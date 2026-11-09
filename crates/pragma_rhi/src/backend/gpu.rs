//! The wgpu backend (DECISIONS.md #8). No wgpu type leaves this module through the public API.
//!
//! wgpu reports most failures asynchronously, through error scopes; every creation runs inside
//! an out-of-memory and a validation scope and turns what they caught into an [`RhiError`]. The
//! uncaptured-error handler is replaced, because wgpu's default one panics. Error scopes are
//! per thread in wgpu, so a pipeline compiled on a blocking-pool thread gets its own scopes
//! there.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{Batch, BufRegion, ComputeOp, Op, RenderOp, TexRegion, convert, wgpu_of};
use crate::binding::BindGroupLayoutEntry;
use crate::pipeline::{ComputeKey, RenderKey};
use crate::{Backend, BufferInfo, Limits, RhiError, SamplerInfo, TextureInfo, TextureViewInfo};

/// What a creation on any thread needs: the device and its loss flag. Cheap to clone (wgpu's
/// device is reference-counted), `Send` and `Sync`.
#[derive(Clone)]
pub(crate) struct GpuContext {
    device: wgpu::Device,
    lost: Arc<AtomicBool>,
}

impl GpuContext {
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Runs `create` inside error scopes and maps what they caught.
    fn scoped<R>(
        &self,
        bytes: u64,
        create: impl FnOnce(&wgpu::Device) -> R,
    ) -> Result<R, RhiError> {
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let value = create(&self.device);
        let validation = pollster::block_on(validation.pop());
        let oom = pollster::block_on(oom.pop());
        if self.is_lost() {
            return Err(RhiError::DeviceLost);
        }
        if oom.is_some() {
            return Err(RhiError::OutOfMemory { bytes });
        }
        match validation {
            None => Ok(value),
            Some(wgpu::Error::Validation { description, .. }) => {
                Err(RhiError::backend(description))
            }
            Some(other) => Err(RhiError::backend(other.to_string())),
        }
    }

    /// Compiles a render pipeline from its validated cache key.
    pub(crate) fn compile_render(
        &self,
        label: Option<&str>,
        layout: &wgpu::PipelineLayout,
        vertex: &wgpu::ShaderModule,
        fragment: Option<&wgpu::ShaderModule>,
        key: &RenderKey,
    ) -> Result<wgpu::RenderPipeline, RhiError> {
        let attributes: Vec<Vec<wgpu::VertexAttribute>> = key
            .buffers
            .iter()
            .map(|buffer| {
                buffer
                    .attributes
                    .iter()
                    .map(|attribute| wgpu::VertexAttribute {
                        format: convert::vertex_format(attribute.format),
                        offset: attribute.offset,
                        shader_location: attribute.shader_location,
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = key
            .buffers
            .iter()
            .zip(&attributes)
            .map(|(buffer, attributes)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: buffer.array_stride,
                    step_mode: convert::step_mode(buffer.step_mode),
                    attributes,
                })
            })
            .collect();
        let targets: Vec<Option<wgpu::ColorTargetState>> = key
            .fragment
            .as_ref()
            .map(|fragment| {
                fragment
                    .targets
                    .iter()
                    .map(|target| target.as_ref().map(convert::color_target))
                    .collect()
            })
            .unwrap_or_default();
        let fragment_state = match (fragment, &key.fragment) {
            (Some(module), Some(fragment)) => Some(wgpu::FragmentState {
                module,
                entry_point: Some(&fragment.stage.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            _ => None,
        };
        let desc = wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex,
                entry_point: Some(&key.vertex.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &buffers,
            },
            primitive: convert::primitive(&key.primitive),
            depth_stencil: key.depth_stencil.as_ref().map(convert::depth_stencil),
            multisample: convert::multisample(&key.multisample),
            fragment: fragment_state,
            multiview_mask: None,
            cache: None,
        };
        self.scoped(0, |device| device.create_render_pipeline(&desc))
    }

    /// Compiles a compute pipeline from its validated cache key.
    pub(crate) fn compile_compute(
        &self,
        label: Option<&str>,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        key: &ComputeKey,
    ) -> Result<wgpu::ComputePipeline, RhiError> {
        let desc = wgpu::ComputePipelineDescriptor {
            label,
            layout: Some(layout),
            module,
            entry_point: Some(&key.stage.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        };
        self.scoped(0, |device| device.create_compute_pipeline(&desc))
    }
}

pub(crate) struct GpuDevice {
    context: GpuContext,
    queue: wgpu::Queue,
    backend: Backend,
}

impl core::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("backend", &self.backend)
            .field("lost", &self.is_lost())
            .finish_non_exhaustive()
    }
}

/// Maps a wgpu backend to ours; `None` for backends this crate does not enable.
pub(crate) fn map_backend(backend: wgpu::Backend) -> Option<Backend> {
    match backend {
        wgpu::Backend::Vulkan => Some(Backend::Vulkan),
        wgpu::Backend::Dx12 => Some(Backend::Dx12),
        _ => None,
    }
}

/// A resource bound to a bind group entry, resolved to its wgpu object.
pub(crate) enum GpuBinding<'a> {
    Buffer {
        buffer: &'a wgpu::Buffer,
        offset: u64,
        size: core::num::NonZeroU64,
    },
    Sampler(&'a wgpu::Sampler),
    TextureView(&'a wgpu::TextureView),
}

impl GpuDevice {
    pub(crate) fn request(
        adapter: &wgpu::Adapter,
        backend: Backend,
        label: Option<&str>,
        limits: &Limits,
    ) -> Result<Self, RhiError> {
        // The adapter's own limits are always requestable; ours narrow the ones we validate.
        let mut required_limits = adapter.limits();
        limits.apply_to_wgpu(&mut required_limits);
        let desc = wgpu::DeviceDescriptor {
            label,
            required_limits,
            ..wgpu::DeviceDescriptor::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc))
            .map_err(|err| RhiError::backend(err.to_string()))?;
        let lost = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&lost);
        device.set_device_lost_callback(move |_reason, _message| {
            flag.store(true, Ordering::Release);
        });
        // Every operation runs in error scopes; anything that still escapes is dropped instead
        // of hitting wgpu's default handler, which panics.
        device.on_uncaptured_error(Arc::new(|_error: wgpu::Error| {}));
        Ok(Self {
            context: GpuContext { device, lost },
            queue,
            backend,
        })
    }

    pub(crate) const fn backend(&self) -> Backend {
        self.backend
    }

    pub(crate) const fn context(&self) -> &GpuContext {
        &self.context
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.context.is_lost()
    }

    pub(crate) fn lose(&mut self) {
        self.context.device.destroy();
        self.context.lost.store(true, Ordering::Release);
    }

    pub(crate) fn create_buffer(
        &self,
        label: Option<&str>,
        info: &BufferInfo,
    ) -> Result<wgpu::Buffer, RhiError> {
        let desc = wgpu::BufferDescriptor {
            label,
            size: info.size,
            usage: convert::buffer_usages(info.usage),
            mapped_at_creation: false,
        };
        self.context
            .scoped(info.size, |device| device.create_buffer(&desc))
    }

    pub(crate) fn create_texture(
        &self,
        label: Option<&str>,
        info: &TextureInfo,
        bytes: u64,
    ) -> Result<wgpu::Texture, RhiError> {
        let desc = wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: info.size.width,
                height: info.size.height,
                depth_or_array_layers: info.size.depth_or_array_layers,
            },
            mip_level_count: info.mip_level_count,
            sample_count: info.sample_count,
            dimension: convert::texture_dimension(info.dimension),
            format: convert::texture_format(info.format),
            usage: convert::texture_usages(info.usage),
            view_formats: &[],
        };
        self.context
            .scoped(bytes, |device| device.create_texture(&desc))
    }

    pub(crate) fn create_texture_view(
        &self,
        label: Option<&str>,
        texture: &wgpu::Texture,
        info: &TextureViewInfo,
    ) -> Result<wgpu::TextureView, RhiError> {
        let desc = wgpu::TextureViewDescriptor {
            label,
            format: Some(convert::texture_format(info.format)),
            dimension: Some(convert::view_dimension(info.dimension)),
            usage: None,
            aspect: convert::texture_aspect(info.aspect),
            base_mip_level: info.base_mip_level,
            mip_level_count: Some(info.mip_level_count),
            base_array_layer: info.base_array_layer,
            array_layer_count: Some(info.array_layer_count),
        };
        self.context.scoped(0, |_| texture.create_view(&desc))
    }

    pub(crate) fn create_sampler(
        &self,
        label: Option<&str>,
        info: &SamplerInfo,
    ) -> Result<wgpu::Sampler, RhiError> {
        let [u, v, w] = info.address_modes;
        let [lod_min_clamp, lod_max_clamp] = info.lod_clamp;
        let desc = wgpu::SamplerDescriptor {
            label,
            address_mode_u: convert::address_mode(u),
            address_mode_v: convert::address_mode(v),
            address_mode_w: convert::address_mode(w),
            mag_filter: convert::filter_mode(info.mag_filter),
            min_filter: convert::filter_mode(info.min_filter),
            mipmap_filter: convert::mipmap_filter_mode(info.mipmap_filter),
            lod_min_clamp,
            lod_max_clamp,
            compare: info.compare.map(convert::compare_function),
            anisotropy_clamp: info.anisotropy_clamp,
            border_color: None,
        };
        self.context
            .scoped(0, |device| device.create_sampler(&desc))
    }

    /// Hands a module naga has already validated to wgpu as IR (DECISIONS.md #13).
    pub(crate) fn create_shader_module(
        &self,
        label: Option<&str>,
        module: &naga::Module,
    ) -> Result<wgpu::ShaderModule, RhiError> {
        let desc = wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Naga(Cow::Owned(module.clone())),
        };
        self.context
            .scoped(0, |device| device.create_shader_module(desc))
    }

    pub(crate) fn create_bind_group_layout(
        &self,
        label: Option<&str>,
        entries: &[BindGroupLayoutEntry],
    ) -> Result<wgpu::BindGroupLayout, RhiError> {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect();
        let desc = wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &entries,
        };
        self.context
            .scoped(0, |device| device.create_bind_group_layout(&desc))
    }

    pub(crate) fn create_bind_group(
        &self,
        label: Option<&str>,
        layout: &wgpu::BindGroupLayout,
        entries: &[(u32, GpuBinding<'_>)],
    ) -> Result<wgpu::BindGroup, RhiError> {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match *resource {
                    GpuBinding::Buffer {
                        buffer,
                        offset,
                        size,
                    } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset,
                        size: Some(size),
                    }),
                    GpuBinding::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    GpuBinding::TextureView(view) => wgpu::BindingResource::TextureView(view),
                },
            })
            .collect();
        let desc = wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &entries,
        };
        self.context
            .scoped(0, |device| device.create_bind_group(&desc))
    }

    pub(crate) fn create_pipeline_layout(
        &self,
        label: Option<&str>,
        groups: &[&wgpu::BindGroupLayout],
    ) -> Result<wgpu::PipelineLayout, RhiError> {
        let groups: Vec<Option<&wgpu::BindGroupLayout>> =
            groups.iter().copied().map(Some).collect();
        let desc = wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &groups,
            immediate_size: 0,
        };
        self.context
            .scoped(0, |device| device.create_pipeline_layout(&desc))
    }

    // -- queue ----------------------------------------------------------------------------

    /// Encodes every validated command buffer and submits them in order, inside error scopes.
    pub(crate) fn execute(&self, batches: &[Batch<'_>]) -> Result<(), RhiError> {
        self.context.scoped(0, |device| -> Result<(), RhiError> {
            let mut buffers = Vec::new();
            buffers
                .try_reserve_exact(batches.len())
                .map_err(|_| RhiError::OutOfMemory {
                    bytes: batches.len() as u64,
                })?;
            for batch in batches {
                let mut encoder = device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: batch.label });
                for op in &batch.ops {
                    encode(&mut encoder, op)?;
                }
                buffers.push(encoder.finish());
            }
            self.queue.submit(buffers);
            Ok(())
        })?
    }

    pub(crate) fn write_buffer(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        self.context.scoped(data.len() as u64, |_| {
            self.queue.write_buffer(buffer, offset, data);
        })
    }

    /// Reads a validated range of a buffer with `MAP_READ` or `COPY_SRC`, after every
    /// submission so far (and every pending queue write) has finished.
    pub(crate) fn read_buffer(
        &self,
        buffer: &wgpu::Buffer,
        info: &BufferInfo,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if info.usage.contains(crate::BufferUsages::MAP_READ) {
            // Flushes pending queue writes, which run before the next submission.
            self.context.scoped(0, |_| {
                self.queue.submit([]);
            })?;
            // A mapping starts at a multiple of `MAP_ALIGNMENT` (8); offsets are multiples of 4.
            let start = offset & !(wgpu::MAP_ALIGNMENT - 1);
            let skip = offset.saturating_sub(start);
            let bytes = self.map_read(buffer, start, size.saturating_add(skip))?;
            let skip = usize::try_from(skip).unwrap_or(0);
            return Ok(bytes.get(skip..).map(<[u8]>::to_vec).unwrap_or_default());
        }
        let staging = self.context.scoped(size, |device| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pragma_rhi readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })?;
        let copied = self.context.scoped(0, |device| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pragma_rhi readback"),
            });
            encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, Some(size));
            self.queue.submit([encoder.finish()]);
        });
        let bytes = copied.and_then(|()| self.map_read(&staging, 0, size));
        staging.destroy();
        bytes
    }

    /// Maps `size` bytes of a `MAP_READ` buffer at `offset`, waits, copies them out, unmaps.
    fn map_read(&self, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Result<Vec<u8>, RhiError> {
        let end = offset.saturating_add(size);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.context.scoped(0, |_| {
            buffer.map_async(wgpu::MapMode::Read, offset..end, move |result| {
                // The receiver waits below; a failed send only means it gave up.
                let _ = sender.send(result);
            });
        })?;
        let polled = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        if self.is_lost() {
            return Err(RhiError::DeviceLost);
        }
        polled.map_err(|err| RhiError::backend(err.to_string()))?;
        receiver
            .recv()
            .map_err(|_| RhiError::backend("the buffer mapping was dropped"))?
            .map_err(|err| RhiError::backend(err.to_string()))?;
        let bytes = buffer
            .get_mapped_range(offset..end)
            .map(|view| view.to_vec())
            .map_err(|err| RhiError::backend(err.to_string()));
        buffer.unmap();
        bytes
    }
}

fn texture_copy<'a>(region: &TexRegion<'a>) -> Result<wgpu::TexelCopyTextureInfo<'a>, RhiError> {
    Ok(wgpu::TexelCopyTextureInfo {
        texture: wgpu_of(region.obj)?,
        mip_level: region.mip,
        origin: convert::origin(region.origin),
        aspect: wgpu::TextureAspect::All,
    })
}

fn buffer_copy<'a>(region: &BufRegion<'a>) -> Result<wgpu::TexelCopyBufferInfo<'a>, RhiError> {
    Ok(wgpu::TexelCopyBufferInfo {
        buffer: wgpu_of(region.obj)?,
        layout: convert::buffer_layout(region.layout),
    })
}

/// Records one validated command into a wgpu encoder.
fn encode(encoder: &mut wgpu::CommandEncoder, op: &Op<'_>) -> Result<(), RhiError> {
    match op {
        Op::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        } => encoder.copy_buffer_to_buffer(
            wgpu_of(src)?,
            *src_offset,
            wgpu_of(dst)?,
            *dst_offset,
            Some(*size),
        ),
        Op::ClearBuffer { obj, offset, size } => {
            encoder.clear_buffer(wgpu_of(obj)?, *offset, Some(*size));
        }
        Op::CopyBufferToTexture { src, dst, size } => encoder.copy_buffer_to_texture(
            buffer_copy(src)?,
            texture_copy(dst)?,
            convert::extent(*size),
        ),
        Op::CopyTextureToBuffer { src, dst, size } => encoder.copy_texture_to_buffer(
            texture_copy(src)?,
            buffer_copy(dst)?,
            convert::extent(*size),
        ),
        Op::CopyTextureToTexture { src, dst, size } => encoder.copy_texture_to_texture(
            texture_copy(src)?,
            texture_copy(dst)?,
            convert::extent(*size),
        ),
        Op::RenderPass(pass) => {
            let colors = pass
                .colors
                .iter()
                .map(|color| {
                    color
                        .as_ref()
                        .map(|color| {
                            Ok(wgpu::RenderPassColorAttachment {
                                view: wgpu_of(color.target.view)?,
                                depth_slice: None,
                                resolve_target: color
                                    .resolve
                                    .map(|resolve| wgpu_of(resolve.view))
                                    .transpose()?,
                                ops: convert::operations(color.ops, convert::color),
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>, RhiError>>()?;
            let depth_stencil_attachment = pass
                .depth
                .as_ref()
                .map(|depth| {
                    Ok::<_, RhiError>(wgpu::RenderPassDepthStencilAttachment {
                        view: wgpu_of(depth.target.view)?,
                        depth_ops: depth.depth.map(|ops| convert::operations(ops, |v| v)),
                        stencil_ops: depth.stencil.map(|ops| convert::operations(ops, |v| v)),
                    })
                })
                .transpose()?;
            let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: pass.label,
                color_attachments: &colors,
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(pipeline) => render.set_pipeline(wgpu_of(pipeline)?),
                    RenderOp::SetBindGroup(index, group, offsets) => {
                        render.set_bind_group(*index, wgpu_of(group)?, offsets);
                    }
                    RenderOp::SetVertexBuffer(slot, buffer, offset, size) => render
                        .set_vertex_buffer(
                            *slot,
                            wgpu_of(buffer)?.slice(*offset..offset.saturating_add(*size)),
                        ),
                    RenderOp::SetIndexBuffer(buffer, format, offset, size) => render
                        .set_index_buffer(
                            wgpu_of(buffer)?.slice(*offset..offset.saturating_add(*size)),
                            convert::index_format(*format),
                        ),
                    RenderOp::SetViewport([x, y, width, height, min, max]) => {
                        render.set_viewport(*x, *y, *width, *height, *min, *max);
                    }
                    RenderOp::SetScissorRect([x, y, width, height]) => {
                        render.set_scissor_rect(*x, *y, *width, *height);
                    }
                    RenderOp::Draw(vertices, instances) => {
                        render.draw(vertices.clone(), instances.clone());
                    }
                    RenderOp::DrawIndexed(indices, base_vertex, instances) => {
                        render.draw_indexed(indices.clone(), *base_vertex, instances.clone());
                    }
                }
            }
        }
        Op::ComputePass(pass) => {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: pass.label,
                timestamp_writes: None,
            });
            for op in &pass.ops {
                match op {
                    ComputeOp::SetPipeline(pipeline) => compute.set_pipeline(wgpu_of(pipeline)?),
                    ComputeOp::SetBindGroup(index, group, offsets) => {
                        compute.set_bind_group(*index, wgpu_of(group)?, offsets);
                    }
                    ComputeOp::Dispatch([x, y, z]) => compute.dispatch_workgroups(*x, *y, *z),
                }
            }
        }
    }
    Ok(())
}
