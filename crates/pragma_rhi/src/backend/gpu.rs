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

use super::convert;
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
    #[expect(
        dead_code,
        reason = "submission arrives with command recording (roadmap M3)"
    )]
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
}
