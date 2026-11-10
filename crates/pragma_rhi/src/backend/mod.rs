//! The seam between the front end and the backends: an enum, not a trait object
//! (DECISIONS.md #10). Every operation matches both variants exhaustively.
//!
//! The front end has already validated every descriptor and every command when a call gets
//! here; commands arrive resolved to backend objects ([`Op`]).

mod convert;
mod gpu;
mod null;
mod texel;

use core::num::NonZeroU64;
use core::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

pub(crate) use gpu::{GpuDevice, capabilities as gpu_surface_capabilities, map_backend};
pub(crate) use null::NullDevice;

use gpu::{GpuBinding, GpuContext};

use crate::binding::BindGroupLayoutEntry;
use crate::pipeline::{ComputeKey, RenderKey};
use crate::{
    Backend, BufferInfo, Color, Extent3d, IndexFormat, Operations, Origin3d, RhiError, SamplerInfo,
    TexelCopyBufferLayout, TextureInfo, TextureViewInfo,
};

/// A device of one of the backends.
#[derive(Debug)]
pub(crate) enum DeviceBackend {
    Wgpu(GpuDevice),
    Null(NullDevice),
}

/// A backend object of a resource. A table entry always holds the variant of its device's
/// backend, because handles cannot cross devices. Cloning a wgpu object clones a reference.
/// A null buffer or texture carries the id of its memory in the null device.
#[derive(Debug, Clone)]
pub(crate) enum Obj<W, N = ()> {
    Wgpu(W),
    Null(N),
}

/// The id of a null buffer's or texture's memory.
pub(crate) type NullId = u64;

pub(crate) type BufferObj = Obj<wgpu::Buffer, NullId>;
pub(crate) type TextureObj = Obj<wgpu::Texture, NullId>;
pub(crate) type TextureViewObj = Obj<wgpu::TextureView>;
pub(crate) type SamplerObj = Obj<wgpu::Sampler>;
pub(crate) type ShaderModuleObj = Obj<wgpu::ShaderModule>;
pub(crate) type BindGroupLayoutObj = Obj<wgpu::BindGroupLayout>;
pub(crate) type BindGroupObj = Obj<wgpu::BindGroup>;
pub(crate) type PipelineLayoutObj = Obj<wgpu::PipelineLayout>;
pub(crate) type RenderPipelineObj = Obj<wgpu::RenderPipeline>;
pub(crate) type ComputePipelineObj = Obj<wgpu::ComputePipeline>;

/// How a buffer is mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MapMode {
    Read,
    Write,
}

/// Where an asynchronous buffer mapping stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MapState {
    Pending,
    Ready,
    Failed,
}

/// The state of one buffer mapping, shared with wgpu's map callback.
#[derive(Debug, Clone, Default)]
pub(crate) struct MapFlag(Arc<AtomicU8>);

impl MapFlag {
    const PENDING: u8 = 0;
    const READY: u8 = 1;
    const FAILED: u8 = 2;

    pub(crate) fn resolve(&self, ok: bool) {
        let state = if ok { Self::READY } else { Self::FAILED };
        self.0.store(state, Ordering::Release);
    }

    pub(crate) fn reset(&self) {
        self.0.store(Self::PENDING, Ordering::Release);
    }

    pub(crate) fn state(&self) -> MapState {
        match self.0.load(Ordering::Acquire) {
            Self::READY => MapState::Ready,
            Self::FAILED => MapState::Failed,
            _ => MapState::Pending,
        }
    }
}

fn wrong_backend() -> RhiError {
    RhiError::backend("a resource of another backend reached this device")
}

/// The wgpu object of a resource of a wgpu device.
fn wgpu_of<W, N>(obj: &Obj<W, N>) -> Result<&W, RhiError> {
    match obj {
        Obj::Wgpu(object) => Ok(object),
        Obj::Null(_) => Err(wrong_backend()),
    }
}

/// The memory id of a buffer or texture of a null device.
fn null_of<W>(obj: &Obj<W, NullId>) -> Result<NullId, RhiError> {
    match obj {
        Obj::Null(id) => Ok(*id),
        Obj::Wgpu(_) => Err(wrong_backend()),
    }
}

/// A resource bound to a bind group entry, already validated against its layout entry.
pub(crate) enum BoundObj<'a> {
    Buffer {
        obj: &'a BufferObj,
        offset: u64,
        size: NonZeroU64,
    },
    Sampler(&'a SamplerObj),
    TextureView(&'a TextureViewObj),
}

// ---------------------------------------------------------------------------------------------
// Resolved commands

/// A texture region of a copy: one mip level from `origin`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TexRegion<'a> {
    pub(crate) obj: &'a TextureObj,
    pub(crate) info: &'a TextureInfo,
    pub(crate) mip: u32,
    pub(crate) origin: Origin3d,
}

/// A buffer side of a texture copy: the layout as recorded (for wgpu) and with every default
/// resolved (for the null backend).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BufRegion<'a> {
    pub(crate) obj: &'a BufferObj,
    pub(crate) layout: TexelCopyBufferLayout,
    pub(crate) bytes_per_row: u64,
    pub(crate) rows_per_image: u64,
}

/// One mip level of one layer of a texture, through a view: what an attachment covers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Subresource<'a> {
    pub(crate) view: &'a TextureViewObj,
    pub(crate) texture: &'a TextureObj,
    pub(crate) info: &'a TextureInfo,
    pub(crate) mip: u32,
    pub(crate) layer: u32,
}

pub(crate) struct ColorOp<'a> {
    pub(crate) target: Subresource<'a>,
    pub(crate) resolve: Option<Subresource<'a>>,
    pub(crate) ops: Operations<Color>,
}

pub(crate) struct DepthOp<'a> {
    pub(crate) target: Subresource<'a>,
    pub(crate) depth: Option<Operations<f32>>,
    pub(crate) stencil: Option<Operations<u32>>,
}

pub(crate) enum RenderOp<'a> {
    SetPipeline(&'a RenderPipelineObj),
    SetBindGroup(u32, &'a BindGroupObj, &'a [u32]),
    /// Slot, buffer, offset, size.
    SetVertexBuffer(u32, &'a BufferObj, u64, u64),
    /// Buffer, format, offset, size.
    SetIndexBuffer(&'a BufferObj, IndexFormat, u64, u64),
    /// x, y, width, height, min depth, max depth.
    SetViewport([f32; 6]),
    /// x, y, width, height.
    SetScissorRect([u32; 4]),
    Draw(Range<u32>, Range<u32>),
    DrawIndexed(Range<u32>, i32, Range<u32>),
}

pub(crate) enum ComputeOp<'a> {
    SetPipeline(&'a ComputePipelineObj),
    SetBindGroup(u32, &'a BindGroupObj, &'a [u32]),
    Dispatch([u32; 3]),
}

pub(crate) struct RenderPassOp<'a> {
    pub(crate) label: Option<&'a str>,
    pub(crate) colors: Vec<Option<ColorOp<'a>>>,
    pub(crate) depth: Option<DepthOp<'a>>,
    pub(crate) ops: Vec<RenderOp<'a>>,
}

pub(crate) struct ComputePassOp<'a> {
    pub(crate) label: Option<&'a str>,
    pub(crate) ops: Vec<ComputeOp<'a>>,
}

/// A validated command, resolved to backend objects.
pub(crate) enum Op<'a> {
    CopyBufferToBuffer {
        src: &'a BufferObj,
        src_offset: u64,
        dst: &'a BufferObj,
        dst_offset: u64,
        size: u64,
    },
    ClearBuffer {
        obj: &'a BufferObj,
        offset: u64,
        size: u64,
    },
    CopyBufferToTexture {
        src: BufRegion<'a>,
        dst: TexRegion<'a>,
        size: Extent3d,
    },
    CopyTextureToBuffer {
        src: TexRegion<'a>,
        dst: BufRegion<'a>,
        size: Extent3d,
    },
    CopyTextureToTexture {
        src: TexRegion<'a>,
        dst: TexRegion<'a>,
        size: Extent3d,
    },
    RenderPass(RenderPassOp<'a>),
    ComputePass(ComputePassOp<'a>),
}

/// One validated command buffer.
pub(crate) struct Batch<'a> {
    pub(crate) label: Option<&'a str>,
    pub(crate) ops: Vec<Op<'a>>,
}

// ---------------------------------------------------------------------------------------------
// Pipeline jobs

/// A pipeline compilation detached from the device: `Send + 'static`, so it can run on a
/// blocking-pool thread or inline.
#[expect(
    clippy::large_enum_variant,
    reason = "one job per compilation, moved once; the null variant is the rare one"
)]
pub(crate) enum RenderJob {
    Wgpu {
        context: GpuContext,
        label: Option<String>,
        layout: wgpu::PipelineLayout,
        vertex: wgpu::ShaderModule,
        fragment: Option<wgpu::ShaderModule>,
        key: RenderKey,
    },
    Null,
}

impl RenderJob {
    pub(crate) fn run(self) -> Result<RenderPipelineObj, RhiError> {
        match self {
            Self::Wgpu {
                context,
                label,
                layout,
                vertex,
                fragment,
                key,
            } => context
                .compile_render(label.as_deref(), &layout, &vertex, fragment.as_ref(), &key)
                .map(Obj::Wgpu),
            Self::Null => Ok(Obj::Null(())),
        }
    }
}

/// See [`RenderJob`].
pub(crate) enum ComputeJob {
    Wgpu {
        context: GpuContext,
        label: Option<String>,
        layout: wgpu::PipelineLayout,
        module: wgpu::ShaderModule,
        key: ComputeKey,
    },
    Null,
}

impl ComputeJob {
    pub(crate) fn run(self) -> Result<ComputePipelineObj, RhiError> {
        match self {
            Self::Wgpu {
                context,
                label,
                layout,
                module,
                key,
            } => context
                .compile_compute(label.as_deref(), &layout, &module, &key)
                .map(Obj::Wgpu),
            Self::Null => Ok(Obj::Null(())),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Device operations

impl DeviceBackend {
    pub(crate) const fn backend(&self) -> Backend {
        match self {
            Self::Wgpu(device) => device.backend(),
            Self::Null(_) => Backend::Null,
        }
    }

    pub(crate) fn is_lost(&self) -> bool {
        match self {
            Self::Wgpu(device) => device.is_lost(),
            Self::Null(device) => device.is_lost(),
        }
    }

    pub(crate) fn lose(&mut self) {
        match self {
            Self::Wgpu(device) => device.lose(),
            Self::Null(device) => device.lose(),
        }
    }

    /// Device memory in use, as far as the backend accounts it (only the null backend does).
    pub(crate) const fn allocated(&self) -> Option<u64> {
        match self {
            Self::Wgpu(_) => None,
            Self::Null(device) => Some(device.allocated()),
        }
    }

    pub(crate) fn create_buffer(
        &mut self,
        label: Option<&str>,
        info: &BufferInfo,
    ) -> Result<BufferObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_buffer(label, info).map(Obj::Wgpu),
            Self::Null(device) => device.allocate(info.size).map(Obj::Null),
        }
    }

    pub(crate) fn destroy_buffer(&mut self, obj: BufferObj, info: &BufferInfo) {
        match (self, obj) {
            (Self::Wgpu(_), Obj::Wgpu(buffer)) => buffer.destroy(),
            (Self::Null(device), Obj::Null(id)) => device.release(id, info.size),
            (Self::Wgpu(_), Obj::Null(_)) | (Self::Null(_), Obj::Wgpu(_)) => {}
        }
    }

    pub(crate) fn create_texture(
        &mut self,
        label: Option<&str>,
        info: &TextureInfo,
        bytes: u64,
    ) -> Result<TextureObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_texture(label, info, bytes).map(Obj::Wgpu),
            Self::Null(device) => device.allocate(bytes).map(Obj::Null),
        }
    }

    pub(crate) fn destroy_texture(&mut self, obj: TextureObj, bytes: u64) {
        match (self, obj) {
            (Self::Wgpu(_), Obj::Wgpu(texture)) => texture.destroy(),
            (Self::Null(device), Obj::Null(id)) => device.release(id, bytes),
            (Self::Wgpu(_), Obj::Null(_)) | (Self::Null(_), Obj::Wgpu(_)) => {}
        }
    }

    pub(crate) fn create_texture_view(
        &mut self,
        label: Option<&str>,
        texture: &TextureObj,
        info: &TextureViewInfo,
    ) -> Result<TextureViewObj, RhiError> {
        match self {
            Self::Wgpu(device) => device
                .create_texture_view(label, wgpu_of(texture)?, info)
                .map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn create_sampler(
        &mut self,
        label: Option<&str>,
        info: &SamplerInfo,
    ) -> Result<SamplerObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_sampler(label, info).map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn create_shader_module(
        &mut self,
        label: Option<&str>,
        module: &naga::Module,
    ) -> Result<ShaderModuleObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_shader_module(label, module).map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn create_bind_group_layout(
        &mut self,
        label: Option<&str>,
        entries: &[BindGroupLayoutEntry],
    ) -> Result<BindGroupLayoutObj, RhiError> {
        match self {
            Self::Wgpu(device) => device
                .create_bind_group_layout(label, entries)
                .map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn create_bind_group(
        &mut self,
        label: Option<&str>,
        layout: &BindGroupLayoutObj,
        entries: &[(u32, BoundObj<'_>)],
    ) -> Result<BindGroupObj, RhiError> {
        match self {
            Self::Wgpu(device) => {
                let entries = entries
                    .iter()
                    .map(|(binding, bound)| {
                        let resource = match *bound {
                            BoundObj::Buffer { obj, offset, size } => GpuBinding::Buffer {
                                buffer: wgpu_of(obj)?,
                                offset,
                                size,
                            },
                            BoundObj::Sampler(obj) => GpuBinding::Sampler(wgpu_of(obj)?),
                            BoundObj::TextureView(obj) => GpuBinding::TextureView(wgpu_of(obj)?),
                        };
                        Ok((*binding, resource))
                    })
                    .collect::<Result<Vec<_>, RhiError>>()?;
                device
                    .create_bind_group(label, wgpu_of(layout)?, &entries)
                    .map(Obj::Wgpu)
            }
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn create_pipeline_layout(
        &mut self,
        label: Option<&str>,
        groups: &[&BindGroupLayoutObj],
    ) -> Result<PipelineLayoutObj, RhiError> {
        match self {
            Self::Wgpu(device) => {
                let groups = groups
                    .iter()
                    .map(|group| wgpu_of(group))
                    .collect::<Result<Vec<_>, RhiError>>()?;
                device.create_pipeline_layout(label, &groups).map(Obj::Wgpu)
            }
            Self::Null(_) => Ok(Obj::Null(())),
        }
    }

    pub(crate) fn render_job(
        &self,
        label: Option<&str>,
        layout: &PipelineLayoutObj,
        vertex: &ShaderModuleObj,
        fragment: Option<&ShaderModuleObj>,
        key: &RenderKey,
    ) -> Result<RenderJob, RhiError> {
        match self {
            Self::Wgpu(device) => Ok(RenderJob::Wgpu {
                context: device.context().clone(),
                label: label.map(str::to_owned),
                layout: wgpu_of(layout)?.clone(),
                vertex: wgpu_of(vertex)?.clone(),
                fragment: fragment.map(wgpu_of).transpose()?.cloned(),
                key: key.clone(),
            }),
            Self::Null(_) => Ok(RenderJob::Null),
        }
    }

    pub(crate) fn compute_job(
        &self,
        label: Option<&str>,
        layout: &PipelineLayoutObj,
        module: &ShaderModuleObj,
        key: &ComputeKey,
    ) -> Result<ComputeJob, RhiError> {
        match self {
            Self::Wgpu(device) => Ok(ComputeJob::Wgpu {
                context: device.context().clone(),
                label: label.map(str::to_owned),
                layout: wgpu_of(layout)?.clone(),
                module: wgpu_of(module)?.clone(),
                key: key.clone(),
            }),
            Self::Null(_) => Ok(ComputeJob::Null),
        }
    }

    // -- queue ----------------------------------------------------------------------------

    /// Runs validated command buffers, in order, as our submission `index`.
    pub(crate) fn execute(&mut self, batches: &[Batch<'_>], index: u64) -> Result<(), RhiError> {
        match self {
            Self::Wgpu(device) => device.execute(batches, index),
            Self::Null(device) => device.execute(batches, index),
        }
    }

    /// The last submission the GPU has finished; with `wait`, blocks until submission `wait`
    /// is finished first.
    pub(crate) fn poll(&mut self, wait: Option<u64>) -> Result<u64, RhiError> {
        match self {
            Self::Wgpu(device) => device.poll(wait),
            Self::Null(device) => device.poll(wait),
        }
    }

    /// Blocks until every mapping requested so far has resolved.
    pub(crate) fn wait_for_maps(&self) -> Result<(), RhiError> {
        match self {
            Self::Wgpu(device) => device.wait_for_maps(),
            Self::Null(device) => device.alive(),
        }
    }

    /// Releases a swapchain texture's object: the surface owns a GPU's swapchain images, the
    /// null backend allocated the memory at acquisition.
    pub(crate) fn release_surface_texture(&mut self, obj: TextureObj, bytes: u64) {
        match (self, obj) {
            (Self::Null(device), Obj::Null(id)) => device.release(id, bytes),
            (Self::Wgpu(_) | Self::Null(_), _) => {}
        }
    }

    // -- mapped buffers -------------------------------------------------------------------

    /// A buffer created mapped for writing (`MAP_WRITE | COPY_SRC` staging).
    pub(crate) fn create_mapped_buffer(
        &mut self,
        label: Option<&str>,
        info: &BufferInfo,
    ) -> Result<BufferObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_mapped_buffer(label, info).map(Obj::Wgpu),
            Self::Null(device) => device.allocate(info.size).map(Obj::Null),
        }
    }

    pub(crate) fn write_mapped(
        &mut self,
        obj: &BufferObj,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        match self {
            Self::Wgpu(_) => GpuDevice::write_mapped(wgpu_of(obj)?, offset, data),
            Self::Null(device) => device.write(null_of(obj)?, offset, data),
        }
    }

    pub(crate) fn read_mapped(
        &self,
        obj: &BufferObj,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        match self {
            Self::Wgpu(_) => GpuDevice::read_mapped(wgpu_of(obj)?, offset, size),
            Self::Null(device) => device.read(null_of(obj)?, offset, size),
        }
    }

    pub(crate) fn unmap(&mut self, obj: &BufferObj) {
        if let (Self::Wgpu(_), Obj::Wgpu(buffer)) = (self, obj) {
            buffer.unmap();
        }
    }

    /// Requests a mapping; `flag` resolves in a later poll (at once on the null backend).
    pub(crate) fn map_async(
        &mut self,
        obj: &BufferObj,
        mode: MapMode,
        offset: u64,
        size: u64,
        flag: &MapFlag,
    ) -> Result<(), RhiError> {
        flag.reset();
        match self {
            Self::Wgpu(device) => device.map_async(wgpu_of(obj)?, mode, offset, size, flag),
            Self::Null(device) => {
                device.alive()?;
                flag.resolve(true);
                Ok(())
            }
        }
    }

    /// Writes `data` into a buffer before the commands of the next submission run.
    pub(crate) fn write_buffer(
        &mut self,
        obj: &BufferObj,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        match self {
            Self::Wgpu(device) => device.write_buffer(wgpu_of(obj)?, offset, data),
            Self::Null(device) => device.write(null_of(obj)?, offset, data),
        }
    }

    /// Waits for every submission so far and reads a validated range of a buffer.
    pub(crate) fn read_buffer(
        &mut self,
        obj: &BufferObj,
        info: &BufferInfo,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        match self {
            Self::Wgpu(device) => device.read_buffer(wgpu_of(obj)?, info, offset, size),
            Self::Null(device) => device.read(null_of(obj)?, offset, size),
        }
    }
}
