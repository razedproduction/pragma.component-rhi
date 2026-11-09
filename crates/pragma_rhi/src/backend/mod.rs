//! The seam between the front end and the backends: an enum, not a trait object
//! (DECISIONS.md #10). Every operation matches both variants exhaustively.
//!
//! The front end has already validated every descriptor when a call gets here.

mod convert;
mod gpu;
mod null;

use core::num::NonZeroU64;

pub(crate) use gpu::{GpuDevice, map_backend};
pub(crate) use null::NullDevice;

use gpu::{GpuBinding, GpuContext};

use crate::binding::BindGroupLayoutEntry;
use crate::pipeline::{ComputeKey, RenderKey};
use crate::{Backend, BufferInfo, RhiError, SamplerInfo, TextureInfo, TextureViewInfo};

/// A device of one of the backends.
#[derive(Debug)]
pub(crate) enum DeviceBackend {
    Wgpu(GpuDevice),
    Null(NullDevice),
}

/// A backend object of a resource. A table entry always holds the variant of its device's
/// backend, because handles cannot cross devices. Cloning a wgpu object clones a reference.
#[derive(Debug, Clone)]
pub(crate) enum Obj<W> {
    Wgpu(W),
    Null,
}

pub(crate) type BufferObj = Obj<wgpu::Buffer>;
pub(crate) type TextureObj = Obj<wgpu::Texture>;
pub(crate) type TextureViewObj = Obj<wgpu::TextureView>;
pub(crate) type SamplerObj = Obj<wgpu::Sampler>;
pub(crate) type ShaderModuleObj = Obj<wgpu::ShaderModule>;
pub(crate) type BindGroupLayoutObj = Obj<wgpu::BindGroupLayout>;
pub(crate) type BindGroupObj = Obj<wgpu::BindGroup>;
pub(crate) type PipelineLayoutObj = Obj<wgpu::PipelineLayout>;
pub(crate) type RenderPipelineObj = Obj<wgpu::RenderPipeline>;
pub(crate) type ComputePipelineObj = Obj<wgpu::ComputePipeline>;

/// The wgpu object of a resource of a wgpu device.
fn wgpu_of<W>(obj: &Obj<W>) -> Result<&W, RhiError> {
    match obj {
        Obj::Wgpu(object) => Ok(object),
        Obj::Null => Err(RhiError::backend(
            "a resource of another backend reached this device",
        )),
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
            Self::Null => Ok(Obj::Null),
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
            Self::Null => Ok(Obj::Null),
        }
    }
}

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
            Self::Null(device) => device.allocate(info.size).map(|()| Obj::Null),
        }
    }

    pub(crate) fn destroy_buffer(&mut self, obj: BufferObj, info: &BufferInfo) {
        match (self, obj) {
            (Self::Wgpu(_), Obj::Wgpu(buffer)) => buffer.destroy(),
            (Self::Null(device), _) => device.release(info.size),
            (Self::Wgpu(_), Obj::Null) => {}
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
            Self::Null(device) => device.allocate(bytes).map(|()| Obj::Null),
        }
    }

    pub(crate) fn destroy_texture(&mut self, obj: TextureObj, bytes: u64) {
        match (self, obj) {
            (Self::Wgpu(_), Obj::Wgpu(texture)) => texture.destroy(),
            (Self::Null(device), _) => device.release(bytes),
            (Self::Wgpu(_), Obj::Null) => {}
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
            Self::Null(_) => Ok(Obj::Null),
        }
    }

    pub(crate) fn create_sampler(
        &mut self,
        label: Option<&str>,
        info: &SamplerInfo,
    ) -> Result<SamplerObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_sampler(label, info).map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null),
        }
    }

    pub(crate) fn create_shader_module(
        &mut self,
        label: Option<&str>,
        module: &naga::Module,
    ) -> Result<ShaderModuleObj, RhiError> {
        match self {
            Self::Wgpu(device) => device.create_shader_module(label, module).map(Obj::Wgpu),
            Self::Null(_) => Ok(Obj::Null),
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
            Self::Null(_) => Ok(Obj::Null),
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
            Self::Null(_) => Ok(Obj::Null),
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
            Self::Null(_) => Ok(Obj::Null),
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
}
