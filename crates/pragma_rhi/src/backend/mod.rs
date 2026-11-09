//! The seam between the front end and the backends: an enum, not a trait object
//! (DECISIONS.md #10). Every operation matches both variants exhaustively.
//!
//! The front end has already validated every descriptor when a call gets here.

mod gpu;
mod null;

pub(crate) use gpu::{GpuDevice, map_backend};
pub(crate) use null::NullDevice;

use crate::{Backend, BufferInfo, RhiError, SamplerInfo, TextureInfo, TextureViewInfo};

/// A device of one of the backends.
#[derive(Debug)]
pub(crate) enum DeviceBackend {
    Wgpu(GpuDevice),
    Null(NullDevice),
}

/// A backend object of a resource. A table entry always holds the variant of its device's
/// backend, because handles cannot cross devices.
#[derive(Debug)]
pub(crate) enum Obj<W> {
    Wgpu(W),
    Null,
}

pub(crate) type BufferObj = Obj<wgpu::Buffer>;
pub(crate) type TextureObj = Obj<wgpu::Texture>;
pub(crate) type TextureViewObj = Obj<wgpu::TextureView>;
pub(crate) type SamplerObj = Obj<wgpu::Sampler>;

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
        match (self, texture) {
            (Self::Wgpu(device), Obj::Wgpu(texture)) => device
                .create_texture_view(label, texture, info)
                .map(Obj::Wgpu),
            (Self::Null(_), Obj::Null) => Ok(Obj::Null),
            (Self::Wgpu(_), Obj::Null) | (Self::Null(_), Obj::Wgpu(_)) => Err(RhiError::backend(
                "a texture of another backend reached this device",
            )),
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
}
