//! The wgpu backend (DECISIONS.md #8). No wgpu type leaves this module through the public API.
//!
//! wgpu reports most failures asynchronously, through error scopes; every creation runs inside
//! an out-of-memory and a validation scope and turns what they caught into an [`RhiError`]. The
//! uncaptured-error handler is replaced, because wgpu's default one panics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    AddressMode, Backend, BufferInfo, BufferUsages, CompareFunction, FilterMode, Limits, RhiError,
    SamplerInfo, TextureAspect, TextureDimension, TextureFormat, TextureInfo, TextureUsages,
    TextureViewDimension, TextureViewInfo,
};

pub(crate) struct GpuDevice {
    device: wgpu::Device,
    #[expect(
        dead_code,
        reason = "submission arrives with command recording (roadmap M3)"
    )]
    queue: wgpu::Queue,
    backend: Backend,
    lost: Arc<AtomicBool>,
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
            device,
            queue,
            backend,
            lost,
        })
    }

    pub(crate) const fn backend(&self) -> Backend {
        self.backend
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    pub(crate) fn lose(&mut self) {
        self.device.destroy();
        self.lost.store(true, Ordering::Release);
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

    pub(crate) fn create_buffer(
        &self,
        label: Option<&str>,
        info: &BufferInfo,
    ) -> Result<wgpu::Buffer, RhiError> {
        let desc = wgpu::BufferDescriptor {
            label,
            size: info.size,
            usage: buffer_usages(info.usage),
            mapped_at_creation: false,
        };
        self.scoped(info.size, |device| device.create_buffer(&desc))
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
            dimension: match info.dimension {
                TextureDimension::D1 => wgpu::TextureDimension::D1,
                TextureDimension::D2 => wgpu::TextureDimension::D2,
                TextureDimension::D3 => wgpu::TextureDimension::D3,
            },
            format: texture_format(info.format),
            usage: texture_usages(info.usage),
            view_formats: &[],
        };
        self.scoped(bytes, |device| device.create_texture(&desc))
    }

    pub(crate) fn create_texture_view(
        &self,
        label: Option<&str>,
        texture: &wgpu::Texture,
        info: &TextureViewInfo,
    ) -> Result<wgpu::TextureView, RhiError> {
        let desc = wgpu::TextureViewDescriptor {
            label,
            format: Some(texture_format(info.format)),
            dimension: Some(match info.dimension {
                TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
                TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
                TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
                TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
                TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
                TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
            }),
            usage: None,
            aspect: match info.aspect {
                TextureAspect::All => wgpu::TextureAspect::All,
                TextureAspect::DepthOnly => wgpu::TextureAspect::DepthOnly,
                TextureAspect::StencilOnly => wgpu::TextureAspect::StencilOnly,
            },
            base_mip_level: info.base_mip_level,
            mip_level_count: Some(info.mip_level_count),
            base_array_layer: info.base_array_layer,
            array_layer_count: Some(info.array_layer_count),
        };
        self.scoped(0, |_| texture.create_view(&desc))
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
            address_mode_u: address_mode(u),
            address_mode_v: address_mode(v),
            address_mode_w: address_mode(w),
            mag_filter: filter_mode(info.mag_filter),
            min_filter: filter_mode(info.min_filter),
            mipmap_filter: match info.mipmap_filter {
                FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
                FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
            },
            lod_min_clamp,
            lod_max_clamp,
            compare: info.compare.map(compare_function),
            anisotropy_clamp: info.anisotropy_clamp,
            border_color: None,
        };
        self.scoped(0, |device| device.create_sampler(&desc))
    }
}

fn buffer_usages(usage: BufferUsages) -> wgpu::BufferUsages {
    let pairs = [
        (BufferUsages::MAP_READ, wgpu::BufferUsages::MAP_READ),
        (BufferUsages::MAP_WRITE, wgpu::BufferUsages::MAP_WRITE),
        (BufferUsages::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsages::COPY_DST, wgpu::BufferUsages::COPY_DST),
        (BufferUsages::INDEX, wgpu::BufferUsages::INDEX),
        (BufferUsages::VERTEX, wgpu::BufferUsages::VERTEX),
        (BufferUsages::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsages::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsages::INDIRECT, wgpu::BufferUsages::INDIRECT),
    ];
    pairs
        .into_iter()
        .filter(|(ours, _)| usage.contains(*ours))
        .fold(wgpu::BufferUsages::empty(), |acc, (_, theirs)| acc | theirs)
}

fn texture_usages(usage: TextureUsages) -> wgpu::TextureUsages {
    let pairs = [
        (TextureUsages::COPY_SRC, wgpu::TextureUsages::COPY_SRC),
        (TextureUsages::COPY_DST, wgpu::TextureUsages::COPY_DST),
        (
            TextureUsages::TEXTURE_BINDING,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        (
            TextureUsages::STORAGE_BINDING,
            wgpu::TextureUsages::STORAGE_BINDING,
        ),
        (
            TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
    ];
    pairs
        .into_iter()
        .filter(|(ours, _)| usage.contains(*ours))
        .fold(wgpu::TextureUsages::empty(), |acc, (_, theirs)| {
            acc | theirs
        })
}

const fn texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
    }
}

const fn address_mode(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

const fn filter_mode(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

const fn compare_function(compare: CompareFunction) -> wgpu::CompareFunction {
    match compare {
        CompareFunction::Never => wgpu::CompareFunction::Never,
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

#[cfg(test)]
mod tests {
    use super::{buffer_usages, texture_usages};
    use crate::{BufferUsages, TextureUsages};

    #[test]
    fn usages_map_bit_for_bit() {
        assert_eq!(
            buffer_usages(BufferUsages::all()).bits().count_ones(),
            BufferUsages::all().bits().count_ones()
        );
        assert_eq!(
            texture_usages(TextureUsages::all()).bits().count_ones(),
            TextureUsages::all().bits().count_ones()
        );
        assert_eq!(
            buffer_usages(BufferUsages::MAP_READ | BufferUsages::COPY_DST),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        );
    }
}
