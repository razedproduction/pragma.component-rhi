//! Resource descriptors and their validation.
//!
//! Every descriptor is validated here, against the device's [`Limits`], before it reaches a
//! backend; a broken rule is a typed [`RhiError`], never a backend panic. The rules are the
//! subset of WebGPU's that the supported types can break; anything a backend still rejects is
//! reported as [`RhiError::Backend`].

use bitflags::bitflags;

use crate::{Limits, ResourceKind, RhiError, Texture};

// ---------------------------------------------------------------------------------------------
// Buffers

bitflags! {
    /// How a [`Buffer`](crate::Buffer) may be used.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct BufferUsages: u16 {
        /// Mappable for reading; may be combined only with `COPY_DST`.
        const MAP_READ = 1 << 0;
        /// Mappable for writing; may be combined only with `COPY_SRC`.
        const MAP_WRITE = 1 << 1;
        /// The source of a copy.
        const COPY_SRC = 1 << 2;
        /// The destination of a copy or a queue write.
        const COPY_DST = 1 << 3;
        /// An index buffer.
        const INDEX = 1 << 4;
        /// A vertex buffer.
        const VERTEX = 1 << 5;
        /// A uniform buffer.
        const UNIFORM = 1 << 6;
        /// A storage buffer.
        const STORAGE = 1 << 7;
        /// The arguments of an indirect draw or dispatch.
        const INDIRECT = 1 << 8;
    }
}

/// Describes a [`Buffer`](crate::Buffer).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct BufferDescriptor<'a> {
    /// A debug label, passed to the backend (graphics debuggers show it).
    pub label: Option<&'a str>,
    /// The size in bytes; not zero, at most [`Limits::max_buffer_size`].
    pub size: u64,
    /// The allowed usages; not empty.
    pub usage: BufferUsages,
}

/// A live buffer's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BufferInfo {
    /// The size in bytes.
    pub size: u64,
    /// The allowed usages.
    pub usage: BufferUsages,
}

pub(crate) fn validate_buffer(
    desc: &BufferDescriptor<'_>,
    limits: &Limits,
) -> Result<BufferInfo, RhiError> {
    const KIND: ResourceKind = ResourceKind::Buffer;
    if desc.usage.is_empty() {
        return Err(RhiError::invalid(KIND, "usage is empty"));
    }
    if desc.usage.contains(BufferUsages::MAP_READ)
        && !(BufferUsages::MAP_READ | BufferUsages::COPY_DST).contains(desc.usage)
    {
        return Err(RhiError::invalid(
            KIND,
            "MAP_READ may be combined only with COPY_DST",
        ));
    }
    if desc.usage.contains(BufferUsages::MAP_WRITE)
        && !(BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).contains(desc.usage)
    {
        return Err(RhiError::invalid(
            KIND,
            "MAP_WRITE may be combined only with COPY_SRC",
        ));
    }
    if desc.size == 0 {
        return Err(RhiError::invalid(KIND, "size is zero"));
    }
    if desc.size > limits.max_buffer_size {
        return Err(RhiError::limit(
            "max_buffer_size",
            desc.size,
            limits.max_buffer_size,
        ));
    }
    Ok(BufferInfo {
        size: desc.size,
        usage: desc.usage,
    })
}

// ---------------------------------------------------------------------------------------------
// Textures

/// The size of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent3d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels (1 for 1D textures).
    pub height: u32,
    /// Depth of a 3D texture, or the number of array layers of a 1D/2D texture.
    pub depth_or_array_layers: u32,
}

impl Default for Extent3d {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        }
    }
}

/// The dimensionality of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureDimension {
    /// One-dimensional.
    D1,
    /// Two-dimensional, possibly an array.
    #[default]
    D2,
    /// Three-dimensional.
    D3,
}

/// A texel format. Compressed formats come later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TextureFormat {
    /// One 8-bit normalized channel.
    R8Unorm,
    /// Two 8-bit normalized channels.
    Rg8Unorm,
    /// Four 8-bit normalized channels.
    #[default]
    Rgba8Unorm,
    /// Four 8-bit normalized channels, sRGB-encoded color.
    Rgba8UnormSrgb,
    /// Four 8-bit normalized channels in BGRA order.
    Bgra8Unorm,
    /// Four 8-bit normalized channels in BGRA order, sRGB-encoded color.
    Bgra8UnormSrgb,
    /// One 16-bit float channel.
    R16Float,
    /// Four 16-bit float channels.
    Rgba16Float,
    /// One 32-bit float channel.
    R32Float,
    /// Two 32-bit float channels.
    Rg32Float,
    /// Four 32-bit float channels.
    Rgba32Float,
    /// One 32-bit unsigned integer channel.
    R32Uint,
    /// 32-bit float depth.
    Depth32Float,
    /// Depth with at least 24 bits of precision.
    Depth24Plus,
    /// Depth with at least 24 bits of precision and 8-bit stencil.
    Depth24PlusStencil8,
}

impl TextureFormat {
    /// Bytes per texel. For the depth formats whose layout the backend picks this is an upper
    /// estimate, used only for memory accounting.
    #[must_use]
    pub const fn block_size(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::R16Float => 2,
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::R32Float
            | Self::R32Uint
            | Self::Depth32Float
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8 => 4,
            Self::Rgba16Float | Self::Rg32Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Whether the format has a depth aspect.
    #[must_use]
    pub const fn has_depth(self) -> bool {
        matches!(
            self,
            Self::Depth32Float | Self::Depth24Plus | Self::Depth24PlusStencil8
        )
    }

    /// Whether the format has a stencil aspect.
    #[must_use]
    pub const fn has_stencil(self) -> bool {
        matches!(self, Self::Depth24PlusStencil8)
    }

    /// Whether the color is sRGB-encoded.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    /// Whether the format may be used with [`TextureUsages::STORAGE_BINDING`] without optional
    /// features (WebGPU's core set).
    #[must_use]
    pub const fn supports_storage(self) -> bool {
        matches!(
            self,
            Self::Rgba8Unorm
                | Self::Rgba16Float
                | Self::R32Float
                | Self::Rg32Float
                | Self::Rgba32Float
                | Self::R32Uint
        )
    }

    /// Whether the format may be multisampled without optional features.
    #[must_use]
    pub const fn supports_multisampling(self) -> bool {
        !matches!(self, Self::Rg32Float | Self::Rgba32Float)
    }
}

bitflags! {
    /// How a [`Texture`] may be used.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct TextureUsages: u8 {
        /// The source of a copy.
        const COPY_SRC = 1 << 0;
        /// The destination of a copy or a queue write.
        const COPY_DST = 1 << 1;
        /// Sampled or loaded in a shader.
        const TEXTURE_BINDING = 1 << 2;
        /// Written in a shader as a storage texture.
        const STORAGE_BINDING = 1 << 3;
        /// A color or depth/stencil attachment of a render pass.
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

/// Describes a [`Texture`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The size; no dimension may be zero.
    pub size: Extent3d,
    /// The number of mip levels, from 1 to the full chain of `size`.
    pub mip_level_count: u32,
    /// 1, or 4 for a multisampled 2D render attachment.
    pub sample_count: u32,
    /// The dimensionality.
    pub dimension: TextureDimension,
    /// The texel format.
    pub format: TextureFormat,
    /// The allowed usages; not empty.
    pub usage: TextureUsages,
}

impl Default for TextureDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            size: Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::default(),
            format: TextureFormat::default(),
            usage: TextureUsages::empty(),
        }
    }
}

/// A live texture's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TextureInfo {
    /// The size.
    pub size: Extent3d,
    /// The number of mip levels.
    pub mip_level_count: u32,
    /// The sample count.
    pub sample_count: u32,
    /// The dimensionality.
    pub dimension: TextureDimension,
    /// The texel format.
    pub format: TextureFormat,
    /// The allowed usages.
    pub usage: TextureUsages,
}

impl TextureInfo {
    /// The number of array layers (1 for a 3D texture).
    const fn array_layers(&self) -> u32 {
        match self.dimension {
            TextureDimension::D3 => 1,
            TextureDimension::D1 | TextureDimension::D2 => self.size.depth_or_array_layers,
        }
    }
}

/// The length of the full mip chain of a texture whose largest side is `side`.
const fn full_mip_chain(side: u32) -> u32 {
    // floor(log2(side)) + 1; `side` is non-zero here.
    u32::BITS.saturating_sub(side.leading_zeros())
}

/// Validates a texture descriptor; returns the texture's properties and its size in bytes.
pub(crate) fn validate_texture(
    desc: &TextureDescriptor<'_>,
    limits: &Limits,
) -> Result<(TextureInfo, u64), RhiError> {
    const KIND: ResourceKind = ResourceKind::Texture;
    let Extent3d {
        width,
        height,
        depth_or_array_layers: depth,
    } = desc.size;
    if desc.usage.is_empty() {
        return Err(RhiError::invalid(KIND, "usage is empty"));
    }
    if width == 0 || height == 0 || depth == 0 {
        return Err(RhiError::invalid(KIND, "a size dimension is zero"));
    }
    let largest_side = match desc.dimension {
        TextureDimension::D1 => {
            if height != 1 || depth != 1 {
                return Err(RhiError::invalid(
                    KIND,
                    "a 1D texture has a height and depth of 1",
                ));
            }
            check_side(
                "max_texture_dimension_1d",
                width,
                limits.max_texture_dimension_1d,
            )?;
            width
        }
        TextureDimension::D2 => {
            check_side(
                "max_texture_dimension_2d",
                width,
                limits.max_texture_dimension_2d,
            )?;
            check_side(
                "max_texture_dimension_2d",
                height,
                limits.max_texture_dimension_2d,
            )?;
            check_side(
                "max_texture_array_layers",
                depth,
                limits.max_texture_array_layers,
            )?;
            width.max(height)
        }
        TextureDimension::D3 => {
            for side in [width, height, depth] {
                check_side(
                    "max_texture_dimension_3d",
                    side,
                    limits.max_texture_dimension_3d,
                )?;
            }
            width.max(height).max(depth)
        }
    };
    if desc.mip_level_count == 0 {
        return Err(RhiError::invalid(KIND, "mip_level_count is zero"));
    }
    if desc.mip_level_count > full_mip_chain(largest_side) {
        return Err(RhiError::invalid(
            KIND,
            "mip_level_count is longer than the full mip chain of the size",
        ));
    }
    if desc.format.has_depth() && desc.dimension != TextureDimension::D2 {
        return Err(RhiError::invalid(KIND, "a depth format needs a 2D texture"));
    }
    if desc.usage.contains(TextureUsages::RENDER_ATTACHMENT)
        && desc.dimension == TextureDimension::D1
    {
        return Err(RhiError::invalid(
            KIND,
            "a 1D texture cannot be a render attachment",
        ));
    }
    if desc.usage.contains(TextureUsages::STORAGE_BINDING) && !desc.format.supports_storage() {
        return Err(RhiError::invalid(
            KIND,
            "the format does not support STORAGE_BINDING",
        ));
    }
    match desc.sample_count {
        1 => {}
        4 => {
            if desc.dimension != TextureDimension::D2 {
                return Err(RhiError::invalid(KIND, "a multisampled texture is 2D"));
            }
            if desc.mip_level_count != 1 || depth != 1 {
                return Err(RhiError::invalid(
                    KIND,
                    "a multisampled texture has one mip level and one layer",
                ));
            }
            if !desc.usage.contains(TextureUsages::RENDER_ATTACHMENT)
                || desc.usage.contains(TextureUsages::STORAGE_BINDING)
            {
                return Err(RhiError::invalid(
                    KIND,
                    "a multisampled texture is a render attachment and not a storage texture",
                ));
            }
            if !desc.format.supports_multisampling() {
                return Err(RhiError::invalid(
                    KIND,
                    "the format does not support multisampling",
                ));
            }
        }
        _ => return Err(RhiError::invalid(KIND, "sample_count is neither 1 nor 4")),
    }
    let info = TextureInfo {
        size: desc.size,
        mip_level_count: desc.mip_level_count,
        sample_count: desc.sample_count,
        dimension: desc.dimension,
        format: desc.format,
        usage: desc.usage,
    };
    let bytes = texture_bytes(&info).ok_or(RhiError::OutOfMemory { bytes: u64::MAX })?;
    Ok((info, bytes))
}

fn check_side(limit: &'static str, side: u32, max: u32) -> Result<(), RhiError> {
    if side > max {
        Err(RhiError::limit(limit, side, max))
    } else {
        Ok(())
    }
}

/// The size of mip level `mip`: width and height (1 for 1D) halve per level, never below 1;
/// `depth_or_array_layers` is the halved depth of a 3D texture and the layer count otherwise.
pub(crate) fn mip_extent(info: &TextureInfo, mip: u32) -> Extent3d {
    let shrink = |side: u32| side.checked_shr(mip).unwrap_or(0).max(1);
    let size = info.size;
    match info.dimension {
        TextureDimension::D1 => Extent3d {
            width: shrink(size.width),
            height: 1,
            depth_or_array_layers: size.depth_or_array_layers,
        },
        TextureDimension::D2 => Extent3d {
            width: shrink(size.width),
            height: shrink(size.height),
            depth_or_array_layers: size.depth_or_array_layers,
        },
        TextureDimension::D3 => Extent3d {
            width: shrink(size.width),
            height: shrink(size.height),
            depth_or_array_layers: shrink(size.depth_or_array_layers),
        },
    }
}

/// The memory a texture needs: every mip level of every layer of every sample. `None` on overflow.
fn texture_bytes(info: &TextureInfo) -> Option<u64> {
    let mut per_layer: u64 = 0;
    for level in 0..info.mip_level_count {
        let shrink = |side: u32| u64::from(side.checked_shr(level).unwrap_or(0).max(1));
        let width = shrink(info.size.width);
        let height = match info.dimension {
            TextureDimension::D1 => 1,
            TextureDimension::D2 | TextureDimension::D3 => shrink(info.size.height),
        };
        let depth = match info.dimension {
            TextureDimension::D3 => shrink(info.size.depth_or_array_layers),
            TextureDimension::D1 | TextureDimension::D2 => 1,
        };
        let level_bytes = width
            .checked_mul(height)?
            .checked_mul(depth)?
            .checked_mul(u64::from(info.format.block_size()))?;
        per_layer = per_layer.checked_add(level_bytes)?;
    }
    per_layer
        .checked_mul(u64::from(info.array_layers()))?
        .checked_mul(u64::from(info.sample_count))
}

// ---------------------------------------------------------------------------------------------
// Texture views

/// How a [`TextureView`](crate::TextureView) interprets its texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    /// A 1D texture.
    D1,
    /// One layer of a 2D texture.
    D2,
    /// Layers of a 2D texture as an array.
    D2Array,
    /// Six layers of a square 2D texture as a cube.
    Cube,
    /// A multiple of six layers of a square 2D texture as an array of cubes.
    CubeArray,
    /// A 3D texture.
    D3,
}

/// Which aspect of a texture a view covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureAspect {
    /// Every aspect of the format.
    #[default]
    All,
    /// Only the depth aspect of a depth format.
    DepthOnly,
    /// Only the stencil aspect of a depth/stencil format.
    StencilOnly,
}

/// Describes a [`TextureView`](crate::TextureView). The view has the texture's format.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TextureViewDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The view dimension; `None` picks the texture's own (`D2Array` for a 2D texture with
    /// several layers).
    pub dimension: Option<TextureViewDimension>,
    /// The aspect.
    pub aspect: TextureAspect,
    /// The first mip level.
    pub base_mip_level: u32,
    /// The number of mip levels; `None` takes the rest of the chain.
    pub mip_level_count: Option<u32>,
    /// The first array layer.
    pub base_array_layer: u32,
    /// The number of array layers; `None` takes the rest.
    pub array_layer_count: Option<u32>,
}

/// A live texture view's properties, with every default resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TextureViewInfo {
    /// The viewed texture.
    pub texture: Texture,
    /// The format (the texture's).
    pub format: TextureFormat,
    /// The view dimension.
    pub dimension: TextureViewDimension,
    /// The aspect.
    pub aspect: TextureAspect,
    /// The first mip level.
    pub base_mip_level: u32,
    /// The number of mip levels.
    pub mip_level_count: u32,
    /// The first array layer.
    pub base_array_layer: u32,
    /// The number of array layers.
    pub array_layer_count: u32,
}

pub(crate) fn validate_texture_view(
    texture: Texture,
    tex: &TextureInfo,
    desc: &TextureViewDescriptor<'_>,
) -> Result<TextureViewInfo, RhiError> {
    const KIND: ResourceKind = ResourceKind::TextureView;
    let range = |base: u32, count: Option<u32>, total: u32, what: &'static str| {
        let count = match count {
            Some(count) => count,
            None => total
                .checked_sub(base)
                .ok_or(RhiError::invalid(KIND, what))?,
        };
        match base.checked_add(count) {
            Some(end) if count > 0 && end <= total => Ok(count),
            _ => Err(RhiError::invalid(KIND, what)),
        }
    };
    let mip_level_count = range(
        desc.base_mip_level,
        desc.mip_level_count,
        tex.mip_level_count,
        "the mip range is empty or outside the texture",
    )?;
    let layers = tex.array_layers();
    let array_layer_count = range(
        desc.base_array_layer,
        desc.array_layer_count,
        layers,
        "the layer range is empty or outside the texture",
    )?;
    let dimension = desc.dimension.unwrap_or(match tex.dimension {
        TextureDimension::D1 => TextureViewDimension::D1,
        TextureDimension::D2 if layers == 1 => TextureViewDimension::D2,
        TextureDimension::D2 => TextureViewDimension::D2Array,
        TextureDimension::D3 => TextureViewDimension::D3,
    });
    let square = tex.size.width == tex.size.height;
    let compatible = match (tex.dimension, dimension) {
        (TextureDimension::D1, TextureViewDimension::D1)
        | (TextureDimension::D2, TextureViewDimension::D2) => array_layer_count == 1,
        (TextureDimension::D2, TextureViewDimension::D2Array)
        | (TextureDimension::D3, TextureViewDimension::D3) => true,
        (TextureDimension::D2, TextureViewDimension::Cube) => square && array_layer_count == 6,
        (TextureDimension::D2, TextureViewDimension::CubeArray) => {
            square && array_layer_count % 6 == 0
        }
        _ => false,
    };
    if !compatible {
        return Err(RhiError::invalid(
            KIND,
            "the view dimension does not fit the texture's dimension, size or layer range",
        ));
    }
    if tex.sample_count > 1 && dimension != TextureViewDimension::D2 {
        return Err(RhiError::invalid(
            KIND,
            "a view of a multisampled texture is 2D",
        ));
    }
    match desc.aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if tex.format.has_depth() => {}
        TextureAspect::StencilOnly if tex.format.has_stencil() => {}
        TextureAspect::DepthOnly | TextureAspect::StencilOnly => {
            return Err(RhiError::invalid(
                KIND,
                "the aspect is not part of the texture's format",
            ));
        }
    }
    Ok(TextureViewInfo {
        texture,
        format: tex.format,
        dimension,
        aspect: desc.aspect,
        base_mip_level: desc.base_mip_level,
        mip_level_count,
        base_array_layer: desc.base_array_layer,
        array_layer_count,
    })
}

// ---------------------------------------------------------------------------------------------
// Samplers

/// What a sampler returns outside `[0, 1]` texture coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    /// The edge texel.
    #[default]
    ClampToEdge,
    /// The texture repeats.
    Repeat,
    /// The texture repeats, mirrored every other time.
    MirrorRepeat,
}

/// Texel filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    /// The nearest texel.
    #[default]
    Nearest,
    /// Linear interpolation.
    Linear,
}

/// A comparison for comparison samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// `new < old`.
    Less,
    /// `new == old`.
    Equal,
    /// `new <= old`.
    LessEqual,
    /// `new > old`.
    Greater,
    /// `new != old`.
    NotEqual,
    /// `new >= old`.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// Describes a [`Sampler`](crate::Sampler).
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// Addressing along `u`.
    pub address_mode_u: AddressMode,
    /// Addressing along `v`.
    pub address_mode_v: AddressMode,
    /// Addressing along `w`.
    pub address_mode_w: AddressMode,
    /// Filtering when magnified.
    pub mag_filter: FilterMode,
    /// Filtering when minified.
    pub min_filter: FilterMode,
    /// Filtering between mip levels.
    pub mipmap_filter: FilterMode,
    /// The smallest level of detail; finite and at least 0.
    pub lod_min_clamp: f32,
    /// The largest level of detail; finite and at least `lod_min_clamp`.
    pub lod_max_clamp: f32,
    /// `Some` makes a comparison sampler.
    pub compare: Option<CompareFunction>,
    /// 1 to 16; above 1 every filter must be [`FilterMode::Linear`].
    pub anisotropy_clamp: u16,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::default(),
            address_mode_v: AddressMode::default(),
            address_mode_w: AddressMode::default(),
            mag_filter: FilterMode::default(),
            min_filter: FilterMode::default(),
            mipmap_filter: FilterMode::default(),
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
        }
    }
}

/// A live sampler's properties.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SamplerInfo {
    /// Addressing along `u`, `v`, `w`.
    pub address_modes: [AddressMode; 3],
    /// Filtering when magnified.
    pub mag_filter: FilterMode,
    /// Filtering when minified.
    pub min_filter: FilterMode,
    /// Filtering between mip levels.
    pub mipmap_filter: FilterMode,
    /// The level-of-detail range.
    pub lod_clamp: [f32; 2],
    /// The comparison, for a comparison sampler.
    pub compare: Option<CompareFunction>,
    /// The anisotropy clamp.
    pub anisotropy_clamp: u16,
}

pub(crate) fn validate_sampler(desc: &SamplerDescriptor<'_>) -> Result<SamplerInfo, RhiError> {
    const KIND: ResourceKind = ResourceKind::Sampler;
    let (min, max) = (desc.lod_min_clamp, desc.lod_max_clamp);
    if !min.is_finite() || !max.is_finite() || min < 0.0 || max < min {
        return Err(RhiError::invalid(
            KIND,
            "the LOD clamp is not a finite range starting at 0 or above",
        ));
    }
    if !(1..=16).contains(&desc.anisotropy_clamp) {
        return Err(RhiError::invalid(KIND, "anisotropy_clamp is not in 1..=16"));
    }
    let all_linear = [desc.mag_filter, desc.min_filter, desc.mipmap_filter]
        .iter()
        .all(|filter| *filter == FilterMode::Linear);
    if desc.anisotropy_clamp > 1 && !all_linear {
        return Err(RhiError::invalid(
            KIND,
            "anisotropic filtering needs every filter to be Linear",
        ));
    }
    Ok(SamplerInfo {
        address_modes: [
            desc.address_mode_u,
            desc.address_mode_v,
            desc.address_mode_w,
        ],
        mag_filter: desc.mag_filter,
        min_filter: desc.min_filter,
        mipmap_filter: desc.mipmap_filter,
        lod_clamp: [min, max],
        compare: desc.compare,
        anisotropy_clamp: desc.anisotropy_clamp,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tex(size: (u32, u32, u32), dimension: TextureDimension) -> TextureDescriptor<'static> {
        TextureDescriptor {
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: size.2,
            },
            dimension,
            usage: TextureUsages::TEXTURE_BINDING,
            ..TextureDescriptor::default()
        }
    }

    fn reason(err: RhiError) -> &'static str {
        match err {
            RhiError::InvalidDescriptor { reason, .. } => reason,
            other => panic!("expected InvalidDescriptor, got {other:?}"),
        }
    }

    #[test]
    fn buffer_rules() {
        let limits = Limits::default();
        let ok = BufferDescriptor {
            size: 256,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ..BufferDescriptor::default()
        };
        assert_eq!(validate_buffer(&ok, &limits).unwrap().size, 256);
        let empty = BufferDescriptor {
            usage: BufferUsages::empty(),
            ..ok.clone()
        };
        assert_eq!(
            reason(validate_buffer(&empty, &limits).unwrap_err()),
            "usage is empty"
        );
        let zero = BufferDescriptor {
            size: 0,
            ..ok.clone()
        };
        assert_eq!(
            reason(validate_buffer(&zero, &limits).unwrap_err()),
            "size is zero"
        );
        let mixed = BufferDescriptor {
            usage: BufferUsages::MAP_READ | BufferUsages::VERTEX,
            ..ok.clone()
        };
        assert!(validate_buffer(&mixed, &limits).is_err());
        let huge = BufferDescriptor {
            size: limits.max_buffer_size + 1,
            ..ok
        };
        assert!(matches!(
            validate_buffer(&huge, &limits),
            Err(RhiError::LimitExceeded {
                limit: "max_buffer_size",
                ..
            })
        ));
    }

    #[test]
    fn texture_bytes_cover_the_mip_chain() {
        let limits = Limits::default();
        let desc = TextureDescriptor {
            mip_level_count: 3,
            ..tex((4, 4, 2), TextureDimension::D2)
        };
        // (16 + 4 + 1) texels * 4 bytes * 2 layers.
        assert_eq!(validate_texture(&desc, &limits).unwrap().1, 168);
        let volume = tex((4, 4, 4), TextureDimension::D3);
        assert_eq!(validate_texture(&volume, &limits).unwrap().1, 256);
    }

    #[test]
    fn texture_rules() {
        let limits = Limits::default();
        let too_many_mips = TextureDescriptor {
            mip_level_count: 4,
            ..tex((4, 4, 1), TextureDimension::D2)
        };
        assert!(validate_texture(&too_many_mips, &limits).is_err());
        assert!(validate_texture(&tex((4, 2, 1), TextureDimension::D1), &limits).is_err());
        assert!(matches!(
            validate_texture(&tex((8193, 1, 1), TextureDimension::D2), &limits),
            Err(RhiError::LimitExceeded {
                limit: "max_texture_dimension_2d",
                ..
            })
        ));
        let depth_3d = TextureDescriptor {
            format: TextureFormat::Depth32Float,
            ..tex((4, 4, 4), TextureDimension::D3)
        };
        assert!(validate_texture(&depth_3d, &limits).is_err());
        let srgb_storage = TextureDescriptor {
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::STORAGE_BINDING,
            ..tex((4, 4, 1), TextureDimension::D2)
        };
        assert!(validate_texture(&srgb_storage, &limits).is_err());
        let msaa = TextureDescriptor {
            sample_count: 4,
            usage: TextureUsages::RENDER_ATTACHMENT,
            ..tex((64, 64, 1), TextureDimension::D2)
        };
        assert_eq!(validate_texture(&msaa, &limits).unwrap().1, 64 * 64 * 4 * 4);
        let msaa_sampled_only = TextureDescriptor {
            usage: TextureUsages::TEXTURE_BINDING,
            ..msaa.clone()
        };
        assert!(validate_texture(&msaa_sampled_only, &limits).is_err());
        let three_samples = TextureDescriptor {
            sample_count: 3,
            ..msaa
        };
        assert!(validate_texture(&three_samples, &limits).is_err());
    }

    #[test]
    fn view_defaults_and_rules() {
        let limits = Limits::default();
        use crate::handle::Handle as _;
        let handle = Texture::from_raw(crate::handle::RawHandle {
            device: 1,
            index: 0,
            generation: 0,
        });
        let (cube_tex, _) = validate_texture(
            &TextureDescriptor {
                mip_level_count: 2,
                ..tex((8, 8, 6), TextureDimension::D2)
            },
            &limits,
        )
        .unwrap();
        let all =
            validate_texture_view(handle, &cube_tex, &TextureViewDescriptor::default()).unwrap();
        assert_eq!(all.dimension, TextureViewDimension::D2Array);
        assert_eq!((all.mip_level_count, all.array_layer_count), (2, 6));
        let cube = TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..TextureViewDescriptor::default()
        };
        assert!(validate_texture_view(handle, &cube_tex, &cube).is_ok());
        let one_layer_cube = TextureViewDescriptor {
            array_layer_count: Some(1),
            ..cube
        };
        assert!(validate_texture_view(handle, &cube_tex, &one_layer_cube).is_err());
        let past_mips = TextureViewDescriptor {
            base_mip_level: 2,
            ..TextureViewDescriptor::default()
        };
        assert!(validate_texture_view(handle, &cube_tex, &past_mips).is_err());
        let overflow = TextureViewDescriptor {
            base_array_layer: u32::MAX,
            array_layer_count: Some(2),
            ..TextureViewDescriptor::default()
        };
        assert!(validate_texture_view(handle, &cube_tex, &overflow).is_err());
        let depth_aspect = TextureViewDescriptor {
            aspect: TextureAspect::DepthOnly,
            ..TextureViewDescriptor::default()
        };
        assert!(validate_texture_view(handle, &cube_tex, &depth_aspect).is_err());
    }

    #[test]
    fn sampler_rules() {
        assert!(validate_sampler(&SamplerDescriptor::default()).is_ok());
        let nan = SamplerDescriptor {
            lod_max_clamp: f32::NAN,
            ..SamplerDescriptor::default()
        };
        assert!(validate_sampler(&nan).is_err());
        let aniso_nearest = SamplerDescriptor {
            anisotropy_clamp: 8,
            ..SamplerDescriptor::default()
        };
        assert!(validate_sampler(&aniso_nearest).is_err());
        let aniso = SamplerDescriptor {
            anisotropy_clamp: 16,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..SamplerDescriptor::default()
        };
        assert!(validate_sampler(&aniso).is_ok());
        let zero = SamplerDescriptor {
            anisotropy_clamp: 0,
            ..SamplerDescriptor::default()
        };
        assert!(validate_sampler(&zero).is_err());
    }
}
