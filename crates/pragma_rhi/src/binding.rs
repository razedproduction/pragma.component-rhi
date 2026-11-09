//! Bind group layouts, bind groups and pipeline layouts, and their validation.
//!
//! The rules are WebGPU's for the binding types this crate supports, checked against the
//! device's [`Limits`] before anything reaches a backend. Checks that need a shader — whether a
//! layout fits what an entry point declares — run at pipeline creation (`pipeline.rs`).

use core::num::NonZeroU64;

use crate::{
    BindGroupLayout, Buffer, BufferInfo, BufferUsages, FilterMode, Limits, ResourceKind, RhiError,
    Sampler, SamplerInfo, ShaderStages, TextureAspect, TextureFormat, TextureInfo, TextureUsages,
    TextureView, TextureViewDimension, TextureViewInfo,
};

// ---------------------------------------------------------------------------------------------
// Binding types

/// The kind of a buffer binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferBindingType {
    /// A uniform buffer (`var<uniform>`); the buffer needs [`BufferUsages::UNIFORM`].
    #[default]
    Uniform,
    /// A storage buffer (`var<storage, read>` when `read_only`, else `var<storage, read_write>`);
    /// the buffer needs [`BufferUsages::STORAGE`].
    Storage {
        /// Whether the shader only reads it.
        read_only: bool,
    },
}

/// The kind of a sampler binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SamplerBindingType {
    /// A sampler that may filter (`sampler`).
    #[default]
    Filtering,
    /// A sampler whose every filter is [`FilterMode::Nearest`] (`sampler`).
    NonFiltering,
    /// A comparison sampler (`sampler_comparison`).
    Comparison,
}

/// What a sampled texture binding yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    /// Floats (`texture_*<f32>`); `filterable` lets a filtering sampler read it.
    Float {
        /// Whether the bound format must be filterable.
        filterable: bool,
    },
    /// Depth (`texture_depth_*`).
    Depth,
    /// Signed integers (`texture_*<i32>`).
    Sint,
    /// Unsigned integers (`texture_*<u32>`).
    Uint,
}

impl Default for TextureSampleType {
    fn default() -> Self {
        Self::Float { filterable: true }
    }
}

/// How a shader accesses a storage texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StorageTextureAccess {
    /// `write` access.
    #[default]
    WriteOnly,
    /// `read` access.
    ReadOnly,
    /// `read_write` access; only [`TextureFormat::R32Float`] and [`TextureFormat::R32Uint`].
    ReadWrite,
}

/// The type of one binding of a [`BindGroupLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindingType {
    /// A uniform or storage buffer.
    Buffer {
        /// Uniform or storage.
        ty: BufferBindingType,
        /// Whether the offset is given when the bind group is set (roadmap M3).
        has_dynamic_offset: bool,
        /// The smallest size a bound range may have; `None` defers the check to draw time.
        min_binding_size: Option<NonZeroU64>,
    },
    /// A sampler.
    Sampler(SamplerBindingType),
    /// A sampled texture; the texture needs [`TextureUsages::TEXTURE_BINDING`].
    Texture {
        /// What it yields.
        sample_type: TextureSampleType,
        /// The view dimension the shader declares.
        view_dimension: TextureViewDimension,
        /// Whether the texture is multisampled (`texture_multisampled_2d`).
        multisampled: bool,
    },
    /// A storage texture; the texture needs [`TextureUsages::STORAGE_BINDING`].
    StorageTexture {
        /// The access the shader declares.
        access: StorageTextureAccess,
        /// The texel format the shader declares; the view must have exactly this format.
        format: TextureFormat,
        /// The view dimension the shader declares; not a cube.
        view_dimension: TextureViewDimension,
    },
}

impl Default for BindingType {
    /// A uniform buffer.
    fn default() -> Self {
        Self::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }
}

/// One binding of a [`BindGroupLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BindGroupLayoutEntry {
    /// The binding number (`@binding(n)`); less than
    /// [`Limits::max_bindings_per_bind_group`], unique in the layout.
    pub binding: u32,
    /// The stages that see the binding.
    pub visibility: ShaderStages,
    /// The binding's type.
    pub ty: BindingType,
}

/// Describes a [`BindGroupLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The bindings, in any order.
    pub entries: &'a [BindGroupLayoutEntry],
}

/// A live bind group layout's properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BindGroupLayoutInfo {
    /// The bindings, sorted by binding number.
    pub entries: Vec<BindGroupLayoutEntry>,
}

// ---------------------------------------------------------------------------------------------
// Bind groups

/// A range of a buffer bound to a buffer binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferBinding {
    /// The buffer.
    pub buffer: Buffer,
    /// The first byte; a multiple of the device's `min_*_buffer_offset_alignment`.
    pub offset: u64,
    /// The size of the range; `None` takes the rest of the buffer.
    pub size: Option<NonZeroU64>,
}

/// A resource bound to one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindingResource {
    /// A buffer range, for a [`BindingType::Buffer`].
    Buffer(BufferBinding),
    /// A sampler, for a [`BindingType::Sampler`].
    Sampler(Sampler),
    /// A texture view, for a [`BindingType::Texture`] or [`BindingType::StorageTexture`].
    TextureView(TextureView),
}

/// One binding of a bind group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupEntry {
    /// The binding number of the layout entry it fills.
    pub binding: u32,
    /// The resource.
    pub resource: BindingResource,
}

/// Describes a [`BindGroup`](crate::BindGroup); the layout is passed to
/// [`Device::create_bind_group`](crate::Device::create_bind_group) alongside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BindGroupDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// Exactly one entry for every entry of the layout, in any order.
    pub entries: &'a [BindGroupEntry],
}

/// A live bind group's properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BindGroupInfo {
    /// The layout.
    pub layout: BindGroupLayout,
    /// The entries, sorted by binding number, with every buffer range's size resolved.
    pub entries: Vec<BindGroupEntry>,
}

// ---------------------------------------------------------------------------------------------
// Pipeline layouts

/// Describes a [`PipelineLayout`](crate::PipelineLayout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineLayoutDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The layout of bind group `i` is `bind_group_layouts[i]`; at most
    /// [`Limits::max_bind_groups`].
    pub bind_group_layouts: &'a [BindGroupLayout],
}

/// A live pipeline layout's properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PipelineLayoutInfo {
    /// The bind group layouts, by group index.
    pub bind_group_layouts: Vec<BindGroupLayout>,
}

// ---------------------------------------------------------------------------------------------
// Validation

/// Validates a layout; returns its entries sorted by binding number.
pub(crate) fn validate_bind_group_layout(
    desc: &BindGroupLayoutDescriptor<'_>,
    limits: &Limits,
) -> Result<Vec<BindGroupLayoutEntry>, RhiError> {
    const KIND: ResourceKind = ResourceKind::BindGroupLayout;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(desc.entries.len())
        .map_err(|_| RhiError::OutOfMemory {
            bytes: u64::try_from(desc.entries.len()).unwrap_or(u64::MAX),
        })?;
    for entry in desc.entries {
        if entry.binding >= limits.max_bindings_per_bind_group {
            return Err(RhiError::limit(
                "max_bindings_per_bind_group",
                u64::from(entry.binding) + 1,
                limits.max_bindings_per_bind_group,
            ));
        }
        if !ShaderStages::all().contains(entry.visibility) {
            return Err(RhiError::invalid(KIND, "the visibility has unknown stages"));
        }
        let vertex = entry.visibility.contains(ShaderStages::VERTEX);
        match entry.ty {
            BindingType::Buffer { ty, .. } => {
                if vertex && ty == (BufferBindingType::Storage { read_only: false }) {
                    return Err(RhiError::invalid(
                        KIND,
                        "a storage buffer visible to the vertex stage must be read-only",
                    ));
                }
            }
            BindingType::Sampler(_) => {}
            BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
            } => {
                if multisampled
                    && (view_dimension != TextureViewDimension::D2
                        || sample_type == (TextureSampleType::Float { filterable: true }))
                {
                    return Err(RhiError::invalid(
                        KIND,
                        "a multisampled texture binding is 2D and not filterable",
                    ));
                }
            }
            BindingType::StorageTexture {
                access,
                format,
                view_dimension,
            } => {
                if !format.supports_storage() {
                    return Err(RhiError::invalid(
                        KIND,
                        "the storage texture format does not support STORAGE_BINDING",
                    ));
                }
                if matches!(
                    view_dimension,
                    TextureViewDimension::Cube | TextureViewDimension::CubeArray
                ) {
                    return Err(RhiError::invalid(
                        KIND,
                        "a storage texture binding cannot be a cube",
                    ));
                }
                if access == StorageTextureAccess::ReadWrite
                    && !matches!(format, TextureFormat::R32Float | TextureFormat::R32Uint)
                {
                    return Err(RhiError::invalid(
                        KIND,
                        "read-write storage textures need R32Float or R32Uint",
                    ));
                }
                if vertex && access != StorageTextureAccess::ReadOnly {
                    return Err(RhiError::invalid(
                        KIND,
                        "a storage texture visible to the vertex stage must be read-only",
                    ));
                }
            }
        }
        entries.push(*entry);
    }
    entries.sort_by_key(|entry| entry.binding);
    if entries
        .windows(2)
        .any(|pair| matches!(pair, [a, b] if a.binding == b.binding))
    {
        return Err(RhiError::invalid(KIND, "a binding number appears twice"));
    }
    check_layout_limits(entries.iter(), limits)?;
    Ok(entries)
}

/// Validates the bind group layouts of a pipeline layout (their entries, by group).
pub(crate) fn validate_pipeline_layout(
    groups: &[&[BindGroupLayoutEntry]],
    limits: &Limits,
) -> Result<(), RhiError> {
    let count = u32::try_from(groups.len()).unwrap_or(u32::MAX);
    if count > limits.max_bind_groups {
        return Err(RhiError::limit(
            "max_bind_groups",
            count,
            limits.max_bind_groups,
        ));
    }
    check_layout_limits(groups.iter().flat_map(|entries| entries.iter()), limits)
}

/// The per-stage and per-layout binding counts of WebGPU.
fn check_layout_limits<'e>(
    entries: impl Iterator<Item = &'e BindGroupLayoutEntry> + Clone,
    limits: &Limits,
) -> Result<(), RhiError> {
    let count = |filter: &dyn Fn(&BindGroupLayoutEntry) -> bool| {
        u32::try_from(entries.clone().filter(|entry| filter(entry)).count()).unwrap_or(u32::MAX)
    };
    let dynamic = |uniform: bool| {
        move |entry: &BindGroupLayoutEntry| match entry.ty {
            BindingType::Buffer {
                ty,
                has_dynamic_offset: true,
                ..
            } => (ty == BufferBindingType::Uniform) == uniform,
            _ => false,
        }
    };
    let layout_limits = [
        (
            "max_dynamic_uniform_buffers_per_pipeline_layout",
            count(&dynamic(true)),
            limits.max_dynamic_uniform_buffers_per_pipeline_layout,
        ),
        (
            "max_dynamic_storage_buffers_per_pipeline_layout",
            count(&dynamic(false)),
            limits.max_dynamic_storage_buffers_per_pipeline_layout,
        ),
    ];
    for (name, requested, max) in layout_limits {
        if requested > max {
            return Err(RhiError::limit(name, requested, max));
        }
    }
    for stage in [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::COMPUTE,
    ] {
        let in_stage = |kind: fn(&BindingType) -> bool| {
            count(&move |entry: &BindGroupLayoutEntry| {
                entry.visibility.contains(stage) && kind(&entry.ty)
            })
        };
        let stage_limits = [
            (
                "max_uniform_buffers_per_shader_stage",
                in_stage(|ty| {
                    matches!(
                        ty,
                        BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            ..
                        }
                    )
                }),
                limits.max_uniform_buffers_per_shader_stage,
            ),
            (
                "max_storage_buffers_per_shader_stage",
                in_stage(|ty| {
                    matches!(
                        ty,
                        BindingType::Buffer {
                            ty: BufferBindingType::Storage { .. },
                            ..
                        }
                    )
                }),
                limits.max_storage_buffers_per_shader_stage,
            ),
            (
                "max_samplers_per_shader_stage",
                in_stage(|ty| matches!(ty, BindingType::Sampler(_))),
                limits.max_samplers_per_shader_stage,
            ),
            (
                "max_sampled_textures_per_shader_stage",
                in_stage(|ty| matches!(ty, BindingType::Texture { .. })),
                limits.max_sampled_textures_per_shader_stage,
            ),
            (
                "max_storage_textures_per_shader_stage",
                in_stage(|ty| matches!(ty, BindingType::StorageTexture { .. })),
                limits.max_storage_textures_per_shader_stage,
            ),
        ];
        for (name, requested, max) in stage_limits {
            if requested > max {
                return Err(RhiError::limit(name, requested, max));
            }
        }
    }
    Ok(())
}

const GROUP: ResourceKind = ResourceKind::BindGroup;

/// Checks a buffer range against its layout entry; returns the resolved size.
pub(crate) fn check_buffer_binding(
    ty: BufferBindingType,
    min_binding_size: Option<NonZeroU64>,
    buffer: &BufferInfo,
    binding: &BufferBinding,
    limits: &Limits,
) -> Result<u64, RhiError> {
    let (usage, alignment, max_size, max_name) = match ty {
        BufferBindingType::Uniform => (
            BufferUsages::UNIFORM,
            limits.min_uniform_buffer_offset_alignment,
            limits.max_uniform_buffer_binding_size,
            "max_uniform_buffer_binding_size",
        ),
        BufferBindingType::Storage { .. } => (
            BufferUsages::STORAGE,
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_binding_size,
            "max_storage_buffer_binding_size",
        ),
    };
    if !buffer.usage.contains(usage) {
        return Err(RhiError::invalid(
            GROUP,
            "the buffer lacks the UNIFORM or STORAGE usage its binding needs",
        ));
    }
    if !binding.offset.is_multiple_of(u64::from(alignment.max(1))) {
        return Err(RhiError::invalid(
            GROUP,
            "the buffer offset is not a multiple of the device's offset alignment",
        ));
    }
    let size = match binding.size {
        Some(size) => size.get(),
        None => buffer.size.saturating_sub(binding.offset),
    };
    let fits = binding
        .offset
        .checked_add(size)
        .is_some_and(|end| end <= buffer.size);
    if size == 0 || !fits {
        return Err(RhiError::invalid(
            GROUP,
            "the buffer range is empty or outside the buffer",
        ));
    }
    if size > max_size {
        return Err(RhiError::limit(max_name, size, max_size));
    }
    if matches!(ty, BufferBindingType::Storage { .. }) && !size.is_multiple_of(4) {
        return Err(RhiError::invalid(
            GROUP,
            "a storage buffer range is a multiple of 4 bytes",
        ));
    }
    if min_binding_size.is_some_and(|min| size < min.get()) {
        return Err(RhiError::invalid(
            GROUP,
            "the buffer range is smaller than the layout's min_binding_size",
        ));
    }
    Ok(size)
}

/// Checks a sampler against its layout entry.
pub(crate) fn check_sampler_binding(
    ty: SamplerBindingType,
    sampler: &SamplerInfo,
) -> Result<(), RhiError> {
    let filtering = [
        sampler.mag_filter,
        sampler.min_filter,
        sampler.mipmap_filter,
    ]
    .contains(&FilterMode::Linear);
    let fits = match ty {
        SamplerBindingType::Comparison => sampler.compare.is_some(),
        SamplerBindingType::Filtering => sampler.compare.is_none(),
        SamplerBindingType::NonFiltering => sampler.compare.is_none() && !filtering,
    };
    if fits {
        Ok(())
    } else {
        Err(RhiError::invalid(
            GROUP,
            "the sampler's comparison or filtering does not fit its binding",
        ))
    }
}

/// Checks a texture view against a texture or storage texture layout entry.
pub(crate) fn check_texture_binding(
    ty: &BindingType,
    view: &TextureViewInfo,
    texture: &TextureInfo,
) -> Result<(), RhiError> {
    match *ty {
        BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            if !texture.usage.contains(TextureUsages::TEXTURE_BINDING) {
                return Err(RhiError::invalid(
                    GROUP,
                    "the texture lacks TEXTURE_BINDING usage",
                ));
            }
            if view.dimension != view_dimension {
                return Err(RhiError::invalid(
                    GROUP,
                    "the view dimension differs from the binding's",
                ));
            }
            if (texture.sample_count > 1) != multisampled {
                return Err(RhiError::invalid(
                    GROUP,
                    "the texture's multisampling differs from the binding's",
                ));
            }
            if !sample_type_fits(view.format, view.aspect, sample_type) {
                return Err(RhiError::invalid(
                    GROUP,
                    "the view's format and aspect do not yield the binding's sample type",
                ));
            }
            Ok(())
        }
        BindingType::StorageTexture {
            format,
            view_dimension,
            ..
        } => {
            if !texture.usage.contains(TextureUsages::STORAGE_BINDING) {
                return Err(RhiError::invalid(
                    GROUP,
                    "the texture lacks STORAGE_BINDING usage",
                ));
            }
            if view.dimension != view_dimension || view.format != format {
                return Err(RhiError::invalid(
                    GROUP,
                    "the view's dimension or format differs from the storage binding's",
                ));
            }
            if view.mip_level_count != 1 {
                return Err(RhiError::invalid(
                    GROUP,
                    "a storage texture view has exactly one mip level",
                ));
            }
            Ok(())
        }
        BindingType::Buffer { .. } | BindingType::Sampler(_) => Err(RhiError::invalid(
            GROUP,
            "a texture view is bound to a buffer or sampler binding",
        )),
    }
}

/// Whether a view of `format` and `aspect` can be bound as `sample_type` (WebGPU's table for
/// the supported formats).
const fn sample_type_fits(
    format: TextureFormat,
    aspect: TextureAspect,
    sample_type: TextureSampleType,
) -> bool {
    if format.has_depth() {
        let depth = matches!(aspect, TextureAspect::DepthOnly)
            || (matches!(aspect, TextureAspect::All) && !format.has_stencil());
        let stencil = matches!(aspect, TextureAspect::StencilOnly);
        return match sample_type {
            TextureSampleType::Depth | TextureSampleType::Float { filterable: false } => depth,
            TextureSampleType::Uint => stencil,
            TextureSampleType::Float { filterable: true } | TextureSampleType::Sint => false,
        };
    }
    match format {
        TextureFormat::R32Uint => matches!(sample_type, TextureSampleType::Uint),
        TextureFormat::R32Float | TextureFormat::Rg32Float | TextureFormat::Rgba32Float => {
            matches!(sample_type, TextureSampleType::Float { filterable: false })
        }
        _ => matches!(sample_type, TextureSampleType::Float { .. }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::handle::{Handle as _, RawHandle};
    use crate::{AddressMode, CompareFunction, Extent3d, TextureDimension};

    fn reason(err: RhiError) -> &'static str {
        match err {
            RhiError::InvalidDescriptor { reason, .. } => reason,
            other => panic!("expected InvalidDescriptor, got {other:?}"),
        }
    }

    fn entry(binding: u32, visibility: ShaderStages, ty: BindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
        }
    }

    fn storage(read_only: bool) -> BindingType {
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }

    #[test]
    fn layout_entries_are_sorted_and_unique() {
        let limits = Limits::default();
        let entries = [
            entry(3, ShaderStages::FRAGMENT, BindingType::default()),
            entry(
                1,
                ShaderStages::FRAGMENT,
                BindingType::Sampler(SamplerBindingType::Filtering),
            ),
        ];
        let desc = BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        };
        let sorted = validate_bind_group_layout(&desc, &limits).unwrap();
        assert_eq!(sorted.iter().map(|e| e.binding).collect::<Vec<_>>(), [1, 3]);
        let twice = [entries[0], entries[0]];
        let desc = BindGroupLayoutDescriptor {
            label: None,
            entries: &twice,
        };
        assert_eq!(
            reason(validate_bind_group_layout(&desc, &limits).unwrap_err()),
            "a binding number appears twice"
        );
    }

    #[test]
    fn layout_rules() {
        let limits = Limits::default();
        let check = |entries: &[BindGroupLayoutEntry]| {
            validate_bind_group_layout(
                &BindGroupLayoutDescriptor {
                    label: None,
                    entries,
                },
                &limits,
            )
        };
        assert!(check(&[entry(0, ShaderStages::VERTEX, storage(false))]).is_err());
        assert!(check(&[entry(0, ShaderStages::VERTEX, storage(true))]).is_ok());
        assert!(matches!(
            check(&[entry(1000, ShaderStages::COMPUTE, storage(true))]),
            Err(RhiError::LimitExceeded {
                limit: "max_bindings_per_bind_group",
                ..
            })
        ));
        let rw_rgba = BindingType::StorageTexture {
            access: StorageTextureAccess::ReadWrite,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        };
        assert!(check(&[entry(0, ShaderStages::COMPUTE, rw_rgba)]).is_err());
        let ms_filterable = BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: true,
        };
        assert!(check(&[entry(0, ShaderStages::FRAGMENT, ms_filterable)]).is_err());
        // Nine storage buffers in one stage break max_storage_buffers_per_shader_stage (8).
        let nine: Vec<_> = (0..9)
            .map(|i| entry(i, ShaderStages::COMPUTE, storage(true)))
            .collect();
        assert!(matches!(
            check(&nine),
            Err(RhiError::LimitExceeded {
                limit: "max_storage_buffers_per_shader_stage",
                requested: 9,
                ..
            })
        ));
        // Spread over two stages they fit.
        let split: Vec<_> = (0..9)
            .map(|i| {
                let stage = if i < 5 {
                    ShaderStages::COMPUTE
                } else {
                    ShaderStages::FRAGMENT
                };
                entry(i, stage, storage(true))
            })
            .collect();
        assert!(check(&split).is_ok());
    }

    #[test]
    fn pipeline_layout_counts_groups_and_dynamic_offsets() {
        let limits = Limits::default();
        let dynamic = [entry(
            0,
            ShaderStages::COMPUTE,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: None,
            },
        )];
        let groups: Vec<&[BindGroupLayoutEntry]> = vec![&dynamic; 4];
        assert!(validate_pipeline_layout(&groups, &limits).is_ok());
        let groups: Vec<&[BindGroupLayoutEntry]> = vec![&dynamic; 5];
        assert!(matches!(
            validate_pipeline_layout(&groups, &limits),
            Err(RhiError::LimitExceeded {
                limit: "max_bind_groups",
                ..
            })
        ));
        let tight = Limits {
            max_dynamic_storage_buffers_per_pipeline_layout: 3,
            ..Limits::default()
        };
        let groups: Vec<&[BindGroupLayoutEntry]> = vec![&dynamic; 4];
        assert!(matches!(
            validate_pipeline_layout(&groups, &tight),
            Err(RhiError::LimitExceeded {
                limit: "max_dynamic_storage_buffers_per_pipeline_layout",
                ..
            })
        ));
    }

    fn buffer_handle() -> Buffer {
        Buffer::from_raw(RawHandle {
            device: 1,
            index: 0,
            generation: 0,
        })
    }

    #[test]
    fn buffer_binding_rules() {
        let limits = Limits::default();
        let info = BufferInfo {
            size: 1024,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE,
        };
        let range = |offset: u64, size: Option<u64>| BufferBinding {
            buffer: buffer_handle(),
            offset,
            size: size.and_then(NonZeroU64::new),
        };
        let uniform = BufferBindingType::Uniform;
        assert_eq!(
            check_buffer_binding(uniform, None, &info, &range(0, None), &limits),
            Ok(1024)
        );
        assert_eq!(
            check_buffer_binding(uniform, None, &info, &range(256, Some(64)), &limits),
            Ok(64)
        );
        assert!(check_buffer_binding(uniform, None, &info, &range(4, None), &limits).is_err());
        assert!(check_buffer_binding(uniform, None, &info, &range(1024, None), &limits).is_err());
        assert!(
            check_buffer_binding(uniform, None, &info, &range(768, Some(512)), &limits).is_err()
        );
        let min = NonZeroU64::new(128);
        assert!(check_buffer_binding(uniform, min, &info, &range(0, Some(64)), &limits).is_err());
        let storage = BufferBindingType::Storage { read_only: true };
        assert!(check_buffer_binding(storage, None, &info, &range(0, Some(6)), &limits).is_err());
        let vertex_only = BufferInfo {
            usage: BufferUsages::VERTEX,
            ..info
        };
        assert!(
            check_buffer_binding(uniform, None, &vertex_only, &range(0, None), &limits).is_err()
        );
    }

    #[test]
    fn sampler_binding_rules() {
        let nearest =
            crate::resource::validate_sampler(&crate::SamplerDescriptor::default()).unwrap();
        let linear = SamplerInfo {
            mag_filter: FilterMode::Linear,
            ..nearest
        };
        let compare = SamplerInfo {
            compare: Some(CompareFunction::Less),
            address_modes: [AddressMode::ClampToEdge; 3],
            ..nearest
        };
        assert!(check_sampler_binding(SamplerBindingType::NonFiltering, &nearest).is_ok());
        assert!(check_sampler_binding(SamplerBindingType::NonFiltering, &linear).is_err());
        assert!(check_sampler_binding(SamplerBindingType::Filtering, &linear).is_ok());
        assert!(check_sampler_binding(SamplerBindingType::Filtering, &compare).is_err());
        assert!(check_sampler_binding(SamplerBindingType::Comparison, &compare).is_ok());
        assert!(check_sampler_binding(SamplerBindingType::Comparison, &nearest).is_err());
    }

    #[test]
    fn sample_types_follow_the_format() {
        use TextureSampleType as S;
        let float = S::Float { filterable: true };
        let unfilterable = S::Float { filterable: false };
        assert!(sample_type_fits(
            TextureFormat::Rgba8Unorm,
            TextureAspect::All,
            float
        ));
        assert!(!sample_type_fits(
            TextureFormat::Rgba32Float,
            TextureAspect::All,
            float
        ));
        assert!(sample_type_fits(
            TextureFormat::Rgba32Float,
            TextureAspect::All,
            unfilterable
        ));
        assert!(sample_type_fits(
            TextureFormat::R32Uint,
            TextureAspect::All,
            S::Uint
        ));
        assert!(sample_type_fits(
            TextureFormat::Depth32Float,
            TextureAspect::All,
            S::Depth
        ));
        assert!(!sample_type_fits(
            TextureFormat::Depth24PlusStencil8,
            TextureAspect::All,
            S::Depth
        ));
        assert!(sample_type_fits(
            TextureFormat::Depth24PlusStencil8,
            TextureAspect::StencilOnly,
            S::Uint
        ));
    }

    #[test]
    fn texture_binding_rules() {
        let texture = TextureInfo {
            size: Extent3d::default(),
            mip_level_count: 2,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
        };
        let view = TextureViewInfo {
            texture: crate::Texture::from_raw(RawHandle {
                device: 1,
                index: 0,
                generation: 0,
            }),
            format: TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::D2,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: 2,
            base_array_layer: 0,
            array_layer_count: 1,
        };
        let sampled = BindingType::Texture {
            sample_type: TextureSampleType::default(),
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        };
        assert!(check_texture_binding(&sampled, &view, &texture).is_ok());
        let cube = BindingType::Texture {
            sample_type: TextureSampleType::default(),
            view_dimension: TextureViewDimension::Cube,
            multisampled: false,
        };
        assert!(check_texture_binding(&cube, &view, &texture).is_err());
        let storage = BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        };
        assert_eq!(
            reason(check_texture_binding(&storage, &view, &texture).unwrap_err()),
            "the texture lacks STORAGE_BINDING usage"
        );
        let storage_texture = TextureInfo {
            usage: TextureUsages::STORAGE_BINDING,
            ..texture
        };
        assert_eq!(
            reason(check_texture_binding(&storage, &view, &storage_texture).unwrap_err()),
            "a storage texture view has exactly one mip level"
        );
        let one_mip = TextureViewInfo {
            mip_level_count: 1,
            ..view
        };
        assert!(check_texture_binding(&storage, &one_mip, &storage_texture).is_ok());
        assert!(check_texture_binding(&BindingType::default(), &view, &texture).is_err());
    }
}
