//! Conversions from `pragma_rhi`'s descriptor types to wgpu's. Every mapping is total: a new
//! variant on our side is a compile error here, not a silent default.

use crate::binding::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType, StorageTextureAccess,
    TextureSampleType,
};
use crate::pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrites,
    DepthStencilKey, Face, FrontFace, IndexFormat, MultisampleState, PrimitiveState,
    PrimitiveTopology, StencilFaceState, StencilOperation, VertexFormat, VertexStepMode,
};
use crate::{
    AddressMode, BufferUsages, CompareFunction, FilterMode, ShaderStages, TextureAspect,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDimension,
};

fn map_flags<O: bitflags::Flags + Copy, T: bitflags::Flags + Copy>(ours: O, pairs: &[(O, T)]) -> T {
    pairs
        .iter()
        .filter(|(flag, _)| ours.contains(*flag))
        .fold(T::empty(), |acc, (_, theirs)| acc.union(*theirs))
}

pub(super) fn buffer_usages(usage: BufferUsages) -> wgpu::BufferUsages {
    map_flags(
        usage,
        &[
            (BufferUsages::MAP_READ, wgpu::BufferUsages::MAP_READ),
            (BufferUsages::MAP_WRITE, wgpu::BufferUsages::MAP_WRITE),
            (BufferUsages::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
            (BufferUsages::COPY_DST, wgpu::BufferUsages::COPY_DST),
            (BufferUsages::INDEX, wgpu::BufferUsages::INDEX),
            (BufferUsages::VERTEX, wgpu::BufferUsages::VERTEX),
            (BufferUsages::UNIFORM, wgpu::BufferUsages::UNIFORM),
            (BufferUsages::STORAGE, wgpu::BufferUsages::STORAGE),
            (BufferUsages::INDIRECT, wgpu::BufferUsages::INDIRECT),
        ],
    )
}

pub(super) fn texture_usages(usage: TextureUsages) -> wgpu::TextureUsages {
    map_flags(
        usage,
        &[
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
        ],
    )
}

pub(super) const fn texture_dimension(dimension: TextureDimension) -> wgpu::TextureDimension {
    match dimension {
        TextureDimension::D1 => wgpu::TextureDimension::D1,
        TextureDimension::D2 => wgpu::TextureDimension::D2,
        TextureDimension::D3 => wgpu::TextureDimension::D3,
    }
}

pub(super) const fn texture_aspect(aspect: TextureAspect) -> wgpu::TextureAspect {
    match aspect {
        TextureAspect::All => wgpu::TextureAspect::All,
        TextureAspect::DepthOnly => wgpu::TextureAspect::DepthOnly,
        TextureAspect::StencilOnly => wgpu::TextureAspect::StencilOnly,
    }
}

pub(super) const fn address_mode(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

pub(super) const fn filter_mode(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

pub(super) const fn mipmap_filter_mode(mode: FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

pub(super) fn shader_stages(stages: ShaderStages) -> wgpu::ShaderStages {
    map_flags(
        stages,
        &[
            (ShaderStages::VERTEX, wgpu::ShaderStages::VERTEX),
            (ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT),
            (ShaderStages::COMPUTE, wgpu::ShaderStages::COMPUTE),
        ],
    )
}

pub(super) fn color_writes(writes: ColorWrites) -> wgpu::ColorWrites {
    map_flags(
        writes,
        &[
            (ColorWrites::RED, wgpu::ColorWrites::RED),
            (ColorWrites::GREEN, wgpu::ColorWrites::GREEN),
            (ColorWrites::BLUE, wgpu::ColorWrites::BLUE),
            (ColorWrites::ALPHA, wgpu::ColorWrites::ALPHA),
        ],
    )
}

pub(super) const fn texture_format(format: TextureFormat) -> wgpu::TextureFormat {
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

pub(super) const fn view_dimension(dimension: TextureViewDimension) -> wgpu::TextureViewDimension {
    match dimension {
        TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
        TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
    }
}

pub(super) const fn compare_function(compare: CompareFunction) -> wgpu::CompareFunction {
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

pub(super) fn bind_group_layout_entry(entry: &BindGroupLayoutEntry) -> wgpu::BindGroupLayoutEntry {
    let ty = match entry.ty {
        BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: match ty {
                BufferBindingType::Uniform => wgpu::BufferBindingType::Uniform,
                BufferBindingType::Storage { read_only } => {
                    wgpu::BufferBindingType::Storage { read_only }
                }
            },
            has_dynamic_offset,
            min_binding_size,
        },
        BindingType::Sampler(ty) => wgpu::BindingType::Sampler(match ty {
            SamplerBindingType::Filtering => wgpu::SamplerBindingType::Filtering,
            SamplerBindingType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            SamplerBindingType::Comparison => wgpu::SamplerBindingType::Comparison,
        }),
        BindingType::Texture {
            sample_type,
            view_dimension: dimension,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: match sample_type {
                TextureSampleType::Float { filterable } => {
                    wgpu::TextureSampleType::Float { filterable }
                }
                TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
                TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
                TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
            },
            view_dimension: view_dimension(dimension),
            multisampled,
        },
        BindingType::StorageTexture {
            access,
            format,
            view_dimension: dimension,
        } => wgpu::BindingType::StorageTexture {
            access: match access {
                StorageTextureAccess::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
                StorageTextureAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                StorageTextureAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
            },
            format: texture_format(format),
            view_dimension: view_dimension(dimension),
        },
    };
    wgpu::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: shader_stages(entry.visibility),
        ty,
        count: None,
    }
}

pub(super) const fn vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    match format {
        VertexFormat::Uint8x2 => wgpu::VertexFormat::Uint8x2,
        VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
        VertexFormat::Sint8x2 => wgpu::VertexFormat::Sint8x2,
        VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
        VertexFormat::Unorm8x2 => wgpu::VertexFormat::Unorm8x2,
        VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        VertexFormat::Snorm8x2 => wgpu::VertexFormat::Snorm8x2,
        VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
        VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
        VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
        VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
        VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
        VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
        VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
        VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
        VertexFormat::Float16x4 => wgpu::VertexFormat::Float16x4,
        VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
        VertexFormat::Uint32x2 => wgpu::VertexFormat::Uint32x2,
        VertexFormat::Uint32x3 => wgpu::VertexFormat::Uint32x3,
        VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
        VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
        VertexFormat::Sint32x2 => wgpu::VertexFormat::Sint32x2,
        VertexFormat::Sint32x3 => wgpu::VertexFormat::Sint32x3,
        VertexFormat::Sint32x4 => wgpu::VertexFormat::Sint32x4,
    }
}

pub(super) const fn step_mode(mode: VertexStepMode) -> wgpu::VertexStepMode {
    match mode {
        VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

pub(super) fn primitive(state: &PrimitiveState) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: match state.topology {
            PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
            PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
            PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        },
        strip_index_format: state.strip_index_format.map(|format| match format {
            IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
            IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
        }),
        front_face: match state.front_face {
            FrontFace::Ccw => wgpu::FrontFace::Ccw,
            FrontFace::Cw => wgpu::FrontFace::Cw,
        },
        cull_mode: state.cull_mode.map(|face| match face {
            Face::Front => wgpu::Face::Front,
            Face::Back => wgpu::Face::Back,
        }),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    }
}

const fn stencil_operation(op: StencilOperation) -> wgpu::StencilOperation {
    match op {
        StencilOperation::Keep => wgpu::StencilOperation::Keep,
        StencilOperation::Zero => wgpu::StencilOperation::Zero,
        StencilOperation::Replace => wgpu::StencilOperation::Replace,
        StencilOperation::Invert => wgpu::StencilOperation::Invert,
        StencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        StencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
    }
}

const fn stencil_face(face: &StencilFaceState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare_function(face.compare),
        fail_op: stencil_operation(face.fail_op),
        depth_fail_op: stencil_operation(face.depth_fail_op),
        pass_op: stencil_operation(face.pass_op),
    }
}

pub(super) fn depth_stencil(state: &DepthStencilKey) -> wgpu::DepthStencilState {
    let bias = state.bias();
    wgpu::DepthStencilState {
        format: texture_format(state.format),
        depth_write_enabled: Some(state.depth_write_enabled),
        depth_compare: Some(compare_function(state.depth_compare)),
        stencil: wgpu::StencilState {
            front: stencil_face(&state.stencil.front),
            back: stencil_face(&state.stencil.back),
            read_mask: state.stencil.read_mask,
            write_mask: state.stencil.write_mask,
        },
        bias: wgpu::DepthBiasState {
            constant: bias.constant,
            slope_scale: bias.slope_scale,
            clamp: bias.clamp,
        },
    }
}

pub(super) const fn multisample(state: &MultisampleState) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: state.count,
        mask: state.mask,
        alpha_to_coverage_enabled: state.alpha_to_coverage_enabled,
    }
}

const fn blend_factor(factor: BlendFactor) -> wgpu::BlendFactor {
    match factor {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::Src => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::Dst => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => wgpu::BlendFactor::SrcAlphaSaturated,
        BlendFactor::Constant => wgpu::BlendFactor::Constant,
        BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    }
}

const fn blend_component(component: &BlendComponent) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: blend_factor(component.src_factor),
        dst_factor: blend_factor(component.dst_factor),
        operation: match component.operation {
            BlendOperation::Add => wgpu::BlendOperation::Add,
            BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
            BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            BlendOperation::Min => wgpu::BlendOperation::Min,
            BlendOperation::Max => wgpu::BlendOperation::Max,
        },
    }
}

const fn blend(state: &BlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: blend_component(&state.color),
        alpha: blend_component(&state.alpha),
    }
}

pub(super) fn color_target(target: &ColorTargetState) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: texture_format(target.format),
        blend: target.blend.as_ref().map(blend),
        write_mask: color_writes(target.write_mask),
    }
}

#[cfg(test)]
mod tests {
    use super::{buffer_usages, shader_stages, texture_usages};
    use crate::{BufferUsages, ShaderStages, TextureUsages};

    #[test]
    fn flags_map_bit_for_bit() {
        assert_eq!(
            buffer_usages(BufferUsages::all()).bits().count_ones(),
            BufferUsages::all().bits().count_ones()
        );
        assert_eq!(
            texture_usages(TextureUsages::all()).bits().count_ones(),
            TextureUsages::all().bits().count_ones()
        );
        assert_eq!(
            shader_stages(ShaderStages::all()),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE
        );
        assert_eq!(
            buffer_usages(BufferUsages::MAP_READ | BufferUsages::COPY_DST),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        );
    }
}
