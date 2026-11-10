//! Render and compute pipeline descriptors, their validation and their cache keys.
//!
//! A pipeline is validated in two parts: the descriptor's own rules (vertex layouts, primitive,
//! depth/stencil, multisample and color target states, against the device's [`Limits`]) and the
//! shader interface — every resource an entry point uses must be in the pipeline layout with a
//! fitting type and visibility, every vertex input needs an attribute of the same scalar kind,
//! every written color target a fragment output, every fragment input a vertex output, and a
//! compute workgroup must fit the limits. Both run before a backend sees the pipeline.

use bitflags::bitflags;

use crate::binding::{BindGroupLayoutEntry, BindingType, BufferBindingType, SamplerBindingType};
use crate::shader::{EntryReflection, IoKind, ShaderBindingType, ShaderSampleKind};
use crate::{
    CompareFunction, Limits, PipelineLayout, ResourceKind, RhiError, ShaderModule, ShaderStage,
    TextureFormat, TextureSampleType,
};

// ---------------------------------------------------------------------------------------------
// Vertex input

/// The format of one vertex attribute.
///
/// ```
/// use pragma_rhi::VertexFormat;
///
/// // `vec2<f32>` positions and `vec4<f32>` colors packed as four normalized bytes.
/// let formats = [VertexFormat::Float32x2, VertexFormat::Unorm8x4];
/// # let _ = formats;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs, reason = "the names say it: components x count")]
pub enum VertexFormat {
    Uint8x2,
    Uint8x4,
    Sint8x2,
    Sint8x4,
    Unorm8x2,
    Unorm8x4,
    Snorm8x2,
    Snorm8x4,
    Uint16x2,
    Uint16x4,
    Sint16x2,
    Sint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl VertexFormat {
    /// The size in bytes.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }

    /// The scalar kind a shader reads it as.
    pub(crate) const fn kind(self) -> IoKind {
        match self {
            Self::Uint8x2
            | Self::Uint8x4
            | Self::Uint16x2
            | Self::Uint16x4
            | Self::Uint32
            | Self::Uint32x2
            | Self::Uint32x3
            | Self::Uint32x4 => IoKind::Uint,
            Self::Sint8x2
            | Self::Sint8x4
            | Self::Sint16x2
            | Self::Sint16x4
            | Self::Sint32
            | Self::Sint32x2
            | Self::Sint32x3
            | Self::Sint32x4 => IoKind::Sint,
            _ => IoKind::Float,
        }
    }
}

/// Whether a vertex buffer advances per vertex or per instance.
///
/// ```
/// use pragma_rhi::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};
///
/// // One `vec4<f32>` offset per instance, at `@location(1)`.
/// let per_instance = VertexBufferLayout {
///     array_stride: 16,
///     step_mode: VertexStepMode::Instance,
///     attributes: &[VertexAttribute { format: VertexFormat::Float32x4, offset: 0, shader_location: 1 }],
/// };
/// # let _ = per_instance;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VertexStepMode {
    /// Per vertex.
    #[default]
    Vertex,
    /// Per instance.
    Instance,
}

/// One attribute of a vertex buffer.
///
/// ```
/// use pragma_rhi::{VertexAttribute, VertexFormat};
///
/// // An interleaved vertex: position (8 bytes), then UV (8 bytes).
/// let attributes = [
///     VertexAttribute { format: VertexFormat::Float32x2, offset: 0, shader_location: 0 },
///     VertexAttribute { format: VertexFormat::Float32x2, offset: 8, shader_location: 1 },
/// ];
/// # let _ = attributes;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// The format.
    pub format: VertexFormat,
    /// The byte offset in the vertex; a multiple of `min(4, format.size())`.
    pub offset: u64,
    /// The `@location` it feeds; unique over all buffers.
    pub shader_location: u32,
}

/// The layout of one vertex buffer.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let attributes = [VertexAttribute { format: VertexFormat::Float32x2, offset: 0, shader_location: 0 }];
/// let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: Some("mesh"),
///     layout,
///     vertex: VertexState {
///         module,
///         entry_point: Some("vs_mesh"),
///         buffers: &[VertexBufferLayout { array_stride: 8, step_mode: VertexStepMode::Vertex, attributes: &attributes }],
///     },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState::default(),
///     fragment: Some(FragmentState {
///         module,
///         entry_point: Some("fs"),
///         targets: &[Some(TextureFormat::Rgba8Unorm.into())],
///     }),
/// })?;
/// assert_eq!(device.render_pipeline_info(pipeline)?.vertex_strides, [8]);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VertexBufferLayout<'a> {
    /// The bytes between consecutive elements; a multiple of 4, at most
    /// [`Limits::max_vertex_buffer_array_stride`]. 0 repeats one element.
    pub array_stride: u64,
    /// Per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes.
    pub attributes: &'a [VertexAttribute],
}

/// The vertex stage of a render pipeline.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// // `vs_mesh` reads `@location(0)`: a pipeline without a vertex buffer for it is rejected.
/// let result = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: None,
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs_mesh"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState::default(),
///     fragment: None,
/// });
/// assert!(matches!(result, Err(RhiError::ShaderInterface { stage: ShaderStage::Vertex, .. })));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexState<'a> {
    /// The module.
    pub module: ShaderModule,
    /// The entry point; `None` takes the module's only vertex entry point.
    pub entry_point: Option<&'a str>,
    /// The vertex buffers, by slot; at most [`Limits::max_vertex_buffers`].
    pub buffers: &'a [VertexBufferLayout<'a>],
}

// ---------------------------------------------------------------------------------------------
// Primitive, depth/stencil, multisample

/// How vertices form primitives.
///
/// ```
/// use pragma_rhi::{IndexFormat, PrimitiveState, PrimitiveTopology};
///
/// // A strip with primitive restart at 0xFFFF.
/// let strip = PrimitiveState {
///     topology: PrimitiveTopology::TriangleStrip,
///     strip_index_format: Some(IndexFormat::Uint16),
///     ..PrimitiveState::default()
/// };
/// assert_eq!(PrimitiveState::default().topology, PrimitiveTopology::TriangleList);
/// # let _ = strip;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PrimitiveTopology {
    /// Points.
    PointList,
    /// Separate lines.
    LineList,
    /// A connected line strip.
    LineStrip,
    /// Separate triangles.
    #[default]
    TriangleList,
    /// A connected triangle strip.
    TriangleStrip,
}

impl PrimitiveTopology {
    const fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }
}

/// The type of indices.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let indices = device.create_buffer(&BufferDescriptor {
///     label: Some("indices"),
///     size: 6 * 2,
///     usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
/// })?;
/// # let target = device.create_texture(&TextureDescriptor { usage: TextureUsages::RENDER_ATTACHMENT, ..Default::default() })?;
/// # let view = device.create_texture_view(target, &TextureViewDescriptor::default())?;
/// let mut encoder = CommandEncoder::new(None);
/// let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
///     color_attachments: &[Some(RenderPassColorAttachment::clear(view, Color::BLACK))],
///     ..RenderPassDescriptor::default()
/// });
/// pass.set_index_buffer(indices, IndexFormat::Uint16, 0, None);
/// pass.end();
/// device.submit([encoder.finish()])?;
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    /// 16-bit indices.
    Uint16,
    /// 32-bit indices.
    Uint32,
}

/// The winding of front-facing triangles.
///
/// ```
/// use pragma_rhi::{Face, FrontFace, PrimitiveState};
///
/// // Clockwise front faces (a left-handed convention), back faces culled.
/// let primitive = PrimitiveState { front_face: FrontFace::Cw, cull_mode: Some(Face::Back), ..PrimitiveState::default() };
/// # let _ = primitive;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrontFace {
    /// Counter-clockwise.
    #[default]
    Ccw,
    /// Clockwise.
    Cw,
}

/// A face of a triangle.
///
/// ```
/// use pragma_rhi::{Face, PrimitiveState};
///
/// // Draw only back faces, e.g. for the inside of a skybox.
/// let inside = PrimitiveState { cull_mode: Some(Face::Front), ..PrimitiveState::default() };
/// assert_eq!(PrimitiveState::default().cull_mode, None);
/// # let _ = inside;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    /// The front face.
    Front,
    /// The back face.
    Back,
}

/// Primitive assembly and rasterization.
///
/// ```
/// use pragma_rhi::{Face, FrontFace, PrimitiveState, PrimitiveTopology};
///
/// let primitive = PrimitiveState::default();
/// assert_eq!(primitive.topology, PrimitiveTopology::TriangleList);
/// assert_eq!((primitive.front_face, primitive.cull_mode), (FrontFace::Ccw, None));
/// let culled = PrimitiveState { cull_mode: Some(Face::Back), ..primitive };
/// # let _ = culled;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PrimitiveState {
    /// The topology.
    pub topology: PrimitiveTopology,
    /// The index format of indexed strip draws (strip topologies only).
    pub strip_index_format: Option<IndexFormat>,
    /// The front face.
    pub front_face: FrontFace,
    /// The face to cull, if any.
    pub cull_mode: Option<Face>,
}

/// What a stencil test does to the stencil value.
///
/// ```
/// use pragma_rhi::{CompareFunction, StencilFaceState, StencilOperation};
///
/// // Mark every pixel a mask covers: write the reference value where the test passes.
/// let mark = StencilFaceState {
///     compare: CompareFunction::Always,
///     pass_op: StencilOperation::Replace,
///     ..StencilFaceState::IGNORE
/// };
/// # let _ = mark;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs, reason = "WebGPU's stencil operations")]
pub enum StencilOperation {
    #[default]
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

/// The stencil test of one face.
///
/// ```
/// use pragma_rhi::{CompareFunction, StencilFaceState, StencilOperation};
///
/// // Draw only where the stencil equals the reference, keeping it unchanged.
/// let inside_mask = StencilFaceState { compare: CompareFunction::Equal, ..StencilFaceState::IGNORE };
/// assert_eq!(StencilFaceState::default(), StencilFaceState::IGNORE);
/// assert_eq!(inside_mask.pass_op, StencilOperation::Keep);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// The comparison.
    pub compare: CompareFunction,
    /// The operation when the stencil test fails.
    pub fail_op: StencilOperation,
    /// The operation when the stencil test passes and the depth test fails.
    pub depth_fail_op: StencilOperation,
    /// The operation when both pass.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// A face state that leaves the stencil alone.
    pub const IGNORE: Self = Self {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

/// The stencil test.
///
/// ```
/// use pragma_rhi::{CompareFunction, DepthStencilState, StencilFaceState, StencilState, TextureFormat};
///
/// let face = StencilFaceState { compare: CompareFunction::Equal, ..StencilFaceState::IGNORE };
/// let masked = DepthStencilState {
///     format: TextureFormat::Depth24PlusStencil8,
///     stencil: StencilState { front: face, back: face, ..StencilState::default() },
///     ..DepthStencilState::default()
/// };
/// # let _ = masked;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// The front face.
    pub front: StencilFaceState,
    /// The back face.
    pub back: StencilFaceState,
    /// The bits read.
    pub read_mask: u32,
    /// The bits written.
    pub write_mask: u32,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            front: StencilFaceState::IGNORE,
            back: StencilFaceState::IGNORE,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

impl StencilState {
    /// Whether the state does anything.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilFaceState::IGNORE || self.back != StencilFaceState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
}

/// The depth bias.
///
/// ```
/// use pragma_rhi::{DepthBiasState, DepthStencilState};
///
/// // Push shadow casters away from the light to avoid shadow acne.
/// let shadow_caster = DepthStencilState {
///     bias: DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
///     ..DepthStencilState::default()
/// };
/// # let _ = shadow_caster;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DepthBiasState {
    /// The constant bias.
    pub constant: i32,
    /// The bias scaled by the slope; finite.
    pub slope_scale: f32,
    /// The largest bias (0 for none); finite.
    pub clamp: f32,
}

impl DepthBiasState {
    /// Whether the state biases anything.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.constant != 0 || self.slope_scale != 0.0
    }
}

/// The depth/stencil state of a render pipeline.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: Some("depth prepass"),
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: Some(DepthStencilState::default()),
///     multisample: MultisampleState::default(),
///     fragment: None,
/// })?;
/// let info = device.render_pipeline_info(pipeline)?;
/// assert_eq!(info.depth_stencil_format, Some(TextureFormat::Depth32Float));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilState {
    /// The depth/stencil attachment's format; a depth format.
    pub format: TextureFormat,
    /// Whether depth is written.
    pub depth_write_enabled: bool,
    /// The depth test.
    pub depth_compare: CompareFunction,
    /// The stencil test; only for a format with stencil.
    pub stencil: StencilState,
    /// The depth bias; only for triangle topologies.
    pub bias: DepthBiasState,
}

impl Default for DepthStencilState {
    /// `Depth32Float`, written, passing when closer (`Less`).
    fn default() -> Self {
        Self {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }
    }
}

/// Multisampling.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: Some("msaa"),
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState { count: 4, ..MultisampleState::default() },
///     fragment: Some(FragmentState { module, entry_point: Some("fs"), targets: &[Some(TextureFormat::Rgba8Unorm.into())] }),
/// })?;
/// assert_eq!(device.render_pipeline_info(pipeline)?.sample_count, 4);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    /// 1 or 4; every attachment must have this sample count.
    pub count: u32,
    /// The samples written.
    pub mask: u64,
    /// Whether alpha drives coverage (needs `count > 1`).
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Color targets

/// A blend factor.
///
/// ```
/// use pragma_rhi::{BlendComponent, BlendFactor, BlendOperation};
///
/// // Additive blending (particles, light accumulation): dst + src.
/// let additive = BlendComponent {
///     src_factor: BlendFactor::One,
///     dst_factor: BlendFactor::One,
///     operation: BlendOperation::Add,
/// };
/// # let _ = additive;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs, reason = "WebGPU's blend factors")]
#[non_exhaustive]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

/// A blend operation.
///
/// ```
/// use pragma_rhi::{BlendComponent, BlendFactor, BlendOperation};
///
/// // Keep the larger value, e.g. for a max-intensity projection.
/// let max = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Max };
/// # let _ = max;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs, reason = "WebGPU's blend operations")]
pub enum BlendOperation {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    /// Needs both factors to be [`BlendFactor::One`].
    Min,
    /// Needs both factors to be [`BlendFactor::One`].
    Max,
}

/// The blending of the color or the alpha channel.
///
/// ```
/// use pragma_rhi::{BlendComponent, BlendState};
///
/// assert_eq!(BlendComponent::default(), BlendComponent::REPLACE);
/// // Straight alpha: color "over", alpha "over" too.
/// assert_eq!(BlendState::ALPHA_BLENDING.alpha, BlendComponent::OVER);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// The factor of the new value.
    pub src_factor: BlendFactor,
    /// The factor of the old value.
    pub dst_factor: BlendFactor,
    /// How they combine.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// The new value replaces the old one.
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };
    /// Premultiplied "over".
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self::REPLACE
    }
}

/// The blending of a color target.
///
/// ```
/// use pragma_rhi::{BlendState, ColorTargetState, ColorWrites, TextureFormat};
///
/// let transparent = ColorTargetState {
///     format: TextureFormat::Bgra8UnormSrgb,
///     blend: Some(BlendState::ALPHA_BLENDING),
///     write_mask: ColorWrites::ALL,
/// };
/// # let _ = transparent;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlendState {
    /// The color channels.
    pub color: BlendComponent,
    /// The alpha channel.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// No blending.
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };
    /// Straight alpha blending.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };
    /// Premultiplied alpha blending.
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

bitflags! {
    /// The channels a color target writes.
    ///
    /// ```
    /// use pragma_rhi::{ColorTargetState, ColorWrites, TextureFormat};
    ///
    /// // Write color but keep the alpha already in the target.
    /// let target = ColorTargetState { write_mask: ColorWrites::COLOR, ..TextureFormat::Rgba8Unorm.into() };
    /// assert_eq!(ColorWrites::default(), ColorWrites::ALL);
    /// # let _ = target;
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ColorWrites: u8 {
        /// Red.
        const RED = 1 << 0;
        /// Green.
        const GREEN = 1 << 1;
        /// Blue.
        const BLUE = 1 << 2;
        /// Alpha.
        const ALPHA = 1 << 3;
        /// Red, green and blue.
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        /// Every channel.
        const ALL = Self::COLOR.bits() | Self::ALPHA.bits();
    }
}

impl Default for ColorWrites {
    fn default() -> Self {
        Self::ALL
    }
}

/// One color target of a render pipeline.
///
/// ```
/// use pragma_rhi::{ColorTargetState, ColorWrites, TextureFormat};
///
/// // From a format: no blending, every channel written.
/// let target = ColorTargetState::from(TextureFormat::Rgba16Float);
/// assert_eq!((target.blend, target.write_mask), (None, ColorWrites::ALL));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ColorTargetState {
    /// The attachment's format; a color format.
    pub format: TextureFormat,
    /// The blending; only for blendable formats (not `R32Uint` or the 32-bit floats).
    pub blend: Option<BlendState>,
    /// The channels written.
    pub write_mask: ColorWrites,
}

impl From<TextureFormat> for ColorTargetState {
    fn from(format: TextureFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }
}

/// The fragment stage of a render pipeline.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// // Two color targets; the second slot is unused (`None`) by this pipeline.
/// let targets = [Some(TextureFormat::Rgba8Unorm.into()), None];
/// let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: None,
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState::default(),
///     fragment: Some(FragmentState { module, entry_point: Some("fs"), targets: &targets }),
/// })?;
/// assert_eq!(device.render_pipeline_info(pipeline)?.color_formats, [Some(TextureFormat::Rgba8Unorm), None]);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentState<'a> {
    /// The module.
    pub module: ShaderModule,
    /// The entry point; `None` takes the module's only fragment entry point.
    pub entry_point: Option<&'a str>,
    /// The color targets, by attachment index (`None` leaves one out); at most
    /// [`Limits::max_color_attachments`].
    pub targets: &'a [Option<ColorTargetState>],
}

// ---------------------------------------------------------------------------------------------
// Pipelines

/// Describes a [`RenderPipeline`](crate::RenderPipeline).
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let targets = [Some(TextureFormat::Rgba8Unorm.into())];
/// let desc = RenderPipelineDescriptor {
///     label: Some("opaque"),
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState::default(),
///     fragment: Some(FragmentState { module, entry_point: Some("fs"), targets: &targets }),
/// };
/// let pipeline = device.create_render_pipeline(&desc)?;
/// // The same descriptor again is a cache hit: the backend compiles nothing.
/// let again = device.create_render_pipeline(&desc)?;
/// assert_ne!(pipeline, again);
/// assert_eq!(device.pipeline_cache_stats().hits, 1);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPipelineDescriptor<'a> {
    /// A debug label, passed to the backend (a pipeline taken from the cache keeps the label it
    /// was compiled with).
    pub label: Option<&'a str>,
    /// The layout; it cannot be destroyed while the pipeline lives.
    pub layout: PipelineLayout,
    /// The vertex stage.
    pub vertex: VertexState<'a>,
    /// Primitive assembly.
    pub primitive: PrimitiveState,
    /// The depth/stencil state, if the pipeline has a depth/stencil attachment.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisampling.
    pub multisample: MultisampleState,
    /// The fragment stage, if any.
    pub fragment: Option<FragmentState<'a>>,
}

/// Describes a [`ComputePipeline`](crate::ComputePipeline).
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
///     label: Some("simulate"),
///     layout,
///     module,
///     entry_point: Some("cs"),
/// })?;
/// assert_eq!(device.compute_pipeline_info(pipeline)?.workgroup_size, Some([64, 1, 1]));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The layout; it cannot be destroyed while the pipeline lives.
    pub layout: PipelineLayout,
    /// The module.
    pub module: ShaderModule,
    /// The entry point; `None` takes the module's only compute entry point.
    pub entry_point: Option<&'a str>,
}

/// A live render pipeline's properties.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
///     label: None,
///     layout,
///     vertex: VertexState { module, entry_point: Some("vs"), buffers: &[] },
///     primitive: PrimitiveState::default(),
///     depth_stencil: None,
///     multisample: MultisampleState::default(),
///     fragment: Some(FragmentState { module, entry_point: Some("fs"), targets: &[Some(TextureFormat::Rgba8Unorm.into())] }),
/// })?;
/// let info = device.render_pipeline_info(pipeline)?;
/// assert_eq!(info.layout, layout);
/// assert_eq!((info.color_formats.as_slice(), info.sample_count), (&[Some(TextureFormat::Rgba8Unorm)][..], 1));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RenderPipelineInfo {
    /// The layout.
    pub layout: PipelineLayout,
    /// The color target formats, by attachment index.
    pub color_formats: Vec<Option<TextureFormat>>,
    /// The depth/stencil format, if any.
    pub depth_stencil_format: Option<TextureFormat>,
    /// The sample count.
    pub sample_count: u32,
    /// The stride of every vertex buffer slot.
    pub vertex_strides: Vec<u64>,
}

/// A live compute pipeline's properties.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor { label: None, layout, module, entry_point: Some("cs") })?;
/// let info = device.compute_pipeline_info(pipeline)?;
/// assert_eq!((info.layout, info.workgroup_size), (layout, Some([64, 1, 1])));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ComputePipelineInfo {
    /// The layout.
    pub layout: PipelineLayout,
    /// The workgroup size, if it does not depend on overridable constants.
    pub workgroup_size: Option<[u32; 3]>,
}

/// Counters of a device's pipeline cache.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// # let module = device.create_shader_module(&ShaderModuleDescriptor {
/// #     label: None,
/// #     source: ShaderSource::Wgsl(
/// #         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
/// #          @vertex fn vs_mesh(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 0.0, 1.0); }
/// #          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
/// #          @compute @workgroup_size(64) fn cs() {}",
/// #     ),
/// # })?;
/// # let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
/// let desc = ComputePipelineDescriptor { label: None, layout, module, entry_point: Some("cs") };
/// device.create_compute_pipeline(&desc)?;
/// device.create_compute_pipeline(&desc)?;
/// let stats = device.pipeline_cache_stats();
/// assert_eq!((stats.misses, stats.hits, stats.entries), (1, 1, 1));
/// device.clear_pipeline_cache();
/// assert_eq!(device.pipeline_cache_stats().entries, 0);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct PipelineCacheStats {
    /// Creations served from the cache.
    pub hits: u64,
    /// Creations that compiled a pipeline.
    pub misses: u64,
    /// Compiled pipelines held.
    pub entries: usize,
}

// ---------------------------------------------------------------------------------------------
// Cache keys (internal)

/// Everything a compiled pipeline depends on, owned: equal keys compile to equal pipelines.
/// Shader modules and pipeline layouts are named by serials that are never reused on a device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RenderKey {
    pub(crate) layout: u64,
    pub(crate) vertex: StageKey,
    pub(crate) buffers: Vec<VertexBufferKey>,
    pub(crate) primitive: PrimitiveState,
    pub(crate) depth_stencil: Option<DepthStencilKey>,
    pub(crate) multisample: MultisampleState,
    pub(crate) fragment: Option<FragmentKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ComputeKey {
    pub(crate) layout: u64,
    pub(crate) stage: StageKey,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct StageKey {
    pub(crate) module: u64,
    pub(crate) entry_point: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VertexBufferKey {
    pub(crate) array_stride: u64,
    pub(crate) step_mode: VertexStepMode,
    pub(crate) attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DepthStencilKey {
    pub(crate) format: TextureFormat,
    pub(crate) depth_write_enabled: bool,
    pub(crate) depth_compare: CompareFunction,
    pub(crate) stencil: StencilState,
    pub(crate) bias_constant: i32,
    pub(crate) bias_slope_scale: u32,
    pub(crate) bias_clamp: u32,
}

impl DepthStencilKey {
    pub(crate) const fn bias(&self) -> DepthBiasState {
        DepthBiasState {
            constant: self.bias_constant,
            slope_scale: f32::from_bits(self.bias_slope_scale),
            clamp: f32::from_bits(self.bias_clamp),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FragmentKey {
    pub(crate) stage: StageKey,
    pub(crate) targets: Vec<Option<ColorTargetState>>,
}

impl RenderKey {
    pub(crate) fn uses(&self, serial: u64) -> bool {
        self.layout == serial
            || self.vertex.module == serial
            || self
                .fragment
                .as_ref()
                .is_some_and(|fragment| fragment.stage.module == serial)
    }
}

impl ComputeKey {
    pub(crate) const fn uses(&self, serial: u64) -> bool {
        self.layout == serial || self.stage.module == serial
    }
}

// ---------------------------------------------------------------------------------------------
// What command validation needs (internal)

/// A buffer binding whose layout leaves the size check to draw time (`min_binding_size:
/// None`): the bound range must hold at least the shader's `min_size` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LateSize {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) min_size: u64,
}

/// One vertex buffer slot of a render pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VertexSlot {
    pub(crate) stride: u64,
    pub(crate) step: VertexStepMode,
    /// The end of the slot's last attribute in an element (0 without attributes).
    pub(crate) last_end: u64,
}

/// What draws and dispatches are checked against, beyond the pipeline's public info.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub(crate) struct PipelineMeta {
    pub(crate) late_sizes: Vec<LateSize>,
    pub(crate) slots: Vec<VertexSlot>,
    pub(crate) strip_index_format: Option<IndexFormat>,
}

/// The buffer bindings of `entries` whose layout entry has no `min_binding_size`.
pub(crate) fn late_sizes(
    entries: &[&EntryReflection],
    groups: &[&[BindGroupLayoutEntry]],
) -> Vec<LateSize> {
    let mut sizes: Vec<LateSize> = Vec::new();
    for entry in entries {
        for binding in &entry.bindings {
            let min_size = match &binding.ty {
                ShaderBindingType::Uniform { min_size }
                | ShaderBindingType::Storage { min_size, .. } => *min_size,
                _ => continue,
            };
            let deferred = usize::try_from(binding.group)
                .ok()
                .and_then(|group| groups.get(group))
                .and_then(|layout| layout.iter().find(|slot| slot.binding == binding.binding))
                .is_some_and(|slot| {
                    matches!(
                        slot.ty,
                        BindingType::Buffer {
                            min_binding_size: None,
                            ..
                        }
                    )
                });
            if !deferred || min_size == 0 {
                continue;
            }
            match sizes
                .iter_mut()
                .find(|late| late.group == binding.group && late.binding == binding.binding)
            {
                Some(late) => late.min_size = late.min_size.max(min_size),
                None => sizes.push(LateSize {
                    group: binding.group,
                    binding: binding.binding,
                    min_size,
                }),
            }
        }
    }
    sizes
}

impl RenderKey {
    /// The draw-time metadata of a render pipeline compiled from this key.
    pub(crate) fn meta(&self, late_sizes: Vec<LateSize>) -> PipelineMeta {
        PipelineMeta {
            late_sizes,
            slots: self
                .buffers
                .iter()
                .map(|buffer| VertexSlot {
                    stride: buffer.array_stride,
                    step: buffer.step_mode,
                    last_end: buffer
                        .attributes
                        .iter()
                        .map(|attribute| attribute.offset.saturating_add(attribute.format.size()))
                        .max()
                        .unwrap_or(0),
                })
                .collect(),
            strip_index_format: self.primitive.strip_index_format,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Validation

/// A resolved stage: its module's serial and the entry point's reflection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stage<'r> {
    pub(crate) module: u64,
    pub(crate) entry: &'r EntryReflection,
}

impl Stage<'_> {
    fn key(&self) -> StageKey {
        StageKey {
            module: self.module,
            entry_point: self.entry.name.clone(),
        }
    }
}

const RENDER: ResourceKind = ResourceKind::RenderPipeline;

fn mismatch(stage: ShaderStage, message: String) -> RhiError {
    RhiError::ShaderInterface { stage, message }
}

/// Validates a render pipeline; returns its cache key and properties.
pub(crate) fn validate_render(
    desc: &RenderPipelineDescriptor<'_>,
    layout_serial: u64,
    groups: &[&[BindGroupLayoutEntry]],
    vertex: Stage<'_>,
    fragment: Option<Stage<'_>>,
    limits: &Limits,
) -> Result<(RenderKey, RenderPipelineInfo), RhiError> {
    let buffers = validate_vertex_buffers(desc.vertex.buffers, groups.len(), vertex.entry, limits)?;
    check_bindings(vertex.entry, groups)?;

    let primitive = desc.primitive;
    if primitive.strip_index_format.is_some() && !primitive.topology.is_strip() {
        return Err(RhiError::invalid(
            RENDER,
            "strip_index_format is only for strip topologies",
        ));
    }

    let multisample = desc.multisample;
    if !matches!(multisample.count, 1 | 4) {
        return Err(RhiError::invalid(
            RENDER,
            "the sample count is neither 1 nor 4",
        ));
    }
    if multisample.alpha_to_coverage_enabled && multisample.count == 1 {
        return Err(RhiError::invalid(
            RENDER,
            "alpha to coverage needs multisampling",
        ));
    }

    let depth_stencil = desc
        .depth_stencil
        .map(|state| validate_depth_stencil(&state, primitive.topology))
        .transpose()?;

    let fragment_key = match (desc.fragment, fragment) {
        (Some(state), Some(stage)) => {
            let targets = validate_targets(state.targets, multisample.count, stage.entry, limits)?;
            check_bindings(stage.entry, groups)?;
            check_inter_stage(vertex.entry, stage.entry)?;
            Some(FragmentKey {
                stage: stage.key(),
                targets,
            })
        }
        (None, None) => None,
        _ => {
            return Err(RhiError::invalid(
                RENDER,
                "the fragment state and its resolved stage disagree",
            ));
        }
    };

    let info = RenderPipelineInfo {
        layout: desc.layout,
        color_formats: fragment_key.as_ref().map_or_else(Vec::new, |fragment| {
            fragment
                .targets
                .iter()
                .map(|target| target.map(|target| target.format))
                .collect()
        }),
        depth_stencil_format: depth_stencil.map(|state| state.format),
        sample_count: multisample.count,
        vertex_strides: buffers.iter().map(|buffer| buffer.array_stride).collect(),
    };
    let key = RenderKey {
        layout: layout_serial,
        vertex: vertex.key(),
        buffers,
        primitive,
        depth_stencil,
        multisample,
        fragment: fragment_key,
    };
    Ok((key, info))
}

/// Validates a compute pipeline; returns its cache key and properties.
pub(crate) fn validate_compute(
    desc: &ComputePipelineDescriptor<'_>,
    layout_serial: u64,
    groups: &[&[BindGroupLayoutEntry]],
    stage: Stage<'_>,
    limits: &Limits,
) -> Result<(ComputeKey, ComputePipelineInfo), RhiError> {
    if let Some(size) = stage.entry.workgroup_size {
        let [x, y, z] = size;
        let checks = [
            (
                "max_compute_workgroup_size_x",
                x,
                limits.max_compute_workgroup_size_x,
            ),
            (
                "max_compute_workgroup_size_y",
                y,
                limits.max_compute_workgroup_size_y,
            ),
            (
                "max_compute_workgroup_size_z",
                z,
                limits.max_compute_workgroup_size_z,
            ),
        ];
        for (name, requested, max) in checks {
            if requested > max {
                return Err(mismatch(
                    ShaderStage::Compute,
                    format!("the workgroup size breaks `{name}`: {requested} > {max}"),
                ));
            }
        }
        let invocations = u64::from(x)
            .saturating_mul(u64::from(y))
            .saturating_mul(u64::from(z));
        let max = u64::from(limits.max_compute_invocations_per_workgroup);
        if invocations > max {
            return Err(mismatch(
                ShaderStage::Compute,
                format!(
                    "the workgroup has {invocations} invocations, more than \
                     `max_compute_invocations_per_workgroup` ({max})"
                ),
            ));
        }
    }
    check_bindings(stage.entry, groups)?;
    Ok((
        ComputeKey {
            layout: layout_serial,
            stage: stage.key(),
        },
        ComputePipelineInfo {
            layout: desc.layout,
            workgroup_size: stage.entry.workgroup_size,
        },
    ))
}

fn validate_vertex_buffers(
    buffers: &[VertexBufferLayout<'_>],
    groups: usize,
    entry: &EntryReflection,
    limits: &Limits,
) -> Result<Vec<VertexBufferKey>, RhiError> {
    let count = u32::try_from(buffers.len()).unwrap_or(u32::MAX);
    if count > limits.max_vertex_buffers {
        return Err(RhiError::limit(
            "max_vertex_buffers",
            count,
            limits.max_vertex_buffers,
        ));
    }
    let with_groups = count.saturating_add(u32::try_from(groups).unwrap_or(u32::MAX));
    if with_groups > limits.max_bind_groups_plus_vertex_buffers {
        return Err(RhiError::limit(
            "max_bind_groups_plus_vertex_buffers",
            with_groups,
            limits.max_bind_groups_plus_vertex_buffers,
        ));
    }
    let max_stride = u64::from(limits.max_vertex_buffer_array_stride);
    let mut locations: Vec<(u32, IoKind)> = Vec::new();
    let mut keys = Vec::new();
    for buffer in buffers {
        if buffer.array_stride > max_stride {
            return Err(RhiError::limit(
                "max_vertex_buffer_array_stride",
                buffer.array_stride,
                max_stride,
            ));
        }
        if !buffer.array_stride.is_multiple_of(4) {
            return Err(RhiError::invalid(
                RENDER,
                "a vertex buffer stride is not a multiple of 4",
            ));
        }
        let span = if buffer.array_stride == 0 {
            max_stride
        } else {
            buffer.array_stride
        };
        for attribute in buffer.attributes {
            let size = attribute.format.size();
            if !attribute.offset.is_multiple_of(size.min(4)) {
                return Err(RhiError::invalid(
                    RENDER,
                    "a vertex attribute offset is not aligned to min(4, its size)",
                ));
            }
            if attribute
                .offset
                .checked_add(size)
                .is_none_or(|end| end > span)
            {
                return Err(RhiError::invalid(
                    RENDER,
                    "a vertex attribute does not fit its buffer's stride",
                ));
            }
            if attribute.shader_location >= limits.max_vertex_attributes {
                return Err(RhiError::limit(
                    "max_vertex_attributes",
                    u64::from(attribute.shader_location) + 1,
                    limits.max_vertex_attributes,
                ));
            }
            if locations
                .iter()
                .any(|(location, _)| *location == attribute.shader_location)
            {
                return Err(RhiError::invalid(
                    RENDER,
                    "two vertex attributes have the same shader location",
                ));
            }
            locations.push((attribute.shader_location, attribute.format.kind()));
        }
        keys.push(VertexBufferKey {
            array_stride: buffer.array_stride,
            step_mode: buffer.step_mode,
            attributes: buffer.attributes.to_vec(),
        });
    }
    let total = u32::try_from(locations.len()).unwrap_or(u32::MAX);
    if total > limits.max_vertex_attributes {
        return Err(RhiError::limit(
            "max_vertex_attributes",
            total,
            limits.max_vertex_attributes,
        ));
    }
    for (location, kind) in &entry.inputs {
        match locations.iter().find(|(at, _)| at == location) {
            None => {
                return Err(mismatch(
                    ShaderStage::Vertex,
                    format!("the input @location({location}) has no vertex attribute"),
                ));
            }
            Some((_, attribute)) if attribute != kind => {
                return Err(mismatch(
                    ShaderStage::Vertex,
                    format!(
                        "the input @location({location}) is {kind}, its attribute's format is \
                         read as {attribute}"
                    ),
                ));
            }
            Some(_) => {}
        }
    }
    Ok(keys)
}

fn validate_depth_stencil(
    state: &DepthStencilState,
    topology: PrimitiveTopology,
) -> Result<DepthStencilKey, RhiError> {
    if !state.format.has_depth() {
        return Err(RhiError::invalid(
            RENDER,
            "the depth/stencil format is not a depth format",
        ));
    }
    if state.stencil.is_enabled() && !state.format.has_stencil() {
        return Err(RhiError::invalid(
            RENDER,
            "the stencil test needs a format with stencil",
        ));
    }
    let bias = state.bias;
    if !bias.slope_scale.is_finite() || !bias.clamp.is_finite() {
        return Err(RhiError::invalid(RENDER, "the depth bias is not finite"));
    }
    let triangles = matches!(
        topology,
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
    );
    if bias.is_enabled() && !triangles {
        return Err(RhiError::invalid(
            RENDER,
            "a depth bias is only for triangle topologies",
        ));
    }
    Ok(DepthStencilKey {
        format: state.format,
        depth_write_enabled: state.depth_write_enabled,
        depth_compare: state.depth_compare,
        stencil: state.stencil,
        bias_constant: bias.constant,
        // -0.0 and 0.0 compile to the same pipeline.
        bias_slope_scale: (bias.slope_scale + 0.0).to_bits(),
        bias_clamp: (bias.clamp + 0.0).to_bits(),
    })
}

/// The scalar kind a fragment output for a color format must have.
const fn target_kind(format: TextureFormat) -> IoKind {
    match format {
        TextureFormat::R32Uint => IoKind::Uint,
        _ => IoKind::Float,
    }
}

const fn is_blendable(format: TextureFormat) -> bool {
    !matches!(
        format,
        TextureFormat::R32Uint
            | TextureFormat::R32Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Float
    )
}

fn validate_targets(
    targets: &[Option<ColorTargetState>],
    sample_count: u32,
    entry: &EntryReflection,
    limits: &Limits,
) -> Result<Vec<Option<ColorTargetState>>, RhiError> {
    let count = u32::try_from(targets.len()).unwrap_or(u32::MAX);
    if count > limits.max_color_attachments {
        return Err(RhiError::limit(
            "max_color_attachments",
            count,
            limits.max_color_attachments,
        ));
    }
    for (index, target) in targets.iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        if target.format.has_depth() {
            return Err(RhiError::invalid(
                RENDER,
                "a color target has a depth format",
            ));
        }
        if !ColorWrites::all().contains(target.write_mask) {
            return Err(RhiError::invalid(
                RENDER,
                "a color write mask has unknown bits",
            ));
        }
        if sample_count > 1 && !target.format.supports_multisampling() {
            return Err(RhiError::invalid(
                RENDER,
                "a color target's format does not support multisampling",
            ));
        }
        if let Some(blend) = target.blend {
            if !is_blendable(target.format) {
                return Err(RhiError::invalid(
                    RENDER,
                    "blending needs a blendable color format",
                ));
            }
            for component in [blend.color, blend.alpha] {
                let min_max = matches!(
                    component.operation,
                    BlendOperation::Min | BlendOperation::Max
                );
                if min_max
                    && (component.src_factor != BlendFactor::One
                        || component.dst_factor != BlendFactor::One)
                {
                    return Err(RhiError::invalid(
                        RENDER,
                        "the Min and Max blend operations need both factors to be One",
                    ));
                }
            }
        }
        let location = u32::try_from(index).unwrap_or(u32::MAX);
        let output = entry
            .outputs
            .iter()
            .find(|(at, _)| *at == location)
            .map(|(_, kind)| *kind);
        let expected = target_kind(target.format);
        match output {
            None if !target.write_mask.is_empty() => {
                return Err(mismatch(
                    ShaderStage::Fragment,
                    format!("color target {index} is written but has no @location({index}) output"),
                ));
            }
            Some(kind) if kind != expected => {
                return Err(mismatch(
                    ShaderStage::Fragment,
                    format!(
                        "the output @location({index}) is {kind}, color target {index} needs \
                         {expected}"
                    ),
                ));
            }
            _ => {}
        }
    }
    Ok(targets.to_vec())
}

fn check_inter_stage(vertex: &EntryReflection, fragment: &EntryReflection) -> Result<(), RhiError> {
    for (location, kind) in &fragment.inputs {
        match vertex.outputs.iter().find(|(at, _)| at == location) {
            None => {
                return Err(mismatch(
                    ShaderStage::Fragment,
                    format!("the input @location({location}) is not a vertex stage output"),
                ));
            }
            Some((_, output)) if output != kind => {
                return Err(mismatch(
                    ShaderStage::Fragment,
                    format!(
                        "the input @location({location}) is {kind}, the vertex stage outputs \
                         {output}"
                    ),
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Checks every resource an entry point uses against the pipeline layout.
fn check_bindings(
    entry: &EntryReflection,
    groups: &[&[BindGroupLayoutEntry]],
) -> Result<(), RhiError> {
    let stage = entry.stage;
    for binding in &entry.bindings {
        let (group, number) = (binding.group, binding.binding);
        let at = format!("@group({group}) @binding({number})");
        let layout = usize::try_from(group)
            .ok()
            .and_then(|group| groups.get(group))
            .and_then(|entries| entries.iter().find(|entry| entry.binding == number))
            .ok_or_else(|| mismatch(stage, format!("{at} is not in the pipeline layout")))?;
        if !layout.visibility.contains(stage.as_stages()) {
            return Err(mismatch(
                stage,
                format!("{at} is not visible to the {stage} stage"),
            ));
        }
        if let Err(problem) = binding_fits(&binding.ty, &layout.ty) {
            return Err(mismatch(stage, format!("{at}: {problem}")));
        }
    }
    Ok(())
}

fn binding_fits(shader: &ShaderBindingType, layout: &BindingType) -> Result<(), String> {
    match (shader, layout) {
        (ShaderBindingType::Unsupported(reason), _) => Err((*reason).to_owned()),
        (
            ShaderBindingType::Uniform { min_size },
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                min_binding_size,
                ..
            },
        ) => check_min_size(*min_size, *min_binding_size),
        (
            ShaderBindingType::Storage {
                read_only,
                min_size,
            },
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: layout },
                min_binding_size,
                ..
            },
        ) => {
            if read_only != layout {
                return Err(if *read_only {
                    "the shader reads a storage buffer the layout declares read-write".to_owned()
                } else {
                    "the shader writes a storage buffer the layout declares read-only".to_owned()
                });
            }
            check_min_size(*min_size, *min_binding_size)
        }
        (ShaderBindingType::Sampler { comparison }, BindingType::Sampler(ty)) => {
            if *comparison == (*ty == SamplerBindingType::Comparison) {
                Ok(())
            } else {
                Err("the sampler's comparison differs from the layout's".to_owned())
            }
        }
        (
            ShaderBindingType::Texture {
                sample,
                dimension,
                multisampled,
            },
            BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: layout_multisampled,
            },
        ) => {
            let sample_fits = matches!(
                (sample, sample_type),
                (ShaderSampleKind::Float, TextureSampleType::Float { .. })
                    | (ShaderSampleKind::Depth, TextureSampleType::Depth)
                    | (ShaderSampleKind::Sint, TextureSampleType::Sint)
                    | (ShaderSampleKind::Uint, TextureSampleType::Uint)
            );
            if !sample_fits {
                Err("the texture's sample type differs from the layout's".to_owned())
            } else if dimension != view_dimension {
                Err("the texture's dimension differs from the layout's".to_owned())
            } else if multisampled != layout_multisampled {
                Err("the texture's multisampling differs from the layout's".to_owned())
            } else {
                Ok(())
            }
        }
        (
            ShaderBindingType::StorageTexture {
                format,
                access,
                dimension,
            },
            BindingType::StorageTexture {
                access: layout_access,
                format: layout_format,
                view_dimension,
            },
        ) => {
            if format.is_none() {
                Err("the storage texture format is not supported".to_owned())
            } else if *format != Some(*layout_format) {
                Err("the storage texture's format differs from the layout's".to_owned())
            } else if access != layout_access {
                Err("the storage texture's access differs from the layout's".to_owned())
            } else if dimension != view_dimension {
                Err("the storage texture's dimension differs from the layout's".to_owned())
            } else {
                Ok(())
            }
        }
        _ => Err("the shader declares another kind of binding than the layout".to_owned()),
    }
}

fn check_min_size(shader: u64, layout: Option<core::num::NonZeroU64>) -> Result<(), String> {
    match layout {
        Some(min) if min.get() < shader => Err(format!(
            "the layout's min_binding_size ({min}) is smaller than the shader's {shader} bytes"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::binding::StorageTextureAccess;
    use crate::handle::RawHandle;
    use crate::shader::{ShaderSource, parse};
    use crate::{ShaderStages, TextureViewDimension};

    fn handle<H: crate::handle::Handle>() -> H {
        H::from_raw(RawHandle {
            device: 1,
            index: 0,
            generation: 0,
        })
    }

    const SHADER: &str = r"
@group(0) @binding(0) var<uniform> tint: vec4<f32>;

struct Out { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }

@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> Out {
    return Out(vec4<f32>(position, 0.0, 1.0), color);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return in.color * tint;
}

@group(0) @binding(1) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = id.x;
}
";

    fn uniform_entry(visibility: ShaderStages) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: BindingType::default(),
        }
    }

    const ATTRIBUTES: [VertexAttribute; 2] = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Unorm8x4,
            offset: 8,
            shader_location: 1,
        },
    ];

    fn render<'a>(
        buffers: &'a [VertexBufferLayout<'a>],
        targets: &'a [Option<ColorTargetState>],
    ) -> RenderPipelineDescriptor<'a> {
        RenderPipelineDescriptor {
            label: None,
            layout: handle(),
            vertex: VertexState {
                module: handle(),
                entry_point: None,
                buffers,
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: handle(),
                entry_point: None,
                targets,
            }),
        }
    }

    fn run(
        desc: &RenderPipelineDescriptor<'_>,
        group: &[BindGroupLayoutEntry],
    ) -> Result<(RenderKey, RenderPipelineInfo), RhiError> {
        let parsed = parse(ShaderSource::Wgsl(SHADER)).unwrap();
        let vertex = parsed.entry_point(ShaderStage::Vertex, None).unwrap();
        let fragment = parsed.entry_point(ShaderStage::Fragment, None).unwrap();
        validate_render(
            desc,
            7,
            &[group],
            Stage {
                module: 1,
                entry: vertex,
            },
            desc.fragment.map(|_| Stage {
                module: 1,
                entry: fragment,
            }),
            &Limits::default(),
        )
    }

    fn buffers() -> [VertexBufferLayout<'static>; 1] {
        [VertexBufferLayout {
            array_stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }]
    }

    #[test]
    fn a_fitting_pipeline_validates() {
        let buffers = buffers();
        let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
        let (key, info) = run(
            &render(&buffers, &targets),
            &[uniform_entry(ShaderStages::FRAGMENT)],
        )
        .unwrap();
        assert_eq!(info.color_formats, [Some(TextureFormat::Rgba8Unorm)]);
        assert_eq!(info.vertex_strides, [12]);
        assert!(key.uses(7) && key.uses(1) && !key.uses(2));
        let (again, _) = run(
            &render(&buffers, &targets),
            &[uniform_entry(ShaderStages::FRAGMENT)],
        )
        .unwrap();
        assert_eq!(key, again, "equal descriptors give equal keys");
    }

    #[test]
    fn shader_bindings_must_be_in_the_layout() {
        let buffers = buffers();
        let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
        let desc = render(&buffers, &targets);
        assert!(matches!(
            run(&desc, &[]),
            Err(RhiError::ShaderInterface {
                stage: ShaderStage::Fragment,
                ..
            })
        ));
        assert!(
            run(&desc, &[uniform_entry(ShaderStages::VERTEX)]).is_err(),
            "not visible to the fragment stage"
        );
        let sampler = BindGroupLayoutEntry {
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            ..uniform_entry(ShaderStages::FRAGMENT)
        };
        assert!(run(&desc, &[sampler]).is_err(), "another kind of binding");
        let too_small = BindGroupLayoutEntry {
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: core::num::NonZeroU64::new(8),
            },
            ..uniform_entry(ShaderStages::FRAGMENT)
        };
        assert!(run(&desc, &[too_small]).is_err());
    }

    #[test]
    fn vertex_inputs_need_attributes_of_their_kind() {
        let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
        let group = [uniform_entry(ShaderStages::FRAGMENT)];
        let one = [VertexBufferLayout {
            array_stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: &ATTRIBUTES[..1],
        }];
        assert!(matches!(
            run(&render(&one, &targets), &group),
            Err(RhiError::ShaderInterface {
                stage: ShaderStage::Vertex,
                ..
            })
        ));
        let uint = [
            ATTRIBUTES[0],
            VertexAttribute {
                format: VertexFormat::Uint8x4,
                ..ATTRIBUTES[1]
            },
        ];
        let wrong_kind = [VertexBufferLayout {
            array_stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: &uint,
        }];
        assert!(run(&render(&wrong_kind, &targets), &group).is_err());
        let too_narrow = [VertexBufferLayout {
            array_stride: 8,
            ..buffers()[0]
        }];
        assert!(matches!(
            run(&render(&too_narrow, &targets), &group),
            Err(RhiError::InvalidDescriptor { .. })
        ));
    }

    #[test]
    fn color_targets_follow_the_fragment_outputs() {
        let buffers = buffers();
        let group = [uniform_entry(ShaderStages::FRAGMENT)];
        let uint = [Some(ColorTargetState::from(TextureFormat::R32Uint))];
        assert!(
            run(&render(&buffers, &uint), &group).is_err(),
            "f32 into R32Uint"
        );
        let unwritten = [
            Some(ColorTargetState::from(TextureFormat::Rgba8Unorm)),
            Some(ColorTargetState {
                write_mask: ColorWrites::empty(),
                ..ColorTargetState::from(TextureFormat::Rgba8Unorm)
            }),
        ];
        assert!(run(&render(&buffers, &unwritten), &group).is_ok());
        let missing = [
            Some(ColorTargetState::from(TextureFormat::Rgba8Unorm)),
            Some(ColorTargetState::from(TextureFormat::Rgba8Unorm)),
        ];
        assert!(run(&render(&buffers, &missing), &group).is_err());
        let blended_float = [Some(ColorTargetState {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..ColorTargetState::from(TextureFormat::Rgba32Float)
        })];
        assert!(matches!(
            run(&render(&buffers, &blended_float), &group),
            Err(RhiError::InvalidDescriptor { .. })
        ));
    }

    #[test]
    fn fixed_function_rules() {
        let buffers = buffers();
        let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
        let group = [uniform_entry(ShaderStages::FRAGMENT)];
        let mut desc = render(&buffers, &targets);
        desc.primitive.strip_index_format = Some(IndexFormat::Uint16);
        assert!(run(&desc, &group).is_err());
        desc.primitive.topology = PrimitiveTopology::TriangleStrip;
        assert!(run(&desc, &group).is_ok());

        let mut desc = render(&buffers, &targets);
        desc.multisample.count = 2;
        assert!(run(&desc, &group).is_err());
        desc.multisample = MultisampleState {
            alpha_to_coverage_enabled: true,
            ..MultisampleState::default()
        };
        assert!(run(&desc, &group).is_err());

        let mut desc = render(&buffers, &targets);
        desc.depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Rgba8Unorm,
            ..DepthStencilState::default()
        });
        assert!(run(&desc, &group).is_err());
        desc.depth_stencil = Some(DepthStencilState {
            stencil: StencilState {
                front: StencilFaceState {
                    pass_op: StencilOperation::Replace,
                    ..StencilFaceState::IGNORE
                },
                ..StencilState::default()
            },
            ..DepthStencilState::default()
        });
        assert!(run(&desc, &group).is_err(), "stencil on Depth32Float");
        desc.depth_stencil = Some(DepthStencilState {
            bias: DepthBiasState {
                constant: 2,
                ..DepthBiasState::default()
            },
            ..DepthStencilState::default()
        });
        assert!(run(&desc, &group).is_ok());
        desc.primitive.topology = PrimitiveTopology::LineList;
        assert!(run(&desc, &group).is_err(), "bias on lines");
    }

    #[test]
    fn compute_workgroups_and_bindings() {
        let parsed = parse(ShaderSource::Wgsl(SHADER)).unwrap();
        let entry = parsed
            .entry_point(ShaderStage::Compute, Some("cs"))
            .unwrap();
        let desc = ComputePipelineDescriptor {
            label: None,
            layout: handle(),
            module: handle(),
            entry_point: Some("cs"),
        };
        let storage = BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        let stage = Stage { module: 3, entry };
        let limits = Limits::default();
        let (key, info) = validate_compute(&desc, 9, &[&[storage]], stage, &limits).unwrap();
        assert_eq!(info.workgroup_size, Some([64, 1, 1]));
        assert!(key.uses(9) && key.uses(3));
        let read_only = BindGroupLayoutEntry {
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            ..storage
        };
        assert!(validate_compute(&desc, 9, &[&[read_only]], stage, &limits).is_err());
        let small = Limits {
            max_compute_workgroup_size_x: 32,
            ..Limits::default()
        };
        assert!(matches!(
            validate_compute(&desc, 9, &[&[storage]], stage, &small),
            Err(RhiError::ShaderInterface {
                stage: ShaderStage::Compute,
                ..
            })
        ));
    }

    #[test]
    fn storage_texture_bindings_match_exactly() {
        let shader = ShaderBindingType::StorageTexture {
            format: Some(TextureFormat::Rgba8Unorm),
            access: StorageTextureAccess::WriteOnly,
            dimension: TextureViewDimension::D2,
        };
        let layout = BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba8Unorm,
            view_dimension: TextureViewDimension::D2,
        };
        assert!(binding_fits(&shader, &layout).is_ok());
        let other_format = BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2,
        };
        assert!(binding_fits(&shader, &other_format).is_err());
    }
}
