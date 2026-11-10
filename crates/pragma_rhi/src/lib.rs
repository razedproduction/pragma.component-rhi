//! A Render Hardware Interface for game engines and frameworks built on
//! [`pragma_core`](https://docs.rs/pragma_core).
//!
//! An [`Instance`] enumerates [`Adapter`]s of the enabled backends — Vulkan and D3D12 through
//! wgpu, and a headless, deterministic null backend — and an adapter creates a [`Device`]. The
//! device creates buffers, textures, texture views and samplers from descriptors it validates
//! against its [`Limits`], and names them by `Copy` generational handles. Every failure,
//! including device loss and out-of-memory, is an [`RhiError`]; nothing panics.
//!
//! Shader modules (WGSL or SPIR-V) are validated by naga on every backend; bind group layouts,
//! bind groups, pipeline layouts and render and compute pipelines are checked against each
//! other and the shaders' interfaces before a backend compiles anything. Pipelines are cached
//! per device and can be compiled on a pragma.core executor's blocking pool
//! ([`Device::compile_render_pipeline`]).
//!
//! Commands are recorded into [`CommandEncoder`]s without touching a device — in parallel, as
//! tasks of a pragma.core frame graph ([`CommandGraph`]) — and checked as a whole by
//! [`Device::submit`]; [`Device::read_buffer`] and [`Device::read_texture`] read results back.
//! The null backend executes copies, clears and resolves exactly like a GPU, so readback tests
//! run without one.
//!
//! Frames: every submission returns a [`SubmissionIndex`] fence; [`Device::begin_frame`] bounds
//! how many frames are in flight, [`PerFrame`] holds per-frame resources, and destroyed buffers
//! and textures keep their memory until the submissions that may use them retire. Uploads go
//! through a bounded staging belt ([`Device::upload_buffer`], [`Device::upload_texture`], or
//! produced on a pragma.core blocking pool with [`Device::upload_async`]); readback can be
//! asynchronous ([`Device::read_buffer_async`]). A [`Surface`] from any window with a
//! `raw-window-handle` (or a null surface for tests) is configured, acquired and presented by the
//! device; an outdated swapchain is a [`SurfaceError::Outdated`], never a panic.
//!
//! No wgpu type appears in the public API (DECISIONS.md #8). The crate is at milestone M4; see
//! `docs/roadmap.md` for what comes next and `docs/architecture.md` for the design.
//!
//! ```
//! use pragma_rhi::{
//!     Backend, DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, RhiError,
//!     TextureDescriptor, TextureFormat, TextureUsages, TextureViewDescriptor,
//! };
//!
//! let instance = Instance::new(&InstanceDescriptor::null());
//! let adapter = instance
//!     .adapters()
//!     .into_iter()
//!     .find(|adapter| adapter.info().backend == Backend::Null)
//!     .ok_or(RhiError::DeviceLost)?;
//! let mut device = adapter.request_device(&DeviceDescriptor::default())?;
//!
//! let texture = device.create_texture(&TextureDescriptor {
//!     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
//!     mip_level_count: 9,
//!     format: TextureFormat::Rgba8UnormSrgb,
//!     usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
//!     ..TextureDescriptor::default()
//! })?;
//! let view = device.create_texture_view(texture, &TextureViewDescriptor::default())?;
//! assert_eq!(device.texture_view_info(view)?.mip_level_count, 9);
//!
//! // A texture with live views cannot be destroyed.
//! assert!(matches!(device.destroy_texture(texture), Err(RhiError::ResourceInUse { .. })));
//! device.destroy_texture_view(view)?;
//! device.destroy_texture(texture)?;
//! # Ok::<(), RhiError>(())
//! ```
#![forbid(unsafe_code)]

mod backend;
mod binding;
mod commands;
mod device;
mod error;
mod frame;
mod graph;
mod handle;
mod instance;
mod limits;
mod pending;
mod pipeline;
mod resource;
mod shader;
mod submit;
mod surface;
mod upload;

pub use binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupInfo, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutInfo, BindingResource, BindingType, BufferBinding,
    BufferBindingType, PipelineLayoutDescriptor, PipelineLayoutInfo, SamplerBindingType,
    StorageTextureAccess, TextureSampleType,
};
pub use commands::{
    COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT, Color, CommandBuffer, CommandEncoder,
    ComputePass, ComputePassDescriptor, LoadOp, Operations, Origin3d, RenderPass,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
};
pub use device::{Device, DeviceDescriptor};
pub use error::RhiError;
pub use frame::{FrameInfo, MAX_FRAMES_IN_FLIGHT, PerFrame, SubmissionIndex};
pub use graph::{CommandGraph, CommandGraphBuilder};
pub use handle::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, PipelineLayout, RenderPipeline,
    ResourceKind, Sampler, ShaderModule, Texture, TextureView,
};
pub use instance::{
    Adapter, AdapterInfo, Backend, Backends, DeviceType, Instance, InstanceDescriptor, NullConfig,
};
pub use limits::Limits;
pub use pending::{PendingComputePipeline, PendingRenderPipeline};
pub use pipeline::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrites,
    ComputePipelineDescriptor, ComputePipelineInfo, DepthBiasState, DepthStencilState, Face,
    FragmentState, FrontFace, IndexFormat, MultisampleState, PipelineCacheStats, PrimitiveState,
    PrimitiveTopology, RenderPipelineDescriptor, RenderPipelineInfo, StencilFaceState,
    StencilOperation, StencilState, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};
pub use resource::{
    AddressMode, BufferDescriptor, BufferInfo, BufferUsages, CompareFunction, Extent3d, FilterMode,
    SamplerDescriptor, SamplerInfo, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureInfo, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    TextureViewInfo,
};
pub use shader::{
    EntryPointInfo, ShaderModuleDescriptor, ShaderModuleInfo, ShaderSource, ShaderStage,
    ShaderStages, SourceLocation,
};
pub use surface::{
    CompositeAlphaMode, PresentMode, Surface, SurfaceCapabilities, SurfaceConfiguration,
    SurfaceError, SurfaceFrame,
};
pub use upload::{PendingReadback, PendingUpload, UploadConfig, UploadStats, UploadTarget};
