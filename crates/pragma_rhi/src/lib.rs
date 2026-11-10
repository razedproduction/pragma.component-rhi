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
//! No wgpu type appears in the public API (DECISIONS.md #8); `docs/architecture.md` in the
//! repository describes the design.
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
//!
//! ## Semver guarantees
//!
//! The public API follows semver. Enums that will gain variants as GPUs and WebGPU grow
//! ([`RhiError`] and its struct variants, [`SurfaceError`], [`TextureFormat`], [`VertexFormat`],
//! [`BindingType`], [`ResourceKind`], …) and the structs the crate fills in for you
//! ([`Limits`], [`AdapterInfo`], the `*Info` structs, [`UploadStats`], [`PipelineCacheStats`],
//! [`SurfaceCapabilities`], …) are `#[non_exhaustive]`: match them with a wildcard arm, and
//! build a [`Limits`] from [`Limits::default`] or [`Adapter::limits`] and change its fields.
//! Descriptors stay plain structs with public fields, built with `..Default::default()`
//! (DECISIONS.md #27). `pragma_core` types appear in the API ([`ThreadPool`](pragma_core::jobs::ThreadPool), `Resources`,
//! `Executor`, the sources of [`RhiError::Graph`] and [`RhiError::Task`]); the crate re-exports
//! the [`pragma_core`] it is built with, and a new major version of `pragma_core` means a new
//! major version of `pragma_rhi`. wgpu is an implementation detail: changing its version is not
//! a breaking change. The minimum supported Rust version (1.88) is raised only in a minor release.
//!
//! ## Integrating into an engine
//!
//! The RHI is a library: it opens no window, runs no loop and has no global state. The engine
//! owns the pieces and decides when they run. These are the decisions to make, with the defaults
//! the crate is designed and tested for;
//! [`examples/rhi_sim.rs`](https://github.com/razedproduction/pragma.component-rhi/blob/main/crates/pragma_rhi/examples/rhi_sim.rs)
//! is a complete headless loop built this way.
//!
//! 1. **One instance, one adapter, one device.** Create the [`Instance`] once, create the
//!    window's [`Surface`] from it, and pick an adapter whose
//!    [`surface_capabilities`](Adapter::surface_capabilities) support the surface (by
//!    [`DeviceType`] when there are several). Request only the [`Limits`] the engine needs: the
//!    device validates against them, not the adapter's, so a game tested on the defaults behaves
//!    the same on a stronger GPU. Tests, CI and servers use the null backend
//!    ([`InstanceDescriptor::null`]), which also presents to a [`Instance::create_null_surface`].
//! 2. **The device has one owner.** Every [`Device`] method takes `&self` or `&mut self` on the
//!    thread that owns it (the render thread); nothing inside it locks. Parallelism comes from
//!    recording: a [`CommandEncoder`] is plain data, so recordings run as tasks of a `pragma_core`
//!    frame graph ([`CommandGraph`]) on the engine's [`ThreadPool`](pragma_core::jobs::ThreadPool), and only
//!    [`Device::submit`] touches the device.
//! 3. **One pool, owned by the engine.** Create one `pragma_core` [`ThreadPool`](pragma_core::jobs::ThreadPool) per process
//!    (through the re-exported [`pragma_core`], so the versions match) and one `Executor` on it
//!    for the slow work: pipeline compilation ([`Device::compile_render_pipeline`]), asset
//!    decoding into uploads ([`Device::upload_async`]). Results are installed with the `poll_*`
//!    methods once per frame, so the render thread decides when they appear.
//! 4. **The frame.** [`Device::begin_frame`] (bounds the CPU lead to
//!    [`DeviceDescriptor::frames_in_flight`]) → [`Device::acquire_frame`] → per-frame data into
//!    the [`PerFrame`] slot ([`Device::write_buffer`], [`Device::upload_buffer`]) → record →
//!    [`Device::submit`] → [`Device::present`] → [`Device::end_frame`]. Reads back without
//!    stalling go through [`Device::read_buffer_async`] and are polled in a later frame.
//! 5. **Handles are values, lifetimes are explicit.** Handles are `Copy` and fit in ECS
//!    components. Destroy explicitly, dependents first: a bind group pins its layout and what
//!    it binds, a pipeline its layout, a texture its views ([`RhiError::ResourceInUse`]
//!    otherwise). A destroyed buffer or texture keeps its memory until the GPU is done with it;
//!    a stale handle is [`RhiError::StaleHandle`], never undefined behaviour.
//! 6. **Errors to react to.** [`SurfaceError::Outdated`] and [`SurfaceError::Lost`]: configure
//!    the surface again and skip the frame; [`SurfaceError::Timeout`] and
//!    [`SurfaceError::Occluded`]: skip the frame. [`RhiError::DeviceLost`]: request a new device
//!    from the same adapter, configure the surface on it and rebuild resources (old handles are
//!    [`RhiError::ForeignHandle`] there); test this path with [`Device::simulate_device_loss`].
//!    [`RhiError::InvalidCommand`] is a bug in a recording; it names the task and the command,
//!    and nothing of that submission ran. [`RhiError::OutOfMemory`] leaves the device usable.
//! 7. **Determinism and tests.** Recordings come back in the frame graph's topological order on
//!    any number of workers. The null backend executes copies, clears and resolves, keeps
//!    texture bytes, and with [`NullConfig::latency`] keeps submissions in flight, so frame
//!    pacing, deferred destruction and readback are testable without a GPU; pixel results are
//!    checked on software adapters (WARP, lavapipe).
//! 8. **Shutdown.** [`Device::wait_idle`], then drop the device before the executor and the
//!    pool (struct fields drop in declaration order).
//!
//! The skeleton of such a renderer:
//!
//! ```
//! use pragma_rhi::pragma_core::graph::Resources;
//! use pragma_rhi::pragma_core::jobs::{PoolConfig, ThreadPool};
//! use pragma_rhi::pragma_core::tasks::{AsyncConfig, Executor};
//! use pragma_rhi::{
//!     Adapter, Buffer, BufferDescriptor, BufferUsages, Color, CommandGraph, Device,
//!     DeviceDescriptor, Instance, InstanceDescriptor, PerFrame, RenderPassColorAttachment,
//!     RenderPassDescriptor, RhiError, Surface, SurfaceConfiguration, SurfaceError, TextureView,
//! };
//!
//! /// Frame-graph resources, set before recording.
//! struct Camera(Buffer);
//! struct Backbuffer(TextureView);
//!
//! struct Renderer {
//!     graph: CommandGraph,
//!     resources: Resources,
//!     camera: PerFrame<Buffer>,
//!     surface: Surface,
//!     config: SurfaceConfiguration,
//!     device: Device,
//!     adapter: Adapter,
//!     _executor: Executor, // pipeline compilation and asset decoding (not used here)
//!     pool: ThreadPool,    // dropped last
//! }
//!
//! impl Renderer {
//!     fn new(instance: &Instance, surface: Surface) -> Result<Self, Box<dyn std::error::Error>> {
//!         let adapter = instance
//!             .adapters()
//!             .into_iter()
//!             .find(|adapter| adapter.surface_capabilities(&surface).is_supported())
//!             .ok_or(SurfaceError::Unsupported)?;
//!         let pool = ThreadPool::new(PoolConfig::default())?;
//!         let executor = Executor::new(&pool, AsyncConfig::default())?;
//!         let (device, surface, config, camera) = Self::device(&adapter, surface, (320, 240))?;
//!
//!         // Built once: ordering mistakes fail here, at startup.
//!         let mut graph = CommandGraph::builder();
//!         graph.record("main pass", |t| t.reads::<Camera>().reads::<Backbuffer>(), |ctx, encoder| {
//!             let target = ctx.read::<Backbuffer>()?.0;
//!             let _camera = ctx.read::<Camera>()?.0; // bound by the real passes
//!             encoder.begin_render_pass(&RenderPassDescriptor {
//!                 label: Some("main pass"),
//!                 color_attachments: &[Some(RenderPassColorAttachment::clear(target, Color::BLACK))],
//!                 ..RenderPassDescriptor::default()
//!             });
//!             Ok(())
//!         });
//!         let graph = graph.build()?;
//!         let resources = Resources::new();
//!         Ok(Self { graph, resources, camera, surface, config, device, adapter, _executor: executor, pool })
//!     }
//!
//!     /// Everything that belongs to one device; called again after a device loss.
//!     fn device(
//!         adapter: &Adapter,
//!         mut surface: Surface,
//!         (width, height): (u32, u32),
//!     ) -> Result<(Device, Surface, SurfaceConfiguration, PerFrame<Buffer>), RhiError> {
//!         let mut device = adapter.request_device(&DeviceDescriptor::default())?;
//!         let caps = device.surface_capabilities(&surface);
//!         let config = SurfaceConfiguration {
//!             format: caps.formats[0],
//!             width,
//!             height,
//!             ..SurfaceConfiguration::default()
//!         };
//!         device.configure_surface(&mut surface, &config)?;
//!         let usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
//!         let mut buffers = Vec::new();
//!         for _ in 0..device.frames_in_flight() {
//!             buffers.push(device.create_buffer(&BufferDescriptor { label: Some("camera"), size: 256, usage })?);
//!         }
//!         let camera = PerFrame::new(&device, |slot| buffers[slot]);
//!         Ok((device, surface, config, camera))
//!     }
//!
//!     /// One frame; `Ok(false)` when it was skipped.
//!     fn frame(&mut self, view_projection: &[u8; 256]) -> Result<bool, RhiError> {
//!         let frame = self.device.begin_frame()?;
//!         let result = self.draw(frame, view_projection);
//!         self.device.end_frame()?;
//!         match result {
//!             Ok(()) => Ok(true),
//!             Err(RhiError::Surface(SurfaceError::Outdated | SurfaceError::Lost)) => {
//!                 self.device.configure_surface(&mut self.surface, &self.config)?;
//!                 Ok(false)
//!             }
//!             Err(RhiError::Surface(SurfaceError::Timeout | SurfaceError::Occluded)) => Ok(false),
//!             Err(other) => Err(other),
//!         }
//!     }
//!
//!     fn draw(&mut self, frame: pragma_rhi::FrameInfo, view_projection: &[u8; 256]) -> Result<(), RhiError> {
//!         let target = self.device.acquire_frame(&mut self.surface)?;
//!         let camera = *self.camera.get(frame);
//!         self.device.write_buffer(camera, 0, view_projection)?;
//!         self.resources.insert(Camera(camera));
//!         self.resources.insert(Backbuffer(target.view));
//!         let buffers = self.graph.record(&self.pool, &mut self.resources)?;
//!         self.device.submit(buffers)?;
//!         self.device.present(&mut self.surface)
//!     }
//!
//!     /// Recovery from `RhiError::DeviceLost`: the same adapter, everything rebuilt.
//!     fn recover(&mut self, surface: Surface) -> Result<(), RhiError> {
//!         let size = (self.config.width, self.config.height);
//!         let (device, surface, config, camera) = Self::device(&self.adapter, surface, size)?;
//!         (self.device, self.surface, self.config, self.camera) = (device, surface, config, camera);
//!         Ok(())
//!     }
//! }
//!
//! let instance = Instance::new(&InstanceDescriptor::null());
//! let mut renderer = Renderer::new(&instance, instance.create_null_surface(320, 240)?)?;
//! assert!(renderer.frame(&[0; 256])?);
//!
//! // The window is resized: the frame is skipped and the swapchain configured again.
//! renderer.surface.set_null_size(640, 480)?;
//! (renderer.config.width, renderer.config.height) = (640, 480);
//! assert!(!renderer.frame(&[0; 256])?);
//! assert!(renderer.frame(&[0; 256])?);
//!
//! // A driver reset.
//! renderer.device.simulate_device_loss();
//! assert!(matches!(renderer.frame(&[0; 256]), Err(RhiError::DeviceLost)));
//! renderer.recover(instance.create_null_surface(640, 480)?)?;
//! assert!(renderer.frame(&[0; 256])?);
//! renderer.device.wait_idle()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![forbid(unsafe_code)]
// Proving that wgpu's pipeline handles are `Send` walks wgpu-core's registry types deeper than
// the default 128; newer compilers turn that overflow into an error (rust-lang/rust#159228).
#![recursion_limit = "256"]

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

/// The `pragma_core` this crate is built with: its [`ThreadPool`](pragma_core::jobs::ThreadPool), frame graph `Resources` and
/// `Executor` are the ones [`CommandGraph`], [`Device::compile_render_pipeline`] and
/// [`Device::upload_async`] take. Naming them through this re-export keeps the versions equal.
///
/// ```
/// use pragma_rhi::pragma_core::jobs::{PoolConfig, ThreadPool};
///
/// let pool = ThreadPool::new(PoolConfig::default().with_workers(2))?;
/// # let _ = pool;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub use pragma_core;
