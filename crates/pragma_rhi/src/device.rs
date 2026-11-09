//! The device: creates, inspects and destroys resources.

use core::num::NonZeroU64;
use core::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;

use pragma_core::tasks::Executor;

use crate::backend::{
    BindGroupLayoutObj, BindGroupObj, BoundObj, BufferObj, ComputeJob, ComputePipelineObj,
    DeviceBackend, PipelineLayoutObj, RenderJob, RenderPipelineObj, SamplerObj, ShaderModuleObj,
    TextureObj, TextureViewObj,
};
use crate::binding::{
    check_buffer_binding, check_sampler_binding, check_texture_binding, validate_bind_group_layout,
    validate_pipeline_layout,
};
use crate::handle::Table;
use crate::pending::{Finished, Pending};
use crate::pipeline::{
    ComputeKey, PipelineMeta, RenderKey, Stage, late_sizes, validate_compute, validate_render,
};
use crate::resource::{validate_buffer, validate_sampler, validate_texture, validate_texture_view};
use crate::shader::{self, EntryReflection, ParsedShader};
use crate::{
    Backend, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupInfo, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupLayoutInfo, BindingResource,
    BindingType, Buffer, BufferDescriptor, BufferInfo, ComputePipeline, ComputePipelineDescriptor,
    ComputePipelineInfo, Limits, PendingComputePipeline, PendingRenderPipeline, PipelineCacheStats,
    PipelineLayout, PipelineLayoutDescriptor, PipelineLayoutInfo, RenderPipeline,
    RenderPipelineDescriptor, RenderPipelineInfo, ResourceKind, RhiError, Sampler,
    SamplerDescriptor, SamplerInfo, ShaderModule, ShaderModuleDescriptor, ShaderModuleInfo,
    ShaderStage, Texture, TextureDescriptor, TextureInfo, TextureView, TextureViewDescriptor,
    TextureViewInfo,
};

/// Describes a [`Device`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeviceDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The limits the device validates every descriptor against; at most the adapter's.
    pub required_limits: Limits,
}

// Dependency counters (`users`) count the bind groups, pipeline layouts and pipelines that
// reference a resource; a resource with users cannot be destroyed (`RhiError::ResourceInUse`).
// They saturate instead of overflowing: a saturated counter only keeps a resource alive.

#[derive(Debug)]
pub(crate) struct BufferEntry {
    pub(crate) info: BufferInfo,
    users: u32,
    pub(crate) obj: BufferObj,
}

#[derive(Debug)]
pub(crate) struct TextureEntry {
    pub(crate) info: TextureInfo,
    bytes: u64,
    views: u32,
    pub(crate) obj: TextureObj,
}

#[derive(Debug)]
pub(crate) struct ViewEntry {
    pub(crate) info: TextureViewInfo,
    users: u32,
    pub(crate) obj: TextureViewObj,
}

#[derive(Debug)]
struct SamplerEntry {
    info: SamplerInfo,
    users: u32,
    obj: SamplerObj,
}

#[derive(Debug)]
struct ModuleEntry {
    parsed: ParsedShader,
    /// Never reused on this device; names the module in pipeline cache keys.
    serial: u64,
    obj: ShaderModuleObj,
}

#[derive(Debug)]
pub(crate) struct GroupLayoutEntry {
    pub(crate) info: BindGroupLayoutInfo,
    users: u32,
    obj: BindGroupLayoutObj,
}

#[derive(Debug)]
pub(crate) struct GroupEntry {
    pub(crate) info: BindGroupInfo,
    pub(crate) obj: BindGroupObj,
}

#[derive(Debug)]
pub(crate) struct PipelineLayoutEntry {
    pub(crate) info: PipelineLayoutInfo,
    /// See [`ModuleEntry::serial`].
    serial: u64,
    users: u32,
    obj: PipelineLayoutObj,
}

#[derive(Debug)]
pub(crate) struct RenderEntry {
    pub(crate) info: RenderPipelineInfo,
    pub(crate) meta: PipelineMeta,
    pub(crate) obj: RenderPipelineObj,
}

#[derive(Debug)]
pub(crate) struct ComputeEntry {
    pub(crate) info: ComputePipelineInfo,
    pub(crate) meta: PipelineMeta,
    pub(crate) obj: ComputePipelineObj,
}

/// Compiled pipelines by everything they depend on. An entry is dropped when its shader module
/// or pipeline layout is destroyed; pipelines created from it keep their own reference.
#[derive(Debug, Default)]
struct PipelineCache {
    render: HashMap<RenderKey, RenderPipelineObj>,
    compute: HashMap<ComputeKey, ComputePipelineObj>,
    hits: u64,
    misses: u64,
}

impl PipelineCache {
    fn purge(&mut self, serial: u64) {
        self.render.retain(|key, _| !key.uses(serial));
        self.compute.retain(|key, _| !key.uses(serial));
    }
}

/// A validated pipeline: either already in the cache or a compilation to run.
enum Prepared<O, J> {
    Cached(O),
    Compile(J),
}

/// A validated render pipeline: its cache key, its info and draw metadata, its object or job.
type PreparedRender = (
    RenderKey,
    (RenderPipelineInfo, PipelineMeta),
    Prepared<RenderPipelineObj, RenderJob>,
);

/// See [`PreparedRender`].
type PreparedCompute = (
    ComputeKey,
    (ComputePipelineInfo, PipelineMeta),
    Prepared<ComputePipelineObj, ComputeJob>,
);

/// Device ids start at 1 and are never reused within a process, so a handle of a dropped
/// device is foreign to every later device.
static NEXT_DEVICE_ID: AtomicU32 = AtomicU32::new(1);

fn next_device_id() -> Result<u32, RhiError> {
    let mut id = NEXT_DEVICE_ID.load(Ordering::Relaxed);
    loop {
        let next = id
            .checked_add(1)
            .ok_or_else(|| RhiError::limit("devices per process", u32::MAX, u32::MAX - 1))?;
        match NEXT_DEVICE_ID.compare_exchange_weak(id, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return Ok(id),
            Err(current) => id = current,
        }
    }
}

const fn in_use(resource: ResourceKind, users: u32) -> Result<(), RhiError> {
    if users > 0 {
        Err(RhiError::ResourceInUse {
            resource,
            dependents: users,
        })
    } else {
        Ok(())
    }
}

/// A logical device of an [`Adapter`](crate::Adapter): owns its resources.
///
/// Resources are created and destroyed explicitly and named by `Copy` handles
/// (DECISIONS.md #9). Every descriptor is validated against [`Self::limits`] first. A resource
/// another one depends on — a buffer, sampler or view in a bind group, a layout used by a bind
/// group, pipeline layout or pipeline — cannot be destroyed before its dependents. After a
/// device loss every call returns [`RhiError::DeviceLost`]; create a new device.
///
/// ```
/// use pragma_rhi::{
///     BufferDescriptor, BufferUsages, DeviceDescriptor, Instance, InstanceDescriptor, RhiError,
/// };
///
/// let instance = Instance::new(&InstanceDescriptor::null());
/// let adapter = instance.adapters().into_iter().next().ok_or(RhiError::DeviceLost)?;
/// let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let buffer = device.create_buffer(&BufferDescriptor {
///     label: Some("vertices"),
///     size: 1024,
///     usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
/// })?;
/// assert_eq!(device.buffer_info(buffer)?.size, 1024);
/// device.destroy_buffer(buffer)?;
/// assert!(matches!(device.buffer_info(buffer), Err(RhiError::StaleHandle { .. })));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug)]
pub struct Device {
    pub(crate) limits: Limits,
    pub(crate) backend: DeviceBackend,
    next_serial: u64,
    pub(crate) buffers: Table<Buffer, BufferEntry>,
    pub(crate) textures: Table<Texture, TextureEntry>,
    pub(crate) views: Table<TextureView, ViewEntry>,
    samplers: Table<Sampler, SamplerEntry>,
    shader_modules: Table<ShaderModule, ModuleEntry>,
    pub(crate) bind_group_layouts: Table<BindGroupLayout, GroupLayoutEntry>,
    pub(crate) bind_groups: Table<BindGroup, GroupEntry>,
    pub(crate) pipeline_layouts: Table<PipelineLayout, PipelineLayoutEntry>,
    pub(crate) render_pipelines: Table<RenderPipeline, RenderEntry>,
    pub(crate) compute_pipelines: Table<ComputePipeline, ComputeEntry>,
    cache: PipelineCache,
}

impl Device {
    pub(crate) fn new(backend: DeviceBackend, limits: Limits) -> Result<Self, RhiError> {
        let id = next_device_id()?;
        Ok(Self {
            limits,
            backend,
            next_serial: 0,
            buffers: Table::new(id),
            textures: Table::new(id),
            views: Table::new(id),
            samplers: Table::new(id),
            shader_modules: Table::new(id),
            bind_group_layouts: Table::new(id),
            bind_groups: Table::new(id),
            pipeline_layouts: Table::new(id),
            render_pipelines: Table::new(id),
            compute_pipelines: Table::new(id),
            cache: PipelineCache::default(),
        })
    }

    /// The backend.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend.backend()
    }

    /// The limits descriptors are validated against.
    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Whether the device was lost.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.backend.is_lost()
    }

    /// Loses the device on purpose, as a driver reset would, to test recovery paths: every
    /// later call returns [`RhiError::DeviceLost`]. On a GPU backend this destroys the device.
    pub fn simulate_device_loss(&mut self) {
        self.backend.lose();
    }

    /// The number of live resources of a kind.
    #[must_use]
    pub const fn live(&self, kind: ResourceKind) -> usize {
        match kind {
            ResourceKind::Buffer => self.buffers.len(),
            ResourceKind::Texture => self.textures.len(),
            ResourceKind::TextureView => self.views.len(),
            ResourceKind::Sampler => self.samplers.len(),
            ResourceKind::ShaderModule => self.shader_modules.len(),
            ResourceKind::BindGroupLayout => self.bind_group_layouts.len(),
            ResourceKind::BindGroup => self.bind_groups.len(),
            ResourceKind::PipelineLayout => self.pipeline_layouts.len(),
            ResourceKind::RenderPipeline => self.render_pipelines.len(),
            ResourceKind::ComputePipeline => self.compute_pipelines.len(),
        }
    }

    /// Device memory in use by buffers and textures, if the backend accounts it (the null
    /// backend does; wgpu does not expose it).
    #[must_use]
    pub const fn allocated_bytes(&self) -> Option<u64> {
        self.backend.allocated()
    }

    pub(crate) fn alive(&self) -> Result<(), RhiError> {
        if self.backend.is_lost() {
            Err(RhiError::DeviceLost)
        } else {
            Ok(())
        }
    }

    fn next_serial(&mut self) -> Result<u64, RhiError> {
        let serial = self.next_serial;
        self.next_serial = serial.checked_add(1).ok_or_else(|| {
            RhiError::limit(
                "shader modules and pipeline layouts",
                u64::MAX,
                u64::MAX - 1,
            )
        })?;
        Ok(serial)
    }

    // -- buffers --------------------------------------------------------------------------

    /// Creates a buffer.
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidDescriptor`] / [`RhiError::LimitExceeded`] for a descriptor the
    /// device cannot create, [`RhiError::OutOfMemory`], [`RhiError::DeviceLost`],
    /// [`RhiError::Backend`].
    pub fn create_buffer(&mut self, desc: &BufferDescriptor<'_>) -> Result<Buffer, RhiError> {
        self.alive()?;
        let info = validate_buffer(desc, &self.limits)?;
        self.buffers.reserve()?;
        let obj = self.backend.create_buffer(desc.label, &info)?;
        self.buffers.insert(BufferEntry {
            info,
            users: 0,
            obj,
        })
    }

    /// A buffer's properties.
    ///
    /// # Errors
    ///
    /// [`RhiError::StaleHandle`], [`RhiError::ForeignHandle`], [`RhiError::DeviceLost`].
    pub fn buffer_info(&self, buffer: Buffer) -> Result<BufferInfo, RhiError> {
        self.alive()?;
        Ok(self.buffers.get(buffer)?.info)
    }

    /// Destroys a buffer; the handle and its copies go stale.
    ///
    /// # Errors
    ///
    /// [`RhiError::ResourceInUse`] while a bind group uses it, [`RhiError::StaleHandle`] (also
    /// on a second destroy), [`RhiError::ForeignHandle`], [`RhiError::DeviceLost`].
    pub fn destroy_buffer(&mut self, buffer: Buffer) -> Result<(), RhiError> {
        self.alive()?;
        in_use(ResourceKind::Buffer, self.buffers.get(buffer)?.users)?;
        let entry = self.buffers.remove(buffer)?;
        self.backend.destroy_buffer(entry.obj, &entry.info);
        Ok(())
    }

    // -- textures -------------------------------------------------------------------------

    /// Creates a texture.
    ///
    /// # Errors
    ///
    /// As [`Self::create_buffer`].
    pub fn create_texture(&mut self, desc: &TextureDescriptor<'_>) -> Result<Texture, RhiError> {
        self.alive()?;
        let (info, bytes) = validate_texture(desc, &self.limits)?;
        self.textures.reserve()?;
        let obj = self.backend.create_texture(desc.label, &info, bytes)?;
        self.textures.insert(TextureEntry {
            info,
            bytes,
            views: 0,
            obj,
        })
    }

    /// A texture's properties.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn texture_info(&self, texture: Texture) -> Result<TextureInfo, RhiError> {
        self.alive()?;
        Ok(self.textures.get(texture)?.info)
    }

    /// Destroys a texture.
    ///
    /// # Errors
    ///
    /// [`RhiError::ResourceInUse`] while the texture has views; otherwise as
    /// [`Self::destroy_buffer`].
    pub fn destroy_texture(&mut self, texture: Texture) -> Result<(), RhiError> {
        self.alive()?;
        in_use(ResourceKind::Texture, self.textures.get(texture)?.views)?;
        let entry = self.textures.remove(texture)?;
        self.backend.destroy_texture(entry.obj, entry.bytes);
        Ok(())
    }

    // -- texture views --------------------------------------------------------------------

    /// Creates a view of a texture.
    ///
    /// # Errors
    ///
    /// The texture handle's errors, then as [`Self::create_buffer`].
    pub fn create_texture_view(
        &mut self,
        texture: Texture,
        desc: &TextureViewDescriptor<'_>,
    ) -> Result<TextureView, RhiError> {
        self.alive()?;
        let parent = self.textures.get(texture)?;
        let dependents = parent.views.checked_add(1).ok_or(RhiError::limit(
            "views per texture",
            u64::from(u32::MAX) + 1,
            u32::MAX,
        ))?;
        let info = validate_texture_view(texture, &parent.info, desc)?;
        self.views.reserve()?;
        let obj = self
            .backend
            .create_texture_view(desc.label, &parent.obj, &info)?;
        let view = self.views.insert(ViewEntry {
            info,
            users: 0,
            obj,
        })?;
        self.textures.get_mut(texture)?.views = dependents;
        Ok(view)
    }

    /// A view's properties.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn texture_view_info(&self, view: TextureView) -> Result<TextureViewInfo, RhiError> {
        self.alive()?;
        Ok(self.views.get(view)?.info)
    }

    /// Destroys a view.
    ///
    /// # Errors
    ///
    /// As [`Self::destroy_buffer`].
    pub fn destroy_texture_view(&mut self, view: TextureView) -> Result<(), RhiError> {
        self.alive()?;
        in_use(ResourceKind::TextureView, self.views.get(view)?.users)?;
        let entry = self.views.remove(view)?;
        // A texture cannot be destroyed while it has views, so the parent is alive.
        let parent = self.textures.get_mut(entry.info.texture)?;
        parent.views = parent.views.saturating_sub(1);
        Ok(())
    }

    // -- samplers -------------------------------------------------------------------------

    /// Creates a sampler.
    ///
    /// # Errors
    ///
    /// As [`Self::create_buffer`].
    pub fn create_sampler(&mut self, desc: &SamplerDescriptor<'_>) -> Result<Sampler, RhiError> {
        self.alive()?;
        let info = validate_sampler(desc)?;
        self.samplers.reserve()?;
        let obj = self.backend.create_sampler(desc.label, &info)?;
        self.samplers.insert(SamplerEntry {
            info,
            users: 0,
            obj,
        })
    }

    /// A sampler's properties.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn sampler_info(&self, sampler: Sampler) -> Result<SamplerInfo, RhiError> {
        self.alive()?;
        Ok(self.samplers.get(sampler)?.info)
    }

    /// Destroys a sampler.
    ///
    /// # Errors
    ///
    /// As [`Self::destroy_buffer`].
    pub fn destroy_sampler(&mut self, sampler: Sampler) -> Result<(), RhiError> {
        self.alive()?;
        in_use(ResourceKind::Sampler, self.samplers.get(sampler)?.users)?;
        self.samplers.remove(sampler)?;
        Ok(())
    }

    // -- shader modules -------------------------------------------------------------------

    /// Creates a shader module from WGSL or SPIR-V. naga parses and validates it here, on every
    /// backend, so the null backend rejects exactly the shaders wgpu would.
    ///
    /// # Errors
    ///
    /// [`RhiError::Shader`] with the source location of a WGSL error; otherwise as
    /// [`Self::create_buffer`].
    pub fn create_shader_module(
        &mut self,
        desc: &ShaderModuleDescriptor<'_>,
    ) -> Result<ShaderModule, RhiError> {
        self.alive()?;
        let parsed = shader::parse(desc.source)?;
        let serial = self.next_serial()?;
        self.shader_modules.reserve()?;
        let obj = self
            .backend
            .create_shader_module(desc.label, &parsed.module)?;
        self.shader_modules.insert(ModuleEntry {
            parsed,
            serial,
            obj,
        })
    }

    /// A shader module's entry points.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn shader_module_info(&self, module: ShaderModule) -> Result<ShaderModuleInfo, RhiError> {
        self.alive()?;
        Ok(self.shader_modules.get(module)?.parsed.info.clone())
    }

    /// Destroys a shader module. Pipelines created from it stay valid; cached pipelines
    /// compiled from it are dropped from the cache.
    ///
    /// # Errors
    ///
    /// [`RhiError::StaleHandle`], [`RhiError::ForeignHandle`], [`RhiError::DeviceLost`].
    pub fn destroy_shader_module(&mut self, module: ShaderModule) -> Result<(), RhiError> {
        self.alive()?;
        let entry = self.shader_modules.remove(module)?;
        self.cache.purge(entry.serial);
        Ok(())
    }

    // -- bind group layouts ---------------------------------------------------------------

    /// Creates a bind group layout.
    ///
    /// # Errors
    ///
    /// As [`Self::create_buffer`].
    pub fn create_bind_group_layout(
        &mut self,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> Result<BindGroupLayout, RhiError> {
        self.alive()?;
        let entries = validate_bind_group_layout(desc, &self.limits)?;
        self.bind_group_layouts.reserve()?;
        let obj = self
            .backend
            .create_bind_group_layout(desc.label, &entries)?;
        self.bind_group_layouts.insert(GroupLayoutEntry {
            info: BindGroupLayoutInfo { entries },
            users: 0,
            obj,
        })
    }

    /// A bind group layout's bindings.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn bind_group_layout_info(
        &self,
        layout: BindGroupLayout,
    ) -> Result<BindGroupLayoutInfo, RhiError> {
        self.alive()?;
        Ok(self.bind_group_layouts.get(layout)?.info.clone())
    }

    /// Destroys a bind group layout.
    ///
    /// # Errors
    ///
    /// [`RhiError::ResourceInUse`] while a bind group or pipeline layout uses it; otherwise as
    /// [`Self::destroy_shader_module`].
    pub fn destroy_bind_group_layout(&mut self, layout: BindGroupLayout) -> Result<(), RhiError> {
        self.alive()?;
        in_use(
            ResourceKind::BindGroupLayout,
            self.bind_group_layouts.get(layout)?.users,
        )?;
        self.bind_group_layouts.remove(layout)?;
        Ok(())
    }

    // -- bind groups ----------------------------------------------------------------------

    /// Creates a bind group: one resource for every binding of `layout`. While it lives, the
    /// layout and every bound buffer, sampler and texture view cannot be destroyed.
    ///
    /// # Errors
    ///
    /// The handles' errors; [`RhiError::InvalidDescriptor`] for a missing, extra or duplicate
    /// binding, a resource of the wrong kind, usage, range or format; otherwise as
    /// [`Self::create_buffer`].
    pub fn create_bind_group(
        &mut self,
        layout: BindGroupLayout,
        desc: &BindGroupDescriptor<'_>,
    ) -> Result<BindGroup, RhiError> {
        const KIND: ResourceKind = ResourceKind::BindGroup;
        self.alive()?;
        let layout_entry = self.bind_group_layouts.get(layout)?;
        let slots = &layout_entry.info.entries;
        if desc.entries.len() != slots.len() {
            return Err(RhiError::invalid(
                KIND,
                "a bind group has exactly one entry for every binding of its layout",
            ));
        }
        let mut entries: Vec<BindGroupEntry> = desc.entries.to_vec();
        entries.sort_by_key(|entry| entry.binding);
        for (entry, slot) in entries.iter_mut().zip(slots) {
            if entry.binding != slot.binding {
                return Err(RhiError::invalid(
                    KIND,
                    "an entry's binding is missing from the layout or appears twice",
                ));
            }
            match (slot.ty, &mut entry.resource) {
                (
                    BindingType::Buffer {
                        ty,
                        min_binding_size,
                        ..
                    },
                    BindingResource::Buffer(binding),
                ) => {
                    let buffer = &self.buffers.get(binding.buffer)?.info;
                    let size =
                        check_buffer_binding(ty, min_binding_size, buffer, binding, &self.limits)?;
                    binding.size = NonZeroU64::new(size);
                }
                (BindingType::Sampler(ty), BindingResource::Sampler(sampler)) => {
                    check_sampler_binding(ty, &self.samplers.get(*sampler)?.info)?;
                }
                (
                    ty @ (BindingType::Texture { .. } | BindingType::StorageTexture { .. }),
                    BindingResource::TextureView(view),
                ) => {
                    let view = &self.views.get(*view)?.info;
                    let texture = &self.textures.get(view.texture)?.info;
                    check_texture_binding(&ty, view, texture)?;
                }
                _ => {
                    return Err(RhiError::invalid(
                        KIND,
                        "an entry's resource is not of its binding's kind",
                    ));
                }
            }
        }
        self.bind_groups.reserve()?;
        let bound = entries
            .iter()
            .map(|entry| {
                let obj = match entry.resource {
                    BindingResource::Buffer(binding) => BoundObj::Buffer {
                        obj: &self.buffers.get(binding.buffer)?.obj,
                        offset: binding.offset,
                        size: binding
                            .size
                            .ok_or(RhiError::invalid(KIND, "a buffer range is empty"))?,
                    },
                    BindingResource::Sampler(sampler) => {
                        BoundObj::Sampler(&self.samplers.get(sampler)?.obj)
                    }
                    BindingResource::TextureView(view) => {
                        BoundObj::TextureView(&self.views.get(view)?.obj)
                    }
                };
                Ok((entry.binding, obj))
            })
            .collect::<Result<Vec<_>, RhiError>>()?;
        let obj = self
            .backend
            .create_bind_group(desc.label, &layout_entry.obj, &bound)?;
        drop(bound);
        for entry in &entries {
            self.add_user(entry.resource, true)?;
        }
        let layout_entry = self.bind_group_layouts.get_mut(layout)?;
        layout_entry.users = layout_entry.users.saturating_add(1);
        self.bind_groups.insert(GroupEntry {
            info: BindGroupInfo { layout, entries },
            obj,
        })
    }

    /// Counts a bind group as a user of a bound resource (`add`) or stops counting it.
    fn add_user(&mut self, resource: BindingResource, add: bool) -> Result<(), RhiError> {
        let users = match resource {
            BindingResource::Buffer(binding) => &mut self.buffers.get_mut(binding.buffer)?.users,
            BindingResource::Sampler(sampler) => &mut self.samplers.get_mut(sampler)?.users,
            BindingResource::TextureView(view) => &mut self.views.get_mut(view)?.users,
        };
        *users = if add {
            users.saturating_add(1)
        } else {
            users.saturating_sub(1)
        };
        Ok(())
    }

    /// A bind group's layout and entries.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn bind_group_info(&self, group: BindGroup) -> Result<BindGroupInfo, RhiError> {
        self.alive()?;
        Ok(self.bind_groups.get(group)?.info.clone())
    }

    /// Destroys a bind group, releasing its layout and resources.
    ///
    /// # Errors
    ///
    /// As [`Self::destroy_shader_module`].
    pub fn destroy_bind_group(&mut self, group: BindGroup) -> Result<(), RhiError> {
        self.alive()?;
        let entry = self.bind_groups.remove(group)?;
        // Resources with users cannot be destroyed, so everything referenced is alive.
        for bound in &entry.info.entries {
            self.add_user(bound.resource, false)?;
        }
        let layout = self.bind_group_layouts.get_mut(entry.info.layout)?;
        layout.users = layout.users.saturating_sub(1);
        Ok(())
    }

    // -- pipeline layouts -----------------------------------------------------------------

    /// Creates a pipeline layout. Its bind group layouts cannot be destroyed while it lives.
    ///
    /// # Errors
    ///
    /// The handles' errors; [`RhiError::LimitExceeded`] for too many groups or bindings per
    /// stage; otherwise as [`Self::create_buffer`].
    pub fn create_pipeline_layout(
        &mut self,
        desc: &PipelineLayoutDescriptor<'_>,
    ) -> Result<PipelineLayout, RhiError> {
        self.alive()?;
        let serial = self.next_serial()?;
        let groups = desc
            .bind_group_layouts
            .iter()
            .map(|layout| self.bind_group_layouts.get(*layout))
            .collect::<Result<Vec<_>, RhiError>>()?;
        let entries: Vec<&[BindGroupLayoutEntry]> = groups
            .iter()
            .map(|group| group.info.entries.as_slice())
            .collect();
        validate_pipeline_layout(&entries, &self.limits)?;
        self.pipeline_layouts.reserve()?;
        let objs: Vec<&BindGroupLayoutObj> = groups.iter().map(|group| &group.obj).collect();
        let obj = self.backend.create_pipeline_layout(desc.label, &objs)?;
        for layout in desc.bind_group_layouts {
            let group = self.bind_group_layouts.get_mut(*layout)?;
            group.users = group.users.saturating_add(1);
        }
        self.pipeline_layouts.insert(PipelineLayoutEntry {
            info: PipelineLayoutInfo {
                bind_group_layouts: desc.bind_group_layouts.to_vec(),
            },
            serial,
            users: 0,
            obj,
        })
    }

    /// A pipeline layout's bind group layouts.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn pipeline_layout_info(
        &self,
        layout: PipelineLayout,
    ) -> Result<PipelineLayoutInfo, RhiError> {
        self.alive()?;
        Ok(self.pipeline_layouts.get(layout)?.info.clone())
    }

    /// Destroys a pipeline layout; cached pipelines compiled with it are dropped.
    ///
    /// # Errors
    ///
    /// [`RhiError::ResourceInUse`] while a pipeline uses it; otherwise as
    /// [`Self::destroy_shader_module`].
    pub fn destroy_pipeline_layout(&mut self, layout: PipelineLayout) -> Result<(), RhiError> {
        self.alive()?;
        in_use(
            ResourceKind::PipelineLayout,
            self.pipeline_layouts.get(layout)?.users,
        )?;
        let entry = self.pipeline_layouts.remove(layout)?;
        for group in &entry.info.bind_group_layouts {
            let group = self.bind_group_layouts.get_mut(*group)?;
            group.users = group.users.saturating_sub(1);
        }
        self.cache.purge(entry.serial);
        Ok(())
    }

    // -- render pipelines -----------------------------------------------------------------

    fn prepare_render(
        &mut self,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Result<PreparedRender, RhiError> {
        self.alive()?;
        let layout = self.pipeline_layouts.get(desc.layout)?;
        let groups = layout
            .info
            .bind_group_layouts
            .iter()
            .map(|group| {
                self.bind_group_layouts
                    .get(*group)
                    .map(|group| group.info.entries.as_slice())
            })
            .collect::<Result<Vec<_>, RhiError>>()?;
        let vertex_module = self.shader_modules.get(desc.vertex.module)?;
        let vertex = Stage {
            module: vertex_module.serial,
            entry: vertex_module
                .parsed
                .entry_point(ShaderStage::Vertex, desc.vertex.entry_point)?,
        };
        let fragment_module = desc
            .fragment
            .map(|fragment| self.shader_modules.get(fragment.module))
            .transpose()?;
        let fragment = match (desc.fragment, fragment_module) {
            (Some(state), Some(module)) => Some(Stage {
                module: module.serial,
                entry: module
                    .parsed
                    .entry_point(ShaderStage::Fragment, state.entry_point)?,
            }),
            _ => None,
        };
        let (key, info) =
            validate_render(desc, layout.serial, &groups, vertex, fragment, &self.limits)?;
        let stages: Vec<&EntryReflection> = core::iter::once(vertex.entry)
            .chain(fragment.map(|stage| stage.entry))
            .collect();
        let meta = key.meta(late_sizes(&stages, &groups));
        let info = (info, meta);
        if let Some(obj) = self.cache.render.get(&key) {
            let obj = obj.clone();
            self.cache.hits = self.cache.hits.saturating_add(1);
            return Ok((key, info, Prepared::Cached(obj)));
        }
        let job = self.backend.render_job(
            desc.label,
            &layout.obj,
            &vertex_module.obj,
            fragment_module.map(|module| &module.obj),
            &key,
        )?;
        self.cache.misses = self.cache.misses.saturating_add(1);
        Ok((key, info, Prepared::Compile(job)))
    }

    fn install_render(
        &mut self,
        finished: Finished<RenderKey, (RenderPipelineInfo, PipelineMeta), RenderPipelineObj>,
    ) -> Result<RenderPipeline, RhiError> {
        self.alive()?;
        // The layout may have been destroyed while the pipeline compiled.
        self.pipeline_layouts.get(finished.layout)?;
        self.render_pipelines.reserve()?;
        if finished.compiled {
            self.cache
                .render
                .entry(finished.key)
                .or_insert_with(|| finished.obj.clone());
        }
        let layout = self.pipeline_layouts.get_mut(finished.layout)?;
        layout.users = layout.users.saturating_add(1);
        let (info, meta) = finished.info;
        self.render_pipelines.insert(RenderEntry {
            info,
            meta,
            obj: finished.obj,
        })
    }

    /// Creates a render pipeline, compiling it on the calling thread unless an equal one is in
    /// the pipeline cache. Its layout cannot be destroyed while it lives.
    ///
    /// # Errors
    ///
    /// The handles' errors; [`RhiError::ShaderInterface`] when a shader does not fit the
    /// layout, the vertex buffers or the color targets; [`RhiError::InvalidDescriptor`] /
    /// [`RhiError::LimitExceeded`] for the descriptor's own rules; otherwise as
    /// [`Self::create_buffer`].
    pub fn create_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Result<RenderPipeline, RhiError> {
        let (key, info, prepared) = self.prepare_render(desc)?;
        let (obj, compiled) = match prepared {
            Prepared::Cached(obj) => (obj, false),
            Prepared::Compile(job) => (job.run()?, true),
        };
        self.install_render(Finished {
            layout: desc.layout,
            key,
            info,
            obj,
            compiled,
        })
    }

    /// Validates a render pipeline now and compiles it on `executor`'s blocking pool, off the
    /// frame thread (a cache hit needs no compilation). Install the result with
    /// [`Self::poll_render_pipeline`] or [`Self::wait_render_pipeline`].
    ///
    /// # Errors
    ///
    /// The descriptor errors of [`Self::create_render_pipeline`]; compilation errors arrive
    /// when polling.
    pub fn compile_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor<'_>,
        executor: &Executor,
    ) -> Result<PendingRenderPipeline, RhiError> {
        let (key, info, prepared) = self.prepare_render(desc)?;
        Ok(PendingRenderPipeline(match prepared {
            Prepared::Cached(obj) => Pending::cached(desc.layout, key, info, obj),
            Prepared::Compile(job) => {
                Pending::compiling(desc.layout, key, info, executor.unblock(move || job.run()))
            }
        }))
    }

    /// Installs a compiled render pipeline: `None` while it compiles, then its result once.
    ///
    /// # Errors
    ///
    /// The compilation's errors, [`RhiError::Task`] if the task failed (or the result was
    /// already taken), [`RhiError::StaleHandle`] if the layout was destroyed meanwhile,
    /// [`RhiError::DeviceLost`].
    pub fn poll_render_pipeline(
        &mut self,
        pending: &mut PendingRenderPipeline,
    ) -> Option<Result<RenderPipeline, RhiError>> {
        let finished = pending.0.poll()?;
        Some(finished.and_then(|finished| self.install_render(finished)))
    }

    /// Blocks until a render pipeline is compiled and installs it. Only outside pool work
    /// (a loading screen, a test); from a worker or a frame it returns
    /// [`RhiError::Task`] with `AsyncError::WouldBlock`.
    ///
    /// # Errors
    ///
    /// As [`Self::poll_render_pipeline`].
    pub fn wait_render_pipeline(
        &mut self,
        pending: PendingRenderPipeline,
    ) -> Result<RenderPipeline, RhiError> {
        let finished = pending.0.wait()?;
        self.install_render(finished)
    }

    /// A render pipeline's properties.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn render_pipeline_info(
        &self,
        pipeline: RenderPipeline,
    ) -> Result<RenderPipelineInfo, RhiError> {
        self.alive()?;
        Ok(self.render_pipelines.get(pipeline)?.info.clone())
    }

    /// Destroys a render pipeline, releasing its layout. A cached copy stays in the cache.
    ///
    /// # Errors
    ///
    /// As [`Self::destroy_shader_module`].
    pub fn destroy_render_pipeline(&mut self, pipeline: RenderPipeline) -> Result<(), RhiError> {
        self.alive()?;
        let entry = self.render_pipelines.remove(pipeline)?;
        let layout = self.pipeline_layouts.get_mut(entry.info.layout)?;
        layout.users = layout.users.saturating_sub(1);
        Ok(())
    }

    // -- compute pipelines ----------------------------------------------------------------

    fn prepare_compute(
        &mut self,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Result<PreparedCompute, RhiError> {
        self.alive()?;
        let layout = self.pipeline_layouts.get(desc.layout)?;
        let groups = layout
            .info
            .bind_group_layouts
            .iter()
            .map(|group| {
                self.bind_group_layouts
                    .get(*group)
                    .map(|group| group.info.entries.as_slice())
            })
            .collect::<Result<Vec<_>, RhiError>>()?;
        let module = self.shader_modules.get(desc.module)?;
        let stage = Stage {
            module: module.serial,
            entry: module
                .parsed
                .entry_point(ShaderStage::Compute, desc.entry_point)?,
        };
        let (key, info) = validate_compute(desc, layout.serial, &groups, stage, &self.limits)?;
        let meta = PipelineMeta {
            late_sizes: late_sizes(&[stage.entry], &groups),
            ..PipelineMeta::default()
        };
        let info = (info, meta);
        if let Some(obj) = self.cache.compute.get(&key) {
            let obj = obj.clone();
            self.cache.hits = self.cache.hits.saturating_add(1);
            return Ok((key, info, Prepared::Cached(obj)));
        }
        let job = self
            .backend
            .compute_job(desc.label, &layout.obj, &module.obj, &key)?;
        self.cache.misses = self.cache.misses.saturating_add(1);
        Ok((key, info, Prepared::Compile(job)))
    }

    fn install_compute(
        &mut self,
        finished: Finished<ComputeKey, (ComputePipelineInfo, PipelineMeta), ComputePipelineObj>,
    ) -> Result<ComputePipeline, RhiError> {
        self.alive()?;
        self.pipeline_layouts.get(finished.layout)?;
        self.compute_pipelines.reserve()?;
        if finished.compiled {
            self.cache
                .compute
                .entry(finished.key)
                .or_insert_with(|| finished.obj.clone());
        }
        let layout = self.pipeline_layouts.get_mut(finished.layout)?;
        layout.users = layout.users.saturating_add(1);
        let (info, meta) = finished.info;
        self.compute_pipelines.insert(ComputeEntry {
            info,
            meta,
            obj: finished.obj,
        })
    }

    /// Creates a compute pipeline; see [`Self::create_render_pipeline`].
    ///
    /// # Errors
    ///
    /// As [`Self::create_render_pipeline`]; a workgroup over the limits is a
    /// [`RhiError::ShaderInterface`].
    pub fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Result<ComputePipeline, RhiError> {
        let (key, info, prepared) = self.prepare_compute(desc)?;
        let (obj, compiled) = match prepared {
            Prepared::Cached(obj) => (obj, false),
            Prepared::Compile(job) => (job.run()?, true),
        };
        self.install_compute(Finished {
            layout: desc.layout,
            key,
            info,
            obj,
            compiled,
        })
    }

    /// Compiles a compute pipeline off the frame thread; see
    /// [`Self::compile_render_pipeline`].
    ///
    /// # Errors
    ///
    /// The descriptor errors of [`Self::create_compute_pipeline`].
    pub fn compile_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor<'_>,
        executor: &Executor,
    ) -> Result<PendingComputePipeline, RhiError> {
        let (key, info, prepared) = self.prepare_compute(desc)?;
        Ok(PendingComputePipeline(match prepared {
            Prepared::Cached(obj) => Pending::cached(desc.layout, key, info, obj),
            Prepared::Compile(job) => {
                Pending::compiling(desc.layout, key, info, executor.unblock(move || job.run()))
            }
        }))
    }

    /// Installs a compiled compute pipeline; see [`Self::poll_render_pipeline`].
    ///
    /// # Errors
    ///
    /// As [`Self::poll_render_pipeline`].
    pub fn poll_compute_pipeline(
        &mut self,
        pending: &mut PendingComputePipeline,
    ) -> Option<Result<ComputePipeline, RhiError>> {
        let finished = pending.0.poll()?;
        Some(finished.and_then(|finished| self.install_compute(finished)))
    }

    /// Blocks until a compute pipeline is compiled; see [`Self::wait_render_pipeline`].
    ///
    /// # Errors
    ///
    /// As [`Self::poll_render_pipeline`].
    pub fn wait_compute_pipeline(
        &mut self,
        pending: PendingComputePipeline,
    ) -> Result<ComputePipeline, RhiError> {
        let finished = pending.0.wait()?;
        self.install_compute(finished)
    }

    /// A compute pipeline's properties.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer_info`].
    pub fn compute_pipeline_info(
        &self,
        pipeline: ComputePipeline,
    ) -> Result<ComputePipelineInfo, RhiError> {
        self.alive()?;
        Ok(self.compute_pipelines.get(pipeline)?.info)
    }

    /// Destroys a compute pipeline; see [`Self::destroy_render_pipeline`].
    ///
    /// # Errors
    ///
    /// As [`Self::destroy_shader_module`].
    pub fn destroy_compute_pipeline(&mut self, pipeline: ComputePipeline) -> Result<(), RhiError> {
        self.alive()?;
        let entry = self.compute_pipelines.remove(pipeline)?;
        let layout = self.pipeline_layouts.get_mut(entry.info.layout)?;
        layout.users = layout.users.saturating_sub(1);
        Ok(())
    }

    // -- pipeline cache -------------------------------------------------------------------

    /// The pipeline cache's counters.
    #[must_use]
    pub fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            hits: self.cache.hits,
            misses: self.cache.misses,
            entries: self
                .cache
                .render
                .len()
                .saturating_add(self.cache.compute.len()),
        }
    }

    /// Drops every cached pipeline; live pipelines are not affected. The counters stay.
    pub fn clear_pipeline_cache(&mut self) {
        self.cache.render.clear();
        self.cache.compute.clear();
    }
}
