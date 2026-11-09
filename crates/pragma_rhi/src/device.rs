//! The device: creates, inspects and destroys resources.

use core::sync::atomic::{AtomicU32, Ordering};

use crate::backend::{BufferObj, DeviceBackend, SamplerObj, TextureObj, TextureViewObj};
use crate::handle::Table;
use crate::resource::{validate_buffer, validate_sampler, validate_texture, validate_texture_view};
use crate::{
    Backend, Buffer, BufferDescriptor, BufferInfo, Limits, ResourceKind, RhiError, Sampler,
    SamplerDescriptor, SamplerInfo, Texture, TextureDescriptor, TextureInfo, TextureView,
    TextureViewDescriptor, TextureViewInfo,
};

/// Describes a [`Device`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct DeviceDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The limits the device validates every descriptor against; at most the adapter's.
    pub required_limits: Limits,
}

#[derive(Debug)]
struct BufferEntry {
    info: BufferInfo,
    obj: BufferObj,
}

#[derive(Debug)]
struct TextureEntry {
    info: TextureInfo,
    bytes: u64,
    views: u32,
    obj: TextureObj,
}

#[derive(Debug)]
struct ViewEntry {
    info: TextureViewInfo,
    // Kept alive until the view is destroyed; read by command recording (roadmap M3).
    #[expect(dead_code, reason = "read by command recording (roadmap M3)")]
    obj: TextureViewObj,
}

#[derive(Debug)]
struct SamplerEntry {
    info: SamplerInfo,
    #[expect(dead_code, reason = "read by bind groups (roadmap M2)")]
    obj: SamplerObj,
}

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

/// A logical device of an [`Adapter`](crate::Adapter): owns its resources.
///
/// Resources are created and destroyed explicitly and named by `Copy` handles
/// (DECISIONS.md #9). Every descriptor is validated against [`Self::limits`] first. After a
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
    limits: Limits,
    backend: DeviceBackend,
    buffers: Table<Buffer, BufferEntry>,
    textures: Table<Texture, TextureEntry>,
    views: Table<TextureView, ViewEntry>,
    samplers: Table<Sampler, SamplerEntry>,
}

impl Device {
    pub(crate) fn new(backend: DeviceBackend, limits: Limits) -> Result<Self, RhiError> {
        let id = next_device_id()?;
        Ok(Self {
            limits,
            backend,
            buffers: Table::new(id),
            textures: Table::new(id),
            views: Table::new(id),
            samplers: Table::new(id),
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
        }
    }

    /// Device memory in use by buffers and textures, if the backend accounts it (the null
    /// backend does; wgpu does not expose it).
    #[must_use]
    pub const fn allocated_bytes(&self) -> Option<u64> {
        self.backend.allocated()
    }

    fn alive(&self) -> Result<(), RhiError> {
        if self.backend.is_lost() {
            Err(RhiError::DeviceLost)
        } else {
            Ok(())
        }
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
        self.buffers.insert(BufferEntry { info, obj })
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
    /// [`RhiError::StaleHandle`] (also on a second destroy), [`RhiError::ForeignHandle`],
    /// [`RhiError::DeviceLost`].
    pub fn destroy_buffer(&mut self, buffer: Buffer) -> Result<(), RhiError> {
        self.alive()?;
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
        let views = self.textures.get(texture)?.views;
        if views > 0 {
            return Err(RhiError::ResourceInUse {
                resource: ResourceKind::Texture,
                dependents: views,
            });
        }
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
        let view = self.views.insert(ViewEntry { info, obj })?;
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
        self.samplers.insert(SamplerEntry { info, obj })
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
        self.samplers.remove(sampler)?;
        Ok(())
    }
}
