//! Surfaces and the swapchain (DECISIONS.md #18).
//!
//! A [`Surface`] comes from a window the engine owns — anything with `raw-window-handle`'s
//! [`HasWindowHandle`] and [`HasDisplayHandle`]; this crate opens no windows — or, for tests,
//! from the null backend ([`Instance::create_null_surface`]). A device configures it
//! ([`Device::configure_surface`]), acquires one frame at a time ([`Device::acquire_frame`]),
//! whose texture is an ordinary [`Texture`] handle for the frame's commands, and presents it
//! ([`Device::present`]). A resized window makes acquiring fail with
//! [`SurfaceError::Outdated`] until the surface is configured again with the new size.

use core::fmt;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::backend::{DeviceBackend, NullId, Obj, gpu_surface_capabilities};
use crate::device::TextureEntry;
use crate::frame::{Garbage, MAX_FRAMES_IN_FLIGHT};
use crate::instance::AdapterInner;
use crate::{
    Adapter, Device, Extent3d, Instance, ResourceKind, RhiError, Texture, TextureDimension,
    TextureFormat, TextureInfo, TextureUsages, TextureView, TextureViewDescriptor,
};

/// How presented frames reach the display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PresentMode {
    /// [`Self::FifoRelaxed`] where supported, otherwise [`Self::Fifo`].
    AutoVsync,
    /// [`Self::Immediate`], else [`Self::Mailbox`], else [`Self::Fifo`].
    AutoNoVsync,
    /// Wait for vertical blank; frames queue up. Supported everywhere.
    #[default]
    Fifo,
    /// As [`Self::Fifo`], but a late frame is shown at once (may tear).
    FifoRelaxed,
    /// No waiting: lowest latency, may tear.
    Immediate,
    /// Wait for vertical blank, replacing the queued frame with a newer one.
    Mailbox,
}

/// How the compositor blends a presented frame's alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CompositeAlphaMode {
    /// The first mode the surface supports.
    #[default]
    Auto,
    /// Alpha is ignored.
    Opaque,
    /// Color is premultiplied by alpha.
    PreMultiplied,
    /// Color is not premultiplied; the compositor multiplies.
    PostMultiplied,
    /// Whatever the platform's window system sets.
    Inherit,
}

/// What a surface supports on an adapter or a device. No formats means it cannot present
/// there at all.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct SurfaceCapabilities {
    /// The texture formats of the swapchain, preferred first (only formats of
    /// [`TextureFormat`]).
    pub formats: Vec<TextureFormat>,
    /// The concrete present modes ([`PresentMode::Fifo`] and others, not the `Auto` ones).
    pub present_modes: Vec<PresentMode>,
    /// The alpha modes, preferred first.
    pub alpha_modes: Vec<CompositeAlphaMode>,
    /// The usages a swapchain texture may have.
    pub usages: TextureUsages,
}

impl SurfaceCapabilities {
    /// Whether the surface can present on this adapter or device.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        !self.formats.is_empty()
    }
}

/// Configures a [`Surface`] for a device ([`Device::configure_surface`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceConfiguration {
    /// The usages of the swapchain textures; within [`SurfaceCapabilities::usages`].
    pub usage: TextureUsages,
    /// The swapchain format; one of [`SurfaceCapabilities::formats`].
    pub format: TextureFormat,
    /// The width in pixels, usually the window's inner width; not zero.
    pub width: u32,
    /// The height in pixels; not zero.
    pub height: u32,
    /// How frames reach the display.
    pub present_mode: PresentMode,
    /// How the compositor blends alpha.
    pub alpha_mode: CompositeAlphaMode,
    /// How many frames the presentation engine may queue (1 to 8); 2 by default.
    pub desired_maximum_frame_latency: u32,
}

impl Default for SurfaceConfiguration {
    fn default() -> Self {
        Self {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8UnormSrgb,
            width: 0,
            height: 0,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 2,
        }
    }
}

/// Why a surface operation failed, or what state the surface is in that the call does not
/// allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum SurfaceError {
    /// The window changed (usually resized): configure the surface again, then retry.
    #[error("the surface is outdated (the window changed); configure it again")]
    Outdated,
    /// The surface was lost: create a new one from the window and configure it.
    #[error("the surface was lost; create it again")]
    Lost,
    /// Acquiring the next frame timed out: skip this frame.
    #[error("acquiring the next frame timed out")]
    Timeout,
    /// The window is occluded or minimized: skip frames until it is visible again.
    #[error("the window is occluded")]
    Occluded,
    /// The adapter or device cannot present to this surface (a null surface on a GPU device,
    /// a window surface on the null device, an instance without the needed backend).
    #[error("this adapter or device cannot present to the surface")]
    Unsupported,
    /// The surface has not been configured, or not since it was lost.
    #[error("the surface is not configured")]
    NotConfigured,
    /// The surface is configured for another device.
    #[error("the surface is configured for another device")]
    OtherDevice,
    /// A frame is acquired and not yet presented or discarded.
    #[error("a frame of the surface is acquired and not yet presented")]
    FrameAcquired,
    /// No frame is acquired.
    #[error("no frame of the surface is acquired")]
    NoFrame,
    /// The texture is a swapchain texture: it goes away with [`Device::present`] or
    /// [`Device::discard_frame`], not with `destroy_texture`.
    #[error("the texture belongs to a surface; present or discard its frame instead")]
    SurfaceTexture,
    /// The configuration does not fit the surface's capabilities or the device's limits.
    #[error("invalid surface configuration: {0}")]
    InvalidConfiguration(&'static str),
}

/// An acquired frame: render into [`Self::view`] (or copy into [`Self::texture`]), submit, then
/// [`Device::present`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SurfaceFrame {
    /// The swapchain texture, valid until the frame is presented or discarded.
    pub texture: Texture,
    /// A default view of [`Self::texture`], destroyed with the frame.
    pub view: TextureView,
    /// The swapchain no longer matches the window exactly; it still works, but configuring
    /// the surface again is recommended.
    pub suboptimal: bool,
}

/// The null backend's stand-in for a window.
#[derive(Debug)]
pub(crate) struct NullSurface {
    width: u32,
    height: u32,
    lost: bool,
    presented: Option<Vec<u8>>,
}

pub(crate) enum SurfaceBackend {
    Wgpu(wgpu::Surface<'static>),
    Null(NullSurface),
}

/// The backend part of an acquired frame.
pub(crate) enum Acquired {
    Wgpu(wgpu::SurfaceTexture),
    Null(NullId),
}

#[derive(Debug, Clone, Copy)]
struct Configured {
    device: u32,
    config: SurfaceConfiguration,
}

struct Frame {
    device: u32,
    texture: Texture,
    view: TextureView,
    bytes: u64,
    acquired: Acquired,
}

/// A presentable surface: a window's swapchain, or a null surface for headless tests.
///
/// ```
/// use pragma_rhi::{
///     Color, CommandEncoder, DeviceDescriptor, Instance, InstanceDescriptor,
///     RenderPassColorAttachment, RenderPassDescriptor, RhiError, SurfaceConfiguration,
///     SurfaceError, TextureFormat, TextureUsages,
/// };
///
/// let instance = Instance::new(&InstanceDescriptor::null());
/// let mut surface = instance.create_null_surface(4, 2)?;
/// let adapter = instance.adapters().remove(0);
/// let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let mut config = SurfaceConfiguration {
///     format: TextureFormat::Rgba8Unorm,
///     usage: TextureUsages::RENDER_ATTACHMENT,
///     width: 4,
///     height: 2,
///     ..SurfaceConfiguration::default()
/// };
/// device.configure_surface(&mut surface, &config)?;
///
/// let frame = device.acquire_frame(&mut surface)?;
/// let mut encoder = CommandEncoder::new(Some("clear"));
/// encoder.begin_render_pass(&RenderPassDescriptor {
///     color_attachments: &[Some(RenderPassColorAttachment::clear(frame.view, Color::WHITE))],
///     ..RenderPassDescriptor::default()
/// });
/// device.submit([encoder.finish()])?;
/// device.present(&mut surface)?;
/// assert_eq!(surface.null_presented(), Some(&[255u8; 4 * 2 * 4][..]));
///
/// // The window is resized: acquiring fails until the surface is configured again.
/// surface.set_null_size(8, 8)?;
/// assert!(matches!(device.acquire_frame(&mut surface), Err(RhiError::Surface(SurfaceError::Outdated))));
/// (config.width, config.height) = (8, 8);
/// device.configure_surface(&mut surface, &config)?;
/// device.acquire_frame(&mut surface)?;
/// device.discard_frame(&mut surface)?;
/// # Ok::<(), RhiError>(())
/// ```
pub struct Surface {
    inner: SurfaceBackend,
    configured: Option<Configured>,
    frame: Option<Frame>,
    presents: u64,
}

impl fmt::Debug for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backend = match &self.inner {
            SurfaceBackend::Wgpu(_) => "wgpu",
            SurfaceBackend::Null(_) => "null",
        };
        f.debug_struct("Surface")
            .field("backend", &backend)
            .field("configuration", &self.configuration())
            .field("frame", &self.frame.as_ref().map(|frame| frame.texture))
            .field("presents", &self.presents)
            .finish()
    }
}

impl Surface {
    const fn new(inner: SurfaceBackend) -> Self {
        Self {
            inner,
            configured: None,
            frame: None,
            presents: 0,
        }
    }

    /// The configuration in effect, if any (with `Auto` modes resolved).
    #[must_use]
    pub fn configuration(&self) -> Option<SurfaceConfiguration> {
        self.configured.map(|configured| configured.config)
    }

    /// Whether a frame is acquired and not yet presented or discarded.
    #[must_use]
    pub const fn has_frame(&self) -> bool {
        self.frame.is_some()
    }

    /// How many frames have been presented.
    #[must_use]
    pub const fn presents(&self) -> u64 {
        self.presents
    }

    /// Whether this is a null surface.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self.inner, SurfaceBackend::Null(_))
    }

    fn null_mut(&mut self) -> Result<&mut NullSurface, RhiError> {
        match &mut self.inner {
            SurfaceBackend::Null(surface) => Ok(surface),
            SurfaceBackend::Wgpu(_) => Err(SurfaceError::Unsupported.into()),
        }
    }

    /// Resizes a null surface's "window": acquiring a frame of a configuration of another size
    /// fails with [`SurfaceError::Outdated`], as it does after a real window is resized.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] on a window surface.
    pub fn set_null_size(&mut self, width: u32, height: u32) -> Result<(), RhiError> {
        let surface = self.null_mut()?;
        (surface.width, surface.height) = (width, height);
        Ok(())
    }

    /// Loses a null surface on purpose, as a window system can: the next acquire fails with
    /// [`SurfaceError::Lost`] and drops the configuration, so the surface must be configured
    /// again.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] on a window surface.
    pub fn simulate_surface_loss(&mut self) -> Result<(), RhiError> {
        self.null_mut()?.lost = true;
        Ok(())
    }

    /// The texels of the frame a null surface presented last, row after row with no padding;
    /// `None` on a window surface or before the first present.
    #[must_use]
    pub fn null_presented(&self) -> Option<&[u8]> {
        match &self.inner {
            SurfaceBackend::Null(surface) => surface.presented.as_deref(),
            SurfaceBackend::Wgpu(_) => None,
        }
    }
}

const fn null_capabilities() -> (
    [TextureFormat; 4],
    [PresentMode; 4],
    [CompositeAlphaMode; 1],
    TextureUsages,
) {
    (
        [
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm,
        ],
        [
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
            PresentMode::Immediate,
            PresentMode::Mailbox,
        ],
        [CompositeAlphaMode::Opaque],
        TextureUsages::RENDER_ATTACHMENT
            .union(TextureUsages::COPY_SRC)
            .union(TextureUsages::COPY_DST)
            .union(TextureUsages::TEXTURE_BINDING),
    )
}

fn null_surface_capabilities() -> SurfaceCapabilities {
    let (formats, present_modes, alpha_modes, usages) = null_capabilities();
    SurfaceCapabilities {
        formats: formats.to_vec(),
        present_modes: present_modes.to_vec(),
        alpha_modes: alpha_modes.to_vec(),
        usages,
    }
}

const fn invalid(reason: &'static str) -> RhiError {
    RhiError::Surface(SurfaceError::InvalidConfiguration(reason))
}

/// Checks a configuration against the capabilities and the limits and resolves its `Auto`
/// modes.
fn validate(
    config: &SurfaceConfiguration,
    caps: &SurfaceCapabilities,
    max_side: u32,
) -> Result<SurfaceConfiguration, RhiError> {
    if config.width == 0 || config.height == 0 {
        return Err(invalid("the width and the height must not be zero"));
    }
    for side in [config.width, config.height] {
        if side > max_side {
            return Err(RhiError::limit(
                "max_texture_dimension_2d",
                u64::from(side),
                u64::from(max_side),
            ));
        }
    }
    if !caps.formats.contains(&config.format) {
        return Err(invalid("the surface does not support the format"));
    }
    if config.usage.is_empty() || !caps.usages.contains(config.usage) {
        return Err(invalid(
            "the usage is empty or outside the surface's supported usages",
        ));
    }
    if !(1..=MAX_FRAMES_IN_FLIGHT).contains(&config.desired_maximum_frame_latency) {
        return Err(invalid(
            "desired_maximum_frame_latency must be between 1 and 8",
        ));
    }
    let supports = |mode: PresentMode| caps.present_modes.contains(&mode);
    let present_mode = match config.present_mode {
        PresentMode::AutoVsync => [PresentMode::FifoRelaxed, PresentMode::Fifo]
            .into_iter()
            .find(|mode| supports(*mode)),
        PresentMode::AutoNoVsync => [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ]
        .into_iter()
        .find(|mode| supports(*mode)),
        mode => supports(mode).then_some(mode),
    }
    .ok_or(invalid("the surface does not support the present mode"))?;
    let alpha_mode = match config.alpha_mode {
        CompositeAlphaMode::Auto => caps.alpha_modes.first().copied(),
        mode => caps.alpha_modes.contains(&mode).then_some(mode),
    }
    .ok_or(invalid("the surface does not support the alpha mode"))?;
    Ok(SurfaceConfiguration {
        present_mode,
        alpha_mode,
        ..*config
    })
}

impl Instance {
    /// Creates a surface for a window the engine owns (winit, SDL, a native window): anything
    /// with a window and a display handle. The surface keeps `window` alive.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] when the instance has no GPU backend;
    /// [`RhiError::Backend`] when the window system refuses.
    pub fn create_surface<W>(&self, window: W) -> Result<Surface, RhiError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = self.wgpu.as_ref().ok_or(SurfaceError::Unsupported)?;
        let surface = instance
            .create_surface(window)
            .map_err(|err| RhiError::backend(err.to_string()))?;
        Ok(Surface::new(SurfaceBackend::Wgpu(surface)))
    }

    /// Creates a surface of the null backend, with a "window" of `width` × `height` pixels:
    /// configure, acquire and present work as on a window, and the presented texels can be
    /// read back ([`Surface::null_presented`]).
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] when the instance has no null backend.
    pub fn create_null_surface(&self, width: u32, height: u32) -> Result<Surface, RhiError> {
        if self.null.is_none() {
            return Err(SurfaceError::Unsupported.into());
        }
        Ok(Surface::new(SurfaceBackend::Null(NullSurface {
            width,
            height,
            lost: false,
            presented: None,
        })))
    }
}

impl Adapter {
    /// What `surface` supports on this adapter; no formats when the adapter cannot present
    /// to it. Pick an adapter for a window with this.
    #[must_use]
    pub fn surface_capabilities(&self, surface: &Surface) -> SurfaceCapabilities {
        match (&self.inner, &surface.inner) {
            (AdapterInner::Wgpu(adapter), SurfaceBackend::Wgpu(surface)) => {
                gpu_surface_capabilities(surface, adapter)
            }
            (AdapterInner::Null(_), SurfaceBackend::Null(_)) => null_surface_capabilities(),
            _ => SurfaceCapabilities::default(),
        }
    }
}

impl Device {
    /// What `surface` supports on this device; see [`Adapter::surface_capabilities`].
    #[must_use]
    pub fn surface_capabilities(&self, surface: &Surface) -> SurfaceCapabilities {
        match (&self.backend, &surface.inner) {
            (DeviceBackend::Wgpu(device), SurfaceBackend::Wgpu(surface)) => {
                device.surface_capabilities(surface)
            }
            (DeviceBackend::Null(_), SurfaceBackend::Null(_)) => null_surface_capabilities(),
            _ => SurfaceCapabilities::default(),
        }
    }

    /// Configures (or reconfigures, after a resize) `surface` for this device. A frame acquired
    /// on another device is dropped unpresented.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::FrameAcquired`] while a frame of this device is acquired;
    /// [`SurfaceError::Unsupported`]; [`SurfaceError::InvalidConfiguration`] or
    /// [`RhiError::LimitExceeded`] for a configuration the surface or the device does not
    /// support; [`RhiError::DeviceLost`]; [`RhiError::Backend`].
    pub fn configure_surface(
        &mut self,
        surface: &mut Surface,
        config: &SurfaceConfiguration,
    ) -> Result<(), RhiError> {
        self.alive()?;
        if let Some(frame) = &surface.frame {
            if frame.device == self.id {
                return Err(SurfaceError::FrameAcquired.into());
            }
            surface.frame = None;
        }
        let caps = self.surface_capabilities(surface);
        if !caps.is_supported() {
            return Err(SurfaceError::Unsupported.into());
        }
        let config = validate(config, &caps, self.limits.max_texture_dimension_2d)?;
        match (&self.backend, &mut surface.inner) {
            (DeviceBackend::Wgpu(device), SurfaceBackend::Wgpu(inner)) => {
                device.configure_surface(inner, &config)?;
            }
            (DeviceBackend::Null(_), SurfaceBackend::Null(inner)) => inner.lost = false,
            _ => return Err(SurfaceError::Unsupported.into()),
        }
        surface.configured = Some(Configured {
            device: self.id,
            config,
        });
        Ok(())
    }

    /// Acquires the surface's next frame. Its texture is a [`Texture`] of this device with the
    /// configured format and usages, for the frame's commands; present it with
    /// [`Self::present`] (or drop it with [`Self::discard_frame`]).
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Outdated`] after the window changed — configure again and retry;
    /// [`SurfaceError::Lost`]; [`SurfaceError::Timeout`] / [`SurfaceError::Occluded`] — skip
    /// the frame; [`SurfaceError::NotConfigured`], [`SurfaceError::OtherDevice`],
    /// [`SurfaceError::FrameAcquired`]; [`RhiError::OutOfMemory`]; [`RhiError::DeviceLost`];
    /// [`RhiError::Backend`].
    pub fn acquire_frame(&mut self, surface: &mut Surface) -> Result<SurfaceFrame, RhiError> {
        self.alive()?;
        let configured = surface.configured.ok_or(SurfaceError::NotConfigured)?;
        if configured.device != self.id {
            return Err(SurfaceError::OtherDevice.into());
        }
        if surface.frame.is_some() {
            return Err(SurfaceError::FrameAcquired.into());
        }
        let config = configured.config;
        let info = TextureInfo {
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: config.format,
            usage: config.usage,
        };
        let bytes = u64::from(config.width)
            .saturating_mul(u64::from(config.height))
            .saturating_mul(u64::from(config.format.block_size()));
        self.textures.reserve()?;
        self.views.reserve()?;
        let (obj, acquired, suboptimal) = match (&mut self.backend, &mut surface.inner) {
            (DeviceBackend::Wgpu(device), SurfaceBackend::Wgpu(inner)) => {
                let (texture, suboptimal) = device.acquire(inner)?;
                (
                    Obj::Wgpu(texture.texture.clone()),
                    Acquired::Wgpu(texture),
                    suboptimal,
                )
            }
            (DeviceBackend::Null(device), SurfaceBackend::Null(inner)) => {
                if inner.lost {
                    surface.configured = None;
                    return Err(SurfaceError::Lost.into());
                }
                if (inner.width, inner.height) != (config.width, config.height) {
                    return Err(SurfaceError::Outdated.into());
                }
                let id = device.allocate(bytes)?;
                (Obj::Null(id), Acquired::Null(id), false)
            }
            _ => return Err(SurfaceError::Unsupported.into()),
        };
        let texture = self.textures.insert(TextureEntry {
            info,
            bytes,
            views: 0,
            surface: true,
            obj,
        })?;
        let view = match self.create_texture_view(
            texture,
            &TextureViewDescriptor {
                label: Some("pragma_rhi surface"),
                ..TextureViewDescriptor::default()
            },
        ) {
            Ok(view) => view,
            Err(err) => {
                if let Ok(entry) = self.textures.remove(texture) {
                    self.release_swapchain_texture(entry.obj, bytes, acquired);
                }
                return Err(err);
            }
        };
        surface.frame = Some(Frame {
            device: self.id,
            texture,
            view,
            bytes,
            acquired,
        });
        Ok(SurfaceFrame {
            texture,
            view,
            suboptimal,
        })
    }

    /// Presents the acquired frame: its texture and view go stale. Submit the frame's
    /// commands first; they are not waited for.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::NoFrame`], [`SurfaceError::OtherDevice`];
    /// [`RhiError::ResourceInUse`] while the frame's texture has views other than
    /// [`SurfaceFrame::view`] or that view is in a bind group; [`RhiError::DeviceLost`];
    /// [`RhiError::Backend`].
    pub fn present(&mut self, surface: &mut Surface) -> Result<(), RhiError> {
        self.end_surface_frame(surface, true)
    }

    /// Releases the acquired frame without presenting it (an error path, a skipped frame).
    ///
    /// # Errors
    ///
    /// As [`Self::present`].
    pub fn discard_frame(&mut self, surface: &mut Surface) -> Result<(), RhiError> {
        self.end_surface_frame(surface, false)
    }

    fn end_surface_frame(&mut self, surface: &mut Surface, present: bool) -> Result<(), RhiError> {
        self.alive()?;
        let frame = surface.frame.as_ref().ok_or(SurfaceError::NoFrame)?;
        if frame.device != self.id {
            return Err(SurfaceError::OtherDevice.into());
        }
        let entry = self.textures.get(frame.texture)?;
        if entry.views > 1 {
            return Err(RhiError::ResourceInUse {
                resource: ResourceKind::Texture,
                dependents: entry.views.saturating_sub(1),
            });
        }
        self.destroy_texture_view(frame.view)?;
        let Some(frame) = surface.frame.take() else {
            return Err(SurfaceError::NoFrame.into());
        };
        let entry = self.textures.remove(frame.texture)?;
        if present {
            if let (Acquired::Null(id), DeviceBackend::Null(device), SurfaceBackend::Null(inner)) =
                (&frame.acquired, &self.backend, &mut surface.inner)
            {
                inner.presented = Some(device.read(*id, 0, frame.bytes)?);
            }
            surface.presents = surface.presents.saturating_add(1);
        }
        match frame.acquired {
            Acquired::Wgpu(texture) => {
                if present && let DeviceBackend::Wgpu(device) = &self.backend {
                    device.present(texture)?;
                }
                // Dropped unpresented, wgpu discards it.
            }
            Acquired::Null(_) => {}
        }
        self.bury(Garbage::SurfaceTexture(entry.obj, frame.bytes))
    }

    /// Gives a swapchain texture that never made it into a frame back.
    fn release_swapchain_texture(
        &mut self,
        obj: crate::backend::TextureObj,
        bytes: u64,
        acquired: Acquired,
    ) {
        drop(acquired);
        self.backend.release_surface_texture(obj, bytes);
    }
}
