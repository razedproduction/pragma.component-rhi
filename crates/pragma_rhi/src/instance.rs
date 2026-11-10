//! The instance, adapter enumeration and device creation.

use bitflags::bitflags;

use crate::backend::{DeviceBackend, GpuDevice, NullDevice, map_backend};
use crate::{Device, DeviceDescriptor, Limits, RhiError};

bitflags! {
    /// The backends an [`Instance`] enumerates adapters of.
    ///
    /// ```
    /// use pragma_rhi::{Backends, Instance, InstanceDescriptor};
    ///
    /// // The GPU backends only, without the null adapter.
    /// let gpus = Instance::new(&InstanceDescriptor {
    ///     backends: Backends::VULKAN | Backends::DX12,
    ///     ..InstanceDescriptor::default()
    /// });
    /// # let _ = gpus;
    /// assert!(Backends::all().contains(Backends::NULL));
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Backends: u8 {
        /// Vulkan (Windows, Linux).
        const VULKAN = 1 << 0;
        /// Direct3D 12 (Windows).
        const DX12 = 1 << 1;
        /// The headless null backend (everywhere).
        const NULL = 1 << 2;
    }
}

impl Default for Backends {
    fn default() -> Self {
        Self::all()
    }
}

/// The backend of an adapter or a device.
///
/// ```
/// use pragma_rhi::{Backend, Instance, InstanceDescriptor};
///
/// let instance = Instance::new(&InstanceDescriptor::default());
/// // Prefer a GPU and fall back to the null adapter, which always exists when enabled.
/// let adapter = instance
///     .adapters()
///     .into_iter()
///     .min_by_key(|adapter| adapter.info().backend == Backend::Null);
/// assert!(adapter.is_some());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Vulkan, through wgpu.
    Vulkan,
    /// Direct3D 12, through wgpu.
    Dx12,
    /// The headless null backend.
    Null,
}

/// The kind of an adapter's device.
///
/// ```
/// use pragma_rhi::{DeviceType, Instance, InstanceDescriptor};
///
/// // Rank discrete GPUs first, software adapters last.
/// let rank = |ty: DeviceType| match ty {
///     DeviceType::DiscreteGpu => 0,
///     DeviceType::IntegratedGpu => 1,
///     DeviceType::Cpu => 3,
///     _ => 2,
/// };
/// let mut adapters = Instance::new(&InstanceDescriptor::default()).adapters();
/// adapters.sort_by_key(|adapter| rank(adapter.info().device_type));
/// # let _ = adapters;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceType {
    /// Unknown.
    Other,
    /// A GPU sharing memory with the CPU.
    IntegratedGpu,
    /// A GPU with its own memory.
    DiscreteGpu,
    /// A GPU of a virtual machine.
    VirtualGpu,
    /// A software rasterizer (WARP, lavapipe) or the null backend.
    Cpu,
}

/// What an adapter is.
///
/// ```
/// use pragma_rhi::{Backend, DeviceType, Instance, InstanceDescriptor};
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// let info = adapter.info();
/// assert_eq!((info.backend, info.device_type), (Backend::Null, DeviceType::Cpu));
/// println!("{} ({:04x}:{:04x})", info.name, info.vendor, info.device);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct AdapterInfo {
    /// The adapter's name, as the driver reports it.
    pub name: String,
    /// The PCI vendor id (0 when unknown).
    pub vendor: u32,
    /// The PCI device id (0 when unknown).
    pub device: u32,
    /// The kind of device.
    pub device_type: DeviceType,
    /// The backend.
    pub backend: Backend,
}

/// The configuration of the null adapter.
///
/// ```
/// use pragma_rhi::{BufferDescriptor, BufferUsages, DeviceDescriptor, Instance, InstanceDescriptor, NullConfig, RhiError};
///
/// // A null adapter with 1 KiB of device memory, to test out-of-memory handling.
/// let instance = Instance::new(&InstanceDescriptor {
///     null: NullConfig { memory_budget: 1024, ..NullConfig::default() },
///     ..InstanceDescriptor::null()
/// });
/// let mut device = instance.adapters().remove(0).request_device(&DeviceDescriptor::default())?;
/// let big = BufferDescriptor { label: None, size: 2048, usage: BufferUsages::STORAGE };
/// assert!(matches!(device.create_buffer(&big), Err(RhiError::OutOfMemory { .. })));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NullConfig {
    /// The limits the null adapter reports.
    pub limits: Limits,
    /// The device memory, in bytes, a null device may allocate before it reports
    /// [`RhiError::OutOfMemory`].
    pub memory_budget: u64,
    /// How many submissions a null device keeps "in flight" when polled without waiting:
    /// commands run at once, but a submission is reported complete only `latency` submissions
    /// later (or when waited for), so frame pacing and deferred destruction can be tested.
    /// 0 (the default) completes every submission at once.
    pub latency: u32,
}

impl Default for NullConfig {
    fn default() -> Self {
        Self {
            limits: Limits::default(),
            memory_budget: 1 << 30,
            latency: 0,
        }
    }
}

/// Describes an [`Instance`].
///
/// ```
/// use pragma_rhi::{Backend, Instance, InstanceDescriptor};
///
/// let adapters = Instance::new(&InstanceDescriptor::null()).adapters();
/// assert!(adapters.iter().all(|adapter| adapter.info().backend == Backend::Null));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InstanceDescriptor {
    /// The backends to enumerate; all by default.
    pub backends: Backends,
    /// The null adapter's configuration, if [`Backends::NULL`] is enabled.
    pub null: NullConfig,
}

impl InstanceDescriptor {
    /// Only the null backend: no GPU is touched (tests, CI, servers).
    #[must_use]
    pub fn null() -> Self {
        Self {
            backends: Backends::NULL,
            null: NullConfig::default(),
        }
    }
}

/// The entry point: enumerates adapters.
///
/// ```
/// use pragma_rhi::{Instance, InstanceDescriptor};
///
/// // Every backend: the GPU adapters found on this machine, then the null adapter.
/// let instance = Instance::new(&InstanceDescriptor::default());
/// for adapter in instance.adapters() {
///     println!("{:?}: {}", adapter.info().backend, adapter.info().name);
/// }
/// ```
pub struct Instance {
    pub(crate) wgpu: Option<wgpu::Instance>,
    pub(crate) null: Option<NullConfig>,
}

impl core::fmt::Debug for Instance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Instance")
            .field("gpu", &self.wgpu.is_some())
            .field("null", &self.null)
            .finish()
    }
}

impl Instance {
    /// Creates an instance. Loading a GPU API never fails here: a missing driver only means
    /// that [`Self::adapters`] lists no adapter of that backend.
    #[must_use]
    pub fn new(desc: &InstanceDescriptor) -> Self {
        let mut gpu = wgpu::Backends::empty();
        if desc.backends.contains(Backends::VULKAN) {
            gpu |= wgpu::Backends::VULKAN;
        }
        if desc.backends.contains(Backends::DX12) {
            gpu |= wgpu::Backends::DX12;
        }
        let wgpu = (!gpu.is_empty()).then(|| {
            let mut instance = wgpu::InstanceDescriptor::new_without_display_handle();
            instance.backends = gpu;
            wgpu::Instance::new(instance)
        });
        Self {
            wgpu,
            null: desc.backends.contains(Backends::NULL).then_some(desc.null),
        }
    }

    /// Every adapter of the enabled backends: GPU adapters in the driver's order, then the null
    /// adapter.
    ///
    /// ```
    /// use pragma_rhi::{Backend, Instance, InstanceDescriptor};
    ///
    /// let adapters = Instance::new(&InstanceDescriptor::default()).adapters();
    /// // The null adapter comes last, after every GPU adapter.
    /// assert_eq!(adapters.last().map(|adapter| adapter.info().backend), Some(Backend::Null));
    /// ```
    #[must_use]
    pub fn adapters(&self) -> Vec<Adapter> {
        let mut adapters: Vec<Adapter> = match &self.wgpu {
            Some(instance) => {
                pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                    .into_iter()
                    .filter_map(Adapter::from_wgpu)
                    .collect()
            }
            None => Vec::new(),
        };
        if let Some(config) = self.null {
            adapters.push(Adapter::null(config));
        }
        adapters
    }
}

#[derive(Debug, Clone)]
pub(crate) enum AdapterInner {
    Wgpu(wgpu::Adapter),
    Null(NullConfig),
}

/// A physical (or software, or null) device that devices are created on.
///
/// ```
/// use pragma_rhi::{DeviceDescriptor, Instance, InstanceDescriptor, RhiError};
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// // Ask for what the adapter supports, never more.
/// let mut limits = *adapter.limits();
/// limits.max_buffer_size = limits.max_buffer_size.min(1 << 28);
/// let device = adapter.request_device(&DeviceDescriptor { required_limits: limits, ..DeviceDescriptor::default() })?;
/// assert_eq!(device.limits().max_buffer_size, limits.max_buffer_size);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Adapter {
    info: AdapterInfo,
    limits: Limits,
    pub(crate) inner: AdapterInner,
}

impl Adapter {
    fn from_wgpu(adapter: wgpu::Adapter) -> Option<Self> {
        let info = adapter.get_info();
        let backend = map_backend(info.backend)?;
        let device_type = match info.device_type {
            wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
            wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
            wgpu::DeviceType::Cpu => DeviceType::Cpu,
            wgpu::DeviceType::Other => DeviceType::Other,
        };
        Some(Self {
            limits: Limits::from_wgpu(&adapter.limits()),
            info: AdapterInfo {
                name: info.name,
                vendor: info.vendor,
                device: info.device,
                device_type,
                backend,
            },
            inner: AdapterInner::Wgpu(adapter),
        })
    }

    fn null(config: NullConfig) -> Self {
        Self {
            info: AdapterInfo {
                name: "pragma null adapter".to_owned(),
                vendor: 0,
                device: 0,
                device_type: DeviceType::Cpu,
                backend: Backend::Null,
            },
            limits: config.limits,
            inner: AdapterInner::Null(config),
        }
    }

    /// What the adapter is.
    #[must_use]
    pub const fn info(&self) -> &AdapterInfo {
        &self.info
    }

    /// The best limits a device of this adapter can request.
    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Creates a device.
    ///
    /// # Errors
    ///
    /// [`RhiError::LimitExceeded`] when `desc.required_limits` exceed [`Self::limits`];
    /// [`RhiError::Backend`] when the driver refuses to create the device.
    ///
    /// ```
    /// use pragma_rhi::{DeviceDescriptor, Instance, InstanceDescriptor, RhiError};
    ///
    /// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
    /// let mut one = adapter.request_device(&DeviceDescriptor { label: Some("one"), ..DeviceDescriptor::default() })?;
    /// let two = adapter.request_device(&DeviceDescriptor { label: Some("two"), ..DeviceDescriptor::default() })?;
    /// // Devices are independent: a lost one leaves the other working.
    /// one.simulate_device_loss();
    /// assert!(one.is_lost() && !two.is_lost());
    /// # Ok::<(), RhiError>(())
    /// ```
    pub fn request_device(&self, desc: &DeviceDescriptor<'_>) -> Result<Device, RhiError> {
        desc.required_limits.check_within(&self.limits)?;
        let backend = match &self.inner {
            AdapterInner::Wgpu(adapter) => DeviceBackend::Wgpu(GpuDevice::request(
                adapter,
                self.info.backend,
                desc.label,
                &desc.required_limits,
            )?),
            AdapterInner::Null(config) => {
                DeviceBackend::Null(NullDevice::new(config.memory_budget, config.latency))
            }
        };
        Device::new(backend, desc)
    }
}
