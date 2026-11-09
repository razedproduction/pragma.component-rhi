//! The instance, adapter enumeration and device creation.

use bitflags::bitflags;

use crate::backend::{DeviceBackend, GpuDevice, NullDevice, map_backend};
use crate::{Device, DeviceDescriptor, Limits, RhiError};

bitflags! {
    /// The backends an [`Instance`] enumerates adapters of.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NullConfig {
    /// The limits the null adapter reports.
    pub limits: Limits,
    /// The device memory, in bytes, a null device may allocate before it reports
    /// [`RhiError::OutOfMemory`].
    pub memory_budget: u64,
}

impl Default for NullConfig {
    fn default() -> Self {
        Self {
            limits: Limits::default(),
            memory_budget: 1 << 30,
        }
    }
}

/// Describes an [`Instance`].
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
pub struct Instance {
    wgpu: Option<wgpu::Instance>,
    null: Option<NullConfig>,
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
enum AdapterInner {
    Wgpu(wgpu::Adapter),
    Null(NullConfig),
}

/// A physical (or software, or null) device that devices are created on.
#[derive(Debug, Clone)]
pub struct Adapter {
    info: AdapterInfo,
    limits: Limits,
    inner: AdapterInner,
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
                DeviceBackend::Null(NullDevice::new(config.memory_budget))
            }
        };
        Device::new(backend, desc.required_limits)
    }
}
