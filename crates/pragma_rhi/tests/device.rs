//! Device behavior on the null backend, and a smoke test on whatever GPU adapters exist.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use pragma_rhi::{
    Backend, Backends, BufferDescriptor, BufferUsages, Device, DeviceDescriptor, Extent3d,
    Instance, InstanceDescriptor, Limits, NullConfig, ResourceKind, RhiError, SamplerDescriptor,
    TextureDescriptor, TextureFormat, TextureUsages, TextureViewDescriptor,
};

fn null_device() -> Device {
    Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}

fn buffer(size: u64) -> BufferDescriptor<'static> {
    BufferDescriptor {
        label: Some("test"),
        size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    }
}

fn texture(width: u32, height: u32) -> TextureDescriptor<'static> {
    TextureDescriptor {
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        ..TextureDescriptor::default()
    }
}

#[test]
fn null_instance_lists_exactly_the_null_adapter() {
    let adapters = Instance::new(&InstanceDescriptor::null()).adapters();
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0].info().backend, Backend::Null);
    assert_eq!(*adapters[0].limits(), Limits::default());
}

#[test]
fn required_limits_are_checked_against_the_adapter() {
    let adapter = Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0);
    let desc = DeviceDescriptor {
        label: None,
        required_limits: limits_with_max_buffer_size(u64::MAX),
        ..DeviceDescriptor::default()
    };
    assert!(matches!(
        adapter.request_device(&desc),
        Err(RhiError::LimitExceeded {
            limit: "max_buffer_size",
            ..
        })
    ));
}

#[test]
fn device_limits_not_the_adapters_bound_descriptors() {
    let adapter = Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0);
    let mut device = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_limits: limits_with_max_buffer_size(1024),
            ..DeviceDescriptor::default()
        })
        .expect("a null device");
    assert!(device.create_buffer(&buffer(1024)).is_ok());
    assert!(matches!(
        device.create_buffer(&buffer(1025)),
        Err(RhiError::LimitExceeded { .. })
    ));
}

#[test]
fn handles_do_not_cross_devices() {
    let mut one = null_device();
    let mut two = null_device();
    let a = one.create_buffer(&buffer(16)).expect("buffer");
    assert!(matches!(
        two.destroy_buffer(a),
        Err(RhiError::ForeignHandle {
            resource: ResourceKind::Buffer,
            ..
        })
    ));
    assert!(one.buffer_info(a).is_ok());
}

#[test]
fn null_budget_reports_out_of_memory_and_recovers() {
    let desc = InstanceDescriptor {
        null: NullConfig {
            memory_budget: 64 * 64 * 4,
            ..NullConfig::default()
        },
        ..InstanceDescriptor::null()
    };
    let mut device = Instance::new(&desc)
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device");
    let full = device
        .create_texture(&texture(64, 64))
        .expect("fits exactly");
    assert!(matches!(
        device.create_buffer(&buffer(4)),
        Err(RhiError::OutOfMemory { bytes: 4, .. })
    ));
    assert_eq!(
        device.live(ResourceKind::Buffer),
        0,
        "a failed create leaves nothing behind"
    );
    device.destroy_texture(full).expect("destroy");
    assert_eq!(device.allocated_bytes(), Some(0));
    assert!(device.create_buffer(&buffer(4)).is_ok());
}

#[test]
fn views_pin_their_texture() {
    let mut device = null_device();
    let tex = device.create_texture(&texture(32, 32)).expect("texture");
    let a = device
        .create_texture_view(tex, &TextureViewDescriptor::default())
        .expect("view");
    let b = device
        .create_texture_view(tex, &TextureViewDescriptor::default())
        .expect("view");
    assert!(matches!(
        device.destroy_texture(tex),
        Err(RhiError::ResourceInUse {
            resource: ResourceKind::Texture,
            dependents: 2,
            ..
        })
    ));
    device.destroy_texture_view(a).expect("destroy view");
    device.destroy_texture_view(b).expect("destroy view");
    device.destroy_texture(tex).expect("destroy texture");
    assert!(matches!(
        device.create_texture_view(tex, &TextureViewDescriptor::default()),
        Err(RhiError::StaleHandle {
            resource: ResourceKind::Texture,
            ..
        })
    ));
}

#[test]
fn device_loss_is_an_error_everywhere() {
    let mut device = null_device();
    let buf = device.create_buffer(&buffer(16)).expect("buffer");
    let sampler = device
        .create_sampler(&SamplerDescriptor::default())
        .expect("sampler");
    device.simulate_device_loss();
    assert!(device.is_lost());
    assert_eq!(device.buffer_info(buf), Err(RhiError::DeviceLost));
    assert_eq!(device.destroy_buffer(buf), Err(RhiError::DeviceLost));
    assert_eq!(
        device.sampler_info(sampler).err(),
        Some(RhiError::DeviceLost)
    );
    assert_eq!(device.create_buffer(&buffer(16)), Err(RhiError::DeviceLost));
    assert_eq!(
        device.create_texture(&texture(4, 4)),
        Err(RhiError::DeviceLost)
    );
}

/// Runs on every Vulkan / D3D12 adapter of the machine (none on CI runners, where this passes
/// trivially): the same resources as on the null backend, and wgpu's own validation must agree
/// with ours, so nothing valid here is rejected by the backend.
#[test]
fn gpu_adapters_create_and_destroy_resources() {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::VULKAN | Backends::DX12,
        ..InstanceDescriptor::default()
    });
    for adapter in instance.adapters() {
        let name = format!("{:?} {}", adapter.info().backend, adapter.info().name);
        let mut device = match adapter.request_device(&DeviceDescriptor::default()) {
            Ok(device) => device,
            // A driver may refuse WebGPU's default limits; that is reported, not a panic.
            Err(RhiError::Backend { message, .. }) => {
                eprintln!("skipping {name}: {message}");
                continue;
            }
            Err(other) => panic!("{name}: {other}"),
        };
        assert_eq!(device.backend(), adapter.info().backend);
        let buf = device.create_buffer(&buffer(256)).expect(&name);
        let tex = device
            .create_texture(&TextureDescriptor {
                mip_level_count: 7,
                ..texture(64, 64)
            })
            .expect(&name);
        let view = device
            .create_texture_view(tex, &TextureViewDescriptor::default())
            .expect(&name);
        let sampler = device
            .create_sampler(&SamplerDescriptor::default())
            .expect(&name);
        device.destroy_texture_view(view).expect(&name);
        device.destroy_texture(tex).expect(&name);
        device.destroy_buffer(buf).expect(&name);
        device.destroy_sampler(sampler).expect(&name);
        device.simulate_device_loss();
        assert_eq!(device.create_buffer(&buffer(4)), Err(RhiError::DeviceLost));
        eprintln!("checked {name}");
    }
}

/// `Limits` is `#[non_exhaustive]`: outside the crate it is built from the defaults and changed.
fn limits_with_max_buffer_size(bytes: u64) -> Limits {
    let mut limits = Limits::default();
    limits.max_buffer_size = bytes;
    limits
}
