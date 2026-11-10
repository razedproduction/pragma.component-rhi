//! Surfaces and the swapchain (roadmap M4) on the null backend: configuration rules, the
//! acquire / present protocol, resize and loss handling, and the presented texels. Window
//! surfaces need a window, which this crate does not open; they share every front-end rule
//! tested here.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use pragma_rhi::{
    Backends, BufferDescriptor, BufferUsages, Color, CommandEncoder, CompositeAlphaMode, Device,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, PresentMode,
    RenderPassColorAttachment, RenderPassDescriptor, ResourceKind, RhiError, Surface,
    SurfaceConfiguration, SurfaceError, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureFormat, TextureUsages, TextureViewDescriptor,
};

fn setup(width: u32, height: u32) -> (Instance, Surface, Device) {
    let instance = Instance::new(&InstanceDescriptor::null());
    let surface = instance
        .create_null_surface(width, height)
        .expect("a null surface");
    let adapter = instance.adapters().remove(0);
    assert!(adapter.surface_capabilities(&surface).is_supported());
    let device = adapter
        .request_device(&DeviceDescriptor::default())
        .expect("a null device");
    (instance, surface, device)
}

fn config(width: u32, height: u32) -> SurfaceConfiguration {
    SurfaceConfiguration {
        format: TextureFormat::Bgra8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        width,
        height,
        ..SurfaceConfiguration::default()
    }
}

fn surface_error(result: Result<impl core::fmt::Debug, RhiError>) -> SurfaceError {
    match result {
        Err(RhiError::Surface(error)) => error,
        other => panic!("expected a surface error, got {other:?}"),
    }
}

fn clear(device: &mut Device, view: pragma_rhi::TextureView, color: Color) {
    let mut encoder = CommandEncoder::new(Some("clear"));
    encoder
        .begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment::clear(view, color))],
            ..RenderPassDescriptor::default()
        })
        .end();
    device.submit([encoder.finish()]).expect("a submission");
}

#[test]
fn capabilities_depend_on_the_adapter() {
    let instance = Instance::new(&InstanceDescriptor::null());
    let surface = instance.create_null_surface(4, 4).expect("a surface");
    let caps = instance.adapters()[0].surface_capabilities(&surface);
    assert!(caps.formats.contains(&TextureFormat::Bgra8UnormSrgb));
    assert!(caps.present_modes.contains(&PresentMode::Fifo));
    assert!(caps.usages.contains(TextureUsages::RENDER_ATTACHMENT));

    // GPU adapters cannot present to a null surface, and a GPU-only instance cannot make one.
    let gpus = Instance::new(&InstanceDescriptor {
        backends: Backends::VULKAN | Backends::DX12,
        ..InstanceDescriptor::default()
    });
    for adapter in gpus.adapters() {
        assert!(!adapter.surface_capabilities(&surface).is_supported());
    }
    assert_eq!(
        surface_error(gpus.create_null_surface(4, 4)),
        SurfaceError::Unsupported
    );
}

#[test]
fn configurations_are_validated_against_the_capabilities() {
    let (_instance, mut surface, mut device) = setup(8, 8);
    let invalid = |device: &mut Device, surface: &mut Surface, config: SurfaceConfiguration| {
        device.configure_surface(surface, &config)
    };
    assert!(matches!(
        surface_error(invalid(&mut device, &mut surface, config(0, 8))),
        SurfaceError::InvalidConfiguration(_)
    ));
    let unsupported_format = SurfaceConfiguration {
        format: TextureFormat::Rgba16Float,
        ..config(8, 8)
    };
    assert!(matches!(
        surface_error(invalid(&mut device, &mut surface, unsupported_format)),
        SurfaceError::InvalidConfiguration(_)
    ));
    let storage = SurfaceConfiguration {
        usage: TextureUsages::STORAGE_BINDING,
        ..config(8, 8)
    };
    assert!(matches!(
        surface_error(invalid(&mut device, &mut surface, storage)),
        SurfaceError::InvalidConfiguration(_)
    ));
    let huge = config(device.limits().max_texture_dimension_2d + 1, 8);
    assert!(matches!(
        invalid(&mut device, &mut surface, huge),
        Err(RhiError::LimitExceeded { .. })
    ));
    assert_eq!(surface.configuration(), None);

    // `Auto` modes resolve to what the surface supports.
    let auto = SurfaceConfiguration {
        present_mode: PresentMode::AutoNoVsync,
        alpha_mode: CompositeAlphaMode::Auto,
        ..config(8, 8)
    };
    device
        .configure_surface(&mut surface, &auto)
        .expect("configured");
    let resolved = surface.configuration().expect("a configuration");
    assert_eq!(resolved.present_mode, PresentMode::Immediate);
    assert_eq!(resolved.alpha_mode, CompositeAlphaMode::Opaque);
}

#[test]
fn frames_are_presented_with_the_rendered_texels() {
    let (_instance, mut surface, mut device) = setup(3, 2);
    device
        .configure_surface(&mut surface, &config(3, 2))
        .expect("configured");
    let base = device.allocated_bytes().expect("accounted");
    for (index, color) in [
        Color::new(1.0, 0.0, 0.0, 1.0),
        Color::new(0.0, 0.0, 1.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let frame = device.acquire_frame(&mut surface).expect("a frame");
        assert!(!frame.suboptimal);
        let info = device.texture_info(frame.texture).expect("a live texture");
        assert_eq!(info.format, TextureFormat::Bgra8Unorm);
        assert_eq!((info.size.width, info.size.height), (3, 2));
        clear(&mut device, frame.view, color);
        device.present(&mut surface).expect("presented");
        assert!(matches!(
            device.texture_info(frame.texture),
            Err(RhiError::StaleHandle { .. })
        ));
        assert_eq!(surface.presents(), index as u64 + 1);
    }
    // BGRA order: blue, green, red, alpha.
    assert_eq!(
        surface.null_presented(),
        Some(&[255, 0, 0, 0].repeat(6)[..])
    );
    device.poll().expect("a poll");
    assert_eq!(
        device.allocated_bytes(),
        Some(base),
        "swapchain memory is released"
    );
}

#[test]
fn a_frame_texture_is_an_ordinary_copy_source() {
    let (_instance, mut surface, mut device) = setup(2, 2);
    device
        .configure_surface(&mut surface, &config(2, 2))
        .expect("configured");
    let frame = device.acquire_frame(&mut surface).expect("a frame");
    clear(&mut device, frame.view, Color::WHITE);
    let screenshot = device
        .create_buffer(&BufferDescriptor {
            label: Some("screenshot"),
            size: 512,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        })
        .expect("a buffer");
    let mut encoder = CommandEncoder::new(Some("screenshot"));
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo::new(frame.texture),
        TexelCopyBufferInfo {
            buffer: screenshot,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    device.submit([encoder.finish()]).expect("a submission");
    device.present(&mut surface).expect("presented");
    let bytes = device.read_buffer(screenshot, 0, 8).expect("read");
    assert_eq!(bytes, [255; 8]);
}

#[test]
fn the_acquire_present_protocol_is_checked() {
    let (instance, mut surface, mut device) = setup(4, 4);
    assert_eq!(
        surface_error(device.acquire_frame(&mut surface)),
        SurfaceError::NotConfigured
    );
    assert_eq!(
        surface_error(device.present(&mut surface)),
        SurfaceError::NoFrame
    );
    device
        .configure_surface(&mut surface, &config(4, 4))
        .expect("configured");
    let frame = device.acquire_frame(&mut surface).expect("a frame");
    assert!(surface.has_frame());
    assert_eq!(
        surface_error(device.acquire_frame(&mut surface)),
        SurfaceError::FrameAcquired
    );
    assert_eq!(
        surface_error(device.configure_surface(&mut surface, &config(4, 4))),
        SurfaceError::FrameAcquired
    );
    assert_eq!(
        surface_error(device.destroy_texture(frame.texture)),
        SurfaceError::SurfaceTexture
    );
    // An extra view must go before the frame does.
    let extra = device
        .create_texture_view(frame.texture, &TextureViewDescriptor::default())
        .expect("a view");
    assert!(matches!(
        device.present(&mut surface),
        Err(RhiError::ResourceInUse {
            resource: ResourceKind::Texture,
            dependents: 1,
            ..
        })
    ));
    device.destroy_texture_view(extra).expect("destroyed");
    device.discard_frame(&mut surface).expect("discarded");
    assert_eq!(surface.presents(), 0, "a discarded frame is not presented");

    // Another device must configure the surface for itself first.
    let mut other = instance.adapters()[0]
        .request_device(&DeviceDescriptor::default())
        .expect("a second device");
    assert_eq!(
        surface_error(other.acquire_frame(&mut surface)),
        SurfaceError::OtherDevice
    );
    device.acquire_frame(&mut surface).expect("a frame");
    // Configuring on another device drops the first device's frame unpresented.
    other
        .configure_surface(&mut surface, &config(4, 4))
        .expect("configured");
    other.acquire_frame(&mut surface).expect("a frame");
    other.present(&mut surface).expect("presented");
}

#[test]
fn resizes_and_losses_are_reported_and_recovered_from() {
    let (_instance, mut surface, mut device) = setup(4, 4);
    device
        .configure_surface(&mut surface, &config(4, 4))
        .expect("configured");
    surface.set_null_size(6, 3).expect("resized");
    assert_eq!(
        surface_error(device.acquire_frame(&mut surface)),
        SurfaceError::Outdated
    );
    device
        .configure_surface(&mut surface, &config(6, 3))
        .expect("reconfigured");
    let frame = device.acquire_frame(&mut surface).expect("a frame");
    assert_eq!(
        device.texture_info(frame.texture).expect("live").size.width,
        6
    );
    device.present(&mut surface).expect("presented");

    surface.simulate_surface_loss().expect("lost");
    assert_eq!(
        surface_error(device.acquire_frame(&mut surface)),
        SurfaceError::Lost
    );
    assert_eq!(
        surface_error(device.acquire_frame(&mut surface)),
        SurfaceError::NotConfigured,
        "a lost surface needs a new configuration"
    );
    device
        .configure_surface(&mut surface, &config(6, 3))
        .expect("reconfigured");
    device
        .acquire_frame(&mut surface)
        .expect("a frame after recovery");
    device.discard_frame(&mut surface).expect("discarded");
}

#[test]
fn frames_and_surfaces_together() {
    let (_instance, mut surface, mut device) = setup(4, 4);
    device
        .configure_surface(&mut surface, &config(4, 4))
        .expect("configured");
    for index in 0..6_u64 {
        let info = device.begin_frame().expect("a frame");
        assert_eq!(info.index, index);
        let frame = device
            .acquire_frame(&mut surface)
            .expect("a swapchain image");
        clear(&mut device, frame.view, Color::BLACK);
        device.present(&mut surface).expect("presented");
        device.end_frame().expect("the frame ends");
    }
    assert_eq!(surface.presents(), 6);
    assert_eq!(device.live(ResourceKind::Texture), 0);
    assert_eq!(device.live(ResourceKind::TextureView), 0);
}
