//! Device-loss recovery on the null backend (roadmap M5, DECISIONS #22).
//!
//! A device can be lost at any point of a frame: with resources of every kind alive, pipelines
//! compiling on a blocking pool, bytes being produced for an upload, uploads in the staging belt,
//! readbacks and submissions in flight, deferred destructions waiting, a swapchain frame
//! acquired. Every entry point must then report [`RhiError::DeviceLost`] — never panic, never
//! hang — and the recovery an engine performs must work: drop the device, request a new one from
//! the same adapter, rebuild, reconfigure the surface, and get exactly the results it got before.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::mpsc;

use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_core::tasks::{AsyncConfig, Executor};
use pragma_rhi::{
    Adapter, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType, BufferDescriptor,
    BufferUsages, Color, CommandEncoder, ComputePassDescriptor, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, NullConfig, PipelineLayoutDescriptor,
    RenderPassColorAttachment, RenderPassDescriptor, ResourceKind, RhiError, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, SurfaceConfiguration, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor, TextureFormat,
    TextureUsages, TextureViewDescriptor, UploadTarget,
};

const SHADER: &str = r"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] * 2u + 1u;
    }
}
";

const LOST: Result<(), RhiError> = Err(RhiError::DeviceLost);

fn adapter(latency: u32) -> (Instance, Adapter) {
    let instance = Instance::new(&InstanceDescriptor {
        null: NullConfig {
            latency,
            ..NullConfig::default()
        },
        ..InstanceDescriptor::null()
    });
    let adapter = instance.adapters().remove(0);
    (instance, adapter)
}

fn device(adapter: &Adapter) -> Device {
    adapter
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(seed))
        .collect()
}

const fn extent(width: u32, height: u32) -> Extent3d {
    Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

/// The resources of a small frame: a buffer, an image, a render target.
struct Scene {
    source: Buffer,
    image: Texture,
    target: Texture,
    output: Buffer,
}

fn build(device: &mut Device) -> Result<Scene, RhiError> {
    let copy = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let source = device.create_buffer(&BufferDescriptor {
        label: Some("source"),
        size: 1024,
        usage: copy,
    })?;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("output"),
        size: 2048,
        usage: copy,
    })?;
    let image = device.create_texture(&TextureDescriptor {
        label: Some("image"),
        size: extent(8, 8),
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        ..TextureDescriptor::default()
    })?;
    let target = device.create_texture(&TextureDescriptor {
        label: Some("target"),
        size: extent(8, 8),
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ..TextureDescriptor::default()
    })?;
    Ok(Scene {
        source,
        image,
        target,
        output,
    })
}

/// Three frames of uploads, copies, clears and readback; the bytes they produce.
fn run(device: &mut Device, scene: &Scene) -> Result<Vec<u8>, RhiError> {
    let view = device.create_texture_view(scene.target, &TextureViewDescriptor::default())?;
    for frame in 0..3_u8 {
        device.begin_frame()?;
        device.upload_buffer(scene.source, 0, &pattern(1024, frame))?;
        let mut encoder = CommandEncoder::new(Some("frame"));
        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: scene.source,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            TexelCopyTextureInfo::new(scene.image),
            extent(8, 4),
        );
        encoder
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(RenderPassColorAttachment::clear(
                    view,
                    Color::new(0.25, 0.5, f64::from(frame) / 4.0, 1.0),
                ))],
                depth_stencil_attachment: None,
            })
            .end();
        for (texture, offset) in [(scene.image, 0), (scene.target, 1024)] {
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo::new(texture),
                TexelCopyBufferInfo {
                    buffer: scene.output,
                    layout: TexelCopyBufferLayout {
                        offset,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                extent(8, 4),
            );
        }
        device.submit([encoder.finish()])?;
        device.end_frame()?;
    }
    let readback = device.read_buffer_async(scene.output, 0, 2048)?;
    let bytes = device.wait_readback(readback)?;
    device.destroy_texture_view(view)?;
    Ok(bytes)
}

#[test]
fn loss_reaches_every_entry_point() {
    let (instance, adapter) = adapter(3);
    let mut device = device(&adapter);
    let pool = ThreadPool::new(PoolConfig::default().with_workers(2)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let scene = build(&mut device).expect("a scene");

    // Something of every kind.
    let module = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: Some("double"),
            source: ShaderSource::Wgsl(SHADER),
        })
        .expect("a module");
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        })
        .expect("a group layout");
    let storage = device
        .create_buffer(&BufferDescriptor {
            label: Some("storage"),
            size: 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
        .expect("a buffer");
    let group = device
        .create_bind_group(
            group_layout,
            &BindGroupDescriptor {
                label: None,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: storage,
                        offset: 0,
                        size: None,
                    }),
                }],
            },
        )
        .expect("a group");
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[group_layout],
        })
        .expect("a layout");
    let compute = ComputePipelineDescriptor {
        label: Some("double"),
        layout,
        module,
        entry_point: None,
    };
    let pipeline = device
        .create_compute_pipeline(&compute)
        .expect("a pipeline");
    let view = device
        .create_texture_view(scene.target, &TextureViewDescriptor::default())
        .expect("a view");
    let sampler = device
        .create_sampler(&SamplerDescriptor::default())
        .expect("a sampler");
    let mut surface = instance.create_null_surface(16, 16).expect("a surface");
    device
        .configure_surface(&mut surface, &surface_config())
        .expect("configured");
    let _swapchain = device.acquire_frame(&mut surface).expect("a frame");

    // Work in flight: a frame open with a submission, a deferred destruction, a readback, a
    // pipeline compiling, bytes being produced, an upload waiting in the belt.
    device.begin_frame().expect("a frame");
    let mut encoder = CommandEncoder::new(Some("double"));
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    device.submit([encoder.finish()]).expect("a submission");
    let doomed = device
        .create_buffer(&BufferDescriptor {
            label: Some("doomed"),
            size: 64,
            usage: BufferUsages::COPY_DST,
        })
        .expect("a buffer");
    device.destroy_buffer(doomed).expect("destroyed");
    assert_eq!(device.deferred_destructions(), 1);
    let mut readback = device.read_buffer_async(storage, 0, 256).expect("started");
    let compiling = device
        .compile_compute_pipeline(
            &ComputePipelineDescriptor {
                label: Some("again"),
                ..compute
            },
            &executor,
        )
        .expect("started");
    let (release, gate) = mpsc::channel::<()>();
    let producing = device
        .upload_async(
            UploadTarget::Buffer {
                buffer: scene.source,
                offset: 0,
            },
            &executor,
            move || {
                gate.recv().ok();
                Ok(pattern(64, 1))
            },
        )
        .expect("started");
    device
        .upload_buffer(scene.source, 64, &pattern(64, 2))
        .expect("staged");

    device.simulate_device_loss();
    assert!(device.is_lost());
    release.send(()).expect("the producer waits");

    let config = surface_config();
    let buffer_desc = BufferDescriptor {
        label: None,
        size: 16,
        usage: BufferUsages::COPY_DST,
    };
    let results: Vec<(&str, Result<(), RhiError>)> = vec![
        (
            "create_buffer",
            device.create_buffer(&buffer_desc).map(drop),
        ),
        ("buffer_info", device.buffer_info(storage).map(drop)),
        ("texture_info", device.texture_info(scene.image).map(drop)),
        (
            "texture_view_info",
            device.texture_view_info(view).map(drop),
        ),
        ("sampler_info", device.sampler_info(sampler).map(drop)),
        ("bind_group_info", device.bind_group_info(group).map(drop)),
        (
            "compute_pipeline_info",
            device.compute_pipeline_info(pipeline).map(drop),
        ),
        (
            "create_compute_pipeline",
            device.create_compute_pipeline(&compute).map(drop),
        ),
        (
            "compile_compute_pipeline",
            device
                .compile_compute_pipeline(&compute, &executor)
                .map(drop),
        ),
        (
            "wait_compute_pipeline",
            device.wait_compute_pipeline(compiling).map(drop),
        ),
        ("wait_upload", device.wait_upload(producing)),
        (
            "poll_readback",
            device
                .poll_readback(&mut readback)
                .map_or(Ok(()), |r| r.map(drop)),
        ),
        (
            "write_buffer",
            device.write_buffer(scene.source, 0, &[0; 4]),
        ),
        (
            "upload_buffer",
            device.upload_buffer(scene.source, 0, &[0; 4]),
        ),
        ("flush_uploads", device.flush_uploads().map(drop)),
        ("read_buffer", device.read_buffer(storage, 0, 16).map(drop)),
        (
            "read_texture",
            device
                .read_texture(TexelCopyTextureInfo::new(scene.image), extent(1, 1))
                .map(drop),
        ),
        (
            "submit",
            device
                .submit([CommandEncoder::new(None).finish()])
                .map(drop),
        ),
        ("end_frame", device.end_frame()),
        ("begin_frame", device.begin_frame().map(drop)),
        ("poll", device.poll()),
        ("wait_idle", device.wait_idle()),
        ("present", device.present(&mut surface)),
        (
            "acquire_frame",
            device.acquire_frame(&mut surface).map(drop),
        ),
        (
            "configure_surface",
            device.configure_surface(&mut surface, &config),
        ),
        ("destroy_texture_view", device.destroy_texture_view(view)),
        ("destroy_bind_group", device.destroy_bind_group(group)),
        ("destroy_buffer", device.destroy_buffer(storage)),
        ("destroy_sampler", device.destroy_sampler(sampler)),
    ];
    for (name, result) in results {
        assert_eq!(result, LOST, "{name} after the loss");
    }
    // Dropping a lost device with everything still alive and in flight is clean.
    drop(device);
    drop(surface);
}

fn surface_config() -> SurfaceConfiguration {
    SurfaceConfiguration {
        format: TextureFormat::Bgra8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        width: 16,
        height: 16,
        ..SurfaceConfiguration::default()
    }
}

#[test]
fn a_new_device_recovers_the_same_results() {
    for latency in [0, 2] {
        let (_instance, adapter) = adapter(latency);
        let mut first = device(&adapter);
        let scene = build(&mut first).expect("a scene");
        let expected = run(&mut first, &scene).expect("the first run");

        // Lost in the middle of the next run, with a frame open and submissions in flight.
        first.begin_frame().expect("a frame");
        first
            .upload_buffer(scene.source, 0, &pattern(1024, 9))
            .expect("staged");
        first.flush_uploads().expect("submitted");
        first.simulate_device_loss();
        assert_eq!(run(&mut first, &scene), Err(RhiError::DeviceLost));

        // The engine's recovery: a new device from the same adapter, everything rebuilt.
        let mut second = device(&adapter);
        assert_eq!(second.live(ResourceKind::Buffer), 0);
        assert!(matches!(
            second.buffer_info(scene.source),
            Err(RhiError::ForeignHandle { .. })
        ));
        let rebuilt = build(&mut second).expect("a scene");
        assert_eq!(
            run(&mut second, &rebuilt),
            Ok(expected),
            "latency {latency}"
        );
        drop(first);
        // The scene and the staging belt; nothing of the lost device is accounted here.
        let belt = second.upload_stats().allocated;
        assert_eq!(second.allocated_bytes(), Some(2048 + 1024 + 2 * 256 + belt));
    }
}

#[test]
fn a_surface_moves_to_the_new_device() {
    let (instance, adapter) = adapter(1);
    let mut surface = instance.create_null_surface(16, 16).expect("a surface");
    let mut first = device(&adapter);
    first
        .configure_surface(&mut surface, &surface_config())
        .expect("configured");
    let _frame = first.acquire_frame(&mut surface).expect("a frame");
    first.simulate_device_loss();
    assert_eq!(first.present(&mut surface), LOST);

    let mut second = device(&adapter);
    second
        .configure_surface(&mut surface, &surface_config())
        .expect("a lost device's frame does not block the new one");
    for frame in 0..3 {
        let acquired = second.acquire_frame(&mut surface).expect("a frame");
        let mut encoder = CommandEncoder::new(Some("present"));
        encoder
            .begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment::clear(
                    acquired.view,
                    Color::new(0.0, 0.0, f64::from(frame) / 2.0, 1.0),
                ))],
                depth_stencil_attachment: None,
            })
            .end();
        second.submit([encoder.finish()]).expect("a submission");
        second.present(&mut surface).expect("presented");
    }
    // Blue 255 in BGRA order.
    let presented = surface.null_presented().expect("presented texels");
    assert_eq!(&presented[..4], &[255, 0, 0, 255]);
    drop(first);
}

#[test]
fn blocking_pool_work_survives_the_loss() {
    // The pool and the executor outlive the device: work started for a lost device finishes
    // with DeviceLost, and the same executor serves the new device.
    let (_instance, adapter) = adapter(0);
    let pool = ThreadPool::new(PoolConfig::default().with_workers(2)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let mut first = device(&adapter);
    let scene = build(&mut first).expect("a scene");
    let pending: Vec<_> = (0..8_u8)
        .map(|seed| {
            first
                .upload_async(
                    UploadTarget::Buffer {
                        buffer: scene.source,
                        offset: u64::from(seed) * 64,
                    },
                    &executor,
                    move || Ok(pattern(64, seed)),
                )
                .expect("started")
        })
        .collect();
    first.simulate_device_loss();
    for upload in pending {
        assert_eq!(first.wait_upload(upload), LOST);
    }

    let mut second = device(&adapter);
    let rebuilt = build(&mut second).expect("a scene");
    let upload = second
        .upload_async(
            UploadTarget::Buffer {
                buffer: rebuilt.source,
                offset: 0,
            },
            &executor,
            || Ok(pattern(64, 5)),
        )
        .expect("started");
    second.wait_upload(upload).expect("uploaded");
    assert_eq!(
        second.read_buffer(rebuilt.source, 0, 64).expect("read"),
        pattern(64, 5)
    );
}
