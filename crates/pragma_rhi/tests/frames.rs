//! Frames in flight, deferred destruction, uploads and asynchronous readback (roadmap M4):
//! the rules on the null backend, whose latency setting keeps submissions "in flight", and the
//! same bytes on every GPU adapter present.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_core::tasks::{AsyncConfig, AsyncError, Executor};
use pragma_rhi::{
    Backends, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, DeviceDescriptor,
    Extent3d, Instance, InstanceDescriptor, NullConfig, PerFrame, ResourceKind, RhiError,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor, TextureFormat,
    TextureUsages, UploadConfig, UploadTarget,
};

// ---------------------------------------------------------------------------------------------
// Helpers

fn null_device_with(latency: u32, desc: &DeviceDescriptor<'_>) -> Device {
    Instance::new(&InstanceDescriptor {
        null: NullConfig {
            latency,
            ..NullConfig::default()
        },
        ..InstanceDescriptor::null()
    })
    .adapters()
    .remove(0)
    .request_device(desc)
    .expect("a null device")
}

fn null_device(latency: u32) -> Device {
    null_device_with(latency, &DeviceDescriptor::default())
}

/// A device on every Vulkan / D3D12 adapter of the machine (none on CI runners).
fn gpu_devices() -> Vec<(String, Device)> {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::VULKAN | Backends::DX12,
        ..InstanceDescriptor::default()
    });
    instance
        .adapters()
        .into_iter()
        .filter_map(|adapter| {
            let name = format!("{:?} {}", adapter.info().backend, adapter.info().name);
            match adapter.request_device(&DeviceDescriptor::default()) {
                Ok(device) => Some((name, device)),
                Err(err) => {
                    eprintln!("skipping {name}: {err}");
                    None
                }
            }
        })
        .collect()
}

const COPY: BufferUsages = BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST);

fn buffer(device: &mut Device, size: u64, usage: BufferUsages) -> Buffer {
    device
        .create_buffer(&BufferDescriptor {
            label: Some("test"),
            size,
            usage,
        })
        .expect("a buffer")
}

fn texture(device: &mut Device, width: u32, height: u32) -> Texture {
    device
        .create_texture(&TextureDescriptor {
            label: Some("test"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            ..TextureDescriptor::default()
        })
        .expect("a texture")
}

/// One submission that does a little work.
fn tick(device: &mut Device, scratch: Buffer) {
    let mut encoder = CommandEncoder::new(Some("tick"));
    encoder.clear_buffer(scratch, 0, None);
    device.submit([encoder.finish()]).expect("a submission");
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Fences and frames

#[test]
fn submissions_are_numbered_and_complete_after_the_latency() {
    let mut device = null_device(2);
    let scratch = buffer(&mut device, 16, COPY);
    assert_eq!(device.last_submission(), None);
    let mut fences = Vec::new();
    for _ in 0..4 {
        tick(&mut device, scratch);
        fences.push(device.last_submission().expect("a submission"));
    }
    assert_eq!(
        fences.iter().map(|fence| fence.get()).collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert!(
        !device.is_complete(fences[0]),
        "nothing is known before a poll"
    );
    device.poll().expect("a poll");
    // Two submissions stay in flight.
    assert!(device.is_complete(fences[1]));
    assert!(!device.is_complete(fences[2]));
    device.wait(fences[2]).expect("a wait");
    assert!(device.is_complete(fences[2]));
    assert!(!device.is_complete(fences[3]));
    device.wait_idle().expect("a wait");
    assert!(device.is_complete(fences[3]));
}

#[test]
fn begin_frame_bounds_the_frames_in_flight() {
    // The null GPU never finishes on its own: only begin_frame's waits complete frames.
    let mut device = null_device_with(
        u32::MAX,
        &DeviceDescriptor {
            frames_in_flight: 2,
            ..DeviceDescriptor::default()
        },
    );
    let scratch = buffer(&mut device, 16, COPY);
    let mut fences = Vec::new();
    for expected in 0..5_u64 {
        let frame = device.begin_frame().expect("a frame");
        assert_eq!(frame.index, expected);
        assert_eq!(frame.slot, (expected % 2) as usize);
        if let Some(older) = expected.checked_sub(2) {
            // Two frames ago has retired: at most one frame is in flight besides this one.
            assert!(device.is_complete(fences[older as usize]));
        }
        if let Some(previous) = expected.checked_sub(1) {
            assert!(
                !device.is_complete(fences[previous as usize]),
                "the previous frame may still run"
            );
        }
        tick(&mut device, scratch);
        fences.push(device.last_submission().expect("a submission"));
        device.end_frame().expect("the frame ends");
    }
}

#[test]
fn the_frame_protocol_is_checked() {
    let mut device = null_device(0);
    assert!(matches!(device.end_frame(), Err(RhiError::Frame { .. })));
    device.begin_frame().expect("a frame");
    assert!(device.in_frame());
    assert!(matches!(device.begin_frame(), Err(RhiError::Frame { .. })));
    device.end_frame().expect("the frame ends");
    let adapter = Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0);
    for frames_in_flight in [0, 9] {
        assert!(matches!(
            adapter.request_device(&DeviceDescriptor {
                frames_in_flight,
                ..DeviceDescriptor::default()
            }),
            Err(RhiError::Frame { .. })
        ));
    }
}

#[test]
fn per_frame_values_rotate_with_the_slot() {
    let mut device = null_device_with(
        0,
        &DeviceDescriptor {
            frames_in_flight: 3,
            ..DeviceDescriptor::default()
        },
    );
    let mut writes = PerFrame::new(&device, |_| Vec::new());
    assert_eq!(writes.len(), 3);
    for _ in 0..7 {
        let frame = device.begin_frame().expect("a frame");
        writes.get_mut(frame).push(frame.index);
        device.end_frame().expect("the frame ends");
    }
    let slots: Vec<Vec<u64>> = writes.iter().cloned().collect();
    assert_eq!(slots, [vec![0, 3, 6], vec![1, 4], vec![2, 5]]);
}

// ---------------------------------------------------------------------------------------------
// Deferred destruction

#[test]
fn destroyed_resources_keep_their_memory_until_their_submissions_retire() {
    let mut device = null_device(1);
    let scratch = buffer(&mut device, 16, COPY);
    let base = device
        .allocated_bytes()
        .expect("the null backend accounts memory");

    // Nothing in flight: freed at once.
    let idle = buffer(&mut device, 1024, COPY);
    device.destroy_buffer(idle).expect("destroyed");
    assert_eq!(device.allocated_bytes(), Some(base));
    assert_eq!(device.deferred_destructions(), 0);

    let doomed = buffer(&mut device, 1024, COPY);
    let image = texture(&mut device, 8, 8);
    tick(&mut device, scratch); // submission 1, in flight (latency 1)
    device.destroy_buffer(doomed).expect("destroyed");
    device.destroy_texture(image).expect("destroyed");
    // The handles are stale at once; the memory is not freed yet.
    assert!(matches!(
        device.buffer_info(doomed),
        Err(RhiError::StaleHandle {
            resource: ResourceKind::Buffer,
            ..
        })
    ));
    assert_eq!(device.deferred_destructions(), 2);
    assert_eq!(device.allocated_bytes(), Some(base + 1024 + 8 * 8 * 4));
    device.poll().expect("a poll");
    assert_eq!(
        device.deferred_destructions(),
        2,
        "submission 1 is in flight"
    );

    tick(&mut device, scratch); // submission 2: submission 1 retires on the next poll
    device.poll().expect("a poll");
    assert_eq!(device.deferred_destructions(), 0);
    assert_eq!(device.allocated_bytes(), Some(base));
}

#[test]
fn begin_frame_frees_what_retired() {
    let mut device = null_device(u32::MAX);
    let scratch = buffer(&mut device, 16, COPY);
    let base = device.allocated_bytes().expect("accounted");
    for _ in 0..4 {
        device.begin_frame().expect("a frame");
        let transient = buffer(&mut device, 4096, COPY);
        tick(&mut device, scratch);
        device.destroy_buffer(transient).expect("destroyed");
        device.end_frame().expect("the frame ends");
        // At most two frames' transients are alive: the bounded latency bounds the memory.
        assert!(device.allocated_bytes().expect("accounted") <= base + 2 * 4096);
    }
    device.wait_idle().expect("idle");
    assert_eq!(device.allocated_bytes(), Some(base));
}

// ---------------------------------------------------------------------------------------------
// Uploads

#[test]
fn uploads_land_at_the_next_submission_after_queue_writes() {
    let mut device = null_device(0);
    let target = buffer(&mut device, 64, COPY);
    let data = pattern(32, 1);
    device.upload_buffer(target, 16, &data).expect("an upload");
    assert_eq!(device.upload_stats().pending, 32);
    // A queue write to the same range made later still runs first.
    device
        .write_buffer(target, 16, &[0xEE; 32])
        .expect("a write");
    device
        .flush_uploads()
        .expect("flushed")
        .expect("a submission");
    let bytes = device.read_buffer(target, 0, 64).expect("read");
    assert_eq!(&bytes[..16], &[0; 16]);
    assert_eq!(&bytes[16..48], data.as_slice());
    let stats = device.upload_stats();
    assert_eq!((stats.pending, stats.uploaded), (0, 32));
    assert_eq!(device.flush_uploads(), Ok(None), "nothing left to flush");
}

#[test]
fn uploads_are_validated_when_recorded() {
    let mut device = null_device(0);
    let target = buffer(&mut device, 64, COPY);
    let read_only = buffer(&mut device, 64, BufferUsages::COPY_SRC);
    let reason = |result: Result<(), RhiError>| match result {
        Err(RhiError::InvalidCommand {
            command, reason, ..
        }) => (command, reason),
        other => panic!("expected an invalid command, got {other:?}"),
    };
    assert_eq!(
        reason(device.upload_buffer(read_only, 0, &[0; 4])).0,
        "upload_buffer"
    );
    assert!(
        reason(device.upload_buffer(target, 2, &[0; 4]))
            .1
            .contains("multiples of 4")
    );
    assert!(
        reason(device.upload_buffer(target, 0, &[0; 6]))
            .1
            .contains("multiples of 4")
    );
    assert!(
        reason(device.upload_buffer(target, 48, &[0; 32]))
            .1
            .contains("out of bounds")
    );
    let image = texture(&mut device, 4, 4);
    let short = device.upload_texture(
        TexelCopyTextureInfo::new(image),
        &[0; 60],
        TexelCopyBufferLayout::default(),
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    assert!(reason(short).1.contains("too short"));
    assert_eq!(device.upload_stats().pending, 0, "nothing was recorded");
}

#[test]
fn uploads_survive_a_rejected_submission() {
    let mut device = null_device(0);
    let target = buffer(&mut device, 16, COPY);
    let stale = buffer(&mut device, 16, COPY);
    let mut encoder = CommandEncoder::new(Some("stale"));
    encoder.clear_buffer(stale, 0, None);
    let broken = encoder.finish();
    device.destroy_buffer(stale).expect("destroyed");
    device
        .upload_buffer(target, 0, &[9; 16])
        .expect("an upload");
    assert!(matches!(
        device.submit([broken]),
        Err(RhiError::StaleHandle { .. })
    ));
    assert_eq!(
        device.upload_stats().pending,
        16,
        "kept for the next submission"
    );
    device.submit([]).expect("a submission");
    assert_eq!(device.read_buffer(target, 0, 16).expect("read"), [9; 16]);
}

#[test]
fn texture_uploads_repack_any_row_stride() {
    let mut device = null_device(0);
    let image = texture(&mut device, 3, 2);
    // 3 texels (12 bytes) per row, given with a stride of 20 bytes after a 4-byte header.
    let mut data = vec![0xAA; 4];
    let rows = [pattern(12, 10), pattern(12, 20)];
    for row in &rows {
        data.extend_from_slice(row);
        data.extend_from_slice(&[0xBB; 8]);
    }
    let layout = TexelCopyBufferLayout {
        offset: 4,
        bytes_per_row: Some(20),
        rows_per_image: None,
    };
    let size = Extent3d {
        width: 3,
        height: 2,
        depth_or_array_layers: 1,
    };
    device
        .upload_texture(TexelCopyTextureInfo::new(image), &data, layout, size)
        .expect("an upload");
    let texels = device
        .read_texture(TexelCopyTextureInfo::new(image), size)
        .expect("read");
    assert_eq!(texels, [rows[0].clone(), rows[1].clone()].concat());
}

#[test]
fn the_belt_is_bounded_and_recycles_retired_chunks() {
    let mut device = null_device_with(
        1,
        &DeviceDescriptor {
            upload: UploadConfig {
                chunk_size: 1024,
                capacity: 2048,
            },
            ..DeviceDescriptor::default()
        },
    );
    let target = buffer(&mut device, 1024, COPY);
    for round in 0..10_u8 {
        let data = pattern(1024, round);
        device.upload_buffer(target, 0, &data).expect("an upload");
        device.submit([]).expect("a submission");
        let stats = device.upload_stats();
        assert!(stats.allocated <= 2048, "round {round}: {stats:?}");
        assert!(stats.chunks <= 2);
    }
    assert!(
        device.upload_stats().stalls > 0,
        "a full belt waited for the GPU"
    );
    assert_eq!(
        device.read_buffer(target, 0, 1024).expect("read"),
        pattern(1024, 9)
    );
    let large = buffer(&mut device, 4096, COPY);
    assert!(matches!(
        device.upload_buffer(large, 0, &vec![0; 4096]),
        Err(RhiError::LimitExceeded {
            limit: "upload capacity",
            ..
        })
    ));
}

#[test]
fn many_small_uploads_share_a_chunk() {
    let mut device = null_device(0);
    let target = buffer(&mut device, 4096, COPY);
    for slot in 0..64_u64 {
        let data = pattern(64, slot as u8);
        device
            .upload_buffer(target, slot * 64, &data)
            .expect("an upload");
    }
    assert_eq!(device.upload_stats().chunks, 1);
    let bytes = device.read_buffer(target, 0, 4096).expect("read");
    for slot in 0..64_usize {
        assert_eq!(&bytes[slot * 64..slot * 64 + 64], pattern(64, slot as u8));
    }
}

#[test]
fn async_uploads_produce_off_the_frame_thread() {
    let pool = ThreadPool::new(PoolConfig::default().with_workers(1)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let mut device = null_device(0);
    let target = buffer(&mut device, 256, COPY);

    let mut pending = device
        .upload_async(
            UploadTarget::Buffer {
                buffer: target,
                offset: 0,
            },
            &executor,
            || Ok(pattern(256, 7)),
        )
        .expect("started");
    let uploaded = loop {
        if let Some(result) = device.poll_upload(&mut pending) {
            break result;
        }
        std::thread::yield_now();
    };
    uploaded.expect("uploaded");
    assert!(matches!(
        device.poll_upload(&mut pending),
        Some(Err(RhiError::Task(AsyncError::Consumed)))
    ));
    assert_eq!(
        device.read_buffer(target, 0, 256).expect("read"),
        pattern(256, 7)
    );

    let image = texture(&mut device, 2, 2);
    let pending = device
        .upload_async(
            UploadTarget::Texture {
                texture: TexelCopyTextureInfo::new(image),
                layout: TexelCopyBufferLayout::default(),
                size: Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            },
            &executor,
            || Err(RhiError::DeviceLost),
        )
        .expect("started");
    assert_eq!(
        device.wait_upload(pending),
        Err(RhiError::DeviceLost),
        "the producer's error comes back as it was"
    );
}

// ---------------------------------------------------------------------------------------------
// Asynchronous readback

#[test]
fn async_readback_arrives_after_its_submission_retires() {
    let mut device = null_device(2);
    let source = buffer(&mut device, 64, COPY);
    let scratch = buffer(&mut device, 16, COPY);
    device
        .write_buffer(source, 0, &pattern(64, 3))
        .expect("a write");
    let mut readback = device.read_buffer_async(source, 0, 64).expect("started");
    assert_eq!(device.poll_readback(&mut readback), None, "still in flight");
    tick(&mut device, scratch);
    assert_eq!(device.poll_readback(&mut readback), None, "still in flight");
    tick(&mut device, scratch);
    let bytes = device
        .poll_readback(&mut readback)
        .expect("ready")
        .expect("the bytes");
    assert_eq!(bytes, pattern(64, 3));
    assert!(matches!(
        device.poll_readback(&mut readback),
        Some(Err(RhiError::Task(AsyncError::Consumed)))
    ));

    let image = texture(&mut device, 3, 2);
    let texels = pattern(24, 5);
    let size = Extent3d {
        width: 3,
        height: 2,
        depth_or_array_layers: 1,
    };
    device
        .upload_texture(
            TexelCopyTextureInfo::new(image),
            &texels,
            TexelCopyBufferLayout::default(),
            size,
        )
        .expect("an upload");
    let readback = device
        .read_texture_async(TexelCopyTextureInfo::new(image), size)
        .expect("started");
    assert_eq!(device.wait_readback(readback).expect("the texels"), texels);
}

#[test]
fn dropped_readbacks_free_their_staging() {
    let mut device = null_device(0);
    let source = buffer(&mut device, 64, COPY);
    let live = device.live(ResourceKind::Buffer);
    let readback = device.read_buffer_async(source, 0, 64).expect("started");
    assert_eq!(device.live(ResourceKind::Buffer), live + 1);
    drop(readback);
    device.poll().expect("a poll");
    assert_eq!(device.live(ResourceKind::Buffer), live);
}

#[test]
fn device_loss_ends_frames_uploads_and_readbacks() {
    let mut device = null_device(0);
    let target = buffer(&mut device, 16, COPY);
    let mut readback = device.read_buffer_async(target, 0, 16).expect("started");
    device.simulate_device_loss();
    assert_eq!(device.begin_frame(), Err(RhiError::DeviceLost));
    assert_eq!(device.poll(), Err(RhiError::DeviceLost));
    assert_eq!(
        device.upload_buffer(target, 0, &[0; 16]),
        Err(RhiError::DeviceLost)
    );
    assert_eq!(
        device.poll_readback(&mut readback),
        Some(Err(RhiError::DeviceLost))
    );
}

// ---------------------------------------------------------------------------------------------
// The same bytes on GPUs

fn upload_scene(device: &mut Device) -> Result<Vec<u8>, RhiError> {
    let target = buffer(device, 4096, COPY);
    let image = texture(device, 5, 3);
    let size = Extent3d {
        width: 5,
        height: 3,
        depth_or_array_layers: 1,
    };
    let stride = 5 * 4 + 12;
    let mut out = Vec::new();
    for frame in 0..6_u8 {
        device.begin_frame()?;
        device.upload_buffer(target, u64::from(frame) * 256, &pattern(256, frame))?;
        let texels: Vec<u8> = (0..3)
            .flat_map(|row| {
                let mut line = pattern(20, frame.wrapping_add(row));
                line.resize(stride, 0xCC);
                line
            })
            .collect();
        device.upload_texture(
            TexelCopyTextureInfo::new(image),
            &texels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: None,
            },
            size,
        )?;
        device.submit([])?;
        let readback = device.read_texture_async(TexelCopyTextureInfo::new(image), size)?;
        device.end_frame()?;
        out.extend(device.wait_readback(readback)?);
    }
    let readback = device.read_buffer_async(target, 0, 4096)?;
    out.extend(device.wait_readback(readback)?);
    Ok(out)
}

#[test]
fn gpu_adapters_match_the_null_backend_on_uploads_and_readback() {
    let expected = upload_scene(&mut null_device(1)).expect("the null scene");
    for (name, mut device) in gpu_devices() {
        let bytes = upload_scene(&mut device).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert!(
            bytes == expected,
            "{name}: uploads differ from the null backend"
        );
        eprintln!("{name}: uploads and readbacks match");
    }
}
