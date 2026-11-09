//! Command recording, submission and readback (roadmap M3): the rules on the null backend, the
//! same transfers on every GPU adapter present, and recording on the pragma.core frame graph.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use core::num::NonZeroU64;

use pragma_core::graph::{GraphError, Resources};
use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_rhi::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, Color, ColorTargetState, CommandEncoder, CommandGraph,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Extent3d, FragmentState, Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations,
    Origin3d, PipelineLayoutDescriptor, PrimitiveState, RenderPass, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RhiError,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState, VertexStepMode,
};

// ---------------------------------------------------------------------------------------------
// Helpers

fn null_device() -> Device {
    Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
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

fn buffer(device: &mut Device, size: u64, usage: BufferUsages) -> Buffer {
    device
        .create_buffer(&BufferDescriptor {
            label: Some("test"),
            size,
            usage,
        })
        .expect("a buffer")
}

fn texture(
    device: &mut Device,
    (width, height): (u32, u32),
    format: TextureFormat,
    usage: TextureUsages,
    sample_count: u32,
) -> Texture {
    device
        .create_texture(&TextureDescriptor {
            label: Some("test"),
            size: extent(width, height),
            sample_count,
            format,
            usage,
            ..TextureDescriptor::default()
        })
        .expect("a texture")
}

fn view(device: &mut Device, texture: Texture) -> TextureView {
    device
        .create_texture_view(texture, &TextureViewDescriptor::default())
        .expect("a view")
}

const fn extent(width: u32, height: u32) -> Extent3d {
    Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

const COPY: BufferUsages = BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST);

/// The command and reason of an [`RhiError::InvalidCommand`].
fn reason(err: RhiError) -> (&'static str, String) {
    match err {
        RhiError::InvalidCommand {
            command, reason, ..
        } => (command, reason),
        other => panic!("expected an invalid command, got {other:?}"),
    }
}

fn submit_one(
    device: &mut Device,
    record: impl FnOnce(&mut CommandEncoder),
) -> Result<(), RhiError> {
    let mut encoder = CommandEncoder::new(Some("test"));
    record(&mut encoder);
    device.submit([encoder.finish()])
}

// ---------------------------------------------------------------------------------------------
// Transfers: the same bytes on every backend

/// Uploads, buffer and texture copies, clears, attachment clears, a resolve and a discard;
/// returns everything read back. Only transfers and load/store operations: the part of a
/// frame the null backend executes like a GPU.
fn transfer_scene(device: &mut Device) -> Result<Vec<u8>, RhiError> {
    let pattern: Vec<u8> = (0..1024_u32).map(|i| (i % 251) as u8).collect();
    let upload = buffer(device, 1024, COPY);
    let other = buffer(device, 256, COPY);
    device.write_buffer(upload, 0, &pattern)?;

    let usage = TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
    let image = texture(device, (8, 4), TextureFormat::Rgba8Unorm, usage, 1);
    let tile = texture(device, (4, 2), TextureFormat::Rgba8Unorm, usage, 1);
    let attachment = TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    let target = texture(device, (4, 4), TextureFormat::Rgba8UnormSrgb, attachment, 1);
    let msaa = texture(
        device,
        (4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
        4,
    );
    let resolved = texture(device, (4, 4), TextureFormat::Rgba8Unorm, attachment, 1);
    let discarded = texture(device, (4, 4), TextureFormat::R32Float, attachment, 1);
    let (target_view, msaa_view, resolved_view, discarded_view) = (
        view(device, target),
        view(device, msaa),
        view(device, resolved),
        view(device, discarded),
    );

    let mut encoder = CommandEncoder::new(Some("transfers"));
    encoder.copy_buffer_to_buffer(upload, 0, other, 16, 128);
    encoder.clear_buffer(other, 192, Some(32));
    encoder.copy_buffer_to_texture(
        TexelCopyBufferInfo {
            buffer: upload,
            layout: TexelCopyBufferLayout {
                offset: 4,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        TexelCopyTextureInfo::new(image),
        extent(8, 4),
    );
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: image,
            mip_level: 0,
            origin: Origin3d { x: 2, y: 1, z: 0 },
        },
        TexelCopyTextureInfo::new(tile),
        extent(4, 2),
    );
    encoder
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("clears"),
            color_attachments: &[
                Some(RenderPassColorAttachment::clear(
                    target_view,
                    Color::new(0.0, 1.0, 0.2, 0.6),
                )),
                None,
            ],
            depth_stencil_attachment: None,
        })
        .end();
    encoder
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(resolved_view),
                ops: Operations {
                    load: LoadOp::Clear(Color::new(0.2, 0.4, 1.0, 1.0)),
                    store: StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
        })
        .end();
    encoder
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("discard"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: discarded_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::new(0.5, 0.0, 0.0, 0.0)),
                    store: StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
        })
        .end();
    device.submit([encoder.finish()])?;

    let mut out = device.read_buffer(other, 0, 256)?;
    out.extend(device.read_texture(TexelCopyTextureInfo::new(image), extent(8, 4))?);
    out.extend(device.read_texture(TexelCopyTextureInfo::new(tile), extent(4, 2))?);
    out.extend(device.read_texture(TexelCopyTextureInfo::new(target), extent(4, 4))?);
    out.extend(device.read_texture(TexelCopyTextureInfo::new(resolved), extent(4, 4))?);
    out.extend(device.read_texture(TexelCopyTextureInfo::new(discarded), extent(4, 4))?);
    Ok(out)
}

#[test]
fn transfers_produce_the_expected_bytes_on_the_null_backend() {
    let mut device = null_device();
    let out = transfer_scene(&mut device).expect("the transfer scene");
    let pattern = |i: usize| (i % 251) as u8;
    let (buffer, rest) = out.split_at(256);
    // copy_buffer_to_buffer at 16, then a cleared range at 192.
    assert_eq!(&buffer[..16], &[0; 16]);
    assert!(
        buffer[16..144]
            .iter()
            .enumerate()
            .all(|(i, b)| *b == pattern(i))
    );
    assert_eq!(&buffer[192..224], &[0; 32]);
    let (image, rest) = rest.split_at(8 * 4 * 4);
    // Row y of the image starts at byte 4 + 256 * y of the upload.
    for y in 0..4 {
        let row = &image[y * 32..(y + 1) * 32];
        assert!(
            row.iter()
                .enumerate()
                .all(|(i, b)| *b == pattern(4 + 256 * y + i))
        );
    }
    let (tile, rest) = rest.split_at(4 * 2 * 4);
    // The tile is the image's region (2, 1)..(6, 3).
    for y in 0..2 {
        assert_eq!(
            &tile[y * 16..(y + 1) * 16],
            &image[(y + 1) * 32 + 8..(y + 1) * 32 + 24]
        );
    }
    let (target, rest) = rest.split_at(64);
    // sRGB: 0.2 encodes to 124; alpha stays linear (0.6 -> 153).
    assert!(target.chunks(4).all(|texel| texel == [0, 255, 124, 153]));
    let (resolved, discarded) = rest.split_at(64);
    assert!(resolved.chunks(4).all(|texel| texel == [51, 102, 255, 255]));
    assert_eq!(discarded, &[0; 64]);
}

#[test]
fn gpu_adapters_match_the_null_backend() {
    let expected = transfer_scene(&mut null_device()).expect("the null scene");
    for (name, mut device) in gpu_devices() {
        let out = transfer_scene(&mut device).expect(&name);
        assert_eq!(out, expected, "{name} differs from the null backend");
        eprintln!("checked {name}");
    }
}

// ---------------------------------------------------------------------------------------------
// Draws and dispatches

const SHADER: &str = r"
@group(0) @binding(0) var<uniform> tint: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // One triangle that covers the whole target.
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return tint;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = id.x * 3u;
}
";

struct Shaded {
    group: BindGroup,
    render: RenderPipeline,
    compute: ComputePipeline,
    tint: Buffer,
    data: Buffer,
}

fn buffer_entry(binding: u32, buffer: Buffer) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }),
    }
}

fn shaded(device: &mut Device) -> Result<Shaded, RhiError> {
    let module = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("shaded"),
        source: ShaderSource::Wgsl(SHADER),
    })?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
        ],
    })?;
    let tint = buffer(device, 16, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
    let data = buffer(device, 1024, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
    let group = device.create_bind_group(
        layout,
        &BindGroupDescriptor {
            label: None,
            entries: &[buffer_entry(0, tint), buffer_entry(1, data)],
        },
    )?;
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
    })?;
    let render = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("fill"),
        layout: pipeline_layout,
        vertex: VertexState {
            module,
            entry_point: None,
            buffers: &[],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: None,
            targets: &[Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))],
        }),
    })?;
    let compute = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("ids"),
        layout: pipeline_layout,
        module,
        entry_point: None,
    })?;
    Ok(Shaded {
        group,
        render,
        compute,
        tint,
        data,
    })
}

/// Clears a target to black, draws a triangle tinted green over it, fills the storage buffer
/// from a compute pass; returns the target's texels and the buffer.
fn draw_and_dispatch(device: &mut Device) -> Result<(Vec<u8>, Vec<u8>), RhiError> {
    let scene = shaded(device)?;
    let tint: Vec<u8> = [0.0_f32, 1.0, 0.0, 1.0]
        .iter()
        .flat_map(|c| c.to_le_bytes())
        .collect();
    device.write_buffer(scene.tint, 0, &tint)?;
    let usage = TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    let target = texture(device, (8, 8), TextureFormat::Rgba8Unorm, usage, 1);
    let target_view = view(device, target);

    let mut encoder = CommandEncoder::new(Some("frame"));
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("fill"),
            color_attachments: &[Some(RenderPassColorAttachment::clear(
                target_view,
                Color::BLACK,
            ))],
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(scene.render);
        pass.set_bind_group(0, scene.group, &[]);
        pass.draw(0..3, 0..1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("ids") });
        pass.set_pipeline(scene.compute);
        pass.set_bind_group(0, scene.group, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    device.submit([encoder.finish()])?;
    let pixels = device.read_texture(TexelCopyTextureInfo::new(target), extent(8, 8))?;
    let data = device.read_buffer(scene.data, 0, 1024)?;
    Ok((pixels, data))
}

#[test]
fn draws_and_dispatches_are_validated_but_run_no_shaders_on_null() {
    let mut device = null_device();
    let (pixels, data) = draw_and_dispatch(&mut device).expect("the frame");
    assert!(pixels.chunks(4).all(|texel| texel == [0, 0, 0, 255]));
    assert_eq!(data, vec![0; 1024]);
}

#[test]
fn gpu_adapters_draw_and_dispatch() {
    for (name, mut device) in gpu_devices() {
        let (pixels, data) = draw_and_dispatch(&mut device).expect(&name);
        assert!(
            pixels.chunks(4).all(|texel| texel == [0, 255, 0, 255]),
            "{name}: the triangle covers the target"
        );
        for (index, word) in data.chunks(4).enumerate() {
            let value = u32::from_le_bytes(word.try_into().expect("4 bytes"));
            assert_eq!(value, index as u32 * 3, "{name}: element {index}");
        }
        eprintln!("checked {name}");
    }
}

// ---------------------------------------------------------------------------------------------
// Rules

type Record = Box<dyn FnOnce(&mut CommandEncoder)>;

#[test]
fn buffer_copies_follow_webgpus_rules() {
    let mut device = null_device();
    let a = buffer(&mut device, 64, COPY);
    let b = buffer(&mut device, 64, COPY);
    let read_only = buffer(&mut device, 64, BufferUsages::COPY_SRC);

    let cases: [(&str, Record); 5] = [
        (
            "same buffer",
            Box::new(move |e| e.copy_buffer_to_buffer(a, 0, a, 32, 16)),
        ),
        (
            "COPY_DST",
            Box::new(move |e| e.copy_buffer_to_buffer(a, 0, read_only, 0, 16)),
        ),
        (
            "multiples of 4",
            Box::new(move |e| e.copy_buffer_to_buffer(a, 2, b, 0, 16)),
        ),
        (
            "out of bounds",
            Box::new(move |e| e.copy_buffer_to_buffer(a, 32, b, 0, 64)),
        ),
        (
            "past the end",
            Box::new(move |e| e.clear_buffer(b, 128, None)),
        ),
    ];
    for (expected, record) in cases {
        let (_, why) = reason(submit_one(&mut device, record).expect_err(expected));
        assert!(
            why.contains(expected),
            "`{why}` should mention `{expected}`"
        );
    }
    assert!(submit_one(&mut device, |e| e.copy_buffer_to_buffer(a, 0, b, 0, 64)).is_ok());
}

#[test]
fn texture_copies_follow_webgpus_rules() {
    let mut device = null_device();
    let usage = TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
    let image = texture(&mut device, (8, 8), TextureFormat::Rgba8Unorm, usage, 1);
    let staging = buffer(&mut device, 4096, COPY);
    let copy = |bytes_per_row, size| {
        move |e: &mut CommandEncoder| {
            e.copy_texture_to_buffer(
                TexelCopyTextureInfo::new(image),
                TexelCopyBufferInfo {
                    buffer: staging,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: None,
                    },
                },
                size,
            );
        }
    };
    let failures = [
        (copy(None, extent(8, 2)), "required"),
        (copy(Some(100), extent(8, 2)), "multiple of 256"),
        (copy(Some(256), extent(9, 2)), "bounds"),
        (copy(Some(1024), extent(8, 8)), "overruns"),
    ];
    for (record, expected) in failures {
        let (command, why) = reason(submit_one(&mut device, record).expect_err(expected));
        assert_eq!(command, "copy_texture_to_buffer");
        assert!(
            why.contains(expected),
            "`{why}` should mention `{expected}`"
        );
    }
    assert!(submit_one(&mut device, copy(None, extent(8, 1))).is_ok());
    assert!(submit_one(&mut device, copy(Some(256), extent(8, 8))).is_ok());
}

#[test]
fn a_rejected_submission_runs_nothing() {
    let mut device = null_device();
    let a = buffer(&mut device, 64, COPY);
    let b = buffer(&mut device, 64, COPY);
    device.write_buffer(b, 0, &[9; 64]).expect("a write");
    let mut good = CommandEncoder::new(Some("good"));
    good.clear_buffer(b, 0, None);
    let mut bad = CommandEncoder::new(Some("bad"));
    bad.copy_buffer_to_buffer(a, 0, b, 0, 128);
    let err = device
        .submit([good.finish(), bad.finish()])
        .expect_err("rejected");
    assert_eq!(
        err.to_string(),
        "invalid command `copy_buffer_to_buffer` in `bad`: the source range is out of bounds"
    );
    assert_eq!(device.read_buffer(b, 0, 64).expect("a read"), vec![9; 64]);
}

#[test]
fn resources_destroyed_after_recording_are_stale_at_submission() {
    let mut device = null_device();
    let a = buffer(&mut device, 64, COPY);
    let b = buffer(&mut device, 64, COPY);
    let mut encoder = CommandEncoder::new(None);
    encoder.copy_buffer_to_buffer(a, 0, b, 0, 64);
    device.destroy_buffer(b).expect("destroyed");
    assert!(matches!(
        device.submit([encoder.finish()]),
        Err(RhiError::StaleHandle { .. })
    ));
}

#[test]
fn queue_writes_and_reads_check_usage_and_alignment() {
    let mut device = null_device();
    let uniform = buffer(&mut device, 64, BufferUsages::UNIFORM);
    let copy = buffer(&mut device, 64, COPY);
    let err = device.write_buffer(uniform, 0, &[0; 4]).expect_err("usage");
    assert_eq!(reason(err).0, "write_buffer");
    assert!(device.write_buffer(copy, 2, &[0; 4]).is_err());
    assert!(device.write_buffer(copy, 0, &[0; 6]).is_err());
    assert!(device.write_buffer(copy, 60, &[0; 8]).is_err());
    let err = device.read_buffer(uniform, 0, 4).expect_err("usage");
    assert_eq!(reason(err).0, "read_buffer");
    assert!(device.read_buffer(copy, 0, 68).is_err());
    let mapped = buffer(
        &mut device,
        16,
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    device
        .write_buffer(mapped, 8, &[1, 2, 3, 4])
        .expect("a write");
    assert_eq!(
        device.read_buffer(mapped, 4, 8),
        Ok(vec![0, 0, 0, 0, 1, 2, 3, 4])
    );
}

type Draw = Box<dyn Fn(&mut RenderPass<'_>)>;

/// Records a render pass clearing `colors` and running `draw` in it.
fn pass(colors: Vec<TextureView>, draw: Draw) -> impl FnOnce(&mut CommandEncoder) {
    move |e: &mut CommandEncoder| {
        let attachments: Vec<_> = colors
            .iter()
            .map(|view| Some(RenderPassColorAttachment::clear(*view, Color::BLACK)))
            .collect();
        let mut pass = e.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &attachments,
            depth_stencil_attachment: None,
        });
        draw(&mut pass);
    }
}

#[test]
fn render_passes_check_attachments_and_draw_state() {
    let mut device = null_device();
    let scene = shaded(&mut device).expect("the scene");
    let usage = TextureUsages::RENDER_ATTACHMENT;
    let small = texture(&mut device, (4, 4), TextureFormat::Rgba8Unorm, usage, 1);
    let large = texture(&mut device, (8, 8), TextureFormat::Rgba8Unorm, usage, 1);
    let float = texture(&mut device, (4, 4), TextureFormat::R32Float, usage, 1);
    let sampled = texture(
        &mut device,
        (4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING,
        1,
    );
    let (small, large, float, sampled) = (
        view(&mut device, small),
        view(&mut device, large),
        view(&mut device, float),
        view(&mut device, sampled),
    );
    let (render, group) = (scene.render, scene.group);
    let failures: Vec<(Vec<TextureView>, Draw, &str, &str)> = vec![
        (
            vec![],
            Box::new(|_| ()),
            "begin_render_pass",
            "at least one",
        ),
        (
            vec![small, large],
            Box::new(|_| ()),
            "begin_render_pass",
            "differ in size",
        ),
        (
            vec![sampled],
            Box::new(|_| ()),
            "begin_render_pass",
            "RENDER_ATTACHMENT",
        ),
        (
            vec![small, small],
            Box::new(|_| ()),
            "begin_render_pass",
            "attached twice",
        ),
        (
            vec![small],
            Box::new(|p| p.draw(0..3, 0..1)),
            "draw",
            "no pipeline",
        ),
        (
            vec![float],
            Box::new(move |p| p.set_pipeline(render)),
            "set_pipeline",
            "color formats",
        ),
        (
            vec![small],
            Box::new(move |p| {
                p.set_pipeline(render);
                p.draw(0..3, 0..1);
            }),
            "draw",
            "bind group 0 is not set",
        ),
        (
            vec![small],
            Box::new(move |p| p.set_bind_group(0, group, &[0])),
            "set_bind_group",
            "more dynamic offsets",
        ),
        (
            vec![small],
            Box::new(|p| p.set_scissor_rect(2, 2, 4, 1)),
            "set_scissor_rect",
            "outside",
        ),
        (
            vec![small],
            Box::new(|p| p.set_viewport(0.0, 0.0, 4.0, 4.0, 0.5, 0.2)),
            "set_viewport",
            "depth range",
        ),
    ];
    for (colors, draw, command, expected) in failures {
        let err = submit_one(&mut device, pass(colors, draw)).expect_err(expected);
        let (got, why) = reason(err);
        assert_eq!(got, command, "{why}");
        assert!(
            why.contains(expected),
            "`{why}` should mention `{expected}`"
        );
    }
    let ok: Draw = Box::new(move |p| {
        p.set_pipeline(render);
        p.set_bind_group(0, group, &[]);
        p.draw(0..3, 0..1);
    });
    assert!(submit_one(&mut device, pass(vec![small], ok)).is_ok());
}

#[test]
fn vertex_buffers_must_hold_every_drawn_element() {
    let mut device = null_device();
    let module = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(
                "@vertex fn vs(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> {
                     return vec4<f32>(p, 0.0, 1.0);
                 }
                 @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
            ),
        })
        .expect("a module");
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDescriptor::default())
        .expect("a layout");
    let attributes = [VertexAttribute {
        format: VertexFormat::Float32x2,
        offset: 0,
        shader_location: 0,
    }];
    let pipeline = device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout,
            vertex: VertexState {
                module,
                entry_point: None,
                buffers: &[VertexBufferLayout {
                    array_stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module,
                entry_point: None,
                targets: &[Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))],
            }),
        })
        .expect("a pipeline");
    let vertices = buffer(&mut device, 24, BufferUsages::VERTEX);
    let target = texture(
        &mut device,
        (4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
        1,
    );
    let target = view(&mut device, target);
    let draw = |count: u32| -> Draw {
        Box::new(move |p| {
            p.set_pipeline(pipeline);
            p.set_vertex_buffer(0, vertices, 0, None);
            p.draw(0..count, 0..1);
        })
    };
    assert!(
        submit_one(&mut device, pass(vec![target], draw(3))).is_ok(),
        "3 vertices of 8 bytes fit 24 bytes"
    );
    let err = submit_one(&mut device, pass(vec![target], draw(4))).expect_err("too few");
    let (command, why) = reason(err);
    assert_eq!(command, "draw");
    assert!(why.contains("vertex buffer 0"), "{why}");
}

#[test]
fn usage_conflicts_in_a_pass_are_rejected() {
    let mut device = null_device();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        })
        .expect("a layout");
    let shared = buffer(
        &mut device,
        256,
        BufferUsages::STORAGE | BufferUsages::VERTEX,
    );
    let group = device
        .create_bind_group(
            layout,
            &BindGroupDescriptor {
                label: None,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: shared,
                        offset: 0,
                        size: NonZeroU64::new(256),
                    }),
                }],
            },
        )
        .expect("a group");
    let target = texture(
        &mut device,
        (4, 4),
        TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT,
        1,
    );
    let target = view(&mut device, target);
    let conflict: Draw = Box::new(move |p| {
        p.set_bind_group(0, group, &[]);
        p.set_vertex_buffer(0, shared, 0, None);
    });
    let err = submit_one(&mut device, pass(vec![target], conflict)).expect_err("a conflict");
    let (command, why) = reason(err);
    assert_eq!(command, "set_vertex_buffer");
    assert!(why.contains("writable storage"), "{why}");
}

#[test]
fn dispatches_check_the_pipeline_and_the_limits() {
    let mut device = null_device();
    let scene = shaded(&mut device).expect("the scene");
    let dispatch = |groups: bool, x: u32| {
        move |e: &mut CommandEncoder| {
            let mut pass = e.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(scene.compute);
            if groups {
                pass.set_bind_group(0, scene.group, &[]);
            }
            pass.dispatch_workgroups(x, 1, 1);
        }
    };
    assert!(submit_one(&mut device, dispatch(true, 65_535)).is_ok());
    let err = submit_one(&mut device, dispatch(true, 65_536)).expect_err("limit");
    let (command, why) = reason(err);
    assert_eq!(command, "dispatch_workgroups");
    assert!(
        why.contains("max_compute_workgroups_per_dimension"),
        "{why}"
    );
    let err = submit_one(&mut device, dispatch(false, 1)).expect_err("no group");
    assert!(reason(err).1.contains("not set"));
    let unbound = |e: &mut CommandEncoder| {
        e.begin_compute_pass(&ComputePassDescriptor::default())
            .dispatch_workgroups(1, 1, 1);
    };
    let err = submit_one(&mut device, unbound).expect_err("no pipeline");
    assert!(reason(err).1.contains("no pipeline"));
}

// ---------------------------------------------------------------------------------------------
// The frame graph

struct Particles(Buffer);
struct Lights(Buffer);
struct Shadows(Buffer);
struct Upload(Buffer);

fn graph() -> CommandGraph {
    let mut g = CommandGraph::builder();
    g.record(
        "particles",
        |t| t.writes::<Particles>(),
        |ctx, e| {
            e.clear_buffer(ctx.read::<Particles>()?.0, 0, None);
            Ok(())
        },
    );
    g.record(
        "lights",
        |t| t.writes::<Lights>(),
        |ctx, e| {
            e.clear_buffer(ctx.read::<Lights>()?.0, 0, None);
            Ok(())
        },
    );
    g.record(
        "shadows",
        |t| t.reads::<Lights>().writes::<Shadows>().after("lights"),
        |ctx, e| {
            let (lights, shadows) = (ctx.read::<Lights>()?.0, ctx.read::<Shadows>()?.0);
            e.copy_buffer_to_buffer(lights, 0, shadows, 0, 64);
            Ok(())
        },
    );
    g.record(
        "upload",
        |t| t.reads::<Upload>().writes::<Particles>().after("particles"),
        |ctx, e| {
            let (upload, particles) = (ctx.read::<Upload>()?.0, ctx.read::<Particles>()?.0);
            e.copy_buffer_to_buffer(upload, 0, particles, 0, 64);
            Ok(())
        },
    );
    g.build().expect("a graph")
}

fn resources(device: &mut Device) -> Resources {
    let mut resources = Resources::new();
    resources.insert(Particles(buffer(device, 64, COPY)));
    resources.insert(Lights(buffer(device, 64, COPY)));
    resources.insert(Shadows(buffer(device, 64, COPY)));
    resources.insert(Upload(buffer(device, 64, COPY)));
    resources
}

#[test]
fn recordings_come_back_in_one_order_on_any_number_of_workers() {
    let mut device = null_device();
    let mut resources = resources(&mut device);
    let mut graph = graph();
    let order: Vec<String> = graph.recordings().map(str::to_owned).collect();
    let mut runs = Vec::new();
    for workers in [1, 2, 4] {
        let pool = ThreadPool::new(PoolConfig::default().with_workers(workers)).expect("a pool");
        for _ in 0..8 {
            let buffers = graph.record(&pool, &mut resources).expect("recorded");
            let labels: Vec<String> = buffers
                .iter()
                .map(|b| b.label().unwrap_or_default().to_owned())
                .collect();
            assert_eq!(labels, order, "the topological order, on {workers} workers");
            runs.push(buffers);
        }
    }
    assert!(
        runs.windows(2).all(|pair| pair[0] == pair[1]),
        "identical command buffers"
    );
    device.submit(runs.remove(0)).expect("submitted");
}

#[test]
fn unordered_conflicting_recordings_fail_at_build() {
    let mut g = CommandGraph::builder();
    g.record("a", |t| t.writes::<Particles>(), |_, _| Ok(()));
    g.record("b", |t| t.writes::<Particles>(), |_, _| Ok(()));
    assert!(matches!(
        g.build(),
        Err(RhiError::Graph(GraphError::AmbiguousOrder { .. }))
    ));

    let mut g = CommandGraph::builder().ordered();
    g.record("a", |t| t.writes::<Particles>(), |_, _| Ok(()));
    g.record("b", |t| t.writes::<Particles>(), |_, _| Ok(()));
    let graph = g.build().expect("ordered by registration");
    assert_eq!(graph.recordings().collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn recording_errors_stay_typed() {
    let mut device = null_device();
    let mut resources = resources(&mut device);
    let mut g = CommandGraph::builder();
    g.record("fine", |t| t.reads::<Upload>(), |_, _| Ok(()));
    // Reads a resource it did not declare.
    g.record(
        "sloppy",
        |t| t,
        |ctx, _| {
            ctx.read::<Lights>()?;
            Ok(())
        },
    );
    let mut graph = g.build().expect("a graph");
    let pool = ThreadPool::new(PoolConfig::default().with_workers(2)).expect("a pool");
    assert!(matches!(
        graph.record(&pool, &mut resources),
        Err(RhiError::Graph(GraphError::UndeclaredAccess { .. }))
    ));
}
