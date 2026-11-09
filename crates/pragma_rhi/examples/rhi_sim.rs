//! The vertical slice of roadmap M3: device → resources → pipelines → commands recorded in
//! parallel on the pragma.core frame graph → submission → readback.
//!
//! Every adapter of the machine (and the null one) runs the same three frames of a small
//! headless scene — a compute pass moves particles, a render pass draws triangles into an
//! offscreen target, a copy fills a texture and a clear-only pass paints a backdrop — once on
//! one worker and once on several. Two FNV-1a hashes come out per adapter:
//!
//! - **transfer**: the texels produced by copies and attachment clears. They must be the same
//!   on every backend, the null one included (it runs no shaders, but executes these exactly).
//! - **render**: the drawn target and the particle buffer. They must be the same on one and
//!   on N workers; across GPUs rasterization may differ, and on the null backend shaders do
//!   not run.
//!
//! `cargo run --example rhi_sim` checks every adapter; `-- --null` only the null one (CI).

use std::error::Error;
use std::process::ExitCode;

use pragma_core::graph::Resources;
use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_rhi::{
    Backend, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, Color, ColorTargetState, CommandGraph, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, FragmentState,
    Instance, InstanceDescriptor, MultisampleState, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureDescriptor, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

const SHADER: &str = r"
struct Particle { position: vec2<f32>, velocity: vec2<f32> }
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    var particle = particles[i];
    particle.position = particle.position + particle.velocity;
    particle.velocity = particle.velocity * 0.5;
    particles[i] = particle;
}

struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }

@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> VertexOut {
    return VertexOut(vec4<f32>(position, 0.0, 1.0), color);
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
";

const SIZE: u32 = 64;
const PARTICLES: u32 = 256;
const PARTICLE_BYTES: u64 = 16;
const FRAMES: usize = 3;

// Frame-graph resources, declared by type like ECS components.
struct Pipelines {
    simulate: ComputePipeline,
    draw: RenderPipeline,
    group: BindGroup,
}
struct Particles(Buffer);
struct Geometry(Buffer);
struct Staging(Buffer);
struct Target(TextureView, Texture);
struct Snapshot(Texture);
struct Atlas(Texture);
struct Backdrop(TextureView, Texture);

/// FNV-1a, 64 bits.
fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

const fn extent(width: u32, height: u32) -> Extent3d {
    Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn initial_particles() -> Vec<u8> {
    (0..PARTICLES)
        .flat_map(|i| {
            let f = i as f32;
            [f * 0.01, -f * 0.02, 1.0 / (f + 1.0), f * 0.25]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn geometry() -> Vec<u8> {
    // Three triangles: a position (2 x f32) and a color (4 x unorm8) per vertex.
    let triangles: [([f32; 6], [u8; 4]); 3] = [
        ([-0.9, -0.9, 0.0, -0.9, -0.45, 0.0], [255, 0, 0, 255]),
        ([0.0, 0.0, 0.9, 0.0, 0.45, 0.9], [0, 255, 0, 255]),
        ([-0.8, 0.2, -0.2, 0.2, -0.5, 0.8], [40, 80, 255, 255]),
    ];
    let mut bytes = Vec::new();
    for (positions, color) in triangles {
        for vertex in positions.chunks(2) {
            for value in vertex {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(color);
        }
    }
    bytes
}

fn texture(
    device: &mut Device,
    label: &str,
    format: TextureFormat,
    usage: TextureUsages,
) -> Result<Texture, Box<dyn Error>> {
    Ok(device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: extent(SIZE, SIZE),
        format,
        usage,
        ..TextureDescriptor::default()
    })?)
}

fn pipelines(device: &mut Device, particles: Buffer) -> Result<Pipelines, Box<dyn Error>> {
    let module = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("rhi_sim"),
        source: ShaderSource::Wgsl(SHADER),
    })?;
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("particles"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        }],
    })?;
    let compute_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("simulate"),
        bind_group_layouts: &[layout],
    })?;
    let simulate = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("simulate"),
        layout: compute_layout,
        module,
        entry_point: None,
    })?;
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Unorm8x4,
            offset: 8,
            shader_location: 1,
        },
    ];
    let draw_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
    let draw = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("draw"),
        layout: draw_layout,
        vertex: VertexState {
            module,
            entry_point: None,
            buffers: &[VertexBufferLayout {
                array_stride: 12,
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
    })?;
    let group = device.create_bind_group(
        layout,
        &BindGroupDescriptor {
            label: Some("particles"),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: particles,
                    offset: 0,
                    size: None,
                }),
            }],
        },
    )?;
    Ok(Pipelines {
        simulate,
        draw,
        group,
    })
}

/// Creates the scene on `device` and puts it into a fresh set of graph resources.
fn scene(device: &mut Device) -> Result<Resources, Box<dyn Error>> {
    let particles = device.create_buffer(&BufferDescriptor {
        label: Some("particles"),
        size: u64::from(PARTICLES) * PARTICLE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    })?;
    let pipelines = pipelines(device, particles)?;
    let vertices = geometry();
    let geometry = device.create_buffer(&BufferDescriptor {
        label: Some("geometry"),
        size: vertices.len() as u64,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    })?;
    device.write_buffer(geometry, 0, &vertices)?;
    // A SIZE x SIZE pattern, every row padded to 256 bytes.
    let pattern: Vec<u8> = (0..SIZE * 256)
        .map(|i| (i.wrapping_mul(7) % 253) as u8)
        .collect();
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("staging"),
        size: pattern.len() as u64,
        usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    })?;
    device.write_buffer(staging, 0, &pattern)?;

    let copy = TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
    let attachment = TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    let target = texture(device, "target", TextureFormat::Rgba8Unorm, attachment)?;
    let snapshot = texture(device, "snapshot", TextureFormat::Rgba8Unorm, copy)?;
    let atlas = texture(device, "atlas", TextureFormat::Rgba8Unorm, copy)?;
    let backdrop = texture(
        device,
        "backdrop",
        TextureFormat::Bgra8UnormSrgb,
        attachment,
    )?;
    let target_view = device.create_texture_view(target, &TextureViewDescriptor::default())?;
    let backdrop_view = device.create_texture_view(backdrop, &TextureViewDescriptor::default())?;

    let mut resources = Resources::new();
    resources.insert(pipelines);
    resources.insert(Particles(particles));
    resources.insert(Geometry(geometry));
    resources.insert(Staging(staging));
    resources.insert(Target(target_view, target));
    resources.insert(Snapshot(snapshot));
    resources.insert(Atlas(atlas));
    resources.insert(Backdrop(backdrop_view, backdrop));
    Ok(resources)
}

/// The frame: five recordings; `simulate`, `draw`, `atlas` and `backdrop` are independent and
/// record in parallel, `snapshot` waits for `draw`.
fn frame_graph() -> Result<CommandGraph, Box<dyn Error>> {
    let mut g = CommandGraph::builder();
    g.record(
        "simulate",
        |t| t.reads::<Pipelines>().writes::<Particles>(),
        |ctx, encoder| {
            let pipelines = ctx.read::<Pipelines>()?;
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("simulate"),
            });
            pass.set_pipeline(pipelines.simulate);
            pass.set_bind_group(0, pipelines.group, &[]);
            pass.dispatch_workgroups(PARTICLES.div_ceil(64), 1, 1);
            Ok(())
        },
    );
    g.record(
        "draw",
        |t| {
            t.reads::<Pipelines>()
                .reads::<Geometry>()
                .writes::<Target>()
        },
        |ctx, encoder| {
            let (pipelines, geometry) = (ctx.read::<Pipelines>()?, ctx.read::<Geometry>()?);
            let target = ctx.read::<Target>()?;
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("draw"),
                color_attachments: &[Some(RenderPassColorAttachment::clear(
                    target.0,
                    Color::new(0.1, 0.1, 0.1, 1.0),
                ))],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(pipelines.draw);
            pass.set_vertex_buffer(0, geometry.0, 0, None);
            pass.draw(0..9, 0..1);
            Ok(())
        },
    );
    g.record(
        "atlas",
        |t| t.reads::<Staging>().writes::<Atlas>(),
        |ctx, encoder| {
            let (staging, atlas) = (ctx.read::<Staging>()?.0, ctx.read::<Atlas>()?.0);
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: staging,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                TexelCopyTextureInfo::new(atlas),
                extent(SIZE, SIZE),
            );
            Ok(())
        },
    );
    g.record(
        "backdrop",
        |t| t.writes::<Backdrop>(),
        |ctx, encoder| {
            let backdrop = ctx.read::<Backdrop>()?.0;
            encoder
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("backdrop"),
                    color_attachments: &[Some(RenderPassColorAttachment::clear(
                        backdrop,
                        Color::new(0.25, 0.5, 0.75, 1.0),
                    ))],
                    depth_stencil_attachment: None,
                })
                .end();
            Ok(())
        },
    );
    g.record(
        "snapshot",
        |t| t.reads::<Target>().writes::<Snapshot>().after("draw"),
        |ctx, encoder| {
            let (target, snapshot) = (ctx.read::<Target>()?.1, ctx.read::<Snapshot>()?.0);
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo::new(target),
                TexelCopyTextureInfo::new(snapshot),
                extent(SIZE, SIZE),
            );
            Ok(())
        },
    );
    Ok(g.build()?)
}

/// Resets the particles, runs `FRAMES` frames on `workers` and hashes what they produced.
fn run_frames(
    device: &mut Device,
    resources: &mut Resources,
    workers: usize,
) -> Result<(u64, u64), Box<dyn Error>> {
    let pool = ThreadPool::new(PoolConfig::default().with_workers(workers))?;
    let mut graph = frame_graph()?;
    let particles = resources.get::<Particles>().ok_or("no particles")?.0;
    let atlas = resources.get::<Atlas>().ok_or("no atlas")?.0;
    let backdrop = resources.get::<Backdrop>().ok_or("no backdrop")?.1;
    let snapshot = resources.get::<Snapshot>().ok_or("no snapshot")?.0;
    device.write_buffer(particles, 0, &initial_particles())?;
    for _ in 0..FRAMES {
        let buffers = graph.record(&pool, resources)?;
        device.submit(buffers)?;
    }
    let full = extent(SIZE, SIZE);
    let mut transfer = device.read_texture(TexelCopyTextureInfo::new(atlas), full)?;
    transfer.extend(device.read_texture(TexelCopyTextureInfo::new(backdrop), full)?);
    let mut render = device.read_texture(TexelCopyTextureInfo::new(snapshot), full)?;
    render.extend(device.read_buffer(particles, 0, u64::from(PARTICLES) * PARTICLE_BYTES)?);
    Ok((hash(&transfer), hash(&render)))
}

fn run(null_only: bool) -> Result<bool, Box<dyn Error>> {
    let backends = if null_only {
        Backends::NULL
    } else {
        Backends::all()
    };
    let instance = Instance::new(&InstanceDescriptor {
        backends,
        ..InstanceDescriptor::default()
    });
    let workers = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .clamp(2, 8);
    let mut transfers = Vec::new();
    let mut ok = true;
    for adapter in instance.adapters() {
        let backend = adapter.info().backend;
        let name = format!("{backend:?} {}", adapter.info().name);
        let mut device = match adapter.request_device(&DeviceDescriptor::default()) {
            Ok(device) => device,
            Err(err) => {
                println!("{name}: skipped ({err})");
                continue;
            }
        };
        let mut resources = scene(&mut device)?;
        let one = run_frames(&mut device, &mut resources, 1)?;
        let many = run_frames(&mut device, &mut resources, workers)?;
        ok &= one == many;
        println!(
            "{name}: transfer {:016x}  render {:016x}  1 vs {workers} workers: {}",
            one.0,
            one.1,
            if one == many {
                "identical"
            } else {
                "DIFFERENT"
            }
        );
        transfers.push((backend, one.0));
    }
    if let Some((_, first)) = transfers.first() {
        let agree = transfers.iter().all(|(_, hash)| hash == first);
        let null = transfers
            .iter()
            .any(|(backend, _)| *backend == Backend::Null);
        println!(
            "transfer hashes of {} adapters{}: {}",
            transfers.len(),
            if null { ", null included" } else { "" },
            if agree { "identical" } else { "DIFFERENT" }
        );
        ok &= agree;
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let null_only = std::env::args().any(|arg| arg == "--null");
    match run(null_only) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("rhi_sim: {err}");
            ExitCode::FAILURE
        }
    }
}
