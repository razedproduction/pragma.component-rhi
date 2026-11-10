//! The golden-image suite (roadmap M5, DECISIONS #20).
//!
//! A fixed set of small scenes — interpolated colors, alpha blending, depth testing with
//! back-face culling, a 4x multisampled resolve, a sampled texture drawn with an index buffer,
//! instances, a viewport and a scissor, an sRGB target, compute shaders writing a storage
//! texture and a storage buffer — renders on the software adapters CI has: **WARP** (D3D12,
//! Windows) and **lavapipe** (Vulkan, Mesa's llvmpipe, Linux). Each scene's readback is hashed
//! (FNV-1a, 64 bits) and compared with the hash committed for that adapter in `golden.txt`
//! next to this file. Rasterizers may legitimately differ in the last bit, so every adapter has
//! its own hashes; within one adapter any change is a failure. Every scene also runs twice on
//! fresh resources and must give the same bytes both times.
//!
//! ```text
//! cargo run -p pragma_rhi --example rhi_golden                     # every software adapter found
//! cargo run -p pragma_rhi --example rhi_golden -- --adapter warp   # WARP must be present
//! cargo run -p pragma_rhi --example rhi_golden -- --adapter lavapipe --bless   # rewrite its hashes
//! cargo run -p pragma_rhi --example rhi_golden -- --dump target/golden        # images on failure
//! ```
//!
//! `--dump DIR` writes, when a hash differs or is missing, every scene image of that adapter as
//! a PPM file and the `golden.txt` the run would bless, so CI can upload them. Hardware adapters
//! run the scenes too, but only for information: their hashes are printed, never compared.
// An example: indices into fixed scene arrays and small integer arithmetic.
#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pragma_rhi::{
    Adapter, Backend, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, CommandEncoder, ComputePassDescriptor, ComputePipelineDescriptor,
    DepthStencilState, Device, DeviceDescriptor, DeviceType, Extent3d, Face, FilterMode,
    FragmentState, IndexFormat, Instance, InstanceDescriptor, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, StoreOp,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureDescriptor, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

type Res<T> = Result<T, Box<dyn Error>>;

/// The side of every scene image.
const SIZE: u32 = 64;
const GOLDEN: &str = "golden.txt";

/// What a scene produced: RGBA8 texels of a `SIZE` x `SIZE` image, or raw bytes.
struct Output {
    bytes: Vec<u8>,
    image: bool,
}

type Scene = fn(&mut Device) -> Res<Output>;

const SCENES: [(&str, Scene); 7] = [
    ("triangles", triangles),
    ("blend", blend),
    ("depth_cull", depth_cull),
    ("msaa_resolve", msaa_resolve),
    ("textured", textured),
    ("srgb", srgb),
    ("compute", compute),
];

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

// ---------------------------------------------------------------------------------------------
// Building blocks

fn module(device: &mut Device, label: &str, source: &str) -> Res<ShaderModule> {
    Ok(device.create_shader_module(&ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source),
    })?)
}

fn target(device: &mut Device, format: TextureFormat, samples: u32) -> Res<(Texture, TextureView)> {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("target"),
        size: extent(SIZE, SIZE),
        sample_count: samples,
        format,
        usage: if samples == 1 {
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC
        } else {
            TextureUsages::RENDER_ATTACHMENT
        },
        ..TextureDescriptor::default()
    })?;
    let view = device.create_texture_view(texture, &TextureViewDescriptor::default())?;
    Ok((texture, view))
}

fn vertices(device: &mut Device, usage: BufferUsages, bytes: &[u8]) -> Res<Buffer> {
    let len = bytes.len().next_multiple_of(4);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("vertices"),
        size: len as u64,
        usage: usage | BufferUsages::COPY_DST,
    })?;
    let mut padded = bytes.to_vec();
    padded.resize(len, 0);
    device.upload_buffer(buffer, 0, &padded)?;
    Ok(buffer)
}

fn read_image(device: &mut Device, texture: Texture) -> Res<Output> {
    let bytes = device.read_texture(TexelCopyTextureInfo::new(texture), extent(SIZE, SIZE))?;
    Ok(Output { bytes, image: true })
}

/// Vertices of colored triangles: a position (2 x f32) and a color (4 x unorm8).
fn colored(triangles: &[([f32; 6], [[u8; 4]; 3])]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (positions, colors) in triangles {
        for (vertex, color) in positions.chunks(2).zip(colors) {
            for value in vertex {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(color);
        }
    }
    bytes
}

const COLOR_SHADER: &str = r"
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

const COLOR_ATTRIBUTES: [VertexAttribute; 2] = [
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

/// The colored-triangle pipeline for a target format, blending and sample count.
fn color_pipeline(
    device: &mut Device,
    format: TextureFormat,
    blend: Option<BlendState>,
    samples: u32,
) -> Res<RenderPipeline> {
    let module = module(device, "color", COLOR_SHADER)?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
    Ok(device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("color"),
        layout,
        vertex: VertexState {
            module,
            entry_point: None,
            buffers: &[VertexBufferLayout {
                array_stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: &COLOR_ATTRIBUTES,
            }],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState {
            count: samples,
            ..MultisampleState::default()
        },
        fragment: Some(FragmentState {
            module,
            entry_point: None,
            targets: &[Some(ColorTargetState {
                format,
                blend,
                ..ColorTargetState::default()
            })],
        }),
    })?)
}

/// Draws colored triangles into a fresh single-sampled target and reads it back.
fn draw_colored(
    device: &mut Device,
    format: TextureFormat,
    blend: Option<BlendState>,
    clear: Color,
    triangles: &[([f32; 6], [[u8; 4]; 3])],
) -> Res<Output> {
    let pipeline = color_pipeline(device, format, blend, 1)?;
    let (texture, view) = target(device, format, 1)?;
    let geometry = vertices(device, BufferUsages::VERTEX, &colored(triangles))?;
    let mut encoder = CommandEncoder::new(Some("colored"));
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("colored"),
            color_attachments: &[Some(RenderPassColorAttachment::clear(view, clear))],
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, geometry, 0, None);
        pass.draw(0..3 * triangles.len() as u32, 0..1);
    }
    device.submit([encoder.finish()])?;
    read_image(device, texture)
}

// ---------------------------------------------------------------------------------------------
// Scenes

/// Three triangles with a color per vertex: interpolation and the rasterization rules on edges.
fn triangles(device: &mut Device) -> Res<Output> {
    draw_colored(
        device,
        TextureFormat::Rgba8Unorm,
        None,
        Color::new(0.1, 0.1, 0.1, 1.0),
        &[
            (
                [-0.9, -0.9, 0.9, -0.7, -0.2, 0.9],
                [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            ),
            (
                [0.1, 0.2, 0.95, 0.95, 0.3, 0.95],
                [[255, 255, 0, 255], [0, 255, 255, 255], [255, 0, 255, 255]],
            ),
            (
                [-1.0, 1.0, -0.6, 0.3, -0.3, 1.0],
                [[40, 80, 120, 255], [200, 160, 120, 255], [10, 250, 90, 255]],
            ),
        ],
    )
}

/// Overlapping translucent triangles with alpha blending.
fn blend(device: &mut Device) -> Res<Output> {
    draw_colored(
        device,
        TextureFormat::Rgba8Unorm,
        Some(BlendState::ALPHA_BLENDING),
        Color::new(0.0, 0.0, 0.0, 1.0),
        &[
            (
                [-0.8, -0.8, 0.8, -0.8, 0.0, 0.8],
                [[255, 0, 0, 160], [255, 0, 0, 160], [255, 0, 0, 160]],
            ),
            (
                [-0.8, 0.8, 0.8, 0.8, 0.0, -0.8],
                [[0, 0, 255, 96], [0, 255, 0, 200], [0, 0, 255, 32]],
            ),
        ],
    )
}

const DEPTH_SHADER: &str = r"
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VertexOut {
    return VertexOut(vec4<f32>(position, 1.0), color);
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
";

/// Intersecting triangles with a depth test, drawn near-first; a back-facing one is culled.
fn depth_cull(device: &mut Device) -> Res<Output> {
    let module = module(device, "depth", DEPTH_SHADER)?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default())?;
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Unorm8x4,
            offset: 12,
            shader_location: 1,
        },
    ];
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("depth"),
        layout,
        vertex: VertexState {
            module,
            entry_point: None,
            buffers: &[VertexBufferLayout {
                array_stride: 16,
                step_mode: VertexStepMode::Vertex,
                attributes: &attributes,
            }],
        },
        primitive: PrimitiveState {
            cull_mode: Some(Face::Back),
            ..PrimitiveState::default()
        },
        depth_stencil: Some(DepthStencilState::default()),
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module,
            entry_point: None,
            targets: &[Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))],
        }),
    })?;
    // (x, y, depth) per vertex, counter-clockwise unless noted.
    let triangles: [([f32; 9], [u8; 4]); 4] = [
        // Tilted in depth: from 0.1 on the left to 0.9 on the right.
        (
            [-0.9, -0.6, 0.1, 0.9, -0.6, 0.9, 0.0, 0.8, 0.5],
            [255, 64, 0, 255],
        ),
        // Tilted the other way; the two cross in the middle.
        (
            [-0.9, 0.6, 0.9, 0.0, -0.8, 0.5, 0.9, 0.6, 0.1],
            [0, 128, 255, 255],
        ),
        // Behind everything.
        (
            [-1.0, -1.0, 0.95, 1.0, -1.0, 0.95, 1.0, 1.0, 0.95],
            [60, 60, 60, 255],
        ),
        // Clockwise: culled.
        (
            [-0.5, -0.5, 0.0, 0.0, 0.5, 0.0, 0.5, -0.5, 0.0],
            [255, 255, 255, 255],
        ),
    ];
    let mut bytes = Vec::new();
    for (positions, color) in triangles {
        for vertex in positions.chunks(3) {
            for value in vertex {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(color);
        }
    }
    let geometry = vertices(device, BufferUsages::VERTEX, &bytes)?;
    let (texture, view) = target(device, TextureFormat::Rgba8Unorm, 1)?;
    let depth = device.create_texture(&TextureDescriptor {
        label: Some("depth"),
        size: extent(SIZE, SIZE),
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        ..TextureDescriptor::default()
    })?;
    let depth_view = device.create_texture_view(depth, &TextureViewDescriptor::default())?;
    let mut encoder = CommandEncoder::new(Some("depth"));
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("depth"),
            color_attachments: &[Some(RenderPassColorAttachment::clear(
                view,
                Color::new(0.0, 0.0, 0.0, 1.0),
            ))],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, geometry, 0, None);
        pass.draw(0..12, 0..1);
    }
    device.submit([encoder.finish()])?;
    read_image(device, texture)
}

/// Thin triangles rendered with 4 samples per pixel and resolved.
fn msaa_resolve(device: &mut Device) -> Res<Output> {
    let format = TextureFormat::Rgba8Unorm;
    let pipeline = color_pipeline(device, format, None, 4)?;
    let (_, multisampled) = target(device, format, 4)?;
    let (resolved, resolve_view) = target(device, format, 1)?;
    let geometry = vertices(
        device,
        BufferUsages::VERTEX,
        &colored(&[
            (
                [-0.95, -0.9, 0.95, -0.75, 0.9, -0.6],
                [
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 128, 0, 255],
                ],
            ),
            (
                [-0.9, -0.95, -0.7, 0.95, -0.85, 0.9],
                [[0, 255, 128, 255], [0, 255, 128, 255], [0, 64, 255, 255]],
            ),
            (
                [-0.5, 0.9, 0.9, -0.3, 0.6, 0.7],
                [[255, 0, 80, 255], [80, 0, 255, 255], [255, 255, 0, 255]],
            ),
        ]),
    )?;
    let mut encoder = CommandEncoder::new(Some("msaa"));
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("msaa"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: multisampled,
                resolve_target: Some(resolve_view),
                ops: Operations {
                    load: LoadOp::Clear(Color::new(0.05, 0.05, 0.1, 1.0)),
                    store: StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, geometry, 0, None);
        pass.draw(0..9, 0..1);
    }
    device.submit([encoder.finish()])?;
    read_image(device, resolved)
}

const TEXTURED_SHADER: &str = r"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@group(0) @binding(2) var bilinear: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance: u32,
}

@vertex
fn vs(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @builtin(instance_index) instance: u32,
) -> VertexOut {
    let offset = vec2<f32>(f32(instance) * 0.5 - 0.25, f32(instance) * 0.25 - 0.125);
    return VertexOut(vec4<f32>(position + offset, 0.0, 1.0), uv, instance);
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
    let sharp = textureSample(image, nearest, input.uv);
    let blurred = textureSample(image, bilinear, input.uv);
    return select(blurred, sharp, input.instance == 0u);
}
";

fn textured_layout(device: &mut Device) -> Res<BindGroupLayout> {
    let sampler = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
    };
    Ok(device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("textured"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            },
            sampler(1),
            sampler(2),
        ],
    })?)
}

/// A checkerboard texture on two instances of an indexed quad — one sampled nearest, one
/// bilinear — inside a viewport, clipped by a scissor rectangle.
fn textured(device: &mut Device) -> Res<Output> {
    let module = module(device, "textured", TEXTURED_SHADER)?;
    let group_layout = textured_layout(device)?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("textured"),
        bind_group_layouts: &[group_layout],
    })?;
    let attributes = [
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 8,
            shader_location: 1,
        },
    ];
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("textured"),
        layout,
        vertex: VertexState {
            module,
            entry_point: None,
            buffers: &[VertexBufferLayout {
                array_stride: 16,
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

    // An 8 x 8 checkerboard with a gradient, uploaded with padded rows.
    let image = device.create_texture(&TextureDescriptor {
        label: Some("checker"),
        size: extent(8, 8),
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        ..TextureDescriptor::default()
    })?;
    let stride: usize = 8 * 4 + 16;
    let mut texels = vec![0_u8; stride * 8];
    for y in 0..8 {
        for x in 0..8 {
            let on = (x + y) % 2 == 0;
            let at = y * stride + x * 4;
            texels[at..at + 4].copy_from_slice(&[
                if on { 255 } else { (x * 32) as u8 },
                (y * 32) as u8,
                if on { 64 } else { 200 },
                255,
            ]);
        }
    }
    device.upload_texture(
        TexelCopyTextureInfo::new(image),
        &texels,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(stride as u32),
            rows_per_image: None,
        },
        extent(8, 8),
    )?;
    let image_view = device.create_texture_view(image, &TextureViewDescriptor::default())?;
    let nearest = device.create_sampler(&SamplerDescriptor::default())?;
    let bilinear = device.create_sampler(&SamplerDescriptor {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..SamplerDescriptor::default()
    })?;
    let group = device.create_bind_group(
        group_layout,
        &BindGroupDescriptor {
            label: Some("textured"),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(image_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(nearest),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(bilinear),
                },
            ],
        },
    )?;

    // A quad of four vertices (position, uv) and six indices.
    let quad: [[f32; 4]; 4] = [
        [-0.6, -0.6, 0.0, 1.0],
        [0.6, -0.6, 1.0, 1.0],
        [0.6, 0.6, 1.0, 0.0],
        [-0.6, 0.6, 0.0, 0.0],
    ];
    let quad_bytes: Vec<u8> = quad
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let geometry = vertices(device, BufferUsages::VERTEX, &quad_bytes)?;
    let indices: Vec<u8> = [0_u16, 1, 2, 0, 2, 3]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let index_buffer = vertices(device, BufferUsages::INDEX, &indices)?;

    let (texture, view) = target(device, TextureFormat::Rgba8Unorm, 1)?;
    let mut encoder = CommandEncoder::new(Some("textured"));
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("textured"),
            color_attachments: &[Some(RenderPassColorAttachment::clear(
                view,
                Color::new(0.2, 0.0, 0.2, 1.0),
            ))],
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_vertex_buffer(0, geometry, 0, None);
        pass.set_index_buffer(index_buffer, IndexFormat::Uint16, 0, None);
        pass.set_viewport(4.0, 2.0, 56.0, 60.0, 0.0, 1.0);
        pass.set_scissor_rect(2, 6, 58, 50);
        pass.draw_indexed(0..6, 0, 0..2);
    }
    device.submit([encoder.finish()])?;
    read_image(device, texture)
}

/// A gradient written to an sRGB target: the encoding on store.
fn srgb(device: &mut Device) -> Res<Output> {
    draw_colored(
        device,
        TextureFormat::Rgba8UnormSrgb,
        None,
        Color::new(0.5, 0.5, 0.5, 1.0),
        &[
            (
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0],
                [[0, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255]],
            ),
            (
                [1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
                [[255, 255, 255, 255], [0, 0, 255, 255], [128, 128, 128, 255]],
            ),
        ],
    )
}

const COMPUTE_SHADER: &str = r"
@group(0) @binding(0) var image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

fn mix_bits(x: u32) -> u32 {
    var h = x * 0x9e3779b9u;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    return h ^ (h >> 13u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(image);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let h = mix_bits(id.y * size.x + id.x);
    let color = vec4<f32>(
        f32(id.x) / 63.0,
        f32(id.y) / 63.0,
        f32(h & 255u) / 255.0,
        1.0,
    );
    textureStore(image, vec2<i32>(id.xy), color);
    data[id.y * size.x + id.x] = h;
}
";

/// A compute shader writing a storage texture and a storage buffer of integer hashes.
fn compute(device: &mut Device) -> Res<Output> {
    let module = module(device, "compute", COMPUTE_SHADER)?;
    let group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("compute"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
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
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute"),
        bind_group_layouts: &[group_layout],
    })?;
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("compute"),
        layout,
        module,
        entry_point: None,
    })?;
    let image = device.create_texture(&TextureDescriptor {
        label: Some("storage"),
        size: extent(SIZE, SIZE),
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        ..TextureDescriptor::default()
    })?;
    let image_view = device.create_texture_view(image, &TextureViewDescriptor::default())?;
    let words = u64::from(SIZE * SIZE) * 4;
    let data = device.create_buffer(&BufferDescriptor {
        label: Some("hashes"),
        size: words,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    })?;
    let group = device.create_bind_group(
        group_layout,
        &BindGroupDescriptor {
            label: Some("compute"),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(image_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: data,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        },
    )?;
    let mut encoder = CommandEncoder::new(Some("compute"));
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute"),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(SIZE / 8, SIZE / 8, 1);
    }
    device.submit([encoder.finish()])?;
    let mut bytes = device.read_texture(TexelCopyTextureInfo::new(image), extent(SIZE, SIZE))?;
    bytes.extend(device.read_buffer(data, 0, words)?);
    Ok(Output {
        bytes,
        image: false,
    })
}

// ---------------------------------------------------------------------------------------------
// The suite

/// The golden name of a software adapter; `None` for hardware.
fn golden_name(adapter: &Adapter) -> Option<&'static str> {
    let info = adapter.info();
    match (info.backend, info.device_type) {
        (Backend::Dx12, DeviceType::Cpu) => Some("warp"),
        (Backend::Vulkan, DeviceType::Cpu) if info.name.contains("llvmpipe") => Some("lavapipe"),
        _ => None,
    }
}

/// Every scene on a device: its hash and output; each scene runs twice and must agree.
fn run_scenes(device: &mut Device) -> Res<Vec<(&'static str, u64, Output)>> {
    let mut results = Vec::new();
    for (name, scene) in SCENES {
        let first = scene(device).map_err(|err| format!("{name}: {err}"))?;
        let second = scene(device).map_err(|err| format!("{name}: {err}"))?;
        if first.bytes != second.bytes {
            return Err(format!("{name}: two runs on one device differ").into());
        }
        results.push((name, hash(&first.bytes), first));
    }
    Ok(results)
}

/// `golden.txt`: (adapter, scene) → hash. A missing file is an empty table.
fn load(path: &Path) -> Res<BTreeMap<(String, String), u64>> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut map = BTreeMap::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let [adapter, scene, value] = parts[..] else {
            return Err(format!("{}: bad line `{line}`", path.display()).into());
        };
        let value = u64::from_str_radix(value, 16)?;
        map.insert((adapter.to_owned(), scene.to_owned()), value);
    }
    Ok(map)
}

fn render(map: &BTreeMap<(String, String), u64>) -> String {
    let mut text = String::from(
        "# Golden image hashes of rhi_golden (DECISIONS #20): <adapter> <scene> <FNV-1a 64>.\n\
         # Rewrite an adapter's lines with:\n\
         #   cargo run -p pragma_rhi --example rhi_golden -- --adapter <adapter> --bless\n",
    );
    for ((adapter, scene), value) in map {
        let _ = writeln!(text, "{adapter} {scene} {value:016x}");
    }
    text
}

/// A binary PPM of an RGBA8 image (alpha dropped).
fn ppm(bytes: &[u8]) -> Vec<u8> {
    let mut out = format!("P6\n{SIZE} {SIZE}\n255\n").into_bytes();
    for texel in bytes.chunks(4) {
        out.extend(&texel[..3]);
    }
    out
}

struct Options {
    adapter: Option<String>,
    bless: bool,
    dump: Option<PathBuf>,
}

fn options() -> Res<Options> {
    let mut options = Options {
        adapter: None,
        bless: false,
        dump: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--adapter" => options.adapter = Some(args.next().ok_or("--adapter needs a name")?),
            "--bless" => options.bless = true,
            "--dump" => options.dump = Some(args.next().ok_or("--dump needs a directory")?.into()),
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    Ok(options)
}

fn suite() -> Res<bool> {
    let options = options()?;
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("rhi_golden")
        .join(GOLDEN);
    let golden = load(&golden_path)?;
    let mut blessed = golden.clone();
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::VULKAN | Backends::DX12,
        ..InstanceDescriptor::default()
    });
    let mut ok = true;
    let mut found = false;
    for adapter in instance.adapters() {
        let golden_name = golden_name(&adapter);
        let label = format!("{:?} {}", adapter.info().backend, adapter.info().name);
        if options.adapter.is_some() && options.adapter.as_deref() != golden_name {
            continue;
        }
        let mut device = match adapter.request_device(&DeviceDescriptor::default()) {
            Ok(device) => device,
            Err(err) if golden_name.is_none() => {
                println!("{label}: skipped ({err})");
                continue;
            }
            Err(err) => return Err(format!("{label}: {err}").into()),
        };
        let results = run_scenes(&mut device)?;
        let Some(name) = golden_name else {
            println!("{label} (hardware, not compared):");
            for (scene, value, _) in &results {
                println!("  {scene:<14} {value:016x}");
            }
            continue;
        };
        found = true;
        println!("{label} -> golden `{name}`:");
        let mut adapter_ok = true;
        for (scene, value, _) in &results {
            let key = (name.to_owned(), (*scene).to_owned());
            let verdict = match golden.get(&key) {
                Some(expected) if expected == value => "ok".to_owned(),
                Some(expected) => format!("DIFFERENT (golden {expected:016x})"),
                None => "MISSING from golden.txt".to_owned(),
            };
            adapter_ok &= verdict == "ok";
            println!("  {scene:<14} {value:016x}  {verdict}");
            blessed.insert(key, *value);
        }
        if !adapter_ok && let Some(dir) = &options.dump {
            std::fs::create_dir_all(dir)?;
            for (scene, _, output) in &results {
                if output.image {
                    std::fs::write(dir.join(format!("{name}-{scene}.ppm")), ppm(&output.bytes))?;
                }
            }
            std::fs::write(dir.join(GOLDEN), render(&blessed))?;
            println!(
                "  images and the blessed golden.txt are in {}",
                dir.display()
            );
        }
        ok &= adapter_ok || options.bless;
    }
    if let Some(name) = &options.adapter
        && !found
    {
        println!("the `{name}` adapter is not present");
        return Ok(false);
    }
    if options.bless && blessed != golden {
        std::fs::write(&golden_path, render(&blessed))?;
        println!("blessed {}", golden_path.display());
    }
    Ok(ok)
}

fn main() -> ExitCode {
    match suite() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(err) => {
            eprintln!("rhi_golden: {err}");
            ExitCode::FAILURE
        }
    }
}
