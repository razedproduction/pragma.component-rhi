//! What the benchmarks share (DECISIONS #23): a null device, a small render scene and the
//! reference workload of the bench gate.
//!
//! Every benchmark runs on the null backend, so it measures `pragma_rhi`'s own work — handle
//! tables, descriptor validation, recording, submission-time validation — and not a driver.
// Not every bench uses every item.
#![allow(dead_code)]

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkGroup, BenchmarkId, measurement::WallTime};
use pragma_rhi::{
    Buffer, BufferDescriptor, BufferUsages, ColorTargetState, Device, DeviceDescriptor, Extent3d,
    FragmentState, Instance, InstanceDescriptor, MultisampleState, PipelineLayoutDescriptor,
    PrimitiveState, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    TextureDescriptor, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};

/// Operations per measurement.
pub const N: usize = 1000;

pub fn null_device() -> Device {
    Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}

/// The gate's reference: inserting and removing `N` entries of a `HashMap` — work of a similar
/// size that no change to `pragma_rhi` can affect, so its shift between two builds is runner
/// noise (`tools/bench_gate.py`).
pub fn reference(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.bench_with_input(BenchmarkId::new("hashmap_churn", N), &N, |b, &n| {
        let mut map: HashMap<u64, [u64; 4]> = HashMap::with_capacity(n);
        b.iter(|| {
            for i in 0..n as u64 {
                map.insert(black_box(i), [i; 4]);
            }
            for i in 0..n as u64 {
                black_box(map.remove(&i));
            }
        });
    });
}

const SHADER: &str = r"
@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.25, 1.0);
}
";

/// A render pipeline, a view of a 64 x 64 target and a vertex buffer of `vertices` positions.
#[derive(Clone, Copy)]
pub struct DrawScene {
    pub pipeline: RenderPipeline,
    pub view: TextureView,
    pub vertices: Buffer,
}

pub fn draw_scene(device: &mut Device, vertices: u64) -> DrawScene {
    let module = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: Some("bench"),
            source: ShaderSource::Wgsl(SHADER),
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
            label: Some("bench"),
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
    let target = device
        .create_texture(&TextureDescriptor {
            label: Some("target"),
            size: Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            ..TextureDescriptor::default()
        })
        .expect("a target");
    let view = device
        .create_texture_view(target, &TextureViewDescriptor::default())
        .expect("a view");
    let vertices = device
        .create_buffer(&BufferDescriptor {
            label: Some("vertices"),
            size: vertices * 8,
            usage: BufferUsages::VERTEX,
        })
        .expect("a vertex buffer");
    DrawScene {
        pipeline,
        view,
        vertices,
    }
}
