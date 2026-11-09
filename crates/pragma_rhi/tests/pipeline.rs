//! Shaders, bind groups and pipelines (roadmap M2) on the null backend, and the same scene on
//! whatever GPU adapters exist.
#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_core::tasks::{AsyncConfig, AsyncError, Executor};
use pragma_rhi::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState,
    ComputePipelineDescriptor, Device, DeviceDescriptor, FragmentState, Instance,
    InstanceDescriptor, MultisampleState, PipelineLayout, PipelineLayoutDescriptor, PrimitiveState,
    RenderPipelineDescriptor, ResourceKind, RhiError, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStage, ShaderStages, TextureFormat, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState, VertexStepMode,
};

const SHADER: &str = r"
@group(0) @binding(0) var<uniform> tint: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

struct Out { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> }

@vertex
fn vs(@location(0) position: vec2<f32>) -> Out {
    return Out(vec4<f32>(position, 0.0, 1.0), tint);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return in.color;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = id.x;
}
";

fn null_device() -> Device {
    Instance::new(&InstanceDescriptor::null())
        .adapters()
        .remove(0)
        .request_device(&DeviceDescriptor::default())
        .expect("a null device")
}

const LAYOUT_ENTRIES: [BindGroupLayoutEntry; 2] = [
    BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX_FRAGMENT,
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
];

const ATTRIBUTES: [VertexAttribute; 1] = [VertexAttribute {
    format: VertexFormat::Float32x2,
    offset: 0,
    shader_location: 0,
}];

/// Everything a pipeline of [`SHADER`] needs.
struct Scene {
    module: ShaderModule,
    group_layout: BindGroupLayout,
    uniform: Buffer,
    storage: Buffer,
    group: BindGroup,
    layout: PipelineLayout,
}

fn scene(device: &mut Device) -> Result<Scene, RhiError> {
    let module = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("scene"),
        source: ShaderSource::Wgsl(SHADER),
    })?;
    let group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &LAYOUT_ENTRIES,
    })?;
    let uniform = device.create_buffer(&BufferDescriptor {
        label: Some("tint"),
        size: 256,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })?;
    let storage = device.create_buffer(&BufferDescriptor {
        label: Some("data"),
        size: 1024,
        usage: BufferUsages::STORAGE,
    })?;
    let group = device.create_bind_group(
        group_layout,
        &BindGroupDescriptor {
            label: None,
            entries: &[
                // Out of order on purpose: entries are matched by binding number.
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: storage,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: uniform,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        },
    )?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[group_layout],
    })?;
    Ok(Scene {
        module,
        group_layout,
        uniform,
        storage,
        group,
        layout,
    })
}

fn render_desc<'a>(
    scene: &Scene,
    buffers: &'a [VertexBufferLayout<'a>],
    targets: &'a [Option<ColorTargetState>],
) -> RenderPipelineDescriptor<'a> {
    RenderPipelineDescriptor {
        label: Some("triangle"),
        layout: scene.layout,
        vertex: VertexState {
            module: scene.module,
            entry_point: None,
            buffers,
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: scene.module,
            entry_point: Some("fs"),
            targets,
        }),
    }
}

fn vertex_buffers() -> [VertexBufferLayout<'static>; 1] {
    [VertexBufferLayout {
        array_stride: 8,
        step_mode: VertexStepMode::Vertex,
        attributes: &ATTRIBUTES,
    }]
}

fn compute_desc(scene: &Scene) -> ComputePipelineDescriptor<'static> {
    ComputePipelineDescriptor {
        label: Some("fill"),
        layout: scene.layout,
        module: scene.module,
        entry_point: None,
    }
}

#[test]
fn shader_errors_carry_the_source_location() {
    let mut device = null_device();
    let error = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(
                "@compute @workgroup_size(1)\nfn main() {\n    let x = ;\n}",
            ),
        })
        .expect_err("a syntax error");
    match error {
        RhiError::Shader {
            location: Some(at), ..
        } => assert_eq!(at.line, 3),
        other => panic!("expected a located shader error, got {other}"),
    }
    assert_eq!(device.live(ResourceKind::ShaderModule), 0);
}

#[test]
fn module_info_lists_the_entry_points() {
    let mut device = null_device();
    let module = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(SHADER),
        })
        .expect("a valid module");
    let info = device.shader_module_info(module).expect("live");
    let stages: Vec<_> = info
        .entry_points
        .iter()
        .map(|entry| (entry.name.as_str(), entry.stage))
        .collect();
    assert_eq!(
        stages,
        [
            ("vs", ShaderStage::Vertex),
            ("fs", ShaderStage::Fragment),
            ("cs", ShaderStage::Compute)
        ]
    );
    assert_eq!(info.entry_points[2].workgroup_size, Some([64, 1, 1]));
}

#[test]
fn pipelines_are_built_and_dependencies_pin_resources() {
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");
    let buffers = vertex_buffers();
    let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
    let render = device
        .create_render_pipeline(&render_desc(&scene, &buffers, &targets))
        .expect("a render pipeline");
    let compute = device
        .create_compute_pipeline(&compute_desc(&scene))
        .expect("a compute pipeline");
    let info = device.render_pipeline_info(render).expect("live");
    assert_eq!(info.color_formats, [Some(TextureFormat::Rgba8Unorm)]);
    assert_eq!(info.vertex_strides, [8]);
    assert_eq!(
        device
            .compute_pipeline_info(compute)
            .expect("live")
            .workgroup_size,
        Some([64, 1, 1])
    );
    // The bind group resolved the buffer ranges.
    let group = device.bind_group_info(scene.group).expect("live");
    assert_eq!(group.entries[0].binding, 0);
    assert!(matches!(
        group.entries[1].resource,
        BindingResource::Buffer(BufferBinding { size: Some(size), .. }) if size.get() == 1024
    ));

    let in_use = |result: Result<(), RhiError>, kind: ResourceKind, count: u32| {
        assert!(
            matches!(
                result,
                Err(RhiError::ResourceInUse { resource, dependents, .. })
                    if resource == kind && dependents == count
            ),
            "{kind}: {result:?}"
        );
    };
    in_use(
        device.destroy_buffer(scene.uniform),
        ResourceKind::Buffer,
        1,
    );
    in_use(
        device.destroy_pipeline_layout(scene.layout),
        ResourceKind::PipelineLayout,
        2,
    );
    // Used by the bind group and the pipeline layout.
    in_use(
        device.destroy_bind_group_layout(scene.group_layout),
        ResourceKind::BindGroupLayout,
        2,
    );

    // A shader module is not pinned: the pipelines keep their compiled code.
    device
        .destroy_shader_module(scene.module)
        .expect("unpinned");
    device.destroy_render_pipeline(render).expect("live");
    device.destroy_compute_pipeline(compute).expect("live");
    device
        .destroy_pipeline_layout(scene.layout)
        .expect("unpinned now");
    device.destroy_bind_group(scene.group).expect("live");
    device
        .destroy_bind_group_layout(scene.group_layout)
        .expect("unpinned now");
    device.destroy_buffer(scene.uniform).expect("unpinned now");
    device.destroy_buffer(scene.storage).expect("unpinned now");
    for kind in [
        ResourceKind::Buffer,
        ResourceKind::ShaderModule,
        ResourceKind::BindGroupLayout,
        ResourceKind::BindGroup,
        ResourceKind::PipelineLayout,
        ResourceKind::RenderPipeline,
        ResourceKind::ComputePipeline,
    ] {
        assert_eq!(device.live(kind), 0, "{kind}");
    }
}

#[test]
fn bind_groups_match_their_layout() {
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");
    let uniform = BindGroupEntry {
        binding: 0,
        resource: BindingResource::Buffer(BufferBinding {
            buffer: scene.uniform,
            offset: 0,
            size: None,
        }),
    };
    let reject = |device: &mut Device, entries: &[BindGroupEntry]| {
        let result = device.create_bind_group(
            scene.group_layout,
            &BindGroupDescriptor {
                label: None,
                entries,
            },
        );
        assert!(
            matches!(
                result,
                Err(RhiError::InvalidDescriptor {
                    resource: ResourceKind::BindGroup,
                    ..
                })
            ),
            "{result:?}"
        );
    };
    // Missing binding 1; binding 0 twice; the uniform buffer bound as storage (no STORAGE
    // usage); a range outside the buffer.
    reject(&mut device, &[uniform]);
    reject(&mut device, &[uniform, uniform]);
    reject(
        &mut device,
        &[
            uniform,
            BindGroupEntry {
                binding: 1,
                ..uniform
            },
        ],
    );
    reject(
        &mut device,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: scene.uniform,
                    offset: 512,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: scene.storage,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    );
    assert_eq!(device.live(ResourceKind::BindGroup), 1);
}

#[test]
fn shaders_must_fit_the_pipeline_layout() {
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");
    // A layout without the uniform the vertex stage reads.
    let empty = device
        .create_pipeline_layout(&PipelineLayoutDescriptor::default())
        .expect("an empty layout");
    let buffers = vertex_buffers();
    let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
    let desc = RenderPipelineDescriptor {
        layout: empty,
        ..render_desc(&scene, &buffers, &targets)
    };
    assert!(matches!(
        device.create_render_pipeline(&desc),
        Err(RhiError::ShaderInterface {
            stage: ShaderStage::Vertex,
            ..
        })
    ));
    // A vertex input without an attribute.
    let desc = render_desc(&scene, &[], &targets);
    assert!(matches!(
        device.create_render_pipeline(&desc),
        Err(RhiError::ShaderInterface {
            stage: ShaderStage::Vertex,
            ..
        })
    ));
    // An entry point that does not exist.
    let desc = ComputePipelineDescriptor {
        entry_point: Some("missing"),
        ..compute_desc(&scene)
    };
    assert!(matches!(
        device.create_compute_pipeline(&desc),
        Err(RhiError::ShaderInterface {
            stage: ShaderStage::Compute,
            ..
        })
    ));
    assert_eq!(device.live(ResourceKind::RenderPipeline), 0);
    assert_eq!(device.live(ResourceKind::ComputePipeline), 0);
}

#[test]
fn equal_descriptors_hit_the_pipeline_cache() {
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");
    let first = device
        .create_compute_pipeline(&compute_desc(&scene))
        .expect("compiled");
    // A different label is not a different pipeline.
    let second = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("again"),
            ..compute_desc(&scene)
        })
        .expect("cached");
    assert_ne!(first, second, "every creation returns its own handle");
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 1, 1));

    // Destroying the module drops its cache entries; the next creation compiles again.
    device.destroy_shader_module(scene.module).expect("live");
    assert_eq!(device.pipeline_cache_stats().entries, 0);
    let module = device
        .create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(SHADER),
        })
        .expect("a module");
    device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            module,
            ..compute_desc(&scene)
        })
        .expect("compiled");
    let stats = device.pipeline_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (1, 2, 1));
    device.clear_pipeline_cache();
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

#[test]
fn pipelines_compile_on_the_async_pool() {
    let pool = ThreadPool::new(PoolConfig::default().with_workers(1)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");

    let mut pending = device
        .compile_compute_pipeline(&compute_desc(&scene), &executor)
        .expect("valid");
    let pipeline = loop {
        if let Some(result) = device.poll_compute_pipeline(&mut pending) {
            break result.expect("compiled");
        }
        std::thread::yield_now();
    };
    assert!(device.compute_pipeline_info(pipeline).is_ok());
    // The result is handed out once.
    assert_eq!(
        device.poll_compute_pipeline(&mut pending),
        Some(Err(RhiError::Task(AsyncError::Consumed)))
    );

    // Descriptor errors are synchronous; a cache hit needs no task.
    assert!(matches!(
        device.compile_compute_pipeline(
            &ComputePipelineDescriptor {
                entry_point: Some("missing"),
                ..compute_desc(&scene)
            },
            &executor
        ),
        Err(RhiError::ShaderInterface { .. })
    ));
    let cached = device
        .compile_compute_pipeline(&compute_desc(&scene), &executor)
        .expect("valid");
    assert!(cached.is_ready());
    device
        .wait_compute_pipeline(cached)
        .expect("from the cache");

    let buffers = vertex_buffers();
    let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
    let pending = device
        .compile_render_pipeline(&render_desc(&scene, &buffers, &targets), &executor)
        .expect("valid");
    device.wait_render_pipeline(pending).expect("compiled");
    assert_eq!(device.live(ResourceKind::ComputePipeline), 2);
    assert_eq!(device.live(ResourceKind::RenderPipeline), 1);
}

#[test]
fn a_layout_destroyed_during_compilation_is_reported() {
    let pool = ThreadPool::new(PoolConfig::default().with_workers(1)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let mut device = null_device();
    let scene = scene(&mut device).expect("the scene");
    let pending = device
        .compile_compute_pipeline(&compute_desc(&scene), &executor)
        .expect("valid");
    device
        .destroy_pipeline_layout(scene.layout)
        .expect("not pinned by a pending compilation");
    assert!(matches!(
        device.wait_compute_pipeline(pending),
        Err(RhiError::StaleHandle {
            resource: ResourceKind::PipelineLayout,
            ..
        })
    ));
    assert_eq!(device.pipeline_cache_stats().entries, 0);
}

/// Runs on every Vulkan / D3D12 adapter of the machine (none on CI runners, where this passes
/// trivially): the null backend's scene must be accepted by wgpu's own validation too.
#[test]
fn gpu_adapters_build_the_scene() {
    let pool = ThreadPool::new(PoolConfig::default().with_workers(1)).expect("a pool");
    let executor = Executor::new(&pool, AsyncConfig::default()).expect("an executor");
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::VULKAN | Backends::DX12,
        ..InstanceDescriptor::default()
    });
    for adapter in instance.adapters() {
        let name = format!("{:?} {}", adapter.info().backend, adapter.info().name);
        let mut device = match adapter.request_device(&DeviceDescriptor::default()) {
            Ok(device) => device,
            Err(RhiError::Backend { message, .. }) => {
                eprintln!("skipping {name}: {message}");
                continue;
            }
            Err(other) => panic!("{name}: {other}"),
        };
        let scene = scene(&mut device).expect(&name);
        let buffers = vertex_buffers();
        let targets = [Some(ColorTargetState::from(TextureFormat::Rgba8Unorm))];
        device
            .create_render_pipeline(&render_desc(&scene, &buffers, &targets))
            .expect(&name);
        let pending = device
            .compile_compute_pipeline(&compute_desc(&scene), &executor)
            .expect(&name);
        device.wait_compute_pipeline(pending).expect(&name);
        eprintln!("checked {name}");
    }
}
