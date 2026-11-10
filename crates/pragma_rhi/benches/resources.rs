//! Resource churn (roadmap M5, DECISIONS #23): creating and destroying 1000 buffers, 1000
//! textures with a view each, and 1000 bind groups on a null device — descriptor validation,
//! the handle tables, dependency pinning and deferred destruction — against the `HashMap`
//! reference.
//!
//! `cargo bench -p pragma_rhi --bench resources` (report: `target/criterion/report/index.html`).
#![allow(
    missing_docs,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

mod common;

use std::hint::black_box;

use common::{N, null_device, reference};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pragma_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages,
    Extent3d, ShaderStages, TextureDescriptor, TextureFormat, TextureUsages, TextureViewDescriptor,
};

fn resources(c: &mut Criterion) {
    let mut group = c.benchmark_group("resources");
    reference(&mut group);

    group.bench_with_input(BenchmarkId::new("buffer_churn", N), &N, |b, &n| {
        let mut device = null_device();
        let mut buffers = Vec::with_capacity(n);
        b.iter(|| {
            for i in 0..n {
                let buffer = device
                    .create_buffer(&BufferDescriptor {
                        label: None,
                        size: 256 + (i as u64 % 16) * 64,
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    })
                    .expect("a buffer");
                buffers.push(buffer);
            }
            for buffer in buffers.drain(..) {
                device.destroy_buffer(black_box(buffer)).expect("destroyed");
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("texture_view_churn", N), &N, |b, &n| {
        let mut device = null_device();
        let mut textures = Vec::with_capacity(n);
        b.iter(|| {
            for i in 0..n {
                let side = 16 << (i % 4);
                let texture = device
                    .create_texture(&TextureDescriptor {
                        label: None,
                        size: Extent3d {
                            width: side,
                            height: side,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 3,
                        format: TextureFormat::Rgba8UnormSrgb,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                        ..TextureDescriptor::default()
                    })
                    .expect("a texture");
                let view = device
                    .create_texture_view(texture, &TextureViewDescriptor::default())
                    .expect("a view");
                textures.push((texture, view));
            }
            for (texture, view) in textures.drain(..) {
                device.destroy_texture_view(view).expect("destroyed");
                device
                    .destroy_texture(black_box(texture))
                    .expect("destroyed");
            }
        });
    });

    group.bench_with_input(BenchmarkId::new("bind_group_churn", N), &N, |b, &n| {
        let mut device = null_device();
        let layout = device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                }],
            })
            .expect("a layout");
        let uniforms: Vec<_> = (0..16)
            .map(|_| {
                device
                    .create_buffer(&BufferDescriptor {
                        label: None,
                        size: 256,
                        usage: BufferUsages::UNIFORM,
                    })
                    .expect("a buffer")
            })
            .collect();
        let mut groups = Vec::with_capacity(n);
        b.iter(|| {
            for i in 0..n {
                let group = device
                    .create_bind_group(
                        layout,
                        &BindGroupDescriptor {
                            label: None,
                            entries: &[BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: uniforms[i % uniforms.len()],
                                    offset: 0,
                                    size: None,
                                }),
                            }],
                        },
                    )
                    .expect("a bind group");
                groups.push(group);
            }
            for group in groups.drain(..) {
                device
                    .destroy_bind_group(black_box(group))
                    .expect("destroyed");
            }
        });
    });
    group.finish();
}

criterion_group!(benches, resources);
criterion_main!(benches);
