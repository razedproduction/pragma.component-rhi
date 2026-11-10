//! Command recording (roadmap M5, DECISIONS #23): 1000 draws or 1000 copies into one encoder,
//! and 1000 draws recorded by 8 frame-graph tasks in parallel (`CommandGraph` on a pragma.core
//! pool) — recording is plain data and touches no device (DECISIONS #15) — against the
//! `HashMap` reference.
//!
//! `cargo bench -p pragma_rhi --bench recording` (report: `target/criterion/report/index.html`).
#![allow(
    missing_docs,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

mod common;

use std::hint::black_box;

use common::{DrawScene, N, draw_scene, null_device, reference};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pragma_core::graph::Resources;
use pragma_core::jobs::{PoolConfig, ThreadPool};
use pragma_rhi::{
    Buffer, BufferDescriptor, BufferUsages, Color, CommandEncoder, CommandGraph,
    RenderPassColorAttachment, RenderPassDescriptor,
};

const TASKS: usize = 8;

fn record_draws(scene: &DrawScene, encoder: &mut CommandEncoder, draws: usize) {
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("draws"),
        color_attachments: &[Some(RenderPassColorAttachment::clear(
            scene.view,
            Color::new(0.0, 0.0, 0.0, 1.0),
        ))],
        depth_stencil_attachment: None,
    });
    pass.set_pipeline(scene.pipeline);
    pass.set_vertex_buffer(0, scene.vertices, 0, None);
    for i in 0..draws as u32 {
        let first = (i % 64) * 3;
        pass.draw(first..first + 3, 0..1);
    }
}

fn record_copies(src: Buffer, dst: Buffer, encoder: &mut CommandEncoder, copies: usize) {
    for i in 0..copies as u64 {
        encoder.copy_buffer_to_buffer(src, (i % 64) * 16, dst, ((i + 7) % 64) * 16, 16);
    }
}

fn recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("recording");
    reference(&mut group);
    let mut device = null_device();
    let scene = draw_scene(&mut device, 192);
    let buffer = |device: &mut pragma_rhi::Device| {
        device
            .create_buffer(&BufferDescriptor {
                label: None,
                size: 1024,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            })
            .expect("a buffer")
    };
    let (src, dst) = (buffer(&mut device), buffer(&mut device));

    group.bench_with_input(BenchmarkId::new("encode_draws", N), &N, |b, &n| {
        b.iter(|| {
            let mut encoder = CommandEncoder::new(Some("draws"));
            record_draws(&scene, &mut encoder, n);
            black_box(encoder.finish())
        });
    });

    group.bench_with_input(BenchmarkId::new("encode_copies", N), &N, |b, &n| {
        b.iter(|| {
            let mut encoder = CommandEncoder::new(Some("copies"));
            record_copies(src, dst, &mut encoder, n);
            black_box(encoder.finish())
        });
    });

    group.bench_with_input(BenchmarkId::new("graph_8_tasks", N), &N, |b, &n| {
        let pool = ThreadPool::new(PoolConfig::default().with_workers(4)).expect("a pool");
        let mut builder = CommandGraph::builder();
        for task in 0..TASKS {
            builder.record(
                format!("draws_{task}"),
                |t| t.reads::<DrawScene>(),
                move |ctx, encoder| {
                    let scene = ctx.read::<DrawScene>()?;
                    record_draws(&scene, encoder, n / TASKS);
                    Ok(())
                },
            );
        }
        let mut graph = builder.build().expect("a graph");
        let mut resources = Resources::new();
        resources.insert(scene);
        b.iter(|| black_box(graph.record(&pool, &mut resources).expect("recorded")));
    });
    group.finish();
}

criterion_group!(benches, recording);
criterion_main!(benches);
