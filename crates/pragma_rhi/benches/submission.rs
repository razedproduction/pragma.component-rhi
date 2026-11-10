//! Submission (roadmap M5, DECISIONS #23): validating and executing on a null device a command
//! buffer of 1000 draws (pipeline compatibility, vertex ranges, attachments, usage scopes) or
//! 1000 buffer copies (usages, alignment, ranges), and a frame of `begin_frame` → submit →
//! `end_frame` with 1000 draws. The null backend runs no shaders and its copies are a few
//! bytes, so this is `pragma_rhi`'s own submission-time validation. Against the `HashMap`
//! reference.
//!
//! `cargo bench -p pragma_rhi --bench submission` (report: `target/criterion/report/index.html`).
#![allow(
    missing_docs,
    clippy::expect_used,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]

mod common;

use std::hint::black_box;

use common::{N, draw_scene, null_device, reference};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use pragma_rhi::{
    BufferDescriptor, BufferUsages, Color, CommandEncoder, RenderPassColorAttachment,
    RenderPassDescriptor,
};

fn submission(c: &mut Criterion) {
    let mut group = c.benchmark_group("submission");
    reference(&mut group);
    let mut device = null_device();
    let scene = draw_scene(&mut device, 192);
    let draws = |n: usize| {
        let mut encoder = CommandEncoder::new(Some("draws"));
        {
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
            for i in 0..n as u32 {
                let first = (i % 64) * 3;
                pass.draw(first..first + 3, 0..1);
            }
        }
        encoder.finish()
    };

    group.bench_with_input(BenchmarkId::new("submit_draws", N), &N, |b, &n| {
        b.iter_batched(
            || draws(n),
            |buffer| black_box(device.submit([buffer]).expect("submitted")),
            BatchSize::SmallInput,
        );
    });

    let copy = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let src = device
        .create_buffer(&BufferDescriptor {
            label: Some("src"),
            size: 1024,
            usage: copy,
        })
        .expect("a buffer");
    let dst = device
        .create_buffer(&BufferDescriptor {
            label: Some("dst"),
            size: 1024,
            usage: copy,
        })
        .expect("a buffer");
    group.bench_with_input(BenchmarkId::new("submit_copies", N), &N, |b, &n| {
        b.iter_batched(
            || {
                let mut encoder = CommandEncoder::new(Some("copies"));
                for i in 0..n as u64 {
                    encoder.copy_buffer_to_buffer(src, (i % 64) * 16, dst, ((i + 7) % 64) * 16, 16);
                }
                encoder.finish()
            },
            |buffer| black_box(device.submit([buffer]).expect("submitted")),
            BatchSize::SmallInput,
        );
    });

    group.bench_with_input(BenchmarkId::new("frame_draws", N), &N, |b, &n| {
        b.iter_batched(
            || draws(n),
            |buffer| {
                device.begin_frame().expect("a frame");
                black_box(device.submit([buffer]).expect("submitted"));
                device.end_frame().expect("ended");
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, submission);
criterion_main!(benches);
