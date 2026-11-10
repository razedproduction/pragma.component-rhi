# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning is [SemVer](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-11-10

### Added

- Golden-image suite (M5, DECISIONS #20): example `rhi_golden` renders seven scenes on the
  software adapters — WARP on Windows, lavapipe on Linux — and compares each readback's hash with
  `examples/rhi_golden/golden.txt` (hashes per adapter); `--adapter`, `--bless`, `--dump DIR`
  (PPM images and the blessed file). Run in CI on both platforms.
- Fuzzing (DECISIONS #21): a `fuzz/` workspace (cargo-fuzz) with the targets `handles`,
  `descriptors` and `transfers`; their oracles in `tests/common` also run as property tests
  (`tests/oracle.rs`). 60 s per target in PRs, 10 min nightly.
- Device-loss recovery tests (DECISIONS #22): every entry point after a loss with work in flight,
  recovery on a new device with identical results, a surface moving to the new device, blocking
  pool work across the loss.
- Benchmarks (DECISIONS #23): `resources`, `recording` and `submission` on the null backend
  (`cargo bench -p pragma_rhi`), a bench gate in PRs (`tools/bench_gate.py`, DECISIONS #24) and a
  nightly benchmark run.
- CI: `cargo semver-checks` against the PR base (DECISIONS #25); MSRV runs the tests, benches and
  doctests on 1.88; a `changes` job skips the heavy jobs on docs-only PRs.

### Changed

- The version is 0.5.0.
- `pragma_rhi` sets `#![recursion_limit = "256"]`: newer compilers warn (and will fail) proving
  wgpu-core's pipeline types `Send` at the default depth.
- The library target is no longer a bench target (`[lib] bench = false`), so
  `cargo bench -p pragma_rhi -- <criterion flags>` works.

## [0.4.0] - 2026-11-10

### Added

- Surfaces (M4, DECISIONS #18): `Instance::create_surface` from any window with
  `raw-window-handle` 0.6 window and display handles (the crate opens no windows) and
  `Instance::create_null_surface` for headless tests; `Adapter::surface_capabilities` /
  `Device::surface_capabilities`; `Device::configure_surface` validates a
  `SurfaceConfiguration` (format, usages, size, `PresentMode`, `CompositeAlphaMode`, frame
  latency) and resolves the `Auto` modes; `Device::acquire_frame` returns a `SurfaceFrame` whose
  texture and view are ordinary handles; `Device::present`, `Device::discard_frame`. New
  `RhiError::Surface(SurfaceError)` with `Outdated`, `Lost`, `Timeout`, `Occluded` and the
  protocol errors. The null surface simulates resizes and loss and keeps the presented texels.
- Frames in flight (DECISIONS #17): `Device::submit` returns a `SubmissionIndex`;
  `Device::poll`, `wait`, `wait_idle`, `is_complete`, `last_submission`;
  `Device::begin_frame` / `end_frame` bound the frames in flight
  (`DeviceDescriptor::frames_in_flight`, default 2, `MAX_FRAMES_IN_FLIGHT` = 8) and return a
  `FrameInfo`; `PerFrame<T>` for per-frame resources. New `RhiError::Frame`.
  `NullConfig::latency` keeps null submissions in flight for tests.
- Deferred destruction: `destroy_buffer` / `destroy_texture` free memory only after every
  submission made before them has retired; `Device::deferred_destructions`.
- Uploads (DECISIONS #19): a bounded staging belt (`DeviceDescriptor::upload`, `UploadConfig`)
  behind `Device::upload_buffer` and `Device::upload_texture` (any row stride, repacked);
  `Device::flush_uploads`, `Device::upload_stats` (`UploadStats`); `Device::upload_async` with
  `poll_upload` / `wait_upload` produces the bytes on a pragma.core executor's blocking pool
  (`UploadTarget`, `PendingUpload`).
- Asynchronous readback: `Device::read_buffer_async`, `read_texture_async`, `poll_readback`,
  `wait_readback` (`PendingReadback`); a dropped readback's staging is freed on the next poll.
- Tests: fences, frame pacing, deferred destruction, belt bounds and recycling, async uploads and
  readback on the null backend; uploads and readbacks byte-identical between the null backend and
  every GPU adapter present (checked on Vulkan, D3D12 and WARP); the surface protocol on null
  surfaces. Window surfaces were checked locally on real windows (Vulkan, D3D12, WARP).

### Changed

- `Device::submit` returns `Result<SubmissionIndex, RhiError>` instead of `Result<(), RhiError>`.
- `DeviceDescriptor` has `frames_in_flight` and `upload`; it no longer derives `Default` but
  implements it (`frames_in_flight: 2`). `NullConfig` has `latency`.
- `read_buffer` and `read_texture` submit pending uploads first; `read_texture` uses the
  asynchronous path and waits.
- `RhiError::Task` covers every blocking-pool task (pipeline compilations and async uploads).
- Example `rhi_sim` paces its frames, uploads through the belt and reads back asynchronously;
  its hashes are unchanged.

## [0.3.0] - 2026-11-09

### Added

- Command recording and submission (M3): `CommandEncoder` with `copy_buffer_to_buffer`,
  `clear_buffer`, `copy_buffer_to_texture`, `copy_texture_to_buffer`, `copy_texture_to_texture`,
  render passes (`RenderPass`: pipelines, bind groups with dynamic offsets, vertex and index
  buffers, viewport, scissor, `draw`, `draw_indexed`; color attachments with resolve targets, a
  depth/stencil attachment, load and store operations) and compute passes (`ComputePass`:
  `dispatch_workgroups`). Recording needs no device and cannot fail (DECISIONS #15).
- `Device::submit` validates every command of every command buffer before the backend sees any
  of it: handles, usages, alignments and ranges, attachments, pipeline/attachment and bind group
  layout compatibility, dynamic offsets, vertex and index ranges of draws, late
  `min_binding_size` checks, WebGPU usage scopes. A broken rule rejects the whole submission
  with the new `RhiError::InvalidCommand` (command buffer label, command, reason).
- `CommandGraph`: recording tasks on a pragma.core frame graph, with resource access declared by
  type, so conflicting recordings without an order fail at build time
  (`RhiError::Graph(GraphError::AmbiguousOrder)`); the command buffers come back in the graph's
  topological order on any number of workers. New `RhiError::Graph`.
- Queue writes (`Device::write_buffer`) and blocking readback (`Device::read_buffer`,
  `Device::read_texture`).
- The null backend keeps buffer and texture contents and executes writes, copies, clears,
  attachment clears, resolves and discards like a GPU; draws and dispatches change nothing
  (DECISIONS #16).
- `Limits::max_compute_workgroups_per_dimension` (WebGPU default 65535).
- Example `rhi_sim`: the vertical slice on every adapter, with hashes identical on 1 and N
  workers and transfer hashes identical across backends, the null one included; run in CI on
  the null backend.
- Tests: copy, pass, draw and dispatch rules; atomic rejection; stale handles at submission;
  transfers byte-identical between the null backend and every GPU adapter present (checked on
  Vulkan, D3D12 and WARP); draws and dispatches on GPUs; recording order on 1, 2 and 4 workers.

## [0.2.0] - 2026-11-09

### Added

- Shaders, bind groups and pipelines (M2): `ShaderModule` from WGSL or SPIR-V, parsed and
  validated by naga on every backend, with entry-point reflection (`shader_module_info`);
  WGSL errors carry line and column (`RhiError::Shader`, DECISIONS #13).
- `BindGroupLayout`, `BindGroup`, `PipelineLayout`, `RenderPipeline`, `ComputePipeline` handles
  with typed descriptors validated against WebGPU's rules and 22 new `Limits` fields
  (bind groups, bindings per stage, buffer binding sizes and alignments, vertex input, color
  targets, compute workgroups). Bind groups, pipeline layouts and pipelines pin what they
  reference (`RhiError::ResourceInUse`, DECISIONS #14).
- The shader interface of a pipeline is checked against its layout, vertex buffers, color
  targets and the limits (`RhiError::ShaderInterface`).
- A per-device pipeline cache keyed by the validated descriptor (`pipeline_cache_stats`,
  `clear_pipeline_cache`).
- Asynchronous compilation on a pragma.core `Executor`'s blocking pool:
  `compile_render_pipeline` / `compile_compute_pipeline` → `Pending*Pipeline`, installed with
  `poll_*` or `wait_*` (`RhiError::Task` for a failed task). `pragma_core`'s `async` feature
  is enabled.
- `RhiError::LimitBelowMinimum` for an offset alignment requested below the adapter's.
- Tests: the whole M2 chain on the null backend, cache hits and purges, async compilation, and
  the same scene on every GPU adapter present.

### Changed

- wgpu takes shaders as naga IR (`naga-ir` feature instead of `wgsl`); `naga` 30 is a direct
  dependency.

## [0.1.0] - 2026-11-09

### Added

- Instance, devices and resources (M1): `Instance` enumerates `Adapter`s (Vulkan and D3D12
  through wgpu 30, and a headless null backend with a device-memory budget); an `Adapter`
  creates a `Device` with requested `Limits`.
- `Buffer`, `Texture`, `TextureView`, `Sampler` as `Copy` generational handles with explicit
  destruction (DECISIONS #9); typed descriptors validated against the device's limits;
  `*_info` queries.
- `RhiError` variants `StaleHandle`, `ForeignHandle`, `InvalidDescriptor`, `LimitExceeded`,
  `ResourceInUse`, `Backend`. Device loss and out-of-memory are errors on every backend;
  `Device::simulate_device_loss` for recovery tests.
- Decisions #9–#12 (handles, the enum backend seam, the wgpu tree, descriptor style).
- Tests: descriptor rules, a property test of handles against a `HashMap` oracle, null-device
  behavior, and a smoke test on every GPU adapter present.

## [0.0.1] - 2026-11-09

### Added

- Repository foundation (M0): the `pragma_rhi` crate skeleton with `RhiError`, workspace lints,
  pinned toolchain, `deny.toml`, `layers.toml` with `tools/audit_workspace.py`, CI (Windows +
  Linux: fmt, clippy, test, doc, MSRV, cargo-deny, audit), documentation and the decision log
  (wgpu as the backend behind the crate's own types, DECISIONS #8).
