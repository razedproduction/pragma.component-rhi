# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning is [SemVer](https://semver.org/).

## [Unreleased]

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
