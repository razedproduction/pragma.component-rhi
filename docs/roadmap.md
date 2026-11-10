# pragma.component-rhi roadmap

The principle, as in pragma.core and pragma.component-ecs: **a vertical slice before breadth**.
A small headless example should run through device → resources → pipeline → commands recorded on
the frame graph → readback as early as possible, and every later milestone is verified on it.

Legend: ✅ done · 🚧 in progress · 📋 planned

| Milestone | Version | Essence | Status |
|---|---|---|---|
| M0 | 0.0.1 | Repository foundation | ✅ |
| M1 | 0.1.0 | Instance, devices and resources; the null backend | ✅ |
| M2 | 0.2.0 | Shaders, bind groups, pipelines | ✅ |
| M3 | 0.3.0 | Command recording on the pragma.core frame graph, vertical slice | ✅ |
| M4 | 0.4.0 | Surfaces, frames in flight, uploads, deferred destruction | ✅ |
| M5 | 0.5.0 | Hardening: golden images, fuzz, benches, semver | 📋 |
| M6 | 1.0.0 | Stable API, publication | 📋 |

---

## M0 — Foundation ✅

- [x] Workspace, edition 2024, pinned toolchain, MSRV 1.88, strict lints (no panics)
- [x] `deny.toml`, `layers.toml`, `tools/audit_workspace.py`
- [x] CI: Windows + Linux, fmt, clippy, test, doc, MSRV, deny, audit
- [x] `pragma_rhi` skeleton with `RhiError`, depending on `pragma_core` 1.x
- [x] Docs: README, architecture, roadmap, DECISIONS, CONTRIBUTING, CHANGELOG
- [x] **Decision:** wgpu as the backend, hidden behind own types (DECISIONS #8)

## M1 — Instance, devices, resources (0.1.0) ✅

- [x] Add `wgpu`; audit its tree on both targets (`cargo deny`, `tools/audit_workspace.py`); every
      accepted exception recorded in `DECISIONS.md` (#11: none was needed)
- [x] **Decision:** handle model — generational ids, explicit destruction (DECISIONS #9)
- [x] **Decision:** backend seam — an enum over {wgpu, null} (DECISIONS #10)
- [x] `Instance` → `Adapter` enumeration (with info and limits) → `Device`, headless; the wgpu
      queue stays inside the device until submission (M3)
- [x] Null backend: same validation and handle rules as wgpu, no GPU, deterministic, with a
      device-memory budget for out-of-memory tests
- [x] `Buffer`, `Texture`, `TextureView`, `Sampler` with typed descriptors; descriptors validated
      against the device's limits before reaching the backend → `RhiError`
- [x] Device loss and out-of-memory reported as `RhiError`, never a panic or an abort
      (`Device::simulate_device_loss` for recovery tests)
- [x] Property tests: handles against a naive `HashMap` oracle (create / destroy / stale use)

## M2 — Shaders and pipelines (0.2.0) ✅

- [x] Shader modules from WGSL and SPIR-V, parsed and validated by `naga` in `pragma_rhi` on every
      backend → `RhiError::Shader` with the source location; handed to wgpu as naga IR. SPIR-V is
      translated and validated like WGSL rather than passed through raw, which would need
      `unsafe` (DECISIONS #13)
- [x] Bind group layouts, bind groups, pipeline layouts, validated against WebGPU's rules and the
      device's limits; dependents pin what they reference (DECISIONS #14)
- [x] Render and compute pipelines; the shader interface (bindings, vertex inputs, color
      targets, inter-stage locations, workgroup size) checked against the descriptor →
      `RhiError::ShaderInterface`; a per-device pipeline cache keyed by the descriptor
- [x] Pipeline compilation off the frame thread on the pragma.core executor's blocking pool
      (`compile_*` → `poll_*` / `wait_*`)

## M3 — Commands and the vertical slice (0.3.0) ✅

- [x] **Decision:** recording is plain data, validated as a whole at submission (DECISIONS #15)
- [x] **Decision:** the null backend keeps resource contents and executes transfers
      (DECISIONS #16)
- [x] Command encoders, render and compute passes, copies; recorded in parallel as frame-graph
      tasks (`CommandGraph`), submitted in the graph's topological order
- [x] Resource access declared per recording task, so conflicting recordings without an order
      fail at build time (pragma.core `GraphError::AmbiguousOrder`)
- [x] Submission-time validation: handles, usages, alignments and ranges, attachments, pipeline
      and bind group compatibility, dynamic offsets, draw ranges against vertex and index
      buffers, late `min_binding_size` checks, usage scopes → `RhiError::InvalidCommand`
- [x] Queue writes; buffer and texture readback (`read_buffer`, `read_texture`)
- [x] Example `rhi_sim`: offscreen render + compute, readback, an image hash identical on 1 and N
      workers and on the null and software backends where they overlap (copies, clears)

## M4 — Frames (0.4.0) ✅

- [x] **Decision:** fences, frames in flight and deferred destruction (DECISIONS #17)
- [x] **Decision:** surfaces behind `raw-window-handle`, swapchain textures as table textures
      (DECISIONS #18)
- [x] **Decision:** a bounded staging belt, async uploads and async readback (DECISIONS #19)
- [x] Surfaces from `raw-window-handle` (the crate opens no windows), swapchain configure / acquire
      / present, resize and out-of-date handling; a null surface for headless tests; checked
      locally against real windows on Vulkan, D3D12 and WARP
- [x] Frames in flight: fences (`SubmissionIndex`, `poll` / `wait`), per-frame resources
      (`PerFrame`), a bounded frame latency (`begin_frame` / `end_frame`); the null backend's
      `latency` keeps submissions in flight for tests
- [x] Upload ring (staging belt) with a memory bound and recycling, async uploads through
      pragma.core's blocking pool, asynchronous buffer and texture readback
- [x] Deferred destruction: buffers and textures freed only after the submissions that may use
      them retired
- [x] Example `rhi_sim` runs on frames, the belt and async readback with unchanged hashes

## M5 — Hardening (0.5.0)

- [ ] Golden-image suite on software adapters (WARP, lavapipe) in CI
- [ ] Fuzzing of descriptor validation and handle operations against the oracle
- [ ] Benchmarks (`criterion`): resource churn, command recording, submission; bench gate
- [ ] Device-loss recovery tests on the null backend
- [ ] `cargo-semver-checks` in CI

## M6 — 1.0 (1.0.0)

- [ ] API review, rustdoc with examples, an "Integrating into an engine" section
- [ ] Publication on crates.io

---

## Out of scope

Windowing and input, the renderer itself (materials, meshes, scenes, lighting, a high-level
render graph), asset loading and texture/mesh formats, shader authoring beyond WGSL / SPIR-V
input, ECS integration, editor, the game loop (DECISIONS #1). Own native backends on `ash` /
`windows` are out of scope before 1.0 (DECISIONS #8). These belong to engines built on top.
