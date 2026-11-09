# pragma.component-rhi roadmap

The principle, as in pragma.core and pragma.component-ecs: **a vertical slice before breadth**.
A small headless example should run through device → resources → pipeline → commands recorded on
the frame graph → readback as early as possible, and every later milestone is verified on it.

Legend: ✅ done · 🚧 in progress · 📋 planned

| Milestone | Version | Essence | Status |
|---|---|---|---|
| M0 | 0.0.1 | Repository foundation | ✅ |
| M1 | 0.1.0 | Instance, devices and resources; the null backend | 📋 |
| M2 | 0.2.0 | Shaders, bind groups, pipelines | 📋 |
| M3 | 0.3.0 | Command recording on the pragma.core frame graph, vertical slice | 📋 |
| M4 | 0.4.0 | Surfaces, frames in flight, uploads, deferred destruction | 📋 |
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

## M1 — Instance, devices, resources (0.1.0)

- [ ] Add `wgpu`; audit its tree on both targets (`cargo deny`, `tools/audit_workspace.py`); every
      accepted exception recorded in `DECISIONS.md` (#5)
- [ ] **Decision:** handle model (generational ids into a resource table vs. reference-counted
      handles) and who owns destruction — see `docs/architecture.md` §3
- [ ] **Decision:** backend seam: an internal trait vs. an enum over {wgpu, null}
- [ ] `Instance` → `Adapter` enumeration (with info and limits) → `Device` + `Queue`, headless
- [ ] Null backend: same validation and handle rules as wgpu, no GPU, deterministic
- [ ] `Buffer`, `Texture`, `TextureView`, `Sampler` with typed descriptors; descriptors validated
      against the adapter's limits before reaching the backend → `RhiError`
- [ ] Device loss and out-of-memory reported as `RhiError`, never a panic or an abort
- [ ] Property tests: handles against a naive `HashMap` oracle (create / destroy / stale use)

## M2 — Shaders and pipelines (0.2.0)

- [ ] Shader modules from WGSL (and SPIR-V passthrough), validated by `naga` → `RhiError` with
      the source location
- [ ] Bind group layouts, bind groups, pipeline layouts
- [ ] Render and compute pipelines; a pipeline cache keyed by descriptor hash
- [ ] Pipeline compilation off the frame thread as a pragma.core async task

## M3 — Commands and the vertical slice (0.3.0)

- [ ] Command encoders, render and compute passes, copies; recorded in parallel as frame-graph
      tasks, submitted in a fixed order
- [ ] Resource access declared per recording task, so conflicting recordings without an order
      fail at build time (pragma.core `GraphError::AmbiguousOrder`)
- [ ] Buffer and texture readback
- [ ] Example `rhi_sim`: offscreen render + compute, readback, an image hash identical on 1 and N
      workers and on the null and software backends where they overlap

## M4 — Frames (0.4.0)

- [ ] Surfaces from `raw-window-handle` (the crate opens no windows), swapchain configure / acquire
      / present, resize and out-of-date handling
- [ ] Frames in flight: fences, per-frame resources, a bounded frame latency
- [ ] Upload ring (staging belt) and async uploads through pragma.core
- [ ] Deferred destruction: resources freed only after the GPU frame that used them retired

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
