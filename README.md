# pragma.component-rhi

[![CI](https://github.com/razedproduction/pragma.component-rhi/actions/workflows/ci.yml/badge.svg)](https://github.com/razedproduction/pragma.component-rhi/actions/workflows/ci.yml)
![MSRV](https://img.shields.io/badge/MSRV-1.88-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

**pragma.component-rhi** is a pure-Rust Render Hardware Interface for game engines and frameworks
built on [pragma.core](https://github.com/razedproduction/pragma.core).

It sits *on top of* the core: it owns no threads of its own. Command buffers are recorded as tasks
of the pragma.core frame graph on its work-stealing pool, so two recordings that write the same
GPU resource without an order are caught when the graph is built, not as corrupted frames at run
time. The hardware backend is [wgpu](https://wgpu.rs) (Vulkan, D3D12), kept behind the crate's own
types; a null backend runs everything except pixels without a GPU.

## Status

**M4 (0.4.0) — frames.** Adapters of Vulkan, D3D12 and the null backend; devices; buffers,
textures, views and samplers; shader modules validated by naga on every backend; bind groups,
pipeline layouts, render and compute pipelines with a pipeline cache and compilation on
pragma.core's async pool; command encoders with render and compute passes and copies, recorded in
parallel as frame-graph tasks and validated as a whole at submission. Frames: surfaces from any
`raw-window-handle` window (the crate opens none) with configure / acquire / present and
resize handling; submission fences and a bounded number of frames in flight; deferred
destruction; a bounded staging belt for uploads, uploads produced on pragma.core's blocking pool,
and asynchronous readback. The null backend executes copies, clears and resolves byte for byte
like a GPU and has a headless surface. `cargo run --example rhi_sim` runs the whole slice on every
adapter. Hardening comes next (M5). See the [roadmap](docs/roadmap.md) and the
[architecture](docs/architecture.md).

## Building

```bash
cargo test --workspace --all-features
```

The toolchain is pinned in `rust-toolchain.toml` (1.99.0); the minimum supported Rust version is
1.88, the same as pragma.core. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full check list.

## Reliability

The same rules as pragma.core, checked in CI on every PR:

- **Pure Rust.** No C/C++ compiled into the dependency tree (`cargo deny`, `tools/audit_workspace.py`).
- **No panics in the library.** `unwrap`/`expect`/`panic!`/unchecked indexing are denied by lints;
  failures, device loss included, are typed `#[non_exhaustive]` errors.
- **No `unsafe`.** The FFI of the GPU APIs stays inside wgpu ([DECISIONS #6, #8](DECISIONS.md)).
- **Layered.** Crates depend only on lower layers (`layers.toml`).

## Platforms

Windows (Vulkan, D3D12) and Linux (Vulkan), x86_64, tested in CI on the null backend and software
adapters.

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Decision log](DECISIONS.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

## License

[MIT](LICENSE) © 2026 razed
