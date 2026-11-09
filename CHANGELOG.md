# Changelog

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning is [SemVer](https://semver.org/).

## [Unreleased]

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
