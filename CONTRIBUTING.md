# Contributing

## Before a PR

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
python tools/audit_workspace.py .
cargo deny check
# On a machine with WARP (Windows) or lavapipe (Linux):
cargo run -p pragma_rhi --example rhi_golden -- --adapter warp
```

A change that moves rendered pixels on purpose updates the golden hashes with `--bless` (one
adapter at a time); the lavapipe lines can be taken from the `golden-lavapipe` CI artifact.
Fuzz targets (Linux, nightly): `cargo +nightly fuzz run handles` (also `descriptors`,
`transfers`); a crash reproduces as a `tests/oracle.rs` input through the same decoder.

## Module contract

A new crate or module is considered done when it has:

1. **A place in a layer**: an entry in `layers.toml`, dependencies only on lower layers.
2. **A typed error**: a `#[non_exhaustive]` enum via `thiserror`, no `unwrap`/`expect`/`panic!`/indexing.
   Device loss and out-of-memory are errors of the call that observed them.
3. **Tests**: unit tests and property tests for invariants (handle reuse, stale handles, descriptor
   validation against a naive oracle), runnable on the null backend without a GPU.
4. **Documentation**: `//!` in `lib.rs` and an example on every public type; `missing_docs` is clean.
5. **A benchmark** (`criterion`), if the module is on a hot path (command recording, resource
   creation and destruction, submission).

## The backend

`wgpu` is an implementation detail (DECISIONS #8): no `wgpu` type, trait or error appears in a
public signature. Every feature lands in the null backend with the same validation rules, so it
can be tested in CI.

## `unsafe`

Forbidden everywhere (`#![forbid(unsafe_code)]`, DECISIONS #6): the FFI of the GPU APIs stays
inside wgpu. Allowing it anywhere needs a decision record first; then the crate goes into
`unsafe_allowed` in `layers.toml`, every file gets a budget in `[unsafe_budget]`, every block a
`// SAFETY:` comment.

## Dependencies

- Pure Rust only (`deny.toml`). Before adding one, run `cargo tree -e build -i cc` and the audit.
- The version goes only into `[workspace.dependencies]`; in the crate, `{ workspace = true }`.
- Depend on the `pragma_core` facade, not on individual `pragma_*` crates of pragma.core.
- A new dependency or an exception to the rules means a row in `DECISIONS.md`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat(device): ...`, `fix(buffer): ...`,
`docs: ...`, `ci: ...`, `refactor(pipeline): ...`. A breaking change is `feat(pipeline)!: ...` plus
an entry in `CHANGELOG.md`.
