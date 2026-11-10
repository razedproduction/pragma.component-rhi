# pragma_rhi

[![crates.io](https://img.shields.io/crates/v/pragma_rhi.svg)](https://crates.io/crates/pragma_rhi)
[![docs.rs](https://img.shields.io/docsrs/pragma_rhi)](https://docs.rs/pragma_rhi)
![MSRV](https://img.shields.io/badge/MSRV-1.88-orange)

A pure-Rust Render Hardware Interface for game engines and frameworks built on
[pragma.core](https://github.com/razedproduction/pragma.core) (`pragma_core` 1.x).

It owns no threads: command buffers are recorded as tasks of the pragma.core frame graph on the
engine's thread pool, and two recordings that write the same resource without an order are
rejected when the graph is built. The device validates every descriptor against its limits and
every submission as a whole, so a mistake is a typed `RhiError` naming the task and the command —
not a driver crash. Resources are `Copy` generational handles with explicit destruction and
deferred freeing; frames in flight are bounded; uploads go through a bounded staging belt. The
GPU backend is [wgpu](https://wgpu.rs) (Vulkan, D3D12) behind the crate's own types, and a null
backend runs copies, clears, resolves, readback and presentation without a GPU, for tests and CI.

```rust
use pragma_rhi::{
    BufferDescriptor, BufferUsages, CommandEncoder, DeviceDescriptor, Instance,
    InstanceDescriptor, RhiError,
};

let instance = Instance::new(&InstanceDescriptor::null()); // or ::default() for the GPUs
let adapter = instance.adapters().remove(0);
let mut device = adapter.request_device(&DeviceDescriptor::default())?;

let usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
let src = device.create_buffer(&BufferDescriptor { label: Some("src"), size: 16, usage })?;
let dst = device.create_buffer(&BufferDescriptor { label: Some("dst"), size: 16, usage })?;
device.write_buffer(src, 0, &[7; 16])?;

let mut encoder = CommandEncoder::new(Some("copy")); // plain data: record on any thread
encoder.copy_buffer_to_buffer(src, 0, dst, 0, 16);
device.submit([encoder.finish()])?;
assert_eq!(device.read_buffer(dst, 0, 16)?, [7; 16]);
```

No `unsafe` in the crate, no panics (typed `#[non_exhaustive]` errors, device loss included),
pure-Rust dependencies. The crate documentation has an "Integrating into an engine" section with
a renderer skeleton; the design, the decision log, the golden-image suite and the benchmarks are
in the [repository](https://github.com/razedproduction/pragma.component-rhi).

License: MIT.
