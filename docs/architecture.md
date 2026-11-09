# Architecture

Status: **intended design** (M0). Sections marked *open* are decided in the milestone named there
and recorded in `DECISIONS.md`.

## 1. Place in the stack

```
engine / game            frame loop, renderer, windowing, input, assets
pragma_ecs (optional)    world, components, systems
pragma_rhi               devices, resources, pipelines, commands     <- this repository
  └─ backend             wgpu (Vulkan, D3D12) | null (headless)
pragma_core (1.x)        thread pool, frame graph, async tasks
```

The RHI owns no threads (DECISIONS #2). The engine creates the `pragma_core::ThreadPool`; the RHI
records command buffers as tasks of a pragma.core `FrameGraph` on that pool and submits them in a
fixed order. It does not depend on pragma_ecs and pragma_ecs does not depend on it: an engine
connects the two.

## 2. Backends

`wgpu` is the only hardware backend (DECISIONS #8) and is an implementation detail: the public API
has its own descriptors, handles and errors, so a wgpu major release does not break `pragma_rhi`.
The **null backend** implements the same validation and handle rules without a GPU; tests and CI
use it for everything that is not pixels. Pixels are checked on software adapters (WARP, lavapipe).

*Open (M1):* the seam between the front end and the backends — an internal trait object, or an
enum over the two backends (no dynamic dispatch, exhaustive matching).

## 3. Resources — *open (M1)*

Candidates for handles:

- **Generational ids** into a per-device resource table (index + generation), as `Entity` in
  pragma_ecs: `Copy`, cheap to store in ECS components, a stale handle is detected and returned as
  `RhiError`. Destruction is explicit.
- **Reference-counted handles** (`Arc`-like): destruction when the last handle drops; simpler to
  use, but lifetimes become implicit and `Drop` cannot report errors.

Either way destruction is **deferred** (M4): a resource is freed only after the last GPU frame
that used it retired. Descriptors are validated against the adapter's limits before they reach the
backend, so a wrong size or format is a typed error, not a backend panic. Allocation sizes set by
input are bounded (`try_reserve` → `RhiError::OutOfMemory`).

## 4. Command recording on the frame graph

Each recording task declares which resources it reads and writes, like an ECS system declares
component access. Tasks become frame-graph tasks with those accesses, so pragma.core rejects two
tasks that write the same resource without an explicit order at build time. Command buffers are
recorded in parallel and submitted in the graph's topological order, which keeps submission
deterministic on any number of workers.

## 5. Failure model

GPU work fails in ways a CPU library does not: device loss (driver reset, removal), out of video
memory, an out-of-date surface. All of them are `RhiError` variants returned from the call that
observed them; none is a panic or an abort. After device loss every handle of that device returns
`RhiError::DeviceLost` until the engine creates a new device.

## 6. Reliability rules

The same as pragma.core: no `unwrap`/`expect`/`panic!`/unchecked indexing in library code (lints),
typed `#[non_exhaustive]` errors, pure-Rust dependencies, no `unsafe` (it stays inside wgpu,
DECISIONS #6), layers checked by `tools/audit_workspace.py`.
