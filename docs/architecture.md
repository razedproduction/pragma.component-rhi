# Architecture

Status: §2, §3, §3a, §4 and §5 are implemented (M1–M3); the rest is the intended design. Sections marked
*open* are decided in the milestone named there and recorded in `DECISIONS.md`.

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

The seam is an internal enum over the two backends (DECISIONS #10): no dynamic dispatch, and a
forgotten backend is a compile error. The front end validates; a backend only creates objects.
wgpu reports failures through error scopes, which every creation opens (DECISIONS #11).

## 3. Resources

Handles are **generational ids** (DECISIONS #9): device id + slot index + generation, `Copy`,
12 bytes, cheap to store in ECS components. Each device keeps one table per resource kind.
Destruction is explicit (`Device::destroy_*`); afterwards every copy of the handle returns
`RhiError::StaleHandle`, and a handle used on another device returns `RhiError::ForeignHandle`.
A slot whose generation would wrap is retired. A texture with live views cannot be destroyed.

Descriptors are validated against the **device's** limits (requested at creation, at most the
adapter's) before they reach the backend, so a wrong size or format is a typed error, not a
backend panic, and a program tested with the default limits behaves the same on a stronger GPU.
Table growth uses `try_reserve` → `RhiError::OutOfMemory`; the table slot is reserved before the
backend object is created, so a failure never has to roll a backend object back.

Resources that reference others pin them (DECISIONS #14): a bind group pins its layout and the
buffers, samplers and views it binds; a pipeline layout pins its bind group layouts; a pipeline
pins its pipeline layout. Destroying a pinned resource is `RhiError::ResourceInUse`.

Destruction is immediate before M4. From M4 it is **deferred**: a resource is freed only after the
last GPU frame that used it retired.

## 3a. Shaders and pipelines

Shaders are parsed and validated by naga inside `pragma_rhi`, before any backend (DECISIONS
#13), so the null backend rejects exactly what wgpu would and errors point at a line and column.
The reflection of each entry point — the resources it uses, its `@location` inputs and outputs,
its workgroup size — is kept with the module. Pipeline creation validates in two parts: the
descriptor's own rules against the device's limits, and the **shader interface** against the
pipeline layout, vertex buffers and color targets (`RhiError::ShaderInterface`). Only then does
a backend compile.

A compiled pipeline is cached per device under a key of everything it depends on (DECISIONS
#14); layouts and modules are named in keys by serials never reused on the device, and an entry
is dropped with its module or layout. `compile_*_pipeline` validates on the calling thread and
runs only the backend compilation on the pragma.core executor's blocking pool; `poll_*` installs
the result on the device at a point the frame thread chooses, so the device has a single owner
and no locks.

## 4. Command recording on the frame graph and submission

Each recording task declares which resources it reads and writes, like an ECS system declares
component access: by type, the engine's types that hold the handles. Tasks become frame-graph
tasks with those accesses, so pragma.core rejects two tasks that write the same resource without
an explicit order at build time. Command buffers are recorded in parallel and returned in the
graph's topological order, which keeps submission deterministic on any number of workers
(`CommandGraph`).

Recording does not touch the device (DECISIONS #15): a `CommandEncoder` appends plain commands
naming resources by handle, so it needs no lock and no `&Device` and cannot fail. `Device::submit`
then validates every command of every buffer against the live tables — handles, usages,
alignments and ranges, attachments, pipeline and bind group compatibility, dynamic offsets, draw
ranges, usage conflicts within a pass or dispatch — resolves handles to backend objects, and only
then hands the whole submission to the backend. A broken rule rejects all of it with
`RhiError::InvalidCommand`, naming the command buffer's label (the task) and the command. wgpu
encodes the validated commands on the frame thread; that translation is cheap next to recording
and is what M5's benchmarks will measure.

The null backend keeps the bytes of buffers and textures (DECISIONS #16) and executes queue
writes, copies, buffer clears, attachment clears, resolves and discards with the results a GPU
produces; draws and dispatches are validated and change nothing. Readback is blocking before
M4 (`read_buffer`, `read_texture`).

## 5. Failure model

GPU work fails in ways a CPU library does not: device loss (driver reset, removal), out of video
memory, an out-of-date surface. All of them are `RhiError` variants returned from the call that
observed them; none is a panic or an abort. After device loss every handle of that device returns
`RhiError::DeviceLost` until the engine creates a new device.

## 6. Reliability rules

The same as pragma.core: no `unwrap`/`expect`/`panic!`/unchecked indexing in library code (lints),
typed `#[non_exhaustive]` errors, pure-Rust dependencies, no `unsafe` (it stays inside wgpu,
DECISIONS #6), layers checked by `tools/audit_workspace.py`.
