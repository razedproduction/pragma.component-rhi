//! Command recording as pragma.core frame-graph tasks (architecture §4, DECISIONS.md #15).
//!
//! A [`CommandGraph`] is a pragma.core [`FrameGraph`] whose recording tasks each fill one
//! [`CommandEncoder`]. A task declares the resources it reads and writes like any frame-graph
//! task — by type, the way ECS systems declare components — so two recordings that write the
//! same resource without an order are [`GraphError::AmbiguousOrder`] when the graph is built,
//! not a race in a frame. Independent recordings run in parallel on the pool; the command
//! buffers come back in the graph's topological order, which does not depend on the number of
//! workers or on scheduling, so a frame submits the same commands on 1 or N threads.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pragma_core::graph::{
    FrameGraph, FrameGraphBuilder, GraphError, Resources, TaskConfig, TaskContext, TaskResult,
};
use pragma_core::jobs::ThreadPool;

use crate::{CommandBuffer, CommandEncoder, RhiError};

/// Where a recording task leaves its command buffer, or its error, for the frame.
type Slot = Arc<Mutex<Option<Result<CommandBuffer, RhiError>>>>;

fn lock(slot: &Slot) -> MutexGuard<'_, Option<Result<CommandBuffer, RhiError>>> {
    // A task never panics while holding the lock; a poisoned slot still holds valid data.
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Recording {
    name: String,
    slot: Slot,
}

/// Describes a [`CommandGraph`]; see [`CommandGraph::builder`].
///
/// ```
/// use pragma_rhi::{CommandGraph, RhiError};
///
/// struct Shadows;
/// struct Lighting;
///
/// let mut builder = CommandGraph::builder().ordered();
/// builder.task("cull", |t| t.writes::<Shadows>(), |_ctx| {});
/// builder.record("shadows", |t| t.reads::<Shadows>().writes::<Lighting>(), |_ctx, _encoder| Ok(()));
/// builder.record("lighting", |t| t.reads::<Lighting>(), |_ctx, _encoder| Ok(()));
/// let graph = builder.build()?;
/// assert_eq!(graph.recordings().collect::<Vec<_>>(), ["shadows", "lighting"]);
/// // Two recordings with the same name are rejected at build time.
/// let mut twice = CommandGraph::builder();
/// twice.record("a", |t| t, |_, _| Ok(())).record("a", |t| t, |_, _| Ok(()));
/// assert!(matches!(twice.build(), Err(RhiError::Graph(_))));
/// # Ok::<(), RhiError>(())
/// ```
pub struct CommandGraphBuilder {
    inner: FrameGraphBuilder,
    recordings: Vec<Recording>,
}

impl fmt::Debug for CommandGraphBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandGraphBuilder")
            .field(
                "recordings",
                &self
                    .recordings
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl CommandGraphBuilder {
    /// pragma.core's `ordered` mode: conflicting tasks without an explicit order run in
    /// registration order instead of failing the build.
    #[must_use]
    pub fn ordered(mut self) -> Self {
        self.inner = self.inner.ordered();
        self
    }

    /// Registers recording task `name`. `configure` declares its resource access and order as
    /// for any frame-graph task; `record` fills an encoder labeled `name`, reading the
    /// resources it declared through the [`TaskContext`] (a [`GraphError`] converts into
    /// [`RhiError::Graph`] with `?`).
    pub fn record<C, F>(&mut self, name: impl Into<String>, configure: C, record: F) -> &mut Self
    where
        C: FnOnce(TaskConfig) -> TaskConfig,
        F: Fn(&TaskContext<'_>, &mut CommandEncoder) -> Result<(), RhiError>
            + Send
            + Sync
            + 'static,
    {
        let name = name.into();
        let slot = Slot::default();
        let output = Arc::clone(&slot);
        self.inner.task(
            name.clone(),
            configure,
            move |ctx: &TaskContext<'_>| -> Result<(), GraphError> {
                let mut encoder = CommandEncoder::new(Some(ctx.name()));
                let result = record(ctx, &mut encoder).map(|()| encoder.finish());
                // A recording error is kept typed and reported by `CommandGraph::record`.
                *lock(&output) = Some(result);
                Ok(())
            },
        );
        self.recordings.push(Recording { name, slot });
        self
    }

    /// Registers a plain frame-graph task that records nothing (culling, uploads to
    /// prepare, …), ordered with the recordings by the same rules.
    pub fn task<C, F, R>(&mut self, name: impl Into<String>, configure: C, system: F) -> &mut Self
    where
        C: FnOnce(TaskConfig) -> TaskConfig,
        F: Fn(&TaskContext<'_>) -> R + Send + Sync + 'static,
        R: TaskResult,
    {
        self.inner.task(name, configure, system);
        self
    }

    /// Validates the graph: duplicate names, unknown dependencies, cycles and unordered access
    /// conflicts are errors here, before the first frame.
    ///
    /// # Errors
    ///
    /// [`RhiError::Graph`] with pragma.core's [`GraphError`].
    pub fn build(self) -> Result<CommandGraph, RhiError> {
        let graph = self.inner.build()?;
        let position: HashMap<&str, usize> = graph
            .order()
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect();
        let mut recordings = self.recordings;
        recordings.sort_by_key(|recording| {
            position
                .get(recording.name.as_str())
                .copied()
                .unwrap_or(usize::MAX)
        });
        Ok(CommandGraph { graph, recordings })
    }
}

/// Command buffers recorded in parallel as frame-graph tasks, returned in a fixed order.
///
/// ```
/// use pragma_core::graph::Resources;
/// use pragma_core::jobs::{PoolConfig, ThreadPool};
/// use pragma_rhi::{
///     Buffer, BufferDescriptor, BufferUsages, CommandGraph, DeviceDescriptor, Instance,
///     InstanceDescriptor, RhiError,
/// };
///
/// // Resources are declared by type, like ECS components.
/// struct Staging(Buffer);
/// struct Particles(Buffer);
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
/// let staging = device.create_buffer(&BufferDescriptor { label: None, size: 64, usage })?;
/// let particles = device.create_buffer(&BufferDescriptor { label: None, size: 64, usage })?;
/// device.write_buffer(staging, 0, &[1; 64])?;
///
/// let mut g = CommandGraph::builder();
/// g.record("clear", |t| t.writes::<Particles>(), |ctx, encoder| {
///     encoder.clear_buffer(ctx.read::<Particles>()?.0, 0, None);
///     Ok(())
/// });
/// g.record(
///     "upload",
///     |t| t.reads::<Staging>().writes::<Particles>().after("clear"),
///     |ctx, encoder| {
///         let (src, dst) = (ctx.read::<Staging>()?.0, ctx.read::<Particles>()?.0);
///         encoder.copy_buffer_to_buffer(src, 0, dst, 0, 32);
///         Ok(())
///     },
/// );
/// let mut graph = g.build()?;
///
/// let pool = ThreadPool::new(PoolConfig::default().with_workers(2))?;
/// let mut resources = Resources::new();
/// resources.insert(Staging(staging));
/// resources.insert(Particles(particles));
/// let buffers = graph.record(&pool, &mut resources)?;
/// assert_eq!(buffers.iter().map(|b| b.label()).collect::<Vec<_>>(), [Some("clear"), Some("upload")]);
/// device.submit(buffers)?;
/// assert_eq!(device.read_buffer(particles, 28, 8)?, [1, 1, 1, 1, 0, 0, 0, 0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CommandGraph {
    graph: FrameGraph,
    /// In the graph's topological order: the submission order.
    recordings: Vec<Recording>,
}

impl fmt::Debug for CommandGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandGraph")
            .field("recordings", &self.recordings().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl CommandGraph {
    /// A builder; see [`CommandGraphBuilder::record`].
    #[must_use]
    pub fn builder() -> CommandGraphBuilder {
        CommandGraphBuilder {
            inner: FrameGraph::builder(),
            recordings: Vec::new(),
        }
    }

    /// The recording tasks in submission order (the graph's topological order).
    pub fn recordings(&self) -> impl Iterator<Item = &str> {
        self.recordings
            .iter()
            .map(|recording| recording.name.as_str())
    }

    /// The underlying frame graph: every task's order, its edges, DOT and Mermaid exports.
    #[must_use]
    pub const fn frame_graph(&self) -> &FrameGraph {
        &self.graph
    }

    /// Runs one frame of the graph on `pool` over `resources` and returns the recorded
    /// command buffers in submission order, ready for [`Device::submit`](crate::Device::submit).
    ///
    /// # Errors
    ///
    /// [`RhiError::Graph`] when the frame graph fails (a missing or undeclared resource, a
    /// panicking task); otherwise the error of the first failed recording in submission
    /// order, so the result does not depend on scheduling.
    pub fn record(
        &mut self,
        pool: &ThreadPool,
        resources: &mut Resources,
    ) -> Result<Vec<CommandBuffer>, RhiError> {
        for recording in &self.recordings {
            *lock(&recording.slot) = None;
        }
        self.graph.run(pool, resources)?;
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(self.recordings.len())
            .map_err(|_| RhiError::OutOfMemory {
                bytes: self.recordings.len() as u64,
            })?;
        for recording in &self.recordings {
            match lock(&recording.slot).take() {
                Some(Ok(buffer)) => buffers.push(buffer),
                Some(Err(err)) => return Err(err),
                // A successful run runs every task.
                None => {
                    return Err(RhiError::backend(format!(
                        "recording task `{}` did not run",
                        recording.name
                    )));
                }
            }
        }
        Ok(buffers)
    }
}
