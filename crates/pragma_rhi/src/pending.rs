//! Pipelines compiled off the frame thread, as pragma.core async tasks (DECISIONS.md #2).
//!
//! [`Device::compile_render_pipeline`](crate::Device::compile_render_pipeline) validates the
//! descriptor on the calling thread — every descriptor error is reported there, synchronously —
//! and hands only the backend compilation to the executor's blocking pool. The result is
//! installed on the device when the caller polls it, at a point of its choosing (usually the
//! start of a frame): the device itself is never touched from another thread.

use core::fmt;

use pragma_core::tasks::{AsyncError, AsyncHandle};

use crate::backend::{ComputePipelineObj, RenderPipelineObj};
use crate::pipeline::{ComputeKey, ComputePipelineInfo, RenderKey, RenderPipelineInfo};
use crate::{PipelineLayout, RhiError};

enum State<O> {
    /// Served from the pipeline cache: nothing to compile.
    Cached(O),
    Compiling(AsyncHandle<Result<O, RhiError>>),
    Taken,
}

/// What a finished compilation installs on the device.
pub(crate) struct Finished<K, I, O> {
    pub(crate) layout: PipelineLayout,
    pub(crate) key: K,
    pub(crate) info: I,
    pub(crate) obj: O,
    /// Whether the object was compiled now (and belongs in the cache) or came from it.
    pub(crate) compiled: bool,
}

pub(crate) struct Pending<K, I, O> {
    layout: PipelineLayout,
    parts: Option<(K, I)>,
    state: State<O>,
}

impl<K, I, O> Pending<K, I, O> {
    pub(crate) const fn cached(layout: PipelineLayout, key: K, info: I, obj: O) -> Self {
        Self {
            layout,
            parts: Some((key, info)),
            state: State::Cached(obj),
        }
    }

    pub(crate) const fn compiling(
        layout: PipelineLayout,
        key: K,
        info: I,
        task: AsyncHandle<Result<O, RhiError>>,
    ) -> Self {
        Self {
            layout,
            parts: Some((key, info)),
            state: State::Compiling(task),
        }
    }

    fn is_ready(&self) -> bool {
        match &self.state {
            State::Cached(_) | State::Taken => true,
            State::Compiling(task) => task.is_finished(),
        }
    }

    /// `None` while compiling; the result once, then [`AsyncError::Consumed`].
    pub(crate) fn poll(&mut self) -> Option<Result<Finished<K, I, O>, RhiError>> {
        if let State::Compiling(task) = &self.state
            && !task.is_finished()
        {
            return None;
        }
        Some(match core::mem::replace(&mut self.state, State::Taken) {
            State::Cached(obj) => self.finish(Ok(Ok(obj)), false),
            // Finished: `poll_ready` returns the result without waiting.
            State::Compiling(mut task) => match task.poll_ready() {
                Some(result) => self.finish(result, true),
                None => {
                    self.state = State::Compiling(task);
                    return None;
                }
            },
            State::Taken => Err(RhiError::Task(AsyncError::Consumed)),
        })
    }

    /// Blocks until the compilation finishes; only outside pool work (see
    /// [`AsyncHandle::wait`]), otherwise [`AsyncError::WouldBlock`].
    pub(crate) fn wait(mut self) -> Result<Finished<K, I, O>, RhiError> {
        match core::mem::replace(&mut self.state, State::Taken) {
            State::Cached(obj) => self.finish(Ok(Ok(obj)), false),
            State::Compiling(task) => {
                let result = task.wait();
                self.finish(result, true)
            }
            State::Taken => Err(RhiError::Task(AsyncError::Consumed)),
        }
    }

    fn finish(
        &mut self,
        result: Result<Result<O, RhiError>, AsyncError>,
        compiled: bool,
    ) -> Result<Finished<K, I, O>, RhiError> {
        let obj = result.map_err(RhiError::Task)??;
        let (key, info) = self
            .parts
            .take()
            .ok_or(RhiError::Task(AsyncError::Consumed))?;
        Ok(Finished {
            layout: self.layout,
            key,
            info,
            obj,
            compiled,
        })
    }
}

impl<K, I, O> fmt::Debug for Pending<K, I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pending")
            .field("layout", &self.layout)
            .field("ready", &self.is_ready())
            .finish_non_exhaustive()
    }
}

/// A render pipeline being compiled on a pragma.core executor's blocking pool; returned by
/// [`Device::compile_render_pipeline`](crate::Device::compile_render_pipeline) and turned into
/// a [`RenderPipeline`](crate::RenderPipeline) by
/// [`Device::poll_render_pipeline`](crate::Device::poll_render_pipeline) or
/// [`Device::wait_render_pipeline`](crate::Device::wait_render_pipeline). Dropping it discards
/// the result.
#[derive(Debug)]
#[must_use = "a compiled pipeline is installed only by polling or waiting on it"]
pub struct PendingRenderPipeline(
    pub(crate) Pending<RenderKey, RenderPipelineInfo, RenderPipelineObj>,
);

impl PendingRenderPipeline {
    /// Whether polling would return a result now.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.0.is_ready()
    }
}

/// A compute pipeline being compiled; see [`PendingRenderPipeline`].
#[derive(Debug)]
#[must_use = "a compiled pipeline is installed only by polling or waiting on it"]
pub struct PendingComputePipeline(
    pub(crate) Pending<ComputeKey, ComputePipelineInfo, ComputePipelineObj>,
);

impl PendingComputePipeline {
    /// Whether polling would return a result now.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.0.is_ready()
    }
}
