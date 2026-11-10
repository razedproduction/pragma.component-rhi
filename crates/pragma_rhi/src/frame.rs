//! Frames in flight: submission fences, a bounded frame latency, per-frame resources and
//! deferred destruction (DECISIONS.md #17).
//!
//! Every [`Device::submit`] gets a [`SubmissionIndex`], increasing by one per submission. The
//! device learns which submissions the GPU has finished when it is polled
//! ([`Device::poll`], [`Device::wait`], [`Device::begin_frame`], blocking readback). A frame is
//! the work between [`Device::begin_frame`] and [`Device::end_frame`]; `begin_frame` blocks until
//! at most `frames_in_flight - 1` earlier frames are still running on the GPU, which bounds the
//! CPU's lead over the GPU. Buffers and textures destroyed while submissions that may use them
//! are in flight keep their memory until those submissions retire.

use std::collections::VecDeque;

use crate::backend::{BufferObj, TextureObj};
use crate::{BufferInfo, Device, RhiError};

/// The largest number of frames a device may keep in flight.
///
/// ```
/// use pragma_rhi::{DeviceDescriptor, Instance, InstanceDescriptor, MAX_FRAMES_IN_FLIGHT};
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// let too_many = DeviceDescriptor { frames_in_flight: MAX_FRAMES_IN_FLIGHT + 1, ..DeviceDescriptor::default() };
/// assert!(adapter.request_device(&too_many).is_err());
/// ```
pub const MAX_FRAMES_IN_FLIGHT: u32 = 8;

/// The fence of one submission: its position in the device's submission order, starting at 1.
///
/// Returned by [`Device::submit`]; a submission is complete once [`Device::is_complete`] says
/// so (after a poll), or after [`Device::wait`] returned for it or a later one.
///
/// ```
/// use pragma_rhi::{DeviceDescriptor, Instance, InstanceDescriptor, NullConfig, RhiError};
///
/// // A null device that reports a submission complete only two submissions later.
/// let instance = Instance::new(&InstanceDescriptor {
///     null: NullConfig { latency: 2, ..NullConfig::default() },
///     ..InstanceDescriptor::null()
/// });
/// let mut device = instance.adapters().remove(0).request_device(&DeviceDescriptor::default())?;
/// let first = device.submit([])?;
/// let second = device.submit([])?;
/// assert!(first < second);
/// device.poll()?;
/// assert!(!device.is_complete(first));
/// device.wait(first)?;
/// assert!(device.is_complete(first));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubmissionIndex(pub(crate) u64);

impl SubmissionIndex {
    /// The index as a number: 1 for a device's first submission, then one more per submission.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The frame [`Device::begin_frame`] opened.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let first = device.begin_frame()?;
/// device.end_frame()?;
/// let second = device.begin_frame()?;
/// device.end_frame()?;
/// // Frames count up; slots cycle through `frames_in_flight` (2 by default).
/// assert_eq!((first.index + 1, first.slot), (second.index, 0));
/// assert_eq!(second.slot, 1);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct FrameInfo {
    /// The frame's number on its device, starting at 0.
    pub index: u64,
    /// `index % frames_in_flight`: the set of per-frame resources ([`PerFrame`]) this frame
    /// may write, because the frame that used them last has retired.
    pub slot: usize,
}

/// One value per frame in flight — uniform buffers, staging rings, descriptor pools — picked
/// by the frame's [`FrameInfo::slot`]. A frame writes only its own slot, whose previous user
/// retired before [`Device::begin_frame`] returned, so it never races the GPU.
///
/// ```
/// use pragma_rhi::{DeviceDescriptor, Instance, InstanceDescriptor, PerFrame, RhiError};
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let mut counters = PerFrame::new(&device, |slot| slot * 10);
/// for _ in 0..4 {
///     let frame = device.begin_frame()?;
///     *counters.get_mut(frame) += 1;
///     device.end_frame()?;
/// }
/// assert_eq!(counters.iter().copied().collect::<Vec<_>>(), [2, 12]);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PerFrame<T> {
    first: T,
    rest: Vec<T>,
}

impl<T> PerFrame<T> {
    /// One value per frame in flight of `device`, made by `make(slot)`.
    pub fn new(device: &Device, mut make: impl FnMut(usize) -> T) -> Self {
        let count = device.frames_in_flight() as usize;
        Self {
            first: make(0),
            rest: (1..count).map(make).collect(),
        }
    }

    /// The number of values: the device's frames in flight.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rest.len().saturating_add(1)
    }

    /// Always `false`: there is a value for every frame in flight, and at least one frame.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    fn slot(&self, frame: FrameInfo) -> usize {
        frame.slot.checked_rem(self.len()).unwrap_or(0)
    }

    /// The value of `frame`'s slot.
    #[must_use]
    pub fn get(&self, frame: FrameInfo) -> &T {
        match self.slot(frame).checked_sub(1) {
            Some(index) => self.rest.get(index).unwrap_or(&self.first),
            None => &self.first,
        }
    }

    /// The value of `frame`'s slot, mutably.
    pub fn get_mut(&mut self, frame: FrameInfo) -> &mut T {
        match self.slot(frame).checked_sub(1) {
            Some(index) => self.rest.get_mut(index).unwrap_or(&mut self.first),
            None => &mut self.first,
        }
    }

    /// Every value, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        core::iter::once(&self.first).chain(&self.rest)
    }
}

/// A backend object whose handle is gone but which a submission in flight may still use.
#[derive(Debug)]
pub(crate) enum Garbage {
    Buffer(BufferObj, BufferInfo),
    Texture(TextureObj, u64),
    /// A presented or discarded swapchain texture: its memory belongs to the surface.
    SurfaceTexture(TextureObj, u64),
}

/// The device's view of the GPU's progress.
#[derive(Debug)]
pub(crate) struct Timeline {
    /// The last submission made (0: none).
    pub(crate) submitted: u64,
    /// The last submission the GPU is known to have finished.
    pub(crate) completed: u64,
    frames_in_flight: u32,
    /// The number of the next frame [`Device::begin_frame`] opens.
    next_frame: u64,
    /// Whether a frame is open.
    open: bool,
    /// The fence (last submission) of every ended frame that may still be in flight, oldest
    /// first.
    fences: VecDeque<u64>,
    /// Destroyed objects with the submission they must outlive, in destruction order (so in
    /// fence order).
    graveyard: VecDeque<(u64, Garbage)>,
}

impl Timeline {
    pub(crate) fn new(frames_in_flight: u32) -> Result<Self, RhiError> {
        if !(1..=MAX_FRAMES_IN_FLIGHT).contains(&frames_in_flight) {
            return Err(RhiError::frame("frames_in_flight must be between 1 and 8"));
        }
        Ok(Self {
            submitted: 0,
            completed: 0,
            frames_in_flight,
            next_frame: 0,
            open: false,
            fences: VecDeque::new(),
            graveyard: VecDeque::new(),
        })
    }

    pub(crate) fn next_submission(&self) -> Result<u64, RhiError> {
        self.submitted
            .checked_add(1)
            .ok_or_else(|| RhiError::limit("submissions per device", u64::MAX, u64::MAX - 1))
    }

    /// Keeps `garbage` until the last submission so far retires; returns it when that
    /// submission already has.
    fn bury(&mut self, garbage: Garbage) -> Result<Option<Garbage>, RhiError> {
        if self.submitted <= self.completed {
            return Ok(Some(garbage));
        }
        self.graveyard
            .try_reserve(1)
            .map_err(|_| RhiError::OutOfMemory { bytes: 64 })?;
        self.graveyard.push_back((self.submitted, garbage));
        Ok(None)
    }

    /// The garbage whose submissions have retired.
    fn exhume(&mut self) -> Vec<Garbage> {
        let mut freed = Vec::new();
        while let Some((fence, _)) = self.graveyard.front() {
            if *fence > self.completed {
                break;
            }
            if let Some((_, garbage)) = self.graveyard.pop_front() {
                freed.push(garbage);
            }
        }
        freed
    }
}

impl Device {
    /// How many frames may be in flight at once
    /// ([`DeviceDescriptor::frames_in_flight`](crate::DeviceDescriptor::frames_in_flight)).
    #[must_use]
    pub const fn frames_in_flight(&self) -> u32 {
        self.timeline.frames_in_flight
    }

    /// The last submission made on this device, if any.
    #[must_use]
    pub const fn last_submission(&self) -> Option<SubmissionIndex> {
        match self.timeline.submitted {
            0 => None,
            index => Some(SubmissionIndex(index)),
        }
    }

    /// Whether the GPU had finished `submission` when the device was last polled. Does not
    /// poll; see [`Self::poll`].
    #[must_use]
    pub const fn is_complete(&self, submission: SubmissionIndex) -> bool {
        submission.0 <= self.timeline.completed
    }

    /// The number of destroyed buffers and textures whose memory waits for a submission in
    /// flight to retire.
    #[must_use]
    pub fn deferred_destructions(&self) -> usize {
        self.timeline.graveyard.len()
    }

    /// Learns which submissions the GPU has finished, without blocking, and frees what waited
    /// for them: destroyed resources, upload chunks, finished readbacks' staging.
    ///
    /// # Errors
    ///
    /// [`RhiError::DeviceLost`], [`RhiError::Backend`].
    pub fn poll(&mut self) -> Result<(), RhiError> {
        self.alive()?;
        self.timeline.completed = self.backend.poll(None)?;
        self.retire()
    }

    /// Blocks until the GPU has finished `submission` (or the last submission, if `submission`
    /// is later), then frees what waited for it, as [`Self::poll`] does.
    ///
    /// # Errors
    ///
    /// As [`Self::poll`].
    pub fn wait(&mut self, submission: SubmissionIndex) -> Result<(), RhiError> {
        self.alive()?;
        let target = submission.0.min(self.timeline.submitted);
        self.timeline.completed = self.backend.poll(Some(target))?;
        self.retire()
    }

    /// Blocks until every submission so far has finished; see [`Self::wait`].
    ///
    /// # Errors
    ///
    /// As [`Self::poll`].
    ///
    /// ```
    /// # use pragma_rhi::*;
    /// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
    /// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
    /// device.submit([CommandEncoder::new(None).finish()])?;
    /// // Before shutdown or a resize that recreates everything.
    /// device.wait_idle()?;
    /// assert!(device.is_complete(device.last_submission().ok_or(RhiError::DeviceLost)?));
    /// # Ok::<(), RhiError>(())
    /// ```
    pub fn wait_idle(&mut self) -> Result<(), RhiError> {
        self.wait(SubmissionIndex(self.timeline.submitted))
    }

    /// Opens the next frame. First blocks until at most `frames_in_flight - 1` earlier frames
    /// are still in flight on the GPU — the frame latency is bounded, and the previous frame of
    /// the returned [`FrameInfo::slot`] has retired — then frees what retired.
    ///
    /// # Errors
    ///
    /// [`RhiError::Frame`] while a frame is open; otherwise as [`Self::poll`].
    ///
    /// ```
    /// # use pragma_rhi::*;
    /// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
    /// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
    /// let mut frame_buffers = PerFrame::new(&device, |_| None);
    /// for _ in 0..3 {
    ///     // Blocks while `frames_in_flight` frames are still on the GPU.
    ///     let frame = device.begin_frame()?;
    ///     // The previous frame of this slot has retired: its resources can be reused.
    ///     let uniforms = match frame_buffers.get_mut(frame) {
    ///         Some(buffer) => *buffer,
    ///         slot => *slot.insert(device.create_buffer(&BufferDescriptor {
    ///             label: Some("per-frame uniforms"),
    ///             size: 256,
    ///             usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    ///         })?),
    ///     };
    ///     device.write_buffer(uniforms, 0, &[frame.index as u8; 256])?;
    ///     device.submit([CommandEncoder::new(Some("frame")).finish()])?;
    ///     device.end_frame()?;
    /// }
    /// assert_eq!(device.live(ResourceKind::Buffer), 2);
    /// # Ok::<(), RhiError>(())
    /// ```
    pub fn begin_frame(&mut self) -> Result<FrameInfo, RhiError> {
        self.alive()?;
        if self.timeline.open {
            return Err(RhiError::frame("begin_frame while a frame is open"));
        }
        let frames = self.timeline.frames_in_flight as usize;
        // The frame being opened is in flight too: keep at most `frames - 1` older ones.
        while self.timeline.fences.len() >= frames {
            if let Some(fence) = self.timeline.fences.pop_front()
                && fence > self.timeline.completed
            {
                self.timeline.completed = self.backend.poll(Some(fence))?;
            }
        }
        self.poll()?;
        let completed = self.timeline.completed;
        self.timeline.fences.retain(|fence| *fence > completed);
        let index = self.timeline.next_frame;
        self.timeline.next_frame = index
            .checked_add(1)
            .ok_or_else(|| RhiError::limit("frames per device", u64::MAX, u64::MAX - 1))?;
        self.timeline.open = true;
        let slot = index
            .checked_rem(u64::from(self.timeline.frames_in_flight))
            .and_then(|slot| usize::try_from(slot).ok())
            .unwrap_or(0);
        Ok(FrameInfo { index, slot })
    }

    /// Closes the open frame: its fence is the last submission so far.
    ///
    /// # Errors
    ///
    /// [`RhiError::Frame`] without an open frame; [`RhiError::DeviceLost`].
    pub fn end_frame(&mut self) -> Result<(), RhiError> {
        self.alive()?;
        if !self.timeline.open {
            return Err(RhiError::frame("end_frame without an open frame"));
        }
        self.timeline
            .fences
            .try_reserve(1)
            .map_err(|_| RhiError::OutOfMemory { bytes: 8 })?;
        self.timeline.fences.push_back(self.timeline.submitted);
        self.timeline.open = false;
        Ok(())
    }

    /// Whether a frame is open.
    #[must_use]
    pub const fn in_frame(&self) -> bool {
        self.timeline.open
    }

    /// Records that submission `index` was made.
    pub(crate) const fn submitted(&mut self, index: u64) {
        self.timeline.submitted = index;
    }

    /// Destroys a backend object now, or when the last submission so far retires.
    pub(crate) fn bury(&mut self, garbage: Garbage) -> Result<(), RhiError> {
        if let Some(garbage) = self.timeline.bury(garbage)? {
            self.free(garbage);
        }
        Ok(())
    }

    fn free(&mut self, garbage: Garbage) {
        match garbage {
            Garbage::Buffer(obj, info) => self.backend.destroy_buffer(obj, &info),
            Garbage::Texture(obj, bytes) => self.backend.destroy_texture(obj, bytes),
            Garbage::SurfaceTexture(obj, bytes) => {
                self.backend.release_surface_texture(obj, bytes);
            }
        }
    }

    /// Frees everything whose submissions have retired, as far as the device knows.
    pub(crate) fn retire(&mut self) -> Result<(), RhiError> {
        for garbage in self.timeline.exhume() {
            self.free(garbage);
        }
        self.retire_uploads()?;
        self.sweep_readbacks()
    }
}
