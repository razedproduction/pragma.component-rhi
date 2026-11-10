//! Uploads through a staging belt, asynchronous uploads and asynchronous readback
//! (DECISIONS.md #19).
//!
//! The belt is a ring of mapped `MAP_WRITE | COPY_SRC` chunks, bounded by
//! [`UploadConfig::capacity`]. [`Device::upload_buffer`] and [`Device::upload_texture`] copy the
//! caller's bytes into a chunk at once and record a copy into the destination; the copies run at
//! the start of the next submission. A chunk goes back into the ring when the submission that
//! read it retires; when the ring is full, an upload first reuses retired chunks, then waits for
//! the oldest chunk in flight. [`Device::upload_async`] produces the bytes on a pragma.core
//! executor's blocking pool (decoding, file reads) and uploads them when polled.
//!
//! [`Device::read_buffer_async`] and [`Device::read_texture_async`] copy into a `MAP_READ`
//! buffer as their own submission; [`Device::poll_readback`] returns the bytes once that
//! submission has retired and the buffer is mapped, without blocking the frame.

use core::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pragma_core::tasks::{AsyncError, AsyncHandle, Executor};

use crate::backend::{DeviceBackend, MapFlag, MapMode, MapState};
use crate::commands::{COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT};
use crate::device::BufferEntry;
use crate::frame::Garbage;
use crate::handle::Table;
use crate::resource::validate_buffer;
use crate::{
    Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder, Device, Extent3d,
    ResourceKind, RhiError, SubmissionIndex, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo,
};

const LABEL: &str = "pragma_rhi uploads";

/// Sizes the upload belt ([`DeviceDescriptor::upload`](crate::DeviceDescriptor::upload)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UploadConfig {
    /// The size of a staging chunk (1 MiB by default); larger uploads get a chunk of their
    /// own size.
    pub chunk_size: u64,
    /// The most staging memory the belt holds at once (64 MiB by default), and so the largest
    /// single upload. A full belt waits for the GPU before it grows.
    pub capacity: u64,
}

impl Default for UploadConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1 << 20,
            capacity: 64 << 20,
        }
    }
}

/// The upload belt's state ([`Device::upload_stats`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct UploadStats {
    /// Staging chunks allocated.
    pub chunks: usize,
    /// Staging memory allocated, in bytes (at most [`UploadConfig::capacity`]).
    pub allocated: u64,
    /// Bytes recorded for the next submission.
    pub pending: u64,
    /// Bytes uploaded through the belt over the device's life.
    pub uploaded: u64,
    /// How often an upload had to wait for the GPU because the belt was full.
    pub stalls: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkState {
    /// Mapped: the CPU may write from `cursor` on.
    Mapped,
    /// Read by submission `.0`; unmapped.
    InFlight(u64),
    /// Its submission retired; mapping again.
    Remapping,
}

#[derive(Debug)]
struct Chunk {
    buffer: Buffer,
    size: u64,
    cursor: u64,
    state: ChunkState,
    flag: MapFlag,
}

/// The device's staging belt.
#[derive(Debug)]
pub(crate) struct Belt {
    config: UploadConfig,
    chunks: Vec<Chunk>,
    allocated: u64,
    /// The copies recorded for the next submission.
    encoder: Option<CommandEncoder>,
    pending: u64,
    uploaded: u64,
    stalls: u64,
}

const fn round_up(value: u64, alignment: u64) -> Option<u64> {
    value.checked_next_multiple_of(alignment)
}

impl Belt {
    pub(crate) fn new(config: UploadConfig) -> Result<Self, RhiError> {
        if config.chunk_size == 0 || config.capacity == 0 {
            return Err(RhiError::invalid(
                ResourceKind::Buffer,
                "the upload belt's chunk_size and capacity must not be zero",
            ));
        }
        Ok(Self {
            config,
            chunks: Vec::new(),
            allocated: 0,
            encoder: None,
            pending: 0,
            uploaded: 0,
            stalls: 0,
        })
    }

    /// The recorded copies, to run first in the next submission.
    pub(crate) fn take_pending(&mut self) -> Option<CommandBuffer> {
        self.encoder.take().map(CommandEncoder::finish)
    }

    /// Puts back copies a rejected submission did not run.
    pub(crate) fn restore_pending(&mut self, uploads: CommandBuffer) {
        self.encoder = Some(CommandEncoder::from_buffer(uploads));
    }

    /// Forgets copies that can no longer run; their chunk space is reused.
    pub(crate) fn discard_pending(&mut self) {
        self.encoder = None;
        self.pending = 0;
        for chunk in &mut self.chunks {
            if chunk.state == ChunkState::Mapped {
                chunk.cursor = 0;
            }
        }
    }

    /// Unmaps every chunk written since the last submission: submission `index` reads them.
    pub(crate) fn unmap_pending(
        &mut self,
        backend: &mut DeviceBackend,
        buffers: &Table<Buffer, BufferEntry>,
        index: u64,
    ) {
        for chunk in &mut self.chunks {
            if chunk.state == ChunkState::Mapped && chunk.cursor > 0 {
                if let Ok(entry) = buffers.get(chunk.buffer) {
                    backend.unmap(&entry.obj);
                }
                chunk.state = ChunkState::InFlight(index);
            }
        }
        self.uploaded = self.uploaded.saturating_add(self.pending);
        self.pending = 0;
    }

    /// A mapped chunk with room for `len` bytes at `alignment`, partly used chunks first.
    fn fit(&self, len: u64, alignment: u64) -> Option<(usize, u64)> {
        let room = |chunk: &Chunk| {
            let at = round_up(chunk.cursor, alignment)?;
            (chunk.state == ChunkState::Mapped && at.checked_add(len)? <= chunk.size).then_some(at)
        };
        let used = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| chunk.cursor > 0);
        let idle = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, chunk)| chunk.cursor == 0);
        used.chain(idle)
            .find_map(|(index, chunk)| room(chunk).map(|at| (index, at)))
    }

    fn idle_chunk(&self) -> Option<usize> {
        self.chunks
            .iter()
            .position(|chunk| chunk.state == ChunkState::Mapped && chunk.cursor == 0)
    }

    fn oldest_in_flight(&self) -> Option<u64> {
        self.chunks
            .iter()
            .filter_map(|chunk| match chunk.state {
                ChunkState::InFlight(fence) => Some(fence),
                ChunkState::Mapped | ChunkState::Remapping => None,
            })
            .min()
    }
}

/// Staging buffers of readbacks still in progress, with a flag their [`PendingReadback`] sets
/// when dropped unread; abandoned ones are destroyed on the next poll.
#[derive(Debug, Default)]
pub(crate) struct Readbacks(Vec<(Buffer, Arc<AtomicBool>)>);

/// Where [`Device::upload_async`] puts the bytes it produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UploadTarget {
    /// A range of a buffer, as [`Device::upload_buffer`].
    Buffer {
        /// The buffer, with `COPY_DST`.
        buffer: Buffer,
        /// Where the bytes go; a multiple of 4.
        offset: u64,
    },
    /// A region of a texture, as [`Device::upload_texture`].
    Texture {
        /// The texture region's origin.
        texture: TexelCopyTextureInfo,
        /// How the produced bytes are laid out.
        layout: TexelCopyBufferLayout,
        /// The region's size.
        size: Extent3d,
    },
}

/// Bytes being produced on a pragma.core executor's blocking pool, to be uploaded when
/// [`Device::poll_upload`] (or [`Device::wait_upload`]) sees them. Dropping it discards the
/// upload.
#[must_use = "the bytes are uploaded only by polling or waiting on the upload"]
pub struct PendingUpload {
    target: UploadTarget,
    task: Option<AsyncHandle<Result<Vec<u8>, RhiError>>>,
}

impl fmt::Debug for PendingUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingUpload")
            .field("target", &self.target)
            .field("ready", &self.is_ready())
            .finish()
    }
}

impl PendingUpload {
    /// Whether polling would upload (or fail) now.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.task.as_ref().is_none_or(AsyncHandle::is_finished)
    }

    /// Where the bytes go.
    #[must_use]
    pub const fn target(&self) -> UploadTarget {
        self.target
    }
}

/// A readback in flight; returned by [`Device::read_buffer_async`] and
/// [`Device::read_texture_async`], finished by [`Device::poll_readback`] or
/// [`Device::wait_readback`]. Dropping it discards the result; the staging buffer is freed on
/// a later poll.
#[must_use = "the bytes arrive only by polling or waiting on the readback"]
#[derive(Debug)]
pub struct PendingReadback {
    device: u32,
    staging: Option<Buffer>,
    submission: u64,
    size: u64,
    /// Bytes of a texel row and of a padded staging row, for texture readbacks.
    rows: Option<(usize, usize)>,
    flag: MapFlag,
    mapping: bool,
    abandoned: Arc<AtomicBool>,
}

impl PendingReadback {
    /// The submission that copies the data; the bytes are available once it has retired.
    #[must_use]
    pub const fn submission(&self) -> SubmissionIndex {
        SubmissionIndex(self.submission)
    }
}

impl Drop for PendingReadback {
    fn drop(&mut self) {
        self.abandoned.store(true, Ordering::Release);
    }
}

fn site_error(command: &'static str, reason: &str) -> RhiError {
    RhiError::InvalidCommand {
        label: None,
        command,
        reason: reason.to_owned(),
    }
}

/// The bytes of a padded `rows`-shaped readback without the padding.
fn strip_rows(bytes: &[u8], row: usize, padded: usize) -> Result<Vec<u8>, RhiError> {
    let lines = bytes.len().checked_div(padded).unwrap_or(0);
    let mut out = Vec::new();
    out.try_reserve_exact(lines.saturating_mul(row))
        .map_err(|_| RhiError::OutOfMemory {
            bytes: lines.saturating_mul(row) as u64,
        })?;
    for line in bytes.chunks(padded) {
        out.extend_from_slice(line.get(..row).unwrap_or(line));
    }
    Ok(out)
}

impl Device {
    // -- the belt -------------------------------------------------------------------------

    /// The upload belt's state.
    #[must_use]
    pub fn upload_stats(&self) -> UploadStats {
        UploadStats {
            chunks: self.belt.chunks.len(),
            allocated: self.belt.allocated,
            pending: self.belt.pending,
            uploaded: self.belt.uploaded,
            stalls: self.belt.stalls,
        }
    }

    /// Copies `data` into the upload belt now; it lands in `buffer` (with
    /// [`BufferUsages::COPY_DST`]) at the start of the next submission, after queue writes.
    /// Unlike [`Self::write_buffer`], the staging memory is bounded and recycled per frame.
    /// Offset and length are multiples of [`COPY_BUFFER_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`upload_buffer`) for a missing usage, a misaligned or
    /// out-of-bounds range; [`RhiError::LimitExceeded`] for more bytes than
    /// [`UploadConfig::capacity`]; the handle's errors; [`RhiError::OutOfMemory`];
    /// [`RhiError::DeviceLost`]; [`RhiError::Backend`].
    pub fn upload_buffer(
        &mut self,
        buffer: Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        const COMMAND: &str = "upload_buffer";
        self.alive()?;
        let len = data.len() as u64;
        let entry = self.buffers.get(buffer)?;
        if !entry.info.usage.contains(BufferUsages::COPY_DST) {
            return Err(site_error(COMMAND, "the buffer lacks COPY_DST usage"));
        }
        if !offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
            || !len.is_multiple_of(COPY_BUFFER_ALIGNMENT)
        {
            return Err(site_error(
                COMMAND,
                "the offset and length must be multiples of 4",
            ));
        }
        if offset
            .checked_add(len)
            .is_none_or(|end| end > entry.info.size)
        {
            return Err(site_error(COMMAND, "the range is out of bounds"));
        }
        if data.is_empty() {
            return Ok(());
        }
        // Mapped ranges start at multiples of 8.
        let (chunk, at) = self.reserve_upload(len, 8)?;
        let staging = self.chunk_buffer(chunk)?;
        let mut copy = CommandEncoder::new(Some(LABEL));
        copy.copy_buffer_to_buffer(staging, at, buffer, offset, len);
        self.check_commands(&copy.finish())?;
        self.write_chunk(chunk, at, data)?;
        self.belt
            .encoder
            .get_or_insert_with(|| CommandEncoder::new(Some(LABEL)))
            .copy_buffer_to_buffer(staging, at, buffer, offset, len);
        self.commit_upload(chunk, at, len);
        Ok(())
    }

    /// Copies texels into the upload belt now; they land in the texture region at the start of
    /// the next submission. `data` is laid out as `layout` says (any row stride, unlike a
    /// buffer-to-texture copy); the belt repacks the rows to the copy alignment.
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`upload_texture` for a short `data` or a layout whose rows
    /// are too short, `copy_buffer_to_texture` for a region the texture cannot be copied to);
    /// otherwise as [`Self::upload_buffer`].
    pub fn upload_texture(
        &mut self,
        dst: TexelCopyTextureInfo,
        data: &[u8],
        layout: TexelCopyBufferLayout,
        size: Extent3d,
    ) -> Result<(), RhiError> {
        const COMMAND: &str = "upload_texture";
        self.alive()?;
        let format = self.textures.get(dst.texture)?.info.format;
        let row = u64::from(size.width).saturating_mul(u64::from(format.block_size()));
        let (height, images) = (
            u64::from(size.height),
            u64::from(size.depth_or_array_layers),
        );
        if row == 0 || height == 0 || images == 0 {
            return Ok(());
        }
        let stride = layout.bytes_per_row.map_or(row, u64::from);
        let rows_per_image = layout.rows_per_image.map_or(height, u64::from);
        if stride < row || rows_per_image < height {
            return Err(site_error(
                COMMAND,
                "bytes_per_row or rows_per_image is smaller than the region",
            ));
        }
        let overflow = || RhiError::OutOfMemory { bytes: u64::MAX };
        // The last row of the last image ends `row` bytes after its start.
        let last_row = rows_per_image
            .checked_mul(images.saturating_sub(1))
            .and_then(|rows| rows.checked_add(height.saturating_sub(1)))
            .ok_or_else(overflow)?;
        let needed = stride
            .checked_mul(last_row)
            .and_then(|bytes| bytes.checked_add(row))
            .and_then(|bytes| bytes.checked_add(layout.offset))
            .ok_or_else(overflow)?;
        if (data.len() as u64) < needed {
            return Err(site_error(COMMAND, "the data is too short for the region"));
        }
        let alignment = u64::from(COPY_BYTES_PER_ROW_ALIGNMENT);
        let padded = round_up(row, alignment).ok_or_else(overflow)?;
        let bytes_per_row = u32::try_from(padded)
            .map_err(|_| RhiError::limit("bytes_per_row", padded, u32::MAX))?;
        let total = padded
            .checked_mul(height)
            .and_then(|bytes| bytes.checked_mul(images))
            .ok_or_else(overflow)?;
        let (chunk, at) = self.reserve_upload(total, alignment)?;
        let staging = self.chunk_buffer(chunk)?;
        let src = TexelCopyBufferInfo {
            buffer: staging,
            layout: TexelCopyBufferLayout {
                offset: at,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(size.height),
            },
        };
        let mut copy = CommandEncoder::new(Some(LABEL));
        copy.copy_buffer_to_texture(src, dst, size);
        self.check_commands(&copy.finish())?;

        let mut packed = Vec::new();
        let len = usize::try_from(total).map_err(|_| overflow())?;
        packed
            .try_reserve_exact(len)
            .map_err(|_| RhiError::OutOfMemory { bytes: total })?;
        let (row_len, padding) = (
            usize::try_from(row).map_err(|_| overflow())?,
            usize::try_from(padded.saturating_sub(row)).map_err(|_| overflow())?,
        );
        for image in 0..images {
            for line in 0..height {
                let start = image
                    .saturating_mul(rows_per_image)
                    .saturating_add(line)
                    .saturating_mul(stride)
                    .saturating_add(layout.offset);
                let start = usize::try_from(start).map_err(|_| overflow())?;
                let bytes = data
                    .get(start..start.saturating_add(row_len))
                    .ok_or_else(|| site_error(COMMAND, "the data is too short for the region"))?;
                packed.extend_from_slice(bytes);
                packed.resize(packed.len().saturating_add(padding), 0);
            }
        }
        self.write_chunk(chunk, at, &packed)?;
        self.belt
            .encoder
            .get_or_insert_with(|| CommandEncoder::new(Some(LABEL)))
            .copy_buffer_to_texture(src, dst, size);
        self.commit_upload(chunk, at, total);
        Ok(())
    }

    /// Submits the pending uploads on their own, if there are any, and returns that
    /// submission. [`Self::submit`] does this anyway; call it to make uploads land without
    /// other commands (at the end of loading, before a blocking read).
    ///
    /// # Errors
    ///
    /// As [`Self::submit`].
    pub fn flush_uploads(&mut self) -> Result<Option<SubmissionIndex>, RhiError> {
        if self.belt.encoder.is_none() {
            return Ok(None);
        }
        self.submit([]).map(Some)
    }

    fn chunk_buffer(&self, chunk: usize) -> Result<Buffer, RhiError> {
        self.belt
            .chunks
            .get(chunk)
            .map(|chunk| chunk.buffer)
            .ok_or_else(|| RhiError::backend("an upload chunk disappeared"))
    }

    fn write_chunk(&mut self, chunk: usize, at: u64, data: &[u8]) -> Result<(), RhiError> {
        let buffer = self.chunk_buffer(chunk)?;
        let obj = self.buffers.get(buffer)?.obj.clone();
        self.backend.write_mapped(&obj, at, data)
    }

    fn commit_upload(&mut self, chunk: usize, at: u64, len: u64) {
        if let Some(chunk) = self.belt.chunks.get_mut(chunk) {
            chunk.cursor = at.saturating_add(len);
        }
        self.belt.pending = self.belt.pending.saturating_add(len);
    }

    /// Finds room for `len` bytes: a mapped chunk, a retired one, a new one within the
    /// capacity — or, when the belt is full, the oldest chunk in flight after the GPU is done
    /// with it.
    fn reserve_upload(&mut self, len: u64, alignment: u64) -> Result<(usize, u64), RhiError> {
        let capacity = self.belt.config.capacity;
        let needed = round_up(len, 256).ok_or(RhiError::OutOfMemory { bytes: len })?;
        if needed > capacity {
            return Err(RhiError::limit("upload capacity", needed, capacity));
        }
        let size = needed.max(self.belt.config.chunk_size.min(capacity));
        let size = round_up(size, 256).ok_or(RhiError::OutOfMemory { bytes: size })?;
        // Every round frees or recycles at least one chunk, so this bound is never reached
        // while the GPU makes progress.
        let rounds = self.belt.chunks.len().saturating_add(2).saturating_mul(2);
        for _ in 0..rounds {
            if let Some(found) = self.belt.fit(len, alignment) {
                return Ok(found);
            }
            self.retire_uploads()?;
            if let Some(found) = self.belt.fit(len, alignment) {
                return Ok(found);
            }
            while self.belt.allocated.saturating_add(size) > capacity {
                let Some(idle) = self.belt.idle_chunk() else {
                    break;
                };
                self.drop_chunk(idle)?;
            }
            if self.belt.allocated.saturating_add(size) <= capacity {
                return self.new_chunk(size).map(|chunk| (chunk, 0));
            }
            // Full: let the GPU catch up with the oldest chunk.
            self.belt.stalls = self.belt.stalls.saturating_add(1);
            if self.belt.oldest_in_flight().is_none() && self.belt.encoder.is_some() {
                self.flush_uploads()?;
            }
            if let Some(fence) = self.belt.oldest_in_flight() {
                self.wait(SubmissionIndex(fence))?;
            }
            self.backend.wait_for_maps()?;
            self.retire_uploads()?;
        }
        Err(RhiError::backend("the upload belt made no progress"))
    }

    fn new_chunk(&mut self, size: u64) -> Result<usize, RhiError> {
        let info = validate_buffer(
            &BufferDescriptor {
                label: Some(LABEL),
                size,
                usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            },
            &self.limits,
        )?;
        self.belt
            .chunks
            .try_reserve(1)
            .map_err(|_| RhiError::OutOfMemory { bytes: 64 })?;
        self.buffers.reserve()?;
        let obj = self.backend.create_mapped_buffer(Some(LABEL), &info)?;
        let buffer = self.buffers.insert(BufferEntry {
            info,
            users: 0,
            obj,
        })?;
        self.belt.chunks.push(Chunk {
            buffer,
            size,
            cursor: 0,
            state: ChunkState::Mapped,
            flag: MapFlag::default(),
        });
        self.belt.allocated = self.belt.allocated.saturating_add(size);
        Ok(self.belt.chunks.len().saturating_sub(1))
    }

    fn drop_chunk(&mut self, index: usize) -> Result<(), RhiError> {
        if index >= self.belt.chunks.len() {
            return Ok(());
        }
        let chunk = self.belt.chunks.swap_remove(index);
        self.belt.allocated = self.belt.allocated.saturating_sub(chunk.size);
        let entry = self.buffers.remove(chunk.buffer)?;
        self.bury(Garbage::Buffer(entry.obj, entry.info))
    }

    /// Moves chunks along: retired ones are mapped again, mapped ones become writable.
    pub(crate) fn retire_uploads(&mut self) -> Result<(), RhiError> {
        let completed = self.timeline.completed;
        let mut failed = Vec::new();
        for (index, chunk) in self.belt.chunks.iter_mut().enumerate() {
            match chunk.state {
                ChunkState::InFlight(fence) if fence <= completed => {
                    let obj = &self.buffers.get(chunk.buffer)?.obj;
                    self.backend
                        .map_async(obj, MapMode::Write, 0, chunk.size, &chunk.flag)?;
                    chunk.state = ChunkState::Remapping;
                }
                ChunkState::InFlight(_) | ChunkState::Mapped => {}
                ChunkState::Remapping => {}
            }
            if chunk.state == ChunkState::Remapping {
                match chunk.flag.state() {
                    MapState::Ready => {
                        chunk.state = ChunkState::Mapped;
                        chunk.cursor = 0;
                    }
                    MapState::Failed => failed.push(index),
                    MapState::Pending => {}
                }
            }
        }
        // Highest index first: `swap_remove` only moves later chunks.
        for index in failed.into_iter().rev() {
            self.drop_chunk(index)?;
        }
        Ok(())
    }

    // -- asynchronous uploads -------------------------------------------------------------

    /// Runs `produce` on `executor`'s blocking pool — decoding an image, reading a file — and
    /// uploads what it returns to `target` once [`Self::poll_upload`] sees it finished. The
    /// device is not touched off the calling thread.
    ///
    /// # Errors
    ///
    /// The target handle's errors; [`RhiError::DeviceLost`]. Errors of `produce` and of the
    /// upload itself arrive when polling.
    pub fn upload_async<F>(
        &mut self,
        target: UploadTarget,
        executor: &Executor,
        produce: F,
    ) -> Result<PendingUpload, RhiError>
    where
        F: FnOnce() -> Result<Vec<u8>, RhiError> + Send + 'static,
    {
        self.alive()?;
        match target {
            UploadTarget::Buffer { buffer, .. } => {
                self.buffers.get(buffer)?;
            }
            UploadTarget::Texture { texture, .. } => {
                self.textures.get(texture.texture)?;
            }
        }
        Ok(PendingUpload {
            target,
            task: Some(executor.unblock(produce)),
        })
    }

    fn upload_to(&mut self, target: UploadTarget, data: &[u8]) -> Result<(), RhiError> {
        match target {
            UploadTarget::Buffer { buffer, offset } => self.upload_buffer(buffer, offset, data),
            UploadTarget::Texture {
                texture,
                layout,
                size,
            } => self.upload_texture(texture, data, layout, size),
        }
    }

    /// Uploads the produced bytes if they are ready: `None` while `produce` runs, then its
    /// result once.
    ///
    /// # Errors
    ///
    /// The error `produce` returned; [`RhiError::Task`] when the task failed (or the result was
    /// already taken); the errors of [`Self::upload_buffer`] / [`Self::upload_texture`].
    pub fn poll_upload(&mut self, pending: &mut PendingUpload) -> Option<Result<(), RhiError>> {
        let Some(task) = pending.task.as_mut() else {
            return Some(Err(RhiError::Task(AsyncError::Consumed)));
        };
        if !task.is_finished() {
            return None;
        }
        let result = task.poll_ready()?;
        pending.task = None;
        Some(
            result
                .map_err(RhiError::Task)
                .and_then(|produced| produced)
                .and_then(|data| self.upload_to(pending.target, &data)),
        )
    }

    /// Blocks until the bytes are produced and uploads them. Only outside pool work; see
    /// [`Self::wait_render_pipeline`].
    ///
    /// # Errors
    ///
    /// As [`Self::poll_upload`].
    pub fn wait_upload(&mut self, mut pending: PendingUpload) -> Result<(), RhiError> {
        let task = pending
            .task
            .take()
            .ok_or(RhiError::Task(AsyncError::Consumed))?;
        let data = task.wait().map_err(RhiError::Task)??;
        self.upload_to(pending.target, &data)
    }

    // -- asynchronous readback ------------------------------------------------------------

    fn begin_readback(
        &mut self,
        size: u64,
        rows: Option<(usize, usize)>,
        record: impl FnOnce(&mut CommandEncoder, Buffer),
    ) -> Result<PendingReadback, RhiError> {
        self.readbacks
            .0
            .try_reserve(1)
            .map_err(|_| RhiError::OutOfMemory { bytes: 16 })?;
        let staging = self.create_buffer(&BufferDescriptor {
            label: Some("pragma_rhi readback"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        })?;
        let mut encoder = CommandEncoder::new(Some("pragma_rhi readback"));
        record(&mut encoder, staging);
        let submission = match self.submit([encoder.finish()]) {
            Ok(submission) => submission,
            Err(err) => {
                // Never used by a submission: freed at once.
                if let Ok(entry) = self.buffers.remove(staging) {
                    self.backend.destroy_buffer(entry.obj, &entry.info);
                }
                return Err(err);
            }
        };
        let abandoned = Arc::new(AtomicBool::new(false));
        self.readbacks.0.push((staging, Arc::clone(&abandoned)));
        Ok(PendingReadback {
            device: self.id,
            staging: Some(staging),
            submission: submission.0,
            size,
            rows,
            flag: MapFlag::default(),
            mapping: false,
            abandoned,
        })
    }

    /// Copies a range of a buffer with [`BufferUsages::COPY_SRC`] into a staging buffer, as a
    /// submission of its own (after the pending uploads), without waiting. Offset and size are
    /// multiples of [`COPY_BUFFER_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`copy_buffer_to_buffer`) for a missing usage, a
    /// misaligned or out-of-bounds range; otherwise as [`Self::submit`].
    pub fn read_buffer_async(
        &mut self,
        buffer: Buffer,
        offset: u64,
        size: u64,
    ) -> Result<PendingReadback, RhiError> {
        self.alive()?;
        self.buffers.get(buffer)?;
        if size == 0 {
            return Err(site_error("read_buffer_async", "the size is zero"));
        }
        self.begin_readback(size, None, |encoder, staging| {
            encoder.copy_buffer_to_buffer(buffer, offset, staging, 0, size);
        })
    }

    /// Copies a texture region with [`TextureUsages::COPY_SRC`](crate::TextureUsages::COPY_SRC)
    /// into a staging buffer, as a submission of its own, without waiting; the bytes come back
    /// row after row with no padding, as [`Self::read_texture`] returns them.
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`copy_texture_to_buffer`) for a region the texture cannot
    /// be copied from or an empty one; [`RhiError::LimitExceeded`] when the padded region is
    /// larger than [`Limits::max_buffer_size`](crate::Limits::max_buffer_size); otherwise as
    /// [`Self::submit`].
    pub fn read_texture_async(
        &mut self,
        src: TexelCopyTextureInfo,
        size: Extent3d,
    ) -> Result<PendingReadback, RhiError> {
        self.alive()?;
        let format = self.textures.get(src.texture)?.info.format;
        let row = u64::from(size.width).saturating_mul(u64::from(format.block_size()));
        let (height, images) = (
            u64::from(size.height),
            u64::from(size.depth_or_array_layers),
        );
        if row == 0 || height == 0 || images == 0 {
            return Err(site_error("copy_texture_to_buffer", "the region is empty"));
        }
        let overflow = || RhiError::OutOfMemory { bytes: u64::MAX };
        let padded = round_up(row, u64::from(COPY_BYTES_PER_ROW_ALIGNMENT)).ok_or_else(overflow)?;
        let bytes_per_row = u32::try_from(padded)
            .map_err(|_| RhiError::limit("bytes_per_row", padded, u32::MAX))?;
        let total = padded
            .checked_mul(height)
            .and_then(|bytes| bytes.checked_mul(images))
            .ok_or_else(overflow)?;
        let rows = (
            usize::try_from(row).map_err(|_| overflow())?,
            usize::try_from(padded).map_err(|_| overflow())?,
        );
        self.begin_readback(total, Some(rows), |encoder, staging| {
            encoder.copy_texture_to_buffer(
                src,
                TexelCopyBufferInfo {
                    buffer: staging,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(size.height),
                    },
                },
                size,
            );
        })
    }

    /// The bytes of a readback, if they are ready: `None` until its submission has retired
    /// and the staging buffer is mapped (polls the device), then the bytes once.
    ///
    /// # Errors
    ///
    /// [`RhiError::Task`] (`Consumed`) after the bytes were taken;
    /// [`RhiError::ForeignHandle`] for another device's readback; [`RhiError::DeviceLost`];
    /// [`RhiError::Backend`] when the mapping failed.
    pub fn poll_readback(
        &mut self,
        pending: &mut PendingReadback,
    ) -> Option<Result<Vec<u8>, RhiError>> {
        if let Err(err) = self.poll() {
            return Some(Err(err));
        }
        self.advance_readback(pending).transpose()
    }

    /// Blocks until a readback's bytes are ready and returns them.
    ///
    /// # Errors
    ///
    /// As [`Self::poll_readback`].
    pub fn wait_readback(&mut self, mut pending: PendingReadback) -> Result<Vec<u8>, RhiError> {
        self.wait(SubmissionIndex(pending.submission))?;
        if let Some(bytes) = self.advance_readback(&mut pending)? {
            return Ok(bytes);
        }
        self.backend.wait_for_maps()?;
        self.advance_readback(&mut pending)?
            .ok_or_else(|| RhiError::backend("a readback mapping did not resolve"))
    }

    /// Requests the mapping once the copy retired, and takes the bytes once mapped.
    fn advance_readback(
        &mut self,
        pending: &mut PendingReadback,
    ) -> Result<Option<Vec<u8>>, RhiError> {
        if pending.device != self.id {
            return Err(RhiError::ForeignHandle {
                resource: ResourceKind::Buffer,
            });
        }
        let staging = pending
            .staging
            .ok_or(RhiError::Task(AsyncError::Consumed))?;
        if !self.is_complete(SubmissionIndex(pending.submission)) {
            return Ok(None);
        }
        let obj = self.buffers.get(staging)?.obj.clone();
        if !pending.mapping {
            self.backend
                .map_async(&obj, MapMode::Read, 0, pending.size, &pending.flag)?;
            pending.mapping = true;
        }
        let bytes = match pending.flag.state() {
            MapState::Pending => return Ok(None),
            MapState::Failed => Err(RhiError::backend("mapping a readback buffer failed")),
            MapState::Ready => self.backend.read_mapped(&obj, 0, pending.size),
        };
        pending.staging = None;
        self.backend.unmap(&obj);
        self.end_readback(staging)?;
        let bytes = bytes?;
        match pending.rows {
            Some((row, padded)) => strip_rows(&bytes, row, padded).map(Some),
            None => Ok(Some(bytes)),
        }
    }

    /// Destroys a readback's staging buffer; destruction waits for its submission, and a
    /// mapping still pending is cancelled.
    fn end_readback(&mut self, staging: Buffer) -> Result<(), RhiError> {
        self.readbacks.0.retain(|(buffer, _)| *buffer != staging);
        let entry = self.buffers.remove(staging)?;
        self.bury(Garbage::Buffer(entry.obj, entry.info))
    }

    /// Destroys the staging buffers of readbacks dropped unread.
    pub(crate) fn sweep_readbacks(&mut self) -> Result<(), RhiError> {
        let abandoned: Vec<Buffer> = self
            .readbacks
            .0
            .iter()
            .filter(|(_, flag)| flag.load(Ordering::Acquire))
            .map(|(buffer, _)| *buffer)
            .collect();
        for staging in abandoned {
            self.end_readback(staging)?;
        }
        Ok(())
    }
}
