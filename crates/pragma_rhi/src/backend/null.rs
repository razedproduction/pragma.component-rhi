//! The null backend: no GPU, deterministic.
//!
//! It accounts device memory against a fixed budget, so out-of-memory paths can be tested on
//! any machine, and it keeps the contents of buffers and textures (DECISIONS.md #16): queue
//! writes, buffer clears, every copy, attachment clears, resolves and discards run on the CPU
//! with the results a GPU produces, so readback tests pass on CI without one. Shaders do not
//! run: draws and dispatches are validated like on every backend and then change nothing.
//!
//! Memory is allocated when a resource is first written; until then it reads as zeros, which
//! is what WebGPU guarantees for a new resource.

use std::collections::HashMap;

use super::texel::{self, Layout};
use super::{
    Batch, BufRegion, ColorOp, DepthOp, NullId, Op, RenderPassOp, Subresource, TexRegion, null_of,
};
use crate::resource::mip_extent;
use crate::{Extent3d, LoadOp, RhiError, StoreOp};

#[derive(Debug)]
pub(crate) struct NullDevice {
    budget: u64,
    allocated: u64,
    lost: bool,
    next_id: NullId,
    /// The bytes of every resource written so far, by id.
    memory: HashMap<NullId, Vec<u8>>,
    /// How many submissions the null "GPU" keeps in flight when polled without waiting.
    latency: u32,
    submitted: u64,
    completed: u64,
}

const fn oom(bytes: u64) -> RhiError {
    RhiError::OutOfMemory { bytes }
}

fn byte_range(offset: u64, len: u64) -> Result<core::ops::Range<usize>, RhiError> {
    let start = usize::try_from(offset).map_err(|_| oom(len))?;
    let count = usize::try_from(len).map_err(|_| oom(len))?;
    let end = start.checked_add(count).ok_or(oom(u64::MAX))?;
    Ok(start..end)
}

impl NullDevice {
    pub(crate) fn new(budget: u64, latency: u32) -> Self {
        Self {
            budget,
            allocated: 0,
            lost: false,
            next_id: 0,
            memory: HashMap::new(),
            latency,
            submitted: 0,
            completed: 0,
        }
    }

    pub(crate) const fn is_lost(&self) -> bool {
        self.lost
    }

    pub(crate) const fn alive(&self) -> Result<(), RhiError> {
        if self.lost {
            Err(RhiError::DeviceLost)
        } else {
            Ok(())
        }
    }

    /// Submissions execute at once; they *complete* `latency` submissions later, or when
    /// waited for, so frames in flight and deferred destruction behave as on a GPU.
    pub(crate) fn poll(&mut self, wait: Option<u64>) -> Result<u64, RhiError> {
        self.alive()?;
        let settled = self.submitted.saturating_sub(u64::from(self.latency));
        let waited = wait.map_or(0, |index| index.min(self.submitted));
        self.completed = self.completed.max(settled).max(waited);
        Ok(self.completed)
    }

    pub(crate) fn lose(&mut self) {
        self.lost = true;
        self.memory.clear();
    }

    pub(crate) const fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Accounts `bytes` against the budget and names the new resource's memory.
    pub(crate) fn allocate(&mut self, bytes: u64) -> Result<NullId, RhiError> {
        match self.allocated.checked_add(bytes) {
            Some(total) if total <= self.budget => {
                let id = self.next_id;
                self.next_id = id.checked_add(1).ok_or(oom(bytes))?;
                self.allocated = total;
                Ok(id)
            }
            _ => Err(oom(bytes)),
        }
    }

    pub(crate) fn release(&mut self, id: NullId, bytes: u64) {
        self.memory.remove(&id);
        self.allocated = self.allocated.saturating_sub(bytes);
    }

    /// Writes `data` at `offset`, growing the resource's memory (zero-filled) to `len` bytes,
    /// and at least to the end of the write, on first use.
    fn store(&mut self, id: NullId, len: u64, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        let at = byte_range(offset, data.len() as u64)?;
        let size = usize::try_from(len).map_err(|_| oom(len))?.max(at.end);
        let memory = self.memory.entry(id).or_default();
        if memory.len() < size {
            memory
                .try_reserve_exact(size.saturating_sub(memory.len()))
                .map_err(|_| oom(len))?;
            memory.resize(size, 0);
        }
        let target = memory
            .get_mut(at)
            .ok_or_else(|| RhiError::backend("null backend write out of bounds"))?;
        target.copy_from_slice(data);
        Ok(())
    }

    /// Reads `len` bytes at `offset`; memory never written reads as zeros.
    fn load(&self, id: NullId, offset: u64, len: u64) -> Result<Vec<u8>, RhiError> {
        let at = byte_range(offset, len)?;
        let mut out = Vec::new();
        out.try_reserve_exact(at.len()).map_err(|_| oom(len))?;
        if let Some(memory) = self.memory.get(&id) {
            let end = at.end.min(memory.len());
            let start = at.start.min(end);
            out.extend_from_slice(memory.get(start..end).unwrap_or(&[]));
        }
        out.resize(at.len(), 0);
        Ok(out)
    }

    pub(crate) fn write(&mut self, id: NullId, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        self.store(id, 0, offset, data)
    }

    pub(crate) fn read(&self, id: NullId, offset: u64, size: u64) -> Result<Vec<u8>, RhiError> {
        self.load(id, offset, size)
    }

    // -- execution ------------------------------------------------------------------------

    pub(crate) fn execute(&mut self, batches: &[Batch<'_>], index: u64) -> Result<(), RhiError> {
        if self.lost {
            return Err(RhiError::DeviceLost);
        }
        for batch in batches {
            for op in &batch.ops {
                self.run(op)?;
            }
        }
        self.submitted = index;
        Ok(())
    }

    fn run(&mut self, op: &Op<'_>) -> Result<(), RhiError> {
        match op {
            Op::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let data = self.load(null_of(src)?, *src_offset, *size)?;
                self.write(null_of(dst)?, *dst_offset, &data)
            }
            Op::ClearBuffer { obj, offset, size } => {
                let count = usize::try_from(*size).map_err(|_| oom(*size))?;
                let mut zeros = Vec::new();
                zeros.try_reserve_exact(count).map_err(|_| oom(*size))?;
                zeros.resize(count, 0);
                self.write(null_of(obj)?, *offset, &zeros)
            }
            Op::CopyBufferToTexture { src, dst, size } => self.buffer_to_texture(src, dst, *size),
            Op::CopyTextureToBuffer { src, dst, size } => self.texture_to_buffer(src, dst, *size),
            Op::CopyTextureToTexture { src, dst, size } => self.texture_to_texture(src, dst, *size),
            Op::RenderPass(pass) => self.render_pass(pass),
            // Shaders do not run on the null backend.
            Op::ComputePass(_) => Ok(()),
        }
    }

    /// Calls `row(buffer_offset, texture_offset, row_bytes)` for every row of a texture copy.
    fn rows(
        texture: &TexRegion<'_>,
        buffer: &BufRegion<'_>,
        size: Extent3d,
        mut row: impl FnMut(u64, u64, u64) -> Result<(), RhiError>,
    ) -> Result<(), RhiError> {
        let layout = Layout::new(texture.info);
        let row_bytes = u64::from(size.width).saturating_mul(layout.block());
        let x = u64::from(texture.origin.x).saturating_mul(layout.block());
        for z in 0..size.depth_or_array_layers {
            let image = u64::from(z)
                .saturating_mul(buffer.rows_per_image)
                .saturating_mul(buffer.bytes_per_row);
            for y in 0..size.height {
                let at = buffer
                    .layout
                    .offset
                    .saturating_add(image)
                    .saturating_add(u64::from(y).saturating_mul(buffer.bytes_per_row));
                let texel = layout
                    .row(
                        texture.mip,
                        texture.origin.y.saturating_add(y),
                        texture.origin.z.saturating_add(z),
                    )
                    .saturating_add(x);
                row(at, texel, row_bytes)?;
            }
        }
        Ok(())
    }

    fn buffer_to_texture(
        &mut self,
        src: &BufRegion<'_>,
        dst: &TexRegion<'_>,
        size: Extent3d,
    ) -> Result<(), RhiError> {
        let (from, to) = (null_of(src.obj)?, null_of(dst.obj)?);
        let total = Layout::new(dst.info).total();
        Self::rows(dst, src, size, |at, texel, len| {
            let data = self.load(from, at, len)?;
            self.store(to, total, texel, &data)
        })
    }

    fn texture_to_buffer(
        &mut self,
        src: &TexRegion<'_>,
        dst: &BufRegion<'_>,
        size: Extent3d,
    ) -> Result<(), RhiError> {
        let (from, to) = (null_of(src.obj)?, null_of(dst.obj)?);
        Self::rows(src, dst, size, |at, texel, len| {
            let data = self.load(from, texel, len)?;
            self.write(to, at, &data)
        })
    }

    fn texture_to_texture(
        &mut self,
        src: &TexRegion<'_>,
        dst: &TexRegion<'_>,
        size: Extent3d,
    ) -> Result<(), RhiError> {
        let (from, to) = (null_of(src.obj)?, null_of(dst.obj)?);
        let (src_layout, dst_layout) = (Layout::new(src.info), Layout::new(dst.info));
        let total = dst_layout.total();
        let row_bytes = u64::from(size.width).saturating_mul(src_layout.block());
        let at = |region: &TexRegion<'_>, layout: &Layout<'_>, y: u32, z: u32| {
            layout
                .row(
                    region.mip,
                    region.origin.y.saturating_add(y),
                    region.origin.z.saturating_add(z),
                )
                .saturating_add(u64::from(region.origin.x).saturating_mul(layout.block()))
        };
        for z in 0..size.depth_or_array_layers {
            for y in 0..size.height {
                let data = self.load(from, at(src, &src_layout, y, z), row_bytes)?;
                self.store(to, total, at(dst, &dst_layout, y, z), &data)?;
            }
        }
        Ok(())
    }

    /// Fills a whole attachment subresource with one texel value.
    fn fill(&mut self, target: &Subresource<'_>, texel: &[u8]) -> Result<(), RhiError> {
        let id = null_of(target.texture)?;
        let layout = Layout::new(target.info);
        let total = layout.total();
        let size = mip_extent(target.info, target.mip);
        let row_len = usize::try_from(size.width)
            .ok()
            .and_then(|width| width.checked_mul(texel.len()))
            .ok_or(oom(u64::MAX))?;
        let mut row = Vec::new();
        row.try_reserve_exact(row_len)
            .map_err(|_| oom(row_len as u64))?;
        for _ in 0..size.width {
            row.extend_from_slice(texel);
        }
        for y in 0..size.height {
            self.store(id, total, layout.row(target.mip, y, target.layer), &row)?;
        }
        Ok(())
    }

    /// Copies a whole attachment subresource to another of the same format and size.
    fn copy_subresource(
        &mut self,
        from: &Subresource<'_>,
        to: &Subresource<'_>,
    ) -> Result<(), RhiError> {
        let (src, dst) = (null_of(from.texture)?, null_of(to.texture)?);
        let (src_layout, dst_layout) = (Layout::new(from.info), Layout::new(to.info));
        let total = dst_layout.total();
        let size = mip_extent(from.info, from.mip);
        let row_bytes = u64::from(size.width).saturating_mul(src_layout.block());
        for y in 0..size.height {
            let data = self.load(src, src_layout.row(from.mip, y, from.layer), row_bytes)?;
            self.store(dst, total, dst_layout.row(to.mip, y, to.layer), &data)?;
        }
        Ok(())
    }

    fn zero(&mut self, target: &Subresource<'_>) -> Result<(), RhiError> {
        let block = target.info.format.block_size() as usize;
        let zeros = [0_u8; 16];
        self.fill(target, zeros.get(..block).unwrap_or(&[]))
    }

    fn render_pass(&mut self, pass: &RenderPassOp<'_>) -> Result<(), RhiError> {
        // Load operations, in attachment order.
        for color in pass.colors.iter().flatten() {
            if let LoadOp::Clear(value) = color.ops.load
                && let Some(texel) = texel::encode_color(color.target.info.format, value)
            {
                self.fill(&color.target, texel.as_bytes())?;
            }
        }
        if let Some(DepthOp {
            target,
            depth: Some(depth),
            ..
        }) = &pass.depth
            && let LoadOp::Clear(value) = depth.load
            && let Some(texel) = texel::encode_depth(target.info.format, value)
        {
            self.fill(target, texel.as_bytes())?;
        }
        // Draws run no shaders here. Then resolves and store operations.
        for ColorOp {
            target,
            resolve,
            ops,
        } in pass.colors.iter().flatten()
        {
            if let Some(resolve) = resolve {
                // Every sample of a null texel holds the same value: the resolve is a copy.
                self.copy_subresource(target, resolve)?;
            }
            if ops.store == StoreOp::Discard {
                self.zero(target)?;
            }
        }
        if let Some(DepthOp { target, depth, .. }) = &pass.depth
            && depth.is_some_and(|depth| depth.store == StoreOp::Discard)
        {
            self.zero(target)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::NullDevice;
    use crate::RhiError;

    #[test]
    fn budget_is_enforced() {
        let mut device = NullDevice::new(100, 0);
        assert_eq!(device.allocate(60), Ok(0));
        assert_eq!(
            device.allocate(41),
            Err(RhiError::OutOfMemory { bytes: 41 })
        );
        assert_eq!(
            device.allocate(u64::MAX),
            Err(RhiError::OutOfMemory { bytes: u64::MAX })
        );
        device.release(0, 60);
        assert_eq!(device.allocate(100), Ok(1));
        assert_eq!(device.allocated(), 100);
    }

    #[test]
    fn memory_reads_zero_until_written() {
        let mut device = NullDevice::new(1 << 20, 0);
        let id = device.allocate(16).unwrap_or_default();
        assert_eq!(device.read(id, 0, 8), Ok(vec![0; 8]));
        assert_eq!(device.write(id, 4, &[1, 2, 3, 4]), Ok(()));
        assert_eq!(
            device.read(id, 0, 12),
            Ok(vec![0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0])
        );
        device.release(id, 16);
        assert_eq!(device.read(id, 0, 4), Ok(vec![0; 4]));
    }
}
