//! Buffers, textures and views against a naive model of handles, memory and the GPU timeline.
//!
//! The model knows which handles are live, what each one costs on the null backend, which views
//! pin which texture, and when deferred destruction (DECISIONS #17) frees memory: a destroyed
//! buffer or texture is freed at once if every submission so far has completed, and otherwise
//! when the device learns that the last submission at its destruction has (`poll`, `wait_idle`,
//! `begin_frame`). The null device runs with a submission latency, so destruction really is
//! deferred. After every operation the device's live counts, allocated bytes, deferred
//! destructions and fences must agree with the model. Used by `tests/oracle.rs` and the
//! `handles` fuzz target.

use std::collections::VecDeque;

use pragma_rhi::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, ResourceKind,
    RhiError, SubmissionIndex, Texture, TextureDescriptor, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

use super::bytes::Bytes;

/// The null device's memory budget: small enough that creations run out of it.
pub const BUDGET: u64 = 256 * 1024;
/// The null device's submission latency.
pub const LATENCY: u32 = 2;
const FRAMES_IN_FLIGHT: usize = 2;
const MAX_OPS: usize = 400;

/// One operation; indices pick among every handle of the kind ever created (modulo their number),
/// so destroyed handles are used again.
#[derive(Clone, Debug)]
pub enum Op {
    CreateBuffer(u64),
    /// A square `Rgba8Unorm` texture: side, mip levels, array layers (all ≥ 1, possibly invalid).
    CreateTexture(u32, u32, u32),
    /// A view of a texture starting at a mip level.
    CreateView(usize, u32),
    DestroyBuffer(usize),
    DestroyTexture(usize),
    DestroyView(usize),
    QueryBuffer(usize),
    QueryTexture(usize),
    QueryView(usize),
    /// A submission (a clear of a scratch buffer).
    Submit,
    Poll,
    WaitIdle,
    /// `begin_frame`, one submission, `end_frame`.
    Frame,
}

struct Live<H> {
    handle: H,
    bytes: u64,
    live: bool,
}

struct Model {
    buffers: Vec<Live<Buffer>>,
    textures: Vec<(Live<Texture>, u32)>,
    /// Views: the handle, the index of its texture, live.
    views: Vec<(TextureView, usize, bool)>,
    /// Deferred memory: the fence (last submission at destruction) and the bytes.
    graveyard: VecDeque<(u64, u64)>,
    submitted: u64,
    completed: u64,
    fences: VecDeque<u64>,
}

impl Model {
    fn allocated(&self) -> u64 {
        let buffers: u64 = self
            .buffers
            .iter()
            .filter(|b| b.live)
            .map(|b| b.bytes)
            .sum();
        let textures: u64 = self
            .textures
            .iter()
            .filter(|(t, _)| t.live)
            .map(|(t, _)| t.bytes)
            .sum();
        let deferred: u64 = self.graveyard.iter().map(|(_, bytes)| bytes).sum();
        SCRATCH + buffers + textures + deferred
    }

    fn bury(&mut self, bytes: u64) {
        if self.submitted > self.completed {
            self.graveyard.push_back((self.submitted, bytes));
        }
    }

    fn retire(&mut self) {
        while self
            .graveyard
            .front()
            .is_some_and(|(fence, _)| *fence <= self.completed)
        {
            self.graveyard.pop_front();
        }
    }

    /// What the null backend reports when polled, waiting for `wait` if given.
    fn poll(&mut self, wait: Option<u64>) {
        let settled = self.submitted.saturating_sub(u64::from(LATENCY));
        let waited = wait.map_or(0, |index| index.min(self.submitted));
        self.completed = self.completed.max(settled).max(waited);
    }
}

const SCRATCH: u64 = 4;

/// The memory of a square `Rgba8Unorm` texture with a mip chain and layers.
pub fn texture_bytes(side: u32, mips: u32, layers: u32) -> u64 {
    let per_layer: u64 = (0..mips)
        .map(|level| u64::from((side >> level).max(1)).pow(2) * 4)
        .sum();
    per_layer * u64::from(layers)
}

fn full_chain(side: u32) -> u32 {
    32 - side.leading_zeros()
}

struct Harness {
    device: Device,
    model: Model,
    scratch: Buffer,
    /// The index of every submission so far, in order.
    indices: Vec<SubmissionIndex>,
}

impl Harness {
    fn submit(&mut self) {
        let mut encoder = CommandEncoder::new(Some("scratch"));
        encoder.clear_buffer(self.scratch, 0, None);
        let index = self
            .device
            .submit([encoder.finish()])
            .expect("a submission");
        self.model.submitted += 1;
        assert_eq!(Some(index), self.device.last_submission());
        assert!(self.indices.last().is_none_or(|last| *last < index));
        self.indices.push(index);
    }

    fn apply(&mut self, op: Op) {
        let m = &mut self.model;
        match op {
            Op::CreateBuffer(size) => {
                let result = self.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size,
                    usage: BufferUsages::STORAGE,
                });
                if m.allocated() + size > BUDGET {
                    assert!(
                        matches!(result, Err(RhiError::OutOfMemory { bytes, .. }) if bytes == size),
                        "over budget is OutOfMemory, got {result:?}"
                    );
                } else {
                    let handle = result.expect("a buffer within budget");
                    assert!(
                        m.buffers.iter().all(|b| b.handle != handle),
                        "a buffer handle is never handed out twice"
                    );
                    m.buffers.push(Live {
                        handle,
                        bytes: size,
                        live: true,
                    });
                }
            }
            Op::CreateTexture(side, mips, layers) => {
                let result = self.device.create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: side,
                        height: side,
                        depth_or_array_layers: layers,
                    },
                    mip_level_count: mips,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    ..TextureDescriptor::default()
                });
                let bytes = texture_bytes(side, mips, layers);
                if mips > full_chain(side) {
                    assert!(
                        matches!(result, Err(RhiError::InvalidDescriptor { .. })),
                        "a mip chain longer than the size allows is invalid, got {result:?}"
                    );
                } else if m.allocated() + bytes > BUDGET {
                    assert!(
                        matches!(result, Err(RhiError::OutOfMemory { bytes: b, .. }) if b == bytes),
                        "over budget is OutOfMemory, got {result:?}"
                    );
                } else {
                    let handle = result.expect("a texture within budget");
                    assert!(
                        m.textures.iter().all(|(t, _)| t.handle != handle),
                        "a texture handle is never handed out twice"
                    );
                    m.textures.push((
                        Live {
                            handle,
                            bytes,
                            live: true,
                        },
                        mips,
                    ));
                }
            }
            Op::CreateView(n, base_mip) if !m.textures.is_empty() => {
                let index = n % m.textures.len();
                let (texture, mips) = &m.textures[index];
                let result = self.device.create_texture_view(
                    texture.handle,
                    &TextureViewDescriptor {
                        base_mip_level: base_mip,
                        ..TextureViewDescriptor::default()
                    },
                );
                if !texture.live {
                    assert!(
                        matches!(result, Err(RhiError::StaleHandle { .. })),
                        "a view of a destroyed texture is StaleHandle, got {result:?}"
                    );
                } else if base_mip >= *mips {
                    assert!(
                        matches!(result, Err(RhiError::InvalidDescriptor { .. })),
                        "a view past the mip chain is invalid, got {result:?}"
                    );
                } else {
                    let view = result.expect("a view");
                    let info = self.device.texture_view_info(view).expect("view info");
                    assert_eq!(info.texture, texture.handle);
                    assert_eq!(info.base_mip_level, base_mip);
                    assert_eq!(info.mip_level_count, mips - base_mip);
                    m.views.push((view, index, true));
                }
            }
            Op::DestroyBuffer(n) if !m.buffers.is_empty() => {
                let index = n % m.buffers.len();
                let result = self.device.destroy_buffer(m.buffers[index].handle);
                if m.buffers[index].live {
                    assert_eq!(result, Ok(()));
                    m.buffers[index].live = false;
                    let bytes = m.buffers[index].bytes;
                    m.bury(bytes);
                } else {
                    assert!(
                        matches!(result, Err(RhiError::StaleHandle { .. })),
                        "a second destroy is StaleHandle, got {result:?}"
                    );
                }
            }
            Op::DestroyTexture(n) if !m.textures.is_empty() => {
                let index = n % m.textures.len();
                let result = self.device.destroy_texture(m.textures[index].0.handle);
                let views = m
                    .views
                    .iter()
                    .filter(|(_, texture, live)| *live && *texture == index)
                    .count();
                if !m.textures[index].0.live {
                    assert!(
                        matches!(result, Err(RhiError::StaleHandle { .. })),
                        "a second destroy is StaleHandle, got {result:?}"
                    );
                } else if views > 0 {
                    assert!(
                        matches!(result, Err(RhiError::ResourceInUse { dependents, .. }) if dependents as usize == views),
                        "a texture with live views is in use, got {result:?}"
                    );
                } else {
                    assert_eq!(result, Ok(()));
                    m.textures[index].0.live = false;
                    let bytes = m.textures[index].0.bytes;
                    m.bury(bytes);
                }
            }
            Op::DestroyView(n) if !m.views.is_empty() => {
                let index = n % m.views.len();
                let (view, _, live) = m.views[index];
                let result = self.device.destroy_texture_view(view);
                if live {
                    assert_eq!(result, Ok(()));
                    m.views[index].2 = false;
                } else {
                    assert!(
                        matches!(result, Err(RhiError::StaleHandle { .. })),
                        "a second destroy is StaleHandle, got {result:?}"
                    );
                }
            }
            Op::QueryBuffer(n) if !m.buffers.is_empty() => {
                let buffer = &m.buffers[n % m.buffers.len()];
                let got = self.device.buffer_info(buffer.handle).map(|info| info.size);
                if buffer.live {
                    assert_eq!(got, Ok(buffer.bytes));
                } else {
                    assert!(matches!(got, Err(RhiError::StaleHandle { .. })));
                }
            }
            Op::QueryTexture(n) if !m.textures.is_empty() => {
                let (texture, mips) = &m.textures[n % m.textures.len()];
                let got = self
                    .device
                    .texture_info(texture.handle)
                    .map(|info| info.mip_level_count);
                if texture.live {
                    assert_eq!(got, Ok(*mips));
                } else {
                    assert!(matches!(got, Err(RhiError::StaleHandle { .. })));
                }
            }
            Op::QueryView(n) if !m.views.is_empty() => {
                let (view, texture, live) = m.views[n % m.views.len()];
                let got = self.device.texture_view_info(view).map(|info| info.texture);
                if live {
                    assert_eq!(got, Ok(m.textures[texture].0.handle));
                } else {
                    assert!(matches!(got, Err(RhiError::StaleHandle { .. })));
                }
            }
            Op::Submit => self.submit(),
            Op::Poll => {
                self.device.poll().expect("poll");
                self.model.poll(None);
                self.model.retire();
            }
            Op::WaitIdle => {
                self.device.wait_idle().expect("wait_idle");
                let last = self.model.submitted;
                self.model.poll(Some(last));
                self.model.retire();
            }
            Op::Frame => {
                self.device.begin_frame().expect("begin_frame");
                let m = &mut self.model;
                while m.fences.len() >= FRAMES_IN_FLIGHT {
                    if let Some(fence) = m.fences.pop_front()
                        && fence > m.completed
                    {
                        m.poll(Some(fence));
                    }
                }
                m.poll(None);
                m.retire();
                let completed = m.completed;
                m.fences.retain(|fence| *fence > completed);
                self.submit();
                self.device.end_frame().expect("end_frame");
                self.model.fences.push_back(self.model.submitted);
            }
            Op::CreateView(..)
            | Op::DestroyBuffer(_)
            | Op::DestroyTexture(_)
            | Op::DestroyView(_)
            | Op::QueryBuffer(_)
            | Op::QueryTexture(_)
            | Op::QueryView(_) => {}
        }
    }

    fn check(&self) {
        let m = &self.model;
        let live_buffers = m.buffers.iter().filter(|b| b.live).count();
        let live_textures = m.textures.iter().filter(|(t, _)| t.live).count();
        let live_views = m.views.iter().filter(|(_, _, live)| *live).count();
        assert_eq!(self.device.live(ResourceKind::Buffer), live_buffers + 1);
        assert_eq!(self.device.live(ResourceKind::Texture), live_textures);
        assert_eq!(self.device.live(ResourceKind::TextureView), live_views);
        assert_eq!(self.device.deferred_destructions(), m.graveyard.len());
        assert_eq!(self.device.allocated_bytes(), Some(m.allocated()));
        assert_eq!(self.device.last_submission(), self.indices.last().copied());
        for (fence, index) in (1..).zip(&self.indices) {
            assert_eq!(
                self.device.is_complete(*index),
                fence <= m.completed,
                "submission {fence} of {}, completed {}",
                m.submitted,
                m.completed
            );
        }
    }
}

/// Runs the operations, checking the device against the model after each one; at the end,
/// `wait_idle` frees every deferred destruction.
pub fn check(ops: impl IntoIterator<Item = Op>) {
    let mut device = super::null_device(BUDGET, LATENCY);
    let scratch = device
        .create_buffer(&BufferDescriptor {
            label: Some("scratch"),
            size: SCRATCH,
            usage: BufferUsages::COPY_DST,
        })
        .expect("the scratch buffer");
    let mut h = Harness {
        device,
        model: Model {
            buffers: Vec::new(),
            textures: Vec::new(),
            views: Vec::new(),
            graveyard: VecDeque::new(),
            submitted: 0,
            completed: 0,
            fences: VecDeque::new(),
        },
        scratch,
        indices: Vec::new(),
    };
    for op in ops {
        h.apply(op);
        h.check();
    }
    h.device.wait_idle().expect("wait_idle");
    let last = h.model.submitted;
    h.model.poll(Some(last));
    h.model.retire();
    h.check();
    assert_eq!(h.device.deferred_destructions(), 0);
}

/// Arbitrary bytes as a sequence of operations.
pub fn decode(data: &[u8]) -> Vec<Op> {
    let mut bytes = Bytes::new(data);
    let mut ops = Vec::new();
    while !bytes.is_empty() && ops.len() < MAX_OPS {
        let op = match bytes.below(16) {
            0 | 1 => Op::CreateBuffer(1 + u64::from(bytes.u16() % 8192)),
            2 => {
                let side = 1 + u32::from(bytes.below(64));
                let mips = 1 + u32::from(bytes.below(8));
                Op::CreateTexture(side, mips, 1 + u32::from(bytes.below(4)))
            }
            3 | 4 => {
                let n = bytes.index();
                Op::CreateView(n, u32::from(bytes.below(8)))
            }
            5 => Op::DestroyBuffer(bytes.index()),
            6 => Op::DestroyTexture(bytes.index()),
            7 => Op::DestroyView(bytes.index()),
            8 => Op::QueryBuffer(bytes.index()),
            9 => Op::QueryTexture(bytes.index()),
            10 => Op::QueryView(bytes.index()),
            11 | 12 => Op::Submit,
            13 => Op::Poll,
            14 => Op::WaitIdle,
            _ => Op::Frame,
        };
        ops.push(op);
    }
    ops
}
