//! Queue writes, buffer copies and clears against a model of buffer contents.
//!
//! A few buffers with different sizes and usages (one not a multiple of 4 bytes, one that cannot
//! be a copy source, one mappable for writing that nothing can write) receive queue writes and
//! submissions of one to three command buffers of copies and clears, some naming a destroyed
//! buffer. The model applies the WebGPU rules (DECISIONS #15): usages, 4-byte alignment, ranges,
//! no copy within one buffer, a stale handle; a submission with one broken command is rejected
//! as a whole and changes nothing. Every readable buffer must then read back exactly what the
//! model holds. A recreated buffer reads as zeros even while the memory of its predecessor waits
//! for deferred destruction. Used by `tests/oracle.rs` and the `transfers` fuzz target.

use pragma_rhi::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, RhiError};

use super::bytes::Bytes;

const MAX_OPS: usize = 200;
const MAX_COMMANDS: usize = 4;
const MAX_BUFFERS: usize = 3;

/// The buffers: size and usages.
pub const BUFFERS: [(u64, BufferUsages); 5] = [
    (64, BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST)),
    (128, BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST)),
    (102, BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST)),
    (48, BufferUsages::MAP_READ.union(BufferUsages::COPY_DST)),
    (32, BufferUsages::MAP_WRITE.union(BufferUsages::COPY_SRC)),
];

/// A buffer reference: one of [`BUFFERS`] (`index < BUFFERS.len()`) or, above, a handle that was
/// destroyed by [`Op::Recreate`] (modulo their number; the first buffer if there is none).
#[derive(Clone, Copy, Debug)]
pub struct Ref(pub usize);

#[derive(Clone, Debug)]
pub enum Command {
    Copy {
        src: Ref,
        src_offset: u64,
        dst: Ref,
        dst_offset: u64,
        size: u64,
    },
    Clear {
        buffer: Ref,
        offset: u64,
        size: Option<u64>,
    },
}

#[derive(Clone, Debug)]
pub enum Op {
    /// A queue write of `len` bytes derived from the seed.
    Write {
        buffer: Ref,
        offset: u64,
        len: u64,
        seed: u8,
    },
    /// One submission: its command buffers, each a list of commands.
    Submit(Vec<Vec<Command>>),
    /// Destroys a buffer and creates it again (zeroed); the old handle goes stale.
    Recreate(usize),
    /// Reads every buffer back and compares it with the model.
    ReadAll,
}

fn data(len: u64, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

const fn aligned(value: u64) -> bool {
    value.is_multiple_of(4)
}

fn within(offset: u64, size: u64, len: u64) -> bool {
    offset.checked_add(size).is_some_and(|end| end <= len)
}

struct Harness {
    device: Device,
    handles: Vec<Buffer>,
    stale: Vec<Buffer>,
    contents: Vec<Vec<u8>>,
}

/// Why the model rejects a command.
#[derive(Debug, PartialEq)]
enum Reject {
    Stale,
    Invalid,
}

impl Harness {
    /// The buffer a reference names: `Ok(index)` for a live one.
    fn resolve(&self, r: Ref) -> (Buffer, Option<usize>) {
        if r.0 < BUFFERS.len() || self.stale.is_empty() {
            let index = r.0 % BUFFERS.len();
            (self.handles[index], Some(index))
        } else {
            (self.stale[(r.0 - BUFFERS.len()) % self.stale.len()], None)
        }
    }

    /// Applies a command to `contents` if the rules allow it.
    fn model(&self, contents: &mut [Vec<u8>], command: &Command) -> Result<(), Reject> {
        match *command {
            Command::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let (Some(s), Some(d)) = (self.resolve(src).1, self.resolve(dst).1) else {
                    return Err(Reject::Stale);
                };
                let (from, to) = (BUFFERS[s], BUFFERS[d]);
                let ok = s != d
                    && from.1.contains(BufferUsages::COPY_SRC)
                    && to.1.contains(BufferUsages::COPY_DST)
                    && aligned(src_offset)
                    && aligned(dst_offset)
                    && aligned(size)
                    && within(src_offset, size, from.0)
                    && within(dst_offset, size, to.0);
                if !ok {
                    return Err(Reject::Invalid);
                }
                let (so, dof, n) = (src_offset as usize, dst_offset as usize, size as usize);
                let bytes = contents[s][so..so + n].to_vec();
                contents[d][dof..dof + n].copy_from_slice(&bytes);
                Ok(())
            }
            Command::Clear {
                buffer,
                offset,
                size,
            } => {
                let Some(b) = self.resolve(buffer).1 else {
                    return Err(Reject::Stale);
                };
                let (len, usage) = BUFFERS[b];
                if !usage.contains(BufferUsages::COPY_DST) {
                    return Err(Reject::Invalid);
                }
                let Some(size) = size.or_else(|| len.checked_sub(offset)) else {
                    return Err(Reject::Invalid);
                };
                if !(aligned(offset) && aligned(size) && within(offset, size, len)) {
                    return Err(Reject::Invalid);
                }
                contents[b][offset as usize..(offset + size) as usize].fill(0);
                Ok(())
            }
        }
    }

    fn record(&self, commands: &[Command]) -> pragma_rhi::CommandBuffer {
        let mut encoder = CommandEncoder::new(Some("transfers"));
        for command in commands {
            match *command {
                Command::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => encoder.copy_buffer_to_buffer(
                    self.resolve(src).0,
                    src_offset,
                    self.resolve(dst).0,
                    dst_offset,
                    size,
                ),
                Command::Clear {
                    buffer,
                    offset,
                    size,
                } => encoder.clear_buffer(self.resolve(buffer).0, offset, size),
            }
        }
        encoder.finish()
    }

    fn apply(&mut self, op: Op) {
        match op {
            Op::Write {
                buffer,
                offset,
                len,
                seed,
            } => {
                let bytes = data(len, seed);
                let (handle, index) = self.resolve(buffer);
                let result = self.device.write_buffer(handle, offset, &bytes);
                match index {
                    None => assert!(
                        matches!(result, Err(RhiError::StaleHandle { .. })),
                        "a write to a destroyed buffer is StaleHandle, got {result:?}"
                    ),
                    Some(b) => {
                        let (size, usage) = BUFFERS[b];
                        let ok = usage.contains(BufferUsages::COPY_DST)
                            && aligned(offset)
                            && aligned(len)
                            && within(offset, len, size);
                        if ok {
                            assert_eq!(result, Ok(()));
                            self.contents[b][offset as usize..(offset + len) as usize]
                                .copy_from_slice(&bytes);
                        } else {
                            assert!(
                                matches!(
                                    result,
                                    Err(RhiError::InvalidCommand {
                                        command: "write_buffer",
                                        ..
                                    })
                                ),
                                "the write breaks a rule, got {result:?}"
                            );
                        }
                    }
                }
            }
            Op::Submit(buffers) => {
                let mut contents = self.contents.clone();
                let mut verdict = Ok(());
                'outer: for commands in &buffers {
                    for command in commands {
                        if let Err(reject) = self.model(&mut contents, command) {
                            verdict = Err(reject);
                            break 'outer;
                        }
                    }
                }
                let recorded: Vec<_> = buffers.iter().map(|c| self.record(c)).collect();
                let before = self.device.last_submission();
                let result = self.device.submit(recorded);
                match verdict {
                    Ok(()) => {
                        let index = result.expect("a valid submission");
                        assert!(before.is_none_or(|b| b < index));
                        self.contents = contents;
                    }
                    Err(Reject::Stale) => {
                        assert!(
                            matches!(result, Err(RhiError::StaleHandle { .. })),
                            "a command naming a destroyed buffer is StaleHandle, got {result:?}"
                        );
                        assert_eq!(self.device.last_submission(), before);
                    }
                    Err(Reject::Invalid) => {
                        assert!(
                            matches!(result, Err(RhiError::InvalidCommand { .. })),
                            "a broken command rejects the submission, got {result:?}"
                        );
                        assert_eq!(self.device.last_submission(), before);
                    }
                }
            }
            Op::Recreate(n) => {
                let index = n % BUFFERS.len();
                let old = self.handles[index];
                self.device.destroy_buffer(old).expect("destroy");
                self.stale.push(old);
                self.handles[index] = create(&mut self.device, index);
                self.contents[index].fill(0);
            }
            Op::ReadAll => self.read_all(),
        }
    }

    fn read_all(&mut self) {
        for (index, (size, usage)) in BUFFERS.iter().enumerate() {
            let readable = usage.intersects(BufferUsages::MAP_READ | BufferUsages::COPY_SRC);
            let len = size - size % 4;
            let got = self.device.read_buffer(self.handles[index], 0, len);
            if readable {
                assert_eq!(
                    got.expect("a readable buffer"),
                    self.contents[index][..len as usize],
                    "buffer {index}"
                );
            } else {
                assert!(matches!(got, Err(RhiError::InvalidCommand { .. })));
            }
        }
    }
}

fn create(device: &mut Device, index: usize) -> Buffer {
    let (size, usage) = BUFFERS[index];
    device
        .create_buffer(&BufferDescriptor {
            label: Some("transfers"),
            size,
            usage,
        })
        .expect("a buffer")
}

/// Runs the operations against the model and reads every buffer back at the end.
pub fn check(ops: impl IntoIterator<Item = Op>) {
    let mut device = super::null_device(1 << 20, 2);
    let handles = (0..BUFFERS.len()).map(|i| create(&mut device, i)).collect();
    let mut h = Harness {
        device,
        handles,
        stale: Vec::new(),
        contents: BUFFERS
            .iter()
            .map(|(size, _)| vec![0; *size as usize])
            .collect(),
    };
    for op in ops {
        h.apply(op);
    }
    h.read_all();
}

/// An offset or size: mostly multiples of 4 within the buffers, sometimes not.
fn amount(bytes: &mut Bytes<'_>) -> u64 {
    match bytes.below(8) {
        0 => u64::from(bytes.byte()),
        1 => u64::from(bytes.u32()) | (u64::from(bytes.u32()) << 32),
        _ => u64::from(bytes.below(33)) * 4,
    }
}

fn reference(bytes: &mut Bytes<'_>) -> Ref {
    Ref(usize::from(bytes.below(BUFFERS.len() as u8 + 2)))
}

fn command(bytes: &mut Bytes<'_>) -> Command {
    if bytes.below(3) == 0 {
        let buffer = reference(bytes);
        let offset = amount(bytes);
        let size = if bytes.bool() {
            Some(amount(bytes))
        } else {
            None
        };
        Command::Clear {
            buffer,
            offset,
            size,
        }
    } else {
        Command::Copy {
            src: reference(bytes),
            src_offset: amount(bytes),
            dst: reference(bytes),
            dst_offset: amount(bytes),
            size: amount(bytes),
        }
    }
}

/// Arbitrary bytes as a sequence of operations.
pub fn decode(data: &[u8]) -> Vec<Op> {
    let mut bytes = Bytes::new(data);
    let mut ops = Vec::new();
    while !bytes.is_empty() && ops.len() < MAX_OPS {
        let op = match bytes.below(10) {
            0..=2 => Op::Write {
                buffer: reference(&mut bytes),
                offset: amount(&mut bytes),
                // Small: the bytes are allocated.
                len: match bytes.below(4) {
                    0 => u64::from(bytes.byte()),
                    _ => u64::from(bytes.below(33)) * 4,
                },
                seed: bytes.byte(),
            },
            3..=7 => {
                let count = 1 + usize::from(bytes.below(MAX_BUFFERS as u8));
                let buffers = (0..count)
                    .map(|_| {
                        let n = 1 + usize::from(bytes.below(MAX_COMMANDS as u8));
                        (0..n).map(|_| command(&mut bytes)).collect()
                    })
                    .collect();
                Op::Submit(buffers)
            }
            8 => Op::Recreate(bytes.index()),
            _ => Op::ReadAll,
        };
        ops.push(op);
    }
    ops
}
