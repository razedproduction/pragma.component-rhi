//! Descriptor validation against independent models of the rules.
//!
//! Arbitrary buffer, texture, view and sampler descriptors — zero and huge sizes, every usage
//! combination, mip chains, sample counts, depth formats in every dimension, LOD ranges with
//! NaN and infinities — go to a null device. Buffer and sampler rules are modeled exactly. For
//! textures and views the model holds rules each of which alone rejects a descriptor: a
//! descriptor breaking one must fail, and an accepted one must satisfy them all and cost
//! exactly the bytes of its mip chain. Every failure must leave the device unchanged (live
//! counts, allocated bytes), and nothing may panic. Used by `tests/oracle.rs` and the
//! `descriptors` fuzz target.

use pragma_rhi::{
    AddressMode, BufferDescriptor, BufferUsages, CompareFunction, Device, Extent3d, FilterMode,
    Limits, ResourceKind, RhiError, SamplerDescriptor, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureInfo, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

use super::bytes::Bytes;

const BUDGET: u64 = 64 << 20;
const MAX_OPS: usize = 120;

pub const FORMATS: [TextureFormat; 15] = [
    TextureFormat::R8Unorm,
    TextureFormat::Rg8Unorm,
    TextureFormat::Rgba8Unorm,
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Bgra8Unorm,
    TextureFormat::Bgra8UnormSrgb,
    TextureFormat::R16Float,
    TextureFormat::Rgba16Float,
    TextureFormat::R32Float,
    TextureFormat::Rg32Float,
    TextureFormat::Rgba32Float,
    TextureFormat::R32Uint,
    TextureFormat::Depth32Float,
    TextureFormat::Depth24Plus,
    TextureFormat::Depth24PlusStencil8,
];
pub const DIMENSIONS: [TextureDimension; 3] = [
    TextureDimension::D1,
    TextureDimension::D2,
    TextureDimension::D3,
];
pub const VIEW_DIMENSIONS: [Option<TextureViewDimension>; 7] = [
    None,
    Some(TextureViewDimension::D1),
    Some(TextureViewDimension::D2),
    Some(TextureViewDimension::D2Array),
    Some(TextureViewDimension::Cube),
    Some(TextureViewDimension::CubeArray),
    Some(TextureViewDimension::D3),
];
pub const ASPECTS: [TextureAspect; 3] = [
    TextureAspect::All,
    TextureAspect::DepthOnly,
    TextureAspect::StencilOnly,
];
const ADDRESS: [AddressMode; 3] = [
    AddressMode::ClampToEdge,
    AddressMode::Repeat,
    AddressMode::MirrorRepeat,
];
const FILTERS: [FilterMode; 2] = [FilterMode::Nearest, FilterMode::Linear];
const COMPARE: [Option<CompareFunction>; 3] = [
    None,
    Some(CompareFunction::Less),
    Some(CompareFunction::Always),
];
const LODS: [f32; 8] = [0.0, 1.0, 4.5, 32.0, -1.0, f32::NAN, f32::INFINITY, 1000.0];

#[derive(Clone, Debug)]
pub struct TextureOp {
    pub size: Extent3d,
    pub mips: u32,
    pub samples: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
}

#[derive(Clone, Debug)]
pub struct ViewOp {
    /// Picks among the textures created so far.
    pub texture: usize,
    pub dimension: Option<TextureViewDimension>,
    pub aspect: TextureAspect,
    pub base_mip: u32,
    pub mips: Option<u32>,
    pub base_layer: u32,
    pub layers: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct SamplerOp {
    pub address: [AddressMode; 3],
    pub filters: [FilterMode; 3],
    pub lod: [f32; 2],
    pub compare: Option<CompareFunction>,
    pub anisotropy: u16,
}

#[derive(Clone, Debug)]
pub enum Op {
    Buffer(u64, BufferUsages),
    Texture(TextureOp),
    View(ViewOp),
    Sampler(SamplerOp),
}

/// Whether the buffer rules accept a descriptor (the budget aside).
fn buffer_valid(size: u64, usage: BufferUsages, limits: &Limits) -> bool {
    let map_read_ok = !usage.contains(BufferUsages::MAP_READ)
        || (BufferUsages::MAP_READ | BufferUsages::COPY_DST).contains(usage);
    let map_write_ok = !usage.contains(BufferUsages::MAP_WRITE)
        || (BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).contains(usage);
    !usage.is_empty() && map_read_ok && map_write_ok && size > 0 && size <= limits.max_buffer_size
}

fn sampler_valid(op: &SamplerOp) -> bool {
    let [min, max] = op.lod;
    let lod_ok = min.is_finite() && max.is_finite() && min >= 0.0 && max >= min;
    let linear = op.filters.iter().all(|f| *f == FilterMode::Linear);
    lod_ok && (1..=16).contains(&op.anisotropy) && (op.anisotropy == 1 || linear)
}

fn chain(side: u32) -> u32 {
    32 - side.leading_zeros()
}

fn has_depth(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Depth32Float
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
    )
}

/// Rules each of which alone rejects a texture descriptor.
fn texture_rules(op: &TextureOp, limits: &Limits) -> bool {
    let Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: d,
    } = op.size;
    let sides = match op.dimension {
        TextureDimension::D1 => h == 1 && d == 1 && w <= limits.max_texture_dimension_1d,
        TextureDimension::D2 => {
            w <= limits.max_texture_dimension_2d
                && h <= limits.max_texture_dimension_2d
                && d <= limits.max_texture_array_layers
        }
        TextureDimension::D3 => [w, h, d]
            .iter()
            .all(|side| *side <= limits.max_texture_dimension_3d),
    };
    let largest = match op.dimension {
        TextureDimension::D1 => w,
        TextureDimension::D2 => w.max(h),
        TextureDimension::D3 => w.max(h).max(d),
    };
    let render = op.usage.contains(TextureUsages::RENDER_ATTACHMENT);
    let samples_ok = match op.samples {
        1 => true,
        4 => op.dimension == TextureDimension::D2 && op.mips == 1 && d == 1 && render,
        _ => false,
    };
    !op.usage.is_empty()
        && w > 0
        && h > 0
        && d > 0
        && sides
        && (1..=chain(largest)).contains(&op.mips)
        && (!has_depth(op.format) || op.dimension == TextureDimension::D2)
        && !(render && op.dimension == TextureDimension::D1)
        && samples_ok
}

/// The memory of a texture: its mip chain, every layer and sample.
fn texture_bytes(info: &TextureInfo) -> u64 {
    let Extent3d {
        width,
        height,
        depth_or_array_layers: depth,
    } = info.size;
    let shrink = |side: u32, level: u32| u64::from((side >> level).max(1));
    let (layers, d3) = match info.dimension {
        TextureDimension::D2 => (depth, false),
        TextureDimension::D1 | TextureDimension::D3 => (1, info.dimension == TextureDimension::D3),
    };
    let per_layer: u64 = (0..info.mip_level_count)
        .map(|level| {
            let h = if info.dimension == TextureDimension::D1 {
                1
            } else {
                shrink(height, level)
            };
            let d = if d3 { shrink(depth, level) } else { 1 };
            shrink(width, level) * h * d * u64::from(info.format.block_size())
        })
        .sum();
    per_layer * u64::from(layers) * u64::from(info.sample_count)
}

const KINDS: [ResourceKind; 4] = [
    ResourceKind::Buffer,
    ResourceKind::Texture,
    ResourceKind::TextureView,
    ResourceKind::Sampler,
];

fn snapshot(device: &Device) -> ([usize; 4], Option<u64>) {
    (
        KINDS.map(|kind| device.live(kind)),
        device.allocated_bytes(),
    )
}

fn rejected<T: core::fmt::Debug>(result: &Result<T, RhiError>) -> bool {
    matches!(
        result,
        Err(RhiError::InvalidDescriptor { .. }
            | RhiError::LimitExceeded { .. }
            | RhiError::OutOfMemory { .. })
    )
}

/// Runs the operations against the models.
pub fn check(ops: impl IntoIterator<Item = Op>) {
    let mut device = super::null_device(BUDGET, 0);
    let limits = *device.limits();
    let mut textures: Vec<(Texture, TextureInfo)> = Vec::new();
    for op in ops {
        let before = snapshot(&device);
        let failed = match op {
            Op::Buffer(size, usage) => {
                let result = device.create_buffer(&BufferDescriptor {
                    label: Some("fuzz"),
                    size,
                    usage,
                });
                let valid = buffer_valid(size, usage, &limits);
                let fits = before.1.is_some_and(|used| used + size <= BUDGET);
                match &result {
                    Ok(buffer) => {
                        assert!(valid && fits, "accepted {size} bytes of {usage:?}");
                        let info = device.buffer_info(*buffer).expect("buffer info");
                        assert_eq!((info.size, info.usage), (size, usage));
                        assert_eq!(device.allocated_bytes(), before.1.map(|b| b + size));
                    }
                    Err(RhiError::OutOfMemory { .. }) => assert!(valid && !fits),
                    Err(err) => assert!(!valid, "rejected a valid buffer: {err}"),
                }
                result.is_err()
            }
            Op::Texture(op) => {
                let result = device.create_texture(&TextureDescriptor {
                    label: Some("fuzz"),
                    size: op.size,
                    mip_level_count: op.mips,
                    sample_count: op.samples,
                    dimension: op.dimension,
                    format: op.format,
                    usage: op.usage,
                });
                let rules = texture_rules(&op, &limits);
                match &result {
                    Ok(texture) => {
                        assert!(rules, "accepted a texture breaking a rule: {op:?}");
                        let info = device.texture_info(*texture).expect("texture info");
                        assert_eq!(
                            (info.size, info.mip_level_count, info.sample_count),
                            (op.size, op.mips, op.samples)
                        );
                        assert_eq!(
                            (info.dimension, info.format, info.usage),
                            (op.dimension, op.format, op.usage)
                        );
                        assert_eq!(
                            device.allocated_bytes(),
                            before.1.map(|b| b + texture_bytes(&info))
                        );
                        textures.push((*texture, info));
                    }
                    Err(_) => assert!(rejected(&result), "unexpected error {result:?}"),
                }
                result.is_err()
            }
            Op::View(op) if !textures.is_empty() => {
                let (texture, tex) = textures[op.texture % textures.len()];
                let result = device.create_texture_view(
                    texture,
                    &TextureViewDescriptor {
                        label: Some("fuzz"),
                        dimension: op.dimension,
                        aspect: op.aspect,
                        base_mip_level: op.base_mip,
                        mip_level_count: op.mips,
                        base_array_layer: op.base_layer,
                        array_layer_count: op.layers,
                    },
                );
                let layers = if tex.dimension == TextureDimension::D2 {
                    tex.size.depth_or_array_layers
                } else {
                    1
                };
                let in_range = |base: u32, count: Option<u32>, total: u32| {
                    let count = count.unwrap_or(total.saturating_sub(base));
                    count > 0 && base.checked_add(count).is_some_and(|end| end <= total)
                };
                let ranges = in_range(op.base_mip, op.mips, tex.mip_level_count)
                    && in_range(op.base_layer, op.layers, layers);
                match &result {
                    Ok(view) => {
                        assert!(
                            ranges,
                            "accepted a view outside its texture: {op:?} of {tex:?}"
                        );
                        let info = device.texture_view_info(*view).expect("view info");
                        assert_eq!((info.texture, info.base_mip_level), (texture, op.base_mip));
                        assert!(info.base_mip_level + info.mip_level_count <= tex.mip_level_count);
                        assert!(info.base_array_layer + info.array_layer_count <= layers);
                        assert_eq!(device.allocated_bytes(), before.1);
                    }
                    Err(_) => assert!(rejected(&result), "unexpected error {result:?}"),
                }
                result.is_err()
            }
            Op::View(_) => false,
            Op::Sampler(op) => {
                let result = device.create_sampler(&SamplerDescriptor {
                    label: Some("fuzz"),
                    address_mode_u: op.address[0],
                    address_mode_v: op.address[1],
                    address_mode_w: op.address[2],
                    mag_filter: op.filters[0],
                    min_filter: op.filters[1],
                    mipmap_filter: op.filters[2],
                    lod_min_clamp: op.lod[0],
                    lod_max_clamp: op.lod[1],
                    compare: op.compare,
                    anisotropy_clamp: op.anisotropy,
                });
                assert_eq!(result.is_ok(), sampler_valid(&op), "{op:?}: {result:?}");
                if let Ok(sampler) = result {
                    let info = device.sampler_info(sampler).expect("sampler info");
                    assert_eq!(info.anisotropy_clamp, op.anisotropy);
                    assert_eq!(info.compare, op.compare);
                }
                result.is_err()
            }
        };
        if failed {
            assert_eq!(
                snapshot(&device),
                before,
                "a failed creation changes nothing"
            );
        }
    }
}

fn side(bytes: &mut Bytes<'_>) -> u32 {
    match bytes.below(16) {
        0 => 0,
        1 => bytes.u32(),
        2 => 8192 + u32::from(bytes.below(2)),
        3 => 2048 + u32::from(bytes.below(2)),
        4..=7 => 1,
        _ => 1 + u32::from(bytes.below(70)),
    }
}

fn small(bytes: &mut Bytes<'_>) -> u32 {
    match bytes.below(8) {
        0 => bytes.u32(),
        _ => u32::from(bytes.below(10)),
    }
}

fn optional(bytes: &mut Bytes<'_>) -> Option<u32> {
    if bytes.bool() {
        Some(small(bytes))
    } else {
        None
    }
}

fn pick<T: Copy>(items: &[T], bytes: &mut Bytes<'_>) -> T {
    items[usize::from(bytes.byte()) % items.len()]
}

/// Arbitrary bytes as a sequence of operations.
pub fn decode(data: &[u8]) -> Vec<Op> {
    let mut bytes = Bytes::new(data);
    let mut ops = Vec::new();
    while !bytes.is_empty() && ops.len() < MAX_OPS {
        let op = match bytes.below(8) {
            0 | 1 => {
                let size = match bytes.below(6) {
                    0 => 0,
                    1 => u64::from(bytes.u32()) << bytes.below(40),
                    _ => u64::from(bytes.u16()),
                };
                Op::Buffer(size, BufferUsages::from_bits_truncate(bytes.u16()))
            }
            2..=4 => Op::Texture(TextureOp {
                size: Extent3d {
                    width: side(&mut bytes),
                    height: side(&mut bytes),
                    depth_or_array_layers: side(&mut bytes),
                },
                mips: small(&mut bytes),
                samples: pick(&[1, 1, 4, 2, 0], &mut bytes),
                dimension: pick(&DIMENSIONS, &mut bytes),
                format: pick(&FORMATS, &mut bytes),
                usage: TextureUsages::from_bits_truncate(bytes.byte()),
            }),
            5 | 6 => Op::View(ViewOp {
                texture: bytes.index(),
                dimension: pick(&VIEW_DIMENSIONS, &mut bytes),
                aspect: pick(&ASPECTS, &mut bytes),
                base_mip: small(&mut bytes),
                mips: optional(&mut bytes),
                base_layer: small(&mut bytes),
                layers: optional(&mut bytes),
            }),
            _ => Op::Sampler(SamplerOp {
                address: [0; 3].map(|_| pick(&ADDRESS, &mut bytes)),
                filters: [0; 3].map(|_| pick(&FILTERS, &mut bytes)),
                lod: [pick(&LODS, &mut bytes), pick(&LODS, &mut bytes)],
                compare: pick(&COMPARE, &mut bytes),
                anisotropy: pick(&[1, 1, 4, 16, 0, 17], &mut bytes),
            }),
        };
        ops.push(op);
    }
    ops
}
