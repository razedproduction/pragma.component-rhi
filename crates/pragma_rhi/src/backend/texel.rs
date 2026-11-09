//! Texel encoding and texture memory layout for the null backend.
//!
//! The null backend keeps texture memory linear: layer after layer, each layer its mip levels
//! from 0, each level row after row, with no padding and one sample per texel. Clear values
//! are encoded the way WebGPU converts them: unorm channels clamp to `0..=1` and round to
//! nearest, sRGB formats encode the color channels (not alpha) first, half floats round to
//! nearest even.

use crate::resource::mip_extent;
use crate::{Color, TextureDimension, TextureFormat, TextureInfo};

/// The linear layout of a null texture's memory.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout<'a> {
    info: &'a TextureInfo,
    block: u64,
}

impl<'a> Layout<'a> {
    pub(crate) const fn new(info: &'a TextureInfo) -> Self {
        Self {
            info,
            block: info.format.block_size() as u64,
        }
    }

    pub(crate) const fn block(&self) -> u64 {
        self.block
    }

    /// The bytes of one mip level of one layer (all slices of a 3D level).
    fn level_bytes(&self, mip: u32) -> u64 {
        let size = mip_extent(self.info, mip);
        let depth = match self.info.dimension {
            TextureDimension::D3 => u64::from(size.depth_or_array_layers),
            TextureDimension::D1 | TextureDimension::D2 => 1,
        };
        u64::from(size.width)
            .saturating_mul(u64::from(size.height))
            .saturating_mul(depth)
            .saturating_mul(self.block)
    }

    fn level_offset(&self, mip: u32) -> u64 {
        (0..mip).fold(0_u64, |sum, level| {
            sum.saturating_add(self.level_bytes(level))
        })
    }

    fn layer_bytes(&self) -> u64 {
        self.level_offset(self.info.mip_level_count)
    }

    /// The bytes of the whole texture, one sample per texel.
    pub(crate) fn total(&self) -> u64 {
        let layers = match self.info.dimension {
            TextureDimension::D3 => 1,
            TextureDimension::D1 | TextureDimension::D2 => self.info.size.depth_or_array_layers,
        };
        self.layer_bytes().saturating_mul(u64::from(layers))
    }

    /// The offset of texel `(0, y)` of layer or slice `z` of mip level `mip`.
    pub(crate) fn row(&self, mip: u32, y: u32, z: u32) -> u64 {
        let size = mip_extent(self.info, mip);
        let row_bytes = u64::from(size.width).saturating_mul(self.block);
        let level = self.level_offset(mip);
        match self.info.dimension {
            TextureDimension::D3 => {
                let image = u64::from(size.height).saturating_mul(row_bytes);
                level
                    .saturating_add(u64::from(z).saturating_mul(image))
                    .saturating_add(u64::from(y).saturating_mul(row_bytes))
            }
            TextureDimension::D1 | TextureDimension::D2 => u64::from(z)
                .saturating_mul(self.layer_bytes())
                .saturating_add(level)
                .saturating_add(u64::from(y).saturating_mul(row_bytes)),
        }
    }
}

/// An encoded texel of at most 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Texel {
    bytes: [u8; 16],
    len: usize,
}

impl Texel {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.bytes.get(..self.len).unwrap_or(&[])
    }

    fn push(&mut self, part: &[u8]) {
        for byte in part {
            if let Some(slot) = self.bytes.get_mut(self.len) {
                *slot = *byte;
                self.len = self.len.saturating_add(1);
            }
        }
    }

    fn of(parts: &[&[u8]]) -> Self {
        let mut texel = Self {
            bytes: [0; 16],
            len: 0,
        };
        for part in parts {
            texel.push(part);
        }
        texel
    }

    fn f32s(channels: &[f64]) -> Self {
        let mut texel = Self::of(&[]);
        for channel in channels {
            texel.push(&(*channel as f32).to_le_bytes());
        }
        texel
    }

    fn f16s(channels: &[f64]) -> Self {
        let mut texel = Self::of(&[]);
        for channel in channels {
            texel.push(&f16_bits(*channel as f32).to_le_bytes());
        }
        texel
    }
}

/// Clamped to `0..=255` before the cast.
fn unorm8(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn srgb8(linear: f64) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    unorm8(encoded)
}

/// `f32` to IEEE half precision bits, rounding to nearest even.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "bit manipulation on values whose ranges are checked on every path"
)]
pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        // Infinity, or a quiet NaN.
        let nan = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7c00 | nan;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // A subnormal half, or zero.
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let half = full >> shift;
        let rest = full & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        let round_up = rest > midpoint || (rest == midpoint && half & 1 == 1);
        return sign | (half + u32::from(round_up)) as u16;
    }
    let half = ((half_exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    let round_up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa rounds up into the exponent, as it should.
    sign | (half + u32::from(round_up)) as u16
}

/// The bytes of a color clear value in `format`; `None` for a depth format.
pub(crate) fn encode_color(format: TextureFormat, color: Color) -> Option<Texel> {
    let Color { r, g, b, a } = color;
    Some(match format {
        TextureFormat::R8Unorm => Texel::of(&[&[unorm8(r)]]),
        TextureFormat::Rg8Unorm => Texel::of(&[&[unorm8(r), unorm8(g)]]),
        TextureFormat::Rgba8Unorm => Texel::of(&[&[unorm8(r), unorm8(g), unorm8(b), unorm8(a)]]),
        TextureFormat::Rgba8UnormSrgb => Texel::of(&[&[srgb8(r), srgb8(g), srgb8(b), unorm8(a)]]),
        TextureFormat::Bgra8Unorm => Texel::of(&[&[unorm8(b), unorm8(g), unorm8(r), unorm8(a)]]),
        TextureFormat::Bgra8UnormSrgb => Texel::of(&[&[srgb8(b), srgb8(g), srgb8(r), unorm8(a)]]),
        TextureFormat::R16Float => Texel::f16s(&[r]),
        TextureFormat::Rgba16Float => Texel::f16s(&[r, g, b, a]),
        TextureFormat::R32Float => Texel::f32s(&[r]),
        TextureFormat::Rg32Float => Texel::f32s(&[r, g]),
        TextureFormat::Rgba32Float => Texel::f32s(&[r, g, b, a]),
        // An integer clear value converts by truncation, saturating at the range (as `as`).
        TextureFormat::R32Uint => Texel::of(&[&(r as u32).to_le_bytes()]),
        TextureFormat::Depth32Float
        | TextureFormat::Depth24Plus
        | TextureFormat::Depth24PlusStencil8 => return None,
    })
}

/// The bytes of a depth clear value; only `Depth32Float` has a layout copies can observe.
pub(crate) fn encode_depth(format: TextureFormat, depth: f32) -> Option<Texel> {
    match format {
        TextureFormat::Depth32Float => Some(Texel::of(&[&depth.to_le_bytes()])),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{Layout, encode_color, f16_bits};
    use crate::{Color, Extent3d, TextureDimension, TextureFormat, TextureInfo, TextureUsages};

    #[test]
    fn half_floats_round_to_nearest_even() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1.0e6), 0x7c00);
        assert_eq!(f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits(f32::NAN) & 0x7e00, 0x7e00);
        // The smallest subnormal, and a value that rounds to zero.
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(2.0e-8), 0x0000);
        // 1 + 2^-11 lies halfway between 1 and the next half; the tie goes to even.
        assert_eq!(f16_bits(1.000_488_3), 0x3c00);
    }

    #[test]
    fn colors_encode_per_format() {
        let color = Color::new(0.2, 0.4, 1.0, 1.0);
        let bytes = |format, color| encode_color(format, color).unwrap().as_bytes().to_vec();
        assert_eq!(bytes(TextureFormat::Rgba8Unorm, color), [51, 102, 255, 255]);
        assert_eq!(bytes(TextureFormat::Bgra8Unorm, color), [255, 102, 51, 255]);
        assert_eq!(
            bytes(
                TextureFormat::Rgba8UnormSrgb,
                Color::new(0.0, 1.0, 0.5, 0.5)
            ),
            [0, 255, 188, 128]
        );
        assert_eq!(
            bytes(TextureFormat::R32Float, Color::new(0.5, 0.0, 0.0, 0.0)),
            0.5_f32.to_le_bytes()
        );
        assert_eq!(
            bytes(TextureFormat::Rgba16Float, Color::WHITE),
            [0, 0x3c, 0, 0x3c, 0, 0x3c, 0, 0x3c]
        );
        assert!(encode_color(TextureFormat::Depth32Float, color).is_none());
    }

    #[test]
    fn layout_is_linear_per_layer_and_level() {
        let info = TextureInfo {
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 2,
            },
            mip_level_count: 3,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_DST,
        };
        let layout = Layout::new(&info);
        // Levels of 64, 16 and 4 bytes: 84 per layer.
        assert_eq!(layout.total(), 168);
        assert_eq!(layout.row(0, 1, 0), 16);
        assert_eq!(layout.row(1, 0, 0), 64);
        assert_eq!(layout.row(2, 0, 1), 84 + 80);

        let volume = TextureInfo {
            dimension: TextureDimension::D3,
            mip_level_count: 1,
            ..info
        };
        let layout = Layout::new(&volume);
        assert_eq!(layout.total(), 128);
        assert_eq!(layout.row(0, 2, 1), 64 + 32);
    }
}
