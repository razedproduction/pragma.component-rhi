//! Device limits that `pragma_rhi` validates descriptors against.

use crate::RhiError;

/// The limits of a device: the largest sizes a descriptor may request.
///
/// Requested in [`DeviceDescriptor::required_limits`](crate::DeviceDescriptor) and checked against
/// the adapter's [`Adapter::limits`](crate::Adapter::limits). The device validates every
/// descriptor against the limits it was created with, not the adapter's, so a program tested
/// with the defaults behaves the same on a stronger GPU. The defaults are WebGPU's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Limits {
    /// The largest `width` of a 1D texture.
    pub max_texture_dimension_1d: u32,
    /// The largest `width` and `height` of a 2D texture.
    pub max_texture_dimension_2d: u32,
    /// The largest `width`, `height` and `depth_or_array_layers` of a 3D texture.
    pub max_texture_dimension_3d: u32,
    /// The largest number of array layers of a 1D or 2D texture.
    pub max_texture_array_layers: u32,
    /// The largest buffer size in bytes.
    pub max_buffer_size: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_buffer_size: 1 << 28,
        }
    }
}

impl Limits {
    /// Checks that every limit is at most the matching one of `available`.
    pub(crate) fn check_within(&self, available: &Self) -> Result<(), RhiError> {
        let pairs: [(&'static str, u64, u64); 5] = [
            (
                "max_texture_dimension_1d",
                self.max_texture_dimension_1d.into(),
                available.max_texture_dimension_1d.into(),
            ),
            (
                "max_texture_dimension_2d",
                self.max_texture_dimension_2d.into(),
                available.max_texture_dimension_2d.into(),
            ),
            (
                "max_texture_dimension_3d",
                self.max_texture_dimension_3d.into(),
                available.max_texture_dimension_3d.into(),
            ),
            (
                "max_texture_array_layers",
                self.max_texture_array_layers.into(),
                available.max_texture_array_layers.into(),
            ),
            (
                "max_buffer_size",
                self.max_buffer_size,
                available.max_buffer_size,
            ),
        ];
        for (name, requested, max) in pairs {
            if requested > max {
                return Err(RhiError::limit(name, requested, max));
            }
        }
        Ok(())
    }

    pub(crate) fn from_wgpu(limits: &wgpu::Limits) -> Self {
        Self {
            max_texture_dimension_1d: limits.max_texture_dimension_1d,
            max_texture_dimension_2d: limits.max_texture_dimension_2d,
            max_texture_dimension_3d: limits.max_texture_dimension_3d,
            max_texture_array_layers: limits.max_texture_array_layers,
            max_buffer_size: limits.max_buffer_size,
        }
    }

    pub(crate) fn apply_to_wgpu(&self, limits: &mut wgpu::Limits) {
        limits.max_texture_dimension_1d = self.max_texture_dimension_1d;
        limits.max_texture_dimension_2d = self.max_texture_dimension_2d;
        limits.max_texture_dimension_3d = self.max_texture_dimension_3d;
        limits.max_texture_array_layers = self.max_texture_array_layers;
        limits.max_buffer_size = self.max_buffer_size;
    }
}

#[cfg(test)]
mod tests {
    use super::Limits;
    use crate::RhiError;

    #[test]
    fn defaults_match_webgpu() {
        let ours = Limits::default();
        assert_eq!(ours, Limits::from_wgpu(&wgpu::Limits::default()));
    }

    #[test]
    fn check_within_names_the_limit() {
        let available = Limits::default();
        assert_eq!(available.check_within(&available), Ok(()));
        let requested = Limits {
            max_texture_array_layers: 1024,
            ..Limits::default()
        };
        assert_eq!(
            requested.check_within(&available),
            Err(RhiError::limit(
                "max_texture_array_layers",
                1024_u64,
                256_u64
            ))
        );
    }
}
