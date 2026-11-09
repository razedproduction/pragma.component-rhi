//! Device limits that `pragma_rhi` validates descriptors against.

use crate::RhiError;

/// The limits of a device: the largest sizes and counts a descriptor may request.
///
/// Requested in [`DeviceDescriptor::required_limits`](crate::DeviceDescriptor) and checked against
/// the adapter's [`Adapter::limits`](crate::Adapter::limits). The device validates every
/// descriptor against the limits it was created with, not the adapter's, so a program tested
/// with the defaults behaves the same on a stronger GPU. The defaults are WebGPU's.
///
/// Every `max_*` limit is an upper bound (a device may request at most the adapter's value);
/// the two `min_*_offset_alignment` limits are lower bounds (a device may request at least the
/// adapter's value, and only a power of two).
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
    /// The largest number of bind groups in a pipeline layout.
    pub max_bind_groups: u32,
    /// The largest number of bind groups plus vertex buffers of a render pipeline.
    pub max_bind_groups_plus_vertex_buffers: u32,
    /// One more than the largest binding number in a bind group layout.
    pub max_bindings_per_bind_group: u32,
    /// The largest number of uniform buffers with a dynamic offset in a pipeline layout.
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// The largest number of storage buffers with a dynamic offset in a pipeline layout.
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// The largest number of sampled textures one shader stage sees in a pipeline layout.
    pub max_sampled_textures_per_shader_stage: u32,
    /// The largest number of samplers one shader stage sees in a pipeline layout.
    pub max_samplers_per_shader_stage: u32,
    /// The largest number of storage buffers one shader stage sees in a pipeline layout.
    pub max_storage_buffers_per_shader_stage: u32,
    /// The largest number of storage textures one shader stage sees in a pipeline layout.
    pub max_storage_textures_per_shader_stage: u32,
    /// The largest number of uniform buffers one shader stage sees in a pipeline layout.
    pub max_uniform_buffers_per_shader_stage: u32,
    /// The largest size in bytes of a uniform buffer binding.
    pub max_uniform_buffer_binding_size: u64,
    /// The largest size in bytes of a storage buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// The alignment in bytes of a uniform buffer binding's offset (a power of two).
    pub min_uniform_buffer_offset_alignment: u32,
    /// The alignment in bytes of a storage buffer binding's offset (a power of two).
    pub min_storage_buffer_offset_alignment: u32,
    /// The largest number of vertex buffers of a render pipeline.
    pub max_vertex_buffers: u32,
    /// The largest number of vertex attributes of a render pipeline, over all its buffers.
    pub max_vertex_attributes: u32,
    /// The largest stride in bytes of a vertex buffer.
    pub max_vertex_buffer_array_stride: u32,
    /// The largest number of color targets of a render pipeline.
    pub max_color_attachments: u32,
    /// The largest number of invocations in a compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// The largest `x` size of a compute workgroup.
    pub max_compute_workgroup_size_x: u32,
    /// The largest `y` size of a compute workgroup.
    pub max_compute_workgroup_size_y: u32,
    /// The largest `z` size of a compute workgroup.
    pub max_compute_workgroup_size_z: u32,
    /// The largest number of workgroups a dispatch may have in each dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_texture_array_layers: 256,
            max_buffer_size: 1 << 28,
            max_bind_groups: 4,
            max_bind_groups_plus_vertex_buffers: 24,
            max_bindings_per_bind_group: 1000,
            max_dynamic_uniform_buffers_per_pipeline_layout: 8,
            max_dynamic_storage_buffers_per_pipeline_layout: 4,
            max_sampled_textures_per_shader_stage: 16,
            max_samplers_per_shader_stage: 16,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_textures_per_shader_stage: 4,
            max_uniform_buffers_per_shader_stage: 12,
            max_uniform_buffer_binding_size: 64 << 10,
            max_storage_buffer_binding_size: 128 << 20,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_vertex_buffers: 8,
            max_vertex_attributes: 16,
            max_vertex_buffer_array_stride: 2048,
            max_color_attachments: 8,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}

/// Applies `$m!(field, is_upper_bound)` to every limit, in field order.
macro_rules! for_each_limit {
    ($m:ident) => {
        $m! {
            (max_texture_dimension_1d, true),
            (max_texture_dimension_2d, true),
            (max_texture_dimension_3d, true),
            (max_texture_array_layers, true),
            (max_buffer_size, true),
            (max_bind_groups, true),
            (max_bind_groups_plus_vertex_buffers, true),
            (max_bindings_per_bind_group, true),
            (max_dynamic_uniform_buffers_per_pipeline_layout, true),
            (max_dynamic_storage_buffers_per_pipeline_layout, true),
            (max_sampled_textures_per_shader_stage, true),
            (max_samplers_per_shader_stage, true),
            (max_storage_buffers_per_shader_stage, true),
            (max_storage_textures_per_shader_stage, true),
            (max_uniform_buffers_per_shader_stage, true),
            (max_uniform_buffer_binding_size, true),
            (max_storage_buffer_binding_size, true),
            (min_uniform_buffer_offset_alignment, false),
            (min_storage_buffer_offset_alignment, false),
            (max_vertex_buffers, true),
            (max_vertex_attributes, true),
            (max_vertex_buffer_array_stride, true),
            (max_color_attachments, true),
            (max_compute_invocations_per_workgroup, true),
            (max_compute_workgroup_size_x, true),
            (max_compute_workgroup_size_y, true),
            (max_compute_workgroup_size_z, true),
            (max_compute_workgroups_per_dimension, true),
        }
    };
}

macro_rules! limit_entries {
    ($(($field:ident, $upper:expr)),* $(,)?) => {
        fn entries(&self) -> Vec<(&'static str, u64, bool)> {
            vec![$((stringify!($field), u64::from(self.$field), $upper)),*]
        }

        pub(crate) fn from_wgpu(limits: &wgpu::Limits) -> Self {
            Self { $($field: limits.$field),* }
        }

        pub(crate) fn apply_to_wgpu(&self, limits: &mut wgpu::Limits) {
            $(limits.$field = self.$field;)*
        }
    };
}

impl Limits {
    for_each_limit!(limit_entries);

    /// Checks that every upper-bound limit is at most, and every alignment at least, the
    /// matching one of `available`, and that the alignments are powers of two.
    pub(crate) fn check_within(&self, available: &Self) -> Result<(), RhiError> {
        for ((name, requested, upper), (_, bound, _)) in
            self.entries().into_iter().zip(available.entries())
        {
            if upper && requested > bound {
                return Err(RhiError::limit(name, requested, bound));
            }
            if !upper && (requested < bound || !requested.is_power_of_two()) {
                return Err(RhiError::LimitBelowMinimum {
                    limit: name,
                    requested,
                    min: bound,
                });
            }
        }
        Ok(())
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

    #[test]
    fn alignments_are_lower_bounds() {
        let available = Limits::default();
        let looser = Limits {
            min_uniform_buffer_offset_alignment: 512,
            ..Limits::default()
        };
        assert_eq!(looser.check_within(&available), Ok(()));
        let stricter = Limits {
            min_storage_buffer_offset_alignment: 64,
            ..Limits::default()
        };
        assert!(matches!(
            stricter.check_within(&available),
            Err(RhiError::LimitBelowMinimum {
                limit: "min_storage_buffer_offset_alignment",
                requested: 64,
                min: 256,
            })
        ));
        let odd = Limits {
            min_uniform_buffer_offset_alignment: 300,
            ..Limits::default()
        };
        assert!(odd.check_within(&available).is_err());
    }
}
