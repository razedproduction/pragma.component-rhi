//! Submission: command validation, queue writes and readback.
//!
//! [`Device::submit`] checks every command of every command buffer against the device's
//! resources and limits before any of them reaches the backend, and resolves handles to
//! backend objects on the way; a broken rule rejects the whole submission. The rules are
//! WebGPU's for the commands this crate records — the ones wgpu would otherwise report as an
//! opaque validation error — so the null backend rejects exactly what wgpu rejects.

use core::ops::Range;

use crate::backend::{
    Batch, BufRegion, ColorOp, ComputeOp, ComputePassOp, DepthOp, Op, RenderOp, RenderPassOp,
    Subresource, TexRegion,
};
use crate::binding::{BindingType, BufferBindingType, StorageTextureAccess};
use crate::commands::{
    COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT, Command, ComputeCommand, ComputePassData,
    RenderCommand, RenderPassData,
};
use crate::device::{
    BufferEntry, ComputeEntry, GroupEntry, GroupLayoutEntry, PipelineLayoutEntry, RenderEntry,
    TextureEntry, ViewEntry,
};
use crate::handle::Table;
use crate::pipeline::{PipelineMeta, VertexStepMode};
use crate::resource::mip_extent;
use crate::{
    BindGroup, BindGroupLayout, BindingResource, Buffer, BufferDescriptor, BufferUsages,
    CommandBuffer, CommandEncoder, ComputePipeline, Device, Extent3d, IndexFormat, Limits, LoadOp,
    PipelineLayout, RenderPassColorAttachment, RenderPipeline, RhiError, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDimension,
};

// ---------------------------------------------------------------------------------------------
// Errors

/// Where a rule was broken: a command of a command buffer.
#[derive(Debug, Clone, Copy)]
struct Site<'a> {
    label: Option<&'a str>,
    command: &'static str,
}

impl Site<'_> {
    fn fail(&self, reason: impl Into<String>) -> RhiError {
        RhiError::InvalidCommand {
            label: self.label.map(str::to_owned),
            command: self.command,
            reason: reason.into(),
        }
    }

    const fn at(&self, command: &'static str) -> Self {
        Self {
            label: self.label,
            command,
        }
    }

    fn check(&self, ok: bool, reason: impl FnOnce() -> String) -> Result<(), RhiError> {
        if ok { Ok(()) } else { Err(self.fail(reason())) }
    }
}

const fn aligned(value: u64) -> bool {
    value.is_multiple_of(COPY_BUFFER_ALIGNMENT)
}

/// Whether `offset..offset + size` lies within `0..total`.
fn within(offset: u64, size: u64, total: u64) -> bool {
    offset.checked_add(size).is_some_and(|end| end <= total)
}

fn reversed(range: &Range<u32>) -> bool {
    range.start > range.end
}

// ---------------------------------------------------------------------------------------------
// Usage scopes

/// How a resource is used inside a usage scope (a render pass, or one dispatch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Use {
    /// Any read-only use: vertex, index, uniform, read-only storage, sampled.
    Read,
    /// A writable storage binding.
    Storage,
    /// A render pass attachment or resolve target.
    Attachment,
}

impl Use {
    /// WebGPU: a resource in a scope is either only read, or used only as writable storage, or
    /// is exactly one attachment.
    const fn compatible(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Read, Self::Read) | (Self::Storage, Self::Storage)
        )
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Storage => "writable storage",
            Self::Attachment => "attachment",
        }
    }
}

const fn overlaps(a: &Range<u32>, b: &Range<u32>) -> bool {
    a.start < b.end && b.start < a.end
}

#[derive(Debug, Default)]
struct Scope {
    buffers: Vec<(Buffer, Use)>,
    textures: Vec<(Texture, Range<u32>, Range<u32>, Use)>,
}

impl Scope {
    fn buffer(&mut self, site: Site<'_>, buffer: Buffer, usage: Use) -> Result<(), RhiError> {
        for (other, other_use) in &self.buffers {
            if *other != buffer {
                continue;
            }
            if *other_use == usage {
                return Ok(());
            }
            if !usage.compatible(*other_use) {
                return Err(site.fail(format!(
                    "a buffer is used as {} and as {} in one usage scope",
                    other_use.name(),
                    usage.name()
                )));
            }
        }
        self.buffers.push((buffer, usage));
        Ok(())
    }

    fn texture(
        &mut self,
        site: Site<'_>,
        texture: Texture,
        mips: Range<u32>,
        layers: Range<u32>,
        usage: Use,
    ) -> Result<(), RhiError> {
        for (other, other_mips, other_layers, other_use) in &self.textures {
            if *other != texture || !overlaps(other_mips, &mips) || !overlaps(other_layers, &layers)
            {
                continue;
            }
            if *other_use == usage && *other_mips == mips && *other_layers == layers {
                if usage == Use::Attachment {
                    return Err(site.fail("a texture subresource is attached twice"));
                }
                return Ok(());
            }
            if !usage.compatible(*other_use) {
                return Err(site.fail(format!(
                    "a texture subresource is used as {} and as {} in one usage scope",
                    other_use.name(),
                    usage.name()
                )));
            }
        }
        self.textures.push((texture, mips, layers, usage));
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Validation

/// Read access to every table commands refer to: the device's fields one by one, so the
/// backend stays free to be borrowed mutably while validated commands borrow the tables.
struct Ctx<'d> {
    limits: &'d Limits,
    buffers: &'d Table<Buffer, BufferEntry>,
    textures: &'d Table<Texture, TextureEntry>,
    views: &'d Table<TextureView, ViewEntry>,
    bind_group_layouts: &'d Table<BindGroupLayout, GroupLayoutEntry>,
    bind_groups: &'d Table<BindGroup, GroupEntry>,
    pipeline_layouts: &'d Table<PipelineLayout, PipelineLayoutEntry>,
    render_pipelines: &'d Table<RenderPipeline, RenderEntry>,
    compute_pipelines: &'d Table<ComputePipeline, ComputeEntry>,
}

/// A bind group set in a pass.
#[derive(Clone, Copy)]
struct Bound<'d> {
    group: &'d GroupEntry,
}

/// A vertex or index buffer range set in a pass.
#[derive(Clone, Copy)]
struct BufferRange {
    size: u64,
}

impl<'d> Ctx<'d> {
    fn buffer(&self, buffer: Buffer) -> Result<&'d BufferEntry, RhiError> {
        self.buffers.get(buffer)
    }

    fn batch(&self, buffer: &'d CommandBuffer) -> Result<Batch<'d>, RhiError> {
        if buffer.out_of_memory {
            return Err(RhiError::OutOfMemory { bytes: 0 });
        }
        let label = buffer.label.as_deref();
        let mut ops = Vec::new();
        ops.try_reserve_exact(buffer.commands.len())
            .map_err(|_| RhiError::OutOfMemory {
                bytes: buffer.commands.len() as u64,
            })?;
        for command in &buffer.commands {
            let site = Site {
                label,
                command: command.name(),
            };
            ops.push(self.command(site, command)?);
        }
        Ok(Batch { label, ops })
    }

    fn command(&self, site: Site<'_>, command: &'d Command) -> Result<Op<'d>, RhiError> {
        match command {
            Command::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                let (from, to) = (self.buffer(*src)?, self.buffer(*dst)?);
                site.check(src != dst, || {
                    "the source and the destination are the same buffer".to_owned()
                })?;
                site.check(from.info.usage.contains(BufferUsages::COPY_SRC), || {
                    "the source lacks COPY_SRC usage".to_owned()
                })?;
                site.check(to.info.usage.contains(BufferUsages::COPY_DST), || {
                    "the destination lacks COPY_DST usage".to_owned()
                })?;
                site.check(
                    aligned(*src_offset) && aligned(*dst_offset) && aligned(*size),
                    || "offsets and size must be multiples of 4".to_owned(),
                )?;
                site.check(within(*src_offset, *size, from.info.size), || {
                    "the source range is out of bounds".to_owned()
                })?;
                site.check(within(*dst_offset, *size, to.info.size), || {
                    "the destination range is out of bounds".to_owned()
                })?;
                Ok(Op::CopyBufferToBuffer {
                    src: &from.obj,
                    src_offset: *src_offset,
                    dst: &to.obj,
                    dst_offset: *dst_offset,
                    size: *size,
                })
            }
            Command::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let entry = self.buffer(*buffer)?;
                site.check(entry.info.usage.contains(BufferUsages::COPY_DST), || {
                    "the buffer lacks COPY_DST usage".to_owned()
                })?;
                let size = match size {
                    Some(size) => *size,
                    None => entry
                        .info
                        .size
                        .checked_sub(*offset)
                        .ok_or_else(|| site.fail("the offset is past the end of the buffer"))?,
                };
                site.check(aligned(*offset) && aligned(size), || {
                    "the offset and size must be multiples of 4".to_owned()
                })?;
                site.check(within(*offset, size, entry.info.size), || {
                    "the range is out of bounds".to_owned()
                })?;
                Ok(Op::ClearBuffer {
                    obj: &entry.obj,
                    offset: *offset,
                    size,
                })
            }
            Command::CopyBufferToTexture { src, dst, size } => {
                let texture = self.texture_region(site, dst, *size, TextureUsages::COPY_DST)?;
                let buffer = self.buffer_region(
                    site,
                    src,
                    texture.info.format,
                    *size,
                    BufferUsages::COPY_SRC,
                )?;
                Ok(Op::CopyBufferToTexture {
                    src: buffer,
                    dst: texture,
                    size: *size,
                })
            }
            Command::CopyTextureToBuffer { src, dst, size } => {
                let texture = self.texture_region(site, src, *size, TextureUsages::COPY_SRC)?;
                let buffer = self.buffer_region(
                    site,
                    dst,
                    texture.info.format,
                    *size,
                    BufferUsages::COPY_DST,
                )?;
                Ok(Op::CopyTextureToBuffer {
                    src: texture,
                    dst: buffer,
                    size: *size,
                })
            }
            Command::CopyTextureToTexture { src, dst, size } => {
                let from = self.texture_region(site, src, *size, TextureUsages::COPY_SRC)?;
                let to = self.texture_region(site, dst, *size, TextureUsages::COPY_DST)?;
                site.check(from.info.format == to.info.format, || {
                    "the textures have different formats".to_owned()
                })?;
                if src.texture == dst.texture && src.mip_level == dst.mip_level {
                    let layers = |origin: u32| {
                        origin..origin.saturating_add(size.depth_or_array_layers.max(1))
                    };
                    let separate = from.info.dimension != TextureDimension::D3
                        && !overlaps(&layers(src.origin.z), &layers(dst.origin.z));
                    site.check(separate, || {
                        "the source and destination subresources overlap".to_owned()
                    })?;
                }
                Ok(Op::CopyTextureToTexture {
                    src: from,
                    dst: to,
                    size: *size,
                })
            }
            Command::RenderPass(pass) => self.render_pass(site, pass).map(Op::RenderPass),
            Command::ComputePass(pass) => self.compute_pass(site, pass).map(Op::ComputePass),
        }
    }

    // -- copies ---------------------------------------------------------------------------

    fn texture_region(
        &self,
        site: Site<'_>,
        copy: &TexelCopyTextureInfo,
        size: Extent3d,
        usage: TextureUsages,
    ) -> Result<TexRegion<'d>, RhiError> {
        let entry = self.textures.get(copy.texture)?;
        let info = &entry.info;
        site.check(info.usage.contains(usage), || {
            format!("the texture lacks {usage:?} usage")
        })?;
        site.check(info.sample_count == 1, || {
            "multisampled textures cannot be copied".to_owned()
        })?;
        site.check(!info.format.has_depth(), || {
            "copies of depth/stencil textures are not supported".to_owned()
        })?;
        site.check(copy.mip_level < info.mip_level_count, || {
            format!(
                "mip level {} is out of range ({} levels)",
                copy.mip_level, info.mip_level_count
            )
        })?;
        let level = mip_extent(info, copy.mip_level);
        let fits = |origin: u32, extent: u32, side: u32| {
            origin.checked_add(extent).is_some_and(|end| end <= side)
        };
        site.check(
            fits(copy.origin.x, size.width, level.width)
                && fits(copy.origin.y, size.height, level.height)
                && fits(
                    copy.origin.z,
                    size.depth_or_array_layers,
                    level.depth_or_array_layers,
                ),
            || "the region is out of the mip level's bounds".to_owned(),
        )?;
        Ok(TexRegion {
            obj: &entry.obj,
            info,
            mip: copy.mip_level,
            origin: copy.origin,
        })
    }

    fn buffer_region(
        &self,
        site: Site<'_>,
        copy: &TexelCopyBufferInfo,
        format: TextureFormat,
        size: Extent3d,
        usage: BufferUsages,
    ) -> Result<BufRegion<'d>, RhiError> {
        let entry = self.buffer(copy.buffer)?;
        let layout = copy.layout;
        site.check(entry.info.usage.contains(usage), || {
            format!("the buffer lacks {usage:?} usage")
        })?;
        let block = u64::from(format.block_size());
        site.check(layout.offset.is_multiple_of(block), || {
            format!("the buffer offset is not a multiple of the texel size ({block})")
        })?;
        let row_bytes = u64::from(size.width).saturating_mul(block);
        let height = u64::from(size.height);
        let images = u64::from(size.depth_or_array_layers);
        let bytes_per_row = match layout.bytes_per_row {
            Some(bytes) => {
                site.check(bytes.is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT), || {
                    format!("bytes_per_row is not a multiple of {COPY_BYTES_PER_ROW_ALIGNMENT}")
                })?;
                site.check(u64::from(bytes) >= row_bytes, || {
                    "bytes_per_row is smaller than a row".to_owned()
                })?;
                u64::from(bytes)
            }
            None => {
                site.check(height <= 1 && images <= 1, || {
                    "bytes_per_row is required for more than one row".to_owned()
                })?;
                row_bytes
            }
        };
        let rows_per_image = match layout.rows_per_image {
            Some(rows) => {
                site.check(u64::from(rows) >= height, || {
                    "rows_per_image is smaller than the copy's height".to_owned()
                })?;
                u64::from(rows)
            }
            None => {
                site.check(images <= 1, || {
                    "rows_per_image is required for more than one image".to_owned()
                })?;
                height
            }
        };
        let required = if row_bytes == 0 || height == 0 || images == 0 {
            Some(0)
        } else {
            bytes_per_row
                .checked_mul(rows_per_image)
                .and_then(|image| image.checked_mul(images.saturating_sub(1)))
                .and_then(|full| {
                    bytes_per_row
                        .checked_mul(height.saturating_sub(1))
                        .and_then(|rows| full.checked_add(rows))
                })
                .and_then(|rows| rows.checked_add(row_bytes))
        };
        site.check(
            required.is_some_and(|required| within(layout.offset, required, entry.info.size)),
            || "the copy overruns the buffer".to_owned(),
        )?;
        Ok(BufRegion {
            obj: &entry.obj,
            layout,
            bytes_per_row,
            rows_per_image,
        })
    }

    // -- bind groups ----------------------------------------------------------------------

    /// Checks a `set_bind_group` and adds the group's resources to `scope` (render passes;
    /// compute adds them per dispatch).
    fn bind_group(
        &self,
        site: Site<'_>,
        index: u32,
        group: BindGroup,
        offsets: &'d [u32],
    ) -> Result<Bound<'d>, RhiError> {
        site.check(index < self.limits.max_bind_groups, || {
            format!(
                "group index {index} is not below max_bind_groups ({})",
                self.limits.max_bind_groups
            )
        })?;
        let entry = self.bind_groups.get(group)?;
        let layout = &self.bind_group_layouts.get(entry.info.layout)?.info;
        let limits = &self.limits;
        let mut dynamic = offsets.iter();
        for (bound, slot) in entry.info.entries.iter().zip(&layout.entries) {
            let BindingType::Buffer {
                ty,
                has_dynamic_offset: true,
                ..
            } = slot.ty
            else {
                continue;
            };
            let offset = dynamic.next().ok_or_else(|| {
                site.fail("fewer dynamic offsets than dynamic bindings in the layout")
            })?;
            let alignment = match ty {
                BufferBindingType::Uniform => limits.min_uniform_buffer_offset_alignment,
                BufferBindingType::Storage { .. } => limits.min_storage_buffer_offset_alignment,
            };
            site.check(offset.is_multiple_of(alignment.max(1)), || {
                format!("dynamic offset {offset} is not a multiple of {alignment}")
            })?;
            if let BindingResource::Buffer(binding) = bound.resource {
                let size = binding.size.map_or(0, core::num::NonZeroU64::get);
                let buffer = self.buffer(binding.buffer)?;
                site.check(
                    within(
                        binding.offset.saturating_add(u64::from(*offset)),
                        size,
                        buffer.info.size,
                    ),
                    || {
                        format!(
                            "dynamic offset {offset} moves binding {} out of its buffer",
                            bound.binding
                        )
                    },
                )?;
            }
        }
        site.check(dynamic.next().is_none(), || {
            "more dynamic offsets than dynamic bindings in the layout".to_owned()
        })?;
        Ok(Bound { group: entry })
    }

    /// Adds every resource of a bound group to a usage scope.
    fn scope_group(
        &self,
        site: Site<'_>,
        scope: &mut Scope,
        bound: Bound<'d>,
    ) -> Result<(), RhiError> {
        let layout = &self.bind_group_layouts.get(bound.group.info.layout)?.info;
        for (entry, slot) in bound.group.info.entries.iter().zip(&layout.entries) {
            match (entry.resource, slot.ty) {
                (BindingResource::Buffer(binding), BindingType::Buffer { ty, .. }) => {
                    let usage = match ty {
                        BufferBindingType::Storage { read_only: false } => Use::Storage,
                        BufferBindingType::Uniform | BufferBindingType::Storage { .. } => Use::Read,
                    };
                    scope.buffer(site, binding.buffer, usage)?;
                }
                (BindingResource::TextureView(view), ty) => {
                    let usage = match ty {
                        BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadOnly,
                            ..
                        } => Use::Read,
                        BindingType::StorageTexture { .. } => Use::Storage,
                        _ => Use::Read,
                    };
                    let info = &self.views.get(view)?.info;
                    scope.texture(
                        site,
                        info.texture,
                        info.base_mip_level
                            ..info.base_mip_level.saturating_add(info.mip_level_count),
                        info.base_array_layer
                            ..info.base_array_layer.saturating_add(info.array_layer_count),
                        usage,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Checks that the groups a pipeline layout needs are set with fitting layouts, and that
    /// late-sized buffer bindings are large enough.
    fn check_groups(
        &self,
        site: Site<'_>,
        layout: PipelineLayout,
        meta: &PipelineMeta,
        groups: &[Option<Bound<'d>>],
    ) -> Result<(), RhiError> {
        let layout = &self.pipeline_layouts.get(layout)?.info;
        for (index, needed) in layout.bind_group_layouts.iter().enumerate() {
            let bound = groups
                .get(index)
                .copied()
                .flatten()
                .ok_or_else(|| site.fail(format!("bind group {index} is not set")))?;
            if !self.same_layout(bound.group.info.layout, *needed)? {
                return Err(site.fail(format!(
                    "bind group {index} has a layout incompatible with the pipeline's"
                )));
            }
        }
        for late in &meta.late_sizes {
            let size = usize::try_from(late.group)
                .ok()
                .and_then(|group| groups.get(group).copied().flatten())
                .and_then(|bound| {
                    bound
                        .group
                        .info
                        .entries
                        .iter()
                        .find(|entry| entry.binding == late.binding)
                })
                .and_then(|entry| match entry.resource {
                    BindingResource::Buffer(binding) => binding.size,
                    _ => None,
                })
                .map_or(0, core::num::NonZeroU64::get);
            site.check(size >= late.min_size, || {
                format!(
                    "@group({}) @binding({}) is bound to {size} bytes, the shader needs {}",
                    late.group, late.binding, late.min_size
                )
            })?;
        }
        Ok(())
    }

    /// Bind group layouts are compatible when they are the same or have equal entries.
    fn same_layout(&self, a: BindGroupLayout, b: BindGroupLayout) -> Result<bool, RhiError> {
        if a == b {
            return Ok(true);
        }
        let layouts = &self.bind_group_layouts;
        Ok(layouts.get(a)?.info.entries == layouts.get(b)?.info.entries)
    }

    // -- render passes --------------------------------------------------------------------

    /// Resolves an attachment view: one mip level and one layer of a 2D texture.
    fn attachment(
        &self,
        site: Site<'_>,
        view: TextureView,
        depth: bool,
    ) -> Result<(Subresource<'d>, u32, u32), RhiError> {
        let view_entry = self.views.get(view)?;
        let info = &view_entry.info;
        let texture = self.textures.get(info.texture)?;
        site.check(
            texture
                .info
                .usage
                .contains(TextureUsages::RENDER_ATTACHMENT),
            || "an attachment's texture lacks RENDER_ATTACHMENT usage".to_owned(),
        )?;
        site.check(
            info.mip_level_count == 1 && info.array_layer_count == 1,
            || "an attachment view must cover one mip level and one layer".to_owned(),
        )?;
        site.check(
            matches!(
                info.dimension,
                TextureViewDimension::D2 | TextureViewDimension::D2Array
            ),
            || "an attachment view must be 2D".to_owned(),
        )?;
        site.check(info.aspect == TextureAspect::All, || {
            "an attachment view must cover every aspect".to_owned()
        })?;
        site.check(texture.info.format.has_depth() == depth, || {
            if depth {
                "the depth/stencil attachment does not have a depth format".to_owned()
            } else {
                "a color attachment has a depth format".to_owned()
            }
        })?;
        let size = mip_extent(&texture.info, info.base_mip_level);
        Ok((
            Subresource {
                view: &view_entry.obj,
                texture: &texture.obj,
                info: &texture.info,
                mip: info.base_mip_level,
                layer: info.base_array_layer,
            },
            size.width,
            size.height,
        ))
    }

    fn scope_attachment(
        &self,
        site: Site<'_>,
        scope: &mut Scope,
        view: TextureView,
    ) -> Result<(), RhiError> {
        let info = &self.views.get(view)?.info;
        scope.texture(
            site,
            info.texture,
            info.base_mip_level..info.base_mip_level.saturating_add(1),
            info.base_array_layer..info.base_array_layer.saturating_add(1),
            Use::Attachment,
        )
    }

    fn color_attachment(
        &self,
        site: Site<'_>,
        scope: &mut Scope,
        attachment: &RenderPassColorAttachment,
        shape: &mut Option<(u32, u32, u32)>,
    ) -> Result<ColorOp<'d>, RhiError> {
        let (target, width, height) = self.attachment(site, attachment.view, false)?;
        Self::same_shape(site, shape, (width, height, target.info.sample_count))?;
        self.scope_attachment(site, scope, attachment.view)?;
        let resolve = match attachment.resolve_target {
            Some(view) => {
                site.check(target.info.sample_count > 1, || {
                    "a resolve target needs a multisampled attachment".to_owned()
                })?;
                let (resolve, resolve_width, resolve_height) =
                    self.attachment(site, view, false)?;
                site.check(resolve.info.sample_count == 1, || {
                    "a resolve target must be single-sampled".to_owned()
                })?;
                site.check(resolve.info.format == target.info.format, || {
                    "a resolve target must have its attachment's format".to_owned()
                })?;
                site.check((resolve_width, resolve_height) == (width, height), || {
                    "a resolve target must have its attachment's size".to_owned()
                })?;
                self.scope_attachment(site, scope, view)?;
                Some(resolve)
            }
            None => None,
        };
        Ok(ColorOp {
            target,
            resolve,
            ops: attachment.ops,
        })
    }

    fn same_shape(
        site: Site<'_>,
        shape: &mut Option<(u32, u32, u32)>,
        this: (u32, u32, u32),
    ) -> Result<(), RhiError> {
        match shape {
            Some(first) => site.check(*first == this, || {
                "the attachments differ in size or sample count".to_owned()
            }),
            None => {
                *shape = Some(this);
                Ok(())
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one state machine over the pass's commands; splitting it scatters the state"
    )]
    fn render_pass(
        &self,
        site: Site<'_>,
        pass: &'d RenderPassData,
    ) -> Result<RenderPassOp<'d>, RhiError> {
        let limits = &self.limits;
        site.check(
            pass.colors.len() <= limits.max_color_attachments as usize,
            || {
                format!(
                    "{} color attachments exceed max_color_attachments ({})",
                    pass.colors.len(),
                    limits.max_color_attachments
                )
            },
        )?;
        site.check(
            pass.colors.iter().any(Option::is_some) || pass.depth_stencil.is_some(),
            || "a render pass needs at least one attachment".to_owned(),
        )?;
        let mut scope = Scope::default();
        let mut shape = None;
        let mut colors = Vec::new();
        let mut formats = Vec::new();
        for attachment in &pass.colors {
            match attachment {
                Some(attachment) => {
                    let color = self.color_attachment(site, &mut scope, attachment, &mut shape)?;
                    formats.push(Some(color.target.info.format));
                    colors.push(Some(color));
                }
                None => {
                    formats.push(None);
                    colors.push(None);
                }
            }
        }
        while formats.last() == Some(&None) {
            formats.pop();
        }
        let depth = match &pass.depth_stencil {
            Some(attachment) => {
                let (target, width, height) = self.attachment(site, attachment.view, true)?;
                Self::same_shape(site, &mut shape, (width, height, target.info.sample_count))?;
                self.scope_attachment(site, &mut scope, attachment.view)?;
                let format = target.info.format;
                let depth_ops = attachment
                    .depth_ops
                    .ok_or_else(|| site.fail("a depth format needs depth_ops"))?;
                if let LoadOp::Clear(value) = depth_ops.load {
                    site.check((0.0..=1.0).contains(&value), || {
                        format!("the depth clear value {value} is not in 0..=1")
                    })?;
                }
                site.check(
                    attachment.stencil_ops.is_some() == format.has_stencil(),
                    || {
                        if format.has_stencil() {
                            "a stencil format needs stencil_ops".to_owned()
                        } else {
                            "stencil_ops are given for a format without stencil".to_owned()
                        }
                    },
                )?;
                Some(DepthOp {
                    target,
                    depth: Some(depth_ops),
                    stencil: attachment.stencil_ops,
                })
            }
            None => None,
        };
        let depth_format = depth.as_ref().map(|depth| depth.target.info.format);
        let (width, height, samples) = shape.unwrap_or((0, 0, 1));

        let mut pipeline: Option<&'d RenderEntry> = None;
        let mut groups: Vec<Option<Bound<'d>>> = vec![None; limits.max_bind_groups as usize];
        let mut vertex: Vec<Option<BufferRange>> = vec![None; limits.max_vertex_buffers as usize];
        let mut index: Option<(IndexFormat, BufferRange)> = None;
        let mut ops = Vec::new();
        for command in &pass.commands {
            let site = site.at(command.name());
            let op = match command {
                RenderCommand::SetPipeline(handle) => {
                    let entry = self.render_pipelines.get(*handle)?;
                    let mut wanted = entry.info.color_formats.clone();
                    while wanted.last() == Some(&None) {
                        wanted.pop();
                    }
                    site.check(wanted == formats, || {
                        "the pipeline's color formats differ from the pass's attachments".to_owned()
                    })?;
                    site.check(entry.info.depth_stencil_format == depth_format, || {
                        "the pipeline's depth/stencil format differs from the pass's".to_owned()
                    })?;
                    site.check(entry.info.sample_count == samples, || {
                        "the pipeline's sample count differs from the pass's".to_owned()
                    })?;
                    pipeline = Some(entry);
                    RenderOp::SetPipeline(&entry.obj)
                }
                RenderCommand::SetBindGroup {
                    index: slot,
                    group,
                    offsets,
                } => {
                    let bound = self.bind_group(site, *slot, *group, offsets)?;
                    self.scope_group(site, &mut scope, bound)?;
                    if let Some(entry) = groups.get_mut(*slot as usize) {
                        *entry = Some(bound);
                    }
                    RenderOp::SetBindGroup(*slot, &bound.group.obj, offsets)
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                    size,
                } => {
                    let entry = self.buffer(*buffer)?;
                    site.check(*slot < limits.max_vertex_buffers, || {
                        format!("slot {slot} is not below max_vertex_buffers")
                    })?;
                    site.check(entry.info.usage.contains(BufferUsages::VERTEX), || {
                        "the buffer lacks VERTEX usage".to_owned()
                    })?;
                    let size = Self::range(site, entry, *offset, *size, 4)?;
                    scope.buffer(site, *buffer, Use::Read)?;
                    if let Some(entry) = vertex.get_mut(*slot as usize) {
                        *entry = Some(BufferRange { size });
                    }
                    RenderOp::SetVertexBuffer(*slot, &entry.obj, *offset, size)
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    format,
                    offset,
                    size,
                } => {
                    let entry = self.buffer(*buffer)?;
                    site.check(entry.info.usage.contains(BufferUsages::INDEX), || {
                        "the buffer lacks INDEX usage".to_owned()
                    })?;
                    let size = Self::range(site, entry, *offset, *size, index_size(*format))?;
                    scope.buffer(site, *buffer, Use::Read)?;
                    index = Some((*format, BufferRange { size }));
                    RenderOp::SetIndexBuffer(&entry.obj, *format, *offset, size)
                }
                RenderCommand::SetViewport {
                    x,
                    y,
                    width: w,
                    height: h,
                    min_depth,
                    max_depth,
                } => {
                    let values = [*x, *y, *w, *h, *min_depth, *max_depth];
                    let max = limits.max_texture_dimension_2d as f32;
                    let bound = 2.0 * max;
                    site.check(
                        values.iter().all(|value| value.is_finite())
                            && *w >= 0.0
                            && *h >= 0.0
                            && *w <= max
                            && *h <= max
                            && *x >= -bound
                            && *y >= -bound
                            && x + w <= bound - 1.0
                            && y + h <= bound - 1.0,
                        || "the viewport rectangle is invalid".to_owned(),
                    )?;
                    site.check(
                        0.0 <= *min_depth && min_depth <= max_depth && *max_depth <= 1.0,
                        || "the viewport depth range is not within 0 <= min <= max <= 1".to_owned(),
                    )?;
                    RenderOp::SetViewport(values)
                }
                RenderCommand::SetScissorRect {
                    x,
                    y,
                    width: w,
                    height: h,
                } => {
                    let fits = |origin: u32, extent: u32, side: u32| {
                        origin.checked_add(extent).is_some_and(|end| end <= side)
                    };
                    site.check(fits(*x, *w, width) && fits(*y, *h, height), || {
                        "the scissor rectangle is outside the attachments".to_owned()
                    })?;
                    RenderOp::SetScissorRect([*x, *y, *w, *h])
                }
                RenderCommand::Draw {
                    vertices,
                    instances,
                } => {
                    site.check(!reversed(vertices) && !reversed(instances), || {
                        "a range is reversed".to_owned()
                    })?;
                    let entry = self.draw_state(site, pipeline, &groups)?;
                    Self::check_vertices(
                        site,
                        &entry.meta,
                        &vertex,
                        vertices.end,
                        instances.end,
                        true,
                    )?;
                    RenderOp::Draw(vertices.clone(), instances.clone())
                }
                RenderCommand::DrawIndexed {
                    indices,
                    base_vertex,
                    instances,
                } => {
                    site.check(!reversed(indices) && !reversed(instances), || {
                        "a range is reversed".to_owned()
                    })?;
                    let entry = self.draw_state(site, pipeline, &groups)?;
                    let (format, range) =
                        index.ok_or_else(|| site.fail("no index buffer is set"))?;
                    if let Some(strip) = entry.meta.strip_index_format {
                        site.check(strip == format, || {
                            "the index format differs from the pipeline's strip_index_format"
                                .to_owned()
                        })?;
                    }
                    let needed = u64::from(indices.end).saturating_mul(index_size(format));
                    site.check(needed <= range.size, || {
                        format!(
                            "indices up to {} need {needed} bytes, the index buffer range has {}",
                            indices.end, range.size
                        )
                    })?;
                    Self::check_vertices(site, &entry.meta, &vertex, 0, instances.end, false)?;
                    RenderOp::DrawIndexed(indices.clone(), *base_vertex, instances.clone())
                }
            };
            ops.push(op);
        }
        Ok(RenderPassOp {
            label: pass.label.as_deref(),
            colors,
            depth,
            ops,
        })
    }

    /// Resolves a vertex or index buffer range: `size: None` takes the rest of the buffer.
    fn range(
        site: Site<'_>,
        entry: &BufferEntry,
        offset: u64,
        size: Option<u64>,
        alignment: u64,
    ) -> Result<u64, RhiError> {
        site.check(offset.is_multiple_of(alignment), || {
            format!("the offset is not a multiple of {alignment}")
        })?;
        let size = match size {
            Some(size) => size,
            None => entry
                .info
                .size
                .checked_sub(offset)
                .ok_or_else(|| site.fail("the offset is past the end of the buffer"))?,
        };
        site.check(within(offset, size, entry.info.size), || {
            "the range is out of bounds".to_owned()
        })?;
        Ok(size)
    }

    fn draw_state(
        &self,
        site: Site<'_>,
        pipeline: Option<&'d RenderEntry>,
        groups: &[Option<Bound<'d>>],
    ) -> Result<&'d RenderEntry, RhiError> {
        let entry = pipeline.ok_or_else(|| site.fail("no pipeline is set"))?;
        self.check_groups(site, entry.info.layout, &entry.meta, groups)?;
        Ok(entry)
    }

    /// Checks that every vertex buffer slot is set and holds the elements a draw reads:
    /// `vertex_end` vertices (when `per_vertex`) and `instance_end` instances.
    fn check_vertices(
        site: Site<'_>,
        meta: &PipelineMeta,
        vertex: &[Option<BufferRange>],
        vertex_end: u32,
        instance_end: u32,
        per_vertex: bool,
    ) -> Result<(), RhiError> {
        for (slot, layout) in meta.slots.iter().enumerate() {
            let bound = vertex
                .get(slot)
                .copied()
                .flatten()
                .ok_or_else(|| site.fail(format!("vertex buffer {slot} is not set")))?;
            let count = match layout.step {
                VertexStepMode::Vertex if per_vertex => vertex_end,
                VertexStepMode::Vertex => continue,
                VertexStepMode::Instance => instance_end,
            };
            if count == 0 {
                continue;
            }
            let needed = u64::from(count.saturating_sub(1))
                .checked_mul(layout.stride)
                .and_then(|elements| elements.checked_add(layout.last_end));
            site.check(needed.is_some_and(|needed| needed <= bound.size), || {
                format!(
                    "vertex buffer {slot} holds {} bytes, too few for {count} elements",
                    bound.size
                )
            })?;
        }
        Ok(())
    }

    // -- compute passes -------------------------------------------------------------------

    fn compute_pass(
        &self,
        site: Site<'_>,
        pass: &'d ComputePassData,
    ) -> Result<ComputePassOp<'d>, RhiError> {
        let limits = &self.limits;
        let mut pipeline: Option<&'d ComputeEntry> = None;
        let mut groups: Vec<Option<Bound<'d>>> = vec![None; limits.max_bind_groups as usize];
        let mut ops = Vec::new();
        for command in &pass.commands {
            let site = site.at(command.name());
            let op = match command {
                ComputeCommand::SetPipeline(handle) => {
                    let entry = self.compute_pipelines.get(*handle)?;
                    pipeline = Some(entry);
                    ComputeOp::SetPipeline(&entry.obj)
                }
                ComputeCommand::SetBindGroup {
                    index,
                    group,
                    offsets,
                } => {
                    let bound = self.bind_group(site, *index, *group, offsets)?;
                    if let Some(entry) = groups.get_mut(*index as usize) {
                        *entry = Some(bound);
                    }
                    ComputeOp::SetBindGroup(*index, &bound.group.obj, offsets)
                }
                ComputeCommand::Dispatch { x, y, z } => {
                    let entry = pipeline.ok_or_else(|| site.fail("no pipeline is set"))?;
                    self.check_groups(site, entry.info.layout, &entry.meta, &groups)?;
                    let max = limits.max_compute_workgroups_per_dimension;
                    site.check(*x <= max && *y <= max && *z <= max, || {
                        format!(
                            "the workgroup counts ({x}, {y}, {z}) exceed \
                             max_compute_workgroups_per_dimension ({max})"
                        )
                    })?;
                    // Each dispatch is its own usage scope: the groups its pipeline uses.
                    let used = self
                        .pipeline_layouts
                        .get(entry.info.layout)?
                        .info
                        .bind_group_layouts
                        .len();
                    let mut scope = Scope::default();
                    for bound in groups.iter().take(used).flatten() {
                        self.scope_group(site, &mut scope, *bound)?;
                    }
                    ComputeOp::Dispatch([*x, *y, *z])
                }
            };
            ops.push(op);
        }
        Ok(ComputePassOp {
            label: pass.label.as_deref(),
            ops,
        })
    }
}

const fn index_size(format: IndexFormat) -> u64 {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}

// ---------------------------------------------------------------------------------------------
// The device's queue

impl Device {
    /// Validates command buffers and submits them, in order, as one submission.
    ///
    /// Every command of every buffer is checked first — handles, usages, ranges, alignments,
    /// attachments, pipeline and bind group compatibility, draw ranges, usage conflicts — and
    /// a broken rule rejects the whole submission before the backend sees any of it. Queue
    /// writes made before the call take effect before its commands run.
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] naming the command buffer's label and the command; the
    /// handles' errors ([`RhiError::StaleHandle`] for a resource destroyed after recording);
    /// [`RhiError::OutOfMemory`] for a recording that ran out of memory;
    /// [`RhiError::DeviceLost`]; [`RhiError::Backend`].
    pub fn submit(
        &mut self,
        command_buffers: impl IntoIterator<Item = CommandBuffer>,
    ) -> Result<(), RhiError> {
        self.alive()?;
        let buffers: Vec<CommandBuffer> = command_buffers.into_iter().collect();
        let ctx = Ctx {
            limits: &self.limits,
            buffers: &self.buffers,
            textures: &self.textures,
            views: &self.views,
            bind_group_layouts: &self.bind_group_layouts,
            bind_groups: &self.bind_groups,
            pipeline_layouts: &self.pipeline_layouts,
            render_pipelines: &self.render_pipelines,
            compute_pipelines: &self.compute_pipelines,
        };
        let batches = buffers
            .iter()
            .map(|buffer| ctx.batch(buffer))
            .collect::<Result<Vec<_>, RhiError>>()?;
        self.backend.execute(&batches)
    }

    /// Writes `data` into a buffer with [`BufferUsages::COPY_DST`]; it takes effect before the
    /// commands of the next submission run. Offset and length are multiples of
    /// [`COPY_BUFFER_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`write_buffer`) for a missing usage, a misaligned or
    /// out-of-bounds range; the handle's errors; [`RhiError::DeviceLost`];
    /// [`RhiError::Backend`].
    pub fn write_buffer(
        &mut self,
        buffer: Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        self.alive()?;
        let site = Site {
            label: None,
            command: "write_buffer",
        };
        let entry = self.buffers.get(buffer)?;
        let size = data.len() as u64;
        site.check(entry.info.usage.contains(BufferUsages::COPY_DST), || {
            "the buffer lacks COPY_DST usage".to_owned()
        })?;
        site.check(aligned(offset) && aligned(size), || {
            "the offset and length must be multiples of 4".to_owned()
        })?;
        site.check(within(offset, size, entry.info.size), || {
            "the range is out of bounds".to_owned()
        })?;
        if data.is_empty() {
            return Ok(());
        }
        let obj = entry.obj.clone();
        self.backend.write_buffer(&obj, offset, data)
    }

    /// Waits until every submission so far has finished and reads a range of a buffer with
    /// [`BufferUsages::MAP_READ`] or [`BufferUsages::COPY_SRC`]. Offset and size are multiples
    /// of [`COPY_BUFFER_ALIGNMENT`]. Blocks the calling thread; meant for tests, tools and
    /// loading, not for every frame (asynchronous readback comes with frames in flight, M4).
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`read_buffer`) for a missing usage, a misaligned or
    /// out-of-bounds range; the handle's errors; [`RhiError::DeviceLost`];
    /// [`RhiError::Backend`].
    pub fn read_buffer(
        &mut self,
        buffer: Buffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        self.alive()?;
        let site = Site {
            label: None,
            command: "read_buffer",
        };
        let entry = self.buffers.get(buffer)?;
        site.check(
            entry
                .info
                .usage
                .intersects(BufferUsages::MAP_READ | BufferUsages::COPY_SRC),
            || "the buffer has neither MAP_READ nor COPY_SRC usage".to_owned(),
        )?;
        site.check(aligned(offset) && aligned(size), || {
            "the offset and size must be multiples of 4".to_owned()
        })?;
        site.check(within(offset, size, entry.info.size), || {
            "the range is out of bounds".to_owned()
        })?;
        let (obj, info) = (entry.obj.clone(), entry.info);
        self.backend.read_buffer(&obj, &info, offset, size)
    }

    /// Waits until every submission so far has finished and reads a region of a texture with
    /// [`TextureUsages::COPY_SRC`], returning its texels row after row with no padding.
    /// Copies through a temporary buffer; blocks like [`Self::read_buffer`].
    ///
    /// # Errors
    ///
    /// [`RhiError::InvalidCommand`] (`copy_texture_to_buffer` in `read_texture`) for a region
    /// the texture cannot be copied from; [`RhiError::LimitExceeded`] when the padded region
    /// is larger than [`Limits::max_buffer_size`](crate::Limits::max_buffer_size); otherwise as
    /// [`Self::read_buffer`].
    pub fn read_texture(
        &mut self,
        src: TexelCopyTextureInfo,
        size: Extent3d,
    ) -> Result<Vec<u8>, RhiError> {
        self.alive()?;
        let format = self.textures.get(src.texture)?.info.format;
        let block = u64::from(format.block_size());
        let row = u64::from(size.width).saturating_mul(block);
        let (height, images) = (
            u64::from(size.height),
            u64::from(size.depth_or_array_layers),
        );
        if row == 0 || height == 0 || images == 0 {
            return Ok(Vec::new());
        }
        let alignment = u64::from(COPY_BYTES_PER_ROW_ALIGNMENT);
        let padded = row
            .checked_next_multiple_of(alignment)
            .ok_or(RhiError::OutOfMemory { bytes: u64::MAX })?;
        let bytes_per_row = u32::try_from(padded)
            .map_err(|_| RhiError::limit("bytes_per_row", padded, u32::MAX))?;
        let total = padded
            .checked_mul(height)
            .and_then(|bytes| bytes.checked_mul(images))
            .ok_or(RhiError::OutOfMemory { bytes: u64::MAX })?;
        let staging = self.create_buffer(&BufferDescriptor {
            label: Some("pragma_rhi read_texture"),
            size: total,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        })?;
        let mut encoder = CommandEncoder::new(Some("read_texture"));
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
        let read = self
            .submit([encoder.finish()])
            .and_then(|()| self.read_buffer(staging, 0, total));
        let destroyed = self.destroy_buffer(staging);
        let padded_bytes = read?;
        destroyed?;
        let mut out = Vec::new();
        let len = usize::try_from(row.saturating_mul(height).saturating_mul(images))
            .map_err(|_| RhiError::OutOfMemory { bytes: u64::MAX })?;
        out.try_reserve_exact(len)
            .map_err(|_| RhiError::OutOfMemory { bytes: len as u64 })?;
        let (row, padded) = (row as usize, padded as usize);
        for line in padded_bytes.chunks(padded) {
            out.extend_from_slice(line.get(..row).unwrap_or(line));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{Scope, Site, Use};
    use crate::handle::{Handle, RawHandle};
    use crate::{Buffer, RhiError, Texture};

    const SITE: Site<'static> = Site {
        label: Some("pass"),
        command: "draw",
    };

    fn raw(index: u32) -> RawHandle {
        RawHandle {
            device: 1,
            index,
            generation: 0,
        }
    }

    #[test]
    fn buffers_are_read_or_writable_storage_not_both() {
        let mut scope = Scope::default();
        let a = Buffer::from_raw(raw(0));
        let b = Buffer::from_raw(raw(1));
        assert_eq!(scope.buffer(SITE, a, Use::Read), Ok(()));
        assert_eq!(scope.buffer(SITE, a, Use::Read), Ok(()));
        assert_eq!(scope.buffer(SITE, b, Use::Storage), Ok(()));
        assert_eq!(scope.buffer(SITE, b, Use::Storage), Ok(()));
        assert!(matches!(
            scope.buffer(SITE, a, Use::Storage),
            Err(RhiError::InvalidCommand {
                command: "draw",
                ..
            })
        ));
    }

    #[test]
    fn texture_conflicts_need_overlapping_subresources() {
        let mut scope = Scope::default();
        let t = Texture::from_raw(raw(0));
        // Mip 0 as an attachment while mip 1 is sampled: a mip chain generation pattern.
        assert_eq!(scope.texture(SITE, t, 0..1, 0..1, Use::Attachment), Ok(()));
        assert_eq!(scope.texture(SITE, t, 1..2, 0..1, Use::Read), Ok(()));
        assert!(scope.texture(SITE, t, 0..2, 0..1, Use::Read).is_err());
        assert!(
            scope.texture(SITE, t, 0..1, 0..1, Use::Attachment).is_err(),
            "one subresource attached twice"
        );
    }
}
