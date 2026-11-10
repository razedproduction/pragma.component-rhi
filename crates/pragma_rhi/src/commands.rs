//! Command recording: encoders, render and compute passes, command buffers.
//!
//! Recording does not touch a device (DECISIONS.md #15). A [`CommandEncoder`] appends plain
//! commands that name resources by handle; it is `Send`, needs no lock and cannot fail, so any
//! number of encoders can be recorded at once on any threads — as frame-graph tasks
//! ([`CommandGraph`](crate::CommandGraph)) or by hand. Every rule is checked when the finished
//! [`CommandBuffer`]s are submitted ([`Device::submit`](crate::Device::submit)): handles,
//! usages, ranges, alignments, pass attachments, pipeline and bind group compatibility, draw
//! ranges and resource usage conflicts. A submission that breaks a rule is rejected as a whole
//! with a typed error before the backend sees any of it.

use core::ops::Range;

use crate::{
    BindGroup, Buffer, ComputePipeline, Extent3d, IndexFormat, RenderPipeline, Texture, TextureView,
};

// ---------------------------------------------------------------------------------------------
// Pass descriptors

/// An RGBA color, as a render pass clears a color attachment with it. The channels are in the
/// attachment's linear space; an sRGB attachment encodes them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color {
    /// Red.
    pub r: f64,
    /// Green.
    pub g: f64,
    /// Blue.
    pub b: f64,
    /// Alpha.
    pub a: f64,
}

impl Color {
    /// All zero.
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    /// A color from its channels.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }
}

/// What a render pass does with an attachment when it begins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<V> {
    /// Fills the attachment with a value.
    Clear(V),
    /// Keeps the attachment's contents.
    Load,
}

/// What a render pass does with an attachment when it ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StoreOp {
    /// Keeps what the pass wrote.
    #[default]
    Store,
    /// Discards it: the attachment reads as zero afterwards.
    Discard,
}

/// The load and store operations of one attachment aspect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operations<V> {
    /// At the beginning of the pass.
    pub load: LoadOp<V>,
    /// At the end of the pass.
    pub store: StoreOp,
}

impl<V> Operations<V> {
    /// Clears to `value` and stores.
    pub const fn clear(value: V) -> Self {
        Self {
            load: LoadOp::Clear(value),
            store: StoreOp::Store,
        }
    }

    /// Loads and stores.
    pub const fn load() -> Self {
        Self {
            load: LoadOp::Load,
            store: StoreOp::Store,
        }
    }
}

/// A color attachment of a render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPassColorAttachment {
    /// A view of one mip level and one layer of a 2D texture with
    /// [`TextureUsages::RENDER_ATTACHMENT`](crate::TextureUsages::RENDER_ATTACHMENT) and a color
    /// format.
    pub view: TextureView,
    /// For a multisampled `view`: a single-sampled view of the same format and size that
    /// receives the resolved samples when the pass ends.
    pub resolve_target: Option<TextureView>,
    /// The operations.
    pub ops: Operations<Color>,
}

impl RenderPassColorAttachment {
    /// An attachment cleared to `color` and stored.
    #[must_use]
    pub const fn clear(view: TextureView, color: Color) -> Self {
        Self {
            view,
            resolve_target: None,
            ops: Operations::clear(color),
        }
    }
}

/// The depth/stencil attachment of a render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderPassDepthStencilAttachment {
    /// A view of one mip level and one layer of a texture with a depth format.
    pub view: TextureView,
    /// The depth operations; required, since every depth format has depth. A clear value is
    /// in `0.0..=1.0`.
    pub depth_ops: Option<Operations<f32>>,
    /// The stencil operations; required exactly when the format has stencil.
    pub stencil_ops: Option<Operations<u32>>,
}

/// Describes a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RenderPassDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The color attachments by index (`None` leaves one out); at most
    /// [`Limits::max_color_attachments`](crate::Limits::max_color_attachments).
    pub color_attachments: &'a [Option<RenderPassColorAttachment>],
    /// The depth/stencil attachment, if any. A pass has at least one attachment.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
}

/// Describes a compute pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ComputePassDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
}

// ---------------------------------------------------------------------------------------------
// Copies

/// A texel position in a mip level: `z` is the array layer of a 1D or 2D texture and the depth
/// slice of a 3D one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Origin3d {
    /// The column.
    pub x: u32,
    /// The row.
    pub y: u32,
    /// The layer or slice.
    pub z: u32,
}

/// How texels are laid out in a buffer for a copy to or from a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TexelCopyBufferLayout {
    /// The first byte of the first texel; a multiple of the format's texel size.
    pub offset: u64,
    /// The bytes from the start of one row to the next; a multiple of
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`] and at least a row's bytes. Required when the copy has
    /// more than one row.
    pub bytes_per_row: Option<u32>,
    /// The rows from the start of one image (layer or slice) to the next; at least the copy's
    /// height. Required when the copy has more than one image.
    pub rows_per_image: Option<u32>,
}

/// A buffer side of a copy to or from a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TexelCopyBufferInfo {
    /// The buffer.
    pub buffer: Buffer,
    /// The layout of the texels in it.
    pub layout: TexelCopyBufferLayout,
}

/// A texture side of a copy: a region of one mip level, starting at `origin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TexelCopyTextureInfo {
    /// The texture; single-sampled, with a color format.
    pub texture: Texture,
    /// The mip level.
    pub mip_level: u32,
    /// The first texel.
    pub origin: Origin3d,
}

impl TexelCopyTextureInfo {
    /// Mip level 0 from its first texel.
    #[must_use]
    pub const fn new(texture: Texture) -> Self {
        Self {
            texture,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
        }
    }
}

/// The alignment of buffer offsets and sizes in buffer copies, clears, queue writes and reads.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// The alignment of [`TexelCopyBufferLayout::bytes_per_row`].
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

// ---------------------------------------------------------------------------------------------
// Recorded commands (internal)

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Command {
    CopyBufferToBuffer {
        src: Buffer,
        src_offset: u64,
        dst: Buffer,
        dst_offset: u64,
        size: u64,
    },
    ClearBuffer {
        buffer: Buffer,
        offset: u64,
        size: Option<u64>,
    },
    CopyBufferToTexture {
        src: TexelCopyBufferInfo,
        dst: TexelCopyTextureInfo,
        size: Extent3d,
    },
    CopyTextureToBuffer {
        src: TexelCopyTextureInfo,
        dst: TexelCopyBufferInfo,
        size: Extent3d,
    },
    CopyTextureToTexture {
        src: TexelCopyTextureInfo,
        dst: TexelCopyTextureInfo,
        size: Extent3d,
    },
    RenderPass(RenderPassData),
    ComputePass(ComputePassData),
}

impl Command {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::CopyBufferToBuffer { .. } => "copy_buffer_to_buffer",
            Self::ClearBuffer { .. } => "clear_buffer",
            Self::CopyBufferToTexture { .. } => "copy_buffer_to_texture",
            Self::CopyTextureToBuffer { .. } => "copy_texture_to_buffer",
            Self::CopyTextureToTexture { .. } => "copy_texture_to_texture",
            Self::RenderPass(_) => "begin_render_pass",
            Self::ComputePass(_) => "begin_compute_pass",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct RenderPassData {
    pub(crate) label: Option<String>,
    pub(crate) colors: Vec<Option<RenderPassColorAttachment>>,
    pub(crate) depth_stencil: Option<RenderPassDepthStencilAttachment>,
    pub(crate) commands: Vec<RenderCommand>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RenderCommand {
    SetPipeline(RenderPipeline),
    SetBindGroup {
        index: u32,
        group: BindGroup,
        offsets: Vec<u32>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: Buffer,
        offset: u64,
        size: Option<u64>,
    },
    SetIndexBuffer {
        buffer: Buffer,
        format: IndexFormat,
        offset: u64,
        size: Option<u64>,
    },
    SetViewport {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    },
    SetScissorRect {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
}

impl RenderCommand {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::SetPipeline(_) => "set_pipeline",
            Self::SetBindGroup { .. } => "set_bind_group",
            Self::SetVertexBuffer { .. } => "set_vertex_buffer",
            Self::SetIndexBuffer { .. } => "set_index_buffer",
            Self::SetViewport { .. } => "set_viewport",
            Self::SetScissorRect { .. } => "set_scissor_rect",
            Self::Draw { .. } => "draw",
            Self::DrawIndexed { .. } => "draw_indexed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct ComputePassData {
    pub(crate) label: Option<String>,
    pub(crate) commands: Vec<ComputeCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ComputeCommand {
    SetPipeline(ComputePipeline),
    SetBindGroup {
        index: u32,
        group: BindGroup,
        offsets: Vec<u32>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
}

impl ComputeCommand {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::SetPipeline(_) => "set_pipeline",
            Self::SetBindGroup { .. } => "set_bind_group",
            Self::Dispatch { .. } => "dispatch_workgroups",
        }
    }
}

/// Appends without aborting on allocation failure: a failed push marks the recording as out of
/// memory, which [`Device::submit`](crate::Device::submit) reports.
fn try_push<T>(list: &mut Vec<T>, value: T, out_of_memory: &mut bool) {
    if list.try_reserve(1).is_ok() {
        list.push(value);
    } else {
        *out_of_memory = true;
    }
}

fn try_copy<T: Clone>(items: &[T], out_of_memory: &mut bool) -> Vec<T> {
    let mut copy = Vec::new();
    if copy.try_reserve_exact(items.len()).is_ok() {
        copy.extend_from_slice(items);
    } else {
        *out_of_memory = true;
    }
    copy
}

// ---------------------------------------------------------------------------------------------
// Encoder

/// Records commands into a [`CommandBuffer`].
///
/// Needs no device: commands name resources by handle and are validated when submitted, so an
/// encoder can be created and filled on any thread. Recording never fails; a rule a command
/// breaks is reported by [`Device::submit`](crate::Device::submit), naming the encoder's label
/// and the command.
///
/// ```
/// use pragma_rhi::{
///     BufferDescriptor, BufferUsages, CommandEncoder, DeviceDescriptor, Instance,
///     InstanceDescriptor, RhiError,
/// };
///
/// let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
/// let src = device.create_buffer(&BufferDescriptor { label: None, size: 16, usage })?;
/// let dst = device.create_buffer(&BufferDescriptor { label: None, size: 16, usage })?;
/// device.write_buffer(src, 0, &[7; 16])?;
///
/// let mut encoder = CommandEncoder::new(Some("copy"));
/// encoder.copy_buffer_to_buffer(src, 0, dst, 0, 16);
/// device.submit([encoder.finish()])?;
/// assert_eq!(device.read_buffer(dst, 0, 16)?, [7; 16]);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Default)]
pub struct CommandEncoder {
    label: Option<String>,
    commands: Vec<Command>,
    out_of_memory: bool,
}

impl CommandEncoder {
    /// A new, empty encoder. The label names the command buffer in errors and debuggers.
    #[must_use]
    pub fn new(label: Option<&str>) -> Self {
        Self {
            label: label.map(str::to_owned),
            commands: Vec::new(),
            out_of_memory: false,
        }
    }

    /// The label.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    fn push(&mut self, command: Command) {
        try_push(&mut self.commands, command, &mut self.out_of_memory);
    }

    /// Copies `size` bytes between two different buffers. The source needs
    /// [`BufferUsages::COPY_SRC`](crate::BufferUsages::COPY_SRC), the destination
    /// [`BufferUsages::COPY_DST`](crate::BufferUsages::COPY_DST); offsets and size are multiples
    /// of [`COPY_BUFFER_ALIGNMENT`].
    pub fn copy_buffer_to_buffer(
        &mut self,
        src: Buffer,
        src_offset: u64,
        dst: Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        self.push(Command::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
    }

    /// Fills a range of a buffer with zeros (`size: None` clears to the end). The buffer needs
    /// [`BufferUsages::COPY_DST`](crate::BufferUsages::COPY_DST); offset and size are multiples
    /// of [`COPY_BUFFER_ALIGNMENT`].
    pub fn clear_buffer(&mut self, buffer: Buffer, offset: u64, size: Option<u64>) {
        self.push(Command::ClearBuffer {
            buffer,
            offset,
            size,
        });
    }

    /// Copies texels from a buffer with
    /// [`BufferUsages::COPY_SRC`](crate::BufferUsages::COPY_SRC) into a region of a texture with
    /// [`TextureUsages::COPY_DST`](crate::TextureUsages::COPY_DST).
    pub fn copy_buffer_to_texture(
        &mut self,
        src: TexelCopyBufferInfo,
        dst: TexelCopyTextureInfo,
        size: Extent3d,
    ) {
        self.push(Command::CopyBufferToTexture { src, dst, size });
    }

    /// Copies a region of a texture with
    /// [`TextureUsages::COPY_SRC`](crate::TextureUsages::COPY_SRC) into a buffer with
    /// [`BufferUsages::COPY_DST`](crate::BufferUsages::COPY_DST).
    pub fn copy_texture_to_buffer(
        &mut self,
        src: TexelCopyTextureInfo,
        dst: TexelCopyBufferInfo,
        size: Extent3d,
    ) {
        self.push(Command::CopyTextureToBuffer { src, dst, size });
    }

    /// Copies a region between two textures of the same format; within one texture the
    /// regions must not overlap.
    pub fn copy_texture_to_texture(
        &mut self,
        src: TexelCopyTextureInfo,
        dst: TexelCopyTextureInfo,
        size: Extent3d,
    ) {
        self.push(Command::CopyTextureToTexture { src, dst, size });
    }

    /// Begins a render pass; it ends when the returned [`RenderPass`] is dropped.
    pub fn begin_render_pass(&mut self, desc: &RenderPassDescriptor<'_>) -> RenderPass<'_> {
        let colors = try_copy(desc.color_attachments, &mut self.out_of_memory);
        RenderPass {
            data: RenderPassData {
                label: desc.label.map(str::to_owned),
                colors,
                depth_stencil: desc.depth_stencil_attachment,
                commands: Vec::new(),
            },
            encoder: self,
        }
    }

    /// Begins a compute pass; it ends when the returned [`ComputePass`] is dropped.
    pub fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor<'_>) -> ComputePass<'_> {
        ComputePass {
            data: ComputePassData {
                label: desc.label.map(str::to_owned),
                commands: Vec::new(),
            },
            encoder: self,
        }
    }

    /// Finishes recording.
    #[must_use]
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            label: self.label,
            commands: self.commands,
            out_of_memory: self.out_of_memory,
        }
    }

    /// Continues recording a finished buffer (the device's pending uploads after a rejected
    /// submission).
    pub(crate) fn from_buffer(buffer: CommandBuffer) -> Self {
        Self {
            label: buffer.label,
            commands: buffer.commands,
            out_of_memory: buffer.out_of_memory,
        }
    }
}

/// Recorded commands, ready for [`Device::submit`](crate::Device::submit). `Send` and `Sync`;
/// it holds handles, not resources, so it keeps nothing alive.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandBuffer {
    pub(crate) label: Option<String>,
    pub(crate) commands: Vec<Command>,
    pub(crate) out_of_memory: bool,
}

impl CommandBuffer {
    /// The label of the encoder it was recorded with.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// The number of commands, a pass counting as one.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ---------------------------------------------------------------------------------------------
// Passes

/// A render pass being recorded; dropping it (or calling [`Self::end`]) ends the pass.
#[derive(Debug)]
pub struct RenderPass<'e> {
    encoder: &'e mut CommandEncoder,
    data: RenderPassData,
}

impl RenderPass<'_> {
    fn push(&mut self, command: RenderCommand) {
        try_push(
            &mut self.data.commands,
            command,
            &mut self.encoder.out_of_memory,
        );
    }

    /// Sets the pipeline of the following draws. Its color formats, depth format and sample
    /// count must be the pass's.
    pub fn set_pipeline(&mut self, pipeline: RenderPipeline) {
        self.push(RenderCommand::SetPipeline(pipeline));
    }

    /// Sets bind group `index`, with one offset for every dynamic-offset binding of its layout,
    /// in binding order.
    pub fn set_bind_group(&mut self, index: u32, group: BindGroup, offsets: &[u32]) {
        let offsets = try_copy(offsets, &mut self.encoder.out_of_memory);
        self.push(RenderCommand::SetBindGroup {
            index,
            group,
            offsets,
        });
    }

    /// Binds a range of a buffer with [`BufferUsages::VERTEX`](crate::BufferUsages::VERTEX) to
    /// vertex buffer `slot` (`size: None` takes the rest of the buffer).
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: Buffer, offset: u64, size: Option<u64>) {
        self.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer,
            offset,
            size,
        });
    }

    /// Binds a range of a buffer with [`BufferUsages::INDEX`](crate::BufferUsages::INDEX) as the
    /// index buffer.
    pub fn set_index_buffer(
        &mut self,
        buffer: Buffer,
        format: IndexFormat,
        offset: u64,
        size: Option<u64>,
    ) {
        self.push(RenderCommand::SetIndexBuffer {
            buffer,
            format,
            offset,
            size,
        });
    }

    /// Sets the viewport: finite, a non-negative size, `0 <= min_depth <= max_depth <= 1`.
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.push(RenderCommand::SetViewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        });
    }

    /// Sets the scissor rectangle; it lies within the attachments.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.push(RenderCommand::SetScissorRect {
            x,
            y,
            width,
            height,
        });
    }

    /// Draws `vertices` of `instances`; the bound vertex buffers must hold them.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.push(RenderCommand::Draw {
            vertices,
            instances,
        });
    }

    /// Draws `indices` of the index buffer, offset by `base_vertex`, of `instances`.
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.push(RenderCommand::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
    }

    /// Ends the pass (as dropping it does).
    pub fn end(self) {}
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        let data = core::mem::take(&mut self.data);
        self.encoder.push(Command::RenderPass(data));
    }
}

/// A compute pass being recorded; dropping it (or calling [`Self::end`]) ends the pass.
#[derive(Debug)]
pub struct ComputePass<'e> {
    encoder: &'e mut CommandEncoder,
    data: ComputePassData,
}

impl ComputePass<'_> {
    fn push(&mut self, command: ComputeCommand) {
        try_push(
            &mut self.data.commands,
            command,
            &mut self.encoder.out_of_memory,
        );
    }

    /// Sets the pipeline of the following dispatches.
    pub fn set_pipeline(&mut self, pipeline: ComputePipeline) {
        self.push(ComputeCommand::SetPipeline(pipeline));
    }

    /// Sets bind group `index`; see [`RenderPass::set_bind_group`].
    pub fn set_bind_group(&mut self, index: u32, group: BindGroup, offsets: &[u32]) {
        let offsets = try_copy(offsets, &mut self.encoder.out_of_memory);
        self.push(ComputeCommand::SetBindGroup {
            index,
            group,
            offsets,
        });
    }

    /// Dispatches `x * y * z` workgroups; each count at most
    /// [`Limits::max_compute_workgroups_per_dimension`](crate::Limits::max_compute_workgroups_per_dimension).
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.push(ComputeCommand::Dispatch { x, y, z });
    }

    /// Ends the pass (as dropping it does).
    pub fn end(self) {}
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        let data = core::mem::take(&mut self.data);
        self.encoder.push(Command::ComputePass(data));
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, CommandEncoder, ComputePassDescriptor, RenderPassDescriptor};

    #[test]
    fn passes_are_pushed_when_they_end() {
        let mut encoder = CommandEncoder::new(Some("frame"));
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder
            .begin_render_pass(&RenderPassDescriptor::default())
            .end();
        let buffer = encoder.finish();
        assert_eq!(buffer.label(), Some("frame"));
        assert_eq!(buffer.len(), 2);
        assert!(matches!(
            buffer.commands.as_slice(),
            [Command::ComputePass(compute), Command::RenderPass(_)] if compute.commands.len() == 1
        ));
    }

    #[test]
    fn buffers_are_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<super::CommandBuffer>();
        check::<CommandEncoder>();
    }
}
