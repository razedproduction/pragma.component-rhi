//! Shader modules: parsed and validated by naga on every backend, then reflected.
//!
//! A module is parsed (WGSL or SPIR-V) and validated here, before any backend sees it, so the
//! null backend rejects exactly the shaders wgpu would and a WGSL error carries its line and
//! column. The reflection kept for each entry point — the resources it uses, its vertex inputs,
//! fragment outputs and workgroup size — is what pipeline creation checks against the pipeline
//! layout and the rest of the descriptor (DECISIONS.md #13).

use core::fmt;
use std::sync::Arc;

use bitflags::bitflags;

use crate::binding::StorageTextureAccess;
use crate::{RhiError, TextureFormat, TextureViewDimension};

/// The source of a [`ShaderModule`](crate::ShaderModule).
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let wgsl = device.create_shader_module(&ShaderModuleDescriptor {
///     label: Some("wgsl"),
///     source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}"),
/// })?;
/// assert_eq!(device.shader_module_info(wgsl)?.entry_points[0].name, "main");
/// // SPIR-V is validated like WGSL; these words are no module.
/// let result = device.create_shader_module(&ShaderModuleDescriptor {
///     label: Some("spir-v"),
///     source: ShaderSource::SpirV(&[0x0723_0203, 0]),
/// });
/// assert!(matches!(result, Err(RhiError::Shader { .. })));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// WGSL text. Errors carry a [`SourceLocation`].
    Wgsl(&'a str),
    /// SPIR-V words, translated and validated by naga like WGSL (not passed through
    /// unvalidated: that would need `unsafe`, DECISIONS.md #6).
    SpirV(&'a [u32]),
}

impl Default for ShaderSource<'_> {
    /// An empty WGSL module (valid, with no entry points).
    fn default() -> Self {
        Self::Wgsl("")
    }
}

/// Describes a [`ShaderModule`](crate::ShaderModule).
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let module = device.create_shader_module(&ShaderModuleDescriptor {
///     label: Some("fullscreen"),
///     source: ShaderSource::Wgsl(
///         "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
///              let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
///              return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
///          }",
///     ),
/// })?;
/// assert_eq!(device.shader_module_info(module)?.entry_points[0].stage, ShaderStage::Vertex);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShaderModuleDescriptor<'a> {
    /// A debug label, passed to the backend.
    pub label: Option<&'a str>,
    /// The source.
    pub source: ShaderSource<'a>,
}

bitflags! {
    /// A set of shader stages, for example the visibility of a binding.
    ///
    /// ```
    /// use pragma_rhi::{ShaderStage, ShaderStages};
    ///
    /// assert_eq!(ShaderStages::VERTEX_FRAGMENT, ShaderStages::VERTEX | ShaderStages::FRAGMENT);
    /// assert!(ShaderStages::VERTEX_FRAGMENT.contains(ShaderStage::Fragment.as_stages()));
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct ShaderStages: u8 {
        /// The vertex stage.
        const VERTEX = 1 << 0;
        /// The fragment stage.
        const FRAGMENT = 1 << 1;
        /// The compute stage.
        const COMPUTE = 1 << 2;
        /// The vertex and fragment stages.
        const VERTEX_FRAGMENT = Self::VERTEX.bits() | Self::FRAGMENT.bits();
    }
}

/// One shader stage.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let module = device.create_shader_module(&ShaderModuleDescriptor {
///     label: None,
///     source: ShaderSource::Wgsl(
///         "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(); }
///          @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
///     ),
/// })?;
/// let stages: Vec<ShaderStage> = device
///     .shader_module_info(module)?
///     .entry_points
///     .iter()
///     .map(|entry| entry.stage)
///     .collect();
/// assert_eq!(stages, [ShaderStage::Vertex, ShaderStage::Fragment]);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ShaderStage {
    /// The vertex stage of a render pipeline.
    Vertex,
    /// The fragment stage of a render pipeline.
    Fragment,
    /// A compute pipeline.
    Compute,
}

impl ShaderStage {
    /// The stage as a one-element [`ShaderStages`] set.
    #[must_use]
    pub const fn as_stages(self) -> ShaderStages {
        match self {
            Self::Vertex => ShaderStages::VERTEX,
            Self::Fragment => ShaderStages::FRAGMENT,
            Self::Compute => ShaderStages::COMPUTE,
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        })
    }
}

/// A position in WGSL source text.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let error = device
///     .create_shader_module(&ShaderModuleDescriptor {
///         label: Some("broken"),
///         source: ShaderSource::Wgsl("@compute @workgroup_size(1)\nfn main() {\n    let x = ;\n}"),
///     })
///     .unwrap_err();
/// let RhiError::Shader { location: Some(at), .. } = error else { panic!("{error}") };
/// assert_eq!(at.line, 3);
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SourceLocation {
    /// The 1-based line.
    pub line: u32,
    /// The 1-based column, in bytes.
    pub column: u32,
}

/// An entry point of a shader module.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let module = device.create_shader_module(&ShaderModuleDescriptor {
///     label: None,
///     source: ShaderSource::Wgsl("@compute @workgroup_size(8, 8) fn blur() {}"),
/// })?;
/// let info = device.shader_module_info(module)?;
/// let blur = &info.entry_points[0];
/// assert_eq!((blur.name.as_str(), blur.stage), ("blur", ShaderStage::Compute));
/// assert_eq!(blur.workgroup_size, Some([8, 8, 1]));
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EntryPointInfo {
    /// The function name.
    pub name: String,
    /// The stage.
    pub stage: ShaderStage,
    /// The workgroup size of a compute entry point (`[0; 3]` for other stages); `None` when it
    /// depends on pipeline-overridable constants.
    pub workgroup_size: Option<[u32; 3]>,
}

/// A live shader module's properties.
///
/// ```
/// # use pragma_rhi::*;
/// # let adapter = Instance::new(&InstanceDescriptor::null()).adapters().remove(0);
/// # let mut device = adapter.request_device(&DeviceDescriptor::default())?;
/// let module = device.create_shader_module(&ShaderModuleDescriptor::default())?;
/// // An empty WGSL module is valid and has no entry points.
/// assert!(device.shader_module_info(module)?.entry_points.is_empty());
/// # Ok::<(), RhiError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ShaderModuleInfo {
    /// The entry points of the supported stages, in source order.
    pub entry_points: Vec<EntryPointInfo>,
}

// ---------------------------------------------------------------------------------------------
// Reflection (internal)

/// The scalar kind of a vertex input or a fragment output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IoKind {
    Float,
    Sint,
    Uint,
}

impl fmt::Display for IoKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Float => "f32",
            Self::Sint => "i32",
            Self::Uint => "u32",
        })
    }
}

/// The sample type a shader declares for a sampled texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShaderSampleKind {
    Float,
    Depth,
    Sint,
    Uint,
}

/// What a shader declares at a `@group(g) @binding(b)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShaderBindingType {
    Uniform {
        min_size: u64,
    },
    Storage {
        read_only: bool,
        min_size: u64,
    },
    Sampler {
        comparison: bool,
    },
    Texture {
        sample: ShaderSampleKind,
        dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        format: Option<TextureFormat>,
        access: StorageTextureAccess,
        dimension: TextureViewDimension,
    },
    Unsupported(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ShaderBinding {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) ty: ShaderBindingType,
}

/// What pipeline creation needs to know about one entry point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct EntryReflection {
    pub(crate) name: String,
    pub(crate) stage: ShaderStage,
    pub(crate) workgroup_size: Option<[u32; 3]>,
    /// The resources the entry point uses, sorted by `(group, binding)`.
    pub(crate) bindings: Vec<ShaderBinding>,
    /// `@location` inputs (the vertex stage's are vertex attributes), sorted by location.
    pub(crate) inputs: Vec<(u32, IoKind)>,
    /// `@location` outputs (the fragment stage's are color targets), sorted by location.
    pub(crate) outputs: Vec<(u32, IoKind)>,
}

/// A parsed, validated and reflected module.
#[derive(Debug, Clone)]
pub(crate) struct ParsedShader {
    pub(crate) module: Arc<naga::Module>,
    pub(crate) info: ShaderModuleInfo,
    pub(crate) entries: Arc<[EntryReflection]>,
}

impl ParsedShader {
    /// Resolves an entry point: the named one of `stage`, or the only one of `stage`.
    pub(crate) fn entry_point(
        &self,
        stage: ShaderStage,
        name: Option<&str>,
    ) -> Result<&EntryReflection, RhiError> {
        let mismatch = |message: String| RhiError::ShaderInterface { stage, message };
        let mut of_stage = self.entries.iter().filter(|entry| entry.stage == stage);
        match name {
            Some(name) => of_stage
                .find(|entry| entry.name == name)
                .ok_or_else(|| mismatch(format!("the module has no {stage} entry point `{name}`"))),
            None => match (of_stage.next(), of_stage.next()) {
                (Some(entry), None) => Ok(entry),
                (None, _) => Err(mismatch(format!("the module has no {stage} entry point"))),
                (Some(_), Some(_)) => Err(mismatch(format!(
                    "the module has several {stage} entry points; name one"
                ))),
            },
        }
    }
}

/// Parses and validates a module and reflects its entry points.
pub(crate) fn parse(source: ShaderSource<'_>) -> Result<ParsedShader, RhiError> {
    let (module, text) = match source {
        ShaderSource::Wgsl(text) => {
            let module = naga::front::wgsl::parse_str(text).map_err(|err| RhiError::Shader {
                message: err.message().to_owned(),
                location: err.location(text).map(location),
            })?;
            (module, Some(text))
        }
        ShaderSource::SpirV(words) => {
            // wgpu's own options for SPIR-V input.
            let options = naga::front::spv::Options {
                adjust_coordinate_space: false,
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            };
            let module = naga::front::spv::Frontend::new(words.iter().copied(), &options)
                .parse()
                .map_err(|err| RhiError::Shader {
                    message: error_chain(&err),
                    location: None,
                })?;
            (module, None)
        }
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    let info = validator
        .validate(&module)
        .map_err(|err| RhiError::Shader {
            message: error_chain(err.as_inner()),
            location: text.and_then(|text| err.location(text)).map(location),
        })?;
    let entries = reflect(&module, &info)?;
    let info = ShaderModuleInfo {
        entry_points: entries
            .iter()
            .map(|entry| EntryPointInfo {
                name: entry.name.clone(),
                stage: entry.stage,
                workgroup_size: entry.workgroup_size,
            })
            .collect(),
    };
    Ok(ParsedShader {
        module: Arc::new(module),
        info,
        entries: entries.into(),
    })
}

const fn location(at: naga::SourceLocation) -> SourceLocation {
    SourceLocation {
        line: at.line_number,
        column: at.line_position,
    }
}

/// An error and its sources, joined: naga's top-level messages name only the function.
fn error_chain(err: &dyn std::error::Error) -> String {
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

fn reflect(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<Vec<EntryReflection>, RhiError> {
    let mut entries = Vec::new();
    for (index, entry) in module.entry_points.iter().enumerate() {
        let stage = match entry.stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
            // Mesh, task and ray tracing stages are not part of this crate's API.
            _ => continue,
        };
        let function = info.get_entry_point(index);
        let mut bindings = Vec::new();
        for (handle, var) in module.global_variables.iter() {
            let Some(binding) = &var.binding else {
                continue;
            };
            let used = function
                .global_uses
                .get(handle.index())
                .is_some_and(|uses| !uses.is_empty());
            if used {
                bindings.push(ShaderBinding {
                    group: binding.group,
                    binding: binding.binding,
                    ty: binding_type(module, var)?,
                });
            }
        }
        bindings.sort_by_key(|binding| (binding.group, binding.binding));
        let mut inputs = Vec::new();
        for argument in &entry.function.arguments {
            collect_io(module, argument.binding.as_ref(), argument.ty, &mut inputs)?;
        }
        let mut outputs = Vec::new();
        if let Some(result) = &entry.function.result {
            collect_io(module, result.binding.as_ref(), result.ty, &mut outputs)?;
        }
        inputs.sort_by_key(|(location, _)| *location);
        outputs.sort_by_key(|(location, _)| *location);
        let workgroup_size = match stage {
            ShaderStage::Compute => entry
                .workgroup_size_overrides
                .is_none()
                .then_some(entry.workgroup_size),
            ShaderStage::Vertex | ShaderStage::Fragment => Some([0; 3]),
        };
        entries.push(EntryReflection {
            name: entry.name.clone(),
            stage,
            workgroup_size,
            bindings,
            inputs,
            outputs,
        });
    }
    Ok(entries)
}

fn reflection_error(message: &str) -> RhiError {
    RhiError::Shader {
        message: format!("reflection: {message}"),
        location: None,
    }
}

fn type_of(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Result<&naga::Type, RhiError> {
    module
        .types
        .get_handle(ty)
        .map_err(|_| reflection_error("a type handle is out of range"))
}

fn binding_type(
    module: &naga::Module,
    var: &naga::GlobalVariable,
) -> Result<ShaderBindingType, RhiError> {
    let ty = type_of(module, var.ty)?;
    let size = || u64::from(ty.inner.size(module.to_ctx()));
    Ok(match (var.space, &ty.inner) {
        (naga::AddressSpace::Uniform, _) => ShaderBindingType::Uniform { min_size: size() },
        (naga::AddressSpace::Storage { access }, _) => ShaderBindingType::Storage {
            read_only: !access.contains(naga::StorageAccess::STORE),
            min_size: size(),
        },
        (naga::AddressSpace::Handle, naga::TypeInner::Sampler { comparison }) => {
            ShaderBindingType::Sampler {
                comparison: *comparison,
            }
        }
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            },
        ) => {
            let dimension = match (dim, arrayed) {
                (naga::ImageDimension::D1, _) => TextureViewDimension::D1,
                (naga::ImageDimension::D2, false) => TextureViewDimension::D2,
                (naga::ImageDimension::D2, true) => TextureViewDimension::D2Array,
                (naga::ImageDimension::D3, _) => TextureViewDimension::D3,
                (naga::ImageDimension::Cube, false) => TextureViewDimension::Cube,
                (naga::ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
            };
            match class {
                naga::ImageClass::Sampled { kind, multi } => ShaderBindingType::Texture {
                    sample: match kind {
                        naga::ScalarKind::Sint => ShaderSampleKind::Sint,
                        naga::ScalarKind::Uint => ShaderSampleKind::Uint,
                        _ => ShaderSampleKind::Float,
                    },
                    dimension,
                    multisampled: *multi,
                },
                naga::ImageClass::Depth { multi } => ShaderBindingType::Texture {
                    sample: ShaderSampleKind::Depth,
                    dimension,
                    multisampled: *multi,
                },
                naga::ImageClass::Storage { format, access } => {
                    let read = access.contains(naga::StorageAccess::LOAD);
                    let write = access.contains(naga::StorageAccess::STORE);
                    ShaderBindingType::StorageTexture {
                        format: storage_format(*format),
                        access: match (read, write) {
                            (true, true) => StorageTextureAccess::ReadWrite,
                            (true, false) => StorageTextureAccess::ReadOnly,
                            (false, _) => StorageTextureAccess::WriteOnly,
                        },
                        dimension,
                    }
                }
                naga::ImageClass::External => {
                    ShaderBindingType::Unsupported("external textures are not supported")
                }
            }
        }
        (naga::AddressSpace::Handle, naga::TypeInner::BindingArray { .. }) => {
            ShaderBindingType::Unsupported("binding arrays are not supported")
        }
        _ => ShaderBindingType::Unsupported("this kind of binding is not supported"),
    })
}

const fn storage_format(format: naga::StorageFormat) -> Option<TextureFormat> {
    Some(match format {
        naga::StorageFormat::R8Unorm => TextureFormat::R8Unorm,
        naga::StorageFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        naga::StorageFormat::R16Float => TextureFormat::R16Float,
        naga::StorageFormat::Rgba16Float => TextureFormat::Rgba16Float,
        naga::StorageFormat::R32Float => TextureFormat::R32Float,
        naga::StorageFormat::Rg32Float => TextureFormat::Rg32Float,
        naga::StorageFormat::Rgba32Float => TextureFormat::Rgba32Float,
        naga::StorageFormat::R32Uint => TextureFormat::R32Uint,
        _ => return None,
    })
}

/// Collects the `@location`s of an argument or a result, descending into structs.
fn collect_io(
    module: &naga::Module,
    binding: Option<&naga::Binding>,
    ty: naga::Handle<naga::Type>,
    out: &mut Vec<(u32, IoKind)>,
) -> Result<(), RhiError> {
    let inner = &type_of(module, ty)?.inner;
    match binding {
        Some(naga::Binding::Location { location, .. }) => {
            let scalar = match inner {
                naga::TypeInner::Scalar(scalar) | naga::TypeInner::Vector { scalar, .. } => {
                    Some(scalar.kind)
                }
                _ => None,
            };
            let kind = match scalar {
                Some(naga::ScalarKind::Sint) => IoKind::Sint,
                Some(naga::ScalarKind::Uint) => IoKind::Uint,
                Some(naga::ScalarKind::Float) => IoKind::Float,
                _ => return Err(reflection_error("a @location has a non-numeric type")),
            };
            out.push((*location, kind));
        }
        Some(naga::Binding::BuiltIn(_)) => {}
        None => {
            if let naga::TypeInner::Struct { members, .. } = inner {
                for member in members {
                    collect_io(module, member.binding.as_ref(), member.ty, out)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    const SHADER: &str = r"
struct Globals { tint: vec4<f32> }
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var<storage, read_write> counters: array<u32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VertexOut {
    return VertexOut(vec4<f32>(position, 1.0), uv);
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * globals.tint;
}

@compute @workgroup_size(8, 4)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    counters[id.x] = counters[id.x] + 1u;
}
";

    #[test]
    fn reflects_entry_points() {
        let parsed = parse(ShaderSource::Wgsl(SHADER)).unwrap();
        let names: Vec<_> = parsed
            .info
            .entry_points
            .iter()
            .map(|entry| (entry.name.as_str(), entry.stage))
            .collect();
        assert_eq!(
            names,
            [
                ("vs", ShaderStage::Vertex),
                ("fs", ShaderStage::Fragment),
                ("count", ShaderStage::Compute)
            ]
        );
        let vs = parsed.entry_point(ShaderStage::Vertex, None).unwrap();
        assert!(vs.bindings.is_empty(), "the vertex stage uses no resource");
        assert_eq!(vs.inputs, [(0, IoKind::Float), (1, IoKind::Float)]);
        let fs = parsed
            .entry_point(ShaderStage::Fragment, Some("fs"))
            .unwrap();
        assert_eq!(fs.inputs, [(0, IoKind::Float)]);
        assert_eq!(fs.outputs, [(0, IoKind::Float)]);
        let used: Vec<_> = fs.bindings.iter().map(|b| (b.group, b.binding)).collect();
        assert_eq!(used, [(0, 0), (0, 1), (0, 2)]);
        assert_eq!(
            fs.bindings.first().map(|b| &b.ty),
            Some(&ShaderBindingType::Uniform { min_size: 16 })
        );
        let cs = parsed.entry_point(ShaderStage::Compute, None).unwrap();
        assert_eq!(cs.workgroup_size, Some([8, 4, 1]));
        assert_eq!(
            cs.bindings.first().map(|b| &b.ty),
            Some(&ShaderBindingType::Storage {
                read_only: false,
                min_size: 4
            })
        );
    }

    #[test]
    fn entry_point_resolution() {
        let parsed = parse(ShaderSource::Wgsl(SHADER)).unwrap();
        assert!(matches!(
            parsed.entry_point(ShaderStage::Vertex, Some("fs")),
            Err(RhiError::ShaderInterface {
                stage: ShaderStage::Vertex,
                ..
            })
        ));
        let empty = parse(ShaderSource::default()).unwrap();
        assert!(empty.info.entry_points.is_empty());
        assert!(empty.entry_point(ShaderStage::Compute, None).is_err());
    }

    #[test]
    fn parse_errors_have_a_location() {
        let err = parse(ShaderSource::Wgsl("fn main() {\n  let x = ;\n}")).unwrap_err();
        match err {
            RhiError::Shader {
                location: Some(at), ..
            } => assert_eq!(at.line, 2),
            other => panic!("expected a located shader error, got {other:?}"),
        }
    }

    #[test]
    fn validation_errors_have_a_location() {
        let source = "@compute @workgroup_size(1)\nfn main() {\n  let x: u32 = 1.5;\n}";
        let err = parse(ShaderSource::Wgsl(source)).unwrap_err();
        assert!(
            matches!(err, RhiError::Shader { .. }),
            "expected a shader error, got {err:?}"
        );
    }

    #[test]
    fn spirv_garbage_is_an_error() {
        let err = parse(ShaderSource::SpirV(&[0xdead_beef, 1, 2, 3, 4])).unwrap_err();
        assert!(matches!(err, RhiError::Shader { location: None, .. }));
    }
}
