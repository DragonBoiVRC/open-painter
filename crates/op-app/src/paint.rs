//! Paint layers: per-node UV-space textures, projective brush stamping, stroke history,
//! and the glue that recomposites the layer stack into the output the viewport samples.
//!
//! Channels are stored premultiplied ("over" compositing). The compositor
//! (`composite.rs`) blends every layer into `PaintGpu::output`, which the PBR shader
//! reads: final = output.rgb + fallback * (1 - output.a).

use crate::composite::Compositor;
use crate::normal_gpu::NormalPass;
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use op_core::{Document, Node, NodeId, NodeKind};
use op_mesh::{Tile, Vertex};
use op_project::{BlobKind, TextureBlob};
use std::collections::{HashMap, HashSet};

pub const BASE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const VALUE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Stamp capacity of the per-frame instance buffer.
pub const MAX_STAMPS: usize = 4096;
/// GPU memory we allow undo snapshots to use.
const HISTORY_BUDGET_BYTES: u64 = 512 << 20;
pub const DEFAULT_TEXTURE_SIZE: u32 = 2048;

fn expected_bytes(size: u32, kind: BlobKind) -> usize {
    size as usize * size as usize * kind.bytes_per_texel()
}

/// One brush dab. Layout must match the instance attributes in `stamp.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Stamp {
    /// Center in viewport pixels (x, y), radius in pixels, hardness 0..1.
    pub brush: [f32; 4],
    /// Linear RGB.
    pub color: [f32; 4],
    /// metallic, roughness, opacity. For mask stamps `x` is the mask value (1 reveals).
    pub values: [f32; 4],
    /// 0/1 per channel: base color, metallic, roughness.
    pub enables: [f32; 4],
    /// Brush shape: rotation in radians, shape kind (see `shape_dist` in the stamp shaders), noise seed.
    pub shape: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StampGlobals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    /// width px, height px, near, far
    viewport: [f32; 4],
    /// UV offset of the tile being painted (u, v, 0, 0).
    tile: [f32; 4],
    /// Stencil: centre px (xy), half-size px (zw).
    stencil: [f32; 4],
    /// Stencil: cos, sin of its rotation, 1 if a stencil is on, 1 if mirrored.
    stencil_xf: [f32; 4],
}

/// Camera state the stamp pass needs; supplied each frame.
pub struct StampView<'a> {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub size_px: [u32; 2],
    pub near_far: [f32; 2],
    pub depth: &'a wgpu::TextureView,
    /// The UV the camera saw at each pixel (Rg32Float, -1 where nothing was drawn).
    pub uv: &'a wgpu::TextureView,
}

pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    /// The triangles to draw: `index_count` indices starting at `first_index` (one texture set).
    pub first_index: u32,
    pub index_count: u32,
}

/// What a stroke writes into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Content,
    Mask,
}

#[derive(Clone, Debug)]
pub enum PaintOp {
    /// Snapshot the target so the stroke that follows can be undone.
    BeginStroke { layer: NodeId, target: Target },
    Stamps { layer: NodeId, target: Target, stamps: Vec<Stamp> },
    /// Stamps that copy paint from elsewhere on the surface (smudge and clone); content only.
    Smudge { layer: NodeId, stamps: Vec<Stamp> },
    /// Stamps drawn directly in UV space: `brush` is (u, v, radius in UV units, hardness).
    Stamps2d { layer: NodeId, target: Target, stamps: Vec<Stamp> },
    /// Reset the target (content: transparent; mask: its initial value). Pair with BeginStroke.
    Clear { layer: NodeId, target: Target },
    UndoStroke,
    RedoStroke,
    /// A new edit invalidated every redo-able stroke.
    DropRedo,
    /// Free textures of nodes that are neither in the document nor in `keep`.
    Gc(Vec<NodeId>),
    /// Drop all paint and stroke history and (re)allocate at this resolution.
    Reset(u32),
    /// Replace a layer's mask with generated content (one byte per texel). Ignored if the
    /// resolution no longer matches or the layer has no mask.
    SetMask { layer: NodeId, size: u32, bytes: Vec<u8> },
    /// Copy each (source, destination) node's paint and mask pixels (layer duplication). The
    /// destinations already exist in the document; missing sources are skipped.
    CopyLayers(Vec<(NodeId, NodeId)>),
    /// The baked tangent-space normal map of this set as RGBA8 (`None` removes it). It is combined
    /// with painted height in the viewport's normal map. Ignored if `size` is not the set's.
    SetBakedNormal { size: u32, rgba8: Option<Vec<u8>> },
    /// Like `Reset`, then restore saved texture contents (opening a project).
    Load { size: u32, blobs: Vec<TextureBlob> },
}

/// The four paintable channels of a layer (and the compositor's working buffers).
pub struct Layer {
    pub size: u32,
    pub base: wgpu::Texture,
    pub metal: wgpu::Texture,
    pub rough: wgpu::Texture,
    /// (height * coverage, coverage); 0.5 is flat.
    pub height: wgpu::Texture,
    /// Tangent-space normal (xyz packed to 0..1) with coverage in alpha; only fill layers with a
    /// normal map put anything here.
    pub normal: wgpu::Texture,
}

fn texture(device: &wgpu::Device, label: &str, size: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

pub fn make_mask_texture(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    texture(device, "paint_mask", size, MASK_FORMAT)
}

fn copy_tex(encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture, size: u32) {
    let extent = wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 };
    encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), extent);
}

fn clear_textures(encoder: &mut wgpu::CommandEncoder, views: &[&wgpu::TextureView], color: wgpu::Color) {
    let ops = wgpu::Operations { load: wgpu::LoadOp::Clear(color), store: wgpu::StoreOp::Store };
    let attachments: Vec<_> = views
        .iter()
        .map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops }))
        .collect();
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("paint_clear"),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

impl Layer {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        Self {
            size,
            base: texture(device, "paint_base", size, BASE_FORMAT),
            metal: texture(device, "paint_metal", size, VALUE_FORMAT),
            rough: texture(device, "paint_rough", size, VALUE_FORMAT),
            height: texture(device, "paint_height", size, VALUE_FORMAT),
            normal: texture(device, "paint_normal_layer", size, crate::normal_gpu::NORMAL_FORMAT),
        }
    }

    fn bytes(size: u32) -> u64 {
        size as u64 * size as u64 * (8 + 2 + 2 + 2 + 4)
    }

    pub fn views(&self) -> [wgpu::TextureView; 5] {
        [&self.base, &self.metal, &self.rough, &self.height, &self.normal].map(|t| t.create_view(&Default::default()))
    }

    pub fn copy_from(&self, encoder: &mut wgpu::CommandEncoder, src: &Layer) {
        for (s, d) in [(&src.base, &self.base), (&src.metal, &self.metal), (&src.rough, &self.rough), (&src.height, &self.height), (&src.normal, &self.normal)] {
            copy_tex(encoder, s, d, self.size);
        }
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        let v = self.views();
        clear_textures(encoder, &[&v[0], &v[1], &v[2], &v[3], &v[4]], wgpu::Color::TRANSPARENT);
    }
}

/// GPU data owned by one document node.
#[derive(Default)]
pub struct NodeGpu {
    pub content: Option<Layer>,
    pub mask: Option<wgpu::Texture>,
}

enum SnapTex {
    Content(Layer),
    Mask(wgpu::Texture),
}

struct StrokeSnap {
    layer: NodeId,
    tex: SnapTex,
}

impl StrokeSnap {
    fn bytes(&self, size: u32) -> u64 {
        match self.tex {
            SnapTex::Content(_) => Layer::bytes(size),
            SnapTex::Mask(_) => size as u64 * size as u64,
        }
    }
}

pub struct PaintGpu {
    pipeline_content: wgpu::RenderPipeline,
    pipeline_mask: wgpu::RenderPipeline,
    pipeline_smudge: wgpu::RenderPipeline,
    smudge_layout: wgpu::BindGroupLayout,
    /// A copy of the layer being smudged, so stamps read what was there before them.
    scratch: Option<Layer>,
    images: Option<crate::images::SharedImages>,
    /// A 1x1 stand-in bound for a brush picture that is not set (the shader ignores it then).
    white: crate::images::GpuImage,
    /// The same brushes drawn directly in UV space (the 2D view).
    pipeline_content_2d: wgpu::RenderPipeline,
    pipeline_mask_2d: wgpu::RenderPipeline,
    stamp_layout: wgpu::BindGroupLayout,
    pub material_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    globals_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    cursor: usize,
    size: u32,
    /// Per-node paint data, keyed by document node id.
    pub store: HashMap<NodeId, NodeGpu>,
    /// Composited result of the whole layer stack; what the viewport samples.
    pub output: Layer,
    material_bind: wgpu::BindGroup,
    pub(crate) normal: NormalPass,
    /// Which UV tile this store holds (the first tile unless UV tiles are in use).
    pub tile: Tile,
    tile_buf: wgpu::Buffer,
    compositor: Compositor,
    last_doc: Option<Document>,
    undo: Vec<StrokeSnap>,
    redo: Vec<StrokeSnap>,
}

impl PaintGpu {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let stamp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stamp_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // The brush's alpha picture and the stencil picture, and how to sample them.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paint_material_layout"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                tex_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/stamp.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stamp_pl"),
            bind_group_layouts: &[Some(&stamp_layout)],
            immediate_size: 0,
        });
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let straight = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let target = |format, c: wgpu::BlendComponent| {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState { color: c, alpha: c }),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        let instance_attrs =
            wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4];
        let make = |entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &vertex_attrs,
                        }),
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Stamp>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &instance_attrs,
                        }),
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: Default::default(), // no culling: UV winding is not guaranteed
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_content =
            make("fs_main", &[target(BASE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over)]);
        let pipeline_mask = make("fs_mask", &[target(MASK_FORMAT, straight)]);
        let smudge_tex = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let smudge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smudge_layout"),
            entries: &[
                smudge_tex(0, false),
                smudge_tex(1, true),
                smudge_tex(2, true),
                smudge_tex(3, true),
                smudge_tex(4, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let smudge_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smudge_pl"),
            bind_group_layouts: &[Some(&stamp_layout), Some(&smudge_layout)],
            immediate_size: 0,
        });
        let pipeline_smudge = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fs_smudge"),
            layout: Some(&smudge_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Stamp>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_smudge"),
                compilation_options: Default::default(),
                targets: &[target(BASE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let shader_2d = device.create_shader_module(wgpu::include_wgsl!("shaders/stamp2d.wgsl"));
        let pl_2d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("stamp2d_pl"), bind_group_layouts: &[], immediate_size: 0 });
        let make_2d = |entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl_2d),
                vertex: wgpu::VertexState {
                    module: &shader_2d,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Stamp>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    })],
                },
                fragment: Some(wgpu::FragmentState { module: &shader_2d, entry_point: Some(entry), compilation_options: Default::default(), targets }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_content_2d =
            make_2d("fs_main", &[target(BASE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over)]);
        let pipeline_mask_2d = make_2d("fs_mask", &[target(MASK_FORMAT, straight)]);

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stamp_globals"),
            size: std::mem::size_of::<StampGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stamp_instances"),
            size: (MAX_STAMPS * std::mem::size_of::<Stamp>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let output = Layer::new(device, size);
        let normal = NormalPass::new(device, size);
        let tile_buf = Self::tile_buffer(device, Tile::FIRST);
        let material_bind = Self::make_material_bind(device, &material_layout, &sampler, &output, &normal, &tile_buf);
        Self {
            pipeline_content,
            pipeline_mask,
            pipeline_smudge,
            smudge_layout,
            scratch: None,
            images: None,
            white: crate::images::dummy_image(device),
            pipeline_content_2d,
            pipeline_mask_2d,
            stamp_layout,
            material_layout,
            sampler,
            globals_buf,
            instance_buf,
            cursor: 0,
            size,
            store: HashMap::new(),
            output,
            material_bind,
            normal,
            tile: Tile::FIRST,
            tile_buf,
            compositor: Compositor::new(device, size),
            last_doc: None,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    fn make_material_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        layer: &Layer,
        normal: &NormalPass,
        tile_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let [b, m, r, _, _] = layer.views();
        let n = normal.target.create_view(&Default::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint_material_bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&m) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&r) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&n) },
                wgpu::BindGroupEntry { binding: 5, resource: tile_buf.as_entire_binding() },
            ],
        })
    }

    fn tile_buffer(device: &wgpu::Device, tile: Tile) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tile_offset"),
            contents: bytemuck::cast_slice(&[tile.u as f32, tile.v as f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Make this store hold `tile`: its textures are sampled and painted at that UV offset.
    pub fn set_tile(&mut self, device: &wgpu::Device, tile: Tile) {
        if self.tile == tile {
            return;
        }
        self.tile = tile;
        self.compositor.set_tile([tile.u as f32, tile.v as f32]);
        self.tile_buf = Self::tile_buffer(device, tile);
        self.material_bind = Self::make_material_bind(device, &self.material_layout, &self.sampler, &self.output, &self.normal, &self.tile_buf);
    }

    pub fn material_bind(&self) -> &wgpu::BindGroup {
        &self.material_bind
    }

    #[cfg(test)]
    pub fn undo_depth(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }

    /// Create textures for every paint layer and mask in `doc` that lacks them.
    fn ensure_store(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, doc: &Document) {
        fn walk(nodes: &[Node], f: &mut impl FnMut(&Node)) {
            for n in nodes {
                f(n);
                if let NodeKind::Group { children } = &n.kind {
                    walk(children, f);
                }
            }
        }
        let size = self.size;
        let store = &mut self.store;
        walk(&doc.nodes, &mut |n| {
            if matches!(n.kind, NodeKind::Paint) {
                let g = store.entry(n.id).or_default();
                if g.content.is_none() {
                    g.content = Some(Layer::new(device, size)); // wgpu zero-initialises
                }
            }
            if let Some(m) = &n.mask {
                let g = store.entry(n.id).or_default();
                if g.mask.is_none() {
                    let t = make_mask_texture(device, size);
                    let v = if m.init_white { 1.0 } else { 0.0 };
                    clear_textures(
                        encoder,
                        &[&t.create_view(&Default::default())],
                        wgpu::Color { r: v, g: 0.0, b: 0.0, a: 0.0 },
                    );
                    g.mask = Some(t);
                }
            }
        });
    }

    fn gc(&mut self, doc: &Document, keep: &[NodeId]) {
        let mut live: HashSet<NodeId> = doc.ids().into_iter().collect();
        live.extend(keep.iter().copied());
        live.extend(self.undo.iter().chain(&self.redo).map(|s| s.layer));
        self.store.retain(|id, _| live.contains(id));
    }

    fn snapshot_of(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, layer: NodeId, target: Target) -> Option<StrokeSnap> {
        let g = self.store.get(&layer)?;
        let tex = match target {
            Target::Content => {
                let snap = Layer::new(device, self.size);
                snap.copy_from(encoder, g.content.as_ref()?);
                SnapTex::Content(snap)
            }
            Target::Mask => {
                let snap = make_mask_texture(device, self.size);
                copy_tex(encoder, g.mask.as_ref()?, &snap, self.size);
                SnapTex::Mask(snap)
            }
        };
        Some(StrokeSnap { layer, tex })
    }

    /// Write `snap` into the live texture and return a snapshot of what was there.
    fn swap_snapshot(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, snap: StrokeSnap) -> Option<StrokeSnap> {
        let target = match snap.tex {
            SnapTex::Content(_) => Target::Content,
            SnapTex::Mask(_) => Target::Mask,
        };
        let current = self.snapshot_of(device, encoder, snap.layer, target)?;
        let g = self.store.get(&snap.layer)?;
        match (&snap.tex, target) {
            (SnapTex::Content(s), Target::Content) => g.content.as_ref()?.copy_from(encoder, s),
            (SnapTex::Mask(s), Target::Mask) => copy_tex(encoder, s, g.mask.as_ref()?, self.size),
            _ => return None,
        }
        Some(current)
    }

    /// Drop all paint and history; reallocate everything at `size`.
    fn reset(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, doc: &Document, size: u32) {
        self.size = size;
        self.store.clear();
        self.undo.clear();
        self.redo.clear();
        self.output = Layer::new(device, size);
        self.output.clear(encoder);
        self.scratch = None;
        self.normal.resize(device, size);
        self.material_bind = Self::make_material_bind(device, &self.material_layout, &self.sampler, &self.output, &self.normal, &self.tile_buf);
        self.compositor.set_size(size);
        self.last_doc = None;
        self.ensure_store(device, encoder, doc);
    }

    fn texture_for(&self, node: NodeId, kind: BlobKind) -> Option<&wgpu::Texture> {
        let g = self.store.get(&node)?;
        match kind {
            BlobKind::Base => g.content.as_ref().map(|c| &c.base),
            BlobKind::Metal => g.content.as_ref().map(|c| &c.metal),
            BlobKind::Rough => g.content.as_ref().map(|c| &c.rough),
            BlobKind::Height => g.content.as_ref().map(|c| &c.height),
            BlobKind::Mask => g.mask.as_ref(),
        }
    }

    fn upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, blob: &TextureBlob) {
        if let Some(tex) = self.texture_for(blob.node, blob.kind) {
            if blob.bytes.len() == expected_bytes(self.size, blob.kind) {
                crate::gpu_io::write_texture(device, encoder, tex, self.size, blob.kind.bytes_per_texel() as u32, &blob.bytes);
            }
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// Forget the last composite, so the next frame composites again (an image arrived, say).
    pub fn invalidate(&mut self) {
        self.last_doc = None;
    }

    /// Let fill layers read from the shared image library.
    pub fn set_images(&mut self, images: crate::images::SharedImages) {
        self.images = Some(images.clone());
        self.compositor.set_images(images);
    }

    /// Reallocate at `size` and composite `doc` right away (no paint is kept). Used to grow a set
    /// that was held at a placeholder size, or to shrink one that has not been touched.
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, doc: &Document, size: u32) {
        let mut enc = device.create_command_encoder(&Default::default());
        self.reset(device, &mut enc, doc, size);
        self.compositor.render(device, &mut enc, doc, &self.store, &self.output);
        self.normal.run(device, &mut enc, &self.output.height, &self.output.normal, doc.height_strength);
        self.last_doc = Some(doc.clone());
        queue.submit([enc.finish()]);
    }

    /// Read back every paint layer's channels and every mask of `doc` for saving.
    pub fn capture_blobs(&self, device: &wgpu::Device, queue: &wgpu::Queue, doc: &Document) -> Vec<TextureBlob> {
        let mut out = Vec::new();
        for id in doc.ids() {
            let Some(node) = doc.find(id) else { continue };
            let mut kinds = Vec::new();
            if matches!(node.kind, NodeKind::Paint) {
                kinds.extend([BlobKind::Base, BlobKind::Metal, BlobKind::Rough, BlobKind::Height]);
            }
            if node.mask.is_some() {
                kinds.push(BlobKind::Mask);
            }
            for kind in kinds {
                if let Some(tex) = self.texture_for(id, kind) {
                    let bytes = crate::gpu_io::read_texture(device, queue, tex, self.size, kind.bytes_per_texel() as u32);
                    out.push(TextureBlob { node: id, kind, bytes });
                }
            }
        }
        out
    }

    /// Read back the composited result: (base RGBA16F, metal RG8, rough RG8, height RG8, layer normal RGBA8).
    pub fn capture_output(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [Vec<u8>; 5] {
        let s = self.size;
        [
            crate::gpu_io::read_texture(device, queue, &self.output.base, s, 8),
            crate::gpu_io::read_texture(device, queue, &self.output.metal, s, 2),
            crate::gpu_io::read_texture(device, queue, &self.output.rough, s, 2),
            crate::gpu_io::read_texture(device, queue, &self.output.height, s, 2),
            crate::gpu_io::read_texture(device, queue, &self.output.normal, s, 4),
        ]
    }

    fn trim_history(&mut self) {
        let size = self.size;
        let mut total: u64 = self.undo.iter().map(|s| s.bytes(size)).sum();
        while total > HISTORY_BUDGET_BYTES && self.undo.len() > 1 {
            total -= self.undo.remove(0).bytes(size);
        }
    }

    /// Apply this frame's operations, then recomposite the layer stack if anything
    /// changed. Must run before the scene pass so the viewport shows the result at once.
    /// Everything the stamp shaders need for this frame: camera, tile and stencil placement.
    fn stamp_globals(&self, view: &StampView<'_>, stencil_on: bool, place: [f32; 4], flip: bool) -> StampGlobals {
        let (w, h) = (view.size_px[0] as f32, view.size_px[1] as f32);
        // `place[2]` is the stencil's height as a fraction of the viewport's.
        let half_h = place[2] * h * 0.5;
        StampGlobals {
            view_proj: view.view_proj,
            camera_pos: view.camera_pos,
            viewport: [w, h, view.near_far[0], view.near_far[1]],
            tile: [self.tile.u as f32, self.tile.v as f32, 0.0, 0.0],
            stencil: [place[0] * w, place[1] * h, half_h, half_h],
            stencil_xf: [place[3].cos(), place[3].sin(), if stencil_on { 1.0 } else { 0.0 }, if flip { 1.0 } else { 0.0 }],
        }
    }

    /// Upload the globals and bind them with the depth buffer and the brush pictures.
    fn stamp_bind(&self, device: &wgpu::Device, queue: &wgpu::Queue, view: &StampView<'_>) -> wgpu::BindGroup {
        let (alpha, stencil, place, flip) = match &self.images {
            Some(lib) => {
                let lib = lib.lock().expect("image library lock");
                let b = lib.brush.clone();
                (b.alpha.as_deref().and_then(|id| lib.get(id)), b.stencil.as_deref().and_then(|id| lib.get(id)), b.stencil_place, b.stencil_flip)
            }
            None => (None, None, [0.5, 0.5, 0.5, 0.0], false),
        };
        let mut globals = self.stamp_globals(view, stencil.is_some(), place, flip);
        if let Some(img) = &stencil {
            // Keep the picture's proportions: `half_h` stays, the half-width follows the aspect.
            let size = img.texture.size();
            globals.stencil[2] = globals.stencil[3] * size.width as f32 / size.height.max(1) as f32;
        }
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        let alpha_view = alpha.as_ref().map(|i| &i.linear).unwrap_or(&self.white.linear);
        let stencil_view = stencil.as_ref().map(|i| &i.linear).unwrap_or(&self.white.linear);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stamp_bind"),
            layout: &self.stamp_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view.depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(alpha_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(stencil_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        ops: &[PaintOp],
        mesh: Option<MeshBuffers<'_>>,
        view: &StampView<'_>,
    ) {
        self.cursor = 0;
        self.ensure_store(device, encoder, doc);
        let mut dirty = false;
        for op in ops {
            match op {
                PaintOp::BeginStroke { layer, target } => {
                    if let Some(s) = self.snapshot_of(device, encoder, *layer, *target) {
                        self.undo.push(s);
                        self.redo.clear();
                        self.trim_history();
                    }
                }
                PaintOp::Clear { layer, target } => {
                    if let Some(g) = self.store.get(layer) {
                        match target {
                            Target::Content => {
                                if let Some(c) = &g.content {
                                    c.clear(encoder);
                                }
                            }
                            Target::Mask => {
                                let white = doc.find(*layer).and_then(|n| n.mask.as_ref()).is_none_or(|m| m.init_white);
                                if let Some(m) = &g.mask {
                                    let v = if white { 1.0 } else { 0.0 };
                                    clear_textures(
                                        encoder,
                                        &[&m.create_view(&Default::default())],
                                        wgpu::Color { r: v, g: 0.0, b: 0.0, a: 0.0 },
                                    );
                                }
                            }
                        }
                        dirty = true;
                    }
                }
                PaintOp::UndoStroke => {
                    if let Some(snap) = self.undo.pop() {
                        if let Some(cur) = self.swap_snapshot(device, encoder, snap) {
                            self.redo.push(cur);
                        }
                        dirty = true;
                    }
                }
                PaintOp::RedoStroke => {
                    if let Some(snap) = self.redo.pop() {
                        if let Some(cur) = self.swap_snapshot(device, encoder, snap) {
                            self.undo.push(cur);
                        }
                        dirty = true;
                    }
                }
                PaintOp::SetMask { layer, size, bytes } => {
                    let expected = *size as usize * *size as usize;
                    if *size == self.size && bytes.len() == expected {
                        if let Some(tex) = self.store.get(layer).and_then(|g| g.mask.as_ref()) {
                            crate::gpu_io::write_texture(device, encoder, tex, *size, 1, bytes);
                            dirty = true;
                        }
                    }
                }
                PaintOp::CopyLayers(pairs) => {
                    for (from, to) in pairs {
                        if from == to {
                            continue;
                        }
                        let size = self.size;
                        let Some(src) = self.store.remove(from) else { continue };
                        if let Some(dst) = self.store.get_mut(to) {
                            if let (Some(s), Some(d)) = (&src.content, &dst.content) {
                                d.copy_from(encoder, s);
                            }
                            if let (Some(s), Some(d)) = (&src.mask, &dst.mask) {
                                copy_tex(encoder, s, d, size);
                            }
                        }
                        self.store.insert(*from, src);
                    }
                    dirty = true;
                }
                PaintOp::SetBakedNormal { size, rgba8 } => {
                    if *size == self.size {
                        self.normal.set_baked(device, queue, rgba8.as_deref());
                        dirty = true;
                    }
                }
                PaintOp::DropRedo => self.redo.clear(),
                PaintOp::Gc(keep) => self.gc(doc, keep),
                PaintOp::Reset(size) => {
                    self.reset(device, encoder, doc, *size);
                    dirty = true;
                }
                PaintOp::Load { size, blobs } => {
                    self.reset(device, encoder, doc, *size);
                    for b in blobs {
                        self.upload(device, encoder, b);
                    }
                    dirty = true;
                }
                PaintOp::Stamps2d { layer, target, stamps } => {
                    let Some(g) = self.store.get(layer) else { continue };
                    let n = stamps.len().min(MAX_STAMPS - self.cursor);
                    if n == 0 {
                        continue;
                    }
                    let (views, pipeline): (Vec<wgpu::TextureView>, _) = match target {
                        Target::Content => match &g.content {
                            Some(c) => (c.views()[..4].to_vec(), &self.pipeline_content_2d),
                            None => continue,
                        },
                        Target::Mask => match &g.mask {
                            Some(m) => (vec![m.create_view(&Default::default())], &self.pipeline_mask_2d),
                            None => continue,
                        },
                    };
                    let stride = std::mem::size_of::<Stamp>() as u64;
                    let offset = self.cursor as u64 * stride;
                    queue.write_buffer(&self.instance_buf, offset, bytemuck::cast_slice(&stamps[..n]));
                    let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
                    let attachments: Vec<_> = views
                        .iter()
                        .map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: load }))
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stamp2d_pass"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, self.instance_buf.slice(offset..offset + n as u64 * stride));
                    pass.draw(0..6, 0..n as u32);
                    self.cursor += n;
                    dirty = true;
                }
                PaintOp::Smudge { layer, stamps } => {
                    let Some(mesh) = &mesh else { continue };
                    let Some(c) = self.store.get(layer).and_then(|g| g.content.as_ref()) else { continue };
                    let n = stamps.len().min(MAX_STAMPS - self.cursor);
                    if n == 0 {
                        continue;
                    }
                    if self.scratch.as_ref().is_none_or(|s| s.size != self.size) {
                        self.scratch = Some(Layer::new(device, self.size));
                    }
                    let scratch = self.scratch.as_ref().unwrap();
                    scratch.copy_from(encoder, c);
                    let sv = scratch.views();
                    let smudge_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("smudge_bind"),
                        layout: &self.smudge_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view.uv) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sv[0]) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&sv[1]) },
                            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sv[2]) },
                            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&sv[3]) },
                            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });
                    let views = c.views()[..4].to_vec();
                    let stride = std::mem::size_of::<Stamp>() as u64;
                    let offset = self.cursor as u64 * stride;
                    queue.write_buffer(&self.instance_buf, offset, bytemuck::cast_slice(&stamps[..n]));
                    let bind = self.stamp_bind(device, queue, view);
                    let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
                    let attachments: Vec<_> = views
                        .iter()
                        .map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: load }))
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("smudge_pass"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&self.pipeline_smudge);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.set_bind_group(1, &smudge_bind, &[]);
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instance_buf.slice(offset..offset + n as u64 * stride));
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(mesh.first_index..mesh.first_index + mesh.index_count, 0, 0..n as u32);
                    self.cursor += n;
                    dirty = true;
                }
                PaintOp::Stamps { layer, target, stamps } => {
                    let Some(mesh) = &mesh else { continue };
                    let Some(g) = self.store.get(layer) else { continue };
                    let n = stamps.len().min(MAX_STAMPS - self.cursor);
                    if n == 0 {
                        continue;
                    }
                    let (views, pipeline): (Vec<wgpu::TextureView>, _) = match target {
                        Target::Content => match &g.content {
                            Some(c) => (c.views()[..4].to_vec(), &self.pipeline_content),
                            None => continue,
                        },
                        Target::Mask => match &g.mask {
                            Some(m) => (vec![m.create_view(&Default::default())], &self.pipeline_mask),
                            None => continue,
                        },
                    };
                    let stride = std::mem::size_of::<Stamp>() as u64;
                    let offset = self.cursor as u64 * stride;
                    queue.write_buffer(&self.instance_buf, offset, bytemuck::cast_slice(&stamps[..n]));
                    let bind = self.stamp_bind(device, queue, view);
                    let load = wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store };
                    let attachments: Vec<_> = views
                        .iter()
                        .map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops: load }))
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("stamp_pass"),
                        color_attachments: &attachments,
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instance_buf.slice(offset..offset + n as u64 * stride));
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(mesh.first_index..mesh.first_index + mesh.index_count, 0, 0..n as u32);
                    self.cursor += n;
                    dirty = true;
                }
            }
        }

        if dirty || self.last_doc.as_ref() != Some(doc) {
            self.compositor.render(device, encoder, doc, &self.store, &self.output);
            self.normal.run(device, encoder, &self.output.height, &self.output.normal, doc.height_strength);
            self.last_doc = Some(doc.clone());
        }
    }
}
