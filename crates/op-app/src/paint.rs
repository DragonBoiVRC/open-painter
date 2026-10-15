//! Paint layers: per-node UV-space textures, projective brush stamping, stroke history,
//! and the glue that recomposites the layer stack into the output the viewport samples.
//!
//! Channels are stored premultiplied ("over" compositing). The compositor
//! (`composite.rs`) blends every layer into `PaintGpu::output`, which the PBR shader
//! reads: final = output.rgb + fallback * (1 - output.a).

use crate::composite::Compositor;
use eframe::egui_wgpu::wgpu;
use op_core::{Document, Node, NodeId, NodeKind};
use op_mesh::Vertex;
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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StampGlobals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    /// width px, height px, near, far
    viewport: [f32; 4],
}

/// Camera state the stamp pass needs; supplied each frame.
pub struct StampView<'a> {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub size_px: [u32; 2],
    pub near_far: [f32; 2],
    pub depth: &'a wgpu::TextureView,
}

pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
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
    /// Like `Reset`, then restore saved texture contents (opening a project).
    Load { size: u32, blobs: Vec<TextureBlob> },
}

/// The three paintable channels of a layer (and the compositor's working buffers).
pub struct Layer {
    pub size: u32,
    pub base: wgpu::Texture,
    pub metal: wgpu::Texture,
    pub rough: wgpu::Texture,
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
        }
    }

    fn bytes(size: u32) -> u64 {
        size as u64 * size as u64 * (8 + 2 + 2)
    }

    pub fn views(&self) -> [wgpu::TextureView; 3] {
        [&self.base, &self.metal, &self.rough].map(|t| t.create_view(&Default::default()))
    }

    pub fn copy_from(&self, encoder: &mut wgpu::CommandEncoder, src: &Layer) {
        for (s, d) in [(&src.base, &self.base), (&src.metal, &self.metal), (&src.rough, &self.rough)] {
            copy_tex(encoder, s, d, self.size);
        }
    }

    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        let v = self.views();
        clear_textures(encoder, &[&v[0], &v[1], &v[2]], wgpu::Color::TRANSPARENT);
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
            wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4];
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
            make("fs_main", &[target(BASE_FORMAT, over), target(VALUE_FORMAT, over), target(VALUE_FORMAT, over)]);
        let pipeline_mask = make("fs_mask", &[target(MASK_FORMAT, straight)]);

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
        let material_bind = Self::make_material_bind(device, &material_layout, &sampler, &output);
        Self {
            pipeline_content,
            pipeline_mask,
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
    ) -> wgpu::BindGroup {
        let [b, m, r] = layer.views();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint_material_bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&b) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&m) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&r) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
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
        self.material_bind = Self::make_material_bind(device, &self.material_layout, &self.sampler, &self.output);
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

    /// Read back every paint layer's channels and every mask of `doc` for saving.
    pub fn capture_blobs(&self, device: &wgpu::Device, queue: &wgpu::Queue, doc: &Document) -> Vec<TextureBlob> {
        let mut out = Vec::new();
        for id in doc.ids() {
            let Some(node) = doc.find(id) else { continue };
            let mut kinds = Vec::new();
            if matches!(node.kind, NodeKind::Paint) {
                kinds.extend([BlobKind::Base, BlobKind::Metal, BlobKind::Rough]);
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

    /// Read back the composited result: (base RGBA16F, metal RG8, rough RG8).
    pub fn capture_output(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [Vec<u8>; 3] {
        let s = self.size;
        [
            crate::gpu_io::read_texture(device, queue, &self.output.base, s, 8),
            crate::gpu_io::read_texture(device, queue, &self.output.metal, s, 2),
            crate::gpu_io::read_texture(device, queue, &self.output.rough, s, 2),
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
                PaintOp::Stamps { layer, target, stamps } => {
                    let Some(mesh) = &mesh else { continue };
                    let Some(g) = self.store.get(layer) else { continue };
                    let n = stamps.len().min(MAX_STAMPS - self.cursor);
                    if n == 0 {
                        continue;
                    }
                    let (views, pipeline): (Vec<wgpu::TextureView>, _) = match target {
                        Target::Content => match &g.content {
                            Some(c) => (c.views().into(), &self.pipeline_content),
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
                    queue.write_buffer(
                        &self.globals_buf,
                        0,
                        bytemuck::bytes_of(&StampGlobals {
                            view_proj: view.view_proj,
                            camera_pos: view.camera_pos,
                            viewport: [
                                view.size_px[0] as f32,
                                view.size_px[1] as f32,
                                view.near_far[0],
                                view.near_far[1],
                            ],
                        }),
                    );
                    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("stamp_bind"),
                        layout: &self.stamp_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view.depth) },
                        ],
                    });
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
                    pass.draw_indexed(0..mesh.index_count, 0, 0..n as u32);
                    self.cursor += n;
                    dirty = true;
                }
            }
        }

        if dirty || self.last_doc.as_ref() != Some(doc) {
            self.compositor.render(device, encoder, doc, &self.store, &self.output);
            self.last_doc = Some(doc.clone());
        }
    }
}
