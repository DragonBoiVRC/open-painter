//! GPU layer compositor.
//!
//! Walks the layer tree bottom to top. Each visible layer is one fullscreen pass that
//! reads the backdrop and the layer's source (its paint textures, a constant fill, or
//! a group's own isolated composite) and writes the blended result to a second buffer;
//! the two buffers then swap (ping-pong). Groups recurse into fresh buffers.

use crate::images::{dummy_image, GpuImage, SharedImages};
use crate::normal_gpu::NORMAL_FORMAT;
use crate::paint::{Layer, NodeGpu, BASE_FORMAT, VALUE_FORMAT, V_NORMAL};
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use op_core::{srgb_to_linear, Document, Filter, FillParams, Node, NodeId, NodeKind, Projection, Slot};
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    color: [f32; 4],
    vals: [f32; 4],
    mode: [u32; 4],
    tex: [u32; 4],
    xf: [f32; 4],
    xf2: [f32; 4],
    dims: [f32; 4],
    /// Mask levels: black, white, gamma, invert (0 or 1).
    lv: [f32; 4],
    /// Colour adjustment: hue (radians), saturation, brightness, contrast.
    adj: [f32; 4],
    /// The blend mode of each channel: base, metallic, roughness, height.
    chan_mode: [u32; 4],
    /// The opacity multiplier of each of those channels (0 when switched off).
    chan_k: [f32; 4],
    /// x: the opacity multiplier of the normal map; y: of the emissive channel; z: its blend mode.
    chan_n: [f32; 4],
    /// Colour of an emissive fill (linear) and, in `w`, 1 if the fill writes it.
    emissive: [f32; 4],
    /// An opacity fill's value and, in `y`, 1 if the fill writes it.
    opac: [f32; 4],
    /// x: the opacity multiplier of the opacity channel; y: its blend mode.
    chan_o: [f32; 4],
}

/// The baked maps a projected fill reads, and the size of the model they were baked from.
#[derive(Clone)]
pub struct ProjectionMaps {
    /// Position, RGBA16F, normalised to the model's bounding box.
    pub position: wgpu::Texture,
    /// World space normal, RGBA8 packed 0..1.
    pub normal: wgpu::Texture,
    /// Where the baked mesh's box starts and how big it is, in units of the whole model's longest
    /// side from its minimum corner (see `ProjectionData`).
    pub offset: [f32; 3],
    pub extent: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProjParams {
    color: [f32; 4],
    vals: [f32; 4],
    emissive: [f32; 4],
    opac: [f32; 4],
    bits: [u32; 4],
    xf: [f32; 4],
    xf2: [f32; 4],
    extent: [f32; 4],
    offset: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    kind: [u32; 4],
    v: [f32; 4],
}

/// The most filters one layer applies, whatever a project file asks for.
const MAX_FILTERS: usize = 16;

enum Src<'a> {
    Fill(&'a FillParams),
    Tex(&'a Layer),
}

pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
    emissive_pipeline: wgpu::RenderPipeline,
    emissive_layout: wgpu::BindGroupLayout,
    proj_pipeline: wgpu::RenderPipeline,
    proj_layout: wgpu::BindGroupLayout,
    projection: Option<ProjectionMaps>,
    filter_pipeline: wgpu::RenderPipeline,
    filter_layout: wgpu::BindGroupLayout,
    layout: wgpu::BindGroupLayout,
    /// 1x1 stand-ins bound when a pass has no source texture / mask.
    dummy: Layer,
    dummy_mask: wgpu::Texture,
    /// Stands in for an image slot a pass does not use.
    dummy_image: GpuImage,
    sampler: wgpu::Sampler,
    images: Option<SharedImages>,
    /// UV offset of the tile this compositor works for, so textures run on across tiles.
    tile: [f32; 2],
    pool: Vec<Layer>,
    size: u32,
}

impl Compositor {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend((1..=11).map(tex));
        let image_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        entries.extend((12..=16).map(image_tex));
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 17,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_layout"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/composite.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let target = |format: wgpu::TextureFormat| Some(wgpu::ColorTargetState::from(format));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[target(BASE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(NORMAL_FORMAT)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // The emissive channel gets its own pass and bindings (see `fs_emissive`).
        let mut emissive_entries = vec![entries[0]];
        emissive_entries.push(tex(7));
        emissive_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 17,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        emissive_entries.push(tex(18));
        emissive_entries.push(tex(19));
        emissive_entries.push(image_tex(20));
        emissive_entries.push(tex(21));
        emissive_entries.push(tex(22));
        emissive_entries.push(image_tex(23));
        let emissive_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("composite_emissive_layout"), entries: &emissive_entries });
        let emissive_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_emissive_pl"),
            bind_group_layouts: &[Some(&emissive_layout)],
            immediate_size: 0,
        });
        let emissive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_emissive"),
            layout: Some(&emissive_pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_emissive"),
                compilation_options: Default::default(),
                targets: &[target(BASE_FORMAT), target(VALUE_FORMAT)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // Projected fills are rendered into a layer of their own first (see `projected_fill`).
        let proj_tex = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let proj_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("projfill_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                proj_tex(1, false),
                proj_tex(2, false),
                proj_tex(3, true),
                proj_tex(4, true),
                proj_tex(5, true),
                proj_tex(6, true),
                proj_tex(7, true),
                proj_tex(9, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let proj_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/projfill.wgsl"));
        let proj_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("projfill_pl"), bind_group_layouts: &[Some(&proj_layout)], immediate_size: 0 });
        let proj_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("projfill"),
            layout: Some(&proj_pl),
            vertex: wgpu::VertexState { module: &proj_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &proj_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[target(BASE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(BASE_FORMAT), target(VALUE_FORMAT), target(NORMAL_FORMAT)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let filter_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let mut filter_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        filter_entries.extend((1..=13).map(filter_tex));
        let filter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("filter_layout"), entries: &filter_entries });
        let filter_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/filter.wgsl"));
        let filter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("filter_pl"),
            bind_group_layouts: &[Some(&filter_layout)],
            immediate_size: 0,
        });
        let filter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("filter"),
            layout: Some(&filter_pl),
            vertex: wgpu::VertexState { module: &filter_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &filter_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[target(BASE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT), target(BASE_FORMAT), target(VALUE_FORMAT), target(NORMAL_FORMAT)],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            emissive_pipeline,
            emissive_layout,
            proj_pipeline,
            proj_layout,
            projection: None,
            filter_pipeline,
            filter_layout,
            layout,
            dummy: Layer::new(device, 1),
            dummy_mask: crate::paint::make_mask_texture(device, 1),
            dummy_image: dummy_image(device),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 8,
                ..Default::default()
            }),
            images: None,
            tile: [0.0, 0.0],
            pool: Vec::new(),
            size,
        }
    }

    pub fn set_images(&mut self, images: SharedImages) {
        self.images = Some(images);
    }

    /// The baked maps projected fills read; without them those fills fall back to the UV layout.
    pub fn set_projection(&mut self, maps: Option<ProjectionMaps>) {
        self.projection = maps;
    }

    pub fn set_tile(&mut self, tile: [f32; 2]) {
        self.tile = tile;
    }

    pub fn set_size(&mut self, size: u32) {
        if size != self.size {
            self.size = size;
            self.pool.clear();
        }
    }

    /// Composite the whole document into `output`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        store: &HashMap<NodeId, NodeGpu>,
        output: &Layer,
    ) {
        let root = self.composite_nodes(device, encoder, &doc.nodes, store);
        output.copy_from(encoder, &root);
        self.pool.push(root);
    }

    fn acquire(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> Layer {
        let layer = self.pool.pop().unwrap_or_else(|| Layer::new(device, self.size));
        layer.clear(encoder);
        layer
    }

    fn composite_nodes(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        nodes: &[Node],
        store: &HashMap<NodeId, NodeGpu>,
    ) -> Layer {
        let mut back = self.acquire(device, encoder);
        let mut front = self.acquire(device, encoder);
        for node in nodes.iter().filter(|n| n.visible) {
            let mask = node
                .mask
                .as_ref()
                .filter(|m| m.enabled)
                .and_then(|_| store.get(&node.id).and_then(|g| g.mask.as_ref()));
            match &node.kind {
                NodeKind::Paint => {
                    let Some(content) = store.get(&node.id).and_then(|g| g.content.as_ref()) else { continue };
                    let filtered = self.filtered(device, encoder, content, &node.filters);
                    self.pass(device, encoder, &back, &front, Src::Tex(filtered.as_ref().unwrap_or(content)), node, mask, true);
                    if let Some(f) = filtered {
                        self.pool.push(f);
                    }
                }
                NodeKind::Fill(f) => match self.projected_fill(device, encoder, f) {
                    Some(layer) => {
                        self.pass(device, encoder, &back, &front, Src::Tex(&layer), node, mask, true);
                        self.pool.push(layer);
                    }
                    None => {
                        // A fill that sets neither emissive nor opacity leaves those channels as they were:
                        // skip their pass and hand the backdrop's textures on instead.
                        let extras = f.emissive.is_some() || f.opacity.is_some() || f.maps.emissive.is_some() || f.maps.opacity.is_some();
                        self.pass(device, encoder, &back, &front, Src::Fill(f), node, mask, extras);
                        if !extras {
                            std::mem::swap(&mut back.emissive, &mut front.emissive);
                            std::mem::swap(&mut back.opacity, &mut front.opacity);
                        }
                    }
                },
                NodeKind::Group { children } => {
                    let result = self.composite_nodes(device, encoder, children, store);
                    let filtered = self.filtered(device, encoder, &result, &node.filters);
                    self.pass(device, encoder, &back, &front, Src::Tex(filtered.as_ref().unwrap_or(&result)), node, mask, true);
                    if let Some(f) = filtered {
                        self.pool.push(f);
                    }
                    self.pool.push(result);
                }
            }
            std::mem::swap(&mut back, &mut front);
        }
        self.pool.push(front);
        back
    }

    /// A fill layer whose textures are projected onto the model, rendered into a layer of its own, or
    /// `None` if it uses the UV layout or the baked maps it needs are missing.
    fn projected_fill(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, f: &FillParams) -> Option<Layer> {
        if f.projection == Projection::Uv || f.maps.is_empty() {
            return None;
        }
        let maps = self.projection.clone()?;
        let lib = self.images.clone()?;
        // The images of the channels that have one, base and emissive through sRGB views.
        let mut views: [Option<std::sync::Arc<GpuImage>>; 7] = Default::default();
        let (mut tex_bits, mut invert_bits) = (0u32, 0u32);
        {
            let lib = lib.lock().expect("image library lock");
            for (n, slot) in Slot::ALL.into_iter().enumerate() {
                if slot == Slot::Normal {
                    continue; // a tangent-space normal map has no meaning on a projection
                }
                if let Some(t) = f.maps.get(slot) {
                    if let Some(img) = lib.get(&t.id) {
                        views[n] = Some(img);
                        tex_bits |= 1 << n;
                        invert_bits |= (t.invert as u32) << n;
                    }
                }
            }
        }
        if tex_bits == 0 {
            return None;
        }
        let constant_bits = f.base.is_some() as u32 | (f.metallic.is_some() as u32) << 1 | (f.roughness.is_some() as u32) << 2 | (f.height.is_some() as u32) << 3;
        let lin = f.base.map_or([0.0; 3], |c| c.map(srgb_to_linear));
        let emissive = f.emissive.map_or([0.0; 4], |c| {
            let l = c.map(srgb_to_linear);
            [l[0], l[1], l[2], 1.0]
        });
        let (sin, cos) = f.uv.rotation.to_radians().sin_cos();
        let sharpness = if f.sharpness.is_finite() { f.sharpness.clamp(0.5, 32.0) } else { 4.0 };
        let params = ProjParams {
            color: [lin[0], lin[1], lin[2], 0.0],
            vals: [0.0, f.metallic.unwrap_or(0.0), f.roughness.unwrap_or(0.0), f.height.unwrap_or(0.0)],
            emissive,
            opac: f.opacity.map_or([0.0; 4], |o| [o, 1.0, 0.0, 0.0]),
            bits: [constant_bits, tex_bits, invert_bits, f.projection.code()],
            xf: [f.uv.tiling[0], f.uv.tiling[1], f.uv.offset[0], f.uv.offset[1]],
            xf2: [cos, sin, sharpness, 0.0],
            extent: [maps.extent[0], maps.extent[1], maps.extent[2], 0.0],
            offset: [maps.offset[0], maps.offset[1], maps.offset[2], 0.0],
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("projfill_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (pos, nrm) = (maps.position.create_view(&Default::default()), maps.normal.create_view(&Default::default()));
        let img = |n: usize| -> &wgpu::TextureView {
            match &views[n] {
                Some(i) if n == 0 || n == 5 => &i.srgb,
                Some(i) => &i.linear,
                None => &self.dummy_image.linear,
            }
        };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("projfill_bind"),
            layout: &self.proj_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&pos) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&nrm) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(img(0)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(img(1)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(img(2)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(img(3)) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(img(5)) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(img(6)) },
            ],
        });
        let layer = self.acquire(device, encoder);
        let dv = layer.views();
        let ops = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        let attachments = dv.each_ref().map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops }));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("projfill_pass"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.proj_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        Some(layer)
    }

    /// `input` run through `filters` in order, or `None` if there is nothing to do.
    fn filtered(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, input: &Layer, filters: &[Filter]) -> Option<Layer> {
        let mut current: Option<Layer> = None;
        for f in filters.iter().take(MAX_FILTERS).map(|f| f.sanitized()) {
            let passes: Vec<(u32, [f32; 4])> = match f {
                Filter::Blur { radius } if radius < 0.5 => continue,
                Filter::Blur { radius } => vec![(0, [radius, 1.0, 0.0, 0.0]), (0, [radius, 0.0, 1.0, 0.0])],
                Filter::Sharpen { amount, .. } if amount <= 0.0 => continue,
                Filter::Sharpen { amount, radius } => vec![(0, [radius, 1.0, 0.0, 0.0]), (0, [radius, 0.0, 1.0, 0.0]), (1, [amount, 0.0, 0.0, 0.0])],
                Filter::Invert => vec![(2, [0.0; 4])],
                Filter::Levels { black, white, gamma } => {
                    if black == 0.0 && white == 1.0 && gamma == 1.0 {
                        continue;
                    }
                    vec![(3, [black, white, gamma, 0.0])]
                }
                Filter::DirectionalBlur { radius, .. } if radius < 0.5 => continue,
                Filter::DirectionalBlur { radius, angle } => {
                    let (s, c) = angle.to_radians().sin_cos();
                    vec![(0, [radius, c, s, 0.0])]
                }
                Filter::Warp { amount, .. } if amount <= 0.0 => continue,
                Filter::Warp { amount, scale } => vec![(4, [amount, scale, 0.0, 0.0])],
                Filter::Threshold { position, softness } => vec![(5, [position, softness, 0.0, 0.0])],
                Filter::Pixelate { size } if size < 1.5 => continue,
                Filter::Pixelate { size } => vec![(6, [size, 0.0, 0.0, 0.0])],
                Filter::HueSaturation { hue, saturation, brightness } => {
                    if hue == 0.0 && saturation == 1.0 && brightness == 1.0 {
                        continue;
                    }
                    vec![(7, [hue / 360.0, saturation, brightness, 0.0])]
                }
            };
            let start = current.take();
            let source = start.as_ref().unwrap_or(input);
            let dst = self.acquire(device, encoder);
            if let Filter::Sharpen { .. } = f {
                // Blur a copy along x then y (blur ping-pongs through `tmp`), then combine with the original.
                let tmp = self.acquire(device, encoder);
                let blurred = self.acquire(device, encoder);
                self.filter_pass(device, encoder, source, source, &tmp, passes[0]);
                self.filter_pass(device, encoder, &tmp, &tmp, &blurred, passes[1]);
                self.filter_pass(device, encoder, source, &blurred, &dst, passes[2]);
                self.pool.push(tmp);
                self.pool.push(blurred);
            } else if passes.len() == 2 {
                let tmp = self.acquire(device, encoder);
                self.filter_pass(device, encoder, source, source, &tmp, passes[0]);
                self.filter_pass(device, encoder, &tmp, &tmp, &dst, passes[1]);
                self.pool.push(tmp);
            } else {
                self.filter_pass(device, encoder, source, source, &dst, passes[0]);
            }
            if let Some(old) = start {
                self.pool.push(old);
            }
            current = Some(dst);
        }
        current
    }

    fn filter_pass(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, a: &Layer, b: &Layer, dst: &Layer, pass: (u32, [f32; 4])) {
        let params = FilterParams { kind: [pass.0, 0, 0, 0], v: pass.1 };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("filter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (av, bv) = (a.views(), b.views());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filter_bind"),
            layout: &self.filter_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&av[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&av[1]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&av[2]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&av[3]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&av[V_NORMAL]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&bv[0]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&bv[1]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&bv[2]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&bv[3]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&av[4]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&bv[4]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&av[5]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&bv[5]) },
            ],
        });
        let dv = dst.views();
        let ops = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        let attachments = dv.each_ref().map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops }));
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("filter_pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(&self.filter_pipeline);
        rp.set_bind_group(0, &bind, &[]);
        rp.draw(0..3, 0..1);
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        backdrop: &Layer,
        dst: &Layer,
        src: Src<'_>,
        node: &Node,
        mask: Option<&wgpu::Texture>,
        extras: bool,
    ) {
        let (is_fill, color, metal, rough, height, bits) = match &src {
            Src::Fill(f) => {
                let c = f.base.map(|c| c.map(srgb_to_linear)).unwrap_or([0.0; 3]);
                let bits = f.base.is_some() as u32
                    | (f.metallic.is_some() as u32) << 1
                    | (f.roughness.is_some() as u32) << 2
                    | (f.height.is_some() as u32) << 3;
                (1, c, f.metallic.unwrap_or(0.0), f.roughness.unwrap_or(0.0), f.height.unwrap_or(0.0), bits)
            }
            Src::Tex(_) => (0, [0.0; 3], 0.0, 0.0, 0.0, 0),
        };
        // Images for the fill's textured channels. A channel whose image is not in the library is
        // simply not textured, so a missing file never breaks the stack.
        let mut slots: [Option<std::sync::Arc<GpuImage>>; 7] = Default::default();
        let (mut tex_bits, mut invert_bits, mut flip) = (0u32, 0u32, 0u32);
        let mut xf = ([1.0f32, 1.0, 0.0, 0.0], [1.0f32, 0.0]);
        if let (Src::Fill(f), Some(lib)) = (&src, &self.images) {
            let lib = lib.lock().expect("image library lock");
            for (n, slot) in Slot::ALL.into_iter().enumerate() {
                if let Some(t) = f.maps.get(slot) {
                    if let Some(img) = lib.get(&t.id) {
                        slots[n] = Some(img);
                        tex_bits |= 1 << n;
                        invert_bits |= (t.invert as u32) << n;
                        if slot == Slot::Normal {
                            flip = t.flip_green as u32;
                        }
                    }
                }
            }
            let (sin, cos) = f.uv.rotation.to_radians().sin_cos();
            xf = ([f.uv.tiling[0], f.uv.tiling[1], f.uv.offset[0], f.uv.offset[1]], [cos, sin]);
        }
        // A textured channel is covered, whatever its constant says.
        let bits = bits | (tex_bits & 0b1111);
        let params = Params {
            color: [color[0], color[1], color[2], 0.0],
            vals: [node.opacity, metal, rough, height],
            mode: [is_fill, mask.is_some() as u32, node.blend.index(), bits],
            chan_mode: [node.channels.base, node.channels.metallic, node.channels.roughness, node.channels.height].map(|c| c.mode(node.blend).index()),
            chan_k: [node.channels.base, node.channels.metallic, node.channels.roughness, node.channels.height].map(|c| c.factor()),
            chan_n: [node.channels.normal.factor(), node.channels.emissive.factor(), node.channels.emissive.mode(node.blend).index() as f32, 0.0],
            emissive: match &src {
                Src::Fill(f) => f.emissive.map_or([0.0; 4], |c| {
                    let l = c.map(srgb_to_linear);
                    [l[0], l[1], l[2], 1.0]
                }),
                Src::Tex(_) => [0.0; 4],
            },
            opac: match &src {
                Src::Fill(f) => f.opacity.map_or([0.0; 4], |o| [o, 1.0, 0.0, 0.0]),
                Src::Tex(_) => [0.0; 4],
            },
            chan_o: [node.channels.opacity.factor(), node.channels.opacity.mode(node.blend).index() as f32, 0.0, 0.0],
            tex: [tex_bits, invert_bits, flip, (!node.adjust.is_identity()) as u32],
            xf: xf.0,
            xf2: [xf.1[0], xf.1[1], self.tile[0], self.tile[1]],
            dims: [self.size as f32, 0.0, 0.0, 0.0],
            adj: {
                let a = node.adjust.sanitized();
                [a.hue.to_radians(), a.saturation, a.brightness, a.contrast]
            },
            lv: {
                let a = node.mask.as_ref().map(|m| m.adjust).unwrap_or_default();
                let finite = |v: f32, d: f32| if v.is_finite() { v } else { d };
                [finite(a.black, 0.0).clamp(0.0, 1.0), finite(a.white, 1.0).clamp(0.0, 1.0), finite(a.gamma, 1.0).clamp(0.1, 10.0), a.invert as u32 as f32]
            },
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bk = backdrop.views();
        let sv = match &src {
            Src::Tex(l) => l.views(),
            Src::Fill(_) => self.dummy.views(),
        };
        let mv = mask.unwrap_or(&self.dummy_mask).create_view(&Default::default());
        let img_view = |n: usize| -> &wgpu::TextureView {
            match &slots[n] {
                // Base colour is read as sRGB; every other slot is data.
                Some(i) if n == 0 || n == 5 => &i.srgb,
                Some(i) => &i.linear,
                None => &self.dummy_image.linear,
            }
        };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&bk[0]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&bk[1]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&bk[2]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&sv[0]) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&sv[1]) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&sv[2]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&mv) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&bk[3]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&sv[3]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&bk[V_NORMAL]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&sv[V_NORMAL]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(img_view(0)) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(img_view(1)) },
                wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::TextureView(img_view(2)) },
                wgpu::BindGroupEntry { binding: 15, resource: wgpu::BindingResource::TextureView(img_view(3)) },
                wgpu::BindGroupEntry { binding: 16, resource: wgpu::BindingResource::TextureView(img_view(4)) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let dv = dst.views();
        let ops = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        // base, metallic, roughness, height and the fill normal; emissive and opacity follow.
        let main = [&dv[0], &dv[1], &dv[2], &dv[3], &dv[V_NORMAL]];
        let attachments = main.map(|v| Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops }));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        if !extras {
            return;
        }
        let emissive_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_emissive_bind"),
            layout: &self.emissive_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&mv) },
                wgpu::BindGroupEntry { binding: 17, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(&bk[4]) },
                wgpu::BindGroupEntry { binding: 19, resource: wgpu::BindingResource::TextureView(&sv[4]) },
                wgpu::BindGroupEntry { binding: 20, resource: wgpu::BindingResource::TextureView(img_view(5)) },
                wgpu::BindGroupEntry { binding: 21, resource: wgpu::BindingResource::TextureView(&bk[5]) },
                wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::TextureView(&sv[5]) },
                wgpu::BindGroupEntry { binding: 23, resource: wgpu::BindingResource::TextureView(img_view(6)) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite_emissive_pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment { view: &dv[4], depth_slice: None, resolve_target: None, ops }),
                Some(wgpu::RenderPassColorAttachment { view: &dv[5], depth_slice: None, resolve_target: None, ops }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.emissive_pipeline);
        pass.set_bind_group(0, &emissive_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
