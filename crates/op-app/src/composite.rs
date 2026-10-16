//! GPU layer compositor.
//!
//! Walks the layer tree bottom to top. Each visible layer is one fullscreen pass that
//! reads the backdrop and the layer's source (its paint textures, a constant fill, or
//! a group's own isolated composite) and writes the blended result to a second buffer;
//! the two buffers then swap (ping-pong). Groups recurse into fresh buffers.

use crate::images::{dummy_image, GpuImage, SharedImages};
use crate::normal_gpu::NORMAL_FORMAT;
use crate::paint::{Layer, NodeGpu, BASE_FORMAT, VALUE_FORMAT};
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use op_core::{srgb_to_linear, Document, FillParams, Node, NodeId, NodeKind, Slot};
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
}

enum Src<'a> {
    Fill(&'a FillParams),
    Tex(&'a Layer),
}

pub struct Compositor {
    pipeline: wgpu::RenderPipeline,
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
        Self {
            pipeline,
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
                    self.pass(device, encoder, &back, &front, Src::Tex(content), node, mask);
                }
                NodeKind::Fill(f) => self.pass(device, encoder, &back, &front, Src::Fill(f), node, mask),
                NodeKind::Group { children } => {
                    let result = self.composite_nodes(device, encoder, children, store);
                    self.pass(device, encoder, &back, &front, Src::Tex(&result), node, mask);
                    self.pool.push(result);
                }
            }
            std::mem::swap(&mut back, &mut front);
        }
        self.pool.push(front);
        back
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
        let mut slots: [Option<std::sync::Arc<GpuImage>>; 5] = Default::default();
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
                Some(i) if n == 0 => &i.srgb,
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
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&bk[4]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&sv[4]) },
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
        let attachments = dv.each_ref().map(|v| {
            Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops })
        });
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
}
