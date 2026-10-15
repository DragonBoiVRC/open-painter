//! GPU layer compositor.
//!
//! Walks the layer tree bottom to top. Each visible layer is one fullscreen pass that
//! reads the backdrop and the layer's source (its paint textures, a constant fill, or
//! a group's own isolated composite) and writes the blended result to a second buffer;
//! the two buffers then swap (ping-pong). Groups recurse into fresh buffers.

use crate::paint::{Layer, NodeGpu, BASE_FORMAT, VALUE_FORMAT};
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt};
use op_core::{srgb_to_linear, Document, FillParams, Node, NodeId, NodeKind};
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    color: [f32; 4],
    vals: [f32; 4],
    mode: [u32; 4],
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
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend((1..=7).map(tex));
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
                targets: &[target(BASE_FORMAT), target(VALUE_FORMAT), target(VALUE_FORMAT)],
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
            pool: Vec::new(),
            size,
        }
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
        let (is_fill, color, metal, rough, bits) = match &src {
            Src::Fill(f) => {
                let c = f.base.map(|c| c.map(srgb_to_linear)).unwrap_or([0.0; 3]);
                let bits = f.base.is_some() as u32 | (f.metallic.is_some() as u32) << 1 | (f.roughness.is_some() as u32) << 2;
                (1, c, f.metallic.unwrap_or(0.0), f.roughness.unwrap_or(0.0), bits)
            }
            Src::Tex(_) => (0, [0.0; 3], 0.0, 0.0, 0),
        };
        let params = Params {
            color: [color[0], color[1], color[2], 0.0],
            vals: [node.opacity, metal, rough, 0.0],
            mode: [is_fill, mask.is_some() as u32, node.blend.index(), bits],
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
