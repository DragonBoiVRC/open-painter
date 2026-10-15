//! 3D viewport: renders the model into an offscreen target, then blits it into
//! the egui panel. egui's own pass has no depth buffer, hence the offscreen step.

use eframe::egui_wgpu::{self, wgpu, wgpu::util::DeviceExt};
use crate::paint::{MeshBuffers, PaintGpu, PaintOp, StampView};
use op_core::Document;
use op_mesh::{Mesh, Vertex};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub base_color: [f32; 4],
    pub params: [f32; 4],
}

struct Offscreen {
    size: [u32; 2],
    color: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blit_bind: wgpu::BindGroup,
}

/// GPU objects shared between frames; lives in egui's callback resources.
pub struct ViewportGpu {
    pbr: wgpu::RenderPipeline,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    globals_buf: wgpu::Buffer,
    globals_bind: wgpu::BindGroup,
    vbuf: Option<wgpu::Buffer>,
    ibuf: Option<wgpu::Buffer>,
    index_count: u32,
    offscreen: Option<Offscreen>,
    pub paint: PaintGpu,
}

impl ViewportGpu {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });

        let paint = PaintGpu::new(device, crate::paint::DEFAULT_TEXTURE_SIZE);
        let pbr_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/pbr.wgsl"));
        let pbr_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_pl"),
            bind_group_layouts: &[Some(&globals_layout), Some(&paint.material_layout)],
            immediate_size: 0,
        });
        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        let pbr = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr"),
            layout: Some(&pbr_pl),
            vertex: wgpu::VertexState {
                module: &pbr_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &pbr_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(COLOR_FORMAT.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/blit.wgsl"));
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pl),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pbr,
            blit,
            blit_layout,
            sampler,
            globals_buf,
            globals_bind,
            vbuf: None,
            ibuf: None,
            index_count: 0,
            offscreen: None,
            paint,
        }
    }

    pub fn set_mesh(&mut self, device: &wgpu::Device, mesh: &Mesh) {
        self.vbuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.ibuf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.index_count = mesh.indices.len() as u32;
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.offscreen.as_ref().is_some_and(|o| o.size == size) {
            return;
        }
        let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let blit_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.offscreen =
            Some(Offscreen { size, color, depth_view: depth.create_view(&Default::default()), blit_bind });
    }


    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, globals: &Globals, size: [u32; 2], doc: &Document, ops: &[PaintOp]) {
        self.ensure_offscreen(device, size);
        queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(globals));
        let off = self.offscreen.as_ref().unwrap();

        // Paint first, using last frame's depth (the camera is static while painting),
        // so this frame's scene pass already shows the new strokes.
        let mesh = match (&self.vbuf, &self.ibuf) {
            (Some(v), Some(i)) => Some(MeshBuffers { vertices: v, indices: i, index_count: self.index_count }),
            _ => None,
        };
        let depth = off.depth_view.clone();
        self.paint.apply(
            device,
            queue,
            encoder,
            doc,
            ops,
            mesh,
            &StampView {
                view_proj: globals.view_proj,
                camera_pos: globals.camera_pos,
                size_px: size,
                near_far: [globals.params[2], globals.params[3]],
                depth: &depth,
            },
        );

        let color_view = off.color.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.09, g: 0.09, b: 0.10, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &off.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let (Some(v), Some(i)) = (&self.vbuf, &self.ibuf) {
            pass.set_pipeline(&self.pbr);
            pass.set_bind_group(0, &self.globals_bind, &[]);
            pass.set_bind_group(1, self.paint.material_bind(), &[]);
            pass.set_vertex_buffer(0, v.slice(..));
            pass.set_index_buffer(i.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
    }
}

/// Per-frame paint callback handed to egui.
pub struct ViewportCallback {
    pub globals: Globals,
    pub size_px: [u32; 2],
    pub doc: Document,
    pub ops: Vec<PaintOp>,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(gpu) = resources.get_mut::<ViewportGpu>() {
            gpu.render(device, queue, encoder, &self.globals, self.size_px, &self.doc, &self.ops);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = resources.get::<ViewportGpu>() else { return };
        let Some(off) = &gpu.offscreen else { return };
        pass.set_pipeline(&gpu.blit);
        pass.set_bind_group(0, &off.blit_bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::paint::{Stamp, Target};
    use glam::Vec3;
    use op_core::{BlendMode, FillParams, Mask, NodeKind};

    const SIZE: u32 = 256;

    fn gpu_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Everything a test needs to drive the viewport headlessly.
    struct Rig {
        device: wgpu::Device,
        queue: wgpu::Queue,
        gpu: ViewportGpu,
        globals: Globals,
        front: Vec3,
    }

    impl Rig {
        fn new() -> Option<Self> {
            let (device, queue) = gpu_device()?;
            let mut gpu = ViewportGpu::new(&device, COLOR_FORMAT);
            gpu.set_mesh(&device, &Mesh::uv_sphere(1.0, 96, 48));
            let mut cam = OrbitCamera::default();
            cam.frame(Vec3::splat(-1.0), Vec3::splat(1.0));
            let (near, far) = cam.near_far();
            let cp = cam.position();
            let globals = Globals {
                view_proj: cam.view_proj(1.0).to_cols_array_2d(),
                camera_pos: [cp.x, cp.y, cp.z, 1.0],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                base_color: [0.18; 4],
                params: [0.0, 0.5, near, far],
            };
            Some(Self { device, queue, gpu, globals, front: cp.normalize() })
        }

        fn run(&mut self, doc: &Document, ops: &[PaintOp]) {
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.gpu.render(&self.device, &self.queue, &mut enc, &self.globals, [SIZE, SIZE], doc, ops);
            self.queue.submit([enc.finish()]);
        }

        /// Raw bytes of a SIZE x SIZE texture with `bpp` bytes per texel.
        fn read(&self, tex: &wgpu::Texture, bpp: u32) -> Vec<u8> {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (SIZE * SIZE * bpp) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(
                tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIZE * bpp), rows_per_image: Some(SIZE) },
                },
                wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            );
            self.queue.submit([enc.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let data = buf.slice(..).get_mapped_range().unwrap().to_vec();
            data
        }

        fn metal(&self) -> Vec<u8> {
            self.read(&self.gpu.paint.output.metal, 2)
        }
    }

    fn texel_rg(data: &[u8], uv: (f32, f32)) -> (u8, u8) {
        let x = ((uv.0 * SIZE as f32) as u32).min(SIZE - 1);
        let y = ((uv.1 * SIZE as f32) as u32).min(SIZE - 1);
        let i = ((y * SIZE + x) * 2) as usize;
        (data[i], data[i + 1])
    }

    /// Inverse of `Mesh::uv_sphere`'s parameterisation.
    fn sphere_uv(n: Vec3) -> (f32, f32) {
        let v = n.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
        let u = n.z.atan2(n.x).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
        (u, v)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let (sign, exp, frac) = ((h >> 15) & 1, (h >> 10) & 0x1f, (h & 0x3ff) as f32);
        let v = match exp {
            0 => frac / 1024.0 * 2f32.powi(-14),
            31 => f32::INFINITY,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e as i32 - 15),
        };
        if sign == 1 { -v } else { v }
    }

    fn dab(center: f32, radius: f32, metal_value: f32, enables: [f32; 4]) -> Stamp {
        Stamp {
            brush: [center, center, radius, 1.0],
            color: [1.0, 0.0, 0.0, 0.0],
            values: [metal_value, 0.5, 1.0, 0.0],
            enables,
        }
    }

    fn fill(base: Option<[f32; 3]>, metallic: Option<f32>, roughness: Option<f32>) -> NodeKind {
        NodeKind::Fill(FillParams { base, metallic, roughness })
    }

    #[test]
    fn projective_brush_paints_visible_side_only_and_undoes() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "paint", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]); // also renders the scene so depth exists

        let stamp = dab(SIZE as f32 / 2.0, 20.0, 1.0, [0.0, 1.0, 0.0, 0.0]); // metallic only
        rig.run(
            &doc,
            &[
                PaintOp::BeginStroke { layer, target: Target::Content },
                PaintOp::Stamps { layer, target: Target::Content, stamps: vec![stamp] },
            ],
        );

        let front = rig.front;
        let metal = rig.metal();
        let (value, cover) = texel_rg(&metal, sphere_uv(front));
        assert!(cover > 240 && value > 240, "front texel not painted: value={value} cover={cover}");
        assert_eq!(texel_rg(&metal, sphere_uv(-front)).1, 0, "back-facing texel must not be painted");
        let painted = metal.chunks_exact(2).filter(|t| t[1] > 0).count();
        assert!(painted > 0 && painted < (SIZE * SIZE) as usize / 10, "implausible painted area: {painted}");
        let rough = rig.read(&rig.gpu.paint.output.rough, 2);
        assert!(rough.iter().all(|&b| b == 0), "roughness written while disabled");

        rig.run(&doc, &[PaintOp::UndoStroke]);
        assert_eq!(texel_rg(&rig.metal(), sphere_uv(front)).1, 0, "undo did not clear the stroke");
        assert_eq!(rig.gpu.paint.undo_depth(), (0, 1));
        rig.run(&doc, &[PaintOp::RedoStroke]);
        assert!(texel_rg(&rig.metal(), sphere_uv(front)).1 > 240, "redo did not restore the stroke");
    }

    /// The GPU compositor must agree with the CPU reference in op-core, including
    /// blend modes, opacity, hidden layers and isolated groups.
    #[test]
    fn gpu_compositor_matches_cpu_reference() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        doc.add(None, None, "bg", fill(Some([0.9, 0.4, 0.2]), Some(0.8), Some(0.3)));
        let mul = doc.add(None, None, "mul", fill(Some([0.5, 0.7, 0.9]), Some(0.5), None));
        doc.find_mut(mul).unwrap().blend = BlendMode::Multiply;
        doc.find_mut(mul).unwrap().opacity = 0.5;
        let hidden = doc.add(None, None, "hidden", fill(Some([0.0; 3]), Some(0.0), Some(0.0)));
        doc.find_mut(hidden).unwrap().visible = false;
        let g = doc.add(None, None, "group", NodeKind::Group { children: vec![] });
        doc.find_mut(g).unwrap().blend = BlendMode::Screen;
        doc.find_mut(g).unwrap().opacity = 0.75;
        let a = doc.add(Some(g), None, "a", fill(Some([0.2, 0.2, 0.6]), None, Some(0.9)));
        let b = doc.add(Some(g), None, "b", fill(Some([0.9, 0.1, 0.1]), Some(0.25), None));
        doc.find_mut(a).unwrap().blend = BlendMode::Overlay;
        doc.find_mut(b).unwrap().blend = BlendMode::Difference;
        doc.find_mut(b).unwrap().opacity = 0.6;

        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        let expect = op_core::blend::eval_fill_only(&doc);

        let base = rig.read(&rig.gpu.paint.output.base, 8);
        let i = ((SIZE / 2 * SIZE + SIZE / 2) * 8) as usize;
        let got_base: Vec<f32> =
            (0..4).map(|c| f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]]))).collect();
        for c in 0..4 {
            assert!((got_base[c] - expect.base[c]).abs() < 2e-3, "base[{c}] gpu {} cpu {}", got_base[c], expect.base[c]);
        }
        let m = rig.metal();
        let r = rig.read(&rig.gpu.paint.output.rough, 2);
        let j = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
        for (name, bytes, want) in [("metal", &m, expect.metal), ("rough", &r, expect.rough)] {
            for c in 0..2 {
                let got = bytes[j + c] as f32 / 255.0;
                assert!((got - want[c]).abs() < 3.0 / 255.0, "{name}[{c}] gpu {got} cpu {}", want[c]);
            }
        }
    }

    /// Every blend mode over a non-empty backdrop (over an empty one all modes
    /// degenerate to Normal, which would hide formula errors).
    #[test]
    fn every_blend_mode_matches_cpu_reference() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        for mode in BlendMode::ALL {
            let mut doc = Document::default();
            // Backdrop channels straddle 0.5 so both Overlay branches are exercised.
            doc.add(None, None, "bg", fill(Some([0.9, 0.4, 0.2]), Some(0.8), Some(0.3)));
            let top = doc.add(None, None, "top", fill(Some([0.5, 0.7, 0.9]), Some(0.4), Some(0.6)));
            doc.find_mut(top).unwrap().blend = mode;
            doc.find_mut(top).unwrap().opacity = 0.7;

            rig.run(&doc, &[PaintOp::Reset(SIZE)]);
            let expect = op_core::blend::eval_fill_only(&doc);
            let base = rig.read(&rig.gpu.paint.output.base, 8);
            let i = ((SIZE / 2 * SIZE + SIZE / 2) * 8) as usize;
            for c in 0..3 {
                let got = f16_to_f32(u16::from_le_bytes([base[i + c * 2], base[i + c * 2 + 1]]));
                assert!((got - expect.base[c]).abs() < 2e-3, "{mode:?} base[{c}] gpu {got} cpu {}", expect.base[c]);
            }
            let (m, r) = (rig.metal(), rig.read(&rig.gpu.paint.output.rough, 2));
            let j = ((SIZE / 2 * SIZE + SIZE / 2) * 2) as usize;
            let (gm, gr) = (m[j] as f32 / 255.0, r[j] as f32 / 255.0);
            assert!((gm - expect.metal[0]).abs() < 3.0 / 255.0, "{mode:?} metal gpu {gm} cpu {}", expect.metal[0]);
            assert!((gr - expect.rough[0]).abs() < 3.0 / 255.0, "{mode:?} rough gpu {gr} cpu {}", expect.rough[0]);
        }
    }

    /// A black mask hides a layer; painting white into it reveals only where stamped.
    #[test]
    fn painted_mask_reveals_layer_locally() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let layer = doc.add(None, None, "metal", fill(None, Some(1.0), None));
        doc.find_mut(layer).unwrap().mask = Some(Mask { enabled: true, init_white: false });
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(rig.metal().iter().all(|&b| b == 0), "black mask should hide the layer entirely");

        let reveal = Stamp {
            brush: [SIZE as f32 / 2.0, SIZE as f32 / 2.0, 20.0, 1.0],
            color: [0.0; 4],
            values: [1.0, 0.0, 1.0, 0.0],
            enables: [0.0; 4],
        };
        rig.run(
            &doc,
            &[
                PaintOp::BeginStroke { layer, target: Target::Mask },
                PaintOp::Stamps { layer, target: Target::Mask, stamps: vec![reveal] },
            ],
        );
        let front = rig.front;
        let metal = rig.metal();
        let (v, c) = texel_rg(&metal, sphere_uv(front));
        assert!(c > 240 && v > 240, "revealed texel: value={v} cover={c}");
        assert_eq!(texel_rg(&metal, sphere_uv(-front)).1, 0, "unpainted mask area must stay hidden");

        // Disabling the mask shows the whole layer; undo of the stroke re-hides the dab.
        doc.find_mut(layer).unwrap().mask.as_mut().unwrap().enabled = false;
        rig.run(&doc, &[]);
        assert!(texel_rg(&rig.metal(), sphere_uv(-front)).1 > 240, "disabled mask should not hide anything");
        doc.find_mut(layer).unwrap().mask.as_mut().unwrap().enabled = true;
        rig.run(&doc, &[PaintOp::UndoStroke]);
        assert_eq!(texel_rg(&rig.metal(), sphere_uv(front)).1, 0, "undo should re-hide the dab");
    }

    /// Textures of deleted layers survive while history can still restore them,
    /// and are freed once nothing references them.
    #[test]
    fn gc_keeps_referenced_nodes_and_frees_the_rest() {
        let Some(mut rig) = Rig::new() else { return eprintln!("no GPU adapter; skipping") };
        let mut doc = Document::default();
        let p = doc.add(None, None, "p", NodeKind::Paint);
        rig.run(&doc, &[PaintOp::Reset(SIZE)]);
        assert!(rig.gpu.paint.store.contains_key(&p));

        let mut gone = doc.clone();
        gone.remove(p);
        rig.run(&gone, &[PaintOp::Gc(vec![p])]); // history still references it
        assert!(rig.gpu.paint.store.contains_key(&p), "referenced node was freed");
        rig.run(&gone, &[PaintOp::Gc(vec![])]);
        assert!(!rig.gpu.paint.store.contains_key(&p), "unreferenced node was kept");
    }
}
