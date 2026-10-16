//! The 2D UV view's GPU side: a pan/zoom view of one texture set's textures with a tile grid and a
//! wireframe of its UVs, drawn inside egui's render pass like the 3D viewport's blit.

use crate::viewport::{SetBatch, ViewportGpu};
use eframe::egui_wgpu::{self, wgpu};
use wgpu::util::DeviceExt;

/// What the 2D view shows of a set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Base,
    Metallic,
    Roughness,
    Normal,
    Emissive,
}

impl Channel {
    pub const ALL: [Channel; 5] = [Channel::Base, Channel::Metallic, Channel::Roughness, Channel::Normal, Channel::Emissive];

    pub fn label(self) -> &'static str {
        match self {
            Channel::Base => "Base color",
            Channel::Metallic => "Metallic",
            Channel::Roughness => "Roughness",
            Channel::Normal => "Normal map",
            Channel::Emissive => "Emissive",
        }
    }

    fn code(self) -> f32 {
        match self {
            Channel::Base => 0.0,
            Channel::Metallic => 1.0,
            Channel::Roughness => 2.0,
            Channel::Normal => 3.0,
            Channel::Emissive => 4.0,
        }
    }
}

/// Where the camera is in UV space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvCamera {
    pub center: [f32; 2],
    /// Screen points per UV unit.
    pub scale: f32,
}

impl UvCamera {
    /// Frame the unit square in a view of `size` points with a small margin.
    pub fn fit(size: [f32; 2]) -> Self {
        Self { center: [0.5, 0.5], scale: (size[0].min(size[1]) * 0.9).max(16.0) }
    }

    /// UV coordinates under a point `from_center` points away from the view's centre.
    pub fn uv_at(&self, from_center: [f32; 2]) -> [f32; 2] {
        [self.center[0] + from_center[0] / self.scale, self.center[1] + from_center[1] / self.scale]
    }

    /// Zoom by `factor` keeping the UV under `from_center` where it is on screen.
    pub fn zoom_at(&mut self, from_center: [f32; 2], factor: f32) {
        let before = self.uv_at(from_center);
        self.scale = (self.scale * factor).clamp(8.0, 200_000.0);
        let after = self.uv_at(from_center);
        self.center = [self.center[0] + before[0] - after[0], self.center[1] + before[1] - after[1]];
    }

    /// Drag the view by a screen delta in points.
    pub fn pan(&mut self, delta: [f32; 2]) {
        self.center = [self.center[0] - delta[0] / self.scale, self.center[1] - delta[1] / self.scale];
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    view: [f32; 4],
    rect: [f32; 4],
    wire: [f32; 4],
}

struct Lines {
    set: usize,
    /// Which mesh upload this was built from.
    generation: u64,
    indices: wgpu::Buffer,
    count: u32,
}

pub struct UvGpu {
    bg_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    params_bind: wgpu::BindGroup,
    lines: Option<Lines>,
    /// Bumped whenever the mesh is replaced, so cached line buffers are rebuilt.
    pub generation: u64,
}

impl UvGpu {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, material_layout: &wgpu::BindGroupLayout) -> Self {
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uv_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uv_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uv_params_bind"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/uv2d.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uv_pl"),
            bind_group_layouts: &[Some(&params_layout), Some(material_layout)],
            immediate_size: 0,
        });
        let pl_params = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uv_pl_params"),
            bind_group_layouts: &[Some(&params_layout)],
            immediate_size: 0,
        });
        let make = |label: &str, layout: &wgpu::PipelineLayout, vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: target_format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let bg_pipeline = make("uv_bg", &pl_params, "vs_full", "fs_bg", None);
        let tile_pipeline = make("uv_tile", &pl, "vs_tile", "fs_tile", None);
        let grid_pipeline = make("uv_grid", &pl_params, "vs_full", "fs_grid", Some(wgpu::BlendState::ALPHA_BLENDING));
        // The wireframe reads only the mesh's UVs (location 2 of the shared vertex layout).
        let attrs = wgpu::vertex_attr_array![2 => Float32x2];
        let wire_attrs = [wgpu::VertexAttribute { offset: 24, ..attrs[0] }];
        let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("uv_wire"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_wire"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<op_mesh::Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wire_attrs,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_wire"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { bg_pipeline, tile_pipeline, grid_pipeline, wire_pipeline, params, params_bind, lines: None, generation: 0 }
    }

    /// Make sure the edge list of `set` exists. `tris` are the set's triangle indices.
    pub fn ensure_lines(&mut self, device: &wgpu::Device, set: usize, tris: &[u32]) {
        if self.lines.as_ref().is_some_and(|l| l.set == set && l.generation == self.generation) {
            return;
        }
        let mut edges = Vec::with_capacity(tris.len() * 2);
        for t in tris.chunks_exact(3) {
            edges.extend_from_slice(&[t[0], t[1], t[1], t[2], t[2], t[0]]);
        }
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uv_wire_indices"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.lines = Some(Lines { set, generation: self.generation, indices, count: edges.len() as u32 });
    }
}

/// One frame of the 2D view.
pub struct UvCallback {
    pub set: usize,
    pub batches: Vec<SetBatch>,
    pub camera: UvCamera,
    pub channel: Channel,
    /// View size in points; the callback scales it by `ppp` to pixels.
    pub size_pts: [f32; 2],
    pub ppp: f32,
    pub wire: Option<[f32; 4]>,
}

impl egui_wgpu::CallbackTrait for UvCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = resources.get_mut::<ViewportGpu>() else { return Vec::new() };
        // Paint operations are applied here too, so the 2D view works while the 3D one is hidden.
        gpu.apply_batches(device, queue, encoder, &self.batches);
        if self.wire.is_some() {
            gpu.prepare_uv_lines(device, self.set);
        }
        let size_px = [self.size_pts[0] * self.ppp, self.size_pts[1] * self.ppp];
        let params = Params {
            view: [self.camera.center[0], self.camera.center[1], self.camera.scale * self.ppp, self.channel.code()],
            rect: [size_px[0], size_px[1], self.wire.is_some() as u32 as f32, 0.0],
            wire: self.wire.unwrap_or([0.0; 4]),
        };
        queue.write_buffer(gpu.uv_params(), 0, bytemuck::bytes_of(&params));
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(gpu) = resources.get::<ViewportGpu>() else { return };
        gpu.paint_uv(pass, self.set, self.wire.is_some());
    }
}

impl ViewportGpu {
    fn uv_params(&self) -> &wgpu::Buffer {
        &self.uv.params
    }

    fn prepare_uv_lines(&mut self, device: &wgpu::Device, set: usize) {
        let Some(&(first, count)) = self.ranges.get(set) else { return };
        let end = ((first + count) as usize).min(self.cpu_indices.len());
        let tris = &self.cpu_indices[(first as usize).min(end)..end];
        self.uv.ensure_lines(device, set, tris);
    }

    fn paint_uv(&self, pass: &mut wgpu::RenderPass<'static>, set: usize, wire: bool) {
        let Some(unit) = self.sets.get(set) else { return };
        pass.set_bind_group(0, &self.uv.params_bind, &[]);
        pass.set_pipeline(&self.uv.bg_pipeline);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.uv.tile_pipeline);
        for paint in &unit.tiles {
            pass.set_bind_group(1, paint.material_bind(), &[]);
            pass.draw(0..6, 0..1);
        }
        pass.set_pipeline(&self.uv.grid_pipeline);
        pass.draw(0..3, 0..1);
        if let (true, Some(lines), Some(v)) = (wire, self.uv.lines.as_ref().filter(|l| l.set == set), &self.vbuf) {
            pass.set_pipeline(&self.uv.wire_pipeline);
            pass.set_vertex_buffer(0, v.slice(..));
            pass.set_index_buffer(lines.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..lines.count, 0, 0..1);
        }
    }

    /// Apply paint operations without drawing the 3D scene (no 3D brush strokes, no depth buffer).
    pub fn apply_batches(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, batches: &[SetBatch]) {
        let queued = std::mem::take(&mut self.queued);
        let depth = self.dummy_depth.create_view(&Default::default());
        let uv = self.dummy_uv.create_view(&Default::default());
        let view = crate::paint::StampView {
            view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0; 4],
            size_px: [1, 1],
            near_far: [0.1, 10.0],
            depth: &depth,
            uv: &uv,
        };
        for b in queued.iter().chain(batches) {
            crate::viewport::apply_batch(&mut self.sets, None, device, queue, encoder, b, &view);
        }
    }
}

/// The UV bounds (min u, min v, max u, max v) of a set's triangles.
pub fn uv_extent(mesh: &op_mesh::Mesh, set: usize) -> Option<[f32; 4]> {
    let ranges = mesh.set_ranges();
    let r = ranges.get(set)?;
    let tris = mesh.indices.get(r.first_index as usize..(r.first_index + r.index_count) as usize)?;
    let mut e = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for &i in tris {
        let uv = mesh.vertices.get(i as usize)?.uv;
        if !(uv[0].is_finite() && uv[1].is_finite()) {
            continue;
        }
        e = [e[0].min(uv[0]), e[1].min(uv[1]), e[2].max(uv[0]), e[3].max(uv[1])];
    }
    (e[0] <= e[2]).then_some(e)
}

/// The UDIM number of the tile containing (u, v): 1001 at the origin, +1 per tile in u, +10 per tile in v.
pub fn udim_tile(u: f32, v: f32) -> u32 {
    let (tu, tv) = (u.floor().max(0.0) as u32, v.floor().max(0.0) as u32);
    1001 + tu.min(9) + 10 * tv
}

/// A short description of which tiles a set's UVs occupy and whether they all fit the painted tile.
pub fn extent_summary(e: [f32; 4]) -> (String, bool) {
    // A coordinate exactly on a tile edge belongs to the tile it closes.
    let last = |x: f32| (x - 1e-4).floor().max(0.0);
    let (lo, hi) = (udim_tile(e[0], e[1]), udim_tile(last(e[2]), last(e[3])));
    let inside = e[0] >= -1e-4 && e[1] >= -1e-4 && e[2] <= 1.0 + 1e-4 && e[3] <= 1.0 + 1e-4;
    if inside {
        ("UVs fit tile 1001".to_string(), true)
    } else if lo == hi {
        (format!("UVs are in tile {lo}, outside 1001"), false)
    } else {
        (format!("UVs span tiles {lo} to {hi}"), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zooming_keeps_the_point_under_the_cursor_fixed() {
        let mut cam = UvCamera::fit([800.0, 600.0]);
        let cursor = [120.0, -80.0];
        let before = cam.uv_at(cursor);
        cam.zoom_at(cursor, 3.0);
        let after = cam.uv_at(cursor);
        assert!((before[0] - after[0]).abs() < 1e-5 && (before[1] - after[1]).abs() < 1e-5);
        assert!(cam.scale > UvCamera::fit([800.0, 600.0]).scale * 2.9);
        // Zoom is bounded.
        for _ in 0..100 {
            cam.zoom_at(cursor, 2.0);
        }
        assert!(cam.scale <= 200_000.0);
        for _ in 0..100 {
            cam.zoom_at(cursor, 0.5);
        }
        assert!(cam.scale >= 8.0);
    }

    #[test]
    fn panning_moves_the_content_with_the_pointer() {
        let mut cam = UvCamera { center: [0.5, 0.5], scale: 100.0 };
        // Dragging right by 50 points moves the picture right, so the centre moves left in UV.
        cam.pan([50.0, -25.0]);
        assert!((cam.center[0] - 0.0).abs() < 1e-6 && (cam.center[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn udim_numbers_follow_the_standard() {
        assert_eq!(udim_tile(0.5, 0.5), 1001);
        assert_eq!(udim_tile(1.2, 0.1), 1002);
        assert_eq!(udim_tile(0.9, 1.5), 1011);
        assert_eq!(udim_tile(3.99, 2.01), 1024);
        assert_eq!(udim_tile(-0.5, -0.5), 1001, "negative space has no tile; it clamps");
    }

    #[test]
    fn extents_are_summarised_by_tile() {
        assert_eq!(extent_summary([0.1, 0.1, 1.0, 1.0]), ("UVs fit tile 1001".to_string(), true));
        assert_eq!(extent_summary([0.0, 0.0, 0.5, 0.5]).1, true);
        assert_eq!(extent_summary([0.0, 0.0, 2.0, 1.0]), ("UVs span tiles 1001 to 1002".to_string(), false));
        assert_eq!(extent_summary([1.1, 0.1, 1.9, 0.9]), ("UVs are in tile 1002, outside 1001".to_string(), false));
        assert_eq!(extent_summary([0.0, 0.0, 1.5, 2.5]).0, "UVs span tiles 1001 to 1022");
    }

    #[test]
    fn a_sets_extent_ignores_other_sets_and_bad_uvs() {
        let mut m = op_mesh::Mesh::uv_sphere(1.0, 8, 4);
        for (i, v) in m.vertices.iter_mut().enumerate() {
            v.uv = [(i % 3) as f32 * 0.4, 0.2];
        }
        let half = (m.indices.len() / 6 * 3) as u32;
        let total = m.indices.len() as u32;
        m.sets = vec![
            op_mesh::SetRange { name: "a".into(), first_index: 0, index_count: half },
            op_mesh::SetRange { name: "b".into(), first_index: half, index_count: total - half },
        ];
        let e = uv_extent(&m, 0).unwrap();
        assert!(e[0] >= 0.0 && e[2] <= 0.8 + 1e-6 && (e[1] - 0.2).abs() < 1e-6);
        assert!(uv_extent(&m, 5).is_none());
        m.vertices[0].uv = [f32::NAN, 0.0];
        assert!(uv_extent(&m, 0).is_some(), "a NaN texel is skipped, not fatal");
    }

    #[test]
    fn fit_frames_the_whole_unit_square() {
        let cam = UvCamera::fit([400.0, 900.0]);
        assert_eq!(cam.center, [0.5, 0.5]);
        assert!(cam.scale <= 400.0 && cam.scale > 300.0);
        let edge = cam.uv_at([-cam.scale * 0.5, 0.0]);
        assert!(edge[0].abs() < 1e-6, "the left edge of the square sits at u = 0: {edge:?}");
    }
}
