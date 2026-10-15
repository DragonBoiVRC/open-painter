//! Small previews of layer textures for the layers panel: a GPU box-filter downscale to 32x32,
//! read back to the CPU.

use crate::gpu_io::read_texture;
use eframe::egui_wgpu::wgpu;

pub const THUMB_SIZE: u32 = 32;

pub struct Thumbnailer {
    base: wgpu::RenderPipeline,
    mask: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    target: wgpu::Texture,
}

impl Thumbnailer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/thumb.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thumb_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thumb_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("thumb"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumb_target"),
            size: wgpu::Extent3d { width: THUMB_SIZE, height: THUMB_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self { base: make("fs_base"), mask: make("fs_mask"), layout, target }
    }

    /// Downscale `src` (an RGBA16F premultiplied layer, or an R8 mask when `mask`) to
    /// `THUMB_SIZE` squared RGBA8 bytes (premultiplied sRGB for layers, gray for masks).
    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Texture, mask: bool) -> Vec<u8> {
        let view = src.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thumb_bind"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }],
        });
        let target = self.target.create_view(&Default::default());
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("thumb_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(if mask { &self.mask } else { &self.base });
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([enc.finish()]);
        read_texture(device, queue, &self.target, THUMB_SIZE, 4)
    }
}

use crate::App;
use op_core::{Node, NodeKind};
use std::time::{Duration, Instant};

/// Refresh at most this often, and only once painting has settled.
const REFRESH_EVERY: Duration = Duration::from_millis(350);

fn wanted(nodes: &[Node], out: &mut Vec<(u64, bool)>) {
    for n in nodes {
        if matches!(n.kind, NodeKind::Paint) {
            out.push((n.id, false));
        }
        if n.mask.is_some() {
            out.push((n.id, true));
        }
        if let NodeKind::Group { children } = &n.kind {
            wanted(children, out);
        }
    }
}

impl App {
    /// Note that layer pixels or the layer list changed, so previews need redrawing.
    pub(crate) fn thumbs_dirty(&mut self) {
        self.paint_rev += 1;
    }

    /// Redraw the previews of the active set if anything changed and painting has paused.
    pub(crate) fn refresh_thumbs(&mut self, ctx: &egui::Context) {
        if self.thumb_rev == self.paint_rev || self.stroke.active || self.thumb_at.elapsed() < REFRESH_EVERY {
            return;
        }
        // Wait until everything queued for the GPU (a project being loaded, say) has been applied.
        let gpu_busy = !self.pending.is_empty()
            || !self.pending_tiles.is_empty()
            || self.render_state.renderer.read().callback_resources.get::<crate::viewport::ViewportGpu>().is_none_or(|g| !g.queued.is_empty());
        if gpu_busy {
            ctx.request_repaint_after(Duration::from_millis(100));
            return;
        }
        self.thumb_rev = self.paint_rev;
        self.thumb_at = Instant::now();
        let mut want = Vec::new();
        wanted(&self.doc.nodes, &mut want);
        let images = {
            let renderer = self.render_state.renderer.read();
            let Some(gpu) = renderer.callback_resources.get::<crate::viewport::ViewportGpu>() else { return };
            gpu.thumbnails(&self.render_state.device, &self.render_state.queue, self.active, &want)
        };
        self.thumbs.retain(|k, _| want.contains(k));
        for ((node, mask), bytes) in images {
            let image = egui::ColorImage::from_rgba_premultiplied([THUMB_SIZE as usize; 2], &bytes);
            match self.thumbs.get_mut(&(node, mask)) {
                Some(h) => h.set(image, egui::TextureOptions::LINEAR),
                None => {
                    let h = ctx.load_texture(format!("thumb-{node}-{mask}"), image, egui::TextureOptions::LINEAR);
                    self.thumbs.insert((node, mask), h);
                }
            }
        }
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::{make_mask_texture, Layer};

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn px(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * THUMB_SIZE + x) * 4) as usize;
        [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
    }

    #[test]
    fn an_empty_layer_is_transparent_and_a_half_filled_mask_is_half_white() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter; skipping") };
        let t = Thumbnailer::new(&device);
        let layer = Layer::new(&device, 64);
        let img = t.render(&device, &queue, &layer.base, false);
        assert!(img.chunks(4).all(|p| p[3] == 0), "nothing painted: fully transparent");

        // A mask whose left half is white: write bytes directly.
        let mask = make_mask_texture(&device, 64);
        let bytes: Vec<u8> = (0..64 * 64).map(|i| if i % 64 < 32 { 255 } else { 0 }).collect();
        let mut enc = device.create_command_encoder(&Default::default());
        crate::gpu_io::write_texture(&device, &mut enc, &mask, 64, 1, &bytes);
        queue.submit([enc.finish()]);
        let img = t.render(&device, &queue, &mask, true);
        assert_eq!(px(&img, 2, 10), [255, 255, 255, 255]);
        assert_eq!(px(&img, 29, 10), [0, 0, 0, 255]);
    }
}
