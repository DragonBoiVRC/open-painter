//! Render preview: a CPU path-traced picture of the model with its finished textures, refining in the
//! background (see `op_bake::render`). It shows the first texture set's tile of each material.

use crate::viewport::ViewportGpu;
use crate::App;
use eframe::egui;
use op_bake::render::{render_pass, to_rgba8, Scene, SetTexture, View};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;
const MAX_SAMPLES: u32 = 256;
/// Textures are read at most this big: the preview needs no more.
const MAX_TEXTURE: u32 = 1024;

pub struct RenderUi {
    rx: Receiver<(u32, Vec<u8>)>,
    cancel: Arc<AtomicBool>,
    texture: Option<egui::TextureHandle>,
    samples: u32,
    rgba: Vec<u8>,
}

impl Drop for RenderUi {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Colour, metallic and roughness of a set's composited output as linear floats, shrunk to `MAX_TEXTURE`.
fn set_texture(planes: &[Vec<u8>; 7], size: u32) -> SetTexture {
    let step = (size / MAX_TEXTURE).max(1);
    let out = size / step;
    let (mut base, mut metal, mut rough) = (Vec::new(), Vec::new(), Vec::new());
    for y in 0..out {
        for x in 0..out {
            let i = ((y * step * size + x * step) as usize) * 8;
            let f = |k: usize| half::f16::from_bits(u16::from_le_bytes([planes[0][i + k * 2], planes[0][i + k * 2 + 1]])).to_f32();
            let alpha = f(3);
            // Premultiplied colour back to colour; bare surface reads mid grey.
            if alpha > 1e-3 {
                base.extend([f(0) / alpha, f(1) / alpha, f(2) / alpha]);
            } else {
                base.extend([0.5; 3]);
            }
            let j = ((y * step * size + x * step) as usize) * 2;
            let value = |plane: &[u8], default: f32| if plane[j + 1] > 0 { plane[j] as f32 / plane[j + 1] as f32 } else { default };
            metal.push(value(&planes[1], 0.0));
            rough.push(value(&planes[2], 0.5));
        }
    }
    SetTexture { size: out, base, metal, rough }
}

impl App {
    fn render_textures(&mut self) -> Vec<SetTexture> {
        let mut textures = Vec::new();
        for i in 0..self.set_ranges_len() {
            let was_open = i == self.active || self.parked.get(i).and_then(Option::as_ref).is_some_and(|s| s.materialized);
            self.ensure_materialized(i);
            {
                let renderer = self.render_state.renderer.read();
                let unit = renderer.callback_resources.get::<ViewportGpu>().and_then(|g| g.sets.get(i)).and_then(|u| u.tiles.first());
                textures.push(match unit {
                    Some(paint) => set_texture(&paint.capture_output(&self.render_state.device, &self.render_state.queue), paint.size()),
                    None => SetTexture::flat([0.5; 3], 0.0, 0.5),
                });
            }
            if !was_open {
                self.release_if_untouched(i);
            }
        }
        textures
    }

    /// Render `samples` passes right now and return the RGBA8 picture (for scripts; the app waits).
    pub(crate) fn render_blocking(&mut self, samples: u32) -> Result<Vec<u8>, String> {
        let textures = self.render_textures();
        let scene = Scene::new(&self.mesh, textures);
        let view = View::new(self.camera.view_proj(WIDTH as f32 / HEIGHT as f32), WIDTH, HEIGHT);
        let mut accum = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
        for pass in 0..samples {
            render_pass(&scene, &view, pass, &mut accum);
        }
        Ok(to_rgba8(&accum, samples))
    }

    pub(crate) fn start_render(&mut self) {
        let textures = self.render_textures();
        let (mesh, view_proj) = (self.mesh.clone(), self.camera.view_proj(WIDTH as f32 / HEIGHT as f32));
        let cancel = Arc::new(AtomicBool::new(false));
        let (flag, (tx, rx)) = (cancel.clone(), mpsc::channel());
        std::thread::spawn(move || {
            let scene = Scene::new(&mesh, textures);
            let view = View::new(view_proj, WIDTH, HEIGHT);
            let mut accum = vec![0.0f32; (WIDTH * HEIGHT * 3) as usize];
            for pass in 0..MAX_SAMPLES {
                if flag.load(Ordering::Relaxed) {
                    return;
                }
                render_pass(&scene, &view, pass, &mut accum);
                let n = pass + 1;
                // Show early passes often, later ones less so.
                if (n <= 8 || n % 4 == 0 || n == MAX_SAMPLES) && tx.send((n, to_rgba8(&accum, n))).is_err() {
                    return;
                }
            }
        });
        self.render_ui = Some(RenderUi { rx, cancel, texture: None, samples: 0, rgba: Vec::new() });
        self.status = "Rendering a preview with the finished textures...".into();
    }

    pub(crate) fn render_window(&mut self, ctx: &egui::Context) {
        let Some(r) = &mut self.render_ui else { return };
        while let Ok((n, rgba)) = r.rx.try_recv() {
            r.samples = n;
            r.texture = Some(ctx.load_texture("render_preview", egui::ColorImage::from_rgba_unmultiplied([WIDTH as usize, HEIGHT as usize], &rgba), egui::TextureOptions::LINEAR));
            r.rgba = rgba;
        }
        let (mut open, mut save) = (true, false);
        let running = r.samples < MAX_SAMPLES;
        egui::Window::new("Render preview").open(&mut open).resizable(false).show(ctx, |ui| {
            match &r.texture {
                Some(t) => {
                    ui.image((t.id(), egui::vec2(WIDTH as f32, HEIGHT as f32)));
                }
                None => {
                    ui.label("Reading the textures and building the scene...");
                }
            }
            ui.horizontal(|ui| {
                ui.label(format!("{} / {MAX_SAMPLES} samples", r.samples));
                if running && ui.button("Stop").clicked() {
                    r.cancel.store(true, Ordering::Relaxed);
                }
                if ui.add_enabled(!r.rgba.is_empty(), egui::Button::new("Save PNG")).clicked() {
                    save = true;
                }
            });
            ui.label(egui::RichText::new("A CPU path trace from the current camera: sky and sun light, soft shadows, diffuse and glossy bounces. Not exact, but it shows how the textures read.").size(11.0).color(crate::ui::theme::text_dim()));
        });
        if running {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if save {
            let rgba = std::mem::take(&mut self.render_ui.as_mut().expect("window is open").rgba);
            self.save_render_png(&rgba);
            if let Some(r) = &mut self.render_ui {
                r.rgba = rgba;
            }
        }
        if !open {
            self.render_ui = None;
        }
    }

    fn save_render_png(&mut self, rgba: &[u8]) {
        let dir = self.default_export_dir();
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let path = dir.join(format!("{}_render_{stamp}.png", self.project_name()));
        let result = op_image::encode_png(WIDTH, HEIGHT, rgba).map_err(|e| e.to_string()).and_then(|png| std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, png)).map_err(|e| e.to_string()));
        self.status = match result {
            Ok(()) => format!("Saved the render to {}", path.display()),
            Err(e) => format!("Could not save the render: {e}"),
        };
    }
}
