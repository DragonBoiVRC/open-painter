//! The 2D UV view tab: the active texture set in UV space, painted on directly.

use super::theme;
use crate::uvview::{self, Channel, UvCallback, UvCamera};
use crate::{App, PaintOp};
use eframe::egui_wgpu;
use egui::{Color32, Pos2, Rect, Sense, Stroke, UiBuilder};
use egui_phosphor::regular as ph;

impl App {
    pub(crate) fn uv_view_ui(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let ppp = ui.pixels_per_point();
        let (mods, primary_down) = ui.input(|i| (i.modifiers, i.pointer.primary_down()));
        let size = [rect.width(), rect.height()];
        let mut cam = self.uv_cam.unwrap_or_else(|| UvCamera::fit(size));
        let from_center = |p: Pos2| [p.x - rect.center().x, p.y - rect.center().y];

        // Navigation: middle drag (or Alt + left drag) pans, scroll zooms around the cursor.
        let d = resp.drag_delta();
        if resp.dragged_by(egui::PointerButton::Middle) || (resp.dragged_by(egui::PointerButton::Primary) && mods.alt) {
            cam.pan([d.x, d.y]);
        }
        if let (true, Some(pos)) = (resp.hovered(), resp.hover_pos()) {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                cam.zoom_at(from_center(pos), (scroll * 0.0015).exp());
            }
            if ui.input(|i| i.key_pressed(egui::Key::F)) && !ui.ctx().egui_wants_keyboard_input() {
                cam = UvCamera::fit(size);
            }
        }
        self.uv_cam = Some(cam);

        // Painting: the brush works in UV units, so its on-screen size follows the zoom.
        let mut stamps = Vec::new();
        let mut stamp_target = None;
        let modal_open = self.dialog.is_some() || self.confirm.is_some();
        let copies = matches!(self.tool, crate::Tool::Smudge | crate::Tool::Clone | crate::Tool::Project | crate::Tool::Fill | crate::Tool::Hide);
        if copies && resp.is_pointer_button_down_on() && primary_down {
            self.status = "Smudge, clone, projection and fill work in the 3D view".into();
        }
        let painting = resp.is_pointer_button_down_on() && primary_down && !mods.alt && !mods.command && !modal_open && !copies;
        match (painting, resp.interact_pointer_pos().or(resp.hover_pos())) {
            (true, Some(pos)) => match self.paint_target() {
                Ok((layer, target)) => {
                    let uv = cam.uv_at(from_center(pos));
                    let radius = self.brush.radius_pts / cam.scale;
                    self.stroke_to_uv(uv, radius, layer, target, &mut stamps);
                    stamp_target = Some((layer, target));
                    ui.ctx().request_repaint();
                }
                Err(msg) => {
                    self.status = msg.to_string();
                    self.stroke.active = false;
                }
            },
            _ => self.stroke.active = false,
        }
        let mut ops = Vec::new();
        if let (false, Some((layer, target))) = (stamps.is_empty(), stamp_target) {
            if self.tool == crate::Tool::Erase && target == crate::Target::Content {
                ops.push(PaintOp::Erase { layer, stamps, uv: true });
            } else {
                ops.push(PaintOp::Stamps2d { layer, target, stamps });
            }
        }
        let batches = self.take_batches(ops);
        if batches.iter().any(|b| !b.ops.is_empty()) {
            self.thumbs_dirty();
        }
        let wire = self.uv_wire.then_some([0.35, 0.75, 1.0, 0.55]);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            UvCallback { set: self.active, batches, camera: cam, channel: self.uv_channel, size_pts: size, ppp, wire },
        ));

        let painter = ui.painter().with_clip_rect(rect);
        self.uv_tile_labels(&painter, rect, &cam);
        if let (false, Some(pos)) = (mods.alt || modal_open, resp.hover_pos()) {
            let r = self.brush.radius_pts;
            painter.circle_stroke(pos, r, Stroke::new(3.0, Color32::from_black_alpha(110)));
            painter.circle_stroke(pos, r, Stroke::new(1.0, Color32::WHITE));
        }
        self.uv_overlay(ui, rect);
    }

    /// UDIM tile numbers in the corner of each visible tile.
    fn uv_tile_labels(&self, painter: &egui::Painter, rect: Rect, cam: &UvCamera) {
        let to_screen = |u: f32, v: f32| Pos2::new(rect.center().x + (u - cam.center[0]) * cam.scale, rect.center().y + (v - cam.center[1]) * cam.scale);
        let tl = cam.uv_at([rect.left() - rect.center().x, rect.top() - rect.center().y]);
        let br = cam.uv_at([rect.right() - rect.center().x, rect.bottom() - rect.center().y]);
        let (u0, u1) = (tl[0].floor().max(0.0) as i32, br[0].floor().min(9.0) as i32);
        let (v0, v1) = (tl[1].floor().max(0.0) as i32, br[1].floor().min(9.0) as i32);
        for tv in v0..=v1.min(v0 + 6) {
            for tu in u0..=u1.min(u0 + 6) {
                let at = to_screen(tu as f32, tv as f32) + egui::vec2(6.0, 4.0);
                if rect.contains(at) {
                    // A dark outline keeps the number readable over any texture.
                    let text = uvview::udim_tile(tu as f32, tv as f32).to_string();
                    let font = egui::FontId::proportional(11.0);
                    for o in [egui::vec2(-1.0, 0.0), egui::vec2(1.0, 0.0), egui::vec2(0.0, -1.0), egui::vec2(0.0, 1.0)] {
                        painter.text(at + o, egui::Align2::LEFT_TOP, &text, font.clone(), Color32::from_black_alpha(220));
                    }
                    painter.text(at, egui::Align2::LEFT_TOP, text, font, Color32::from_rgb(255, 170, 70));
                }
            }
        }
    }

    /// The channel picker, wireframe toggle and UV extent readout.
    fn uv_overlay(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let bar = Rect::from_min_size(rect.min + egui::vec2(8.0, 8.0), egui::vec2((rect.width() - 16.0).max(0.0), 26.0));
        ui.scope_builder(UiBuilder::new().max_rect(bar), |ui| {
            egui::Frame::NONE.fill(Color32::from_black_alpha(150)).corner_radius(4.0).inner_margin(egui::Margin::symmetric(4, 2)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("uv_channel").selected_text(self.uv_channel.label()).width(120.0).show_ui(ui, |ui| {
                        for c in Channel::ALL {
                            ui.selectable_value(&mut self.uv_channel, c, c.label());
                        }
                    });
                    ui.toggle_value(&mut self.uv_wire, format!("{}  UV edges", ph::POLYGON)).on_hover_text("Show the UV layout of this texture set");
                    if ui.button(ph::CORNERS_OUT).on_hover_text("Fit the first tile in the view (F)").clicked() {
                        self.uv_cam = Some(UvCamera::fit([rect.width(), rect.height()]));
                    }
                    if let Some(e) = uvview::uv_extent(&self.mesh, self.active) {
                        let (text, fits) = uvview::extent_summary(e);
                        // With UV tiles on, UVs beyond the first tile are expected.
                        let (text, fits) = if self.udim {
                            let n = self.mesh.set_tiles(self.active).len();
                            (format!("{n} UV tile{}", if n == 1 { "" } else { "s" }), true)
                        } else {
                            (text, fits)
                        };
                        let color = if fits { theme::TEXT_DIM } else { theme::WARN };
                        ui.label(egui::RichText::new(text).color(color).size(11.0));
                    }
                });
            });
        });
    }
}
