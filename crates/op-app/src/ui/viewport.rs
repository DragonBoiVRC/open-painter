//! The 3D viewport tab: navigation, painting, overlays and the axis gizmo.

use super::theme;
use crate::viewport::{Globals, ViewportCallback};
use crate::{srgb_for_display, App, PaintOp, View};
use eframe::egui_wgpu;
use egui::{Color32, Pos2, Rect, Sense, Stroke, UiBuilder};
use egui_phosphor::regular as ph;
use glam::Vec3;
use op_core::{MeshMap, FALLBACK_BASE, FALLBACK_METALLIC, FALLBACK_ROUGHNESS};

impl App {
    /// Re-aim the camera so the whole model is in view.
    pub(crate) fn frame_model(&mut self) {
        let (lo, hi) = self.mesh.bounds();
        self.camera.frame(lo, hi);
    }

    /// A compact "shading mode" dropdown, used by the viewport overlay.
    pub(crate) fn view_combo(&mut self, ui: &mut egui::Ui) {
        let label = |v: View| match v {
            View::Shaded => "Shaded".to_string(),
            View::ShadedAo => "Shaded + baked AO".to_string(),
            View::PaintedNormal => "Normal map".to_string(),
            View::Map(m) => m.label().to_string(),
        };
        let mut view = self.view;
        egui::ComboBox::from_id_salt("viewport_shading").selected_text(label(self.view)).width(150.0).show_ui(ui, |ui| {
            ui.selectable_value(&mut view, View::Shaded, "Shaded");
            ui.selectable_value(&mut view, View::PaintedNormal, "Normal map");
            if self.baked0().contains_key(&MeshMap::AmbientOcclusion) {
                ui.selectable_value(&mut view, View::ShadedAo, "Shaded + baked AO");
            }
            for m in MeshMap::ALL {
                if self.baked0().contains_key(&m) {
                    ui.selectable_value(&mut view, View::Map(m), m.label());
                }
            }
        });
        if view != self.view {
            self.set_view(view);
        }
    }

    pub(crate) fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let ppp = ui.pixels_per_point();
        let (mods, primary_down) = ui.input(|i| (i.modifiers, i.pointer.primary_down()));
        let alt = mods.alt;
        let d = resp.drag_delta();

        // Navigation: Alt+left orbits, middle pans, Alt+right zooms, scroll zooms.
        if resp.dragged_by(egui::PointerButton::Primary) && alt {
            self.camera.orbit(d.x, d.y);
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            self.camera.pan(d.x, d.y, rect.height());
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            if mods.ctrl {
                // Ctrl+right drag: brush size (left/right) and hardness (up/down).
                self.brush.radius_pts = (self.brush.radius_pts + d.x * 0.5).clamp(2.0, 300.0);
                self.brush.hardness = (self.brush.hardness - d.y * 0.005).clamp(0.0, 1.0);
            } else if alt {
                self.camera.zoom(-d.y * 3.0);
            }
        }
        if resp.hovered() {
            self.camera.zoom(ui.input(|i| i.smooth_scroll_delta.y));
        }

        let mut stamps = Vec::new();
        let mut stamp_target = None;
        // A modal dialog must not paint through to the viewport underneath.
        let modal_open = self.dialog.is_some() || self.confirm.is_some();
        let pick = resp.clicked() && mods.command && !alt && !modal_open;
        if let (true, Some(pos)) = (pick, resp.interact_pointer_pos()) {
            self.pick_set_at(rect, pos);
        }
        let painting = resp.is_pointer_button_down_on() && primary_down && !alt && !mods.command && !modal_open;
        match (painting, resp.interact_pointer_pos().or(resp.hover_pos())) {
            (true, Some(pos)) => match self.paint_target() {
                Ok((layer, target)) => {
                    let local = [(pos.x - rect.min.x) * ppp, (pos.y - rect.min.y) * ppp];
                    let cam = self.camera;
                    let camera = [cam.target.x, cam.target.y, cam.target.z, cam.distance, cam.yaw, cam.pitch];
                    // Shift+click continues from where the last stroke ended, as long as nothing moved.
                    if mods.shift && !self.stroke.active {
                        if let Some(a) = self.line_anchor.filter(|a| a.layer == layer && a.target == target && a.camera == camera) {
                            self.begin_line(a.px, ppp, layer, target);
                        }
                    }
                    self.stroke_to(local, ppp, layer, target, &mut stamps);
                    self.line_anchor = Some(crate::LineAnchor { px: local, layer, target, camera });
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
            ops.push(PaintOp::Stamps { layer, target, stamps });
        }
        let batches = self.take_batches(ops);
        if batches.iter().any(|b| !b.ops.is_empty()) {
            self.thumbs_dirty();
        }

        let size_px = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
        let cam = self.camera.position();
        let light = Vec3::new(0.5, 0.8, 0.6).normalize();
        let (near, far) = self.camera.near_far();
        let globals = Globals {
            view_proj: self.camera.view_proj(rect.width() / rect.height().max(1.0)).to_cols_array_2d(),
            camera_pos: [cam.x, cam.y, cam.z, 1.0],
            light_dir: [
                light.x,
                light.y,
                light.z,
                match self.view {
                    View::Shaded => 0.0,
                    View::Map(_) => 1.0,
                    View::ShadedAo => 2.0,
                    View::PaintedNormal => 3.0,
                },
            ],
            // Shown only where the layer stack leaves a pixel uncovered.
            base_color: srgb_for_display(FALLBACK_BASE),
            params: [FALLBACK_METALLIC, FALLBACK_ROUGHNESS, near, far],
        };
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewportCallback { globals, size_px, batches, active: self.active, hidden: self.hidden_flags() },
        ));

        // Brush outline: dark under light so it reads on any surface.
        if let (true, Some(pos)) = (!alt && !modal_open, resp.hover_pos()) {
            let r = self.brush.radius_pts;
            let p = ui.painter().with_clip_rect(rect);
            p.circle_stroke(pos, r, Stroke::new(3.0, Color32::from_black_alpha(110)));
            p.circle_stroke(pos, r, Stroke::new(1.0, Color32::WHITE));
            // A second ring shows where the brush starts to fade.
            let core = r * self.brush.hardness;
            if core > 3.0 {
                p.circle_stroke(pos, core, Stroke::new(1.0, Color32::from_white_alpha(70)));
            }
        }

        self.viewport_overlays(ui, rect);
        draw_axis_gizmo(&ui.painter().with_clip_rect(rect), rect.right_bottom() - egui::vec2(46.0, 46.0), &self.camera);
    }

    fn viewport_overlays(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let bar = Rect::from_min_size(rect.min + egui::vec2(8.0, 8.0), egui::vec2((rect.width() - 16.0).max(0.0), 28.0));
        ui.scope_builder(UiBuilder::new().max_rect(bar), |ui| {
            egui::Frame::NONE
                .fill(Color32::from_black_alpha(130))
                .corner_radius(3)
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(ph::CUBE).color(theme::TEXT_DIM));
                        self.view_combo(ui);
                        if ui.button(ph::FRAME_CORNERS).on_hover_text("Frame model (F)").clicked() {
                            self.frame_model();
                        }
                        let on = self.symmetry != 0;
                        let label = format!("{}  {}", ph::BUTTERFLY, ["Symmetry", "X", "Y", "Z"][self.symmetry as usize]);
                        if ui.selectable_label(on, label).on_hover_text("Mirror strokes (L). Choose the axis in Properties - Paint.").clicked() {
                            self.symmetry = if on { 0 } else { 1 };
                        }
                    });
                });
        });
        // Hint line while a modifier changes what the mouse does.
        let (alt, ctrl) = ui.input(|i| (i.modifiers.alt, i.modifiers.ctrl));
        let hint = if alt {
            Some("Alt: drag to orbit  |  middle drag to pan  |  right drag to zoom")
        } else if ctrl {
            Some("Ctrl + right drag: brush size (left/right) and hardness (up/down)")
        } else {
            None
        };
        if let Some(text) = hint {
            ui.painter().text(
                Pos2::new(rect.center().x, rect.bottom() - 14.0),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(11.5),
                Color32::from_white_alpha(190),
            );
        }
    }
}

/// Small X/Y/Z indicator showing the camera's orientation.
fn draw_axis_gizmo(painter: &egui::Painter, center: Pos2, camera: &crate::camera::OrbitCamera) {
    let (right, up, forward) = camera.basis();
    let len = 26.0;
    let mut axes = [
        (Vec3::X, Color32::from_rgb(0xe0, 0x4b, 0x4b), "X"),
        (Vec3::Y, Color32::from_rgb(0x5c, 0xc0, 0x5c), "Y"),
        (Vec3::Z, Color32::from_rgb(0x4b, 0x8c, 0xe8), "Z"),
    ];
    // Far axes first so the near ones overlap them.
    axes.sort_by(|a, b| b.0.dot(forward).total_cmp(&a.0.dot(forward)));
    painter.circle_filled(center, len + 12.0, Color32::from_black_alpha(60));
    for (axis, color, label) in axes {
        let tip = center + egui::vec2(axis.dot(right), -axis.dot(up)) * len;
        painter.line_segment([center, tip], Stroke::new(2.0, color));
        painter.circle_filled(tip, 7.0, color);
        painter.text(tip, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(9.5), Color32::BLACK);
    }
}
