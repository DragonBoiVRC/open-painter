//! The 3D viewport tab: navigation, painting, overlays and the axis gizmo.

use super::theme;
use crate::viewport::{Globals, ViewportCallback};
use crate::{srgb_for_display, App, PaintOp, Tool, View};
use eframe::egui_wgpu;
use egui::{Color32, Pos2, Rect, Sense, Stroke, UiBuilder};
use egui_phosphor::regular as ph;
use glam::Vec3;
use op_core::{MeshMap, FALLBACK_BASE, FALLBACK_METALLIC, FALLBACK_ROUGHNESS};

/// The viewport mode as shown in menus and the status bar.
pub(crate) fn view_label(v: View) -> String {
    match v {
        View::Shaded => "Shaded".to_string(),
        View::ShadedAo => "Shaded + baked AO".to_string(),
        View::PaintedNormal => "Normal map".to_string(),
        View::Channel(0) => "Base color".to_string(),
        View::Channel(1) => "Roughness".to_string(),
        View::Channel(_) => "Metallic".to_string(),
        View::Map(m) => m.label().to_string(),
    }
}

impl App {
    /// Re-aim the camera so the whole model is in view.
    pub(crate) fn frame_model(&mut self) {
        let (lo, hi) = self.mesh.bounds();
        self.camera.frame(lo, hi);
    }

    /// A compact "shading mode" dropdown, used by the viewport overlay.
    pub(crate) fn view_combo(&mut self, ui: &mut egui::Ui) {
        let label = view_label;
        let mut view = self.view;
        egui::ComboBox::from_id_salt("viewport_shading").selected_text(label(self.view)).width(150.0).show_ui(ui, |ui| {
            ui.selectable_value(&mut view, View::Shaded, "Shaded");
            ui.selectable_value(&mut view, View::PaintedNormal, "Normal map");
            for c in 0..3 {
                ui.selectable_value(&mut view, View::Channel(c), view_label(View::Channel(c)));
            }
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
        self.sync_brush_pictures();
        let (mods, primary_down) = ui.input(|i| (i.modifiers, i.pointer.primary_down()));
        let alt = mods.alt;
        let d = resp.drag_delta();
        // Pen pressure arrives as touch force; a mouse (no touch events) paints at full pressure.
        self.pressure = ui.input(|i| {
            let mut p = if primary_down { self.pressure } else { 1.0 };
            for e in &i.events {
                if let egui::Event::Touch { force: Some(f), phase, .. } = e {
                    p = if matches!(phase, egui::TouchPhase::End | egui::TouchPhase::Cancel) { 1.0 } else { f.clamp(0.0, 1.0) };
                }
            }
            p
        });

        // Navigation: Alt+left orbits, middle pans, Alt+right zooms, scroll zooms.
        if resp.dragged_by(egui::PointerButton::Primary) && alt {
            self.camera.orbit(d.x, d.y);
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            self.camera.pan(d.x, d.y, rect.height());
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            if mods.shift {
                // Shift+right drag: turn the environment (and its light).
                self.env.rotation = (self.env.rotation - d.x * 0.5).rem_euclid(360.0);
            } else if mods.ctrl {
                // Ctrl+right drag: brush size (left/right) and hardness (up/down).
                self.brush.radius_pts = (self.brush.radius_pts + d.x * 0.5).clamp(2.0, 300.0);
                self.brush.hardness = (self.brush.hardness - d.y * 0.005).clamp(0.0, 1.0);
            } else if alt {
                self.camera.zoom(-d.y * 3.0);
            }
        }
        if resp.hovered() && !(self.place_stencil && self.brush_pics.stencil.is_some()) {
            self.camera.zoom(ui.input(|i| i.smooth_scroll_delta.y));
        }

        // I: pick the colour under the cursor into the paint colour.
        if resp.hovered() && !self.dialog.is_some() && ui.input(|i| i.key_pressed(egui::Key::I) && !i.modifiers.any()) && !ui.ctx().egui_wants_keyboard_input() {
            if let Some(pos) = resp.hover_pos() {
                match self.sample_color_at(rect, pos) {
                    Some(c) => {
                        self.channels.base_color = c;
                        self.status = "Picked the colour under the cursor.".into();
                    }
                    None => self.status = "No painted surface under the cursor to pick from.".into(),
                }
            }
        }

        let mut stamps = Vec::new();
        let mut stamp_target = None;
        // A modal dialog must not paint through to the viewport underneath.
        let modal_open = self.dialog.is_some() || self.confirm.is_some();
        let pick = resp.clicked() && mods.command && !alt && !modal_open;
        if let (true, Some(pos)) = (pick, resp.interact_pointer_pos()) {
            if self.tool == Tool::Clone {
                // Ctrl+click sets what the clone tool copies from.
                self.clone_source = Some([(pos.x - rect.min.x) * ppp, (pos.y - rect.min.y) * ppp]);
                self.status = "Clone source set: paint to copy from there.".into();
            } else {
                self.pick_set_at(rect, pos);
            }
        }
        let placing = self.place_stencil && self.brush_pics.stencil.is_some();
        if placing {
            let place = &mut self.brush_pics.stencil_place;
            if resp.dragged_by(egui::PointerButton::Primary) && !alt {
                place[0] += d.x / rect.width().max(1.0);
                place[1] += d.y / rect.height().max(1.0);
            }
            if resp.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                place[2] = (place[2] * (1.0 + scroll * 0.002)).clamp(0.05, 2.0);
            }
        }
        let painting = resp.is_pointer_button_down_on() && primary_down && !alt && !mods.command && !modal_open && !placing;
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
                    let copies = matches!(self.tool, Tool::Smudge | Tool::Clone);
                    let blocked = if copies && target == crate::Target::Mask {
                        Some("Smudge and clone work on layer content, not masks")
                    } else if self.tool == Tool::Clone && !self.stroke.active {
                        match self.clone_source.take() {
                            Some(src) => {
                                self.clone_offset = Some([src[0] - local[0], src[1] - local[1]]);
                                None
                            }
                            None if self.clone_offset.is_none() => Some("Ctrl+click to choose where the clone tool copies from"),
                            None => None,
                        }
                    } else {
                        None
                    };
                    if let Some(msg) = blocked {
                        self.status = msg.into();
                        self.stroke.active = false;
                    } else {
                        self.stroke_to(local, ppp, layer, target, &mut stamps);
                        self.line_anchor = Some(crate::LineAnchor { px: local, layer, target, camera });
                        stamp_target = Some((layer, target));
                        ui.ctx().request_repaint();
                    }
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
            if matches!(self.tool, Tool::Smudge | Tool::Clone) {
                ops.push(PaintOp::Smudge { layer, stamps });
            } else {
                ops.push(PaintOp::Stamps { layer, target, stamps });
            }
        }
        let batches = self.take_batches(ops);
        if batches.iter().any(|b| !b.ops.is_empty()) {
            self.thumbs_dirty();
        }

        let size_px = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
        let cam = self.camera.position();
        // The key light turns with the environment, so highlights and reflections stay consistent.
        let light = {
            let (s, c) = self.env.rotation.to_radians().sin_cos();
            let l = Vec3::new(0.5, 0.8, 0.6);
            Vec3::new(c * l.x - s * l.z, l.y, s * l.x + c * l.z).normalize()
        };
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
                    View::Channel(c) => 4.0 + c as f32,
                },
            ],
            // Shown only where the layer stack leaves a pixel uncovered.
            base_color: srgb_for_display(FALLBACK_BASE),
            params: [FALLBACK_METALLIC, FALLBACK_ROUGHNESS, near, far],
            env: [self.env.rotation.to_radians(), 2f32.powf(self.env.exposure), self.env.light, self.env.preset as f32],
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

        // The stencil, drawn faintly over the model so it can be lined up.
        if let (true, Some(id)) = (self.show_stencil, self.brush_pics.stencil.clone()) {
            self.ensure_tex_thumbs(ui.ctx(), std::slice::from_ref(&id));
            if let Some(tex) = self.tex_thumbs.get(&id) {
                let place = self.brush_pics.stencil_place;
                let [tw, th] = tex.size();
                let half_h = place[2] * rect.height() * 0.5;
                let half_w = half_h * tw as f32 / th.max(1) as f32;
                let centre = Pos2::new(rect.min.x + place[0] * rect.width(), rect.min.y + place[1] * rect.height());
                let (sn, cs) = place[3].sin_cos();
                let flip = if self.brush_pics.stencil_flip { -1.0 } else { 1.0 };
                let corner = |x: f32, y: f32| {
                    let (x, y) = (x * half_w, y * half_h);
                    centre + egui::vec2(cs * x - sn * y, sn * x + cs * y)
                };
                let tint = Color32::from_white_alpha(if placing { 120 } else { 70 });
                let mut mesh = egui::Mesh::with_texture(tex.id());
                for (x, y, u, v) in [(-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0), (-1.0, 1.0, 0.0, 1.0)] {
                    let u = if flip < 0.0 { 1.0 - u } else { u };
                    mesh.vertices.push(egui::epaint::Vertex { pos: corner(x, y), uv: Pos2::new(u, v), color: tint });
                }
                mesh.indices.extend([0, 1, 2, 0, 2, 3]);
                let painter = ui.painter().with_clip_rect(rect);
                painter.add(egui::Shape::mesh(mesh));
                if placing {
                    let outline = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0), corner(-1.0, -1.0)];
                    painter.add(egui::Shape::line(outline.to_vec(), Stroke::new(1.0, Color32::from_rgb(0xff, 0xd0, 0x40))));
                }
            }
        }

        // Clone tool: mark where the paint is being copied from.
        if self.tool == Tool::Clone {
            let to_pos = |px: [f32; 2]| Pos2::new(rect.min.x + px[0] / ppp, rect.min.y + px[1] / ppp);
            let marker = match (self.clone_source, self.clone_offset, resp.hover_pos()) {
                (Some(src), _, _) => Some(to_pos(src)),
                (None, Some(off), Some(h)) => Some(h + egui::vec2(off[0], off[1]) / ppp),
                _ => None,
            };
            if let Some(m) = marker {
                let p = ui.painter().with_clip_rect(rect);
                p.circle_stroke(m, 6.0, Stroke::new(3.0, Color32::from_black_alpha(110)));
                p.circle_stroke(m, 6.0, Stroke::new(1.5, Color32::from_rgb(0xff, 0xd0, 0x40)));
                p.line_segment([m - egui::vec2(10.0, 0.0), m + egui::vec2(10.0, 0.0)], Stroke::new(1.0, Color32::from_rgb(0xff, 0xd0, 0x40)));
                p.line_segment([m - egui::vec2(0.0, 10.0), m + egui::vec2(0.0, 10.0)], Stroke::new(1.0, Color32::from_rgb(0xff, 0xd0, 0x40)));
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
