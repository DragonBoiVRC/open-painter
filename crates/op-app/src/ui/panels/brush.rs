//! "Properties - Paint": the active brush and which channels it writes.

use super::{hint, section, slider_row};
use crate::ui::theme;
use crate::{App, PaintOp, Tool};

/// Names of the stamp shapes, in the order the shaders number them.
const SHAPES: [&str; 6] = ["Round", "Square", "Diamond", "Chalk", "Streaks", "Image"];

impl App {
    pub(crate) fn brush_panel(&mut self, ui: &mut egui::Ui) {
        section(ui, "Brush", |ui| {
            ui.horizontal(|ui| {
                for (tool, label) in [(Tool::Paint, "Paint"), (Tool::Erase, "Erase"), (Tool::Smudge, "Smudge"), (Tool::Clone, "Clone"), (Tool::Project, "Project"), (Tool::Fill, "Fill"), (Tool::Hide, "Hide")] {
                    if ui.selectable_label(self.tool == tool, label).clicked() {
                        self.tool = tool;
                    }
                }
            });
            if self.tool == Tool::Smudge {
                slider_row(ui, "Smear distance", &mut self.brush.smudge_distance, 0.05..=1.5);
                hint(ui, "Drags the paint under the brush along the stroke. Works on the layer's content in the 3D view.");
            }
            if self.tool == Tool::Hide {
                ui.horizontal(|ui| {
                    ui.label("Hide");
                    for scope in crate::FillScope::ALL {
                        if ui.selectable_label(self.fill_scope == scope, scope.label()).clicked() {
                            self.fill_scope = scope;
                        }
                    }
                });
                if ui.button("Show all parts").clicked() {
                    self.pending.push(PaintOp::ShowAll);
                }
                hint(ui, "Click the model to hide that part so you can paint what is behind it. Shift+click shows it again. Hidden parts are not saved with the project.");
            }
            if self.tool == Tool::Fill {
                ui.horizontal(|ui| {
                    ui.label("Fill");
                    for scope in crate::FillScope::ALL {
                        if ui.selectable_label(self.fill_scope == scope, scope.label()).clicked() {
                            self.fill_scope = scope;
                        }
                    }
                });
                hint(ui, "Click the model to fill with the paint colour and channels below (flow sets the opacity). On a mask it reveals; Shift+click hides.");
            }
            if self.tool == Tool::Project {
                hint(ui, "Paints the stencil picture onto the model from the camera's point of view. Set the stencil below, line it up, then paint over it.");
            }
            if self.tool == Tool::Clone {
                hint(ui, "Ctrl+click to choose where to copy from, then paint. The offset stays until you pick a new source.");
            }
            slider_row(ui, "Size", &mut self.brush.radius_pts, 2.0..=300.0);
            slider_row(ui, "Hardness", &mut self.brush.hardness, 0.0..=1.0);
            slider_row(ui, "Flow", &mut self.brush.opacity, 0.0..=1.0).on_hover_text("Opacity of each dab: overlapping dabs build up");
            slider_row(ui, "Stroke opacity", &mut self.brush.stroke_opacity, 0.0..=1.0).on_hover_text("Caps the opacity of a whole stroke, however many dabs overlap. Applies to painting and masks, not to smudge, clone or erase on layer content.");
            slider_row(ui, "Spacing", &mut self.brush.spacing, 0.02..=1.0);
            slider_row(ui, "Lazy mouse", &mut self.brush.lazy_radius, 0.0..=200.0).on_hover_text("The brush trails the cursor on a string this long, which smooths wobbly strokes. 0 is off.");
        });
        section(ui, "Symmetry", |ui| {
            ui.horizontal(|ui| {
                for (axis, label) in [(0, "Off"), (1, "X"), (2, "Y"), (3, "Z")] {
                    if ui.selectable_label(self.symmetry == axis, label).clicked() {
                        self.symmetry = axis;
                    }
                }
            });
            hint(ui, "Mirrors strokes across a plane through the middle of the model (L toggles X).");
        });
        section(ui, "Shape", |ui| {
            ui.horizontal(|ui| {
                ui.label("Stamp");
                egui::ComboBox::from_id_salt("brush_shape").selected_text(SHAPES[self.brush.shape as usize % SHAPES.len()]).show_ui(ui, |ui| {
                    for (i, name) in SHAPES.iter().enumerate() {
                        ui.selectable_value(&mut self.brush.shape, i as u32, *name);
                    }
                });
            });
            slider_row(ui, "Angle", &mut self.brush.angle, 0.0..=360.0);
            ui.checkbox(&mut self.brush.follow_path, "Follow stroke direction");
            slider_row(ui, "Angle jitter", &mut self.brush.angle_jitter, 0.0..=1.0);
            slider_row(ui, "Size jitter", &mut self.brush.size_jitter, 0.0..=1.0);
            slider_row(ui, "Scatter", &mut self.brush.position_jitter, 0.0..=4.0);
            slider_row(ui, "Opacity jitter", &mut self.brush.opacity_jitter, 0.0..=1.0);
        });
        section(ui, "Reach", |ui| {
            ui.checkbox(&mut self.paint_through, "Paint through the model").on_hover_text("Strokes also land on surfaces hidden behind the first one");
            ui.checkbox(&mut self.paint_backfaces, "Paint back faces").on_hover_text("Strokes also land on surfaces turned away from the camera (thin cloth, open shells)");
        });
        section(ui, "Image and stencil", |ui| {
            let names: Vec<(String, String)> = self.assets.iter().map(|a| (a.id.clone(), a.name.clone())).collect();
            if names.is_empty() {
                hint(ui, "Import images into the Shelf (Textures) to use them as a brush shape or a stencil.");
            }
            let name_of = |id: &Option<String>| match id {
                Some(id) => names.iter().find(|(i, _)| i == id).map(|(_, n)| n.clone()).unwrap_or_else(|| "(missing)".into()),
                None => "None".into(),
            };
            let mut alpha = self.brush_pics.alpha.clone();
            ui.horizontal(|ui| {
                ui.label("Stamp image");
                egui::ComboBox::from_id_salt("brush_alpha").selected_text(name_of(&alpha)).show_ui(ui, |ui| {
                    ui.selectable_value(&mut alpha, None, "None");
                    for (id, name) in &names {
                        ui.selectable_value(&mut alpha, Some(id.clone()), name);
                    }
                });
            });
            if alpha != self.brush_pics.alpha {
                if alpha.is_some() {
                    self.brush.shape = 5;
                } else if self.brush.shape == 5 {
                    self.brush.shape = 0;
                }
                self.brush_pics.alpha = alpha;
            }
            hint(ui, "The picture's brightness (and alpha) shapes each stamp. Use the Image stamp above.");
            let mut stencil = self.brush_pics.stencil.clone();
            ui.horizontal(|ui| {
                ui.label("Stencil");
                egui::ComboBox::from_id_salt("brush_stencil").selected_text(name_of(&stencil)).show_ui(ui, |ui| {
                    ui.selectable_value(&mut stencil, None, "None");
                    for (id, name) in &names {
                        ui.selectable_value(&mut stencil, Some(id.clone()), name);
                    }
                });
            });
            self.brush_pics.stencil = stencil;
            if self.brush_pics.stencil.is_some() {
                let place = &mut self.brush_pics.stencil_place;
                slider_row(ui, "Scale", &mut place[2], 0.05..=2.0);
                let mut deg = place[3].to_degrees();
                slider_row(ui, "Rotation", &mut deg, -180.0..=180.0);
                place[3] = deg.to_radians();
                slider_row(ui, "Position X", &mut place[0], -0.5..=1.5);
                slider_row(ui, "Position Y", &mut place[1], -0.5..=1.5);
                ui.checkbox(&mut self.brush_pics.stencil_flip, "Mirror");
                ui.checkbox(&mut self.show_stencil, "Show it over the viewport");
                ui.checkbox(&mut self.place_stencil, "Move it in the viewport");
                hint(ui, "Paint only lands where the stencil is bright. While moving it, drag to position and scroll to resize.");
            } else {
                self.place_stencil = false;
            }
        });
        section(ui, "Pen pressure", |ui| {
            ui.checkbox(&mut self.brush.pressure_size, "Pressure changes size");
            ui.checkbox(&mut self.brush.pressure_opacity, "Pressure changes opacity");
            hint(ui, "Only affects pens that report pressure; a mouse paints at full pressure.");
        });
        section(ui, "Paint channels", |ui| {
            hint(ui, "Used when painting layer content. Masks paint white (reveal) and erase to black (hide).");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.channels.base, "Base color");
                ui.color_edit_button_rgb(&mut self.channels.base_color);
            });
            if !self.recent_colors.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                    for c in self.recent_colors.clone() {
                        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
                        ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(to8(c[0]), to8(c[1]), to8(c[2])));
                        let on = self.channels.base_color == c;
                        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(if on { 1.5 } else { 1.0 }, if on { egui::Color32::WHITE } else { egui::Color32::from_gray(0x55) }), egui::StrokeKind::Inside);
                        if resp.on_hover_text("A colour you painted with").clicked() {
                            self.channels.base_color = c;
                        }
                    }
                });
            }
            ui.checkbox(&mut self.channels.metal, "Metallic");
            ui.add_enabled_ui(self.channels.metal, |ui| slider_row(ui, "Metallic", &mut self.channels.metallic, 0.0..=1.0));
            ui.checkbox(&mut self.channels.rough, "Roughness");
            ui.add_enabled_ui(self.channels.rough, |ui| slider_row(ui, "Roughness", &mut self.channels.roughness, 0.0..=1.0));
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.channels.emissive, "Emissive");
                ui.color_edit_button_rgb(&mut self.channels.emissive_color);
            });
            ui.checkbox(&mut self.channels.opacity, "Opacity");
            ui.add_enabled_ui(self.channels.opacity, |ui| slider_row(ui, "Opacity value", &mut self.channels.opacity_value, 0.0..=1.0).on_hover_text("0 is see-through, 1 is solid"));
            ui.checkbox(&mut self.channels.height, "Height (normal map)");
            ui.add_enabled_ui(self.channels.height, |ui| slider_row(ui, "Height", &mut self.channels.height_value, 0.0..=1.0).on_hover_text("0 digs in, 0.5 is flat, 1 raises"));
        });
        section(ui, "Mouse and keys", |ui| {
            egui::Grid::new("mouse_keys").num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
                for (keys, what) in [
                    ("Left drag", "paint"),
                    ("Ctrl + click", "pick texture set"),
                    ("Alt + left drag", "orbit"),
                    ("Middle drag", "pan"),
                    ("Alt + right drag, scroll", "zoom"),
                    ("Ctrl + right drag", "brush size / hardness"),
                    ("[  ]", "brush size"),
                    ("P  E", "paint / erase"),
                    ("I", "pick the colour under the cursor"),
                    ("L", "mirror strokes (symmetry)"),
                    ("C", "cycle channel views"),
                    ("Tab", "hide panels"),
                    ("F", "frame the model"),
                ] {
                    ui.label(egui::RichText::new(keys).color(theme::TEXT));
                    hint(ui, what);
                    ui.end_row();
                }
            });
        });
    }
}
