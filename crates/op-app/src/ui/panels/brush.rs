//! "Properties - Paint": the active brush and which channels it writes.

use super::{hint, section, slider_row};
use crate::ui::theme;
use crate::{App, Tool};

/// Names of the stamp shapes, in the order the shaders number them.
const SHAPES: [&str; 5] = ["Round", "Square", "Diamond", "Chalk", "Streaks"];

impl App {
    pub(crate) fn brush_panel(&mut self, ui: &mut egui::Ui) {
        section(ui, "Brush", |ui| {
            ui.horizontal(|ui| {
                for (tool, label) in [(Tool::Paint, "Paint"), (Tool::Erase, "Erase")] {
                    if ui.selectable_label(self.tool == tool, label).clicked() {
                        self.tool = tool;
                    }
                }
            });
            slider_row(ui, "Size", &mut self.brush.radius_pts, 2.0..=300.0);
            slider_row(ui, "Hardness", &mut self.brush.hardness, 0.0..=1.0);
            slider_row(ui, "Opacity", &mut self.brush.opacity, 0.0..=1.0);
            slider_row(ui, "Spacing", &mut self.brush.spacing, 0.02..=1.0);
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
            ui.checkbox(&mut self.channels.metal, "Metallic");
            ui.add_enabled_ui(self.channels.metal, |ui| slider_row(ui, "Metallic", &mut self.channels.metallic, 0.0..=1.0));
            ui.checkbox(&mut self.channels.rough, "Roughness");
            ui.add_enabled_ui(self.channels.rough, |ui| slider_row(ui, "Roughness", &mut self.channels.roughness, 0.0..=1.0));
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
