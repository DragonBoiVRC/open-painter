//! "Properties - Paint": the active brush and which channels it writes.

use super::{hint, section, slider_row};
use crate::ui::theme;
use crate::{App, Tool};

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
