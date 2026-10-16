//! "Display Settings": what the viewport shows.

use super::{hint, section};
use crate::{App, View};
use op_core::MeshMap;

const ENVIRONMENTS: [&str; 5] = ["Studio", "Outdoor", "Sunset", "Neutral", "HDRI"];

impl App {
    pub(crate) fn display_panel(&mut self, ui: &mut egui::Ui) {
        section(ui, "Environment", |ui| {
            ui.horizontal(|ui| {
                ui.label("Lighting");
                let shown = match (self.env.preset, &self.hdri_name) {
                    (4, Some(name)) => name.clone(),
                    (p, _) => ENVIRONMENTS[p as usize % 5].to_string(),
                };
                egui::ComboBox::from_id_salt("environment").selected_text(shown).show_ui(ui, |ui| {
                    for (i, name) in ENVIRONMENTS.iter().enumerate().take(4) {
                        ui.selectable_value(&mut self.env.preset, i as u32, *name);
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.label("HDRI");
                let library = self.hdri_library();
                egui::ComboBox::from_id_salt("hdri_library").selected_text(self.hdri_name.clone().unwrap_or_else(|| "None".into())).show_ui(ui, |ui| {
                    for name in &library {
                        if ui.selectable_label(self.hdri_name.as_ref() == Some(name) && self.env.preset == 4, name).clicked() {
                            self.use_hdri_from_library(name);
                        }
                    }
                });
                if ui.button("Load...").on_hover_text("Choose a Radiance .hdr or OpenEXR environment picture").clicked() {
                    self.start_hdri_import();
                }
            });
            super::slider_row(ui, "Rotation", &mut self.env.rotation, 0.0..=360.0);
            super::slider_row(ui, "Exposure", &mut self.env.exposure, -3.0..=3.0);
            super::slider_row(ui, "Key light", &mut self.env.light, 0.0..=3.0);
            hint(ui, "Shift + right drag in the viewport turns the environment.");
            if ui.button("Reset lighting").clicked() {
                self.env = Default::default();
            }
        });
        section(ui, "Shading", |ui| {
            let current = self.view;
            let mut view = current;
            ui.selectable_value(&mut view, View::Shaded, "Shaded");
            ui.selectable_value(&mut view, View::PaintedNormal, "Normal map (painted height + baked normal)");
            for c in 0..3 {
                ui.selectable_value(&mut view, View::Channel(c), crate::ui::viewport::view_label(View::Channel(c)));
            }
            ui.add_enabled_ui(self.baked0().contains_key(&MeshMap::AmbientOcclusion), |ui| {
                ui.selectable_value(&mut view, View::ShadedAo, "Shaded with baked ambient occlusion");
            });
            ui.separator();
            hint(ui, "Baked maps");
            for m in MeshMap::ALL {
                ui.add_enabled_ui(self.baked0().contains_key(&m), |ui| {
                    ui.selectable_value(&mut view, View::Map(m), m.label());
                });
            }
            if self.baked0().is_empty() {
                hint(ui, "Nothing is baked yet. Open Bake Mesh Maps to create maps.");
            }
            if view != current {
                self.set_view(view);
            }
        });
    }
}
