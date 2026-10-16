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
            ui.add_enabled_ui(self.env.preset == 4 && self.hdri_name.is_some(), |ui| {
                ui.checkbox(&mut self.env.show_background, "Show the environment behind the model");
                if self.env.show_background {
                    super::slider_row(ui, "Background blur", &mut self.env.background_blur, 0.0..=1.0);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Tone mapping");
                egui::ComboBox::from_id_salt("tonemap").selected_text(["Filmic", "Reinhard", "None"][self.env.tonemap as usize % 3]).show_ui(ui, |ui| {
                    for (i, name) in ["Filmic", "Reinhard", "None"].iter().enumerate() {
                        ui.selectable_value(&mut self.env.tonemap, i as u32, *name);
                    }
                });
            });
            ui.checkbox(&mut self.env.show_transparency, "Show opacity (see-through paint)").on_hover_text("See-through paint is drawn as dithered holes; switch off to see the model solid");
            ui.checkbox(&mut self.env.fxaa, "Smooth edges (anti-aliasing)");
            super::slider_row(ui, "Rotation", &mut self.env.rotation, 0.0..=360.0);
            super::slider_row(ui, "Exposure", &mut self.env.exposure, -3.0..=3.0);
            super::slider_row(ui, "Key light", &mut self.env.light, 0.0..=3.0);
            hint(ui, "Shift + right drag in the viewport turns the environment.");
            if ui.button("Reset lighting").clicked() {
                self.env = Default::default();
            }
        });
        section(ui, "Camera", |ui| {
            let mut fov = self.camera.fov_y.to_degrees();
            if super::slider_row(ui, "Field of view", &mut fov, 10.0..=110.0).changed() {
                self.camera.fov_y = fov.clamp(10.0, 110.0).to_radians();
            }
            ui.horizontal_wrapped(|ui| {
                for v in crate::camera::ViewPreset::ALL {
                    if ui.button(v.label()).clicked() {
                        self.camera.set_view(v);
                    }
                }
            });
            hint(ui, "Orbit with Alt + left drag. The buttons look along an axis; F frames the model.");
        });
        section(ui, "Shading", |ui| {
            let current = self.view;
            let mut view = current;
            ui.selectable_value(&mut view, View::Shaded, "Shaded");
            ui.selectable_value(&mut view, View::PaintedNormal, "Normal map (painted height + baked normal)");
            for c in 0..5 {
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
