//! "Display Settings": what the viewport shows.

use super::{hint, section};
use crate::{App, View};
use op_core::MeshMap;

impl App {
    pub(crate) fn display_panel(&mut self, ui: &mut egui::Ui) {
        section(ui, "Shading", |ui| {
            let current = self.view;
            let mut view = current;
            ui.selectable_value(&mut view, View::Shaded, "Shaded");
            ui.selectable_value(&mut view, View::PaintedNormal, "Normal map (painted height + baked normal)");
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
