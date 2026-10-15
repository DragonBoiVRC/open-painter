//! "Texture Set Settings": the resolution of the texture set.

use super::{hint, section, slider_row};
use crate::paint::PaintOp;
use crate::{App, TEXTURE_SIZES};

impl App {
    pub(crate) fn texture_set_panel(&mut self, ui: &mut egui::Ui) {
        section(ui, "Resolution", |ui| {
            ui.horizontal(|ui| {
                ui.label("Size");
                let before = self.tex_size;
                egui::ComboBox::from_id_salt("tex_size")
                    .selected_text(format!("{0} x {0}", self.tex_size))
                    .show_ui(ui, |ui| {
                        for s in TEXTURE_SIZES {
                            ui.selectable_value(&mut self.tex_size, s, format!("{s} x {s}"));
                        }
                    });
                if self.tex_size != before {
                    // Changing resolution discards all paint and history.
                    self.clear_active_history();
                    self.pending.push(PaintOp::Reset(self.tex_size));
                    // Baked maps are tied to the resolution they were baked at.
                    self.clear_baked();
                    self.proc.invalidate();
                    self.modified = true;
                }
            });
            hint(ui, "Changing the resolution clears all paint, history and baked maps.");
        });
        section(ui, "Normal map", |ui| {
            hint(ui, "Painted height becomes a tangent-space normal map, combined with the baked normal map.");
            let before = self.doc.clone();
            let mut strength = self.doc.height_strength;
            if slider_row(ui, "Height strength", &mut strength, 0.0..=8.0).on_hover_text("How strongly painted height bends the normal map").changed() {
                if self.gesture_before.is_none() {
                    self.gesture_before = Some(before);
                }
                self.doc.height_strength = strength;
            }
        });
        section(ui, "Model", |ui| {
            let ranges = self.mesh.set_ranges();
            let set = &ranges[self.active.min(ranges.len() - 1)];
            hint(ui, &format!("{}  ({} triangles)", self.mesh_name, self.mesh.indices.len() / 3));
            if ranges.len() > 1 {
                hint(ui, &format!("Texture set '{}': {} triangles", set.name, set.index_count / 3));
            }
        });
    }
}
