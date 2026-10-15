//! "Texture Set Settings": the resolution of the texture set.

use super::{hint, section};
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
                    self.undo.clear();
                    self.redo.clear();
                    self.pending.push(PaintOp::Reset(self.tex_size));
                    // Baked maps are tied to the resolution they were baked at.
                    self.clear_baked();
                    self.proc.invalidate();
                    self.modified = true;
                }
            });
            hint(ui, "Changing the resolution clears all paint, history and baked maps.");
        });
        section(ui, "Model", |ui| {
            hint(ui, &format!("{}  ({} triangles)", self.mesh_name, self.mesh.indices.len() / 3));
        });
    }
}
