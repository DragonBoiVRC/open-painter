//! "Bake Mesh Maps": choose maps and settings, run the baker, show progress.

use super::{hint, section, slider_row};
use crate::ui::theme;
use crate::App;
use op_core::MeshMap;

impl App {
    pub(crate) fn bake_panel(&mut self, ui: &mut egui::Ui) {
        let busy = self.bake.job.is_some();

        section(ui, "Target", |ui| {
            if self.set_ranges_len() > 1 {
                ui.label(egui::RichText::new(format!("Texture set: {}", self.set_name(self.active))).color(theme::TEXT_STRONG));
            }
            hint(ui, &format!("{0} x {0} (set in Texture Set Settings)", self.tex_size));
        });

        section(ui, "Maps", |ui| {
            ui.horizontal_wrapped(|ui| {
                for m in MeshMap::ALL {
                    let mut on = self.bake.maps.contains(&m);
                    let done = self.baked.contains_key(&m);
                    let label = if done { format!("{} \u{2713}", m.label()) } else { m.label().to_string() };
                    if ui.add_enabled(!busy, egui::Checkbox::new(&mut on, label)).on_hover_text(if done { "Already baked" } else { "Not baked yet" }).changed() {
                        if on {
                            self.bake.maps.push(m);
                        } else {
                            self.bake.maps.retain(|x| *x != m);
                        }
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.add_enabled(!busy, egui::Button::new("All").small()).clicked() {
                    self.bake.maps = MeshMap::ALL.to_vec();
                }
                if ui.add_enabled(!busy, egui::Button::new("None").small()).clicked() {
                    self.bake.maps.clear();
                }
            });
        });

        ui.add_enabled_ui(!busy, |ui| {
            section(ui, "Quality", |ui| {
                let mut samples = self.bake.samples as f32;
                if slider_row(ui, "Rays per texel", &mut samples, 4.0..=256.0).on_hover_text("More rays give smoother AO and thickness, and take longer").changed() {
                    self.bake.samples = samples.round() as u32;
                }
                slider_row(ui, "AO distance", &mut self.bake.ao_distance_pct, 1.0..=100.0).on_hover_text("Occluders farther than this (percent of the model size) are ignored");
                slider_row(ui, "Thickness", &mut self.bake.thickness_pct, 1.0..=100.0).on_hover_text("Rays longer than this (percent of the model size) count as fully thick");
                slider_row(ui, "Curvature", &mut self.bake.curvature_strength, 0.1..=8.0).on_hover_text("Contrast of the curvature map");
            });

            section(ui, "High-poly source", |ui| {
                hint(ui, "Optional. Normals, height and opacity are projected from it.");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.bake.high_path).hint_text("path to a .glb/.gltf").desired_width((ui.available_width() - 100.0).max(60.0)));
                    if crate::picker::browse_button(ui, self.picker.is_some()) {
                        let cur = self.bake.high_path.clone();
                        self.browse(crate::picker::Field::HighPoly, &cur);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.add_enabled(!self.bake.high_path.trim().is_empty(), egui::Button::new("Load")).clicked() {
                        self.load_high_poly();
                    }
                    if self.bake.high.is_some() && ui.button("Remove").clicked() {
                        self.bake.high = None;
                    }
                });
                match &self.bake.high {
                    Some((m, name)) => hint(ui, &format!("Using {name} ({} triangles)", m.indices.len() / 3)),
                    None => hint(ui, "None: maps come from the low-poly mesh itself."),
                }
                slider_row(ui, "Search front", &mut self.bake.cage_front_pct, 0.5..=50.0).on_hover_text("How far in front of the low-poly surface to look for the high-poly (percent of size)");
                slider_row(ui, "Search behind", &mut self.bake.cage_rear_pct, 0.5..=50.0).on_hover_text("How far behind the low-poly surface to look (percent of size)");
            });
        });

        ui.add_space(4.0);
        match &self.bake.job {
            Some(job) => {
                ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{:.0} s elapsed", job.started.elapsed().as_secs_f32())).color(theme::TEXT_DIM));
                    if ui.button("Cancel").clicked() {
                        job.progress.cancel();
                    }
                });
            }
            None => {
                let ready = !self.bake.maps.is_empty();
                let n = self.bake.maps.len();
                let button = egui::Button::new(egui::RichText::new(format!("Bake {n} map(s)")).strong().color(theme::TEXT_STRONG))
                    .fill(if ready { theme::ACCENT_DIM } else { theme::CONTROL })
                    .min_size(egui::vec2(ui.available_width(), 26.0));
                if ui.add_enabled(ready, button).on_hover_text("Bake the selected maps for the active texture set").clicked() {
                    self.start_bake();
                }
            }
        }
    }
}
