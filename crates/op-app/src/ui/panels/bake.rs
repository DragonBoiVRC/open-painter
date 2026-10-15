//! "Bake Mesh Maps": choose maps and settings, run the baker, show progress.

use crate::App;
use op_core::MeshMap;

impl App {
    pub(crate) fn bake_panel(&mut self, ui: &mut egui::Ui) {
        let busy = self.bake.job.is_some();
        ui.label(format!("Resolution: {}x{} (the texture set resolution)", self.tex_size, self.tex_size));
        ui.separator();
        ui.label("Maps:");
        ui.horizontal_wrapped(|ui| {
            for m in MeshMap::ALL {
                let mut on = self.bake.maps.contains(&m);
                if ui.add_enabled(!busy, egui::Checkbox::new(&mut on, m.label())).changed() {
                    if on {
                        self.bake.maps.push(m);
                    } else {
                        self.bake.maps.retain(|x| *x != m);
                    }
                }
            }
        });
        ui.add_enabled_ui(!busy, |ui| {
            ui.add(egui::Slider::new(&mut self.bake.samples, 4..=256).logarithmic(true).text("Rays per texel"));
            ui.add(egui::Slider::new(&mut self.bake.ao_distance_pct, 1.0..=100.0).text("AO distance (% of size)"));
            ui.add(egui::Slider::new(&mut self.bake.thickness_pct, 1.0..=100.0).text("Thickness distance (% of size)"));
            ui.add(egui::Slider::new(&mut self.bake.curvature_strength, 0.1..=8.0).logarithmic(true).text("Curvature strength"));
            ui.separator();
            ui.label("High-poly source (optional): normals, height and opacity are projected from it.");
            ui.add(egui::TextEdit::singleline(&mut self.bake.high_path).desired_width(420.0));
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.bake.high_path.trim().is_empty(), egui::Button::new("Load high-poly")).clicked() {
                    self.load_high_poly();
                }
                if self.bake.high.is_some() && ui.button("Remove").clicked() {
                    self.bake.high = None;
                }
            });
            match &self.bake.high {
                Some((m, name)) => ui.label(format!("Using {name} ({} triangles)", m.indices.len() / 3)),
                None => ui.label("None: maps come from the low-poly mesh itself."),
            };
            ui.add(egui::Slider::new(&mut self.bake.cage_front_pct, 0.5..=50.0).text("Search in front (% of size)"));
            ui.add(egui::Slider::new(&mut self.bake.cage_rear_pct, 0.5..=50.0).text("Search behind (% of size)"));
        });
        ui.separator();
        match &self.bake.job {
            Some(job) => {
                ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
                ui.label(format!("{:.0} s elapsed", job.started.elapsed().as_secs_f32()));
                if ui.button("Cancel").clicked() {
                    job.progress.cancel();
                }
            }
            None => {
                if ui.add_enabled(!self.bake.maps.is_empty(), egui::Button::new("Bake")).clicked() {
                    self.start_bake();
                }
            }
        }
        if !self.baked.is_empty() {
            let mut names: Vec<_> = self.baked.keys().map(|m| m.label()).collect();
            names.sort_unstable();
            ui.small(format!("Baked: {}", names.join(", ")));
        }
    }
}
