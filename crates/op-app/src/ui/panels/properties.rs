//! "Properties - Layer": name, blending, fill values, mask and generator of the selected layer.

use crate::{builtin_kinds, App, GenAction, Target};
use op_core::{BlendMode, Generator, Mask, NodeKind};

impl App {
    pub(crate) fn properties_panel(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.selected.filter(|id| self.doc.find(*id).is_some()) else {
            ui.label(egui::RichText::new("Select a layer to edit its properties.").color(crate::ui::theme::TEXT_DIM));
            return;
        };
        // Property widgets edit the document in place; the gesture is committed to
        // history once the pointer is released so a slider drag is a single undo step.
        let snapshot = self.doc.clone();
        let mut changed = false;
        let mut mask_action = None;
        let mut gen_action: Option<GenAction> = None;
        if let Some(node) = self.doc.find_mut(id) {
            changed |= ui.text_edit_singleline(&mut node.name).changed();
            egui::ComboBox::from_id_salt("blend").selected_text(node.blend.name()).show_ui(ui, |ui| {
                for m in BlendMode::ALL {
                    changed |= ui.selectable_value(&mut node.blend, m, m.name()).changed();
                }
            });
            changed |= ui.add(egui::Slider::new(&mut node.opacity, 0.0..=1.0).text("Opacity")).changed();

            if let NodeKind::Fill(f) = &mut node.kind {
                ui.add_space(4.0);
                changed |= super::fill::fill_controls(ui, f, &self.assets, &self.tex_thumbs);
            }

            ui.separator();
            match &mut node.mask {
                None => {
                    ui.horizontal(|ui| {
                        if ui.button("Add mask (reveal all)").clicked() {
                            mask_action = Some(Some(true));
                        }
                        if ui.button("Add mask (hide all)").clicked() {
                            mask_action = Some(Some(false));
                        }
                    });
                    egui::ComboBox::from_id_salt("gen_new").selected_text("Add generated mask...").show_ui(ui, |ui| {
                        for k in builtin_kinds() {
                            if ui.selectable_label(false, op_proc::label(&k)).clicked() {
                                gen_action = Some(GenAction::NewMask(Generator::new(k)));
                            }
                        }
                        for c in &self.custom_generators {
                            if ui.selectable_label(false, op_proc::label(&c.kind)).clicked() {
                                gen_action = Some(GenAction::NewMask(c.clone()));
                            }
                        }
                    });
                }
                Some(m) => {
                    ui.horizontal(|ui| {
                        changed |= ui.checkbox(&mut m.enabled, "Mask enabled").changed();
                        if ui.button("Delete mask").clicked() {
                            mask_action = Some(None);
                        }
                    });
                    match &mut m.generator {
                        Some(gen) => {
                            ui.label("Generated from the baked maps:");
                            egui::ComboBox::from_id_salt("gen_kind").selected_text(op_proc::label(&gen.kind)).show_ui(ui, |ui| {
                                for k in builtin_kinds() {
                                    if ui.selectable_label(gen.kind == k, op_proc::label(&k)).clicked() && gen.kind != k {
                                        gen.kind = k;
                                        gen.params.clear();
                                        changed = true;
                                    }
                                }
                                for c in &self.custom_generators {
                                    if ui.selectable_label(gen.kind == c.kind, op_proc::label(&c.kind)).clicked() && gen.kind != c.kind {
                                        gen.kind = c.kind.clone();
                                        gen.params.clear();
                                        changed = true;
                                    }
                                }
                            });
                            for spec in op_proc::param_specs(&gen.kind) {
                                let mut v = op_proc::value(gen, spec.key);
                                let mut slider = egui::Slider::new(&mut v, spec.min..=spec.max).text(spec.label);
                                if spec.integer {
                                    slider = slider.step_by(1.0);
                                }
                                if ui.add(slider).changed() {
                                    gen.params.insert(spec.key.to_string(), v);
                                    changed = true;
                                }
                            }
                            changed |= ui.checkbox(&mut gen.invert, "Invert").changed();
                            let missing: Vec<&str> = op_proc::required_maps(&gen.kind)
                                .into_iter()
                                .filter(|m| !self.baked0().contains_key(m))
                                .map(|m| m.label())
                                .collect();
                            if !missing.is_empty() {
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Not baked yet: {}. Use Bake... to create them.", missing.join(", ")),
                                );
                            }
                            if self.procs.iter().any(|p| p.scheduler.is_busy(id)) {
                                ui.label("Generating...");
                            }
                            if let Some(info) = self.procs.iter().find_map(|p| p.info.get(&id)) {
                                for w in &info.warnings {
                                    ui.small(w);
                                }
                                if let Some(e) = &info.error {
                                    ui.colored_label(egui::Color32::LIGHT_RED, format!("Generation failed: {e}"));
                                }
                            }
                            if ui.button("Freeze (make paintable)").clicked() {
                                gen_action = Some(GenAction::Freeze);
                            }
                        }
                        None => {
                            egui::ComboBox::from_id_salt("gen_replace").selected_text("Replace with generated...").show_ui(ui, |ui| {
                                for k in builtin_kinds() {
                                    if ui.selectable_label(false, op_proc::label(&k)).clicked() {
                                        gen_action = Some(GenAction::Replace(Generator::new(k)));
                                    }
                                }
                                for c in &self.custom_generators {
                                    if ui.selectable_label(false, op_proc::label(&c.kind)).clicked() {
                                        gen_action = Some(GenAction::Replace(c.clone()));
                                    }
                                }
                            });
                            ui.small("Replacing is undoable: Ctrl+Z restores the painted mask.");
                        }
                    }
                }
            }
        }
        if changed && self.gesture_before.is_none() {
            self.gesture_before = Some(snapshot.clone());
        }
        if let Some(action) = gen_action {
            self.apply_gen_action(id, action);
        }
        if let Some(action) = mask_action {
            self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    n.mask = action.map(Mask::new);
                }
            });
            if action.is_none() && self.target == Target::Mask {
                self.target = Target::Content;
            }
        }

        ui.separator();
        ui.label("Paint on:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.target, Target::Content, "Layer content");
            ui.radio_value(&mut self.target, Target::Mask, "Mask");
        });
        if let Err(msg) = self.paint_target() {
            ui.small(msg);
        }
    }
}
