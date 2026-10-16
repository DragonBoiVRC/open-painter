//! "Properties - Layer": name, blending, fill values, mask and generator of the selected layer.

use crate::{builtin_kinds, App, GenAction, Target};
use op_core::{BlendMode, Filter, Generator, Mask, NodeKind};

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

            egui::CollapsingHeader::new("Colour adjustment").id_salt("colour_adjust").default_open(!node.adjust.is_identity()).show(ui, |ui| {
                changed |= super::slider_row(ui, "Hue", &mut node.adjust.hue, -180.0..=180.0).changed();
                changed |= super::slider_row(ui, "Saturation", &mut node.adjust.saturation, 0.0..=2.0).changed();
                changed |= super::slider_row(ui, "Brightness", &mut node.adjust.brightness, -1.0..=1.0).changed();
                changed |= super::slider_row(ui, "Contrast", &mut node.adjust.contrast, 0.0..=2.0).changed();
                if ui.add_enabled(!node.adjust.is_identity(), egui::Button::new("Reset")).clicked() {
                    node.adjust = Default::default();
                    changed = true;
                }
            });

            let is_fill = matches!(node.kind, NodeKind::Fill(_));
            egui::CollapsingHeader::new("Channels").id_salt("layer_channels").default_open(!node.channels.is_default()).show(ui, |ui| {
                let layer_blend = node.blend;
                let rows: [(&str, &mut op_core::ChannelMix, bool); 5] = [
                    ("Base color", &mut node.channels.base, true),
                    ("Metallic", &mut node.channels.metallic, true),
                    ("Roughness", &mut node.channels.roughness, true),
                    ("Height", &mut node.channels.height, true),
                    ("Normal", &mut node.channels.normal, false),
                ];
                for (i, (label, mix, has_blend)) in rows.into_iter().enumerate() {
                    if i == 4 && !is_fill {
                        break;
                    }
                    ui.push_id(i, |ui| {
                        ui.horizontal(|ui| {
                            changed |= ui.checkbox(&mut mix.enabled, label).changed();
                            if has_blend {
                                let shown = mix.blend.map(|b| b.name()).unwrap_or("Layer");
                                egui::ComboBox::from_id_salt("chan_blend").selected_text(shown).width(90.0).show_ui(ui, |ui| {
                                    changed |= ui.selectable_value(&mut mix.blend, None, format!("Layer ({})", layer_blend.name())).changed();
                                    for m in BlendMode::ALL {
                                        changed |= ui.selectable_value(&mut mix.blend, Some(m), m.name()).changed();
                                    }
                                });
                            }
                            changed |= ui.add(egui::Slider::new(&mut mix.opacity, 0.0..=1.0).show_value(true)).changed();
                        });
                    });
                }
                if ui.add_enabled(!node.channels.is_default(), egui::Button::new("Reset")).clicked() {
                    node.channels = Default::default();
                    changed = true;
                }
                super::hint(ui, "Each channel can be switched off, blended differently from the layer, or made more transparent.");
            });

            if !is_fill {
                egui::CollapsingHeader::new("Filters").id_salt("layer_filters").default_open(!node.filters.is_empty()).show(ui, |ui| {
                    changed |= filters_ui(ui, &mut node.filters);
                });
            }

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
                    egui::CollapsingHeader::new("Mask levels").id_salt("mask_levels").default_open(!m.adjust.is_identity()).show(ui, |ui| {
                        changed |= super::slider_row(ui, "Black", &mut m.adjust.black, 0.0..=1.0).changed();
                        changed |= super::slider_row(ui, "White", &mut m.adjust.white, 0.0..=1.0).changed();
                        changed |= super::slider_row(ui, "Gamma", &mut m.adjust.gamma, 0.2..=5.0).changed();
                        ui.horizontal(|ui| {
                            changed |= ui.checkbox(&mut m.adjust.invert, "Invert").changed();
                            if ui.add_enabled(!m.adjust.is_identity(), egui::Button::new("Reset")).clicked() {
                                m.adjust = Default::default();
                                changed = true;
                            }
                        });
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

/// The filter list of a paint layer or group: each filter's settings, with reorder and remove buttons,
/// and a menu to add more. Returns true if anything changed.
fn filters_ui(ui: &mut egui::Ui, filters: &mut Vec<Filter>) -> bool {
    let mut changed = false;
    let mut remove = None;
    let mut swap = None;
    let count = filters.len();
    for (i, f) in filters.iter_mut().enumerate() {
        ui.push_id(i, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(f.name()).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("Remove this filter").clicked() {
                        remove = Some(i);
                    }
                    if ui.add_enabled(i + 1 < count, egui::Button::new("v").small()).on_hover_text("Run later").clicked() {
                        swap = Some((i, i + 1));
                    }
                    if ui.add_enabled(i > 0, egui::Button::new("^").small()).on_hover_text("Run earlier").clicked() {
                        swap = Some((i - 1, i));
                    }
                });
            });
            match f {
                Filter::Blur { radius } => changed |= super::slider_row(ui, "Radius", radius, 0.0..=64.0).changed(),
                Filter::Sharpen { amount, radius } => {
                    changed |= super::slider_row(ui, "Amount", amount, 0.0..=4.0).changed();
                    changed |= super::slider_row(ui, "Radius", radius, 0.5..=32.0).changed();
                }
                Filter::Invert => {
                    super::hint(ui, "Flips colours and values (1 - x) where the layer has paint.");
                }
                Filter::Levels { black, white, gamma } => {
                    changed |= super::slider_row(ui, "Black", black, 0.0..=1.0).changed();
                    changed |= super::slider_row(ui, "White", white, 0.0..=1.0).changed();
                    changed |= super::slider_row(ui, "Gamma", gamma, 0.2..=5.0).changed();
                }
            }
            ui.separator();
        });
    }
    if let Some(i) = remove {
        filters.remove(i);
        changed = true;
    }
    if let Some((a, b)) = swap {
        filters.swap(a, b);
        changed = true;
    }
    egui::ComboBox::from_id_salt("filter_add").selected_text("Add filter...").show_ui(ui, |ui| {
        for f in Filter::defaults() {
            if ui.selectable_label(false, f.name()).clicked() {
                filters.push(f);
                changed = true;
            }
        }
    });
    if filters.is_empty() {
        super::hint(ui, "Filters reshape what this layer puts down before it is blended. They never change the paint itself.");
    }
    changed
}
