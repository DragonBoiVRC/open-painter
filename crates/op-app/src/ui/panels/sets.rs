//! "Texture Sets": one row per material of the model. Pick one to edit its layers.

use super::hint;
use crate::ui::theme;
use crate::App;
use egui_phosphor::regular as ph;

impl App {
    pub(crate) fn sets_panel(&mut self, ui: &mut egui::Ui) {
        let ranges = self.mesh.set_ranges();
        let mut pick = None;
        let mut eye_action: Option<(usize, bool)> = None;
        let mut menu_action: Option<(usize, u8)> = None;
        let mut resize: Option<(usize, u32)> = None;
        egui::Frame::NONE.fill(theme::LIST).stroke(egui::Stroke::new(1.0, theme::BORDER)).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            egui::ScrollArea::vertical().id_salt("set_list").auto_shrink([false, false]).show(ui, |ui| {
                for (i, r) in ranges.iter().enumerate() {
                    let active = i == self.active;
                    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 26.0), egui::Sense::click());
                    let p = ui.painter_at(rect);
                    let fill = if active {
                        theme::ACCENT_DIM
                    } else if resp.hovered() {
                        theme::CONTROL
                    } else {
                        theme::PANEL
                    };
                    p.rect_filled(rect.shrink2(egui::vec2(0.0, 0.5)), 0.0, fill);
                    if active {
                        p.rect_stroke(rect.shrink(0.5), 0.0, egui::Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
                    }
                    let hidden = self.hidden_sets.contains(&i);
                    let text = match (active, hidden) {
                        (_, true) => theme::TEXT_DIM.gamma_multiply(0.7),
                        (true, _) => theme::TEXT_STRONG,
                        _ => theme::TEXT,
                    };
                    // The eye: click to show or hide this material, Alt+click to look at it alone.
                    let eye = egui::Rect::from_center_size(rect.left_center() + egui::vec2(16.0, 0.0), egui::vec2(22.0, 22.0));
                    let eye_resp = ui.interact(eye, resp.id.with("eye"), egui::Sense::click());
                    p.text(
                        eye.center(),
                        egui::Align2::CENTER_CENTER,
                        if hidden { ph::EYE_SLASH } else { ph::EYE },
                        egui::FontId::proportional(15.0),
                        if hidden { theme::TEXT_DIM } else { theme::TEXT },
                    );
                    let eye_hot = eye_resp.hovered();
                    if eye_resp.on_hover_text("Show or hide this material (Alt+click: show only this one)").clicked() {
                        eye_action = Some((i, ui.input(|inp| inp.modifiers.alt)));
                    }
                    let tiles = self.tile_count(i);
                    let name_rect = egui::Rect::from_min_max(rect.min + egui::vec2(34.0, 0.0), rect.max - egui::vec2(if tiles > 1 { 112.0 } else { 64.0 } + 64.0, 0.0));
                    p.with_clip_rect(name_rect).text(name_rect.left_center(), egui::Align2::LEFT_CENTER, &r.name, egui::FontId::proportional(12.5), text);
                    let count = if tiles > 1 { format!("{tiles} tiles  {}", tri_label(r.index_count / 3)) } else { tri_label(r.index_count / 3) };
                    // The set's resolution: click for the list of sizes.
                    let size = if i == self.active { self.tex_size } else { self.parked[i].as_ref().map_or(0, |st| st.tex_size) };
                    if size > 0 {
                        let w = 62.0;
                        let right = rect.right() - if tiles > 1 { 112.0 } else { 64.0 };
                        let size_rect = egui::Rect::from_min_max(egui::pos2(right - w, rect.top() + 3.0), egui::pos2(right, rect.bottom() - 3.0));
                        let mut b = ui.new_child(egui::UiBuilder::new().max_rect(size_rect));
                        b.menu_button(egui::RichText::new(format!("{size}²")).size(11.5).color(theme::ACCENT), |ui| {
                            for s in crate::TEXTURE_SIZES {
                                if ui.selectable_label(s == size, format!("{s}x{s}")).clicked() {
                                    resize = Some((i, s));
                                    ui.close();
                                }
                            }
                        });
                    }
                    p.text(
                        rect.right_center() - egui::vec2(8.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        count,
                        egui::FontId::proportional(10.5),
                        theme::TEXT_DIM,
                    );
                    if active && self.sets_scrolled != i {
                        resp.scroll_to_me(Some(egui::Align::Center));
                        self.sets_scrolled = i;
                    }
                    let resp = resp.on_hover_text(format!("{}: {} triangles", r.name, r.index_count / 3));
                    if resp.clicked() && !eye_hot {
                        pick = Some(i);
                    }
                    resp.context_menu(|ui| {
                        if ui.button(if hidden { "Show" } else { "Hide" }).clicked() {
                            menu_action = Some((i, 0));
                            ui.close();
                        }
                        if ui.button("Show only this material").clicked() {
                            menu_action = Some((i, 1));
                            ui.close();
                        }
                        if ui.button("Show all materials").clicked() {
                            menu_action = Some((i, 2));
                            ui.close();
                        }
                    });
                }
            });
        });
        if ranges.len() == 1 {
            hint(ui, "This model has one material. Import an FBX with several to get a set for each.");
        } else if !self.hidden_sets.is_empty() {
            ui.horizontal(|ui| {
                hint(ui, &format!("{} of {} hidden", self.hidden_sets.len(), ranges.len()));
                if ui.small_button("Show all").clicked() {
                    self.show_all_sets();
                }
            });
        }
        if let Some((i, s)) = resize {
            self.resize_set(i, s);
        }
        if let Some((i, alt)) = eye_action {
            if alt { self.isolate_set(i) } else { self.toggle_set_visible(i) }
        }
        match menu_action {
            Some((i, 0)) => self.toggle_set_visible(i),
            Some((i, 1)) => self.isolate_set(i),
            Some((_, _)) => self.show_all_sets(),
            None => {}
        }
        if let Some(i) = pick {
            self.switch_set(i);
        }
    }
}

/// "812" or "12.4k".
fn tri_label(n: u32) -> String {
    if n >= 1000 { format!("{:.1}k", n as f32 / 1000.0) } else { n.to_string() }
}

#[cfg(test)]
mod tests {
    use super::tri_label;

    #[test]
    fn triangle_counts_are_compact() {
        assert_eq!(tri_label(0), "0");
        assert_eq!(tri_label(999), "999");
        assert_eq!(tri_label(1000), "1.0k");
        assert_eq!(tri_label(12_400), "12.4k");
    }
}
