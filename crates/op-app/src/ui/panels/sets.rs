//! "Texture Sets": one row per material of the model. Pick one to edit its layers.

use super::hint;
use crate::ui::theme;
use crate::App;
use egui_phosphor::regular as ph;

impl App {
    pub(crate) fn sets_panel(&mut self, ui: &mut egui::Ui) {
        let ranges = self.mesh.set_ranges();
        let mut pick = None;
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
                    let text = if active { theme::TEXT_STRONG } else { theme::TEXT };
                    p.text(rect.left_center() + egui::vec2(10.0, 0.0), egui::Align2::LEFT_CENTER, ph::CUBE, egui::FontId::proportional(14.0), text);
                    let name_rect = egui::Rect::from_min_max(rect.min + egui::vec2(32.0, 0.0), rect.max - egui::vec2(64.0, 0.0));
                    p.with_clip_rect(name_rect).text(name_rect.left_center(), egui::Align2::LEFT_CENTER, &r.name, egui::FontId::proportional(12.5), text);
                    p.text(
                        rect.right_center() - egui::vec2(8.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        format!("{}", tri_label(r.index_count / 3)),
                        egui::FontId::proportional(10.5),
                        theme::TEXT_DIM,
                    );
                    if active && self.sets_scrolled != i {
                        resp.scroll_to_me(Some(egui::Align::Center));
                        self.sets_scrolled = i;
                    }
                    if resp.on_hover_text(format!("{}: {} triangles", r.name, r.index_count / 3)).clicked() {
                        pick = Some(i);
                    }
                }
            });
        });
        if ranges.len() == 1 {
            hint(ui, "This model has one material. Import an FBX with several to get a set for each.");
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
