//! The contents of each dockable panel, plus small widgets they share.

mod bake;
mod brush;
mod display;
mod fill;
mod layer_dnd;
pub mod layers;
mod properties;
mod sets;
mod shelf;
mod texture_set;

pub use shelf::{builtin_brush_names, BrushPreset, ShelfState};

/// The shelf category that lists imported textures.
pub fn shelf_textures() -> shelf::Category {
    shelf::textures_category()
}
use crate::ui::theme;
use std::ops::RangeInclusive;

/// A titled group of controls, collapsible like the property groups in professional tools.
pub fn section<R>(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) -> Option<R> {
    let header = egui::RichText::new(title.to_uppercase()).size(11.0).strong().color(theme::TEXT_DIM);
    egui::CollapsingHeader::new(header)
        .default_open(true)
        .show_unindented(ui, |ui| {
            ui.add_space(2.0);
            let out = body(ui);
            ui.add_space(4.0);
            out
        })
        .body_returned
}

/// A labelled slider: the label sits in a fixed column so a stack of rows lines up.
pub fn slider_row(ui: &mut egui::Ui, label: &str, value: &mut f32, range: RangeInclusive<f32>) -> egui::Response {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(86.0, 18.0), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.add(egui::Label::new(egui::RichText::new(label).color(theme::TEXT)).truncate());
        });
        ui.spacing_mut().slider_width = (ui.available_width() - 62.0).max(40.0);
        ui.add(egui::Slider::new(value, range))
    })
    .inner
}

/// A dim one-line note.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(theme::TEXT_DIM).size(11.0));
}
