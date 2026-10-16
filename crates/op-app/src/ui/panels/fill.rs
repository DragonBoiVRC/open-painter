//! The channels of a fill layer: for each, a constant or a texture dragged from the shelf, plus how
//! the textures are placed over the UV layout.

use super::shelf::Asset;
use crate::textures::TextureAsset;
use crate::ui::theme;
use egui_phosphor::regular as ph;
use op_core::{FillParams, Projection, Slot, TexRef};
use std::collections::HashMap;

const SLOT_SIZE: f32 = 42.0;

/// Edit `f` in place; true if anything changed.
pub fn fill_controls(ui: &mut egui::Ui, f: &mut FillParams, assets: &[TextureAsset], thumbs: &HashMap<String, egui::TextureHandle>) -> bool {
    let mut changed = false;
    for slot in Slot::ALL {
        changed |= channel_row(ui, f, slot, assets, thumbs);
        ui.add_space(2.0);
    }
    if !f.maps.is_empty() {
        ui.add_space(2.0);
        ui.label(egui::RichText::new("TEXTURE PLACEMENT").size(11.0).strong().color(theme::TEXT_DIM));
        changed |= placement_controls(ui, f);
    }
    changed
}

fn placement_controls(ui: &mut egui::Ui, f: &mut FillParams) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Projection");
        egui::ComboBox::from_id_salt("fill_projection").selected_text(f.projection.label()).show_ui(ui, |ui| {
            for p in Projection::ALL {
                changed |= ui.selectable_value(&mut f.projection, p, p.label()).changed();
            }
        });
    });
    if f.projection == Projection::Triplanar {
        let mut s = f.sharpness;
        if ui.add(egui::Slider::new(&mut s, 1.0..=16.0).text("Blend sharpness")).changed() {
            f.sharpness = s;
            changed = true;
        }
    }
    if f.projection != Projection::Uv {
        super::hint(ui, "Projections need the Position map (and for triplanar the World space normal map) baked for this texture set: use the Bake panel. Until then the UV layout is used. Tiling 1 spans the model's longest side; normal maps are not projected.");
    }
    egui::Grid::new("tex_placement").num_columns(3).spacing([6.0, 3.0]).show(ui, |ui| {
        let mut pair = |ui: &mut egui::Ui, label: &str, v: &mut [f32; 2], speed: f64, lo: f32, hi: f32| {
            ui.label(label);
            changed |= ui.add(egui::DragValue::new(&mut v[0]).speed(speed).range(lo..=hi)).changed();
            changed |= ui.add(egui::DragValue::new(&mut v[1]).speed(speed).range(lo..=hi)).changed();
            ui.end_row();
        };
        pair(ui, "Tiling", &mut f.uv.tiling, 0.02, -1000.0, 1000.0);
        pair(ui, "Offset", &mut f.uv.offset, 0.005, -1000.0, 1000.0);
        ui.label("Rotation");
        changed |= ui.add(egui::DragValue::new(&mut f.uv.rotation).speed(0.5).suffix("\u{b0}").range(-3600.0..=3600.0)).changed();
        ui.end_row();
    });
    if !f.uv.is_identity() && ui.small_button("Reset placement").clicked() {
        f.uv = Default::default();
        changed = true;
    }
    changed
}

fn constant_controls(ui: &mut egui::Ui, f: &mut FillParams, slot: Slot) -> bool {
    let mut changed = false;
    match slot {
        Slot::BaseColor => {
            let mut on = f.base.is_some();
            if ui.checkbox(&mut on, "Color").changed() {
                f.base = on.then(|| f.base.unwrap_or([0.8, 0.8, 0.8]));
                changed = true;
            }
            if let Some(c) = &mut f.base {
                changed |= ui.color_edit_button_rgb(c).changed();
            }
        }
        Slot::Metallic => changed |= scalar(ui, &mut f.metallic, 0.0, "Value"),
        Slot::Roughness => changed |= scalar(ui, &mut f.roughness, 0.5, "Value"),
        Slot::Height => changed |= scalar(ui, &mut f.height, 0.5, "Value"),
        Slot::Normal => {}
        Slot::Emissive => {
            let mut on = f.emissive.is_some();
            if ui.checkbox(&mut on, "Emissive").changed() {
                f.emissive = on.then(|| f.emissive.unwrap_or([1.0, 0.6, 0.2]));
                changed = true;
            }
            if let Some(c) = &mut f.emissive {
                changed |= ui.color_edit_button_rgb(c).changed();
            }
        }
    }
    changed
}

fn scalar(ui: &mut egui::Ui, v: &mut Option<f32>, default: f32, label: &str) -> bool {
    let mut changed = false;
    let mut on = v.is_some();
    if ui.checkbox(&mut on, label).changed() {
        *v = on.then(|| v.unwrap_or(default));
        changed = true;
    }
    if let Some(x) = v {
        changed |= ui.add(egui::Slider::new(x, 0.0..=1.0)).changed();
    }
    changed
}

/// One channel: its slot (a drop target showing the texture), then either the texture's options or
/// the constant's controls.
fn channel_row(ui: &mut egui::Ui, f: &mut FillParams, slot: Slot, assets: &[TextureAsset], thumbs: &HashMap<String, egui::TextureHandle>) -> bool {
    let mut changed = false;
    let ctx = ui.ctx().clone();
    let current = f.maps.get(slot).cloned();
    ui.horizontal(|ui| {
        // The slot box.
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(SLOT_SIZE, SLOT_SIZE), egui::Sense::click());
        let dragging = egui::DragAndDrop::payload::<Asset>(&ctx).and_then(|p| match &*p {
            Asset::Texture(id) => Some(id.clone()),
            _ => None,
        });
        let over = dragging.is_some() && resp.contains_pointer();
        let painter = ui.painter();
        painter.rect_filled(rect, 3.0, theme::LIST);
        match current.as_ref().and_then(|t| thumbs.get(&t.id)) {
            Some(h) => {
                painter.image(h.id(), rect.shrink(1.0), egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            }
            None if current.is_some() => {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, ph::IMAGE_BROKEN, egui::FontId::proportional(18.0), theme::WARN);
            }
            None => {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, ph::IMAGE, egui::FontId::proportional(18.0), theme::TEXT_DIM);
            }
        }
        let stroke = if over {
            egui::Stroke::new(2.0, theme::ACCENT)
        } else if dragging.is_some() {
            egui::Stroke::new(1.0, theme::ACCENT_DIM)
        } else {
            egui::Stroke::new(1.0, theme::BORDER)
        };
        painter.rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Outside);
        let tip = if current.is_some() { "Drop another texture here to replace it" } else { "Drag a texture here from the shelf" };
        resp.on_hover_text(tip);
        if let (true, Some(id)) = (over && ctx.input(|i| i.pointer.any_released()), dragging) {
            let directx = assets.iter().find(|a| a.id == id).and_then(|a| a.guess()).is_some_and(|g| g.directx);
            f.maps.set(slot, Some(TexRef { flip_green: slot == Slot::Normal && directx, ..TexRef::new(id) }));
            egui::DragAndDrop::clear_payload(&ctx);
            changed = true;
        }

        ui.vertical(|ui| {
            ui.label(egui::RichText::new(slot.label()).color(theme::TEXT_STRONG));
            match f.maps.get(slot).cloned() {
                Some(tex) => {
                    let name = assets.iter().find(|a| a.id == tex.id).map_or("(missing from the library)", |a| a.name.as_str());
                    ui.horizontal(|ui| {
                        ui.add(egui::Label::new(egui::RichText::new(name).size(11.0).color(theme::TEXT_DIM)).truncate());
                        if ui.small_button(ph::X).on_hover_text("Remove the texture").clicked() {
                            f.maps.set(slot, None);
                            changed = true;
                        }
                    });
                    if let Some(t) = f.maps.get(slot).cloned() {
                        let mut t2 = t.clone();
                        match slot {
                            Slot::Normal => {
                                ui.checkbox(&mut t2.flip_green, "Flip green (DirectX)").on_hover_text("Tick for normal maps made for DirectX, where green points down");
                            }
                            Slot::BaseColor => {}
                            _ => {
                                ui.checkbox(&mut t2.invert, "Invert");
                            }
                        }
                        if t2 != t {
                            f.maps.set(slot, Some(t2));
                            changed = true;
                        }
                    }
                }
                None => {
                    ui.horizontal(|ui| changed |= constant_controls(ui, f, slot));
                    if slot == Slot::Normal {
                        ui.label(egui::RichText::new("Drop a normal map here").size(11.0).color(theme::TEXT_DIM));
                    }
                }
            }
        });
    });
    changed
}
