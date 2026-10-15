//! "Layers": the layer stack as rows with visibility, thumbnails, name and blend summary, plus the
//! toolbar, context menu, in-place rename and drag-and-drop reordering.

use super::layer_dnd::{self, Drop, Indicator, Row};
use crate::ui::theme;
use crate::{App, PaintOp};
use egui_phosphor::regular as ph;
use op_core::{Document, FillParams, Mask, Node, NodeId, NodeKind};

const ROW_HEIGHT: f32 = 38.0;
const INDENT: f32 = 14.0;
const THUMB: f32 = 28.0;

/// Everything a row can ask for; applied after the list is drawn so drawing never edits the document.
enum LayerAct {
    Select(NodeId),
    Visible(NodeId, bool),
    ToggleFold(NodeId),
    BeginRename(NodeId),
    CommitRename(NodeId, String),
    CancelRename,
    Duplicate(NodeId),
    Delete(NodeId),
    Wrap(NodeId),
    MoveUp(NodeId),
    MoveDown(NodeId),
    AddMask(NodeId, bool),
    RemoveMask(NodeId),
    Drop(NodeId, Drop),
}

impl App {
    pub(crate) fn layers_panel(&mut self, ui: &mut egui::Ui) {
        self.layer_toolbar(ui);
        ui.add_space(2.0);

        let mut acts = Vec::new();
        let rows = layer_dnd::rows(&self.doc.nodes, &|id| self.collapsed.contains(&id));
        let dragging: Option<NodeId> = egui::DragAndDrop::payload::<NodeId>(ui.ctx()).map(|p| *p);
        let mut hover_drop: Option<(NodeId, Drop)> = None;

        egui::Frame::NONE
            .fill(theme::LIST)
            .stroke(egui::Stroke::new(1.0, theme::BORDER))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                egui::ScrollArea::vertical()
                    .id_salt("layer_list")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if rows.is_empty() {
                            ui.add_space(12.0);
                            ui.vertical_centered(|ui| {
                                super::hint(ui, "No layers yet.");
                                super::hint(ui, "Use the buttons above to add one.");
                            });
                        }
                        for row in &rows {
                            let Some(node) = self.doc.find(row.id) else { continue };
                            let drop = layer_row(ui, &self.doc, self.selected, &mut self.renaming, node, row, dragging, &mut acts);
                            if let Some(d) = drop {
                                hover_drop = Some((row.id, d));
                            }
                        }
                        // Room below the last row to drop at the very bottom is the "below" half of that row.
                        ui.add_space(24.0);
                    });
            });

        // Draw the drop indicator and resolve the drop when the pointer is released.
        if let (Some(dragged), Some((_, d))) = (dragging, hover_drop) {
            if ui.ctx().input(|i| i.pointer.any_released()) {
                acts.push(LayerAct::Drop(dragged, d));
            }
        }
        if dragging.is_some() && ui.ctx().input(|i| i.pointer.any_released()) {
            egui::DragAndDrop::clear_payload(ui.ctx());
        }

        for a in acts {
            self.apply_layer_act(a);
        }
    }

    fn layer_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            let sel = self.selected.filter(|id| self.doc.find(*id).is_some());
            if icon_button(ui, ph::PAINT_BRUSH, "Add paint layer", true).clicked() {
                self.add_node("Paint layer", NodeKind::Paint);
            }
            if icon_button(ui, ph::PAINT_BUCKET, "Add fill layer", true).clicked() {
                let fill = FillParams {
                    base: Some([0.5, 0.5, 0.5]),
                    metallic: None,
                    roughness: None,
                };
                self.add_node("Fill layer", NodeKind::Fill(fill));
            }
            if icon_button(ui, ph::FOLDER_SIMPLE_PLUS, "Add group", true).clicked() {
                self.add_node("Group", NodeKind::Group { children: Vec::new() });
            }
            ui.add_space(6.0);
            let has_mask = sel.and_then(|id| self.doc.find(id)).is_some_and(|n| n.mask.is_some());
            if icon_button(
                ui,
                ph::CIRCLE_HALF,
                "Add white mask (reveals everything)",
                sel.is_some() && !has_mask,
            )
            .clicked()
            {
                if let Some(id) = sel {
                    self.apply_layer_act(LayerAct::AddMask(id, true));
                }
            }
            if icon_button(ui, ph::COPY, "Duplicate layer", sel.is_some()).clicked() {
                if let Some(id) = sel {
                    self.apply_layer_act(LayerAct::Duplicate(id));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if icon_button(ui, ph::TRASH, "Delete layer", sel.is_some()).clicked() {
                    if let Some(id) = sel {
                        self.apply_layer_act(LayerAct::Delete(id));
                    }
                }
            });
        });
    }
    fn apply_layer_act(&mut self, act: LayerAct) {
        match act {
            LayerAct::Select(id) => self.selected = Some(id),
            LayerAct::Visible(id, v) => self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    n.visible = v;
                }
            }),
            LayerAct::ToggleFold(id) => {
                if !self.collapsed.remove(&id) {
                    self.collapsed.insert(id);
                }
            }
            LayerAct::BeginRename(id) => {
                if let Some(n) = self.doc.find(id) {
                    self.renaming = Some((id, n.name.clone()));
                    self.selected = Some(id);
                }
            }
            LayerAct::CancelRename => self.renaming = None,
            LayerAct::CommitRename(id, name) => {
                self.renaming = None;
                let name = name.trim().to_string();
                if !name.is_empty() {
                    self.edit(|d| {
                        if let Some(n) = d.find_mut(id) {
                            n.name = name;
                        }
                    });
                }
            }
            LayerAct::Duplicate(id) => {
                let before = self.doc.clone();
                if let Some((copy, pairs)) = self.doc.duplicate(id) {
                    self.pending.push(PaintOp::CopyLayers(pairs));
                    self.selected = Some(copy);
                    self.record(before);
                }
            }
            LayerAct::Delete(id) => {
                self.edit(|d| {
                    d.remove(id);
                });
                self.fix_selection();
            }
            LayerAct::Wrap(id) => {
                let before = self.doc.clone();
                if let Some(g) = self.doc.wrap_in_group(id) {
                    self.selected = Some(g);
                    self.record(before);
                }
            }
            LayerAct::MoveUp(id) => self.edit(|d| {
                d.move_up(id);
            }),
            LayerAct::MoveDown(id) => self.edit(|d| {
                d.move_down(id);
            }),
            LayerAct::AddMask(id, white) => self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    if n.mask.is_none() {
                        n.mask = Some(Mask::new(white));
                    }
                }
            }),
            LayerAct::RemoveMask(id) => self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    n.mask = None;
                }
            }),
            LayerAct::Drop(id, d) => {
                self.edit(|doc| {
                    doc.move_to(id, d.parent, d.index);
                });
                if let Some(p) = d.parent {
                    self.collapsed.remove(&p);
                }
            }
        }
    }
}

/// Draw one row; returns the drop slot if a layer is being dragged over it.
#[allow(clippy::too_many_arguments)]
fn layer_row(
    ui: &mut egui::Ui,
    doc: &Document,
    selected: Option<NodeId>,
    renaming: &mut Option<(NodeId, String)>,
    node: &Node,
    row: &Row,
    dragging: Option<NodeId>,
    acts: &mut Vec<LayerAct>,
) -> Option<Drop> {
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, ROW_HEIGHT), egui::Sense::click_and_drag());
    let selected = selected == Some(node.id);
    let p = ui.painter_at(rect);

    let fill = if selected {
        theme::ACCENT_DIM
    } else if resp.hovered() {
        theme::CONTROL
    } else {
        theme::PANEL
    };
    p.rect_filled(rect.shrink2(egui::vec2(0.0, 0.5)), 0.0, fill);
    if selected {
        p.rect_stroke(
            rect.shrink(0.5),
            0.0,
            egui::Stroke::new(1.0, theme::ACCENT),
            egui::StrokeKind::Inside,
        );
    }
    let dim = !node.visible;
    let text = if dim { theme::TEXT_DIM } else { theme::TEXT_STRONG };

    // Visibility eye.
    let eye = egui::Rect::from_center_size(egui::pos2(rect.left() + 14.0, rect.center().y), egui::vec2(22.0, 22.0));
    let eye_resp = ui.interact(eye, resp.id.with("eye"), egui::Sense::click());
    p.text(
        eye.center(),
        egui::Align2::CENTER_CENTER,
        if node.visible { ph::EYE } else { ph::EYE_SLASH },
        egui::FontId::proportional(15.0),
        if node.visible { theme::TEXT } else { theme::TEXT_DIM },
    );
    if eye_resp.on_hover_text("Show or hide").clicked() {
        acts.push(LayerAct::Visible(node.id, !node.visible));
    }

    // Fold caret for groups, then the indent.
    let mut x = rect.left() + 30.0 + row.depth as f32 * INDENT;
    if row.is_group {
        let caret = egui::Rect::from_center_size(egui::pos2(x + 6.0, rect.center().y), egui::vec2(16.0, 22.0));
        let caret_resp = ui.interact(caret, resp.id.with("fold"), egui::Sense::click());
        p.text(
            caret.center(),
            egui::Align2::CENTER_CENTER,
            if row.expanded { ph::CARET_DOWN } else { ph::CARET_RIGHT },
            egui::FontId::proportional(12.0),
            theme::TEXT,
        );
        if caret_resp.clicked() {
            acts.push(LayerAct::ToggleFold(node.id));
        }
    }
    x += 14.0;

    // Thumbnails.
    let thumb = egui::Rect::from_min_size(egui::pos2(x, rect.center().y - THUMB / 2.0), egui::vec2(THUMB, THUMB));
    paint_thumb(&p, thumb, node, dim);
    x += THUMB + 4.0;
    if let Some(mask) = &node.mask {
        let m = egui::Rect::from_min_size(egui::pos2(x, rect.center().y - THUMB / 2.0), egui::vec2(THUMB, THUMB));
        paint_mask_thumb(&p, m, mask, dim);
        x += THUMB + 4.0;
    }
    x += 4.0;

    // Name (or the rename editor) with the blend summary beneath.
    let right = rect.right() - 8.0 - channel_bars_width(node);
    let text_rect = egui::Rect::from_min_max(
        egui::pos2(x, rect.top() + 3.0),
        egui::pos2(right.max(x + 20.0), rect.bottom() - 3.0),
    );
    let is_renaming = renaming.as_ref().is_some_and(|(id, _)| *id == node.id);
    if is_renaming {
        if let Some((_, draft)) = renaming.as_mut() {
            let edit_rect = egui::Rect::from_min_size(text_rect.min, egui::vec2(text_rect.width(), 20.0));
            let r = ui.put(edit_rect, egui::TextEdit::singleline(draft).desired_width(edit_rect.width()));
            if !r.has_focus() && !r.lost_focus() {
                r.request_focus();
            }
            if r.lost_focus() {
                let escape = ui.ctx().input(|i| i.key_pressed(egui::Key::Escape));
                acts.push(if escape {
                    LayerAct::CancelRename
                } else {
                    LayerAct::CommitRename(node.id, draft.clone())
                });
            }
        }
    } else {
        p.with_clip_rect(text_rect).text(
            text_rect.left_top() + egui::vec2(0.0, 2.0),
            egui::Align2::LEFT_TOP,
            &node.name,
            egui::FontId::proportional(12.5),
            text,
        );
        let summary = format!("{}  {}%", node.blend.name(), (node.opacity * 100.0).round() as i32);
        p.with_clip_rect(text_rect).text(
            text_rect.left_bottom() - egui::vec2(0.0, 2.0),
            egui::Align2::LEFT_BOTTOM,
            summary,
            egui::FontId::proportional(10.5),
            theme::TEXT_DIM,
        );
    }
    paint_channel_bars(&p, rect, node);

    // Interaction.
    if resp.clicked() {
        acts.push(LayerAct::Select(node.id));
    }
    if resp.double_clicked() {
        acts.push(LayerAct::BeginRename(node.id));
    }
    if resp.drag_started() {
        egui::DragAndDrop::set_payload(ui.ctx(), node.id);
        acts.push(LayerAct::Select(node.id));
    }
    resp.context_menu(|ui| layer_menu(ui, node, acts));

    // Drag feedback.
    let dragged = dragging?;
    let pos = ui.ctx().pointer_hover_pos()?;
    if !rect.contains(pos) {
        return None;
    }
    let frac = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
    let target = layer_dnd::drop_target(&doc, dragged, row, frac);
    let overlay = ui.painter();
    match target {
        Some(t) => {
            let inset = |depth: usize| rect.left() + 8.0 + depth as f32 * INDENT;
            let stroke = egui::Stroke::new(2.0, theme::ACCENT);
            match t.indicator {
                Indicator::Above(d) => {
                    overlay.hline(inset(d)..=rect.right() - 4.0, rect.top() + 1.0, stroke);
                }
                Indicator::Below(d) => {
                    overlay.hline(inset(d)..=rect.right() - 4.0, rect.bottom() - 1.0, stroke);
                }
                Indicator::Into => {
                    overlay.rect_stroke(rect.shrink(1.0), 2.0, stroke, egui::StrokeKind::Inside);
                }
            }
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
        None => ui.ctx().set_cursor_icon(egui::CursorIcon::NoDrop),
    }
    target
}

fn layer_menu(ui: &mut egui::Ui, node: &Node, acts: &mut Vec<LayerAct>) {
    let id = node.id;
    let mut item = |ui: &mut egui::Ui, label: &str, act: LayerAct| {
        if ui.button(label).clicked() {
            acts.push(act);
            ui.close();
        }
    };
    item(ui, "Rename", LayerAct::BeginRename(id));
    item(ui, "Duplicate", LayerAct::Duplicate(id));
    item(ui, "Group", LayerAct::Wrap(id));
    ui.separator();
    item(ui, "Move up", LayerAct::MoveUp(id));
    item(ui, "Move down", LayerAct::MoveDown(id));
    ui.separator();
    if node.mask.is_some() {
        item(ui, "Remove mask", LayerAct::RemoveMask(id));
    } else {
        item(ui, "Add white mask", LayerAct::AddMask(id, true));
        item(ui, "Add black mask", LayerAct::AddMask(id, false));
    }
    ui.separator();
    item(ui, "Delete", LayerAct::Delete(id));
}

/// A small flat toolbar button.
fn icon_button(ui: &mut egui::Ui, icon: &str, tip: &str, enabled: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 22.0), egui::Sense::click());
    let hot = enabled && resp.hovered();
    if hot {
        ui.painter().rect_filled(rect, 3.0, theme::CONTROL_HOVER);
    }
    let color = if !enabled {
        egui::Color32::from_gray(0x6e)
    } else if hot {
        theme::TEXT_STRONG
    } else {
        theme::TEXT
    };
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(16.0),
        color,
    );
    let resp = resp.on_hover_text(tip);
    if enabled {
        resp
    } else {
        resp.on_hover_cursor(egui::CursorIcon::Default).with_new_rect(egui::Rect::NOTHING)
    }
}

/// Which of the three channels a layer writes, as (label, active).
fn channels(node: &Node) -> Option<[(&'static str, bool); 3]> {
    match &node.kind {
        NodeKind::Paint => Some([("C", true), ("M", true), ("R", true)]),
        NodeKind::Fill(f) => Some([("C", f.base.is_some()), ("M", f.metallic.is_some()), ("R", f.roughness.is_some())]),
        NodeKind::Group { .. } => None,
    }
}

fn channel_bars_width(node: &Node) -> f32 {
    if channels(node).is_some() { 26.0 } else { 0.0 }
}

/// Orange ticks at the right edge: one per channel the layer touches, as in a channel indicator.
fn paint_channel_bars(p: &egui::Painter, rect: egui::Rect, node: &Node) {
    let Some(ch) = channels(node) else { return };
    let x = rect.right() - 20.0;
    for (i, (label, on)) in ch.into_iter().enumerate() {
        let y = rect.top() + 6.0 + i as f32 * 9.0;
        let bar = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(5.0, 6.0));
        p.rect_filled(bar, 1.0, if on { theme::ORANGE } else { egui::Color32::from_gray(0x44) });
        p.text(
            egui::pos2(x + 8.0, y + 3.0),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(8.5),
            if on { theme::TEXT } else { egui::Color32::from_gray(0x7a) },
        );
    }
}

fn checker(p: &egui::Painter, rect: egui::Rect) {
    p.rect_filled(rect, 2.0, egui::Color32::from_gray(0x50));
    let cell = rect.width() / 4.0;
    for ix in 0..4 {
        for iy in 0..4 {
            if (ix + iy) % 2 == 0 {
                let r = egui::Rect::from_min_size(rect.min + egui::vec2(ix as f32 * cell, iy as f32 * cell), egui::vec2(cell, cell));
                p.rect_filled(r, 0.0, egui::Color32::from_gray(0x3a));
            }
        }
    }
}

/// A stand-in preview of what the layer holds: a swatch for fills, a folder for groups, a checker
/// with a brush for paint layers.
fn paint_thumb(p: &egui::Painter, rect: egui::Rect, node: &Node, dim: bool) {
    let fade = |c: egui::Color32| if dim { c.gamma_multiply(0.45) } else { c };
    match &node.kind {
        NodeKind::Fill(f) => {
            // The fill color is stored in linear light; the swatch is drawn in sRGB.
            let srgb = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
            let [r, g, b] = f.base.unwrap_or(op_core::FALLBACK_BASE);
            p.rect_filled(rect, 2.0, fade(egui::Color32::from_rgb(srgb(r), srgb(g), srgb(b))));
        }
        NodeKind::Group { .. } => {
            p.rect_filled(rect, 2.0, fade(theme::CONTROL));
            p.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                ph::FOLDER_SIMPLE,
                egui::FontId::proportional(18.0),
                fade(theme::TEXT),
            );
        }
        NodeKind::Paint => {
            checker(p, rect);
            p.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                ph::PAINT_BRUSH,
                egui::FontId::proportional(16.0),
                fade(theme::TEXT),
            );
        }
    }
    p.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, theme::BORDER), egui::StrokeKind::Outside);
}

fn paint_mask_thumb(p: &egui::Painter, rect: egui::Rect, mask: &Mask, dim: bool) {
    let fade = |c: egui::Color32| if dim { c.gamma_multiply(0.45) } else { c };
    let base = if mask.init_white {
        egui::Color32::WHITE
    } else {
        egui::Color32::BLACK
    };
    p.rect_filled(rect, 2.0, fade(base));
    if mask.generator.is_some() {
        // Generated masks show a soft diagonal ramp to say "computed".
        let n = 6;
        for i in 0..n {
            let v = (i * 255 / (n - 1)) as u8;
            let w = rect.width() / n as f32;
            let r = egui::Rect::from_min_size(rect.min + egui::vec2(i as f32 * w, 0.0), egui::vec2(w, rect.height()));
            p.rect_filled(r, 0.0, fade(egui::Color32::from_gray(v)));
        }
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            ph::SPARKLE,
            egui::FontId::proportional(13.0),
            fade(theme::ORANGE),
        );
    }
    p.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0, if mask.enabled { theme::BORDER } else { theme::DANGER }),
        egui::StrokeKind::Outside,
    );
}
