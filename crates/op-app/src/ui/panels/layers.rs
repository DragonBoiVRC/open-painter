//! "Layers": the layer stack list and the buttons that edit its structure.

use crate::{Act, App};
use op_core::{FillParams, Node, NodeId, NodeKind};

impl App {
    fn layer_rows(ui: &mut egui::Ui, nodes: &[Node], depth: usize, selected: Option<NodeId>, acts: &mut Vec<Act>) {
        for n in nodes.iter().rev() {
            ui.horizontal(|ui| {
                ui.add_space(depth as f32 * 14.0);
                let mut vis = n.visible;
                if ui.checkbox(&mut vis, "").changed() {
                    acts.push(Act::Visible(n.id, vis));
                }
                let tag = match n.kind {
                    NodeKind::Paint => "[P]",
                    NodeKind::Fill(_) => "[F]",
                    NodeKind::Group { .. } => "[G]",
                };
                let mask = if n.mask.is_some() { " (mask)" } else { "" };
                if ui.selectable_label(selected == Some(n.id), format!("{tag} {}{mask}", n.name)).clicked() {
                    acts.push(Act::Select(n.id));
                }
            });
            if let NodeKind::Group { children } = &n.kind {
                Self::layer_rows(ui, children, depth + 1, selected, acts);
            }
        }
    }

    pub(crate) fn layers_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("+ Paint").clicked() {
                self.add_node("Paint layer", NodeKind::Paint);
            }
            if ui.button("+ Fill").clicked() {
                self.add_node(
                    "Fill layer",
                    NodeKind::Fill(FillParams { base: Some([0.5, 0.5, 0.5]), metallic: None, roughness: None }),
                );
            }
            if ui.button("+ Group").clicked() {
                self.add_node("Group", NodeKind::Group { children: Vec::new() });
            }
        });

        let mut acts = Vec::new();
        egui::ScrollArea::vertical().id_salt("layer_list").max_height((ui.available_height() - 70.0).max(80.0)).show(ui, |ui| {
            Self::layer_rows(ui, &self.doc.nodes, 0, self.selected, &mut acts);
        });
        for a in acts {
            match a {
                Act::Select(id) => self.selected = Some(id),
                Act::Visible(id, v) => self.edit(|d| {
                    if let Some(n) = d.find_mut(id) {
                        n.visible = v;
                    }
                }),
            }
        }

        let Some(id) = self.selected.filter(|id| self.doc.find(*id).is_some()) else {
            ui.label("No layer selected.");
            return;
        };

        ui.horizontal_wrapped(|ui| {
            if ui.button("Up").clicked() {
                self.edit(|d| {
                    d.move_up(id);
                });
            }
            if ui.button("Down").clicked() {
                self.edit(|d| {
                    d.move_down(id);
                });
            }
            if ui.button("Into group above").clicked() {
                self.edit(|d| {
                    d.indent(id);
                });
            }
            if ui.button("Out of group").clicked() {
                self.edit(|d| {
                    d.outdent(id);
                });
            }
            if ui.button("Delete").clicked() {
                self.edit(|d| {
                    d.remove(id);
                });
                self.fix_selection();
            }
        });
    }
}
