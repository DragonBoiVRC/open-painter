//! "Shelf": a searchable browser of reusable assets — smart materials, smart masks and brush presets.
//! Double-click a tile to use it; drag smart materials and masks onto the layer stack.

use crate::ui::theme;
use crate::{App, GenAction, builtin_kinds};
use egui_phosphor::regular as ph;
use op_core::smart::SmartMaterial;
use op_core::{Generator, NodeKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Category {
    Materials,
    Masks,
    Brushes,
}

impl Category {
    const ALL: [Category; 3] = [Category::Materials, Category::Masks, Category::Brushes];

    fn label(self) -> &'static str {
        match self {
            Category::Materials => "Smart materials",
            Category::Masks => "Smart masks",
            Category::Brushes => "Brushes",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Category::Materials => ph::CUBE,
            Category::Masks => ph::CIRCLE_HALF,
            Category::Brushes => ph::PAINT_BRUSH,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushPreset {
    pub radius_pts: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub spacing: f32,
}

/// What a tile stands for.
#[derive(Clone, Debug)]
pub enum Asset {
    /// Index into the app's smart material list.
    Material(usize),
    Mask(Generator),
    Brush(BrushPreset),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub category: Category,
    pub name: String,
    pub asset: Asset,
    /// sRGB color for the tile, when the asset has an obvious one.
    pub swatch: Option<[u8; 3]>,
}

pub struct ShelfState {
    pub category: Category,
    pub query: String,
    pub tile: f32,
    pub selected: Option<(Category, String)>,
    pub save_name: String,
}

impl Default for ShelfState {
    fn default() -> Self {
        Self {
            category: Category::Materials,
            query: String::new(),
            tile: 76.0,
            selected: None,
            save_name: String::new(),
        }
    }
}

const BRUSHES: [(&str, BrushPreset); 5] = [
    (
        "Soft round",
        BrushPreset {
            radius_pts: 40.0,
            hardness: 0.1,
            opacity: 1.0,
            spacing: 0.15,
        },
    ),
    (
        "Hard round",
        BrushPreset {
            radius_pts: 24.0,
            hardness: 1.0,
            opacity: 1.0,
            spacing: 0.1,
        },
    ),
    (
        "Airbrush",
        BrushPreset {
            radius_pts: 70.0,
            hardness: 0.0,
            opacity: 0.25,
            spacing: 0.08,
        },
    ),
    (
        "Detail",
        BrushPreset {
            radius_pts: 6.0,
            hardness: 0.9,
            opacity: 1.0,
            spacing: 0.1,
        },
    ),
    (
        "Wash",
        BrushPreset {
            radius_pts: 120.0,
            hardness: 0.3,
            opacity: 0.4,
            spacing: 0.2,
        },
    ),
];

/// The first fill color in a tree, as an sRGB byte triple.
fn first_fill_color(nodes: &[op_core::Node]) -> Option<[u8; 3]> {
    for n in nodes {
        match &n.kind {
            NodeKind::Fill(f) => {
                if let Some(c) = f.base {
                    return Some(c.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
                }
            }
            NodeKind::Group { children } => {
                if let Some(c) = first_fill_color(children) {
                    return Some(c);
                }
            }
            NodeKind::Paint => {}
        }
    }
    None
}

/// Every asset the shelf can show, in display order.
pub fn catalog(materials: &[SmartMaterial], custom: &[Generator]) -> Vec<Entry> {
    let mut out = Vec::new();
    for (i, m) in materials.iter().enumerate() {
        out.push(Entry {
            category: Category::Materials,
            name: m.name.clone(),
            asset: Asset::Material(i),
            swatch: first_fill_color(&m.nodes),
        });
    }
    for k in builtin_kinds() {
        out.push(Entry {
            category: Category::Masks,
            name: op_proc::label(&k),
            asset: Asset::Mask(Generator::new(k)),
            swatch: None,
        });
    }
    for c in custom {
        out.push(Entry {
            category: Category::Masks,
            name: op_proc::label(&c.kind),
            asset: Asset::Mask(c.clone()),
            swatch: None,
        });
    }
    for (name, p) in BRUSHES {
        out.push(Entry {
            category: Category::Brushes,
            name: name.into(),
            asset: Asset::Brush(p),
            swatch: None,
        });
    }
    out
}

/// Case-insensitive match: every word of the query must occur in the name.
pub fn matches(query: &str, name: &str) -> bool {
    let name = name.to_lowercase();
    query.split_whitespace().all(|w| name.contains(&w.to_lowercase()))
}

/// The entries to show for a category and search text. A non-empty search looks in every category.
pub fn visible<'a>(entries: &'a [Entry], category: Category, query: &str) -> Vec<&'a Entry> {
    let searching = !query.trim().is_empty();
    entries
        .iter()
        .filter(|e| (searching || e.category == category) && matches(query, &e.name))
        .collect()
}

impl App {
    pub(crate) fn shelf_panel(&mut self, ui: &mut egui::Ui) {
        let entries = catalog(&self.smart_materials, &self.custom_generators);
        let mut chosen: Option<Asset> = None;

        ui.horizontal_top(|ui| {
            // Category list.
            ui.allocate_ui_with_layout(
                egui::vec2(148.0, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    for c in Category::ALL {
                        let on = self.shelf.category == c && self.shelf.query.trim().is_empty();
                        let count = entries.iter().filter(|e| e.category == c).count();
                        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 24.0), egui::Sense::click());
                        let p = ui.painter();
                        if on {
                            p.rect_filled(rect, 2.0, theme::ACCENT_DIM);
                        } else if resp.hovered() {
                            p.rect_filled(rect, 2.0, theme::CONTROL);
                        }
                        let col = if on { theme::TEXT_STRONG } else { theme::TEXT };
                        p.text(
                            rect.left_center() + egui::vec2(8.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            c.icon(),
                            egui::FontId::proportional(14.0),
                            col,
                        );
                        p.text(
                            rect.left_center() + egui::vec2(28.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            c.label(),
                            egui::FontId::proportional(12.5),
                            col,
                        );
                        p.text(
                            rect.right_center() - egui::vec2(8.0, 0.0),
                            egui::Align2::RIGHT_CENTER,
                            count.to_string(),
                            egui::FontId::proportional(10.5),
                            theme::TEXT_DIM,
                        );
                        if resp.clicked() {
                            self.shelf.category = c;
                            self.shelf.query.clear();
                        }
                    }
                },
            );
            ui.separator();

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(ph::MAGNIFYING_GLASS).color(theme::TEXT_DIM));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.shelf.query)
                            .hint_text("Search all assets")
                            .desired_width(200.0),
                    );
                    if !self.shelf.query.is_empty() && ui.small_button(ph::X).on_hover_text("Clear search").clicked() {
                        self.shelf.query.clear();
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().slider_width = 80.0;
                        ui.add(egui::Slider::new(&mut self.shelf.tile, 56.0..=120.0).show_value(false))
                            .on_hover_text("Tile size");
                    });
                });
                let shown = visible(&entries, self.shelf.category, &self.shelf.query);
                let tile = self.shelf.tile;
                let footer = if self.shelf.category == Category::Materials && self.shelf.query.trim().is_empty() {
                    30.0
                } else {
                    0.0
                };
                egui::Frame::NONE
                    .fill(theme::LIST)
                    .stroke(egui::Stroke::new(1.0, theme::BORDER))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("shelf_grid")
                            .auto_shrink([false, false])
                            .max_height((ui.available_height() - footer - 14.0).max(40.0))
                            .show(ui, |ui| {
                                if shown.is_empty() {
                                    super::hint(ui, "Nothing matches.");
                                }
                                ui.horizontal_wrapped(|ui| {
                                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                                    for e in &shown {
                                        if self.shelf_tile(ui, e, tile) {
                                            chosen = Some(e.asset.clone());
                                        }
                                    }
                                });
                            });
                    });
                for w in &self.definition_warnings {
                    ui.colored_label(theme::WARN, format!("Skipped: {w}"));
                }
                if footer > 0.0 {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.shelf.save_name)
                                .hint_text("name")
                                .desired_width(140.0),
                        );
                        let ok = !self.shelf.save_name.trim().is_empty();
                        if ui
                            .add_enabled(ok, egui::Button::new("Save selected layer as smart material"))
                            .clicked()
                        {
                            self.smart_ui.save_name = self.shelf.save_name.clone();
                            self.save_smart_material();
                            if self.smart_ui.message.starts_with("Saved") {
                                self.shelf.save_name.clear();
                            }
                            self.status = self.smart_ui.message.clone();
                        }
                    });
                }
            });
        });

        if let Some(asset) = chosen {
            self.use_asset(asset);
        }
    }

    /// Draw one tile; true when it was double-clicked.
    fn shelf_tile(&mut self, ui: &mut egui::Ui, e: &Entry, size: f32) -> bool {
        let label_h = 30.0;
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size + label_h), egui::Sense::click_and_drag());
        let key = (e.category, e.name.clone());
        let selected = self.shelf.selected.as_ref() == Some(&key);
        let art = egui::Rect::from_min_size(rect.min, egui::vec2(size, size));
        let p = ui.painter();
        if selected || resp.hovered() {
            p.rect_filled(rect.expand(2.0), 3.0, if selected { theme::ACCENT_DIM } else { theme::CONTROL });
        }
        let base = match e.swatch {
            Some([r, g, b]) => egui::Color32::from_rgb(r, g, b),
            None => match e.category {
                Category::Materials => egui::Color32::from_gray(0x55),
                Category::Masks => egui::Color32::from_gray(0x20),
                Category::Brushes => egui::Color32::from_gray(0x3a),
            },
        };
        p.rect_filled(art, 2.0, base);
        if let Asset::Brush(b) = &e.asset {
            // A dab with the preset's softness, so brushes can be told apart at a glance.
            let r = size * 0.18 + size * 0.22 * (b.radius_pts / 120.0).clamp(0.0, 1.0);
            for i in (1..=8).rev() {
                let f = i as f32 / 8.0;
                let edge = 1.0 - b.hardness;
                let a = if f < 1.0 - edge {
                    1.0
                } else {
                    ((1.0 - f) / edge.max(1e-3)).clamp(0.0, 1.0)
                };
                let alpha = (255.0 * (0.15 + 0.85 * a) * b.opacity.max(0.3)) as u8;
                p.circle_filled(art.center(), r * f.max(0.2) + 2.0, egui::Color32::from_white_alpha(alpha / 6));
            }
            p.circle_filled(
                art.center(),
                r * 0.35,
                egui::Color32::from_white_alpha((200.0 * b.opacity.max(0.4)) as u8),
            );
        } else {
            let tint = if e.swatch.is_some() {
                egui::Color32::from_white_alpha(120)
            } else {
                theme::TEXT_DIM
            };
            p.text(
                art.center(),
                egui::Align2::CENTER_CENTER,
                e.category.icon(),
                egui::FontId::proportional(size * 0.4),
                tint,
            );
        }
        p.rect_stroke(art, 2.0, egui::Stroke::new(1.0, theme::BORDER), egui::StrokeKind::Outside);
        let text_rect = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, size + 3.0), egui::vec2(size, label_h - 3.0));
        let galley = ui.painter().layout(
            e.name.clone(),
            egui::FontId::proportional(11.0),
            if selected { theme::TEXT_STRONG } else { theme::TEXT },
            size,
        );
        ui.painter().with_clip_rect(text_rect).galley(text_rect.min, galley, theme::TEXT);

        if resp.clicked() {
            self.shelf.selected = Some(key);
        }
        if resp.drag_started() && !matches!(e.asset, Asset::Brush(_)) {
            egui::DragAndDrop::set_payload(ui.ctx(), e.asset.clone());
        }
        let tip = match &e.asset {
            Asset::Material(_) => "Double-click to add to the layer stack, or drag it there",
            Asset::Mask(_) => "Double-click to give the selected layer this mask, or drag it onto a layer",
            Asset::Brush(_) => "Double-click to use this brush",
        };
        resp.on_hover_text(format!("{}\n{tip}", e.name)).double_clicked()
    }

    /// Apply an asset to the current selection.
    pub(crate) fn use_asset(&mut self, asset: Asset) {
        match asset {
            Asset::Material(i) => {
                self.add_smart_material(i);
                self.status = self.smart_ui.message.clone();
            }
            Asset::Mask(g) => match self.selected.filter(|id| self.doc.find(*id).is_some()) {
                Some(id) => {
                    self.apply_gen_action(id, GenAction::NewMask(g));
                    self.status = "Added a generated mask. It needs baked maps: use Bake to compute them.".into();
                }
                None => self.status = "Select a layer first, then use the mask.".into(),
            },
            Asset::Brush(b) => {
                self.brush.radius_pts = b.radius_pts;
                self.brush.hardness = b.hardness;
                self.brush.opacity = b.opacity;
                self.brush.spacing = b.spacing;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{FillParams, Node};

    fn material(name: &str, color: Option<[f32; 3]>) -> SmartMaterial {
        let mut d = op_core::Document::default();
        let kind = NodeKind::Fill(FillParams {
            base: color,
            metallic: None,
            roughness: None,
        });
        let id = d.add(None, None, name, kind);
        let node: Node = d.find(id).unwrap().clone();
        SmartMaterial {
            name: name.into(),
            nodes: vec![node],
        }
    }

    #[test]
    fn search_requires_every_word_in_any_case() {
        assert!(matches("", "Anything"));
        assert!(matches("worn METAL", "Worn painted metal"));
        assert!(!matches("worn rust", "Worn painted metal"));
        assert!(matches("  edge  ", "Edge wear"));
    }

    #[test]
    fn the_catalog_lists_materials_masks_and_brushes() {
        let c = catalog(&[material("Steel", Some([0.5, 0.5, 0.6]))], &[]);
        let count = |cat| c.iter().filter(|e| e.category == cat).count();
        assert_eq!(count(Category::Materials), 1);
        assert_eq!(count(Category::Masks), builtin_kinds().len());
        assert_eq!(count(Category::Brushes), BRUSHES.len());
        assert_eq!(c[0].swatch, Some([128, 128, 153]));
    }

    #[test]
    fn a_swatch_is_found_inside_groups_and_absent_without_a_fill() {
        let mut d = op_core::Document::default();
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        d.add(Some(g), None, "p", NodeKind::Paint);
        d.add(
            Some(g),
            None,
            "f",
            NodeKind::Fill(FillParams {
                base: Some([1.0, 0.0, 0.0]),
                metallic: None,
                roughness: None,
            }),
        );
        assert_eq!(first_fill_color(&d.nodes), Some([255, 0, 0]));
        let none = material("x", None);
        assert_eq!(first_fill_color(&none.nodes), None);
    }

    #[test]
    fn a_category_filters_but_a_search_looks_everywhere() {
        let c = catalog(&[material("Edge steel", None)], &[]);
        let names = |v: Vec<&Entry>| v.iter().map(|e| e.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(visible(&c, Category::Materials, "")), ["Edge steel"]);
        let found = names(visible(&c, Category::Brushes, "edge"));
        assert!(found.contains(&"Edge steel".to_string()) && found.contains(&"Edge wear".to_string()));
        assert!(!found.contains(&"Soft round".to_string()));
        assert!(visible(&c, Category::Masks, "zzz").is_empty());
    }

    #[test]
    fn brush_presets_are_sane() {
        for (name, b) in BRUSHES {
            assert!(!name.is_empty());
            assert!((1.0..=500.0).contains(&b.radius_pts), "{name}");
            assert!((0.0..=1.0).contains(&b.hardness) && (0.0..=1.0).contains(&b.opacity), "{name}");
            assert!(b.spacing > 0.0, "{name} needs positive spacing");
        }
    }
}
