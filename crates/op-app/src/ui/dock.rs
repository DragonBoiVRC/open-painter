//! Dockable panels: the set of tabs, the default arrangement, saving and restoring it.

use super::theme;
use egui_dock::{DockState, NodeIndex, Style, TabInteractionStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tab {
    Viewport,
    /// The active texture set laid out in UV space, for painting directly on the texture.
    Uv2d,
    /// Brush and paint settings ("Properties - Paint").
    Brush,
    /// The materials of the model; each is edited on its own.
    Sets,
    Layers,
    /// Properties of the selected layer.
    Properties,
    TextureSet,
    Bake,
    Display,
    Shelf,
}

impl Tab {
    pub const ALL: [Tab; 10] =
        [Tab::Viewport, Tab::Uv2d, Tab::Brush, Tab::Sets, Tab::Layers, Tab::Properties, Tab::TextureSet, Tab::Bake, Tab::Display, Tab::Shelf];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Viewport => "Viewport",
            Tab::Uv2d => "UV View",
            Tab::Brush => "Properties - Paint",
            Tab::Sets => "Texture Sets",
            Tab::Layers => "Layers",
            Tab::Properties => "Properties - Layer",
            Tab::TextureSet => "Texture Set Settings",
            Tab::Bake => "Bake Mesh Maps",
            Tab::Display => "Display Settings",
            Tab::Shelf => "Shelf",
        }
    }

    /// A shorter name for the tab header, where room is tight.
    pub fn tab_label(self) -> &'static str {
        match self {
            Tab::TextureSet => "Texture Set",
            Tab::Bake => "Bake",
            Tab::Display => "Display",
            other => other.title(),
        }
    }

    pub fn icon(self) -> &'static str {
        use egui_phosphor::regular as ph;
        match self {
            Tab::Viewport => ph::CUBE,
            Tab::Uv2d => ph::GRID_NINE,
            Tab::Brush => ph::PAINT_BRUSH,
            Tab::Sets => ph::SQUARES_FOUR,
            Tab::Layers => ph::STACK,
            Tab::Properties => ph::SLIDERS,
            Tab::TextureSet => ph::IMAGE,
            Tab::Bake => ph::SUN,
            Tab::Display => ph::EYE,
            Tab::Shelf => ph::GRID_FOUR,
        }
    }

    /// The 3D viewport is where the model is drawn and painted, so it can never be closed.
    pub fn closeable(self) -> bool {
        self != Tab::Viewport
    }
}

/// The arrangement a fresh install starts with: paint properties on the left, the 3D and UV views
/// in the middle with the shelf below them, layers and their properties on the right.
pub fn default_layout() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Viewport]);
    let tree = dock.main_surface_mut();
    let [center, _left] = tree.split_left(NodeIndex::root(), 0.19, vec![Tab::Brush]);
    let [center, right] = tree.split_right(center, 0.76, vec![Tab::Layers]);
    let [right, _sets] = tree.split_above(right, 0.16, vec![Tab::Sets]);
    tree.split_below(right, 0.36, vec![Tab::Properties, Tab::TextureSet, Tab::Bake, Tab::Display]);
    let [top, _shelf] = tree.split_below(center, 0.74, vec![Tab::Shelf]);
    // The 3D view and the 2D UV view side by side.
    tree.split_right(top, 0.5, vec![Tab::Uv2d]);
    dock
}

/// A layout is usable if the viewport is present exactly once and no tab appears twice.
pub fn is_valid(dock: &DockState<Tab>) -> bool {
    let mut seen = HashSet::new();
    for (_, tab) in dock.iter_all_tabs() {
        if !seen.insert(*tab) {
            return false;
        }
    }
    seen.contains(&Tab::Viewport)
}

pub fn to_json(dock: &DockState<Tab>) -> Option<String> {
    serde_json::to_string(dock).ok()
}

/// The dock keeps each panel's last on-screen rectangle as NaN until it is first drawn, and JSON has
/// no NaN: `serde_json` writes `null`, which it then refuses to read back as a number. Rectangles
/// are recomputed every frame, so a point whose coordinates are both `null` is restored as (0, 0).
fn repair_rects(value: &mut serde_json::Value) {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            if map.len() == 2 && map.get("x") == Some(&Value::Null) && map.get("y") == Some(&Value::Null) {
                map.insert("x".into(), 0.0.into());
                map.insert("y".into(), 0.0.into());
            } else {
                map.values_mut().for_each(repair_rects);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(repair_rects),
        _ => {}
    }
}

/// Parse a saved layout; anything unusable gives `None` so the caller falls back to the default.
pub fn from_json(text: &str) -> Option<DockState<Tab>> {
    let mut value: serde_json::Value = serde_json::from_str(text).ok()?;
    repair_rects(&mut value);
    let dock: DockState<Tab> = serde_json::from_value(value).ok()?;
    is_valid(&dock).then_some(dock)
}

pub fn load(path: &Path) -> Option<DockState<Tab>> {
    // A layout file is a few kilobytes; refuse anything absurd.
    if std::fs::metadata(path).ok()?.len() > 1 << 20 {
        return None;
    }
    from_json(&std::fs::read_to_string(path).ok()?)
}

/// Show `tab`: focus it if it is open, otherwise add it next to the focused panel.
pub fn open(dock: &mut DockState<Tab>, tab: Tab) {
    match dock.find_tab(&tab) {
        Some(path) => {
            let _ = dock.set_active_tab(path);
        }
        None => dock.push_to_focused_leaf(tab),
    }
}

pub fn close(dock: &mut DockState<Tab>, tab: Tab) {
    if tab.closeable() {
        if let Some(path) = dock.find_tab(&tab) {
            dock.remove_tab(path);
        }
    }
}

pub fn is_open(dock: &DockState<Tab>, tab: Tab) -> bool {
    dock.iter_all_tabs().any(|(_, t)| *t == tab)
}

/// Docking style matching the theme: flat gray tab bars, a blue line under the active tab.
pub fn style(ctx: &egui::Context) -> Style {
    let mut s = Style::from_egui(&ctx.global_style());
    s.dock_area_padding = None;
    s.main_surface_border_stroke = egui::Stroke::NONE;
    s.separator.color_idle = theme::BORDER;
    s.separator.color_hovered = theme::ACCENT;
    s.separator.color_dragged = theme::ACCENT;
    s.separator.width = 3.0;
    s.tab_bar.bg_fill = theme::CHROME;
    s.tab_bar.height = 24.0;
    s.tab_bar.hline_color = theme::BORDER;
    s.tab_bar.corner_radius = egui::CornerRadius::ZERO;
    s.tab.hline_below_active_tab_name = false;
    s.tab.tab_body.bg_fill = theme::PANEL;
    s.tab.tab_body.stroke = egui::Stroke::NONE;
    s.tab.tab_body.inner_margin = egui::Margin::same(6);
    s.tab.tab_body.corner_radius = egui::CornerRadius::ZERO;
    let tab = |text: egui::Color32, bg: egui::Color32, outline: egui::Color32| TabInteractionStyle {
        outline_color: outline,
        corner_radius: egui::CornerRadius::ZERO,
        bg_fill: bg,
        text_color: text,
    };
    s.tab.inactive = tab(theme::TEXT_DIM, theme::CHROME, egui::Color32::TRANSPARENT);
    s.tab.hovered = tab(theme::TEXT_STRONG, theme::CONTROL, egui::Color32::TRANSPARENT);
    s.tab.active = tab(theme::TEXT_STRONG, theme::PANEL, theme::BORDER);
    s.tab.focused = tab(theme::TEXT_STRONG, theme::PANEL, theme::ACCENT);
    s.tab.inactive_with_kb_focus = s.tab.inactive.clone();
    s.tab.active_with_kb_focus = s.tab.active.clone();
    s.tab.focused_with_kb_focus = s.tab.focused.clone();
    s.buttons.close_tab_color = theme::TEXT_DIM;
    s.buttons.close_tab_active_color = theme::TEXT_STRONG;
    s.overlay.selection_color = egui::Color32::from_rgba_unmultiplied(0x2e, 0x8b, 0xea, 90);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tabs(d: &DockState<Tab>) -> Vec<Tab> {
        let mut v: Vec<Tab> = d.iter_all_tabs().map(|(_, t)| *t).collect();
        v.sort_by_key(|t| t.title());
        v
    }

    #[test]
    fn the_default_layout_has_every_tab_exactly_once() {
        let d = default_layout();
        assert!(is_valid(&d));
        let mut want = Tab::ALL.to_vec();
        want.sort_by_key(|t| t.title());
        assert_eq!(tabs(&d), want);
    }

    #[test]
    fn layouts_survive_a_save_and_load() {
        let d = default_layout();
        let json = to_json(&d).unwrap();
        let back = from_json(&json).expect("a layout we wrote must load");
        assert_eq!(tabs(&back), tabs(&d));
        // The arrangement is identical; only never-drawn rectangles changed (NaN -> 0), after which
        // saving and loading again is a fixed point.
        let again = to_json(&back).unwrap();
        assert_eq!(to_json(&from_json(&again).unwrap()).unwrap(), again, "second round trip is stable");
    }

    /// Regression: a freshly built layout contains NaN rectangles, which JSON writes as `null`.
    #[test]
    fn a_layout_with_never_drawn_panels_still_round_trips() {
        let json = to_json(&default_layout()).unwrap();
        assert!(json.contains("null"), "the default layout really does contain null rectangles");
        assert!(from_json(&json).is_some());
        // A legitimate null elsewhere must not be silently turned into a number.
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let mut count = 0;
        fn visit(v: &serde_json::Value, count: &mut usize) {
            match v {
                serde_json::Value::Object(m) => {
                    if m.get("x").is_some_and(|x| x.is_null()) {
                        *count += 1;
                    }
                    m.values().for_each(|c| visit(c, count));
                }
                serde_json::Value::Array(a) => a.iter().for_each(|c| visit(c, count)),
                _ => {}
            }
        }
        visit(&v, &mut count);
        assert!(count > 0);
        repair_rects(&mut v);
        count = 0;
        visit(&v, &mut count);
        assert_eq!(count, 0, "every null point was repaired");
    }

    #[test]
    fn broken_or_unusable_layouts_are_rejected() {
        assert!(from_json("").is_none());
        assert!(from_json("not json").is_none());
        assert!(from_json("{}").is_none());
        // Valid JSON for a layout without the viewport.
        let no_viewport = DockState::new(vec![Tab::Layers, Tab::Brush]);
        assert!(from_json(&to_json(&no_viewport).unwrap()).is_none(), "the viewport is mandatory");
        // The same tab twice.
        let mut twice = default_layout();
        twice.push_to_first_leaf(Tab::Layers);
        assert!(!is_valid(&twice));
        assert!(from_json(&to_json(&twice).unwrap()).is_none());
    }

    #[test]
    fn corrupting_a_valid_file_never_panics() {
        let json = to_json(&default_layout()).unwrap();
        for cut in (0..json.len()).step_by(7) {
            let _ = from_json(&json[..cut]);
        }
        let flipped: String = json.chars().enumerate().map(|(i, c)| if i % 11 == 3 { '#' } else { c }).collect();
        assert!(from_json(&flipped).is_none());
    }

    #[test]
    fn opening_and_closing_panels() {
        let mut d = default_layout();
        assert!(is_open(&d, Tab::Bake));
        close(&mut d, Tab::Bake);
        assert!(!is_open(&d, Tab::Bake));
        assert!(is_valid(&d));
        open(&mut d, Tab::Bake);
        assert!(is_open(&d, Tab::Bake));
        open(&mut d, Tab::Bake); // already open: focus, do not duplicate
        assert!(is_valid(&d));
        close(&mut d, Tab::Viewport);
        assert!(is_open(&d, Tab::Viewport), "the viewport cannot be closed");
    }

    #[test]
    fn every_tab_has_a_title_and_an_icon() {
        let mut titles = HashSet::new();
        for t in Tab::ALL {
            assert!(!t.title().is_empty() && !t.icon().is_empty());
            assert!(titles.insert(t.title()), "duplicate title {}", t.title());
        }
        assert_eq!(Tab::ALL.iter().filter(|t| !t.closeable()).count(), 1);
    }
}

