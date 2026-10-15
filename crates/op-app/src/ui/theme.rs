//! Visual theme: a compact dark interface in the spirit of professional texturing tools.
//!
//! Neutral grays for chrome and panels, a darker inset for lists and fields, one blue accent for
//! selection and an orange accent for channel indicators. Colors are original constants (not
//! copied from any product); the tests below keep the text readable.

use egui::{Color32, Context, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

const fn gray(v: u8) -> Color32 {
    Color32::from_rgb(v, v, v)
}

/// Title bars, tab bars, toolbars.
pub const CHROME: Color32 = gray(0x38);
/// Panel bodies.
pub const PANEL: Color32 = gray(0x30);
/// Inset areas: lists, text fields, the asset grid.
pub const LIST: Color32 = gray(0x26);
/// Raised controls (buttons, tiles).
pub const CONTROL: Color32 = gray(0x44);
pub const CONTROL_HOVER: Color32 = gray(0x52);
pub const BORDER: Color32 = gray(0x1b);

pub const TEXT: Color32 = gray(0xd0);
pub const TEXT_STRONG: Color32 = gray(0xf2);
pub const TEXT_DIM: Color32 = gray(0xa8);

pub const ACCENT: Color32 = Color32::from_rgb(0x2e, 0x8b, 0xea);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x1f, 0x5f, 0xa0);
/// Channel indicators on layers.
#[allow(dead_code)] // used by the layer rows
pub const ORANGE: Color32 = Color32::from_rgb(0xe8, 0x84, 0x2c);
pub const WARN: Color32 = Color32::from_rgb(0xe6, 0xb4, 0x3c);
pub const DANGER: Color32 = Color32::from_rgb(0xe0, 0x5c, 0x58);

/// Install the fonts (including the icon font) once at startup.
pub fn install_fonts(ctx: &Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

/// Apply colors, spacing and text sizes.
pub fn apply(ctx: &Context) {
    let mut v = Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = CHROME;
    v.extreme_bg_color = LIST;
    v.faint_bg_color = gray(0x2c);
    v.code_bg_color = LIST;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_corner_radius = CornerRadius::same(3);
    v.menu_corner_radius = CornerRadius::same(3);
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = WARN;
    v.error_fg_color = DANGER;
    v.slider_trailing_fill = true;
    v.striped = false;

    let radius = CornerRadius::same(2);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = CHROME;
    w.noninteractive.weak_bg_fill = CHROME;
    w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    w.noninteractive.corner_radius = radius;

    w.inactive.bg_fill = CONTROL;
    w.inactive.weak_bg_fill = CONTROL;
    w.inactive.bg_stroke = Stroke::new(1.0, gray(0x2a));
    w.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    w.inactive.corner_radius = radius;

    w.hovered.bg_fill = CONTROL_HOVER;
    w.hovered.weak_bg_fill = CONTROL_HOVER;
    w.hovered.bg_stroke = Stroke::new(1.0, gray(0x6a));
    w.hovered.fg_stroke = Stroke::new(1.0, TEXT_STRONG);
    w.hovered.corner_radius = radius;

    w.active.bg_fill = ACCENT_DIM;
    w.active.weak_bg_fill = ACCENT_DIM;
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    w.active.fg_stroke = Stroke::new(1.0, TEXT_STRONG);
    w.active.corner_radius = radius;

    w.open = w.active;
    ctx.set_visuals(v);

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(6.0, 2.0);
        style.spacing.interact_size.y = 20.0;
        style.spacing.slider_width = 110.0;
        style.spacing.window_margin = egui::Margin::same(8);
        style.spacing.menu_margin = egui::Margin::same(4);
        style.spacing.scroll = egui::style::ScrollStyle::thin();
        style.text_styles = [
            (TextStyle::Small, FontId::new(11.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(12.5, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(12.5, FontFamily::Proportional)),
            (TextStyle::Heading, FontId::new(13.5, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        ]
        .into();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG relative luminance of an sRGB color.
    fn luminance(c: Color32) -> f32 {
        let lin = |v: u8| {
            let v = v as f32 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
    }

    fn contrast(a: Color32, b: Color32) -> f32 {
        let (la, lb) = (luminance(a), luminance(b));
        (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
    }

    #[test]
    fn body_text_is_readable_on_every_surface() {
        for (name, bg) in [("panel", PANEL), ("chrome", CHROME), ("list", LIST), ("control", CONTROL), ("hovered control", CONTROL_HOVER)] {
            let c = contrast(TEXT, bg);
            assert!(c >= 4.5, "text on {name}: {c:.2}:1 is below 4.5:1");
        }
        assert!(contrast(TEXT_STRONG, PANEL) >= 7.0, "headings should clear 7:1");
    }

    #[test]
    fn secondary_text_stays_legible() {
        for (name, bg) in [("panel", PANEL), ("chrome", CHROME), ("list", LIST)] {
            let c = contrast(TEXT_DIM, bg);
            assert!(c >= 4.5, "dim text on {name}: {c:.2}:1");
        }
    }

    #[test]
    fn accents_are_distinguishable_from_the_surfaces_they_sit_on() {
        // UI components need 3:1 (WCAG 1.4.11).
        for (name, color) in [("accent", ACCENT), ("orange", ORANGE), ("warning", WARN), ("danger", DANGER)] {
            for (bg_name, bg) in [("panel", PANEL), ("list", LIST), ("chrome", CHROME)] {
                let c = contrast(color, bg);
                assert!(c >= 3.0, "{name} on {bg_name}: {c:.2}:1");
            }
        }
        // Selected text sits on the dim accent fill.
        assert!(contrast(TEXT_STRONG, ACCENT_DIM) >= 4.5);
    }

    #[test]
    fn surfaces_form_a_clear_depth_order() {
        // Inset lists are darker than panels, which are darker than chrome and controls.
        assert!(luminance(LIST) < luminance(PANEL));
        assert!(luminance(PANEL) < luminance(CHROME));
        assert!(luminance(CHROME) < luminance(CONTROL));
        assert!(luminance(CONTROL) < luminance(CONTROL_HOVER));
        assert!(luminance(BORDER) < luminance(LIST));
    }
}
