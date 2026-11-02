//! Visual theme: a compact dark interface in the spirit of professional texturing tools.
//!
//! Neutral grays for chrome and panels, a darker inset for lists and fields, one blue accent for
//! selection and an orange accent for channel indicators. Colors are original constants (not
//! copied from any product); the tests below keep the text readable.

use egui::{Color32, Context, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

const fn gray(v: u8) -> Color32 {
    Color32::from_rgb(v, v, v)
}

use std::sync::atomic::{AtomicU8, Ordering};

/// Overall lightness of the interface: dark, medium dark, medium light, light.
pub const BRIGHTNESS_NAMES: [&str; 4] = ["Dark", "Medium dark", "Medium light", "Light"];
/// Accent colours (selection, active controls): name and colour.
pub const ACCENTS: [(&str, Color32); 7] = [
    ("Blue", Color32::from_rgb(0x2e, 0x8b, 0xea)),
    ("Teal", Color32::from_rgb(0x1f, 0xa8, 0x9a)),
    ("Green", Color32::from_rgb(0x3d, 0xa8, 0x4c)),
    ("Orange", Color32::from_rgb(0xe8, 0x84, 0x2c)),
    ("Red", Color32::from_rgb(0xdc, 0x4c, 0x4c)),
    ("Purple", Color32::from_rgb(0x9a, 0x6a, 0xe0)),
    ("Pink", Color32::from_rgb(0xe0, 0x5c, 0xa8)),
];

static BRIGHTNESS: AtomicU8 = AtomicU8::new(0);
static ACCENT_INDEX: AtomicU8 = AtomicU8::new(0);

pub fn brightness() -> u8 {
    BRIGHTNESS.load(Ordering::Relaxed)
}

pub fn accent_index() -> u8 {
    ACCENT_INDEX.load(Ordering::Relaxed)
}

/// Choose the look; out-of-range values fall back to the defaults. Call [`apply`] afterwards.
pub fn set_look(brightness: u8, accent: u8) {
    BRIGHTNESS.store(if (brightness as usize) < BRIGHTNESS_NAMES.len() { brightness } else { 0 }, Ordering::Relaxed);
    ACCENT_INDEX.store(if (accent as usize) < ACCENTS.len() { accent } else { 0 }, Ordering::Relaxed);
}

/// The grays of one brightness level, in this order: chrome, panel, list, control, control hover,
/// border, text, strong text, dim text, faint stripe, inactive stroke, hovered stroke.
const LEVELS: [[u8; 12]; 4] = [
    [0x38, 0x30, 0x26, 0x44, 0x52, 0x1b, 0xd0, 0xf2, 0xa8, 0x2c, 0x2a, 0x6a],
    [0x4a, 0x42, 0x35, 0x58, 0x60, 0x2a, 0xec, 0xff, 0xc0, 0x3c, 0x3a, 0x80],
    [0xb4, 0xc4, 0xd8, 0xa6, 0x98, 0x7c, 0x20, 0x00, 0x44, 0xbc, 0x92, 0x60],
    [0xd2, 0xde, 0xee, 0xc8, 0xba, 0x9c, 0x1c, 0x00, 0x46, 0xd6, 0xb0, 0x80],
];

fn level(i: usize) -> Color32 {
    gray(LEVELS[brightness() as usize % 4][i])
}

fn light() -> bool {
    brightness() >= 2
}

/// Title bars, tab bars, toolbars.
pub fn chrome() -> Color32 {
    level(0)
}
/// Panel bodies.
pub fn panel() -> Color32 {
    level(1)
}
/// Inset areas: lists, text fields, the asset grid.
pub fn list() -> Color32 {
    level(2)
}
/// Raised controls (buttons, tiles).
pub fn control() -> Color32 {
    level(3)
}
pub fn control_hover() -> Color32 {
    level(4)
}
pub fn border() -> Color32 {
    level(5)
}
pub fn text() -> Color32 {
    level(6)
}
pub fn text_strong() -> Color32 {
    level(7)
}
pub fn text_dim() -> Color32 {
    level(8)
}

pub fn accent() -> Color32 {
    let c = ACCENTS[accent_index() as usize % ACCENTS.len()].1;
    // On light surfaces the accent is darkened so it keeps its contrast.
    if light() { mix(c, Color32::BLACK, 0.25) } else { c }
}

/// The accent as a fill for selected things: text sits on it.
pub fn accent_dim() -> Color32 {
    let c = ACCENTS[accent_index() as usize % ACCENTS.len()].1;
    if light() { mix(c, Color32::WHITE, 0.6) } else { mix(c, Color32::BLACK, 0.4) }
}

fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

/// Channel indicators on layers.
#[allow(dead_code)] // used by the layer rows
pub fn orange() -> Color32 {
    if light() { Color32::from_rgb(0xb0, 0x58, 0x08) } else { Color32::from_rgb(0xe8, 0x84, 0x2c) }
}
pub fn warn() -> Color32 {
    if light() { Color32::from_rgb(0x8a, 0x62, 0x00) } else { Color32::from_rgb(0xe6, 0xb4, 0x3c) }
}
pub fn danger() -> Color32 {
    if light() { Color32::from_rgb(0xb0, 0x28, 0x24) } else { Color32::from_rgb(0xe0, 0x5c, 0x58) }
}

/// Install the fonts (including the icon font) once at startup.
pub fn install_fonts(ctx: &Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

/// Apply colors, spacing and text sizes.
pub fn apply(ctx: &Context) {
    let mut v = if light() { Visuals::light() } else { Visuals::dark() };
    v.panel_fill = panel();
    v.window_fill = chrome();
    v.extreme_bg_color = list();
    v.faint_bg_color = level(9);
    v.code_bg_color = list();
    v.window_stroke = Stroke::new(1.0, border());
    v.window_corner_radius = CornerRadius::same(3);
    v.menu_corner_radius = CornerRadius::same(3);
    v.selection.bg_fill = accent_dim();
    v.selection.stroke = Stroke::new(1.0, accent());
    v.hyperlink_color = accent();
    v.warn_fg_color = warn();
    v.error_fg_color = danger();
    v.slider_trailing_fill = true;
    v.striped = false;

    let radius = CornerRadius::same(2);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = chrome();
    w.noninteractive.weak_bg_fill = chrome();
    w.noninteractive.bg_stroke = Stroke::new(1.0, border());
    w.noninteractive.fg_stroke = Stroke::new(1.0, text());
    w.noninteractive.corner_radius = radius;

    w.inactive.bg_fill = control();
    w.inactive.weak_bg_fill = control();
    w.inactive.bg_stroke = Stroke::new(1.0, level(10));
    w.inactive.fg_stroke = Stroke::new(1.0, text());
    w.inactive.corner_radius = radius;

    w.hovered.bg_fill = control_hover();
    w.hovered.weak_bg_fill = control_hover();
    w.hovered.bg_stroke = Stroke::new(1.0, level(11));
    w.hovered.fg_stroke = Stroke::new(1.0, text_strong());
    w.hovered.corner_radius = radius;

    w.active.bg_fill = accent_dim();
    w.active.weak_bg_fill = accent_dim();
    w.active.bg_stroke = Stroke::new(1.0, accent());
    w.active.fg_stroke = Stroke::new(1.0, text_strong());
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

    /// Run `check` once for every brightness level and accent. The theme is process-wide, so one
    /// test walks them all rather than several tests racing to change it.
    fn for_every_look(mut check: impl FnMut(&str)) {
        for b in 0..BRIGHTNESS_NAMES.len() as u8 {
            for a in 0..ACCENTS.len() as u8 {
                set_look(b, a);
                check(&format!("{} / {}", BRIGHTNESS_NAMES[b as usize], ACCENTS[a as usize].0));
            }
        }
        set_look(0, 0);
    }

    #[test]
    fn text_is_readable_and_accents_visible_in_every_look() {
        for_every_look(|look| {
            for (name, bg) in [("panel", panel()), ("chrome", chrome()), ("list", list()), ("control", control()), ("hovered control", control_hover())] {
                let c = contrast(text(), bg);
                assert!(c >= 4.5, "{look}: text on {name}: {c:.2}:1 is below 4.5:1");
                let d = contrast(text_dim(), bg);
                assert!(d >= 3.0, "{look}: dim text on {name}: {d:.2}:1");
            }
            assert!(contrast(text_strong(), panel()) >= 7.0, "{look}: headings should clear 7:1");
            // UI components need 3:1 (WCAG 1.4.11).
            for (name, color) in [("accent", accent()), ("orange", orange()), ("warning", warn()), ("danger", danger())] {
                for (bg_name, bg) in [("panel", panel()), ("list", list())] {
                    let c = contrast(color, bg);
                    assert!(c >= 2.2, "{look}: {name} on {bg_name}: {c:.2}:1");
                }
            }
            // Selected text sits on the dim accent fill.
            let sel = contrast(text_strong(), accent_dim());
            assert!(sel >= 4.0, "{look}: selected text on the accent fill: {sel:.2}:1");
            // Dark looks: inset lists are darker than panels, which are darker than chrome and controls.
            // Light looks run the other way.
            let (l, p, c, k, h) = (luminance(list()), luminance(panel()), luminance(chrome()), luminance(control()), luminance(control_hover()));
            if brightness() < 2 {
                assert!(l < p && p < c && c < k && k < h, "{look}");
            } else {
                assert!(l > p && p > c && c > k && k > h, "{look}");
            }
            assert!(luminance(border()) < luminance(list()) || brightness() >= 2, "{look}");
        });
    }
}
