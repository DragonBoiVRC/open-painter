//! Text to a picture: white letters on a transparent background, to use as a brush alpha, a
//! stencil or a decal. Layout is plain left-to-right with the font's advances (no kerning, ligatures
//! or complex-script shaping), which suits labels, numbers and stencil lettering.

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};
use std::path::{Path, PathBuf};

pub const MAX_TEXT_CHARS: usize = 500;
pub const MAX_SIDE: u32 = 4096;
/// The font used when none is picked: the one egui ships.
pub const BUILT_IN: &str = "Built-in";

/// A font file the user has installed.
#[derive(Clone, Debug)]
pub struct FontFile {
    pub name: String,
    pub path: PathBuf,
}

/// Installed font files (`.ttf`/`.otf`, not collections), by file name, from the usual folders.
pub fn system_fonts() -> Vec<FontFile> {
    let mut roots = vec![PathBuf::from("/usr/share/fonts"), PathBuf::from("/usr/local/share/fonts"), PathBuf::from("/Library/Fonts"), PathBuf::from("C:\\Windows\\Fonts")];
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        roots.push(home.join(".local/share/fonts"));
        roots.push(home.join(".fonts"));
        roots.push(home.join("Library/Fonts"));
    }
    let mut out = Vec::new();
    let mut stack: Vec<(PathBuf, u32)> = roots.into_iter().map(|r| (r, 0)).collect();
    while let Some((dir, depth)) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && depth < 4 {
                stack.push((p, depth + 1));
            } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("ttf") || x.eq_ignore_ascii_case("otf")) {
                if let Some(name) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push(FontFile { name: name.to_string(), path: p });
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out.dedup_by(|a, b| a.name == b.name);
    out
}

/// The bytes of the built-in font.
pub fn built_in_font() -> Vec<u8> {
    let defs = egui::FontDefinitions::default();
    let data = defs.font_data.get("Ubuntu-Light").or_else(|| defs.font_data.values().next());
    data.map(|d| d.font.as_ref().as_ref().to_vec()).unwrap_or_default()
}

pub fn read_font(file: Option<&Path>) -> Result<Vec<u8>, String> {
    match file {
        Some(p) => std::fs::read(p).map_err(|e| format!("could not read {}: {e}", p.display())),
        None => Ok(built_in_font()),
    }
}

#[derive(Default)]
struct Outline {
    /// Closed polygons of the glyph, in pixels, y down, offset to its pen position.
    edges: Vec<[f32; 4]>,
    pen: (f32, f32),
    start: (f32, f32),
    at: (f32, f32),
}

impl Outline {
    fn line(&mut self, x: f32, y: f32) {
        let (px, py) = self.at;
        self.edges.push([px + self.pen.0, self.pen.1 - py, x + self.pen.0, self.pen.1 - y]);
        self.at = (x, y);
    }
}

impl OutlinePen for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.at = (x, y);
        self.start = (x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.line(x, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (x0, y0) = self.at;
        for i in 1..=8 {
            let t = i as f32 / 8.0;
            let u = 1.0 - t;
            self.line(u * u * x0 + 2.0 * u * t * cx + t * t * x, u * u * y0 + 2.0 * u * t * cy + t * t * y);
        }
    }
    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let (x0, y0) = self.at;
        for i in 1..=12 {
            let t = i as f32 / 12.0;
            let u = 1.0 - t;
            self.line(
                u * u * u * x0 + 3.0 * u * u * t * c1x + 3.0 * u * t * t * c2x + t * t * t * x,
                u * u * u * y0 + 3.0 * u * u * t * c1y + 3.0 * u * t * t * c2y + t * t * t * y,
            );
        }
    }
    fn close(&mut self) {
        let (sx, sy) = self.start;
        if self.at != (sx, sy) {
            self.line(sx, sy);
        }
    }
}

/// Coverage (0..255) of the polygon edges over a `w` x `h` grid, non-zero winding, antialiased by
/// four sample rows per pixel with exact horizontal overlap.
fn coverage(edges: &[[f32; 4]], w: usize, h: usize) -> Vec<u8> {
    const SUB: usize = 4;
    let mut acc = vec![0.0f32; w * h];
    let mut cross: Vec<(f32, i32)> = Vec::new();
    for sy in 0..h * SUB {
        let y = (sy as f32 + 0.5) / SUB as f32;
        cross.clear();
        for e in edges {
            let (y0, y1) = (e[1], e[3]);
            if (y0 <= y) != (y1 <= y) {
                let t = (y - y0) / (y1 - y0);
                cross.push((e[0] + t * (e[2] - e[0]), if y1 > y0 { 1 } else { -1 }));
            }
        }
        cross.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut wind, row) = (0, sy / SUB * w);
        for pair in cross.windows(2) {
            wind += pair[0].1;
            if wind == 0 {
                continue;
            }
            let (x0, x1) = (pair[0].0.max(0.0), pair[1].0.min(w as f32));
            let mut px = x0.floor() as usize;
            while (px as f32) < x1 && px < w {
                let overlap = (x1.min(px as f32 + 1.0) - x0.max(px as f32)).max(0.0);
                acc[row + px] += overlap / SUB as f32;
                px += 1;
            }
        }
    }
    acc.iter().map(|a| (a.min(1.0) * 255.0 + 0.5) as u8).collect()
}

/// White letters of `text` (lines split at newlines) in the font, `px` pixels from one baseline to
/// the next, on transparent: (width, height, RGBA8). Errors for an unusable font or empty text.
pub fn render(text: &str, font_bytes: &[u8], px: f32) -> Result<(u32, u32, Vec<u8>), String> {
    if text.trim().is_empty() {
        return Err("Type some text first.".into());
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(format!("Keep the text under {MAX_TEXT_CHARS} characters."));
    }
    let px = px.clamp(8.0, 1024.0);
    let font = FontRef::new(font_bytes).map_err(|e| format!("not a usable font: {e}"))?;
    let (size, loc) = (Size::new(px), LocationRef::default());
    let (charmap, metrics, glyphs, outlines) = (font.charmap(), font.metrics(size, loc), font.glyph_metrics(size, loc), font.outline_glyphs());
    let line_height = (metrics.ascent - metrics.descent + metrics.leading).max(px);
    let mut edges = Vec::new();
    let (mut width, mut lines) = (0.0f32, 0u32);
    for (n, line) in text.lines().enumerate() {
        lines += 1;
        let mut x = 0.0;
        let baseline = metrics.ascent + n as f32 * line_height;
        for ch in line.chars() {
            let Some(gid) = charmap.map(ch) else { continue };
            if let Some(glyph) = outlines.get(gid) {
                let mut o = Outline { pen: (x, baseline), ..Default::default() };
                if glyph.draw(DrawSettings::unhinted(size, loc), &mut o).is_ok() {
                    edges.extend(o.edges);
                }
            }
            x += glyphs.advance_width(gid).unwrap_or(px * 0.5);
        }
        width = width.max(x);
    }
    let pad = (px * 0.1).ceil();
    for e in &mut edges {
        e[0] += pad;
        e[2] += pad;
        e[1] += pad;
        e[3] += pad;
    }
    let w = ((width + 2.0 * pad).ceil() as u32).clamp(1, MAX_SIDE);
    let h = (((lines as f32) * line_height + 2.0 * pad).ceil() as u32).clamp(1, MAX_SIDE);
    let cov = coverage(&edges, w as usize, h as usize);
    let rgba = cov.iter().flat_map(|&a| [255, 255, 255, a]).collect();
    Ok((w, h, rgba))
}

/// The Text section's state in the Brush panel.
pub struct TextUi {
    pub text: String,
    /// Index into `App::fonts`, or the built-in font.
    pub font: Option<usize>,
    pub px: f32,
}

impl Default for TextUi {
    fn default() -> Self {
        Self { text: String::new(), font: None, px: 128.0 }
    }
}

impl crate::App {
    /// Render `text` and add it to the texture library, returning the image id.
    pub(crate) fn make_text_image(&mut self, text: &str, font: Option<&Path>, px: f32) -> Result<String, String> {
        let bytes = read_font(font)?;
        let (w, h, rgba) = render(text, &bytes, px)?;
        let png = op_image::encode_png(w, h, &rgba).map_err(|e| e.to_string())?;
        let name: String = text.lines().next().unwrap_or("text").chars().take(24).collect();
        self.add_texture(&format!("Text {name}"), png)
    }

    /// A new fill layer showing `image` as a decal, waiting for a click on the model to place it.
    pub(crate) fn new_decal_layer(&mut self, image: &str, name: &str) {
        self.add_node(name, op_core::NodeKind::Fill(op_core::FillParams::default()));
        let Some(id) = self.selected else { return };
        self.set_slot_texture(id, op_core::Slot::BaseColor, Some(op_core::TexRef::new(image)));
        self.edit(|d| {
            if let Some(op_core::NodeKind::Fill(f)) = d.find_mut(id).map(|n| &mut n.kind) {
                f.projection = op_core::Projection::Decal;
            }
        });
        self.decal_place = Some(id);
        self.status = "Click the model to place the decal (Esc cancels). It needs the Position and World space normal maps baked.".into();
    }

    pub(crate) fn text_section(&mut self, ui: &mut egui::Ui) {
        if self.fonts.is_none() {
            self.fonts = Some(system_fonts());
        }
        ui.label(egui::RichText::new("TEXT").size(11.0).strong().color(crate::ui::theme::text_dim()));
        ui.add(egui::TextEdit::multiline(&mut self.text_ui.text).desired_rows(2).hint_text("Type text to make a picture of"));
        let fonts = self.fonts.clone().unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label("Font");
            let shown = self.text_ui.font.and_then(|i| fonts.get(i)).map_or(BUILT_IN.to_string(), |f| f.name.clone());
            egui::ComboBox::from_id_salt("text_font").selected_text(shown).width(150.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.text_ui.font, None, BUILT_IN);
                for (i, f) in fonts.iter().enumerate() {
                    ui.selectable_value(&mut self.text_ui.font, Some(i), &f.name);
                }
            });
            ui.add(egui::DragValue::new(&mut self.text_ui.px).range(8.0..=1024.0).speed(1.0).suffix(" px"));
        });
        let mut made: Option<(String, u8)> = None;
        ui.horizontal_wrapped(|ui| {
            for (label, which, tip) in [
                ("Brush stamp", 0u8, "Use the text as the Image stamp of the brush"),
                ("Stencil", 1, "Use the text as the stencil: paint lands only where the letters are"),
                ("Decal layer", 2, "A new fill layer with the text projected onto the model, placed by a click"),
                ("To library", 3, "Just add the picture to the texture library"),
            ] {
                if ui.button(label).on_hover_text(tip).clicked() {
                    let font = self.text_ui.font.and_then(|i| fonts.get(i)).map(|f| f.path.clone());
                    let text = self.text_ui.text.clone();
                    match self.make_text_image(&text, font.as_deref(), self.text_ui.px) {
                        Ok(id) => made = Some((id, which)),
                        Err(e) => self.status = e,
                    }
                }
            }
        });
        if let Some((id, which)) = made {
            match which {
                0 => {
                    self.brush_pics.alpha = Some(id);
                    self.brush.shape = 5;
                }
                1 => self.brush_pics.stencil = Some(id),
                2 => self.new_decal_layer(&id, "Text decal"),
                _ => self.status = "Added the text picture to the library.".into(),
            }
        }
        crate::ui::panels::hint(ui, "Letters are laid out left to right with the font's own spacing (no kerning or complex-script shaping). Newlines start a new line.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_renders_as_white_letters_that_grow_with_the_text() {
        let font = built_in_font();
        assert!(!font.is_empty(), "egui ships a font");
        let (w1, h1, rgba) = render("I", &font, 64.0).unwrap();
        assert!(rgba.chunks(4).any(|p| p[3] > 200), "the letter has solid pixels");
        assert!(rgba.chunks(4).all(|p| p[..3] == [255, 255, 255]));
        assert!(rgba.chunks(4).any(|p| p[3] == 0), "and transparent ones");
        let (w3, _, _) = render("III", &font, 64.0).unwrap();
        assert!(w3 > w1 * 2, "{w3} vs {w1}");
        let (_, h2, _) = render("I\nI", &font, 64.0).unwrap();
        assert!(h2 > h1, "a second line is taller");
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(render("   ", &built_in_font(), 32.0).is_err());
        assert!(render("x", b"not a font", 32.0).is_err());
        assert!(render(&"x".repeat(MAX_TEXT_CHARS + 1), &built_in_font(), 32.0).is_err());
    }
}

