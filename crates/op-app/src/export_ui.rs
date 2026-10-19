//! The Export textures window, laid out like Substance Painter's:
//!
//! * a tab row (Settings, Output templates, List of exports) with "Include all" on the right,
//! * **Settings**: a sidebar with "Global settings" and the texture sets (each with a tick),
//!   and a form with the output directory, template, file type, bit depth and padding,
//! * **Output templates**: the templates and what each one writes, plus the .spexp import,
//! * **List of exports**: the files that will be written for each set,
//! * a footer with Reset all, the number of textures selected, Cancel, Save settings and Export.

use crate::{files, picker, App, ExportUi};
use op_export::{expand_template, BitDepth, ChannelSrc, ExportPreset, FileFormat, MapSpec, Vars};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ExportTab {
    #[default]
    Settings,
    Templates,
    List,
}

/// The export settings kept between sessions ("Save settings").
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SavedExport {
    template: Option<String>,
    padding: Option<u32>,
    file_type: Option<FileFormat>,
    bit_depth: Option<BitDepth>,
}

impl SavedExport {
    pub(crate) fn load() -> Self {
        let path = files::export_settings_path();
        std::fs::metadata(&path)
            .ok()
            .filter(|m| m.len() < 64 * 1024)
            .and_then(|_| std::fs::read_to_string(&path).ok())
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
}

impl ExportUi {
    /// Start from the saved settings. A template that no longer exists, or a bit depth the file type
    /// cannot hold, is ignored.
    pub(crate) fn apply_saved(&mut self, saved: SavedExport, presets: &[ExportPreset]) {
        if let Some(i) = saved.template.and_then(|t| presets.iter().position(|p| p.name == t)) {
            self.preset = i;
        }
        if let Some(p) = saved.padding {
            self.padding = p.min(64);
        }
        self.file_type = saved.file_type;
        self.bit_depth = saved.bit_depth.filter(|d| depths_for(self.file_type).contains(d));
    }
}

/// The bit depths a file type can hold; with no type chosen, the ones PNG offers.
fn depths_for(file_type: Option<FileFormat>) -> &'static [BitDepth] {
    match file_type {
        Some(FileFormat::Tga) => &[BitDepth::U8],
        Some(FileFormat::Exr) => &[BitDepth::F32],
        _ => &[BitDepth::U8, BitDepth::U16],
    }
}

fn depth_name(d: BitDepth) -> &'static str {
    match d {
        BitDepth::U8 => "8 bits",
        BitDepth::U16 => "16 bits",
        BitDepth::F32 => "32 bits (float)",
    }
}

/// The preset with the window's file type and bit depth applied over every map.
pub(crate) fn with_overrides(mut preset: ExportPreset, file_type: Option<FileFormat>, depth: Option<BitDepth>) -> ExportPreset {
    for m in &mut preset.maps {
        if let Some(f) = file_type {
            m.format = f;
            m.bit_depth = match f {
                FileFormat::Png => depth
                    .filter(|d| *d != BitDepth::F32)
                    .unwrap_or(if m.bit_depth == BitDepth::U16 { BitDepth::U16 } else { BitDepth::U8 }),
                FileFormat::Tga => BitDepth::U8,
                FileFormat::Exr => BitDepth::F32,
            };
        } else if let Some(d) = depth {
            // Only PNG has a choice of depth; other formats keep theirs.
            if m.format == FileFormat::Png && d != BitDepth::F32 {
                m.bit_depth = d;
            }
        }
    }
    preset
}

const WARN: egui::Color32 = egui::Color32::from_rgb(235, 190, 70);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(60, 130, 220);
const FOOTER_HEIGHT: f32 = 52.0;
const SIDEBAR_WIDTH: f32 = 250.0;
const LABEL_WIDTH: f32 = 200.0;
const ROW_HEIGHT: f32 = 26.0;

fn channel_label(c: &ChannelSrc) -> (String, egui::Color32) {
    let grey = egui::Color32::from_gray(150);
    match c {
        ChannelSrc::Color { map, comp } => (format!("{map:?} {comp:?}"), egui::Color32::from_rgb(120, 170, 235)),
        ChannelSrc::Metallic => ("Metallic".into(), egui::Color32::from_rgb(190, 190, 205)),
        ChannelSrc::Roughness => ("Roughness".into(), egui::Color32::from_rgb(150, 205, 150)),
        ChannelSrc::Smoothness => ("Smoothness".into(), egui::Color32::from_rgb(150, 205, 150)),
        ChannelSrc::Mesh { map, comp } => (format!("{} {comp:?}", map.label()), egui::Color32::from_rgb(210, 160, 120)),
        ChannelSrc::Normal { directx, comp } => {
            (format!("Normal {comp:?} ({})", if *directx { "DX" } else { "GL" }), egui::Color32::from_rgb(150, 150, 235))
        }
        ChannelSrc::Height => ("Height".into(), egui::Color32::from_gray(200)),
        ChannelSrc::Opacity => ("Opacity".into(), egui::Color32::from_rgb(200, 150, 200)),
        ChannelSrc::AoMixed => ("AO".into(), egui::Color32::from_rgb(210, 160, 120)),
        ChannelSrc::Constant(v) => {
            let name = if *v <= 0.0 {
                "Black".to_string()
            } else if *v >= 1.0 {
                "White".to_string()
            } else {
                format!("{v:.2}")
            };
            (name, grey)
        }
    }
}

fn map_summary(m: &MapSpec) -> String {
    format!("{}  {}", m.format.extension().to_uppercase(), depth_name(m.bit_depth))
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new(text).strong().size(14.0));
    ui.add_space(8.0);
}

/// A tab header: capitals, and the active one gets a darker fill with an accent line on top.
fn tab_button(ui: &mut egui::Ui, current: &mut ExportTab, tab: ExportTab, text: &str) {
    let active = *current == tab;
    let rich = egui::RichText::new(text.to_uppercase()).size(12.5).strong();
    let rich = if active { rich } else { rich.color(ui.visuals().weak_text_color()) };
    let fill = if active { egui::Color32::from_gray(30) } else { egui::Color32::TRANSPARENT };
    let resp = ui.add(egui::Button::new(rich).fill(fill).stroke(egui::Stroke::NONE).corner_radius(0.0).min_size(egui::vec2(0.0, 30.0)));
    if active {
        let r = resp.rect;
        ui.painter().line_segment([r.left_top(), r.right_top()], egui::Stroke::new(2.0, ACCENT));
    }
    if resp.clicked() {
        *current = tab;
    }
}

/// A form row: the label on the left, then whatever `add` puts in the remaining width.
fn form_row(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui, f32)) {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(LABEL_WIDTH, ROW_HEIGHT), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.label(label);
        });
        let width = (ui.available_width() - 24.0).max(120.0);
        add(ui, width);
    });
    ui.add_space(5.0);
}

fn scrolled(ui: &mut egui::Ui, height: f32, add: impl FnOnce(&mut egui::Ui)) {
    egui::ScrollArea::vertical().id_salt("export_body").max_height(height).auto_shrink([false, false]).show(ui, |ui| {
        ui.add_space(6.0);
        add(ui);
    });
}

/// A full-width row in the sidebar, highlighted when selected.
fn sidebar_row(ui: &mut egui::Ui, selected: bool, content: impl FnOnce(&mut egui::Ui)) -> egui::Response {
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 30.0), egui::Sense::click());
    if selected {
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(48));
        ui.painter().rect_stroke(rect, 0.0, egui::Stroke::new(1.0, ACCENT), egui::StrokeKind::Inside);
    } else if resp.hovered() {
        ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(40));
    }
    let mut inner = ui.new_child(
        egui::UiBuilder::new().max_rect(rect.shrink2(egui::vec2(8.0, 0.0))).layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    content(&mut inner);
    resp
}

impl App {
    pub(crate) fn export_window(&mut self, ctx: &egui::Context) {
        if !self.export.open {
            return;
        }
        let mut open = true;
        egui::Window::new("Export textures")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([940.0, 640.0])
            .min_size([640.0, 420.0])
            .show(ctx, |ui| {
                let sets = self.mesh.set_ranges().len();
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    tab_button(ui, &mut self.export.tab, ExportTab::Settings, "Settings");
                    tab_button(ui, &mut self.export.tab, ExportTab::Templates, "Output templates");
                    tab_button(ui, &mut self.export.tab, ExportTab::List, "List of exports");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(sets > 1, egui::Button::new("Include all").min_size(egui::vec2(110.0, 26.0))).clicked() {
                            self.export.skip_sets.clear();
                        }
                    });
                });
                ui.separator();
                let body_height = (ui.available_height() - FOOTER_HEIGHT).max(160.0);
                match self.export.tab {
                    ExportTab::Settings => self.export_settings_tab(ui, body_height),
                    ExportTab::Templates => scrolled(ui, body_height, |ui| self.export_templates_tab(ui)),
                    ExportTab::List => scrolled(ui, body_height, |ui| self.export_list_tab(ui)),
                }
                ui.separator();
                self.export_footer(ui, ctx);
            });
        self.export.open = open && self.export.open;
    }

    /// Sidebar (Global settings and the texture sets) with the form or the selected set on the right.
    fn export_settings_tab(&mut self, ui: &mut egui::Ui, height: f32) {
        let ranges = self.mesh.set_ranges();
        let multi = ranges.len() > 1;
        let names: Vec<String> =
            if multi { ranges.iter().map(|r| r.name.clone()).collect() } else { vec![self.export.set_name.clone()] };
        if self.export.sel.is_some_and(|i| i >= names.len()) {
            self.export.sel = None;
        }
        let width = ui.available_width();
        ui.allocate_ui_with_layout(egui::vec2(width, height), egui::Layout::left_to_right(egui::Align::Min), |ui| {
            // ---- sidebar
            ui.allocate_ui_with_layout(egui::vec2(SIDEBAR_WIDTH, height), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_min_height(height);
                ui.add_space(2.0);
                let global = self.export.sel.is_none();
                if sidebar_row(ui, global, |ui| {
                    ui.label(egui::RichText::new("Global settings").strong());
                })
                .clicked()
                {
                    self.export.sel = None;
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let half = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                    if ui.add_enabled(multi, egui::Button::new("Select all").min_size(egui::vec2(half, 24.0))).clicked() {
                        self.export.skip_sets.clear();
                    }
                    if ui.add_enabled(multi, egui::Button::new("Deselect all").min_size(egui::vec2(half, 24.0))).clicked() {
                        self.export.skip_sets = (0..names.len()).collect();
                    }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical().id_salt("export_sidebar").auto_shrink([false, false]).show(ui, |ui| {
                    for (i, name) in names.iter().enumerate() {
                        let selected = self.export.sel == Some(i);
                        let mut on = !self.export.skip_sets.contains(&i);
                        let row = sidebar_row(ui, selected, |ui| {
                            if ui.add_enabled(multi, egui::Checkbox::without_text(&mut on)).changed() {
                                if on {
                                    self.export.skip_sets.remove(&i);
                                } else {
                                    self.export.skip_sets.insert(i);
                                }
                            }
                            ui.label(egui::RichText::new(name).strong());
                        });
                        if row.clicked() {
                            self.export.sel = Some(i);
                        }
                    }
                });
            });
            ui.separator();
            // ---- right pane
            ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), height), egui::Layout::top_down(egui::Align::Min), |ui| {
                egui::ScrollArea::vertical().id_salt("export_form").auto_shrink([false, false]).show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.set_width(ui.available_width());
                    match self.export.sel {
                        None => self.export_global_form(ui),
                        Some(i) => self.export_set_pane(ui, i, &names[i]),
                    }
                });
            });
        });
    }

    fn export_global_form(&mut self, ui: &mut egui::Ui) {
        let idx = self.export.preset.min(self.presets.len() - 1);
        heading(ui, "General Export Parameters");
        form_row(ui, "Output directory", |ui, w| {
            ui.add(egui::TextEdit::singleline(&mut self.export.dir).desired_width((w - 100.0).max(80.0)));
            if picker::browse_button(ui, self.picker.is_some()) {
                let cur = self.export.dir.clone();
                self.browse(picker::Field::ExportDir, &cur);
            }
        });
        form_row(ui, "Output template", |ui, w| {
            egui::ComboBox::from_id_salt("export_template").width(w).selected_text(&self.presets[idx].name).show_ui(ui, |ui| {
                for (i, p) in self.presets.iter().enumerate() {
                    ui.selectable_value(&mut self.export.preset, i, &p.name);
                }
            });
        });
        form_row(ui, "File type", |ui, w| {
            let half = ((w - 8.0) / 2.0).max(80.0);
            let before = self.export.file_type;
            egui::ComboBox::from_id_salt("export_file_type")
                .width(half)
                .selected_text(self.export.file_type.map_or("Template default", |f| f.extension()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.export.file_type, None, "Template default");
                    for f in [FileFormat::Png, FileFormat::Tga, FileFormat::Exr] {
                        ui.selectable_value(&mut self.export.file_type, Some(f), f.extension());
                    }
                });
            if self.export.file_type != before && !self.export.bit_depth.is_some_and(|d| depths_for(self.export.file_type).contains(&d)) {
                self.export.bit_depth = None;
            }
            egui::ComboBox::from_id_salt("export_bit_depth")
                .width(half)
                .selected_text(self.export.bit_depth.map_or("Template default", depth_name))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.export.bit_depth, None, "Template default");
                    for d in depths_for(self.export.file_type) {
                        ui.selectable_value(&mut self.export.bit_depth, Some(*d), depth_name(*d));
                    }
                });
        });
        form_row(ui, "Size", |ui, w| {
            // The export always writes each texture set at the size it is painted at.
            ui.add_enabled_ui(false, |ui| {
                egui::ComboBox::from_id_salt("export_size").width(w).selected_text("Based on each Texture Set's size").show_ui(ui, |_| {});
            });
        });
        form_row(ui, "Padding", |ui, w| {
            ui.add_enabled_ui(false, |ui| {
                egui::ComboBox::from_id_salt("export_padding_mode").width((w - 88.0).max(80.0)).selected_text("Dilation").show_ui(ui, |_| {});
            });
            egui::ComboBox::from_id_salt("export_padding").width(80.0).selected_text(self.export.padding.to_string()).show_ui(ui, |ui| {
                let mut options = vec![0, 1, 2, 4, 8, 16, 32, 64];
                if !options.contains(&self.export.padding) {
                    options.push(self.export.padding);
                    options.sort_unstable();
                }
                for p in options {
                    ui.selectable_value(&mut self.export.padding, p, p.to_string());
                }
            });
        });
        if self.mesh.set_ranges().len() <= 1 {
            form_row(ui, "Texture set name", |ui, w| {
                ui.add(egui::TextEdit::singleline(&mut self.export.set_name).desired_width(w));
            });
        }
        ui.add_space(6.0);
        ui.weak("Edge padding (dilation) extends colours past the UV borders so mip-maps and filtering don't show seams.");
        self.export_notes(ui);
    }

    /// The pane for one texture set selected in the sidebar.
    fn export_set_pane(&mut self, ui: &mut egui::Ui, i: usize, name: &str) {
        let ranges = self.mesh.set_ranges();
        heading(ui, &format!("Texture set: {name}"));
        if let Some(r) = ranges.get(i) {
            ui.weak(format!("{} triangles", r.index_count / 3));
        }
        ui.add_space(8.0);
        let idx = self.export.preset.min(self.presets.len() - 1);
        let included = !self.export.skip_sets.contains(&i);
        ui.label(if included { "Included in the export." } else { "Not included in the export." });
        ui.add_space(8.0);
        if included {
            self.write_files_list(ui, idx, name);
        }
        self.export_notes(ui);
    }

    fn write_files_list(&self, ui: &mut egui::Ui, idx: usize, set_name: &str) {
        let project = self.project_name();
        let preset = with_overrides(self.presets[idx].clone(), self.export.file_type, self.export.bit_depth);
        for m in &preset.maps {
            let vars = Vars { project: &project, texture_set: set_name, mesh: &self.mesh_name, udim: None };
            let file = expand_template(&m.file, &vars).unwrap_or_else(|e| format!("({e})"));
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{file}.{}", m.format.extension())).monospace());
                ui.weak(depth_name(m.bit_depth));
            });
        }
    }

    fn export_templates_tab(&mut self, ui: &mut egui::Ui) {
        let idx = self.export.preset.min(self.presets.len() - 1);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(SIDEBAR_WIDTH - 20.0);
                heading(ui, "Templates");
                for (i, p) in self.presets.iter().enumerate() {
                    if ui.selectable_label(i == idx, &p.name).clicked() {
                        self.export.preset = i;
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                heading(ui, &self.presets[idx].name);
                for (mi, m) in self.presets[idx].maps.iter().enumerate() {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("{}.{}", m.file, m.format.extension())).strong());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.weak(map_summary(m));
                            });
                        });
                        let names: &[&str] = match m.channels.len() {
                            1 => &["L"],
                            3 => &["R", "G", "B"],
                            _ => &["R", "G", "B", "A"],
                        };
                        egui::Grid::new(("export_map_channels", mi)).num_columns(2).spacing([10.0, 3.0]).show(ui, |ui| {
                            for (name, c) in names.iter().zip(&m.channels) {
                                let (text, color) = channel_label(c);
                                ui.weak(*name);
                                ui.label(egui::RichText::new(text).color(color));
                                ui.end_row();
                            }
                        });
                    });
                }
            });
        });

        ui.add_space(10.0);
        ui.separator();
        heading(ui, "Import a Substance Painter template (.spexp)");
        ui.horizontal(|ui| {
            let width = (ui.available_width() - 210.0).max(120.0);
            ui.add(egui::TextEdit::singleline(&mut self.export.import_path).desired_width(width));
            if picker::browse_button(ui, self.picker.is_some()) {
                let cur = self.export.import_path.clone();
                self.browse(picker::Field::PresetImport, &cur);
            }
            if ui.add_enabled(!self.export.import_path.trim().is_empty(), egui::Button::new("Import")).clicked() {
                self.import_preset();
            }
        });
        if !self.export.report.is_empty() {
            ui.add_space(4.0);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::ScrollArea::vertical().id_salt("import_report").max_height(170.0).show(ui, |ui| {
                    for (i, line) in self.export.report.iter().enumerate() {
                        if i == 0 {
                            ui.label(line);
                        } else {
                            ui.colored_label(WARN, line);
                        }
                    }
                });
            });
        }
    }

    /// Warnings about templates that were skipped and notes from the last export (for example maps
    /// that were not baked).
    fn export_notes(&self, ui: &mut egui::Ui) {
        if self.export.last_notes.is_empty() && self.preset_warnings.is_empty() {
            return;
        }
        ui.add_space(8.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            for w in &self.preset_warnings {
                ui.colored_label(WARN, format!("Template skipped: {w}"));
            }
            for n in &self.export.last_notes {
                ui.colored_label(WARN, n);
            }
        });
    }

    fn export_list_tab(&mut self, ui: &mut egui::Ui) {
        let idx = self.export.preset.min(self.presets.len() - 1);
        let ranges = self.mesh.set_ranges();
        let multi = ranges.len() > 1;
        let names: Vec<String> =
            if multi { ranges.iter().map(|r| r.name.clone()).collect() } else { vec![self.export.set_name.clone()] };

        heading(ui, "List of exports");
        for (i, name) in names.iter().enumerate() {
            if multi && self.export.skip_sets.contains(&i) {
                continue;
            }
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(egui::RichText::new(name).strong());
                ui.add_space(2.0);
                self.write_files_list(ui, idx, name);
            });
        }
        if multi && !self.presets[idx].separates_texture_sets() {
            ui.add_space(4.0);
            ui.colored_label(WARN, "This template's file names have no $textureSet: only one set can be exported at a time.");
        }
        self.export_notes(ui);
    }

    /// Number of image files the current selection will write.
    fn export_texture_count(&self) -> usize {
        let ranges = self.mesh.set_ranges();
        let maps = self.presets[self.export.preset.min(self.presets.len() - 1)].maps.len();
        (0..ranges.len().max(1))
            .filter(|i| !self.export.skip_sets.contains(i))
            .map(|i| maps * if self.udim { self.tile_count(i).max(1) } else { 1 })
            .sum()
    }

    fn export_footer(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let idx = self.export.preset.min(self.presets.len() - 1);
        let ranges = self.mesh.set_ranges();
        let busy = self.export.running.is_some();
        let chosen = (0..ranges.len()).filter(|i| !self.export.skip_sets.contains(i)).count();
        let collides = chosen > 1 && !self.presets[idx].separates_texture_sets();
        let ready = !busy && !self.export.dir.trim().is_empty() && chosen > 0 && !collides;
        let textures = self.export_texture_count();

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.add_sized([110.0, 30.0], egui::Button::new("Reset all")).clicked() {
                self.reset_export_settings();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let export = egui::Button::new(egui::RichText::new("Export").strong()).min_size(egui::vec2(120.0, 30.0));
                if ui.add_enabled(ready, export).clicked() {
                    self.todo.push(crate::Todo::Export);
                    ctx.request_repaint();
                }
                if ui.add_sized([120.0, 30.0], egui::Button::new("Save settings")).clicked() {
                    self.save_export_settings();
                }
                if ui.add_sized([100.0, 30.0], egui::Button::new("Cancel")).clicked() {
                    self.export.open = false;
                }
                if busy {
                    ui.spinner();
                    ui.label("Exporting...");
                } else {
                    // The messages themselves are in the body: the footer must keep a fixed height, or
                    // the window grows to fit it and never settles.
                    let notes = self.export.last_notes.len() + self.preset_warnings.len();
                    if notes > 0 {
                        ui.colored_label(WARN, format!("{notes} note(s) from the last export"));
                        ui.separator();
                    }
                    let what = if textures == 1 { "texture" } else { "textures" };
                    ui.label(format!("{textures} {what} selected for export"));
                }
            });
        });
    }

    fn reset_export_settings(&mut self) {
        self.export.dir = self.default_export_dir().display().to_string();
        self.export.padding = crate::DEFAULT_PADDING;
        self.export.file_type = None;
        self.export.bit_depth = None;
        self.export.preset = 0;
        self.export.skip_sets.clear();
        self.export.sel = None;
    }

    fn save_export_settings(&mut self) {
        let saved = SavedExport {
            template: self.presets.get(self.export.preset).map(|p| p.name.clone()),
            padding: Some(self.export.padding),
            file_type: self.export.file_type,
            bit_depth: self.export.bit_depth,
        };
        let path = files::export_settings_path();
        let result = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&path, serde_json::to_vec_pretty(&saved).unwrap_or_default()));
        self.status = match result {
            Ok(()) => "Export settings saved.".into(),
            Err(e) => format!("Could not save the export settings: {e}"),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_type_and_bit_depth_override_the_template() {
        let base = op_export::builtin_presets().into_iter().next().unwrap();
        let none = with_overrides(base.clone(), None, None);
        assert_eq!(none, base, "no override changes nothing");

        let exr = with_overrides(base.clone(), Some(FileFormat::Exr), Some(BitDepth::U16));
        assert!(exr.maps.iter().all(|m| m.format == FileFormat::Exr && m.bit_depth == BitDepth::F32), "EXR is always float");
        assert!(exr.validate().is_ok());

        let tga = with_overrides(base.clone(), Some(FileFormat::Tga), None);
        assert!(tga.maps.iter().all(|m| m.format == FileFormat::Tga && m.bit_depth == BitDepth::U8));
        assert!(tga.validate().is_ok());

        let png16 = with_overrides(base, None, Some(BitDepth::U16));
        assert!(png16.maps.iter().filter(|m| m.format == FileFormat::Png).all(|m| m.bit_depth == BitDepth::U16));
        assert!(png16.validate().is_ok());
    }
}
