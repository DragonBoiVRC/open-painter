//! The window frame: menu bar, tool strip, status bar and the docking area that holds the panels.

use super::dock::{self, Tab};
use super::theme;
use crate::{App, Pending, PathDialog, Todo, Tool};
use egui_dock::{DockArea, DockState, TabViewer};
use egui_phosphor::regular as ph;
use op_core::MeshMap;

/// Connects the docking library to the application: what each tab shows.
struct Viewer<'a> {
    app: &'a mut App,
}

impl TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::NULL.with(("panel", *tab))
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        egui::RichText::new(format!("{}  {}", tab.icon(), tab.tab_label().to_uppercase())).size(11.0).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        self.app.tab_ui(ui, *tab);
    }

    fn is_closeable(&self, tab: &Tab) -> bool {
        tab.closeable()
    }

    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        // The viewport fills its tab; every other panel scrolls when it outgrows its space.
        if *tab == Tab::Viewport { [false, false] } else { [false, true] }
    }

    fn clear_background(&self, tab: &Tab) -> bool {
        *tab != Tab::Viewport
    }
}

impl App {
    /// Draw one frame of the interface (everything except per-frame housekeeping and shortcuts).
    pub(crate) fn ui_frame(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let mut dock = self.dock.take().unwrap_or_else(dock::default_layout);

        self.menu_bar(ui, &mut dock);
        self.status_bar(ui);
        self.tool_strip(ui);
        self.commit_gesture(ui);
        self.dialogs(&ctx);

        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(theme::BORDER)).show(ui, |ui| {
            DockArea::new(&mut dock)
                .style(dock::style(&ctx))
                .show_add_buttons(false)
                .show_close_buttons(true)
                .show_leaf_close_all_buttons(false)
                .show_leaf_collapse_buttons(false)
                .show_inside(ui, &mut Viewer { app: self });
        });
        self.dock = Some(dock);
    }

    /// The body of one panel.
    pub(crate) fn tab_ui(&mut self, ui: &mut egui::Ui, tab: Tab) {
        match tab {
            Tab::Viewport => self.viewport_ui(ui),
            Tab::Brush => self.brush_panel(ui),
            Tab::Layers => self.layers_panel(ui),
            Tab::Properties => self.properties_panel(ui),
            Tab::TextureSet => self.texture_set_panel(ui),
            Tab::Bake => self.bake_panel(ui),
            Tab::Display => self.display_panel(ui),
            Tab::Shelf => self.shelf_panel(ui),
        }
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, dock: &mut DockState<Tab>) {
        let ctx = ui.ctx().clone();
        egui::Panel::top("menu_bar").frame(egui::Frame::NONE.fill(theme::CHROME).inner_margin(egui::Margin::symmetric(6, 2))).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.add(egui::Button::new("New").shortcut_text("Ctrl+N")).clicked() {
                        self.request(Pending::New, &ctx);
                        ui.close();
                    }
                    if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                        let start = self.project_path.clone().map(|p| p.display().to_string()).unwrap_or_default();
                        self.dialog = Some(PathDialog { save: false, path: start });
                        ui.close();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S")).clicked() {
                        match self.project_path.clone() {
                            Some(p) => self.todo.push(Todo::Save(p)),
                            None => self.dialog = Some(PathDialog { save: true, path: self.default_save_path() }),
                        }
                        ctx.request_repaint();
                        ui.close();
                    }
                    if ui.button("Save as...").clicked() {
                        self.dialog = Some(PathDialog { save: true, path: self.default_save_path() });
                        ui.close();
                    }
                    let recoverable = self.recover_path().exists();
                    if ui.add_enabled(recoverable, egui::Button::new("Recover autosave")).clicked() {
                        self.recover();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Export textures...").clicked() {
                        if self.export.dir.is_empty() {
                            self.export.dir = self.default_export_dir().display().to_string();
                        }
                        self.export.open = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Quit").shortcut_text("Ctrl+Q")).clicked() {
                        self.request(Pending::Quit, &ctx);
                        ui.close();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                        self.do_undo();
                        ui.close();
                    }
                    if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo").shortcut_text("Ctrl+Y")).clicked() {
                        self.do_redo();
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Clear paint target").clicked() {
                        self.clear_target();
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.add(egui::Button::new("Frame model").shortcut_text("F")).clicked() {
                        self.frame_model();
                        ui.close();
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Shading").color(theme::TEXT_DIM).small());
                    let current = self.view;
                    let mut view = current;
                    ui.selectable_value(&mut view, crate::View::Shaded, "Shaded");
                    if self.baked.contains_key(&MeshMap::AmbientOcclusion) {
                        ui.selectable_value(&mut view, crate::View::ShadedAo, "Shaded + baked AO");
                    }
                    for m in MeshMap::ALL {
                        if self.baked.contains_key(&m) {
                            ui.selectable_value(&mut view, crate::View::Map(m), m.label());
                        }
                    }
                    if view != current {
                        self.set_view(view);
                    }
                });
                ui.menu_button("Window", |ui| {
                    for tab in Tab::ALL {
                        let open = dock::is_open(dock, tab);
                        let label = format!("{}  {}", tab.icon(), tab.title());
                        if ui.add_enabled(tab.closeable(), egui::Button::new(label).selected(open)).clicked() {
                            if open { dock::close(dock, tab) } else { dock::open(dock, tab) }
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button("Reset layout").clicked() {
                        *dock = dock::default_layout();
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard shortcuts").clicked() {
                        self.show_shortcuts = true;
                        ui.close();
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status_bar")
            .exact_size(22.0)
            .frame(egui::Frame::NONE.fill(theme::CHROME).inner_margin(egui::Margin::symmetric(8, 2)))
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(egui::RichText::new(self.status.lines().next().unwrap_or("")).color(theme::TEXT_DIM).size(11.5));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let baked = if self.baked.is_empty() { String::new() } else { format!("{} baked  |  ", self.baked.len()) };
                        let tris = self.mesh.indices.len() / 3;
                        ui.label(
                            egui::RichText::new(format!("{baked}{tris} tris  |  {0}x{0}", self.tex_size))
                                .color(theme::TEXT_DIM)
                                .size(11.5),
                        );
                    });
                });
            });
    }

    /// The vertical strip of tools on the left edge.
    fn tool_strip(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("tool_strip")
            .exact_size(38.0)
            .resizable(false)
            .frame(egui::Frame::NONE.fill(theme::CHROME).inner_margin(egui::Margin::symmetric(4, 6)))
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    if tool_button(ui, ph::PAINT_BRUSH, "Paint (P)", self.tool == Tool::Paint).clicked() {
                        self.tool = Tool::Paint;
                    }
                    if tool_button(ui, ph::ERASER, "Erase (E)", self.tool == Tool::Erase).clicked() {
                        self.tool = Tool::Erase;
                    }
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);
                    if tool_button(ui, ph::STACK_SIMPLE, "Paint on the layer's content", self.target == crate::Target::Content).clicked() {
                        self.target = crate::Target::Content;
                    }
                    if tool_button(ui, ph::CIRCLE_HALF, "Paint on the layer's mask", self.target == crate::Target::Mask).clicked() {
                        self.target = crate::Target::Mask;
                    }
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);
                    ui.spacing_mut().interact_size = egui::vec2(28.0, 22.0);
                    ui.color_edit_button_rgb(&mut self.channels.base_color).on_hover_text("Paint color");
                });
            });
    }
}

/// A square icon button for the tool strip; `active` draws it pressed with the accent outline.
pub fn tool_button(ui: &mut egui::Ui, icon: &str, tip: &str, active: bool) -> egui::Response {
    let size = egui::vec2(28.0, 28.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let hovered = resp.hovered();
    let fill = if active {
        theme::ACCENT_DIM
    } else if hovered {
        theme::CONTROL_HOVER
    } else {
        egui::Color32::TRANSPARENT
    };
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, fill);
    if active {
        painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(17.0),
        if active || hovered { theme::TEXT_STRONG } else { theme::TEXT },
    );
    resp.on_hover_text(tip)
}
