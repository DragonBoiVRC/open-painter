mod camera;
mod composite;
mod paint;
mod viewport;

use camera::OrbitCamera;
use eframe::{egui, egui_wgpu};
use glam::Vec3;
use op_core::{srgb_to_linear, BlendMode, Document, FillParams, Mask, Node, NodeId, NodeKind};
use op_mesh::Mesh;
use paint::{PaintOp, Stamp, Target, DEFAULT_TEXTURE_SIZE, MAX_STAMPS};
use std::path::PathBuf;
use viewport::{Globals, ViewportCallback, ViewportGpu};

const TEXTURE_SIZES: [u32; 4] = [512, 1024, 2048, 4096];
const MAX_HISTORY: usize = 300;

struct Brush {
    radius_pts: f32,
    hardness: f32,
    opacity: f32,
    /// Stamp spacing as a fraction of the brush radius.
    spacing: f32,
}

/// What the brush writes into a paint layer. Each channel can be toggled independently.
struct PaintChannels {
    base: bool,
    metal: bool,
    rough: bool,
    base_color: [f32; 3],
    metallic: f32,
    roughness: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Paint,
    Erase,
}

#[derive(Default)]
struct Stroke {
    active: bool,
    last_px: [f32; 2],
    /// Distance still to travel before the next stamp.
    next_in: f32,
}

/// One undoable step. Structure edits store both document states; stroke steps are
/// kept on the GPU (texture snapshots) and only counted here, in order.
enum Hist {
    Doc { before: Document, after: Document },
    Stroke,
}

enum Act {
    Select(NodeId),
    Visible(NodeId, bool),
}

struct App {
    camera: OrbitCamera,
    doc: Document,
    selected: Option<NodeId>,
    target: Target,
    tool: Tool,
    brush: Brush,
    channels: PaintChannels,
    stroke: Stroke,
    tex_size: u32,
    pending: Vec<PaintOp>,
    undo: Vec<Hist>,
    redo: Vec<Hist>,
    /// Document as it was before the widget gesture currently in progress.
    gesture_before: Option<Document>,
    status: String,
    render_state: egui_wgpu::RenderState,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> Self {
        let rs = cc.wgpu_render_state.clone().expect("wgpu backend required");
        let mut gpu = ViewportGpu::new(&rs.device, rs.target_format);
        let mut camera = OrbitCamera::default();
        let mut status = String::from("Select a layer and paint. Orbit: Alt+left or middle. Drop a .glb/.gltf to load.");

        let mesh = match initial.as_deref().map(|p| (p, Mesh::load_gltf(p))) {
            Some((p, Ok(m))) => {
                status = format!("Loaded {} ({} tris)", p.display(), m.indices.len() / 3);
                m
            }
            Some((p, Err(e))) => {
                status = format!("Failed to load {}: {e}", p.display());
                Mesh::uv_sphere(1.0, 96, 48)
            }
            None => Mesh::uv_sphere(1.0, 96, 48),
        };
        let (lo, hi) = mesh.bounds();
        camera.frame(lo, hi);
        gpu.set_mesh(&rs.device, &mesh);
        rs.renderer.write().callback_resources.insert(gpu);

        let doc = Document::with_default_material();
        let selected = doc.nodes.last().map(|n| n.id);
        Self {
            camera,
            doc,
            selected,
            target: Target::Content,
            tool: Tool::Paint,
            brush: Brush { radius_pts: 24.0, hardness: 0.6, opacity: 1.0, spacing: 0.15 },
            channels: PaintChannels {
                base: true,
                metal: false,
                rough: false,
                base_color: [0.15, 0.35, 0.85],
                metallic: 1.0,
                roughness: 0.2,
            },
            stroke: Stroke::default(),
            tex_size: DEFAULT_TEXTURE_SIZE,
            pending: Vec::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            gesture_before: None,
            status,
            render_state: rs,
        }
    }

    fn load(&mut self, path: PathBuf) {
        match Mesh::load_gltf(&path) {
            Ok(mesh) => {
                let (lo, hi) = mesh.bounds();
                self.camera.frame(lo, hi);
                if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
                    gpu.set_mesh(&self.render_state.device, &mesh);
                }
                // Paint belongs to the old mesh's UVs: start a fresh document.
                self.reset_document();
                self.status = format!("Loaded {} ({} tris)", path.display(), mesh.indices.len() / 3);
            }
            Err(e) => self.status = format!("Failed to load {}: {e}", path.display()),
        }
    }

    fn reset_document(&mut self) {
        self.doc = Document::with_default_material();
        self.selected = self.doc.nodes.last().map(|n| n.id);
        self.undo.clear();
        self.redo.clear();
        self.gesture_before = None;
        self.pending.push(PaintOp::Reset(self.tex_size));
    }

    // ---- history -------------------------------------------------------------------

    fn clear_redo(&mut self) {
        if !self.redo.is_empty() {
            self.redo.clear();
            self.pending.push(PaintOp::DropRedo);
        }
    }

    fn push_hist(&mut self, h: Hist) {
        self.undo.push(h);
        if self.undo.len() > MAX_HISTORY {
            self.undo.remove(0);
        }
        self.clear_redo();
    }

    /// Tell the GPU which node ids history may still resurrect, so it can free the rest.
    fn send_gc(&mut self) {
        let mut keep = self.doc.ids();
        for h in self.undo.iter().chain(&self.redo) {
            if let Hist::Doc { before, after } = h {
                keep.extend(before.ids());
                keep.extend(after.ids());
            }
        }
        keep.sort_unstable();
        keep.dedup();
        self.pending.push(PaintOp::Gc(keep));
    }

    /// Record a structural edit that already happened (`before` is the state prior to it).
    fn record(&mut self, before: Document) {
        if self.doc != before {
            let after = self.doc.clone();
            self.push_hist(Hist::Doc { before, after });
            self.send_gc();
        }
    }

    fn edit(&mut self, f: impl FnOnce(&mut Document)) {
        let before = self.doc.clone();
        f(&mut self.doc);
        self.record(before);
    }

    fn fix_selection(&mut self) {
        if self.selected.is_some_and(|id| self.doc.find(id).is_none()) {
            self.selected = None;
        }
    }

    fn do_undo(&mut self) {
        match self.undo.pop() {
            Some(Hist::Doc { before, after }) => {
                self.doc = before.clone();
                self.redo.push(Hist::Doc { before, after });
                self.fix_selection();
                self.send_gc();
            }
            Some(Hist::Stroke) => {
                self.pending.push(PaintOp::UndoStroke);
                self.redo.push(Hist::Stroke);
            }
            None => {}
        }
    }

    fn do_redo(&mut self) {
        match self.redo.pop() {
            Some(Hist::Doc { before, after }) => {
                self.doc = after.clone();
                self.undo.push(Hist::Doc { before, after });
                self.fix_selection();
                self.send_gc();
            }
            Some(Hist::Stroke) => {
                self.pending.push(PaintOp::RedoStroke);
                self.undo.push(Hist::Stroke);
            }
            None => {}
        }
    }

    // ---- painting ------------------------------------------------------------------

    /// The layer and channel set the brush would write to right now.
    fn paint_target(&self) -> Result<(NodeId, Target), &'static str> {
        let node = self.selected.and_then(|id| self.doc.find(id)).ok_or("Select a layer to paint on")?;
        match self.target {
            Target::Mask if node.mask.is_some() => Ok((node.id, Target::Mask)),
            Target::Mask => Err("This layer has no mask: add one in the layer panel"),
            Target::Content if matches!(node.kind, NodeKind::Paint) => Ok((node.id, Target::Content)),
            Target::Content => Err("Only paint layers hold paint: switch 'Paint on' to Mask, or add a paint layer"),
        }
    }

    fn make_stamp(&self, px: [f32; 2], ppp: f32, target: Target) -> Stamp {
        let c = &self.channels;
        let erase = self.tool == Tool::Erase;
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let brush = [px[0], px[1], self.brush.radius_pts * ppp, self.brush.hardness];
        match target {
            // Masks: paint white to reveal, erase to black to hide.
            Target::Mask => Stamp {
                brush,
                color: [0.0; 4],
                values: [if erase { 0.0 } else { 1.0 }, 0.0, self.brush.opacity, 0.0],
                enables: [0.0; 4],
            },
            Target::Content if erase => Stamp {
                brush,
                color: [0.0; 4],
                values: [0.0, 0.0, self.brush.opacity, 0.0],
                enables: [1.0, 1.0, 1.0, 0.0],
            },
            Target::Content => {
                let lin = c.base_color.map(srgb_to_linear);
                Stamp {
                    brush,
                    color: [lin[0], lin[1], lin[2], 0.0],
                    values: [c.metallic, c.roughness, self.brush.opacity, 0.0],
                    enables: [flag(c.base), flag(c.metal), flag(c.rough), 0.0],
                }
            }
        }
    }

    /// Advance the stroke to `to`, emitting evenly spaced stamps along the path.
    fn stroke_to(&mut self, to: [f32; 2], ppp: f32, layer: NodeId, target: Target, out: &mut Vec<Stamp>) {
        let spacing = (self.brush.radius_pts * ppp * self.brush.spacing).max(1.0);
        if !self.stroke.active {
            self.stroke = Stroke { active: true, last_px: to, next_in: spacing };
            self.pending.push(PaintOp::BeginStroke { layer, target });
            self.push_hist(Hist::Stroke);
            out.push(self.make_stamp(to, ppp, target));
            return;
        }
        let from = self.stroke.last_px;
        let delta = [to[0] - from[0], to[1] - from[1]];
        let dist = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        if dist <= f32::EPSILON {
            return;
        }
        let dir = [delta[0] / dist, delta[1] / dist];
        let mut t = self.stroke.next_in;
        while t <= dist && out.len() < MAX_STAMPS {
            out.push(self.make_stamp([from[0] + dir[0] * t, from[1] + dir[1] * t], ppp, target));
            t += spacing;
        }
        self.stroke.next_in = t - dist;
        self.stroke.last_px = to;
    }

    fn clear_target(&mut self) {
        match self.paint_target() {
            Ok((layer, target)) => {
                self.pending.push(PaintOp::BeginStroke { layer, target });
                self.pending.push(PaintOp::Clear { layer, target });
                self.push_hist(Hist::Stroke);
            }
            Err(msg) => self.status = msg.to_string(),
        }
    }

    // ---- layer panel -----------------------------------------------------------------

    fn add_node(&mut self, prefix: &str, kind: NodeKind) {
        let before = self.doc.clone();
        let (parent, index) = match self.selected.and_then(|id| self.doc.locate(id)) {
            Some((parent, i)) => (parent, Some(i + 1)),
            None => (None, None),
        };
        let name = format!("{prefix} {}", self.doc.ids().len() + 1);
        let id = self.doc.add(parent, index, &name, kind);
        self.selected = Some(id);
        self.record(before);
    }

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

    fn layer_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Layers");
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
        egui::ScrollArea::vertical().id_salt("layer_list").max_height(260.0).show(ui, |ui| {
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

        ui.separator();
        // Property widgets edit the document in place; the gesture is committed to
        // history once the pointer is released so a slider drag is a single undo step.
        let snapshot = self.doc.clone();
        let mut changed = false;
        let mut mask_action = None;
        if let Some(node) = self.doc.find_mut(id) {
            changed |= ui.text_edit_singleline(&mut node.name).changed();
            egui::ComboBox::from_id_salt("blend").selected_text(node.blend.name()).show_ui(ui, |ui| {
                for m in BlendMode::ALL {
                    changed |= ui.selectable_value(&mut node.blend, m, m.name()).changed();
                }
            });
            changed |= ui.add(egui::Slider::new(&mut node.opacity, 0.0..=1.0).text("Opacity")).changed();

            if let NodeKind::Fill(f) = &mut node.kind {
                let mut on = f.base.is_some();
                if ui.checkbox(&mut on, "Base color").changed() {
                    f.base = on.then(|| f.base.unwrap_or([0.8, 0.8, 0.8]));
                    changed = true;
                }
                if let Some(c) = &mut f.base {
                    changed |= ui.color_edit_button_rgb(c).changed();
                }
                let mut on = f.metallic.is_some();
                if ui.checkbox(&mut on, "Metallic").changed() {
                    f.metallic = on.then(|| f.metallic.unwrap_or(0.0));
                    changed = true;
                }
                if let Some(v) = &mut f.metallic {
                    changed |= ui.add(egui::Slider::new(v, 0.0..=1.0)).changed();
                }
                let mut on = f.roughness.is_some();
                if ui.checkbox(&mut on, "Roughness").changed() {
                    f.roughness = on.then(|| f.roughness.unwrap_or(0.5));
                    changed = true;
                }
                if let Some(v) = &mut f.roughness {
                    changed |= ui.add(egui::Slider::new(v, 0.0..=1.0)).changed();
                }
            }

            ui.separator();
            match &mut node.mask {
                None => {
                    ui.horizontal(|ui| {
                        if ui.button("Add mask (reveal all)").clicked() {
                            mask_action = Some(Some(true));
                        }
                        if ui.button("Add mask (hide all)").clicked() {
                            mask_action = Some(Some(false));
                        }
                    });
                }
                Some(m) => {
                    ui.horizontal(|ui| {
                        changed |= ui.checkbox(&mut m.enabled, "Mask enabled").changed();
                        if ui.button("Delete mask").clicked() {
                            mask_action = Some(None);
                        }
                    });
                }
            }
        }
        if changed && self.gesture_before.is_none() {
            self.gesture_before = Some(snapshot.clone());
        }
        if let Some(action) = mask_action {
            self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    n.mask = action.map(|init_white| Mask { enabled: true, init_white });
                }
            });
            if action.is_none() && self.target == Target::Mask {
                self.target = Target::Content;
            }
        }

        ui.separator();
        ui.label("Paint on:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.target, Target::Content, "Layer content");
            ui.radio_value(&mut self.target, Target::Mask, "Mask");
        });
        if let Err(msg) = self.paint_target() {
            ui.small(msg);
        }
    }

    fn commit_gesture(&mut self, ui: &egui::Ui) {
        if self.gesture_before.is_some() && !ui.input(|i| i.pointer.any_down()) {
            if let Some(before) = self.gesture_before.take() {
                self.record(before);
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dropped = ui.ctx().input(|i| i.raw.dropped_files.iter().find_map(|f| Some(f.path().to_path_buf())));
        if let Some(p) = dropped {
            self.load(p);
        }

        // Shortcuts: Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, [ ] brush size.
        let (undo, redo, smaller, larger) = ui.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)
                || i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y);
            let undo = i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z);
            (undo, redo, i.key_pressed(egui::Key::OpenBracket), i.key_pressed(egui::Key::CloseBracket))
        });
        if undo {
            self.do_undo();
        }
        if redo {
            self.do_redo();
        }
        if smaller {
            self.brush.radius_pts = (self.brush.radius_pts * 0.85).max(2.0);
        }
        if larger {
            self.brush.radius_pts = (self.brush.radius_pts / 0.85).min(300.0);
        }

        egui::Panel::left("properties").default_size(250.0).show(ui, |ui| {
            ui.heading("Texture set");
            ui.horizontal(|ui| {
                ui.label("Resolution");
                let before = self.tex_size;
                egui::ComboBox::from_id_salt("tex_size")
                    .selected_text(format!("{0}×{0}", self.tex_size))
                    .show_ui(ui, |ui| {
                        for s in TEXTURE_SIZES {
                            ui.selectable_value(&mut self.tex_size, s, format!("{s}×{s}"));
                        }
                    });
                if self.tex_size != before {
                    // Changing resolution discards all paint and history.
                    self.undo.clear();
                    self.redo.clear();
                    self.pending.push(PaintOp::Reset(self.tex_size));
                }
            });
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo")).clicked() {
                    self.do_undo();
                }
                if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo")).clicked() {
                    self.do_redo();
                }
                if ui.button("Clear target").clicked() {
                    self.clear_target();
                }
            });

            ui.separator();
            ui.heading("Brush");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.tool, Tool::Paint, "Paint");
                ui.radio_value(&mut self.tool, Tool::Erase, "Erase");
            });
            ui.add(egui::Slider::new(&mut self.brush.radius_pts, 2.0..=300.0).text("Size"));
            ui.add(egui::Slider::new(&mut self.brush.hardness, 0.0..=1.0).text("Hardness"));
            ui.add(egui::Slider::new(&mut self.brush.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut self.brush.spacing, 0.02..=1.0).text("Spacing"));

            ui.separator();
            ui.heading("Paint channels");
            ui.small("Used when painting layer content. Masks paint white (reveal) / erase black (hide).");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.channels.base, "Base color");
                ui.color_edit_button_rgb(&mut self.channels.base_color);
            });
            ui.checkbox(&mut self.channels.metal, "Metallic");
            ui.add_enabled(self.channels.metal, egui::Slider::new(&mut self.channels.metallic, 0.0..=1.0));
            ui.checkbox(&mut self.channels.rough, "Roughness");
            ui.add_enabled(self.channels.rough, egui::Slider::new(&mut self.channels.roughness, 0.0..=1.0));

            ui.separator();
            ui.small("Paint: left drag\nOrbit: Alt+left / middle drag\nPan: right drag\nZoom: scroll\nBrush size: [ ]\nUndo/Redo: Ctrl+Z / Ctrl+Y");
        });
        egui::Panel::right("layers").default_size(290.0).show(ui, |ui| self.layer_panel(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(&self.status);
        });
        self.commit_gesture(ui);

        egui::CentralPanel::default().show(ui, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let ppp = ui.pixels_per_point();
            let (alt, primary_down) = ui.input(|i| (i.modifiers.alt, i.pointer.primary_down()));
            let d = resp.drag_delta();

            if (resp.dragged_by(egui::PointerButton::Primary) && alt) || resp.dragged_by(egui::PointerButton::Middle) {
                self.camera.orbit(d.x, d.y);
            }
            if resp.dragged_by(egui::PointerButton::Secondary) {
                self.camera.pan(d.x, d.y, rect.height());
            }
            if resp.hovered() {
                self.camera.zoom(ui.input(|i| i.smooth_scroll_delta.y));
            }

            let mut stamps = Vec::new();
            let mut stamp_target = None;
            let painting = resp.is_pointer_button_down_on() && primary_down && !alt;
            match (painting, resp.interact_pointer_pos().or(resp.hover_pos())) {
                (true, Some(pos)) => match self.paint_target() {
                    Ok((layer, target)) => {
                        let local = [(pos.x - rect.min.x) * ppp, (pos.y - rect.min.y) * ppp];
                        self.stroke_to(local, ppp, layer, target, &mut stamps);
                        stamp_target = Some((layer, target));
                        ui.ctx().request_repaint();
                    }
                    Err(msg) => {
                        self.status = msg.to_string();
                        self.stroke.active = false;
                    }
                },
                _ => self.stroke.active = false,
            }

            let mut ops = std::mem::take(&mut self.pending);
            if let (false, Some((layer, target))) = (stamps.is_empty(), stamp_target) {
                ops.push(PaintOp::Stamps { layer, target, stamps });
            }

            let size_px = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
            let cam = self.camera.position();
            let light = Vec3::new(0.5, 0.8, 0.6).normalize();
            let (near, far) = self.camera.near_far();
            let globals = Globals {
                view_proj: self.camera.view_proj(rect.width() / rect.height().max(1.0)).to_cols_array_2d(),
                camera_pos: [cam.x, cam.y, cam.z, 1.0],
                light_dir: [light.x, light.y, light.z, 0.0],
                // Shown only where the layer stack leaves a pixel uncovered.
                base_color: [0.18, 0.18, 0.18, 1.0],
                params: [0.0, 0.5, near, far],
            };
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                ViewportCallback { globals, size_px, doc: self.doc.clone(), ops },
            ));

            if let (true, Some(pos)) = (!alt, resp.hover_pos()) {
                ui.painter().circle_stroke(pos, self.brush.radius_pts, egui::Stroke::new(1.0, egui::Color32::WHITE));
            }
        });
    }
}

fn main() -> eframe::Result {
    let initial = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 850.0]).with_title("open-painter"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("open-painter", options, Box::new(|cc| Ok(Box::new(App::new(cc, initial)))))
}
