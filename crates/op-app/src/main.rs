mod camera;
mod composite;
mod files;
mod gpu_io;
mod paint;
mod procgen;
mod viewport;

use camera::OrbitCamera;
use eframe::{egui, egui_wgpu};
use glam::Vec3;
use op_bake::{BakeSettings, Progress as BakeProgress};
use op_core::smart::{builtin_smart_materials, SmartMaterial};
use op_core::{
    srgb_to_linear, BakedMap, BlendMode, Document, FillParams, Generator, GeneratorKind, Mask, MeshMap, Node, NodeId,
    NodeKind, FALLBACK_BASE, FALLBACK_METALLIC, FALLBACK_ROUGHNESS,
};
use op_export::{ExportContext, ExportOutcome, ExportPreset, TextureSetData};
use op_mesh::Mesh;
use op_project::Project;
use paint::{PaintOp, Stamp, Target, DEFAULT_TEXTURE_SIZE, MAX_STAMPS};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use viewport::{Globals, ViewportCallback, ViewportGpu};

const TEXTURE_SIZES: [u32; 4] = [512, 1024, 2048, 4096];
const MAX_HISTORY: usize = 300;
const DEFAULT_AUTOSAVE_SECS: u64 = 120;
const DEFAULT_PADDING: u32 = 8;

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

/// Actions that would discard unsaved work, held until the user confirms.
#[derive(Clone)]
enum Pending {
    New,
    Open(PathBuf),
    Mesh(PathBuf),
    Quit,
}

/// Work that reads GPU state. Runs at the start of the next frame, when every op queued
/// so far has been applied, so it never sees a half-updated scene.
enum Todo {
    Save(PathBuf),
    Export,
}

struct PathDialog {
    save: bool,
    path: String,
}

struct ExportUi {
    open: bool,
    preset: usize,
    dir: String,
    padding: u32,
    set_name: String,
    running: Option<Receiver<Result<ExportOutcome, String>>>,
    /// Path typed into the "import a Painter preset" field, and what the last import did.
    import_path: String,
    report: Vec<String>,
    /// Notes from the last export (for example maps that were not baked).
    last_notes: Vec<String>,
}

/// What the viewport shows.
#[derive(Clone, Copy, PartialEq)]
enum View {
    Shaded,
    /// Shaded, with the baked ambient occlusion darkening the ambient light.
    ShadedAo,
    /// A baked map, shown as-is.
    Map(MeshMap),
}

/// A mask generation running on a worker thread.
struct ProcJob {
    layer: NodeId,
    key: procgen::Key,
    rx: Receiver<Result<op_proc::Evaluation, String>>,
    #[allow(dead_code)]
    cancel: Arc<AtomicBool>,
}

/// What the layer panel shows about a generated mask.
#[derive(Default)]
struct ProcInfo {
    warnings: Vec<String>,
    error: Option<String>,
}

/// Changes to a mask's generator that need more than editing a field in place.
enum GenAction {
    /// Remove the generator but keep the generated pixels, so the mask can be painted.
    Freeze,
    /// Replace a painted mask's content by a generator (the old content is snapshotted for undo).
    Replace(Generator),
    /// Give a layer a brand-new generated mask.
    NewMask(Generator),
}

#[derive(Default)]
struct SmartUi {
    pick: usize,
    save_name: String,
    message: String,
}

fn builtin_kinds() -> [GeneratorKind; 5] {
    [GeneratorKind::EdgeWear, GeneratorKind::CreviceDirt, GeneratorKind::TopDust, GeneratorKind::HeightGradient, GeneratorKind::Noise]
}

struct BakeJob {
    rx: Receiver<Result<op_bake::BakeResult, String>>,
    progress: Arc<BakeProgress>,
    started: Instant,
}

struct BakeUi {
    open: bool,
    maps: Vec<MeshMap>,
    samples: u32,
    ao_distance_pct: f32,
    thickness_pct: f32,
    curvature_strength: f32,
    cage_front_pct: f32,
    cage_rear_pct: f32,
    high_path: String,
    high: Option<(Arc<Mesh>, String)>,
    job: Option<BakeJob>,
}

struct App {
    camera: OrbitCamera,
    doc: Document,
    mesh: Mesh,
    mesh_name: String,
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

    project_path: Option<PathBuf>,
    modified: bool,
    todo: Vec<Todo>,
    dialog: Option<PathDialog>,
    confirm: Option<Pending>,
    after_save: Option<Pending>,
    allow_close: bool,
    title: String,
    last_autosave: Instant,
    autosave_every: Duration,
    autosave_rx: Option<Receiver<Result<(), String>>>,
    presets: Vec<ExportPreset>,
    preset_warnings: Vec<String>,
    export: ExportUi,
    baked: Arc<HashMap<MeshMap, BakedMap>>,
    bake: BakeUi,
    view: View,
    /// Bumped whenever the baked maps change, so generated masks know to regenerate.
    bake_rev: u64,
    proc: procgen::Scheduler,
    proc_jobs: Vec<ProcJob>,
    proc_info: HashMap<NodeId, ProcInfo>,
    custom_generators: Vec<Generator>,
    smart_materials: Vec<SmartMaterial>,
    /// Problems found while loading the user's smart materials and generator files.
    definition_warnings: Vec<String>,
    smart_ui: SmartUi,
}

fn is_project(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case(op_project::EXTENSION))
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> Self {
        let rs = cc.wgpu_render_state.clone().expect("wgpu backend required");
        let gpu = ViewportGpu::new(&rs.device, &rs.queue, rs.target_format);
        rs.renderer.write().callback_resources.insert(gpu);

        let (user_presets, preset_warnings) = files::load_user_presets(&files::presets_dir());
        let mut presets = op_export::builtin_presets();
        presets.extend(user_presets);
        let autosave_every = std::env::var("OPEN_PAINTER_AUTOSAVE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|s| *s > 0)
            .unwrap_or(DEFAULT_AUTOSAVE_SECS);

        let (user_smart, mut definition_warnings) = files::load_smart_materials(&files::smart_materials_dir());
        let (custom_generators, generator_warnings) = files::load_custom_generators(&files::generators_dir());
        definition_warnings.extend(generator_warnings);
        let mut smart_materials = builtin_smart_materials();
        for mut m in user_smart {
            // A user file never replaces a built-in one silently.
            while smart_materials.iter().any(|b| b.name == m.name) {
                m.name.push_str(" (user)");
            }
            smart_materials.push(m);
        }

        let doc = Document::with_default_material();
        let selected = doc.nodes.last().map(|n| n.id);
        let mut app = Self {
            camera: OrbitCamera::default(),
            doc,
            mesh: Mesh::default(),
            mesh_name: "sphere".into(),
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
            status: String::from("Select a layer and paint. Orbit: Alt+left or middle. Drop a .glb/.gltf or .opproj to open."),
            render_state: rs,
            project_path: None,
            modified: false,
            todo: Vec::new(),
            dialog: None,
            confirm: None,
            after_save: None,
            allow_close: false,
            title: String::new(),
            last_autosave: Instant::now(),
            autosave_every: Duration::from_secs(autosave_every),
            autosave_rx: None,
            presets,
            preset_warnings,
            export: ExportUi {
                open: false,
                preset: 0,
                dir: String::new(),
                padding: DEFAULT_PADDING,
                set_name: "default".into(),
                running: None,
                import_path: String::new(),
                report: Vec::new(),
                last_notes: Vec::new(),
            },
            baked: Arc::new(HashMap::new()),
            bake: BakeUi {
                open: false,
                maps: MeshMap::ALL.to_vec(),
                samples: 32,
                ao_distance_pct: 10.0,
                thickness_pct: 25.0,
                curvature_strength: 1.0,
                cage_front_pct: 5.0,
                cage_rear_pct: 5.0,
                high_path: String::new(),
                high: None,
                job: None,
            },
            view: View::Shaded,
            bake_rev: 0,
            proc: procgen::Scheduler::new(Duration::from_millis(150)),
            proc_jobs: Vec::new(),
            proc_info: HashMap::new(),
            custom_generators,
            smart_materials,
            definition_warnings,
            smart_ui: SmartUi::default(),
        };
        app.set_mesh(Mesh::uv_sphere(1.0, 96, 48), "sphere".into());
        app.pending.push(PaintOp::Reset(app.tex_size));
        if let Some(p) = initial {
            if is_project(&p) {
                app.open_path(&p);
            } else {
                app.load_mesh_file(&p);
            }
        } else if files::data_dir().join(op_project::autosave_path(None, Path::new("")).file_name().unwrap()).exists() {
            app.status = "An autosave from a previous session exists: use 'Recover autosave'.".into();
        }
        app
    }

    // ---- project lifecycle -----------------------------------------------------------

    fn set_mesh(&mut self, mesh: Mesh, name: String) {
        let (lo, hi) = mesh.bounds();
        self.camera.frame(lo, hi);
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.set_mesh(&self.render_state.device, &mesh);
        }
        self.mesh = mesh;
        self.mesh_name = name;
    }

    fn project_name(&self) -> String {
        self.project_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".into())
    }

    fn reset_document(&mut self) {
        self.doc = Document::with_default_material();
        self.selected = self.doc.nodes.last().map(|n| n.id);
        self.undo.clear();
        self.redo.clear();
        self.gesture_before = None;
        self.stroke.active = false;
        self.pending.clear();
        self.pending.push(PaintOp::Reset(self.tex_size));
        self.proc.invalidate();
        self.proc_info.clear();
    }

    fn new_project(&mut self) {
        self.reset_document();
        self.clear_baked();
        self.project_path = None;
        self.modified = false;
        self.status = "New project.".into();
    }

    fn load_mesh_file(&mut self, path: &Path) {
        match Mesh::load_gltf(path) {
            Ok(mesh) => {
                let tris = mesh.indices.len() / 3;
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mesh".into());
                self.set_mesh(mesh, name);
                // Paint belongs to the old mesh's UVs: start a fresh project.
                self.new_project();
                self.status = format!("Loaded {} ({tris} tris)", path.display());
            }
            Err(e) => self.status = format!("Failed to load {}: {e}", path.display()),
        }
    }

    fn apply_project(&mut self, p: Project, path: Option<PathBuf>, modified: bool) {
        self.set_mesh(p.mesh, p.mesh_name);
        self.doc = p.document;
        self.tex_size = p.texture_size;
        self.selected = self.doc.nodes.last().map(|n| n.id);
        self.undo.clear();
        self.redo.clear();
        self.gesture_before = None;
        self.stroke.active = false;
        self.pending.clear();
        self.pending.push(PaintOp::Load { size: p.texture_size, blobs: p.blobs });
        self.baked = Arc::new(p.baked);
        self.bake_rev += 1;
        self.proc.invalidate();
        self.proc_info.clear();
        self.set_view(View::Shaded);
        self.project_path = path;
        self.modified = modified;
    }

    fn open_path(&mut self, path: &Path) {
        match op_project::load(path) {
            Ok(p) => {
                self.apply_project(p, Some(path.to_path_buf()), false);
                self.status = format!("Opened {}", path.display());
            }
            Err(e) => self.status = format!("Could not open {}: {e}", path.display()),
        }
    }

    fn recover_path(&self) -> PathBuf {
        op_project::autosave_path(self.project_path.as_deref(), &files::data_dir())
    }

    fn recover(&mut self) {
        let path = self.recover_path();
        match op_project::load(&path) {
            Ok(p) => {
                let original = self.project_path.clone();
                self.apply_project(p, original, true);
                self.status = format!("Recovered autosave {}. Save to keep it.", path.display());
            }
            Err(e) => self.status = format!("Could not recover {}: {e}", path.display()),
        }
    }

    fn request(&mut self, action: Pending, ctx: &egui::Context) {
        if self.modified {
            self.confirm = Some(action);
        } else {
            self.perform(action, ctx);
        }
    }

    fn perform(&mut self, action: Pending, ctx: &egui::Context) {
        match action {
            Pending::New => self.new_project(),
            Pending::Open(p) => self.open_path(&p),
            Pending::Mesh(p) => self.load_mesh_file(&p),
            Pending::Quit => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn capture_project(&self) -> Result<Project, String> {
        let renderer = self.render_state.renderer.read();
        let gpu = renderer.callback_resources.get::<ViewportGpu>().ok_or("viewport not ready")?;
        let blobs = gpu.paint.capture_blobs(&self.render_state.device, &self.render_state.queue, &self.doc);
        Ok(Project {
            texture_size: gpu.paint.size(),
            document: self.doc.clone(),
            mesh_name: self.mesh_name.clone(),
            mesh: self.mesh.clone(),
            blobs,
            baked: (*self.baked).clone(),
        })
    }

    fn do_save(&mut self, mut path: PathBuf, ctx: &egui::Context) {
        if path.extension().is_none() {
            path.set_extension(op_project::EXTENSION);
        }
        let result = self.capture_project().and_then(|p| op_project::save(&path, &p).map_err(|e| e.to_string()));
        match result {
            Ok(()) => {
                // The explicit save supersedes any autosave of this project (or of the untitled one).
                let _ = std::fs::remove_file(self.recover_path());
                self.project_path = Some(path.clone());
                let _ = std::fs::remove_file(self.recover_path());
                self.modified = false;
                self.status = format!("Saved {}", path.display());
                if let Some(next) = self.after_save.take() {
                    self.perform(next, ctx);
                }
            }
            Err(e) => {
                self.after_save = None;
                self.status = format!("Save failed: {e}");
            }
        }
    }

    fn start_autosave(&mut self) {
        self.last_autosave = Instant::now();
        let path = self.recover_path();
        let project = match self.capture_project() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Autosave failed: {e}");
                return;
            }
        };
        let (tx, rx) = mpsc::channel();
        self.autosave_rx = Some(rx);
        std::thread::spawn(move || {
            let result = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .map_err(|e| e.to_string())
                .and_then(|()| op_project::save(&path, &project).map_err(|e| e.to_string()));
            let _ = tx.send(result);
        });
    }

    fn start_export(&mut self) {
        if self.export.running.is_some() {
            return;
        }
        let Some(preset) = self.presets.get(self.export.preset).cloned() else { return };
        let (size, [base, metal, rough]) = {
            let renderer = self.render_state.renderer.read();
            let Some(gpu) = renderer.callback_resources.get::<ViewportGpu>() else { return };
            (gpu.paint.size(), gpu.paint.capture_output(&self.render_state.device, &self.render_state.queue))
        };
        let dir = PathBuf::from(&self.export.dir);
        let (project, set, padding) = (self.project_name(), self.export.set_name.clone(), self.export.padding);
        let mesh = self.mesh.clone();
        let mesh_name = self.mesh_name.clone();
        let baked = self.baked.clone();
        let (tx, rx) = mpsc::channel();
        self.export.running = Some(rx);
        self.status = format!("Exporting '{}' to {} ...", preset.name, dir.display());
        // Padding and encoding are CPU-heavy; keep them off the UI thread.
        std::thread::spawn(move || {
            let result = (|| -> Result<ExportOutcome, String> {
                let mut data = TextureSetData::from_composite(size, &base, &metal, &rough).map_err(|e| e.to_string())?;
                let mut stale = Vec::new();
                for (map, b) in baked.iter() {
                    if b.size == size {
                        data.set_baked(*map, b.clone()).map_err(|e| e.to_string())?;
                    } else {
                        stale.push(format!(
                            "{} was baked at {}x{}, not {size}x{size}: bake it again to use it.",
                            map.label(),
                            b.size,
                            b.size
                        ));
                    }
                }
                if padding > 0 {
                    data.dilate(&op_export::uv_coverage(&mesh, size), padding);
                }
                let ctx = ExportContext { project: &project, texture_set: &set, mesh: &mesh_name, out_dir: &dir };
                let mut outcome = op_export::export(&preset, &data, &ctx).map_err(|e| e.to_string())?;
                outcome.notes.extend(stale);
                Ok(outcome)
            })();
            let _ = tx.send(result);
        });
    }

    // ---- baking -----------------------------------------------------------------------

    fn clear_baked(&mut self) {
        if !self.baked.is_empty() {
            self.baked = Arc::new(HashMap::new());
            self.bake_rev += 1;
        }
        self.set_view(View::Shaded);
    }

    /// Switch what the viewport shows, uploading the baked map it needs.
    fn set_view(&mut self, view: View) {
        let source = match view {
            View::Shaded => None,
            View::ShadedAo => self.baked.get(&MeshMap::AmbientOcclusion),
            View::Map(m) => self.baked.get(&m),
        };
        // A view whose map is no longer baked falls back to the normal shaded view.
        let view = if source.is_none() { View::Shaded } else { view };
        let image = source.map(|b| {
            let mut rgba = Vec::with_capacity(b.texels() * 4);
            for i in 0..b.texels() {
                let t = b.texel(i);
                rgba.extend([t[0], t[1], t[2]].map(|v| (v * 255.0 + 0.5) as u8));
                rgba.push(255);
            }
            (b.size, rgba)
        });
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.set_preview(
                &self.render_state.device,
                &self.render_state.queue,
                image.as_ref().map(|(size, px)| (*size, px.as_slice())),
            );
        }
        self.view = view;
    }

    fn load_high_poly(&mut self) {
        let path = PathBuf::from(self.bake.high_path.trim());
        match Mesh::load_gltf(&path) {
            Ok(mesh) => {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.status = format!("High-poly '{name}' loaded ({} tris).", mesh.indices.len() / 3);
                self.bake.high = Some((Arc::new(mesh), name));
            }
            Err(e) => self.status = format!("Could not load the high-poly mesh: {e}"),
        }
    }

    fn start_bake(&mut self) {
        if self.bake.job.is_some() {
            return;
        }
        if self.bake.maps.is_empty() {
            self.status = "Select at least one map to bake.".into();
            return;
        }
        let b = &self.bake;
        let settings = BakeSettings {
            size: self.tex_size,
            maps: b.maps.clone(),
            samples: b.samples,
            ao_distance: b.ao_distance_pct / 100.0,
            thickness_distance: b.thickness_pct / 100.0,
            curvature_strength: b.curvature_strength,
            cage_front: b.cage_front_pct / 100.0,
            cage_rear: b.cage_rear_pct / 100.0,
            padding: 16,
        };
        let low = self.mesh.clone();
        let high = b.high.as_ref().map(|(m, _)| m.clone());
        let progress = BakeProgress::new();
        let worker_progress = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = op_bake::bake(&low, high.as_deref(), &settings, &worker_progress).map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.bake.job = Some(BakeJob { rx, progress, started: Instant::now() });
        self.status = format!("Baking {} map(s) at {}x{}...", self.bake.maps.len(), self.tex_size, self.tex_size);
    }

    fn poll_bake(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.bake.job else { return };
        match job.rx.try_recv() {
            Ok(Ok(result)) => {
                let secs = job.started.elapsed().as_secs_f32();
                let mut maps = (*self.baked).clone();
                maps.retain(|_, b| b.size == result.size); // older bakes at another resolution are stale
                let n = result.maps.len();
                maps.extend(result.maps);
                self.baked = Arc::new(maps);
                self.bake_rev += 1;
                self.bake.job = None;
                self.modified = true;
                let view = self.view;
                self.set_view(view);
                self.status = format!("Baked {n} map(s) in {secs:.1} s ({} texels shaded).", result.covered);
            }
            Ok(Err(e)) => {
                self.bake.job = None;
                self.status = format!("Bake stopped: {e}");
            }
            Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(100)),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.bake.job = None;
                self.status = "The bake ended unexpectedly.".into();
            }
        }
    }

    fn import_preset(&mut self) {
        let path = PathBuf::from(self.export.import_path.trim());
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                self.export.report = vec![format!("Could not read {}: {e}", path.display())];
                return;
            }
        };
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "preset".into());
        match op_export::import_spexp(&bytes, &stem) {
            Err(e) => self.export.report = vec![format!("Import failed: {e}")],
            Ok(mut imp) => {
                // Never shadow a built-in preset.
                let builtin: Vec<String> = op_export::builtin_presets().into_iter().map(|p| p.name).collect();
                if builtin.contains(&imp.preset.name) {
                    imp.preset.name.push_str(" (imported)");
                }
                let mut report = vec![format!("Imported '{}' with {} map(s).", imp.preset.name, imp.preset.maps.len())];
                report.extend(imp.warnings.iter().cloned());
                let file_stem: String = imp
                    .preset
                    .name
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() || "-_ ".contains(c) { c } else { '_' })
                    .collect();
                let dir = files::presets_dir();
                let file = dir.join(format!("{}.json", file_stem.trim()));
                let saved = std::fs::create_dir_all(&dir)
                    .and_then(|()| std::fs::write(&file, serde_json::to_vec_pretty(&imp.preset).unwrap_or_default()));
                report.push(match saved {
                    Ok(()) => format!("Saved a copy to {} so it is available next time.", file.display()),
                    Err(e) => format!("Could not save a copy ({e}); the preset stays available until you close the app."),
                });
                match self.presets.iter().position(|p| p.name == imp.preset.name) {
                    Some(i) => {
                        self.presets[i] = imp.preset;
                        self.export.preset = i;
                    }
                    None => {
                        self.presets.push(imp.preset);
                        self.export.preset = self.presets.len() - 1;
                    }
                }
                self.export.report = report;
            }
        }
    }

    /// Runs at the start of every frame: queued GPU reads, autosave timer, background results.
    fn housekeeping(&mut self, ctx: &egui::Context) {
        for t in std::mem::take(&mut self.todo) {
            match t {
                Todo::Save(p) => self.do_save(p, ctx),
                Todo::Export => self.start_export(),
            }
        }
        if self.modified
            && !self.stroke.active
            && self.autosave_rx.is_none()
            && self.last_autosave.elapsed() >= self.autosave_every
        {
            self.start_autosave();
        }
        if let Some(rx) = &self.autosave_rx {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    self.status = format!("Autosaved to {}", self.recover_path().display());
                    self.autosave_rx = None;
                }
                Ok(Err(e)) => {
                    self.status = format!("Autosave failed: {e}");
                    self.autosave_rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(250)),
                Err(mpsc::TryRecvError::Disconnected) => self.autosave_rx = None,
            }
        }
        self.poll_bake(ctx);
        self.drive_procgen(ctx);
        if let Some(rx) = &self.export.running {
            match rx.try_recv() {
                Ok(result) => {
                    self.status = match result {
                        Ok(out) => {
                            let extra = if out.notes.is_empty() { String::new() } else { format!(" ({} note(s): see Export window)", out.notes.len()) };
                            self.export.last_notes = out.notes;
                            format!("Exported {} file(s) to {}{extra}", out.files.len(), self.export.dir)
                        }
                        Err(e) => format!("Export failed: {e}"),
                    };
                    self.export.running = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(250)),
                Err(mpsc::TryRecvError::Disconnected) => self.export.running = None,
            }
        }
        // Keep the autosave timer ticking even when the user is idle.
        if self.modified {
            ctx.request_repaint_after(Duration::from_secs(1));
        }

        let title = format!("{}{} - open-painter", self.project_name(), if self.modified { " *" } else { "" });
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
        if ctx.input(|i| i.viewport().close_requested()) && self.modified && !self.allow_close {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm = Some(Pending::Quit);
        }
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
        self.modified = true;
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
                self.modified = true;
            }
            Some(Hist::Stroke) => {
                self.pending.push(PaintOp::UndoStroke);
                self.redo.push(Hist::Stroke);
                self.modified = true;
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
                self.modified = true;
            }
            Some(Hist::Stroke) => {
                self.pending.push(PaintOp::RedoStroke);
                self.undo.push(Hist::Stroke);
                self.modified = true;
            }
            None => {}
        }
    }

    // ---- painting ------------------------------------------------------------------

    /// The layer and channel set the brush would write to right now.
    fn paint_target(&self) -> Result<(NodeId, Target), &'static str> {
        let node = self.selected.and_then(|id| self.doc.find(id)).ok_or("Select a layer to paint on")?;
        match self.target {
            Target::Mask if node.mask.as_ref().is_some_and(|m| m.generator.is_some()) => {
                Err("This mask is generated from baked maps: use 'Freeze' to make it paintable")
            }
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
        let mut gen_action: Option<GenAction> = None;
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
                    egui::ComboBox::from_id_salt("gen_new").selected_text("Add generated mask...").show_ui(ui, |ui| {
                        for k in builtin_kinds() {
                            if ui.selectable_label(false, op_proc::label(&k)).clicked() {
                                gen_action = Some(GenAction::NewMask(Generator::new(k)));
                            }
                        }
                        for c in &self.custom_generators {
                            if ui.selectable_label(false, op_proc::label(&c.kind)).clicked() {
                                gen_action = Some(GenAction::NewMask(c.clone()));
                            }
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
                    match &mut m.generator {
                        Some(gen) => {
                            ui.label("Generated from the baked maps:");
                            egui::ComboBox::from_id_salt("gen_kind").selected_text(op_proc::label(&gen.kind)).show_ui(ui, |ui| {
                                for k in builtin_kinds() {
                                    if ui.selectable_label(gen.kind == k, op_proc::label(&k)).clicked() && gen.kind != k {
                                        gen.kind = k;
                                        gen.params.clear();
                                        changed = true;
                                    }
                                }
                                for c in &self.custom_generators {
                                    if ui.selectable_label(gen.kind == c.kind, op_proc::label(&c.kind)).clicked() && gen.kind != c.kind {
                                        gen.kind = c.kind.clone();
                                        gen.params.clear();
                                        changed = true;
                                    }
                                }
                            });
                            for spec in op_proc::param_specs(&gen.kind) {
                                let mut v = op_proc::value(gen, spec.key);
                                let mut slider = egui::Slider::new(&mut v, spec.min..=spec.max).text(spec.label);
                                if spec.integer {
                                    slider = slider.step_by(1.0);
                                }
                                if ui.add(slider).changed() {
                                    gen.params.insert(spec.key.to_string(), v);
                                    changed = true;
                                }
                            }
                            changed |= ui.checkbox(&mut gen.invert, "Invert").changed();
                            let missing: Vec<&str> = op_proc::required_maps(&gen.kind)
                                .into_iter()
                                .filter(|m| !self.baked.contains_key(m))
                                .map(|m| m.label())
                                .collect();
                            if !missing.is_empty() {
                                ui.colored_label(
                                    egui::Color32::YELLOW,
                                    format!("Not baked yet: {}. Use Bake... to create them.", missing.join(", ")),
                                );
                            }
                            if self.proc.is_busy(id) {
                                ui.label("Generating...");
                            }
                            if let Some(info) = self.proc_info.get(&id) {
                                for w in &info.warnings {
                                    ui.small(w);
                                }
                                if let Some(e) = &info.error {
                                    ui.colored_label(egui::Color32::LIGHT_RED, format!("Generation failed: {e}"));
                                }
                            }
                            if ui.button("Freeze (make paintable)").clicked() {
                                gen_action = Some(GenAction::Freeze);
                            }
                        }
                        None => {
                            egui::ComboBox::from_id_salt("gen_replace").selected_text("Replace with generated...").show_ui(ui, |ui| {
                                for k in builtin_kinds() {
                                    if ui.selectable_label(false, op_proc::label(&k)).clicked() {
                                        gen_action = Some(GenAction::Replace(Generator::new(k)));
                                    }
                                }
                                for c in &self.custom_generators {
                                    if ui.selectable_label(false, op_proc::label(&c.kind)).clicked() {
                                        gen_action = Some(GenAction::Replace(c.clone()));
                                    }
                                }
                            });
                            ui.small("Replacing is undoable: Ctrl+Z restores the painted mask.");
                        }
                    }
                }
            }
        }
        if changed && self.gesture_before.is_none() {
            self.gesture_before = Some(snapshot.clone());
        }
        if let Some(action) = gen_action {
            self.apply_gen_action(id, action);
        }
        if let Some(action) = mask_action {
            self.edit(|d| {
                if let Some(n) = d.find_mut(id) {
                    n.mask = action.map(Mask::new);
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

    fn apply_gen_action(&mut self, id: NodeId, action: GenAction) {
        let set = |d: &mut Document, g: Option<Generator>, keep_mask: bool| {
            if let Some(n) = d.find_mut(id) {
                match (&mut n.mask, g) {
                    (Some(m), g) if keep_mask => m.generator = g,
                    (slot, Some(g)) => *slot = Some(Mask::generated(g)),
                    _ => {}
                }
            }
        };
        match action {
            GenAction::Freeze => self.edit(|d| set(d, None, true)),
            GenAction::NewMask(g) => self.edit(|d| set(d, Some(g), false)),
            GenAction::Replace(g) => {
                // The generated content overwrites the painted mask, so snapshot it first: the
                // document edit undoes first, then this snapshot restores the pixels.
                self.pending.push(PaintOp::BeginStroke { layer: id, target: Target::Mask });
                self.push_hist(Hist::Stroke);
                self.edit(|d| set(d, Some(g), true));
            }
        }
    }

    /// Start, collect and apply background mask generation.
    fn drive_procgen(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        for req in self.proc.plan(&self.doc, self.bake_rev, self.tex_size, now) {
            let (baked, generator, size) = (self.baked.clone(), req.key.generator.clone(), req.key.size);
            let cancel = Arc::new(AtomicBool::new(false));
            let flag = cancel.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(op_proc::generate(&generator, size, &baked, &flag).map_err(|e| e.to_string()));
            });
            self.proc_jobs.push(ProcJob { layer: req.layer, key: req.key, rx, cancel });
        }

        let mut finished = Vec::new();
        self.proc_jobs.retain(|job| match job.rx.try_recv() {
            Err(mpsc::TryRecvError::Empty) => true,
            Ok(result) => {
                finished.push((job.layer, job.key.clone(), result));
                false
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                finished.push((job.layer, job.key.clone(), Err("the worker stopped unexpectedly".into())));
                false
            }
        });
        for (layer, key, result) in finished {
            match result {
                Ok(eval) => {
                    if self.proc.finish(layer, &key) {
                        self.pending.push(PaintOp::SetMask { layer, size: key.size, bytes: op_proc::to_mask_bytes(&eval.values) });
                        self.proc_info.insert(layer, ProcInfo { warnings: eval.warnings, error: None });
                    }
                }
                Err(e) => {
                    self.proc.failed(layer, &key);
                    self.proc_info.insert(layer, ProcInfo { warnings: Vec::new(), error: Some(e) });
                }
            }
        }
        let live = self.doc.ids();
        self.proc_info.retain(|id, _| live.contains(id));

        if !self.proc_jobs.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(60));
        } else if let Some(wait) = self.proc.next_wakeup(now) {
            ctx.request_repaint_after(wait + Duration::from_millis(5));
        }
    }

    fn add_smart_material(&mut self, index: usize) {
        let Some(material) = self.smart_materials.get(index).cloned() else { return };
        let before = self.doc.clone();
        let (parent, at) = match self.selected.and_then(|id| self.doc.locate(id)) {
            Some((parent, i)) => (parent, Some(i + 1)),
            None => (None, None),
        };
        let ids = self.doc.insert_nodes(parent, at, material.nodes);
        self.selected = ids.last().copied();
        self.record(before);
        let needs_bakes = !self.baked.contains_key(&MeshMap::Curvature);
        self.smart_ui.message = if needs_bakes {
            format!("Added '{}'. Its generated masks need baked maps: use Bake...", material.name)
        } else {
            format!("Added '{}'.", material.name)
        };
    }

    fn save_smart_material(&mut self) {
        let name = self.smart_ui.save_name.trim().to_string();
        let Some(node) = self.selected.and_then(|id| self.doc.extract(id)) else {
            self.smart_ui.message = "Select a layer or group to save.".into();
            return;
        };
        let material = SmartMaterial { name: name.clone(), nodes: vec![node] };
        let json = match material.to_json() {
            Ok(j) => j,
            Err(e) => {
                self.smart_ui.message = format!("Cannot save: {e}");
                return;
            }
        };
        let stem: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect();
        let dir = files::smart_materials_dir();
        let file = dir.join(format!("{}.json", stem.trim()));
        match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file, json)) {
            Ok(()) => {
                match self.smart_materials.iter().position(|m| m.name == name) {
                    Some(i) => self.smart_materials[i] = material,
                    None => self.smart_materials.push(material),
                }
                self.smart_ui.message = format!("Saved '{name}' to {} (paint layers are saved empty).", file.display());
            }
            Err(e) => self.smart_ui.message = format!("Could not save: {e}"),
        }
    }

    fn smart_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Smart materials");
        let current = self.smart_materials.get(self.smart_ui.pick).map_or("(none)", |m| m.name.as_str()).to_string();
        egui::ComboBox::from_id_salt("smart_pick").selected_text(current).show_ui(ui, |ui| {
            for (i, m) in self.smart_materials.iter().enumerate() {
                ui.selectable_value(&mut self.smart_ui.pick, i, &m.name);
            }
        });
        if ui.button("Add to layer stack").clicked() {
            self.add_smart_material(self.smart_ui.pick);
        }
        ui.small("Smart materials use masks generated from the baked maps (Bake...).");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.smart_ui.save_name).hint_text("name").desired_width(120.0));
            if ui.add_enabled(!self.smart_ui.save_name.trim().is_empty(), egui::Button::new("Save selection")).clicked() {
                self.save_smart_material();
            }
        });
        if !self.smart_ui.message.is_empty() {
            ui.small(&self.smart_ui.message);
        }
        for w in &self.definition_warnings {
            ui.colored_label(egui::Color32::YELLOW, format!("Skipped: {w}"));
        }
    }

    fn commit_gesture(&mut self, ui: &egui::Ui) {
        if self.gesture_before.is_some() && !ui.input(|i| i.pointer.any_down()) {
            if let Some(before) = self.gesture_before.take() {
                self.record(before);
            }
        }
    }

    // ---- file bar and dialogs -----------------------------------------------------------

    fn file_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button("New").clicked() {
                self.request(Pending::New, &ctx);
            }
            if ui.button("Open...").clicked() {
                let start = self.project_path.clone().map(|p| p.display().to_string()).unwrap_or_default();
                self.dialog = Some(PathDialog { save: false, path: start });
            }
            if ui.button("Save").clicked() {
                match self.project_path.clone() {
                    Some(p) => {
                        self.todo.push(Todo::Save(p));
                        ctx.request_repaint();
                    }
                    None => self.dialog = Some(PathDialog { save: true, path: self.default_save_path() }),
                }
            }
            if ui.button("Save as...").clicked() {
                self.dialog = Some(PathDialog { save: true, path: self.default_save_path() });
            }
            let recoverable = self.recover_path().exists();
            if ui.add_enabled(recoverable, egui::Button::new("Recover autosave")).clicked() {
                self.recover();
            }
            ui.separator();
            if ui.button("Bake...").clicked() {
                self.bake.open = true;
            }
            let mut view = self.view;
            let label = |v: View| match v {
                View::Shaded => "Shaded".to_string(),
                View::ShadedAo => "Shaded + baked AO".to_string(),
                View::Map(m) => m.label().to_string(),
            };
            egui::ComboBox::from_id_salt("viewmode").selected_text(format!("View: {}", label(self.view))).show_ui(ui, |ui| {
                ui.selectable_value(&mut view, View::Shaded, "Shaded");
                if self.baked.contains_key(&MeshMap::AmbientOcclusion) {
                    ui.selectable_value(&mut view, View::ShadedAo, "Shaded + baked AO");
                }
                for m in MeshMap::ALL {
                    if self.baked.contains_key(&m) {
                        ui.selectable_value(&mut view, View::Map(m), m.label());
                    }
                }
            });
            if view != self.view {
                self.set_view(view);
            }
            ui.separator();
            if ui.button("Export...").clicked() {
                if self.export.dir.is_empty() {
                    self.export.dir = self.default_export_dir().display().to_string();
                }
                self.export.open = true;
            }
        });
    }

    fn default_save_path(&self) -> String {
        match &self.project_path {
            Some(p) => p.display().to_string(),
            None => std::env::current_dir()
                .unwrap_or_default()
                .join(format!("untitled.{}", op_project::EXTENSION))
                .display()
                .to_string(),
        }
    }

    fn default_export_dir(&self) -> PathBuf {
        match self.project_path.as_ref().and_then(|p| p.parent()) {
            Some(d) => d.join("export"),
            None => std::env::current_dir().unwrap_or_default().join("export"),
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        if let Some(mut d) = self.dialog.take() {
            let mut keep = true;
            let mut accepted = false;
            egui::Window::new(if d.save { "Save project as" } else { "Open project" })
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Path:");
                    ui.add(egui::TextEdit::singleline(&mut d.path).desired_width(460.0));
                    if d.save && Path::new(&d.path).exists() {
                        ui.colored_label(egui::Color32::YELLOW, "This file exists and will be replaced.");
                    }
                    ui.horizontal(|ui| {
                        if ui.button(if d.save { "Save" } else { "Open" }).clicked() {
                            accepted = true;
                        }
                        if ui.button("Cancel").clicked() {
                            keep = false;
                        }
                    });
                });
            if accepted && !d.path.trim().is_empty() {
                let path = PathBuf::from(d.path.trim());
                if d.save {
                    self.todo.push(Todo::Save(path));
                    ctx.request_repaint();
                } else {
                    self.request(Pending::Open(path), ctx);
                }
            } else if keep && !accepted {
                self.dialog = Some(d);
            } else if !keep {
                self.after_save = None;
            }
        }

        if let Some(action) = self.confirm.clone() {
            let what = match action {
                Pending::Quit => "quit",
                Pending::New => "start a new project",
                _ => "open another file",
            };
            egui::Window::new("Unsaved changes").collapsible(false).resizable(false).show(ctx, |ui| {
                ui.label(format!("The project has unsaved changes. Save before you {what}?"));
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.confirm = None;
                        self.after_save = Some(action.clone());
                        match self.project_path.clone() {
                            Some(p) => {
                                self.todo.push(Todo::Save(p));
                                ctx.request_repaint();
                            }
                            None => self.dialog = Some(PathDialog { save: true, path: self.default_save_path() }),
                        }
                    }
                    if ui.button("Discard changes").clicked() {
                        self.confirm = None;
                        self.modified = false;
                        self.perform(action.clone(), ctx);
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm = None;
                    }
                });
            });
        }

        if self.bake.open {
            let mut open = true;
            egui::Window::new("Bake mesh maps").open(&mut open).collapsible(false).show(ctx, |ui| {
                let busy = self.bake.job.is_some();
                ui.label(format!("Resolution: {}x{} (the texture set resolution)", self.tex_size, self.tex_size));
                ui.separator();
                ui.label("Maps:");
                ui.horizontal_wrapped(|ui| {
                    for m in MeshMap::ALL {
                        let mut on = self.bake.maps.contains(&m);
                        if ui.add_enabled(!busy, egui::Checkbox::new(&mut on, m.label())).changed() {
                            if on {
                                self.bake.maps.push(m);
                            } else {
                                self.bake.maps.retain(|x| *x != m);
                            }
                        }
                    }
                });
                ui.add_enabled_ui(!busy, |ui| {
                    ui.add(egui::Slider::new(&mut self.bake.samples, 4..=256).logarithmic(true).text("Rays per texel"));
                    ui.add(egui::Slider::new(&mut self.bake.ao_distance_pct, 1.0..=100.0).text("AO distance (% of size)"));
                    ui.add(egui::Slider::new(&mut self.bake.thickness_pct, 1.0..=100.0).text("Thickness distance (% of size)"));
                    ui.add(egui::Slider::new(&mut self.bake.curvature_strength, 0.1..=8.0).logarithmic(true).text("Curvature strength"));
                    ui.separator();
                    ui.label("High-poly source (optional): normals, height and opacity are projected from it.");
                    ui.add(egui::TextEdit::singleline(&mut self.bake.high_path).desired_width(420.0));
                    ui.horizontal(|ui| {
                        if ui.add_enabled(!self.bake.high_path.trim().is_empty(), egui::Button::new("Load high-poly")).clicked() {
                            self.load_high_poly();
                        }
                        if self.bake.high.is_some() && ui.button("Remove").clicked() {
                            self.bake.high = None;
                        }
                    });
                    match &self.bake.high {
                        Some((m, name)) => ui.label(format!("Using {name} ({} triangles)", m.indices.len() / 3)),
                        None => ui.label("None: maps come from the low-poly mesh itself."),
                    };
                    ui.add(egui::Slider::new(&mut self.bake.cage_front_pct, 0.5..=50.0).text("Search in front (% of size)"));
                    ui.add(egui::Slider::new(&mut self.bake.cage_rear_pct, 0.5..=50.0).text("Search behind (% of size)"));
                });
                ui.separator();
                match &self.bake.job {
                    Some(job) => {
                        ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
                        ui.label(format!("{:.0} s elapsed", job.started.elapsed().as_secs_f32()));
                        if ui.button("Cancel").clicked() {
                            job.progress.cancel();
                        }
                    }
                    None => {
                        if ui.add_enabled(!self.bake.maps.is_empty(), egui::Button::new("Bake")).clicked() {
                            self.start_bake();
                        }
                    }
                }
                if !self.baked.is_empty() {
                    let mut names: Vec<_> = self.baked.keys().map(|m| m.label()).collect();
                    names.sort_unstable();
                    ui.small(format!("Baked: {}", names.join(", ")));
                }
            });
            self.bake.open = open;
        }

        if self.export.open {
            let mut open = true;
            egui::Window::new("Export textures").open(&mut open).collapsible(false).show(ctx, |ui| {
                let idx = self.export.preset.min(self.presets.len() - 1);
                egui::ComboBox::from_label("Preset").selected_text(&self.presets[idx].name).show_ui(ui, |ui| {
                    for (i, p) in self.presets.iter().enumerate() {
                        ui.selectable_value(&mut self.export.preset, i, &p.name);
                    }
                });
                ui.small("Files:");
                for m in &self.presets[idx].maps {
                    ui.small(format!("  {}.{}", m.file, m.format.extension()));
                }
                ui.label("Folder:");
                ui.add(egui::TextEdit::singleline(&mut self.export.dir).desired_width(420.0));
                ui.horizontal(|ui| {
                    ui.label("Texture set name:");
                    ui.text_edit_singleline(&mut self.export.set_name);
                });
                ui.add(egui::Slider::new(&mut self.export.padding, 0..=64).text("Edge padding (texels)"));
                let busy = self.export.running.is_some();
                if ui.add_enabled(!busy && !self.export.dir.trim().is_empty(), egui::Button::new("Export")).clicked() {
                    self.todo.push(Todo::Export);
                    ctx.request_repaint();
                }
                if busy {
                    ui.label("Exporting...");
                }
                for w in &self.preset_warnings {
                    ui.colored_label(egui::Color32::YELLOW, format!("Preset skipped: {w}"));
                }
                for n in &self.export.last_notes {
                    ui.colored_label(egui::Color32::YELLOW, n);
                }
                ui.separator();
                ui.label("Import a Substance Painter preset (.spexp):");
                ui.add(egui::TextEdit::singleline(&mut self.export.import_path).desired_width(420.0));
                if ui.add_enabled(!self.export.import_path.trim().is_empty(), egui::Button::new("Import preset")).clicked() {
                    self.import_preset();
                }
                if !self.export.report.is_empty() {
                    egui::ScrollArea::vertical().id_salt("import_report").max_height(180.0).show(ui, |ui| {
                        for (i, line) in self.export.report.iter().enumerate() {
                            if i == 0 {
                                ui.label(line);
                            } else {
                                ui.colored_label(egui::Color32::YELLOW, line);
                            }
                        }
                    });
                }
            });
            self.export.open = open;
        }
    }
}

fn srgb_for_display(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 1.0]
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.housekeeping(&ctx);

        let dropped = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| Some(f.path().to_path_buf())));
        if let Some(p) = dropped {
            let action = if is_project(&p) { Pending::Open(p) } else { Pending::Mesh(p) };
            self.request(action, &ctx);
        }

        // Shortcuts: Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo, Ctrl+S save, [ ] brush size.
        let (undo, redo, save, smaller, larger) = ui.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)
                || i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y);
            let undo = i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z);
            let save = i.consume_key(egui::Modifiers::COMMAND, egui::Key::S);
            (undo, redo, save, i.key_pressed(egui::Key::OpenBracket), i.key_pressed(egui::Key::CloseBracket))
        });
        if undo {
            self.do_undo();
        }
        if redo {
            self.do_redo();
        }
        if save {
            match self.project_path.clone() {
                Some(p) => self.todo.push(Todo::Save(p)),
                None => self.dialog = Some(PathDialog { save: true, path: self.default_save_path() }),
            }
        }
        if smaller {
            self.brush.radius_pts = (self.brush.radius_pts * 0.85).max(2.0);
        }
        if larger {
            self.brush.radius_pts = (self.brush.radius_pts / 0.85).min(300.0);
        }

        egui::Panel::top("file_bar").show(ui, |ui| self.file_bar(ui));
        egui::Panel::left("properties").default_size(250.0).show(ui, |ui| {
            ui.heading("Texture set");
            ui.horizontal(|ui| {
                ui.label("Resolution");
                let before = self.tex_size;
                egui::ComboBox::from_id_salt("tex_size")
                    .selected_text(format!("{0}x{0}", self.tex_size))
                    .show_ui(ui, |ui| {
                        for s in TEXTURE_SIZES {
                            ui.selectable_value(&mut self.tex_size, s, format!("{s}x{s}"));
                        }
                    });
                if self.tex_size != before {
                    // Changing resolution discards all paint and history.
                    self.undo.clear();
                    self.redo.clear();
                    self.pending.push(PaintOp::Reset(self.tex_size));
                    // Baked maps are tied to the resolution they were baked at.
                    self.clear_baked();
                    self.proc.invalidate();
                    self.modified = true;
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
            ui.small("Paint: left drag\nOrbit: Alt+left / middle drag\nPan: right drag\nZoom: scroll\nBrush size: [ ]\nUndo/Redo: Ctrl+Z / Ctrl+Y\nSave: Ctrl+S");
        });
        egui::Panel::right("layers").default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("right_scroll").auto_shrink([false, false]).show(ui, |ui| {
                self.layer_panel(ui);
                ui.separator();
                self.smart_panel(ui);
            });
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(&self.status);
        });
        self.commit_gesture(ui);
        self.dialogs(&ctx);

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
            // A modal dialog must not paint through to the viewport underneath.
            let modal_open = self.dialog.is_some() || self.confirm.is_some();
            let painting = resp.is_pointer_button_down_on() && primary_down && !alt && !modal_open;
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
                light_dir: [
                    light.x,
                    light.y,
                    light.z,
                    match self.view {
                        View::Shaded => 0.0,
                        View::Map(_) => 1.0,
                        View::ShadedAo => 2.0,
                    },
                ],
                // Shown only where the layer stack leaves a pixel uncovered.
                base_color: srgb_for_display(FALLBACK_BASE),
                params: [FALLBACK_METALLIC, FALLBACK_ROUGHNESS, near, far],
            };
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                ViewportCallback { globals, size_px, doc: self.doc.clone(), ops },
            ));

            if let (true, Some(pos)) = (!alt && !modal_open, resp.hover_pos()) {
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
