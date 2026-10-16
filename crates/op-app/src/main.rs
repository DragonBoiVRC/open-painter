mod camera;
mod composite;
mod files;
mod gpu_io;
mod images;
mod paint;
mod normal_gpu;
mod picker;
mod procgen;
mod sets;
mod textures;
mod thumbs;
mod uvview;
mod ui;
mod viewport;

use camera::OrbitCamera;
use egui_dock::DockState;
use ui::dock::{self, Tab};
use ui::panels::layers;
use eframe::{egui, egui_wgpu};
use op_bake::{BakeSettings, Progress as BakeProgress};
use op_core::smart::{builtin_smart_materials, SmartMaterial};
use op_core::{
    srgb_to_linear, BakedMap, Document, Generator, GeneratorKind, Mask, MeshMap, NodeId, NodeKind,
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
use viewport::ViewportGpu;

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
    /// Paint height (which becomes the normal map)?
    height: bool,
    /// 0 digs, 0.5 is flat, 1 raises.
    height_value: f32,
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

/// Actions that would discard unsaved work, held until the user confirms.
#[derive(Clone)]
enum Pending {
    New,
    Open(PathBuf),
    Mesh(PathBuf),
    /// Turn UV tiles (UDIM) on or off; this restarts the texture sets.
    Udim(bool),
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

/// One texture set's pixels, read back and ready for the export thread.
struct ExportJob {
    name: String,
    /// UDIM number of the tile, when the set has UV tiles.
    udim: Option<u32>,
    size: u32,
    planes: [Vec<u8>; 5],
    /// The set's height strength, for deriving its normal map.
    height_strength: f32,
    mesh: Mesh,
    baked: Arc<HashMap<MeshMap, op_core::BakedMap>>,
}

struct ExportUi {
    open: bool,
    preset: usize,
    dir: String,
    padding: u32,
    /// Name used for a model that has a single texture set; several sets use their material names.
    set_name: String,
    /// Texture sets left out of the export (all are exported by default).
    skip_sets: std::collections::HashSet<usize>,
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
    /// The normal map made from painted height and the baked normal, shown as colour.
    PaintedNormal,
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
    save_name: String,
    message: String,
}

fn builtin_kinds() -> [GeneratorKind; 5] {
    [GeneratorKind::EdgeWear, GeneratorKind::CreviceDirt, GeneratorKind::TopDust, GeneratorKind::HeightGradient, GeneratorKind::Noise]
}

struct BakeJob {
    /// The texture set being baked.
    set: usize,
    /// One result per UV tile of the set, in tile order.
    rx: Receiver<Result<Vec<op_bake::BakeResult>, String>>,
    /// Progress of the tile being baked now.
    progress: Arc<BakeProgress>,
    tiles: usize,
    tiles_done: Arc<std::sync::atomic::AtomicUsize>,
    started: Instant,
}

impl BakeJob {
    /// Overall progress across the tiles, 0..1.
    fn fraction(&self) -> f32 {
        let done = self.tiles_done.load(std::sync::atomic::Ordering::Relaxed) as f32;
        ((done + self.progress.fraction()) / self.tiles.max(1) as f32).min(1.0)
    }
}

struct BakeUi {
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
    /// Texture sets other than the active one (the active set's slot is `None`; its state is in the fields above).
    parked: Vec<Option<sets::SetState>>,
    active: usize,
    shelf: ui::panels::ShelfState,
    /// The texture library: images imported into the shelf.
    assets: Vec<textures::TextureAsset>,
    tex_thumbs: HashMap<String, egui::TextureHandle>,
    import_rx: Option<Receiver<Vec<PathBuf>>>,
    /// Groups folded shut in the layers panel.
    collapsed: std::collections::HashSet<NodeId>,
    /// Layer being renamed in place, with its draft name.
    renaming: Option<(NodeId, String)>,
    target: Target,
    tool: Tool,
    brush: Brush,
    channels: PaintChannels,
    stroke: Stroke,
    tex_size: u32,
    pending: Vec<PaintOp>,
    undo: Vec<Hist>,
    redo: Vec<Hist>,
    /// Which texture set each undo entry belongs to, oldest first, so undo follows the order edits were made in.
    hist_order: Vec<usize>,
    redo_order: Vec<usize>,
    /// Document as it was before the widget gesture currently in progress.
    gesture_before: Option<Document>,
    status: String,
    render_state: egui_wgpu::RenderState,

    project_path: Option<PathBuf>,
    modified: bool,
    todo: Vec<Todo>,
    dialog: Option<PathDialog>,
    picker: Option<picker::Picker>,
    /// Previews of the active set's paint and masks, keyed by (layer, is_mask).
    thumbs: HashMap<(NodeId, bool), egui::TextureHandle>,
    paint_rev: u64,
    thumb_rev: u64,
    thumb_at: Instant,
    /// The 2D UV view's camera; `None` until the view is first drawn, then it frames the unit square.
    uv_cam: Option<uvview::UvCamera>,
    uv_channel: uvview::Channel,
    uv_wire: bool,
    /// The set last scrolled into view in the Texture Sets list.
    sets_scrolled: usize,
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
    /// Baked maps of each UV tile of the active set (one entry unless UV tiles are in use).
    bakes: Vec<sets::TileBake>,
    /// Background mask generation, one state per tile.
    procs: Vec<sets::TileProc>,
    /// Texture sets hidden in the 3D view; a view setting, saved with the project.
    hidden_sets: std::collections::HashSet<usize>,
    /// Whether UV tiles (UDIM) are in use: each set then has a paint store per tile its UVs occupy.
    udim: bool,
    /// Paint operations meant for one tile only, as (tile, op); sent with the next frame.
    pending_tiles: Vec<(usize, PaintOp)>,
    bake: BakeUi,
    view: View,
    custom_generators: Vec<Generator>,
    smart_materials: Vec<SmartMaterial>,
    /// Problems found while loading the user's smart materials and generator files.
    definition_warnings: Vec<String>,
    smart_ui: SmartUi,
    /// Panel arrangement; `None` only while a frame is being drawn.
    dock: Option<DockState<Tab>>,
    /// The layout as last written to disk, to avoid rewriting an unchanged file.
    layout_json: String,
    layout_check: Instant,
    show_shortcuts: bool,
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

        ui::theme::install_fonts(&cc.egui_ctx);
        ui::theme::apply(&cc.egui_ctx);
        let dock_state = dock::load(&files::layout_path()).unwrap_or_else(dock::default_layout);

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
            parked: Vec::new(),
            active: 0,
            shelf: Default::default(),
            assets: Vec::new(),
            tex_thumbs: HashMap::new(),
            import_rx: None,
            collapsed: Default::default(),
            renaming: None,
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
                height: false,
                height_value: 0.8,
            },
            stroke: Stroke::default(),
            tex_size: DEFAULT_TEXTURE_SIZE,
            pending: Vec::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            hist_order: Vec::new(),
            redo_order: Vec::new(),
            gesture_before: None,
            status: String::from("Select a layer and paint. Orbit: Alt+left or middle. Drop a .fbx, .obj, .glb/.gltf or .opproj to open."),
            render_state: rs,
            project_path: None,
            modified: false,
            todo: Vec::new(),
            dialog: None,
            picker: None,
            thumbs: HashMap::new(),
            paint_rev: 1,
            thumb_rev: 0,
            thumb_at: Instant::now() - Duration::from_secs(10),
            uv_cam: None,
            uv_channel: uvview::Channel::Base,
            uv_wire: true,
            sets_scrolled: usize::MAX,
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
                skip_sets: Default::default(),
                running: None,
                import_path: String::new(),
                report: Vec::new(),
                last_notes: Vec::new(),
            },
            bakes: vec![sets::TileBake::default()],
            bake: BakeUi {
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
            procs: vec![sets::TileProc::new()],
            udim: false,
            hidden_sets: Default::default(),
            pending_tiles: Vec::new(),
            custom_generators,
            smart_materials,
            definition_warnings,
            smart_ui: SmartUi::default(),
            dock: Some(dock_state),
            layout_json: String::new(),
            layout_check: Instant::now(),
            show_shortcuts: false,
        };
        app.load_texture_library();
        app.set_mesh(Mesh::uv_sphere(1.0, 96, 48), "sphere".into());
        app.init_sets();
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

    fn set_mesh(&mut self, mut mesh: Mesh, name: String) {
        let (lo, hi) = mesh.bounds();
        self.camera.frame(lo, hi);
        if self.udim {
            mesh.partition_tiles();
        }
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.set_mesh_tiles(&self.render_state.device, &mesh, self.udim);
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

    /// Switch between one texture per set and one texture per UDIM tile. Paint and bakes are cleared,
    /// because the textures they belong to are replaced.
    fn set_udim(&mut self, on: bool) {
        if self.udim == on {
            return;
        }
        self.udim = on;
        let (mesh, name) = (self.mesh.clone(), self.mesh_name.clone());
        self.set_mesh(mesh, name);
        self.init_sets();
        self.set_view(View::Shaded);
        self.modified = true;
        let tiles: usize = (0..self.set_ranges_len()).map(|s| self.tile_count(s)).sum();
        self.status = if on {
            format!("UV tiles are on: {tiles} texture(s) across {} set(s).", self.set_ranges_len())
        } else {
            "UV tiles are off: one texture per set.".to_string()
        };
    }

    fn new_project(&mut self) {
        self.init_sets();
        self.set_view(View::Shaded);
        self.project_path = None;
        self.modified = false;
        self.status = "New project.".into();
    }

    fn load_mesh_file(&mut self, path: &Path) {
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let loaded = match ext.as_str() {
            "fbx" => Mesh::load_fbx(path),
            "obj" => Mesh::load_obj(path),
            _ => Mesh::load_gltf(path),
        };
        match loaded {
            Ok(mesh) => {
                let tris = mesh.indices.len() / 3;
                let sets = mesh.set_ranges().len();
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mesh".into());
                self.set_mesh(mesh, name);
                // Paint belongs to the old mesh's UVs: start a fresh project.
                self.new_project();
                self.status = if sets > 1 {
                    format!("Loaded {} ({tris} tris, {sets} texture sets)", path.display())
                } else {
                    format!("Loaded {} ({tris} tris)", path.display())
                };
            }
            Err(e) => self.status = format!("Failed to load {}: {e}", path.display()),
        }
    }

    fn apply_project(&mut self, p: Project, path: Option<PathBuf>, modified: bool) {
        self.udim = p.udim;
        // The project's textures join the library (the id is the content hash, so a mismatch is ignored).
        for img in p.images {
            if op_image::content_id(&img.bytes) == img.id {
                let _ = self.add_texture(&img.name, img.bytes);
            }
        }
        self.set_mesh(p.mesh, p.mesh_name);
        let mut states = Vec::new();
        let mut loads = Vec::new();
        for s in p.sets {
            let (mut blobs, mut baked) = (Vec::new(), Vec::new());
            for t in s.tiles {
                blobs.push(t.blobs);
                baked.push(t.baked);
            }
            loads.push((s.texture_size, blobs));
            states.push(sets::SetState::restored(s.document, s.texture_size, baked));
        }
        self.install_sets(states, loads, p.active);
        self.hidden_sets = p.hidden_sets.into_iter().collect();
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
            Pending::Open(p) if is_project(&p) => self.open_path(&p),
            Pending::Open(p) => self.load_mesh_file(&p),
            Pending::Mesh(p) => self.load_mesh_file(&p),
            Pending::Udim(on) => self.set_udim(on),
            Pending::Quit => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn capture_project(&self) -> Result<Project, String> {
        let renderer = self.render_state.renderer.read();
        let gpu = renderer.callback_resources.get::<ViewportGpu>().ok_or("viewport not ready")?;
        let mut sets = Vec::new();
        for (i, unit) in gpu.sets.iter().enumerate() {
            let (doc, bakes) = if i == self.active {
                (&self.doc, &self.bakes)
            } else {
                let st = self.parked.get(i).and_then(Option::as_ref).ok_or("a texture set is missing")?;
                (&st.doc, &st.bakes)
            };
            let untouched = i != self.active && self.parked.get(i).and_then(Option::as_ref).is_some_and(|s| !s.materialized);
            let texture_size = if untouched { self.parked[i].as_ref().map_or(unit.size(), |s| s.tex_size) } else { unit.size() };
            let tiles = unit
                .tiles
                .iter()
                .enumerate()
                .map(|(t, paint)| op_project::TileData {
                    blobs: if untouched { Vec::new() } else { paint.capture_blobs(&self.render_state.device, &self.render_state.queue, doc) },
                    baked: bakes.get(t).map(|b| (*b.baked).clone()).unwrap_or_default(),
                })
                .collect();
            sets.push(op_project::SetData { texture_size, document: doc.clone(), tiles });
        }
        let mut hidden_sets: Vec<usize> = self.hidden_sets.iter().copied().collect();
        hidden_sets.sort_unstable();
        // Embed the textures any layer uses, so the project does not depend on this machine's library.
        let mut used: Vec<String> = self.doc.texture_ids();
        for st in self.parked.iter().flatten() {
            for id in st.doc.texture_ids() {
                if !used.contains(&id) {
                    used.push(id);
                }
            }
        }
        let images = used
            .iter()
            .filter_map(|id| self.texture(id))
            .map(|a| op_project::ImageData { id: a.id.clone(), name: a.name.clone(), bytes: (*a.bytes).clone() })
            .collect();
        Ok(Project { mesh_name: self.mesh_name.clone(), mesh: self.mesh.clone(), sets, active: self.active, udim: self.udim, hidden_sets, images })
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
        let ranges = self.mesh.set_ranges();
        let chosen: Vec<usize> = (0..ranges.len()).filter(|i| !self.export.skip_sets.contains(i)).collect();
        if chosen.is_empty() {
            self.status = "Select at least one texture set to export.".into();
            return;
        }
        if self.udim && !preset.separates_udim_tiles() && chosen.iter().any(|&i| self.tile_count(i) > 1) {
            self.status = "This preset's file names lack $udim, so the UV tiles would overwrite each other.".into();
            return;
        }
        if chosen.len() > 1 && !preset.separates_texture_sets() {
            self.status = "This preset's file names lack $textureSet, so the sets would overwrite each other.".into();
            return;
        }
        // Read every chosen set's textures back now; the CPU work happens off the UI thread.
        let mut jobs = Vec::new();
        for &i in &chosen {
            // A set nobody opened is held tiny; bring it to full size just long enough to read it.
            let was_open = i == self.active || self.parked.get(i).and_then(Option::as_ref).is_some_and(|s| s.materialized);
            self.ensure_materialized(i);
            {
                let renderer = self.render_state.renderer.read();
                let Some(gpu) = renderer.callback_resources.get::<ViewportGpu>() else { return };
                let Some(unit) = gpu.sets.get(i) else { continue };
                let bakes = if i == self.active { self.bakes.clone() } else { self.parked[i].as_ref().map_or_else(Default::default, |s| s.bakes.clone()) };
                let name = if ranges.len() > 1 { ranges[i].name.clone() } else { self.export.set_name.clone() };
                let height_strength = if i == self.active { self.doc.height_strength } else { self.parked[i].as_ref().map_or(1.0, |s| s.doc.height_strength) };
                for (t, paint) in unit.tiles.iter().enumerate() {
                    let mesh = if self.udim { self.mesh.extract_tile(i, paint.tile) } else { self.mesh.extract_set(i) };
                    let Some(mesh) = mesh else { continue };
                    jobs.push(ExportJob {
                        name: name.clone(),
                        udim: self.udim.then(|| paint.tile.udim()),
                        size: paint.size(),
                        planes: paint.capture_output(&self.render_state.device, &self.render_state.queue),
                        height_strength,
                        mesh,
                        baked: bakes.get(t).map(|b| b.baked.clone()).unwrap_or_default(),
                    });
                }
            }
            if !was_open {
                self.release_if_untouched(i);
            }
        }
        let dir = PathBuf::from(&self.export.dir);
        let (project, padding) = (self.project_name(), self.export.padding);
        let mesh_name = self.mesh_name.clone();
        let (tx, rx) = mpsc::channel();
        self.export.running = Some(rx);
        let what = if jobs.len() > 1 { format!("{} texture sets", jobs.len()) } else { "textures".to_string() };
        self.status = format!("Exporting {what} with '{}' to {} ...", preset.name, dir.display());
        // Padding and encoding are CPU-heavy; keep them off the UI thread.
        std::thread::spawn(move || {
            let result = (|| -> Result<ExportOutcome, String> {
                let mut total = ExportOutcome::default();
                for job in jobs {
                    let [base, metal, rough, height, layer_normal] = &job.planes;
                    let size = job.size;
                    let mut data = TextureSetData::from_composite(size, base, metal, rough).map_err(|e| e.to_string())?;
                    let mut notes = Vec::new();
                    for (map, b) in job.baked.iter() {
                        if b.size == size {
                            data.set_baked(*map, b.clone()).map_err(|e| e.to_string())?;
                        } else {
                            notes.push(format!(
                                "{} was baked at {}x{}, not {size}x{size}: bake it again to use it.",
                                map.label(),
                                b.size,
                                b.size
                            ));
                        }
                    }
                    data.set_height(height).map_err(|e| e.to_string())?;
                    data.set_layer_normal(layer_normal).map_err(|e| e.to_string())?;
                    data.compute_normal(job.height_strength);
                    if padding > 0 {
                        data.dilate(&op_export::uv_coverage(&job.mesh, size), padding);
                    }
                    let ctx = ExportContext { project: &project, texture_set: &job.name, mesh: &mesh_name, out_dir: &dir, udim: job.udim };
                    let label = match job.udim {
                        Some(n) => format!("{} ({n})", job.name),
                        None => job.name.clone(),
                    };
                    let outcome = op_export::export(&preset, &data, &ctx).map_err(|e| format!("{label}: {e}"))?;
                    total.files.extend(outcome.files);
                    total.notes.extend(outcome.notes.into_iter().chain(notes).map(|n| format!("{label}: {n}")));
                }
                Ok(total)
            })();
            let _ = tx.send(result);
        });
    }

    // ---- baking -----------------------------------------------------------------------

    fn clear_baked(&mut self) {
        self.pending.push(PaintOp::SetBakedNormal { size: self.tex_size, rgba8: None });
        for b in &mut self.bakes {
            if !b.baked.is_empty() {
                b.baked = Arc::new(HashMap::new());
                b.rev += 1;
            }
        }
        self.set_view(View::Shaded);
    }

    /// Switch what the viewport shows, uploading the baked maps it needs (one per UV tile).
    fn set_view(&mut self, view: View) {
        let wanted = |b: &sets::TileBake| match view {
            View::Shaded | View::PaintedNormal => None,
            View::ShadedAo => b.baked.get(&MeshMap::AmbientOcclusion).cloned(),
            View::Map(m) => b.baked.get(&m).cloned(),
        };
        let sources: Vec<Option<BakedMap>> = self.bakes.iter().map(wanted).collect();
        // A view none of whose maps are baked falls back to the normal shaded view.
        let view = if sources.iter().all(Option::is_none) && view != View::PaintedNormal { View::Shaded } else { view };
        let images: Vec<Option<(u32, Vec<u8>)>> = sources
            .iter()
            .map(|src| {
                src.as_ref().map(|b| {
                    let mut rgba = Vec::with_capacity(b.texels() * 4);
                    for i in 0..b.texels() {
                        let t = b.texel(i);
                        rgba.extend([t[0], t[1], t[2]].map(|v| (v * 255.0 + 0.5) as u8));
                        rgba.push(255);
                    }
                    (b.size, rgba)
                })
            })
            .collect();
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            let refs: Vec<Option<(u32, &[u8])>> = images.iter().map(|i| i.as_ref().map(|(size, px)| (*size, px.as_slice()))).collect();
            gpu.set_previews(&self.render_state.device, &self.render_state.queue, &refs);
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
        // Bake this texture set's triangles, tile by tile; the rest of the model still shades them.
        let set = self.active;
        let lows: Vec<Mesh> = if self.udim {
            self.mesh.set_tiles(set).into_iter().filter_map(|t| self.mesh.extract_tile(set, t)).collect()
        } else {
            self.mesh.extract_set(set).into_iter().collect()
        };
        if lows.is_empty() {
            return;
        }
        let tiles = lows.len();
        let occluders = (self.set_ranges_len() > 1).then(|| self.mesh.clone());
        let high = b.high.as_ref().map(|(m, _)| m.clone());
        let progress = BakeProgress::new();
        let worker_progress = progress.clone();
        let tiles_done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_done = tiles_done.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = (|| {
                let mut results = Vec::with_capacity(lows.len());
                for low in &lows {
                    results.push(
                        op_bake::bake_with_occluders(low, high.as_deref(), occluders.as_ref(), &settings, &worker_progress).map_err(|e| e.to_string())?,
                    );
                    worker_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(results)
            })();
            let _ = tx.send(result);
        });
        self.bake.job = Some(BakeJob { set, rx, progress, tiles, tiles_done, started: Instant::now() });
        let (n, size) = (self.bake.maps.len(), self.tex_size);
        let of_tiles = if tiles > 1 { format!(" ({tiles} UV tiles)") } else { String::new() };
        self.status = format!("Baking {n} map(s) for '{}'{of_tiles} at {size}x{size}...", self.set_name(set));
    }

    fn poll_bake(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.bake.job else { return };
        let (set, started) = (job.set, job.started);
        match job.rx.try_recv() {
            Ok(Ok(results)) => {
                let secs = started.elapsed().as_secs_f32();
                let n = results.first().map_or(0, |r| r.maps.len());
                let covered: usize = results.iter().map(|r| r.covered).sum();
                // The bake belongs to the set it was started for, which may no longer be the active one.
                let normals = self.with_set(set, |st| {
                    let mut ops = Vec::new();
                    for (t, result) in results.into_iter().enumerate() {
                        let Some(slot) = st.bakes.get_mut(t) else { continue };
                        let mut maps = (*slot.baked).clone();
                        maps.retain(|_, b| b.size == result.size); // older bakes at another resolution are stale
                        maps.extend(result.maps);
                        ops.push((t, sets::baked_normal_op(&maps, result.size)));
                        slot.baked = Arc::new(maps);
                        slot.rev += 1;
                    }
                    (ops, st.doc.clone(), st.materialized)
                });
                // The normal map the viewport uses changes with the bake.
                if let Some((ops, doc, materialized)) = normals {
                    for (t, op) in ops {
                        if set == self.active {
                            self.pending_tiles.push((t, op));
                        } else if materialized {
                            self.enqueue_batch(viewport::SetBatch { tile: Some(t), set, doc: doc.clone(), ops: vec![op] });
                        }
                    }
                }
                self.bake.job = None;
                self.modified = true;
                if set == self.active {
                    let view = self.view;
                    self.set_view(view);
                }
                self.status = format!("Baked {n} map(s) for '{}' in {secs:.1} s ({covered} texels shaded).", self.set_name(set));
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
        self.poll_picker(ctx);
        self.poll_texture_import(ctx);
        let used = self.doc.texture_ids();
        self.ensure_tex_thumbs(ctx, &used);
        self.refresh_thumbs(ctx);
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
        self.save_layout_if_changed();
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

    /// Remember the panel arrangement between sessions (checked every couple of seconds).
    fn save_layout_if_changed(&mut self) {
        if self.layout_check.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.layout_check = Instant::now();
        let Some(json) = self.dock.as_ref().and_then(dock::to_json) else { return };
        if json != self.layout_json {
            let path = files::layout_path();
            let written = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&path, &json));
            if written.is_ok() {
                self.layout_json = json;
            }
        }
    }

    // ---- history -------------------------------------------------------------------

    fn clear_redo(&mut self) {
        if !self.redo.is_empty() {
            self.redo.clear();
            self.pending.push(PaintOp::DropRedo);
        }
        self.drop_other_redo();
    }

    fn push_hist(&mut self, h: Hist) {
        self.undo.push(h);
        self.hist_order.push(self.active);
        if self.undo.len() > MAX_HISTORY {
            self.undo.remove(0);
            // The oldest entry of this set is the first one in the order list that names it.
            if let Some(at) = self.hist_order.iter().position(|s| *s == self.active) {
                self.hist_order.remove(at);
            }
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
            self.thumbs_dirty();
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

    /// Undo the most recent edit in any texture set, switching to its set first.
    fn do_undo(&mut self) {
        let Some(&set) = self.hist_order.last() else { return };
        if set != self.active {
            self.switch_set(set);
        }
        self.hist_order.pop();
        self.redo_order.push(set);
        self.undo_here();
    }

    fn do_redo(&mut self) {
        let Some(&set) = self.redo_order.last() else { return };
        if set != self.active {
            self.switch_set(set);
        }
        self.redo_order.pop();
        self.hist_order.push(set);
        self.redo_here();
    }

    fn can_undo(&self) -> bool {
        !self.hist_order.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo_order.is_empty()
    }

    fn undo_here(&mut self) {
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

    fn redo_here(&mut self) {
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
        self.make_stamp_raw([px[0], px[1], self.brush.radius_pts * ppp, self.brush.hardness], target)
    }

    /// A stamp with the current tool and channels at an explicit `brush` (position, radius, hardness).
    fn make_stamp_raw(&self, brush: [f32; 4], target: Target) -> Stamp {
        let c = &self.channels;
        let erase = self.tool == Tool::Erase;
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
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
                enables: [1.0, 1.0, 1.0, 1.0],
            },
            Target::Content => {
                let lin = c.base_color.map(srgb_to_linear);
                Stamp {
                    brush,
                    color: [lin[0], lin[1], lin[2], 0.0],
                    values: [c.metallic, c.roughness, self.brush.opacity, c.height_value],
                    enables: [flag(c.base), flag(c.metal), flag(c.rough), flag(c.height)],
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

    /// Like [`App::stroke_to`] in UV space: `to` is a UV position and `radius` is in UV units.
    fn stroke_to_uv(&mut self, to: [f32; 2], radius: f32, layer: NodeId, target: Target, out: &mut Vec<Stamp>) {
        let spacing = (radius * self.brush.spacing).max(1e-5);
        let hardness = self.brush.hardness;
        let at = move |p: [f32; 2]| [p[0], p[1], radius, hardness];
        if !self.stroke.active {
            self.stroke = Stroke { active: true, last_px: to, next_in: spacing };
            self.pending.push(PaintOp::BeginStroke { layer, target });
            self.push_hist(Hist::Stroke);
            out.push(self.make_stamp_raw(at(to), target));
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
            out.push(self.make_stamp_raw(at([from[0] + dir[0] * t, from[1] + dir[1] * t]), target));
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

    /// Start, collect and apply background mask generation, separately for every UV tile (a generated
    /// mask is a function of the tile's baked maps).
    fn drive_procgen(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let live = self.doc.ids();
        let mut busy = false;
        let mut wake: Option<Duration> = None;
        for t in 0..self.procs.len().min(self.bakes.len()) {
            let rev = self.bakes[t].rev;
            for req in self.procs[t].scheduler.plan(&self.doc, rev, self.tex_size, now) {
                let (baked, generator, size) = (self.bakes[t].baked.clone(), req.key.generator.clone(), req.key.size);
                let cancel = Arc::new(AtomicBool::new(false));
                let flag = cancel.clone();
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(op_proc::generate(&generator, size, &baked, &flag).map_err(|e| e.to_string()));
                });
                self.procs[t].jobs.push(ProcJob { layer: req.layer, key: req.key, rx, cancel });
            }

            let mut finished = Vec::new();
            self.procs[t].jobs.retain(|job| match job.rx.try_recv() {
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
                        if self.procs[t].scheduler.finish(layer, &key) {
                            self.pending_tiles.push((t, PaintOp::SetMask { layer, size: key.size, bytes: op_proc::to_mask_bytes(&eval.values) }));
                            self.procs[t].info.insert(layer, ProcInfo { warnings: eval.warnings, error: None });
                        }
                    }
                    Err(e) => {
                        self.procs[t].scheduler.failed(layer, &key);
                        self.procs[t].info.insert(layer, ProcInfo { warnings: Vec::new(), error: Some(e) });
                    }
                }
            }
            self.procs[t].info.retain(|id, _| live.contains(id));
            busy |= !self.procs[t].jobs.is_empty();
            if let Some(w) = self.procs[t].scheduler.next_wakeup(now) {
                wake = Some(wake.map_or(w, |x| x.min(w)));
            }
        }
        if busy {
            ctx.request_repaint_after(Duration::from_millis(60));
        } else if let Some(wait) = wake {
            ctx.request_repaint_after(wait + Duration::from_millis(5));
        }
    }

    /// The operations to send with this frame: those for the whole set, then those for single tiles.
    fn take_batches(&mut self, mut ops: Vec<PaintOp>) -> Vec<viewport::SetBatch> {
        self.ensure_doc_images(&self.doc);
        ops.splice(0..0, std::mem::take(&mut self.pending));
        let mut batches = vec![viewport::SetBatch { tile: None, set: self.active, doc: self.doc.clone(), ops }];
        for (tile, op) in std::mem::take(&mut self.pending_tiles) {
            batches.push(viewport::SetBatch { tile: Some(tile), set: self.active, doc: self.doc.clone(), ops: vec![op] });
        }
        batches
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
        let needs_bakes = !self.bakes[0].baked.contains_key(&MeshMap::Curvature);
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

    fn commit_gesture(&mut self, ui: &egui::Ui) {
        if self.gesture_before.is_some() && !ui.input(|i| i.pointer.any_down()) {
            if let Some(before) = self.gesture_before.take() {
                self.record(before);
            }
        }
    }

    // ---- file bar and dialogs -----------------------------------------------------------

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
        if self.show_shortcuts {
            let mut open = true;
            egui::Window::new("Keyboard shortcuts").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                egui::Grid::new("shortcut_grid").num_columns(2).spacing([24.0, 4.0]).show(ui, |ui| {
                    for (keys, what) in [
                        ("Ctrl+N / Ctrl+O / Ctrl+S", "new / open / save"),
                        ("Ctrl+Z / Ctrl+Y", "undo / redo"),
                        ("Ctrl+D / Ctrl+G", "duplicate / group the selected layer"),
                        ("Delete / F2", "delete / rename the selected layer"),
                        ("Ctrl+PageUp / PageDown", "previous / next texture set"),
                        ("Ctrl + click", "pick the texture set under the cursor"),
                        ("H / Shift+H", "hide or show the active material / show all"),
                        ("P / E", "paint / erase"),
                        ("[  ]", "smaller / larger brush"),
                        ("F", "frame the model"),
                        ("Alt + left drag", "orbit"),
                        ("Middle drag", "pan"),
                        ("Alt + right drag, scroll", "zoom"),
                        ("Ctrl + right drag", "brush size and hardness"),
                    ] {
                        ui.label(egui::RichText::new(keys).strong());
                        ui.label(what);
                        ui.end_row();
                    }
                });
            });
            self.show_shortcuts = open;
        }
        if let Some(mut d) = self.dialog.take() {
            let mut keep = true;
            let mut accepted = false;
            let mut browse = false;
            let busy = self.picker.is_some();
            egui::Window::new(if d.save { "Save project as" } else { "Open project or model" })
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Path:");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut d.path).desired_width(360.0));
                        browse = picker::browse_button(ui, busy);
                    });
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
            if browse {
                let field = if d.save { picker::Field::SavePath } else { picker::Field::OpenPath };
                self.browse(field, &d.path);
            }
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
                Pending::Udim(_) => "change the UV tile setting, which restarts the texture sets",
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
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.export.dir).desired_width(330.0));
                    if picker::browse_button(ui, self.picker.is_some()) {
                        let cur = self.export.dir.clone();
                        self.browse(picker::Field::ExportDir, &cur);
                    }
                });
                let ranges = self.mesh.set_ranges();
                if ranges.len() > 1 {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Texture sets:");
                        if ui.small_button("All").clicked() {
                            self.export.skip_sets.clear();
                        }
                        if ui.small_button("None").clicked() {
                            self.export.skip_sets = (0..ranges.len()).collect();
                        }
                    });
                    egui::ScrollArea::vertical().id_salt("export_sets").max_height(140.0).show(ui, |ui| {
                        for (i, r) in ranges.iter().enumerate() {
                            let mut on = !self.export.skip_sets.contains(&i);
                            if ui.checkbox(&mut on, format!("{}  ({} tris)", r.name, r.index_count / 3)).changed() {
                                if on {
                                    self.export.skip_sets.remove(&i);
                                } else {
                                    self.export.skip_sets.insert(i);
                                }
                            }
                        }
                    });
                    if !self.presets[idx].separates_texture_sets() {
                        ui.colored_label(egui::Color32::YELLOW, "This preset's file names have no $textureSet: only one set can be exported at a time.");
                    }
                } else {
                    ui.horizontal(|ui| {
                        ui.label("Texture set name:");
                        ui.text_edit_singleline(&mut self.export.set_name);
                    });
                }
                ui.add(egui::Slider::new(&mut self.export.padding, 0..=64).text("Edge padding (texels)"));
                let busy = self.export.running.is_some();
                let chosen = (0..ranges.len()).filter(|i| !self.export.skip_sets.contains(i)).count();
                let collides = chosen > 1 && !self.presets[idx].separates_texture_sets();
                let ready = !busy && !self.export.dir.trim().is_empty() && chosen > 0 && !collides;
                let label = if ranges.len() > 1 { format!("Export {chosen} texture set(s)") } else { "Export".to_string() };
                if ui.add_enabled(ready, egui::Button::new(label)).clicked() {
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
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.export.import_path).desired_width(330.0));
                    if picker::browse_button(ui, self.picker.is_some()) {
                        let cur = self.export.import_path.clone();
                        self.browse(picker::Field::PresetImport, &cur);
                    }
                });
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

impl App {
    fn handle_shortcuts(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        use egui::{Key, Modifiers};
        let (undo, redo, save, new, open, quit, dup, group, next_set, prev_set) = ui.input_mut(|i| {
            let redo = i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z) || i.consume_key(Modifiers::COMMAND, Key::Y);
            let undo = i.consume_key(Modifiers::COMMAND, Key::Z);
            let save = i.consume_key(Modifiers::COMMAND, Key::S);
            let new = i.consume_key(Modifiers::COMMAND, Key::N);
            let open = i.consume_key(Modifiers::COMMAND, Key::O);
            let quit = i.consume_key(Modifiers::COMMAND, Key::Q);
            let dup = i.consume_key(Modifiers::COMMAND, Key::D);
            let group = i.consume_key(Modifiers::COMMAND, Key::G);
            let next_set = i.consume_key(Modifiers::COMMAND, Key::PageDown);
            let prev_set = i.consume_key(Modifiers::COMMAND, Key::PageUp);
            (undo, redo, save, new, open, quit, dup, group, next_set, prev_set)
        });
        if dup {
            self.layer_command(layers::LayerCmd::Duplicate);
        }
        if group {
            self.layer_command(layers::LayerCmd::Group);
        }
        let sets = self.set_ranges_len();
        if next_set && sets > 1 {
            self.switch_set((self.active + 1) % sets);
        }
        if prev_set && sets > 1 {
            self.switch_set((self.active + sets - 1) % sets);
        }
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
        if new {
            self.request(Pending::New, ctx);
        }
        if open {
            let start = self.project_path.clone().map(|p| p.display().to_string()).unwrap_or_default();
            self.dialog = Some(PathDialog { save: false, path: start });
        }
        if quit {
            self.request(Pending::Quit, ctx);
        }

        // Single-key shortcuts must not fire while a text field has focus.
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (hide, show_all) = ui.input(|i| (i.key_pressed(Key::H) && !i.modifiers.any(), i.key_pressed(Key::H) && i.modifiers.shift && !i.modifiers.command));
        if show_all {
            self.show_all_sets();
        } else if hide && self.set_ranges_len() > 1 {
            self.toggle_set_visible(self.active);
        }
        let (delete, rename) = ui.input(|i| (i.key_pressed(Key::Delete), i.key_pressed(Key::F2)));
        if delete {
            self.layer_command(layers::LayerCmd::Delete);
        }
        if rename {
            self.layer_command(layers::LayerCmd::Rename);
        }
        let (smaller, larger, paint, erase, frame) = ui.input(|i| {
            (
                i.key_pressed(Key::OpenBracket),
                i.key_pressed(Key::CloseBracket),
                i.key_pressed(Key::P),
                i.key_pressed(Key::E),
                i.key_pressed(Key::F) && !i.modifiers.any(),
            )
        });
        if smaller {
            self.brush.radius_pts = (self.brush.radius_pts * 0.85).max(2.0);
        }
        if larger {
            self.brush.radius_pts = (self.brush.radius_pts / 0.85).min(300.0);
        }
        if paint {
            self.tool = Tool::Paint;
        }
        if erase {
            self.tool = Tool::Erase;
        }
        if frame {
            self.frame_model();
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.housekeeping(&ctx);

        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| Some(f.path().to_path_buf())).collect());
        // Image files go to the texture library; the first other file is a model or project to open.
        let (images, others): (Vec<PathBuf>, Vec<PathBuf>) = dropped.into_iter().partition(|p| textures::is_image_path(p));
        if !images.is_empty() {
            self.import_texture_files(&images);
        }
        if let Some(p) = others.into_iter().next() {
            let action = if is_project(&p) { Pending::Open(p) } else { Pending::Mesh(p) };
            self.request(action, &ctx);
        }

        self.handle_shortcuts(ui, &ctx);
        self.ui_frame(ui);
        // A drag that ended anywhere that did not take it is over.
        if ctx.input(|i| !i.pointer.any_down()) {
            egui::DragAndDrop::clear_payload(&ctx);
        }
    }
}

fn main() -> eframe::Result {
    let initial = std::env::args_os().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 850.0])
            .with_title("open-painter")
            // Lets the desktop match the window to the launcher entry (taskbar icon, grouping).
            .with_app_id("open-painter")
            .with_icon(egui::IconData { rgba: include_bytes!("icon.rgba").to_vec(), width: 64, height: 64 }),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("open-painter", options, Box::new(|cc| Ok(Box::new(App::new(cc, initial)))))
}
