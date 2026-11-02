//! Texture sets: a mesh with several materials is edited one set at a time. Each set has its own
//! layer stack, undo history, baked maps and texture resolution. The active set's state lives in
//! the `App` fields the rest of the program already uses; the others are parked here and swapped in
//! when the user picks them.

use crate::spill::Saved;
use crate::viewport::{SetBatch, ViewportGpu};
use crate::paint::ProjectionData;
use crate::{App, Hist, PaintOp, DEFAULT_TEXTURE_SIZE};
use op_core::{BakedMap, Document, MeshMap, NodeId};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Size of the paint store of a set nobody has touched.
pub(crate) const PLACEHOLDER_SIZE: u32 = 64;
/// Size of the picture of its finished look that a set keeps while it is not being edited.
const PREVIEW_SIZE: u32 = 512;

/// What the program keeps track of to stay inside the GPU and RAM: which sets were used when, which
/// still need their preview made, and how much of each kind of memory paint may take.
pub struct Memory {
    clock: u64,
    used_at: HashMap<usize, u64>,
    /// Sets waiting for their preview picture, made one at a time after a project opens.
    queue: Vec<usize>,
    /// GPU bytes the texture sets may hold before the ones not in use are moved out.
    gpu_budget: u64,
    /// Bytes of paint kept in memory for sets off the GPU before the rest goes to disk.
    ram_budget: u64,
}

impl Memory {
    /// Budgets from the machine: half the video memory and half the free RAM. Either can be set with
    /// `OPEN_PAINTER_GPU_BUDGET_MB` / `OPEN_PAINTER_RAM_BUDGET_MB`.
    pub fn detect() -> Self {
        let mb = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok()).map(|v| v << 20);
        let gpu_budget = mb("OPEN_PAINTER_GPU_BUDGET_MB").unwrap_or_else(|| detect_vram().unwrap_or(4 << 30) / 2);
        let ram_budget = mb("OPEN_PAINTER_RAM_BUDGET_MB").unwrap_or_else(|| available_ram().unwrap_or(8 << 30) / 2);
        remove_stale_spill(&crate::files::data_dir().join("spill"));
        Self { clock: 0, used_at: HashMap::new(), queue: Vec::new(), gpu_budget, ram_budget }
    }

    fn touch(&mut self, set: usize) {
        self.clock += 1;
        self.used_at.insert(set, self.clock);
    }

    fn last_used(&self, set: usize) -> u64 {
        self.used_at.get(&set).copied().unwrap_or(0)
    }
}

/// Total video memory, if we can find out (NVIDIA's tool, or the kernel's AMD counters).
fn detect_vram() -> Option<u64> {
    if !cfg!(windows) {
        let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]).output().ok()?;
        if let Some(mib) = String::from_utf8_lossy(&out.stdout).lines().next().and_then(|l| l.trim().parse::<u64>().ok()) {
            return Some(mib << 20);
        }
        let dirs = std::fs::read_dir("/sys/class/drm").ok()?;
        return dirs.flatten().filter_map(|d| std::fs::read_to_string(d.path().join("device/mem_info_vram_total")).ok()).filter_map(|t| t.trim().parse().ok()).max();
    }
    None
}

/// Delete spill files left by a copy of the program that is no longer running (names start with its pid).
fn remove_stale_spill(dir: &std::path::Path) {
    if !cfg!(target_os = "linux") {
        return;
    }
    for f in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let pid = f.file_name().to_string_lossy().split('-').next().map(str::to_owned);
        if pid.is_some_and(|p| !std::path::Path::new("/proc").join(p).exists()) {
            let _ = std::fs::remove_file(f.path());
        }
    }
}

fn available_ram() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = info.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb << 10)
}

/// Baked maps of one UV tile of a set, and a revision that changes whenever they do (generated
/// masks regenerate when it moves).
#[derive(Clone, Default)]
pub struct TileBake {
    pub baked: Arc<HashMap<MeshMap, BakedMap>>,
    pub rev: u64,
}

/// Background mask generation for one tile: it depends on that tile's baked maps.
pub struct TileProc {
    pub scheduler: crate::procgen::Scheduler,
    pub jobs: Vec<crate::ProcJob>,
    pub info: HashMap<NodeId, crate::ProcInfo>,
}

impl TileProc {
    pub fn new() -> Self {
        Self { scheduler: crate::procgen::Scheduler::new(std::time::Duration::from_millis(150)), jobs: Vec::new(), info: HashMap::new() }
    }

    fn cancel(&self) {
        for j in &self.jobs {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }
}

/// Everything that belongs to one texture set while it is not the active one.
pub struct SetState {
    pub doc: Document,
    pub selected: Option<NodeId>,
    pub undo: Vec<Hist>,
    pub redo: Vec<Hist>,
    /// Baked maps per UV tile (one entry unless UV tiles are in use).
    pub bakes: Vec<TileBake>,
    pub tex_size: u32,
    pub collapsed: HashSet<NodeId>,
    /// False while the set's paint store is held at the small placeholder size: nothing has been
    /// painted in it, so it looks the same and costs almost no GPU memory.
    pub materialized: bool,
    /// The set's paint (per tile) while it is not on the GPU: kept in memory, uploaded when opened.
    pub saved: Option<Saved>,
}

impl SetState {
    /// A set restored from a saved project.
    pub fn restored(doc: Document, tex_size: u32, baked: Vec<HashMap<MeshMap, BakedMap>>) -> Self {
        let selected = doc.nodes.last().map(|n| n.id);
        let bakes = baked.into_iter().map(|b| TileBake { baked: Arc::new(b), rev: 1 }).collect();
        Self { doc, selected, undo: Vec::new(), redo: Vec::new(), bakes, tex_size, collapsed: HashSet::new(), materialized: true, saved: None }
    }

    /// A new set with the default material layer.
    pub fn fresh(tex_size: u32, tiles: usize) -> Self {
        let doc = Document::with_default_material();
        let selected = doc.nodes.last().map(|n| n.id);
        Self {
            doc,
            selected,
            undo: Vec::new(),
            redo: Vec::new(),
            bakes: vec![TileBake::default(); tiles.max(1)],
            tex_size,
            collapsed: HashSet::new(),
            materialized: false,
            saved: None,
        }
    }
}

/// The sets to hide so that only `i` is shown; or nothing hidden if `i` is already alone.
pub fn isolate(hidden: &HashSet<usize>, i: usize, count: usize) -> HashSet<usize> {
    let alone = !hidden.contains(&i) && (0..count).filter(|s| *s != i).all(|s| hidden.contains(&s));
    if alone { HashSet::new() } else { (0..count).filter(|s| *s != i).collect() }
}

/// Where a baked mesh sits in the whole model: its box's start and size, in units of the model's
/// longest side measured from the model's minimum corner.
pub type BakeFrame = ([f32; 3], [f32; 3]);

/// The frame of a part of the model (its bounding box) inside the whole model's box.
pub fn bake_frame_of(whole: (glam::Vec3, glam::Vec3), part: (glam::Vec3, glam::Vec3)) -> BakeFrame {
    let largest = (whole.1 - whole.0).max_element().max(1e-6);
    (((part.0 - whole.0) / largest).to_array(), ((part.1 - part.0) / largest).to_array())
}

/// The op that gives a set's paint store its baked normal map and the maps projected fills read
/// (or removes them).
pub fn baked_normal_op(baked: &HashMap<MeshMap, BakedMap>, size: u32, frame: BakeFrame) -> PaintOp {
    let rgba8 = baked.get(&MeshMap::NormalBase).filter(|b| b.size == size && b.channels == 3).map(|b| {
        let mut out = Vec::with_capacity(b.texels() * 4);
        for i in 0..b.texels() {
            let t = b.texel(i);
            out.extend(t.map(|v| (v * 255.0 + 0.5) as u8));
            out.push(255);
        }
        out
    });
    let usable = |m: MeshMap| baked.get(&m).filter(|b| b.size == size && b.channels == 3);
    let projection = usable(MeshMap::Position).zip(usable(MeshMap::WorldSpaceNormal)).map(|(pos, nrm)| {
        let mut position = Vec::with_capacity(pos.texels() * 8);
        let mut normal = Vec::with_capacity(pos.texels() * 4);
        for i in 0..pos.texels() {
            let p = pos.texel(i);
            for v in [p[0], p[1], p[2], 1.0] {
                position.extend(half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            normal.extend(nrm.texel(i).map(|v| (v * 255.0 + 0.5) as u8));
            normal.push(255);
        }
        ProjectionData { position, normal, offset: frame.0, extent: frame.1.map(|e| e.max(1e-3)) }
    });
    PaintOp::SetBakedNormal { size, rgba8, projection }
}

/// True if a set's bakes include anything the paint store wants uploaded.
pub fn has_uploadable_bakes(baked: &HashMap<MeshMap, BakedMap>) -> bool {
    baked.contains_key(&MeshMap::NormalBase) || (baked.contains_key(&MeshMap::Position) && baked.contains_key(&MeshMap::WorldSpaceNormal))
}

impl App {
    /// Where the mesh a tile of `set` is baked from sits in the whole model (what a Position bake is
    /// normalised by), in units of the model's longest side from its minimum corner.
    pub(crate) fn bake_extent(&self, set: usize, tile: usize) -> BakeFrame {
        let low = if self.udim {
            self.mesh.set_tiles(set).get(tile).and_then(|t| self.mesh.extract_tile(set, *t))
        } else {
            self.mesh.extract_set(set)
        };
        match low {
            Some(m) => bake_frame_of(self.mesh.bounds(), m.bounds()),
            None => ([0.0; 3], [1.0; 3]),
        }
    }

    /// Per-set "hidden" flags for the 3D view.
    pub(crate) fn hidden_flags(&self) -> Vec<bool> {
        (0..self.parked.len()).map(|i| self.hidden_sets.contains(&i)).collect()
    }

    pub(crate) fn toggle_set_visible(&mut self, i: usize) {
        if !self.hidden_sets.remove(&i) {
            self.hidden_sets.insert(i);
        }
        self.modified = true;
    }

    pub(crate) fn isolate_set(&mut self, i: usize) {
        self.hidden_sets = isolate(&self.hidden_sets, i, self.parked.len());
        self.modified = true;
    }

    pub(crate) fn show_all_sets(&mut self) {
        if !self.hidden_sets.is_empty() {
            self.hidden_sets.clear();
            self.modified = true;
        }
    }

    pub(crate) fn set_ranges_len(&self) -> usize {
        self.mesh.set_ranges().len()
    }

    pub(crate) fn set_name(&self, i: usize) -> String {
        self.mesh.set_ranges().get(i).map_or_else(|| "?".into(), |s| s.name.clone())
    }

    /// Move the active set's state out of the working fields.
    fn take_live(&mut self) -> SetState {
        SetState {
            doc: std::mem::take(&mut self.doc),
            selected: self.selected.take(),
            undo: std::mem::take(&mut self.undo),
            redo: std::mem::take(&mut self.redo),
            bakes: std::mem::take(&mut self.bakes),
            tex_size: self.tex_size,
            collapsed: std::mem::take(&mut self.collapsed),
            materialized: true,
            saved: None,
        }
    }

    /// Make `s` the working state.
    fn put_live(&mut self, s: SetState) {
        self.doc = s.doc;
        self.selected = s.selected;
        self.undo = s.undo;
        self.redo = s.redo;
        self.bakes = s.bakes;
        self.tex_size = s.tex_size;
        self.collapsed = s.collapsed;
        self.rebuild_procs();
    }

    /// One mask-generation state per tile of the working set, all starting afresh.
    fn rebuild_procs(&mut self) {
        for p in &self.procs {
            p.cancel();
        }
        self.procs = (0..self.bakes.len().max(1)).map(|_| TileProc::new()).collect();
    }

    /// Baked maps of the first tile of the active set, for the places that show "what is baked".
    pub(crate) fn baked0(&self) -> &HashMap<MeshMap, BakedMap> {
        &self.bakes[0].baked
    }

    /// The number of UV tiles of a set (always 1 unless UV tiles are in use).
    pub(crate) fn tile_count(&self, set: usize) -> usize {
        if self.udim { self.mesh.set_tiles(set).len() } else { 1 }
    }

    /// Forget the active set's undo history (its paint was discarded), leaving other sets' alone.
    pub(crate) fn clear_active_history(&mut self) {
        self.undo.clear();
        self.redo.clear();
        let a = self.active;
        self.hist_order.retain(|s| *s != a);
        self.redo_order.retain(|s| *s != a);
    }

    /// A new edit ends every redo chain, including those of sets that are not being edited.
    pub(crate) fn drop_other_redo(&mut self) {
        self.redo_order.clear();
        let active = self.active;
        let mut drops = Vec::new();
        for (i, slot) in self.parked.iter_mut().enumerate() {
            if let Some(st) = slot.as_mut().filter(|_| i != active) {
                if !st.redo.is_empty() {
                    st.redo.clear();
                    drops.push(SetBatch { tile: None, set: i, doc: st.doc.clone(), ops: vec![PaintOp::DropRedo] });
                }
            }
        }
        for b in drops {
            self.enqueue(b);
        }
    }

    /// Restart background mask generation, which belongs to one set's layers and baked maps.
    fn restart_procgen(&mut self) {
        self.thumbs.clear();
        self.thumbs_dirty();
        self.rebuild_procs();
    }

    pub(crate) fn enqueue_batch(&self, batch: SetBatch) {
        self.enqueue(batch);
    }

    fn enqueue(&self, batch: SetBatch) {
        self.ensure_doc_images(&batch.doc);
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.enqueue(batch);
        }
    }

    fn clear_queue(&self) {
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.clear_queue();
        }
    }

    /// Queue the pending paint operations for the active set, so a switch cannot send them to
    /// another set's textures.
    fn flush_pending(&mut self) {
        if !self.pending.is_empty() {
            let ops = std::mem::take(&mut self.pending);
            self.enqueue(SetBatch { tile: None, set: self.active, doc: self.doc.clone(), ops });
        }
    }

    /// Start every texture set of the current mesh from scratch, at the current resolution.
    pub(crate) fn init_sets(&mut self) {
        let size = if self.tex_size == 0 { DEFAULT_TEXTURE_SIZE } else { self.tex_size };
        let n = self.mesh.set_ranges().len();
        self.pending.clear();
        self.clear_queue();
        self.hist_order.clear();
        self.redo_order.clear();
        self.gesture_before = None;
        self.stroke.active = false;
        self.restart_procgen();
        self.hidden_sets.clear();
        self.parked = (0..n).map(|set| Some(SetState::fresh(size, self.tile_count(set)))).collect();
        self.active = 0;
        let first = self.parked[0].take().expect("just created");
        self.put_live(first);
        self.parked[0] = None;
        self.enqueue(SetBatch { tile: None, set: 0, doc: self.doc.clone(), ops: vec![PaintOp::Reset(size)] });
        for set in 1..n {
            let doc = Document::with_default_material();
            self.gpu_rebuild(set, &doc, PLACEHOLDER_SIZE);
        }
    }

    /// Replace every texture set with the saved ones (opening a project), restoring each set's
    /// paint on the GPU. `loads` holds, per set, its resolution and texture contents.
    pub(crate) fn install_sets(&mut self, states: Vec<SetState>, loads: Vec<(u32, Vec<Vec<op_project::TextureBlob>>)>, active: usize) {
        debug_assert_eq!(states.len(), loads.len());
        self.pending.clear();
        self.clear_queue();
        self.hist_order.clear();
        self.redo_order.clear();
        self.gesture_before = None;
        self.stroke.active = false;
        self.restart_procgen();
        let active = active.min(states.len().saturating_sub(1));
        let mut states = states;
        for (set, ((size, blobs), st)) in loads.into_iter().zip(states.iter_mut()).enumerate() {
            if set != active {
                // Not open: keep it small (and its paint in memory) until it is opened.
                st.materialized = false;
                st.saved = blobs.iter().any(|b| !b.is_empty()).then_some(Saved::Ram(blobs));
                self.gpu_rebuild(set, &st.doc, PLACEHOLDER_SIZE);
            } else {
                // Each tile gets its own textures and baked normal.
                for (t, blobs) in blobs.into_iter().enumerate() {
                    let mut ops = vec![PaintOp::Load { size, blobs }];
                    if has_uploadable_bakes(&st.bakes[t].baked) {
                        ops.push(baked_normal_op(&st.bakes[t].baked, size, self.bake_extent(set, t)));
                    }
                    self.enqueue(SetBatch { tile: Some(t), set, doc: st.doc.clone(), ops });
                }
            }
        }
        self.parked = states.into_iter().map(Some).collect();
        self.memory.used_at.clear();
        self.memory.queue = (0..self.parked.len()).rev().filter(|&i| i != active && self.parked[i].as_ref().is_some_and(|st| st.doc != Document::with_default_material())).collect();
        let live = self.parked[active].take().expect("the active set is present");
        self.put_live(live);
        self.active = active;
        self.trim_images();
    }

    /// Make the texture set under `pos` (a point in the viewport `rect`) the active one.
    /// The colour painted at the point under the cursor, as sRGB (what the paint colour swatch holds).
    /// The ID-map colour of the piece of the model under the cursor (first UV tile only), if baked.
    pub(crate) fn sample_id_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<[f32; 3]> {
        let (w, h) = (rect.width().max(1.0), rect.height().max(1.0));
        let ndc = glam::Vec2::new((pos.x - rect.min.x) / w * 2.0 - 1.0, 1.0 - (pos.y - rect.min.y) / h * 2.0);
        let inv = self.camera.view_proj(w / h).inverse();
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        let active = self.active;
        let (set, _, uv) = self.mesh.raycast_uv_with(near, (far - near).normalize_or_zero(), |s| s == active, self.ray_accept())?;
        if set != active || !(0.0..1.0).contains(&uv[0]) || !(0.0..1.0).contains(&uv[1]) {
            return None;
        }
        let ids = self.baked0().get(&MeshMap::Id)?;
        let size = ids.size as usize;
        let (x, y) = (((uv[0] * size as f32) as usize).min(size - 1), ((uv[1] * size as f32) as usize).min(size - 1));
        Some(ids.texel(y * size + x))
    }

    /// Everything painted at the point under the cursor: colour (sRGB) and the metallic, roughness
    /// and height values, with the viewport's fallbacks where the layer stack leaves a texel uncovered.
    pub(crate) fn sample_surface_at(&self, rect: egui::Rect, pos: egui::Pos2) -> Option<SurfaceSample> {
        let (w, h) = (rect.width().max(1.0), rect.height().max(1.0));
        let ndc = glam::Vec2::new((pos.x - rect.min.x) / w * 2.0 - 1.0, 1.0 - (pos.y - rect.min.y) / h * 2.0);
        let inv = self.camera.view_proj(w / h).inverse();
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        let (set, _, uv) = self.mesh.raycast_uv_with(near, (far - near).normalize_or_zero(), |s| !self.hidden_sets.contains(&s), self.ray_accept())?;
        let renderer = self.render_state.renderer.read();
        let gpu = renderer.callback_resources.get::<ViewportGpu>()?;
        let paint = gpu.sets.get(set)?;
        let size = paint.size();
        // Only the first UV tile is sampled; a model that spreads over tiles reports no colour elsewhere.
        if !(0.0..1.0).contains(&uv[0]) || !(0.0..1.0).contains(&uv[1]) {
            return None;
        }
        let (x, y) = (((uv[0] * size as f32) as u32).min(size - 1), ((uv[1] * size as f32) as u32).min(size - 1));
        // One texel per channel, not whole textures.
        let read = |tex: &eframe::wgpu::Texture, bpp: u32| crate::gpu_io::read_texel(&self.render_state.device, &self.render_state.queue, tex, x as u32, y as u32, bpp);
        let base = read(&paint.output.base, 8);
        let c = |k: usize| crate::gpu_io::f16_to_f32(u16::from_le_bytes([base[k * 2], base[k * 2 + 1]]));
        // Premultiplied over the fallback colour the viewport shows where nothing is painted.
        let alpha = c(3);
        let lin = [0, 1, 2].map(|k| c(k) + op_core::FALLBACK_BASE[k] * (1.0 - alpha));
        // Scalar channels hold (value * coverage, coverage) as bytes.
        let scalar = |tex: &eframe::wgpu::Texture, fallback: f32| {
            let t = read(tex, 2);
            let cover = t[1] as f32 / 255.0;
            if cover <= 0.0 { fallback } else { (t[0] as f32 / 255.0 / cover).clamp(0.0, 1.0) }
        };
        let glow = read(&paint.output.emissive, 8);
        let g = |k: usize| crate::gpu_io::f16_to_f32(u16::from_le_bytes([glow[k * 2], glow[k * 2 + 1]]));
        Some(SurfaceSample {
            color: lin.map(|v| linear_to_srgb(v.clamp(0.0, 1.0))),
            metallic: scalar(&paint.output.metal, op_core::FALLBACK_METALLIC),
            roughness: scalar(&paint.output.rough, op_core::FALLBACK_ROUGHNESS),
            height: scalar(&paint.output.height, 0.5),
            emissive: [g(0), g(1), g(2)].map(|v| linear_to_srgb(v.clamp(0.0, 1.0))),
            opacity: scalar(&paint.output.opacity, 1.0),
        })
    }

    pub(crate) fn pick_set_at(&mut self, rect: egui::Rect, pos: egui::Pos2) {
        if self.set_ranges_len() < 2 {
            return;
        }
        let (w, h) = (rect.width().max(1.0), rect.height().max(1.0));
        let ndc = glam::Vec2::new((pos.x - rect.min.x) / w * 2.0 - 1.0, 1.0 - (pos.y - rect.min.y) / h * 2.0);
        let inv = self.camera.view_proj(w / h).inverse();
        // wgpu clip space has depth 0 at the near plane and 1 at the far plane.
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        match self.mesh.raycast_set_where(near, (far - near).normalize_or_zero(), |s| !self.hidden_sets.contains(&s)) {
            Some((set, _)) => self.switch_set(set),
            None => self.status = "No model under the cursor.".into(),
        }
    }

    /// Make set `i` the one being edited.
    pub(crate) fn switch_set(&mut self, i: usize) {
        if i == self.active || i >= self.parked.len() {
            return;
        }
        if let Some(before) = self.gesture_before.take() {
            self.record(before);
        }
        self.stroke.active = false;
        self.flush_pending();
        self.restart_procgen();
        self.ensure_materialized(i);
        self.memory.touch(self.active);
        let quiet = self.undo.is_empty() && self.redo.is_empty();
        let mut old = self.take_live();
        if quiet {
            // Nothing was done here since it was loaded: give its memory back.
            let a = self.active;
            self.release_state(a, &mut old);
        }
        self.parked[self.active] = Some(old);
        let next = self.parked[i].take().expect("a parked set has its state");
        self.put_live(next);
        self.active = i;
        self.trim_images();
        let view = self.view;
        self.set_view(view);
        // Working on a set you cannot see is confusing, so choosing it brings it back.
        self.hidden_sets.remove(&i);
        self.status = format!("Texture set '{}'.", self.set_name(i));
    }

    /// Free the pictures no open set uses (parked sets at the placeholder size draw without them).
    fn trim_images(&self) {
        let open = self.parked.iter().flatten().filter(|s| s.materialized).flat_map(|s| s.doc.texture_ids());
        let keep: HashSet<String> = open.chain(self.doc.texture_ids()).collect();
        if let Some(gpu) = self.render_state.renderer.read().callback_resources.get::<ViewportGpu>() {
            gpu.keep_images(&keep);
        }
    }

    fn gpu_rebuild(&self, set: usize, doc: &Document, size: u32) {
        if size != PLACEHOLDER_SIZE {
            self.ensure_doc_images(doc);
        }
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.rebuild_set(&self.render_state.device, &self.render_state.queue, set, doc, size);
        }
    }

    /// Give a parked set its full-size paint store (it starts empty, as it was). The active set always has one.
    pub(crate) fn ensure_materialized(&mut self, set: usize) {
        let blobs = self.parked.get_mut(set).and_then(Option::as_mut).filter(|st| !st.materialized).and_then(|st| st.saved.take());
        let blobs = blobs.map(|s| self.read_saved(&s));
        self.materialize(set, blobs);
    }

    fn read_saved(&mut self, saved: &Saved) -> crate::spill::Tiles {
        saved.blobs().unwrap_or_else(|e| {
            self.status = format!("A texture set's paint could not be read back from disk: {e}");
            Vec::new()
        })
    }

    fn materialize(&mut self, set: usize, saved: Option<crate::spill::Tiles>) {
        let Some(Some(st)) = self.parked.get(set) else { return };
        if st.materialized {
            return;
        }
        let (doc, size) = (st.doc.clone(), st.tex_size);
        let loaded = saved.is_some();
        self.gpu_rebuild(set, &doc, size);
        for (t, blobs) in saved.into_iter().flatten().enumerate() {
            self.enqueue(SetBatch { tile: Some(t), set, doc: doc.clone(), ops: vec![PaintOp::Load { size, blobs }] });
        }
        let extents: Vec<BakeFrame> = (0..self.parked[set].as_ref().map_or(0, |st| st.bakes.len())).map(|t| self.bake_extent(set, t)).collect();
        let ops: Vec<(usize, PaintOp)> = self.parked[set]
            .as_ref()
            .map(|st| st.bakes.iter().enumerate().filter(|(_, b)| has_uploadable_bakes(&b.baked)).map(|(t, b)| (t, baked_normal_op(&b.baked, size, extents[t]))).collect())
            .unwrap_or_default();
        for (t, op) in ops {
            self.enqueue(SetBatch { tile: Some(t), set, doc: doc.clone(), ops: vec![op] });
        }
        if loaded {
            // Callers read the set back at once, so the upload must not wait for the next frame.
            self.flush_gpu();
        }
        if let Some(Some(st)) = self.parked.get_mut(set) {
            st.materialized = true;
        }
        self.memory.touch(set);
        self.enforce_budgets(set);
    }

    fn flush_gpu(&self) {
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.flush_queue(&self.render_state.device, &self.render_state.queue);
        }
    }

    /// Shrink a set's GPU copy to a picture of its look.
    fn freeze_gpu(&self, set: usize, doc: &Document) {
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.freeze_set(&self.render_state.device, &self.render_state.queue, set, doc, PREVIEW_SIZE);
        }
    }

    /// Give a set's GPU memory back, keeping its paint in memory and its look as a small picture
    /// (`st` is that set's state).
    fn release_state(&mut self, set: usize, st: &mut SetState) {
        self.flush_gpu();
        if st.doc == Document::with_default_material() {
            st.saved = None;
            self.gpu_rebuild(set, &st.doc, PLACEHOLDER_SIZE);
        } else {
            let tiles = {
                let renderer = self.render_state.renderer.read();
                let tiles = renderer.callback_resources.get::<ViewportGpu>().and_then(|g| g.sets.get(set)).map(|u| &u.tiles[..]).unwrap_or_default();
                tiles.iter().map(|p| p.capture_blobs(&self.render_state.device, &self.render_state.queue, &st.doc)).collect()
            };
            st.saved = Some(Saved::Ram(tiles));
            self.freeze_gpu(set, &st.doc);
        }
        st.materialized = false;
    }

    /// Return a set that was materialised only for reading (export) to its small picture.
    pub(crate) fn release_if_untouched(&mut self, set: usize) {
        let Some(mut st) = self.parked.get_mut(set).and_then(Option::take) else { return };
        if st.materialized && st.undo.is_empty() && st.redo.is_empty() {
            self.release_state(set, &mut st);
        }
        self.parked[set] = Some(st);
        self.trim_images();
    }

    /// Make the small picture of a parked set's look (after a project opens), keeping its paint where it is.
    fn cache_set(&mut self, set: usize) {
        let Some(Some(st)) = self.parked.get_mut(set) else { return };
        if st.materialized {
            return;
        }
        let saved = st.saved.take();
        let blobs = saved.as_ref().map(|s| self.read_saved(s));
        self.materialize(set, blobs);
        self.flush_gpu();
        if let Some(mut st) = self.parked.get_mut(set).and_then(Option::take) {
            self.freeze_gpu(set, &st.doc);
            (st.materialized, st.saved) = (false, saved);
            self.parked[set] = Some(st);
        }
        self.trim_images();
        self.enforce_budgets(usize::MAX);
    }

    /// One step of the background work after a project opens: a preview picture for the next set.
    pub(crate) fn cache_step(&mut self) -> bool {
        if self.stroke.active {
            return !self.memory.queue.is_empty();
        }
        if let Some(set) = self.memory.queue.pop() {
            self.cache_set(set);
        }
        !self.memory.queue.is_empty()
    }

    /// Stay inside the memory budgets. The sets that were used longest ago leave the GPU first
    /// (their paint goes to RAM, their look stays as a small picture); when RAM is short, the oldest
    /// paint goes on to disk. The set being edited, `keep` (just loaded for a reader) and sets with undo
    /// history stay where they are.
    pub(crate) fn enforce_budgets(&mut self, keep: usize) {
        let gpu_bytes = |app: &Self| app.render_state.renderer.read().callback_resources.get::<ViewportGpu>().map_or(0, |g| g.memory_bytes());
        if gpu_bytes(self) > self.memory.gpu_budget {
            let mut idle: Vec<usize> = (0..self.parked.len())
                .filter(|&i| i != keep && self.parked[i].as_ref().is_some_and(|st| st.materialized && st.undo.is_empty() && st.redo.is_empty()))
                .collect();
            idle.sort_by_key(|&i| self.memory.last_used(i));
            for i in idle {
                if gpu_bytes(self) <= self.memory.gpu_budget {
                    break;
                }
                self.release_if_untouched(i);
            }
        }
        let ram = |app: &Self| app.parked.iter().flatten().filter_map(|st| st.saved.as_ref()).map(Saved::ram_bytes).sum::<u64>();
        if ram(self) > self.memory.ram_budget {
            let mut held: Vec<usize> = (0..self.parked.len()).filter(|&i| self.parked[i].as_ref().and_then(|st| st.saved.as_ref()).is_some_and(|s| s.ram_bytes() > 0)).collect();
            held.sort_by_key(|&i| self.memory.last_used(i));
            let dir = crate::files::data_dir().join("spill");
            for i in held {
                if ram(self) <= self.memory.ram_budget {
                    break;
                }
                let name = format!("{}-{}-{}.bin", std::process::id(), i, self.memory.clock);
                if let Some(Some(st)) = self.parked.get_mut(i) {
                    if let Some(Err(e)) = st.saved.as_mut().map(|s| s.spill(&dir, &name)) {
                        self.status = format!("Could not move a texture set's paint to disk: {e}");
                        break;
                    }
                }
            }
        }
    }

    /// Apply `f` to a set's state wherever it currently lives.
    pub(crate) fn with_set<R>(&mut self, i: usize, f: impl FnOnce(&mut SetState) -> R) -> Option<R> {
        if i == self.active {
            let mut live = self.take_live();
            let r = f(&mut live);
            self.put_live(live);
            Some(r)
        } else {
            self.parked.get_mut(i)?.as_mut().map(f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolating_hides_everything_else_and_a_second_isolate_restores() {
        let none = HashSet::new();
        let only_2 = isolate(&none, 2, 5);
        assert_eq!(only_2, HashSet::from([0, 1, 3, 4]));
        assert!(isolate(&only_2, 2, 5).is_empty(), "isolating the one that is alone shows everything again");
        // Isolating a different one switches over.
        assert_eq!(isolate(&only_2, 3, 5), HashSet::from([0, 1, 2, 4]));
        // A hidden set being isolated becomes the only visible one.
        assert_eq!(isolate(&HashSet::from([2]), 2, 3), HashSet::from([0, 1]));
        assert!(isolate(&none, 0, 1).is_empty(), "with a single set there is nothing to hide");
    }

    #[test]
    fn a_fresh_set_has_the_default_material_selected() {
        let s = SetState::fresh(1024, 1);
        assert_eq!(s.doc.nodes.len(), 1);
        assert_eq!(s.selected, s.doc.nodes.last().map(|n| n.id));
        assert_eq!(s.tex_size, 1024);
        assert!(s.undo.is_empty() && s.redo.is_empty() && s.bakes[0].baked.is_empty());
        assert_eq!(SetState::fresh(64, 3).bakes.len(), 3, "one baked-map slot per UV tile");
        assert_eq!(SetState::fresh(64, 0).bakes.len(), 1, "at least one");
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// What the layer stack paints at one point of the surface.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceSample {
    /// sRGB.
    pub color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub height: f32,
    /// sRGB.
    pub emissive: [f32; 3],
    pub opacity: f32,
}

#[cfg(test)]
mod frame_tests {
    use super::bake_frame_of;
    use glam::Vec3;

    #[test]
    fn a_part_is_placed_inside_the_whole_in_units_of_its_longest_side() {
        let whole = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 4.0, 2.0));
        let (offset, extent) = bake_frame_of(whole, (Vec3::new(5.0, 0.0, 0.0), Vec3::new(10.0, 4.0, 2.0)));
        assert_eq!(offset, [0.5, 0.0, 0.0]);
        assert_eq!(extent, [0.5, 0.4, 0.2]);
        // The whole thing is its own frame: origin zero, longest side one.
        let (o, e) = bake_frame_of(whole, whole);
        assert_eq!((o, e), ([0.0; 3], [1.0, 0.4, 0.2]));
        // A model that does not start at the origin is measured from its own corner.
        let shifted = (Vec3::new(-3.0, 2.0, 1.0), Vec3::new(7.0, 6.0, 3.0));
        assert_eq!(bake_frame_of(shifted, (Vec3::new(2.0, 2.0, 1.0), Vec3::new(7.0, 6.0, 3.0))).0, [0.5, 0.0, 0.0]);
    }
}
