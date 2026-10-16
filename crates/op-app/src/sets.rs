//! Texture sets: a mesh with several materials is edited one set at a time. Each set has its own
//! layer stack, undo history, baked maps and texture resolution. The active set's state lives in
//! the `App` fields the rest of the program already uses; the others are parked here and swapped in
//! when the user picks them.

use crate::viewport::{SetBatch, ViewportGpu};
use crate::paint::ProjectionData;
use crate::{App, Hist, PaintOp, DEFAULT_TEXTURE_SIZE};
use op_core::{BakedMap, Document, MeshMap, NodeId};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Size of the paint store of a set nobody has touched.
const PLACEHOLDER_SIZE: u32 = 64;

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
}

impl SetState {
    /// A set restored from a saved project.
    pub fn restored(doc: Document, tex_size: u32, baked: Vec<HashMap<MeshMap, BakedMap>>) -> Self {
        let selected = doc.nodes.last().map(|n| n.id);
        let bakes = baked.into_iter().map(|b| TileBake { baked: Arc::new(b), rev: 1 }).collect();
        Self { doc, selected, undo: Vec::new(), redo: Vec::new(), bakes, tex_size, collapsed: HashSet::new(), materialized: true }
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
        }
    }
}

/// The sets to hide so that only `i` is shown; or nothing hidden if `i` is already alone.
pub fn isolate(hidden: &HashSet<usize>, i: usize, count: usize) -> HashSet<usize> {
    let alone = !hidden.contains(&i) && (0..count).filter(|s| *s != i).all(|s| hidden.contains(&s));
    if alone { HashSet::new() } else { (0..count).filter(|s| *s != i).collect() }
}

/// The op that gives a set's paint store its baked normal map and the maps projected fills read
/// (or removes them). `extent` is the size of the baked mesh along x, y, z.
pub fn baked_normal_op(baked: &HashMap<MeshMap, BakedMap>, size: u32, extent: [f32; 3]) -> PaintOp {
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
        let largest = extent.iter().cloned().fold(1e-6f32, f32::max);
        ProjectionData { position, normal, extent: extent.map(|e| (e / largest).max(1e-3)) }
    });
    PaintOp::SetBakedNormal { size, rgba8, projection }
}

/// True if a set's bakes include anything the paint store wants uploaded.
pub fn has_uploadable_bakes(baked: &HashMap<MeshMap, BakedMap>) -> bool {
    baked.contains_key(&MeshMap::NormalBase) || (baked.contains_key(&MeshMap::Position) && baked.contains_key(&MeshMap::WorldSpaceNormal))
}

impl App {
    /// The size of the mesh a tile of `set` is baked from along x, y, z (what a Position bake is
    /// normalised by).
    pub(crate) fn bake_extent(&self, set: usize, tile: usize) -> [f32; 3] {
        let low = if self.udim {
            self.mesh.set_tiles(set).get(tile).and_then(|t| self.mesh.extract_tile(set, *t))
        } else {
            self.mesh.extract_set(set)
        };
        low.map_or([1.0; 3], |m| {
            let (lo, hi) = m.bounds();
            (hi - lo).to_array()
        })
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
            if set != active && blobs.iter().all(|b| b.is_empty()) {
                // Nothing painted: keep it small until it is opened.
                st.materialized = false;
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
        let live = self.parked[active].take().expect("the active set is present");
        self.put_live(live);
        self.active = active;
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
        let (set, _, uv) = self.mesh.raycast_uv_where(near, (far - near).normalize_or_zero(), |s| s == active)?;
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
        let (set, _, uv) = self.mesh.raycast_uv_where(near, (far - near).normalize_or_zero(), |s| !self.hidden_sets.contains(&s))?;
        let renderer = self.render_state.renderer.read();
        let gpu = renderer.callback_resources.get::<ViewportGpu>()?;
        let paint = gpu.sets.get(set)?;
        let size = paint.size();
        // Only the first UV tile is sampled; a model that spreads over tiles reports no colour elsewhere.
        if !(0.0..1.0).contains(&uv[0]) || !(0.0..1.0).contains(&uv[1]) {
            return None;
        }
        let (x, y) = (((uv[0] * size as f32) as u32).min(size - 1), ((uv[1] * size as f32) as u32).min(size - 1));
        let read = |tex: &eframe::wgpu::Texture, bpp: u32| {
            let texels = crate::gpu_io::read_texture(&self.render_state.device, &self.render_state.queue, tex, size, bpp);
            let i = ((y * size + x) * bpp) as usize;
            texels[i..i + bpp as usize].to_vec()
        };
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
        let untouched = self.undo.is_empty() && self.doc == Document::with_default_material();
        let mut old = self.take_live();
        if untouched {
            // Nothing was done here: give its memory back.
            old.materialized = false;
            let (a, d) = (self.active, old.doc.clone());
            self.gpu_rebuild(a, &d, PLACEHOLDER_SIZE);
        }
        self.parked[self.active] = Some(old);
        let next = self.parked[i].take().expect("a parked set has its state");
        self.put_live(next);
        self.active = i;
        let view = self.view;
        self.set_view(view);
        // Working on a set you cannot see is confusing, so choosing it brings it back.
        self.hidden_sets.remove(&i);
        self.status = format!("Texture set '{}'.", self.set_name(i));
    }

    fn gpu_rebuild(&self, set: usize, doc: &Document, size: u32) {
        self.ensure_doc_images(doc);
        if let Some(gpu) = self.render_state.renderer.write().callback_resources.get_mut::<ViewportGpu>() {
            gpu.rebuild_set(&self.render_state.device, &self.render_state.queue, set, doc, size);
        }
    }

    /// Give a parked set its full-size paint store (it starts empty, as it was). The active set always has one.
    pub(crate) fn ensure_materialized(&mut self, set: usize) {
        let Some(Some(st)) = self.parked.get(set) else { return };
        if st.materialized {
            return;
        }
        let (doc, size) = (st.doc.clone(), st.tex_size);
        self.gpu_rebuild(set, &doc, size);
        let extents: Vec<[f32; 3]> = (0..self.parked[set].as_ref().map_or(0, |st| st.bakes.len())).map(|t| self.bake_extent(set, t)).collect();
        let ops: Vec<(usize, PaintOp)> = self.parked[set]
            .as_ref()
            .map(|st| st.bakes.iter().enumerate().filter(|(_, b)| has_uploadable_bakes(&b.baked)).map(|(t, b)| (t, baked_normal_op(&b.baked, size, extents[t]))).collect())
            .unwrap_or_default();
        for (t, op) in ops {
            self.enqueue(SetBatch { tile: Some(t), set, doc: doc.clone(), ops: vec![op] });
        }
        if let Some(Some(st)) = self.parked.get_mut(set) {
            st.materialized = true;
        }
    }

    /// Return a set that was materialised only for reading (export) to the placeholder size.
    pub(crate) fn release_if_untouched(&mut self, set: usize) {
        let Some(Some(st)) = self.parked.get(set) else { return };
        if st.materialized && st.undo.is_empty() && st.doc == Document::with_default_material() {
            let doc = st.doc.clone();
            self.gpu_rebuild(set, &doc, PLACEHOLDER_SIZE);
            if let Some(Some(st)) = self.parked.get_mut(set) {
                st.materialized = false;
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
