//! Texture sets: a mesh with several materials is edited one set at a time. Each set has its own
//! layer stack, undo history, baked maps and texture resolution. The active set's state lives in
//! the `App` fields the rest of the program already uses; the others are parked here and swapped in
//! when the user picks them.

use crate::viewport::{SetBatch, ViewportGpu};
use crate::{App, Hist, PaintOp, DEFAULT_TEXTURE_SIZE};
use op_core::{BakedMap, Document, MeshMap, NodeId};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Size of the paint store of a set nobody has touched.
const PLACEHOLDER_SIZE: u32 = 64;

/// Everything that belongs to one texture set while it is not the active one.
pub struct SetState {
    pub doc: Document,
    pub selected: Option<NodeId>,
    pub undo: Vec<Hist>,
    pub redo: Vec<Hist>,
    pub baked: Arc<HashMap<MeshMap, BakedMap>>,
    pub bake_rev: u64,
    pub tex_size: u32,
    pub collapsed: HashSet<NodeId>,
    /// False while the set's paint store is held at the small placeholder size: nothing has been
    /// painted in it, so it looks the same and costs almost no GPU memory.
    pub materialized: bool,
}

impl SetState {
    /// A set restored from a saved project.
    pub fn restored(doc: Document, tex_size: u32, baked: HashMap<MeshMap, BakedMap>) -> Self {
        let selected = doc.nodes.last().map(|n| n.id);
        Self { doc, selected, undo: Vec::new(), redo: Vec::new(), baked: Arc::new(baked), bake_rev: 1, tex_size, collapsed: HashSet::new(), materialized: true }
    }

    /// A new set with the default material layer.
    pub fn fresh(tex_size: u32) -> Self {
        let doc = Document::with_default_material();
        let selected = doc.nodes.last().map(|n| n.id);
        Self {
            doc,
            selected,
            undo: Vec::new(),
            redo: Vec::new(),
            baked: Arc::new(HashMap::new()),
            bake_rev: 0,
            tex_size,
            collapsed: HashSet::new(),
            materialized: false,
        }
    }
}

/// The op that gives a set's paint store its baked normal map (or removes it).
pub fn baked_normal_op(baked: &HashMap<MeshMap, BakedMap>, size: u32) -> PaintOp {
    let rgba8 = baked.get(&MeshMap::NormalBase).filter(|b| b.size == size && b.channels == 3).map(|b| {
        let mut out = Vec::with_capacity(b.texels() * 4);
        for i in 0..b.texels() {
            let t = b.texel(i);
            out.extend(t.map(|v| (v * 255.0 + 0.5) as u8));
            out.push(255);
        }
        out
    });
    PaintOp::SetBakedNormal { size, rgba8 }
}

impl App {
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
            baked: std::mem::replace(&mut self.baked, Arc::new(HashMap::new())),
            bake_rev: self.bake_rev,
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
        self.baked = s.baked;
        self.bake_rev = s.bake_rev;
        self.tex_size = s.tex_size;
        self.collapsed = s.collapsed;
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
        for j in &self.proc_jobs {
            j.cancel.store(true, Ordering::Relaxed);
        }
        self.proc_jobs.clear();
        self.proc.invalidate();
        self.proc_info.clear();
    }

    pub(crate) fn enqueue_batch(&self, batch: SetBatch) {
        self.enqueue(batch);
    }

    fn enqueue(&self, batch: SetBatch) {
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
        self.parked = (0..n).map(|_| Some(SetState::fresh(size))).collect();
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
    pub(crate) fn install_sets(&mut self, states: Vec<SetState>, loads: Vec<(u32, Vec<op_project::TextureBlob>)>, active: usize) {
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
            if set != active && blobs.is_empty() {
                // Nothing painted: keep it small until it is opened.
                st.materialized = false;
                self.gpu_rebuild(set, &st.doc, PLACEHOLDER_SIZE);
            } else {
                let mut ops = vec![PaintOp::Load { size, blobs }];
                if st.baked.contains_key(&MeshMap::NormalBase) {
                    ops.push(baked_normal_op(&st.baked, size));
                }
                self.enqueue(SetBatch { tile: None, set, doc: st.doc.clone(), ops });
            }
        }
        self.parked = states.into_iter().map(Some).collect();
        let live = self.parked[active].take().expect("the active set is present");
        self.put_live(live);
        self.active = active;
    }

    /// Make the texture set under `pos` (a point in the viewport `rect`) the active one.
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
        match self.mesh.raycast_set(near, (far - near).normalize_or_zero()) {
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
        self.status = format!("Texture set '{}'.", self.set_name(i));
    }

    fn gpu_rebuild(&self, set: usize, doc: &Document, size: u32) {
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
        let op = self.parked[set].as_ref().map(|st| baked_normal_op(&st.baked, size));
        if let (Some(op), true) = (op, self.parked[set].as_ref().is_some_and(|st| st.baked.contains_key(&MeshMap::NormalBase))) {
            self.enqueue(SetBatch { tile: None, set, doc, ops: vec![op] });
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
    fn a_fresh_set_has_the_default_material_selected() {
        let s = SetState::fresh(1024);
        assert_eq!(s.doc.nodes.len(), 1);
        assert_eq!(s.selected, s.doc.nodes.last().map(|n| n.id));
        assert_eq!(s.tex_size, 1024);
        assert!(s.undo.is_empty() && s.redo.is_empty() && s.baked.is_empty());
    }
}
