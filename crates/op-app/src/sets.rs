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
}

impl SetState {
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
        }
    }
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

    /// Restart background mask generation, which belongs to one set's layers and baked maps.
    fn restart_procgen(&mut self) {
        for j in &self.proc_jobs {
            j.cancel.store(true, Ordering::Relaxed);
        }
        self.proc_jobs.clear();
        self.proc.invalidate();
        self.proc_info.clear();
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
            self.enqueue(SetBatch { set: self.active, doc: self.doc.clone(), ops });
        }
    }

    /// Start every texture set of the current mesh from scratch, at the current resolution.
    pub(crate) fn init_sets(&mut self) {
        let size = if self.tex_size == 0 { DEFAULT_TEXTURE_SIZE } else { self.tex_size };
        let n = self.mesh.set_ranges().len();
        self.pending.clear();
        self.clear_queue();
        self.gesture_before = None;
        self.stroke.active = false;
        self.restart_procgen();
        self.parked = (0..n).map(|_| Some(SetState::fresh(size))).collect();
        self.active = 0;
        let first = self.parked[0].take().expect("just created");
        self.put_live(first);
        self.parked[0] = None;
        for set in 0..n {
            let doc = if set == 0 { self.doc.clone() } else { Document::with_default_material() };
            self.enqueue(SetBatch { set, doc, ops: vec![PaintOp::Reset(size)] });
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
        let old = self.take_live();
        self.parked[self.active] = Some(old);
        let next = self.parked[i].take().expect("a parked set has its state");
        self.put_live(next);
        self.active = i;
        let view = self.view;
        self.set_view(view);
        self.status = format!("Texture set '{}'.", self.set_name(i));
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
