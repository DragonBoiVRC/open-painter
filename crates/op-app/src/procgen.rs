//! Decides when procedural masks must be (re)generated.
//!
//! Generated masks are a function of three things: the layer's generator, the baked maps and
//! the texture resolution. The scheduler remembers, per layer, which combination was last
//! generated ("done"), which is being generated ("running") and which is wanted now. It waits
//! for the wanted combination to stop changing (so dragging a slider does not start a job per
//! frame), never runs two jobs for one layer, and rejects results that are no longer wanted.
//!
//! It is a pure state machine: no threads and no clock of its own, so it is easy to test.

use op_core::{Document, Generator, Node, NodeId, NodeKind};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Everything a generated mask depends on.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub generator: Generator,
    pub bake_rev: u64,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct JobRequest {
    pub layer: NodeId,
    pub key: Key,
}

struct Entry {
    wanted: Key,
    wanted_since: Instant,
    running: Option<Key>,
    done: Option<Key>,
    /// False once the layer no longer has an enabled generated mask: late results are dropped.
    alive: bool,
}

pub struct Scheduler {
    entries: HashMap<NodeId, Entry>,
    debounce: Duration,
}

fn generated_masks(nodes: &[Node], out: &mut Vec<(NodeId, Generator)>) {
    for n in nodes {
        if let Some(m) = &n.mask {
            if let (true, Some(g)) = (m.enabled, &m.generator) {
                out.push((n.id, g.clone()));
            }
        }
        if let NodeKind::Group { children } = &n.kind {
            generated_masks(children, out);
        }
    }
}

impl Scheduler {
    pub fn new(debounce: Duration) -> Self {
        Self { entries: HashMap::new(), debounce }
    }

    /// Forget what has been generated: the mask textures were recreated (new project, resolution
    /// change, project load), so everything must be generated again.
    pub fn invalidate(&mut self) {
        for e in self.entries.values_mut() {
            e.done = None;
        }
    }

    /// Look at the document; returns the jobs to start now.
    pub fn plan(&mut self, doc: &Document, bake_rev: u64, size: u32, now: Instant) -> Vec<JobRequest> {
        let mut wanted = Vec::new();
        generated_masks(&doc.nodes, &mut wanted);
        // Layers that no longer want a generated mask are forgotten (running jobs finish harmlessly).
        let ids: Vec<NodeId> = wanted.iter().map(|(id, _)| *id).collect();
        for (id, e) in self.entries.iter_mut() {
            e.alive = ids.contains(id);
        }
        self.entries.retain(|_, e| e.alive || e.running.is_some());

        let mut start = Vec::new();
        for (id, generator) in wanted {
            let key = Key { generator, bake_rev, size };
            let e = self.entries.entry(id).or_insert_with(|| Entry {
                wanted: key.clone(),
                wanted_since: now,
                running: None,
                done: None,
                alive: true,
            });
            if e.wanted != key {
                e.wanted = key.clone();
                e.wanted_since = now;
            }
            let settled = now.saturating_duration_since(e.wanted_since) >= self.debounce;
            if e.done.as_ref() != Some(&key) && e.running.is_none() && settled {
                e.running = Some(key.clone());
                start.push(JobRequest { layer: id, key });
            }
        }
        start
    }

    /// A job finished. Returns true if its result is still what the layer wants.
    pub fn finish(&mut self, layer: NodeId, key: &Key) -> bool {
        let Some(e) = self.entries.get_mut(&layer) else { return false };
        if e.running.as_ref() == Some(key) {
            e.running = None;
        }
        if e.alive && &e.wanted == key {
            e.done = Some(key.clone());
            true
        } else {
            false
        }
    }

    /// A job failed or was cancelled: allow another attempt for the same inputs only if they change.
    pub fn failed(&mut self, layer: NodeId, key: &Key) {
        if let Some(e) = self.entries.get_mut(&layer) {
            if e.running.as_ref() == Some(key) {
                e.running = None;
            }
            // Mark as done so a deterministic failure is not retried every frame.
            if &e.wanted == key {
                e.done = Some(key.clone());
            }
        }
    }

    /// True while this layer's mask is waiting for, or being produced by, a job.
    pub fn is_busy(&self, layer: NodeId) -> bool {
        self.entries.get(&layer).is_some_and(|e| e.alive && (e.running.is_some() || e.done.as_ref() != Some(&e.wanted)))
    }

    /// How long until `plan` could start something, if anything is waiting out its debounce.
    pub fn next_wakeup(&self, now: Instant) -> Option<Duration> {
        self.entries
            .values()
            .filter(|e| e.alive && e.running.is_none() && e.done.as_ref() != Some(&e.wanted))
            .map(|e| self.debounce.saturating_sub(now.saturating_duration_since(e.wanted_since)))
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::{GeneratorKind, Mask, NodeKind};

    const DEBOUNCE: Duration = Duration::from_millis(100);

    fn doc_with_generated(gen: Generator) -> (Document, NodeId) {
        let mut doc = Document::default();
        let id = doc.add(None, None, "layer", NodeKind::Paint);
        doc.find_mut(id).unwrap().mask = Some(Mask::generated(gen));
        (doc, id)
    }

    fn edge() -> Generator {
        Generator::new(GeneratorKind::EdgeWear)
    }

    fn at(t0: Instant, ms: u64) -> Instant {
        t0 + Duration::from_millis(ms)
    }

    #[test]
    fn a_new_generated_mask_starts_after_the_debounce_and_only_once() {
        let t0 = Instant::now();
        let (doc, id) = doc_with_generated(edge());
        let mut s = Scheduler::new(DEBOUNCE);
        assert!(s.plan(&doc, 0, 1024, t0).is_empty(), "too early: the settings might still be changing");
        assert!(s.is_busy(id));
        assert!(s.next_wakeup(t0).is_some());
        let jobs = s.plan(&doc, 0, 1024, at(t0, 120));
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].layer, id);
        assert!(s.plan(&doc, 0, 1024, at(t0, 500)).is_empty(), "already running");
        assert!(s.finish(id, &jobs[0].key), "the result is wanted");
        assert!(!s.is_busy(id));
        assert!(s.plan(&doc, 0, 1024, at(t0, 900)).is_empty(), "nothing changed: nothing to do");
        assert_eq!(s.next_wakeup(at(t0, 900)), None);
    }

    #[test]
    fn dragging_a_slider_starts_one_job_after_it_stops() {
        let t0 = Instant::now();
        let mut s = Scheduler::new(DEBOUNCE);
        let mut started = 0;
        let (mut doc, id) = doc_with_generated(edge());
        for step in 0..30 {
            // 30 frames, 20 ms apart, the amount changes every frame
            doc.find_mut(id).unwrap().mask.as_mut().unwrap().generator = Some(edge().with("amount", step as f32 / 30.0));
            started += s.plan(&doc, 0, 512, at(t0, step * 20)).len();
        }
        assert_eq!(started, 0, "no job while the value keeps changing");
        let jobs = s.plan(&doc, 0, 512, at(t0, 30 * 20 + 150));
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].key.generator.params["amount"], 29.0 / 30.0, "it uses the final value");
    }

    #[test]
    fn a_change_while_running_waits_for_the_job_then_regenerates_with_the_latest_values() {
        let t0 = Instant::now();
        let mut s = Scheduler::new(DEBOUNCE);
        let (mut doc, id) = doc_with_generated(edge());
        let first = s.plan(&doc, 0, 512, at(t0, 200)); // wanted since 0, settled
        assert!(first.is_empty());
        let first = s.plan(&doc, 0, 512, at(t0, 300));
        assert_eq!(first.len(), 1);
        // The user edits while the first job runs.
        doc.find_mut(id).unwrap().mask.as_mut().unwrap().generator = Some(edge().with("amount", 0.9));
        assert!(s.plan(&doc, 0, 512, at(t0, 320)).is_empty());
        assert!(s.plan(&doc, 0, 512, at(t0, 600)).is_empty(), "never two jobs for one layer");
        assert!(!s.finish(id, &first[0].key), "the finished result is already stale");
        let second = s.plan(&doc, 0, 512, at(t0, 700));
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].key.generator.params["amount"], 0.9);
        assert!(s.finish(id, &second[0].key));
    }

    #[test]
    fn new_bakes_resolution_changes_and_invalidation_all_trigger_regeneration() {
        let t0 = Instant::now();
        let (doc, id) = doc_with_generated(edge());
        let mut s = Scheduler::new(DEBOUNCE);
        let mut now = 0u64;
        let mut generate = |s: &mut Scheduler, rev: u64, size: u32| -> usize {
            now += 200;
            let a = s.plan(&doc, rev, size, at(t0, now));
            now += 200;
            let jobs = s.plan(&doc, rev, size, at(t0, now));
            for j in a.iter().chain(&jobs) {
                s.finish(id, &j.key);
            }
            a.len() + jobs.len()
        };
        assert_eq!(generate(&mut s, 0, 512), 1);
        assert_eq!(generate(&mut s, 0, 512), 0, "unchanged");
        assert_eq!(generate(&mut s, 1, 512), 1, "baked maps changed");
        assert_eq!(generate(&mut s, 1, 1024), 1, "resolution changed");
        s.invalidate();
        assert_eq!(generate(&mut s, 1, 1024), 1, "mask textures were recreated");
        assert_eq!(generate(&mut s, 1, 1024), 0);
    }

    #[test]
    fn only_enabled_generated_masks_in_any_group_depth_are_planned() {
        let t0 = Instant::now();
        let mut doc = Document::default();
        let plain = doc.add(None, None, "plain", NodeKind::Paint);
        doc.find_mut(plain).unwrap().mask = Some(Mask::new(true));
        let g = doc.add(None, None, "group", NodeKind::Group { children: vec![] });
        let inner = doc.add(Some(g), None, "inner", NodeKind::Paint);
        doc.find_mut(inner).unwrap().mask = Some(Mask::generated(edge()));
        let off = doc.add(None, None, "off", NodeKind::Paint);
        doc.find_mut(off).unwrap().mask = Some(Mask { enabled: false, ..Mask::generated(edge()) });
        let none = doc.add(None, None, "none", NodeKind::Paint);
        let _ = none;
        let mut s = Scheduler::new(DEBOUNCE);
        s.plan(&doc, 0, 256, t0);
        let jobs = s.plan(&doc, 0, 256, at(t0, 500));
        assert_eq!(jobs.iter().map(|j| j.layer).collect::<Vec<_>>(), vec![inner]);
        assert!(!s.is_busy(plain) && !s.is_busy(off) && !s.is_busy(none));
    }

    #[test]
    fn removed_or_frozen_layers_are_forgotten_and_late_results_are_rejected() {
        let t0 = Instant::now();
        let (mut doc, id) = doc_with_generated(edge());
        let mut s = Scheduler::new(DEBOUNCE);
        s.plan(&doc, 0, 256, t0);
        let job = s.plan(&doc, 0, 256, at(t0, 200)).remove(0);
        // Freeze: the generator is removed while the job runs.
        doc.find_mut(id).unwrap().mask.as_mut().unwrap().generator = None;
        assert!(s.plan(&doc, 0, 256, at(t0, 300)).is_empty());
        assert!(!s.finish(id, &job.key), "a frozen mask must not be overwritten by a late result");
        assert!(!s.is_busy(id));
        // Deleting the layer entirely.
        let (doc2, _) = doc_with_generated(edge());
        let mut s = Scheduler::new(DEBOUNCE);
        s.plan(&doc2, 0, 256, t0);
        let job = s.plan(&doc2, 0, 256, at(t0, 200)).remove(0);
        s.plan(&Document::default(), 0, 256, at(t0, 300));
        assert!(!s.finish(job.layer, &job.key));
    }

    #[test]
    fn a_failed_job_is_not_retried_until_something_changes() {
        let t0 = Instant::now();
        let (mut doc, id) = doc_with_generated(edge());
        let mut s = Scheduler::new(DEBOUNCE);
        s.plan(&doc, 0, 256, t0);
        let job = s.plan(&doc, 0, 256, at(t0, 200)).remove(0);
        s.failed(id, &job.key);
        assert!(s.plan(&doc, 0, 256, at(t0, 2_000)).is_empty(), "no retry loop");
        assert!(!s.is_busy(id));
        doc.find_mut(id).unwrap().mask.as_mut().unwrap().generator = Some(edge().with("amount", 0.2));
        s.plan(&doc, 0, 256, at(t0, 2_100));
        assert_eq!(s.plan(&doc, 0, 256, at(t0, 2_300)).len(), 1, "a change allows another attempt");
    }
}
