//! Decals: a fill layer whose picture is projected onto the model through a placed frame (the
//! Decal projection). This file places the frame on the surface and edits its size.

use crate::App;
use eframe::egui;
use glam::Vec3;
use op_core::{Decal, NodeId, NodeKind, Projection};

/// A frame of the same size as `old`, centred on `center` and facing along `normal` (pointing away
/// from the surface), upright as far as the surface allows.
pub(crate) fn frame_at(old: &Decal, center: Vec3, normal: Vec3) -> Decal {
    let fwd = normal.normalize_or_zero();
    let up = if fwd.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let right = up.cross(fwd).normalize_or_zero();
    let top = fwd.cross(right);
    let (w, h) = (Vec3::from(old.u).length(), Vec3::from(old.v).length());
    Decal { center: center.to_array(), u: (right * w).to_array(), v: (top * h).to_array(), depth: old.depth }
}

/// Edit a decal's size, depth and roll. Returns true if it changed.
pub(crate) fn decal_controls(ui: &mut egui::Ui, d: &mut Decal) -> bool {
    let mut changed = false;
    let (mut w, mut h) = (Vec3::from(d.u).length(), Vec3::from(d.v).length());
    let mut roll = 0.0f32;
    egui::Grid::new("decal_frame").num_columns(2).spacing([6.0, 3.0]).show(ui, |ui| {
        ui.label("Width");
        let cw = ui.add(egui::DragValue::new(&mut w).speed(0.002).range(0.001..=10.0)).changed();
        ui.end_row();
        ui.label("Height");
        let ch = ui.add(egui::DragValue::new(&mut h).speed(0.002).range(0.001..=10.0)).changed();
        ui.end_row();
        ui.label("Depth");
        changed |= ui.add(egui::DragValue::new(&mut d.depth).speed(0.002).range(0.001..=10.0)).changed();
        ui.end_row();
        ui.label("Turn");
        let cr = ui.add(egui::DragValue::new(&mut roll).speed(0.5).suffix("\u{b0}").range(-360.0..=360.0)).changed();
        ui.end_row();
        for (c, which) in [(cw, 0), (ch, 1)] {
            if c {
                let v = if which == 0 { &mut d.u } else { &mut d.v };
                let len = Vec3::from(*v).length().max(1e-6);
                *v = (Vec3::from(*v) * ((if which == 0 { w } else { h }) / len)).to_array();
                changed = true;
            }
        }
        if cr && roll != 0.0 {
            let fwd = Vec3::from(d.u).cross(Vec3::from(d.v)).normalize_or_zero();
            let q = glam::Quat::from_axis_angle(fwd, roll.to_radians());
            d.u = (q * Vec3::from(d.u)).to_array();
            d.v = (q * Vec3::from(d.v)).to_array();
            changed = true;
        }
    });
    changed
}

impl App {
    /// Put the decal of fill layer `layer` where the ray from `origin` along `dir` meets the model.
    pub(crate) fn place_decal(&mut self, layer: NodeId, origin: Vec3, dir: Vec3) -> Result<(), String> {
        let active = self.active;
        let (_, first) = self.mesh.raycast_triangle_with(origin, dir, |s| s == active, self.ray_accept()).ok_or("No model there.")?;
        let p = |k: usize| Vec3::from(self.mesh.vertices[self.mesh.indices[first + k] as usize].position);
        let (a, b, c) = (p(0), p(1), p(2));
        let mut n = (b - a).cross(c - a).normalize_or_zero();
        if n.dot(dir) > 0.0 {
            n = -n; // face the viewer
        }
        let t = (a - origin).dot(n) / dir.dot(n);
        let hit = origin + dir * t;
        let (lo, hi) = self.mesh.bounds();
        let (lo, size) = (Vec3::from(lo), Vec3::from(hi) - Vec3::from(lo));
        let to_units = |v: Vec3| (v - lo) / size.max_element().max(1e-6);
        let normal_units = n; // directions are not affected by the uniform scale
        let node = self.doc.find(layer).ok_or("That layer is gone.")?;
        let NodeKind::Fill(f) = &node.kind else { return Err("Decals live on fill layers.".into()) };
        let frame = frame_at(&f.decal, to_units(hit), normal_units);
        self.edit(|d| {
            if let Some(NodeKind::Fill(f)) = d.find_mut(layer).map(|n| &mut n.kind) {
                f.projection = Projection::Decal;
                f.decal = frame;
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_faces_along_the_normal_with_the_same_size() {
        let old = Decal { u: [0.3, 0.0, 0.0], v: [0.0, 0.1, 0.0], ..Decal::default() };
        let f = frame_at(&old, Vec3::new(0.2, 0.3, 0.4), Vec3::new(0.0, 0.0, 2.0));
        let (u, v) = (Vec3::from(f.u), Vec3::from(f.v));
        assert!((u.length() - 0.3).abs() < 1e-5 && (v.length() - 0.1).abs() < 1e-5);
        assert!(u.cross(v).normalize().dot(Vec3::Z) > 0.999, "it looks along the normal");
        assert!(v.normalize().dot(Vec3::Y) > 0.999, "upright");
    }
}
