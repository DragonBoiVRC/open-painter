//! Lasso and path fill: outline an area in the 3D or UV view and fill it as one undo step.
//! In the UV view the outline is filled exactly (as triangles); in the 3D view every mesh region
//! the outline covers is filled, found by casting rays on a grid inside it.

use crate::paint::PaintOp;
use crate::{App, Hist};
use eframe::egui::{self, Pos2, Rect};

/// Whether `p` is inside the polygon `poly` (even-odd rule).
pub(crate) fn inside(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut odd = false;
    let mut j = poly.len().wrapping_sub(1);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            odd = !odd;
        }
        j = i;
    }
    odd
}

fn area2(poly: &[[f32; 2]]) -> f32 {
    (0..poly.len()).map(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum()
}

fn cross(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Triangles (three points each) covering a simple polygon, by ear clipping. A polygon that crosses
/// itself still gets a result (the leftover is fanned), just not an exact one.
pub(crate) fn triangulate(poly: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut idx: Vec<usize> = (0..poly.len()).collect();
    if area2(poly) < 0.0 {
        idx.reverse();
    }
    let mut out = Vec::new();
    while idx.len() > 3 {
        let n = idx.len();
        let ear = (0..n).find(|&i| {
            let (a, b, c) = (poly[idx[(i + n - 1) % n]], poly[idx[i]], poly[idx[(i + 1) % n]]);
            cross(a, b, c) > 0.0
                && idx.iter().all(|&k| {
                    let p = poly[k];
                    p == a || p == b || p == c || !(cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0)
                })
        });
        let i = ear.unwrap_or(0);
        out.extend([poly[idx[(i + n - 1) % n]], poly[idx[i]], poly[idx[(i + 1) % n]]]);
        idx.remove(i);
    }
    if idx.len() == 3 {
        out.extend(idx.iter().map(|&k| poly[k]));
    }
    out
}

impl App {
    /// Fill the 3D-view outline `poly` (screen points in `rect`). `hide` writes the opposite on a mask.
    pub(crate) fn lasso_fill_3d(&mut self, poly: &[Pos2], rect: Rect, hide: bool) {
        if poly.len() < 3 {
            return;
        }
        let pts: Vec<[f32; 2]> = poly.iter().map(|p| [p.x, p.y]).collect();
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in &pts {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        // About 4000 rays at most: each one tests every triangle, so a dense mesh needs a coarse grid (the UV view is exact).
        let step = (((hi[0] - lo[0]) * (hi[1] - lo[1])) / 4_000.0).sqrt().clamp(3.0, 64.0);
        if !self.begin_fill_drag(poly[0], false) {
            return;
        }
        let (w, h) = (rect.width().max(1.0), rect.height().max(1.0));
        let inv = self.camera.view_proj(w / h).inverse();
        let mut y = lo[1];
        while y <= hi[1] {
            let mut x = lo[0];
            while x <= hi[0] {
                if inside([x, y], &pts) {
                    let ndc = glam::Vec2::new((x - rect.min.x) / w * 2.0 - 1.0, 1.0 - (y - rect.min.y) / h * 2.0);
                    let (near, far) = (inv.project_point3(ndc.extend(0.0)), inv.project_point3(ndc.extend(1.0)));
                    self.fill_drag_at(near, (far - near).normalize_or_zero(), hide);
                }
                x += step;
            }
            y += step;
        }
        if let Some(d) = self.fill_drag.take() {
            self.pending.push(PaintOp::EndStroke);
            self.status = format!("Filled {} region(s).", d.done.len());
        }
    }

    /// Fill the UV-space outline `poly` exactly, as one undo step.
    pub(crate) fn lasso_fill_uv(&mut self, poly: &[[f32; 2]], hide: bool) {
        let (layer, target) = match self.paint_target() {
            Ok(t) => t,
            Err(msg) => {
                self.status = msg.to_string();
                return;
            }
        };
        let uvs = triangulate(poly);
        if uvs.is_empty() {
            return;
        }
        let op = self.fill_op(layer, target, uvs, hide);
        self.stroke.active = false;
        self.pending.push(PaintOp::BeginStroke { layer, target });
        self.pending.push(op);
        self.pending.push(PaintOp::EndStroke);
        self.push_hist(Hist::Stroke);
        self.thumbs_dirty();
        self.status = "Filled the outline.".into();
    }

    /// Feed the pointer to the outline tools (Lasso: drag; Path: click corners, Enter or double-click
    /// to close, Esc to cancel). Returns the finished outline.
    pub(crate) fn outline_input(&mut self, ui: &egui::Ui, resp: &egui::Response, in_uv: bool) -> Option<Vec<Pos2>> {
        if self.outline.is_empty() {
            self.outline_uv = in_uv;
        } else if self.outline_uv != in_uv {
            return None; // the outline belongs to the other view
        }
        let far = |a: &Pos2, b: Pos2| (*a - b).length() > 3.0;
        match self.tool {
            crate::Tool::Lasso => {
                if let (true, Some(pos)) = (resp.dragged_by(egui::PointerButton::Primary), resp.interact_pointer_pos()) {
                    if self.outline.last().is_none_or(|l| far(l, pos)) {
                        self.outline.push(pos);
                    }
                    ui.ctx().request_repaint();
                }
                if resp.drag_stopped() {
                    return Some(std::mem::take(&mut self.outline));
                }
            }
            crate::Tool::Path => {
                if let (true, Some(pos)) = (resp.clicked(), resp.interact_pointer_pos()) {
                    if self.outline.last().is_none_or(|l| far(l, pos)) {
                        self.outline.push(pos);
                    }
                    if resp.double_clicked() {
                        return Some(std::mem::take(&mut self.outline));
                    }
                }
                if resp.hovered() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    return Some(std::mem::take(&mut self.outline));
                }
            }
            _ => {}
        }
        if resp.hovered() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.outline.clear();
        }
        None
    }

    /// Draw the outline being made, closing it back to the first point with a dashed hint.
    pub(crate) fn draw_outline(&self, ui: &egui::Ui, in_uv: bool) {
        if self.outline.len() < 2 || self.outline_uv != in_uv {
            return;
        }
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 210, 80));
        ui.painter().add(egui::Shape::line(self.outline.clone(), stroke));
        if let (Some(a), Some(b)) = (self.outline.last(), self.outline.first()) {
            ui.painter().add(egui::Shape::dashed_line(&[*a, *b], egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 210, 80, 140)), 4.0, 4.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_concave_polygon_triangulates_to_its_own_area() {
        // An L shape, 3 units of area.
        let l = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        let tris = triangulate(&l);
        assert_eq!(tris.len(), 12);
        let total: f32 = tris.chunks(3).map(|t| cross(t[0], t[1], t[2]).abs() / 2.0).sum();
        assert!((total - 3.0).abs() < 1e-5, "area {total}");
        let mut cw = l;
        cw.reverse();
        assert_eq!(triangulate(&cw).len(), 12, "winding does not matter");
    }

    #[test]
    fn points_are_inside_by_the_even_odd_rule() {
        let l = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]];
        assert!(inside([0.5, 1.5], &l));
        assert!(!inside([1.5, 1.5], &l), "the notch of the L");
        assert!(!inside([3.0, 0.5], &l));
    }
}
