//! Polygon triangulation by ear clipping, for the n-gons (possibly concave) that DCC tools export.

use glam::{DVec2, DVec3};

/// Triangles (as indices into `pts`) covering the polygon `pts`, which must be planar-ish and
/// non-self-intersecting. Quads and triangles take fast paths; anything degenerate falls back to
/// a fan so every polygon yields `n - 2` triangles.
pub fn triangulate(pts: &[DVec3]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    // Newell's normal gives the polygon's plane and winding.
    let mut normal = DVec3::ZERO;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        normal += DVec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    let Some(normal) = normal.try_normalize() else { return fan(n) };
    let u = if normal.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let (ax, ay) = (normal.cross(u).normalize(), normal.cross(normal.cross(u)).normalize());
    let p2: Vec<DVec2> = pts.iter().map(|p| DVec2::new(p.dot(ax), p.dot(ay))).collect();
    // With this basis a polygon wound counter-clockwise around `normal` has positive area.
    let area: f64 = (0..n).map(|i| p2[i].perp_dot(p2[(i + 1) % n])).sum();
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let convex = |a: usize, b: usize, c: usize| sign * (p2[b] - p2[a]).perp_dot(p2[c] - p2[b]) > 1e-12;

    let mut ring: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n - 2);
    while ring.len() > 3 {
        let m = ring.len();
        let ear = (0..m).find(|&i| {
            let (a, b, c) = (ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]);
            convex(a, b, c)
                && !ring.iter().any(|&q| {
                    q != a && q != b && q != c && inside(p2[q], p2[a], p2[b], p2[c], sign)
                })
        });
        match ear {
            Some(i) => {
                out.push([ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]]);
                ring.remove(i);
            }
            None => {
                // No ear (degenerate input): finish with a fan so the polygon is still covered.
                for k in 1..ring.len() - 1 {
                    out.push([ring[0], ring[k], ring[k + 1]]);
                }
                return out;
            }
        }
    }
    out.push([ring[0], ring[1], ring[2]]);
    out
}

fn fan(n: usize) -> Vec<[usize; 3]> {
    (1..n - 1).map(|i| [0, i, i + 1]).collect()
}

fn inside(p: DVec2, a: DVec2, b: DVec2, c: DVec2, sign: f64) -> bool {
    let s = |x: DVec2, y: DVec2| sign * (y - x).perp_dot(p - x);
    s(a, b) >= -1e-12 && s(b, c) >= -1e-12 && s(c, a) >= -1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(pts: &[DVec3], tris: &[[usize; 3]]) -> f64 {
        tris.iter().map(|t| 0.5 * (pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]]).length()).sum()
    }

    fn v(x: f64, y: f64) -> DVec3 {
        DVec3::new(x, y, 0.0)
    }

    #[test]
    fn convex_polygons_cover_their_area() {
        let quad = [v(0.0, 0.0), v(2.0, 0.0), v(2.0, 1.0), v(0.0, 1.0)];
        let t = triangulate(&quad);
        assert_eq!(t.len(), 2);
        assert!((area(&quad, &t) - 2.0).abs() < 1e-9);
    }

    /// An L shape: a fan from vertex 0 would cover the notch; ear clipping must not.
    #[test]
    fn concave_polygons_do_not_cover_the_notch() {
        let l = [v(0.0, 0.0), v(2.0, 0.0), v(2.0, 1.0), v(1.0, 1.0), v(1.0, 2.0), v(0.0, 2.0)];
        let t = triangulate(&l);
        assert_eq!(t.len(), 4);
        assert!((area(&l, &t) - 3.0).abs() < 1e-9, "area {}", area(&l, &t));
        // Winding is preserved: every triangle is counter-clockwise like the polygon.
        for tri in &t {
            assert!((l[tri[1]] - l[tri[0]]).cross(l[tri[2]] - l[tri[0]]).z > 0.0);
        }
        // The same shape wound the other way still works and keeps its (clockwise) winding.
        let mut rev = l;
        rev.reverse();
        let t = triangulate(&rev);
        assert!((area(&rev, &t) - 3.0).abs() < 1e-9);
        for tri in &t {
            assert!((rev[tri[1]] - rev[tri[0]]).cross(rev[tri[2]] - rev[tri[0]]).z < 0.0);
        }
    }

    #[test]
    fn polygons_in_any_plane_and_degenerate_input_are_handled() {
        // A concave polygon standing in the XZ plane.
        let l = [v(0.0, 0.0), v(2.0, 0.0), v(2.0, 1.0), v(1.0, 1.0), v(1.0, 2.0), v(0.0, 2.0)].map(|p| DVec3::new(p.x, 5.0, p.y));
        assert!((area(&l, &triangulate(&l)) - 3.0).abs() < 1e-9);
        // Collinear points: no panic, n - 2 triangles.
        let line = [v(0.0, 0.0), v(1.0, 0.0), v(2.0, 0.0), v(3.0, 0.0)];
        assert_eq!(triangulate(&line).len(), 2);
        assert!(triangulate(&[v(0.0, 0.0), v(1.0, 0.0)]).is_empty());
    }
}
