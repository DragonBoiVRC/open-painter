//! Per-mesh preparation for baking: welding, tangent frames, curvature, piece ids.

use glam::{Vec2, Vec3};
use op_mesh::Mesh;
use std::collections::HashMap;

pub struct Geometry {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub tris: Vec<[u32; 3]>,
    /// Unit tangent per vertex: direction of increasing u, perpendicular to the normal.
    pub tangents: Vec<Vec3>,
    /// +1 / -1: `bitangent = cross(normal, tangent) * handedness` points toward the top of the image
    /// (decreasing v), the convention of OpenGL-style normal maps.
    pub handedness: Vec<f32>,
    /// Signed mean curvature per vertex: positive convex, negative concave, about `1 / radius`.
    pub curvature: Vec<f32>,
    /// Connected piece each triangle belongs to.
    pub piece: Vec<u32>,
    pub min: Vec3,
    pub max: Vec3,
}

impl Geometry {
    pub fn new(mesh: &Mesh) -> Self {
        let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from(v.position)).collect();
        let normals: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from(v.normal)).collect();
        let uvs: Vec<Vec2> = mesh.vertices.iter().map(|v| Vec2::from(v.uv)).collect();
        let tris: Vec<[u32; 3]> = mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let (min, max) = mesh.bounds();
        let diag = (max - min).length().max(1e-9);
        let weld = weld_ids(&positions, diag * 1e-6);
        let (tangents, handedness) = tangent_frames(&positions, &normals, &uvs, &tris);
        let curvature = curvature(&positions, &tris, &weld);
        let piece = pieces(&tris, &weld);
        Self { positions, normals, uvs, tris, tangents, handedness, curvature, piece, min, max }
    }

    pub fn diagonal(&self) -> f32 {
        (self.max - self.min).length().max(1e-9)
    }

    pub fn face_normal(&self, tri: usize) -> Vec3 {
        let t = self.tris[tri];
        let (a, b, c) = (self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]);
        (b - a).cross(c - a).normalize_or_zero()
    }

    pub fn triangles(&self) -> Vec<[Vec3; 3]> {
        self.tris
            .iter()
            .map(|t| [self.positions[t[0] as usize], self.positions[t[1] as usize], self.positions[t[2] as usize]])
            .collect()
    }
}

/// Map each vertex to the id of the first vertex at (almost) the same position.
fn weld_ids(positions: &[Vec3], tol: f32) -> Vec<u32> {
    let mut seen: HashMap<[i64; 3], u32> = HashMap::with_capacity(positions.len());
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let key = [(p.x / tol).round() as i64, (p.y / tol).round() as i64, (p.z / tol).round() as i64];
            *seen.entry(key).or_insert(i as u32)
        })
        .collect()
}

fn tangent_frames(positions: &[Vec3], normals: &[Vec3], uvs: &[Vec2], tris: &[[u32; 3]]) -> (Vec<Vec3>, Vec<f32>) {
    let mut t_acc = vec![Vec3::ZERO; positions.len()];
    let mut b_acc = vec![Vec3::ZERO; positions.len()];
    for tri in tris {
        let [a, b, c] = tri.map(|i| i as usize);
        let (e1, e2) = (positions[b] - positions[a], positions[c] - positions[a]);
        let (d1, d2) = (uvs[b] - uvs[a], uvs[c] - uvs[a]);
        let det = d1.x * d2.y - d1.y * d2.x;
        if det.abs() < 1e-12 || !det.is_finite() {
            continue;
        }
        let dp_du = (e1 * d2.y - e2 * d1.y) / det;
        let dp_dv = (e2 * d1.x - e1 * d2.x) / det;
        let weight = e1.cross(e2).length(); // area weighted
        let (t, up) = (dp_du.normalize_or_zero() * weight, (-dp_dv).normalize_or_zero() * weight);
        for i in [a, b, c] {
            t_acc[i] += t;
            b_acc[i] += up;
        }
    }
    let mut tangents = Vec::with_capacity(positions.len());
    let mut hands = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        let n = normals[i].normalize_or_zero();
        let mut t = (t_acc[i] - n * n.dot(t_acc[i])).normalize_or_zero();
        if t == Vec3::ZERO {
            // No usable UV derivative: any vector perpendicular to the normal.
            let helper = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            t = (helper - n * n.dot(helper)).normalize_or_zero();
        }
        let hand = if n.cross(t).dot(b_acc[i]) < 0.0 { -1.0 } else { 1.0 };
        tangents.push(t);
        hands.push(hand);
    }
    (tangents, hands)
}

/// Signed curvature from the angle between adjacent faces, divided by the distance between
/// their centres (about `1 / radius` on a smooth surface). Hard edges register as sharp
/// peaks, which vertex normals alone could not show.
fn curvature(positions: &[Vec3], tris: &[[u32; 3]], weld: &[u32]) -> Vec<f32> {
    let face: Vec<(Vec3, Vec3)> = tris
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            ((b - a).cross(c - a).normalize_or_zero(), (a + b + c) / 3.0)
        })
        .collect();
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (weld[t[k] as usize], weld[t[(k + 1) % 3] as usize]);
            if a != b {
                edges.entry((a.min(b), a.max(b))).or_default().push(ti);
            }
        }
    }
    let mut sum: HashMap<u32, (f32, u32)> = HashMap::new();
    for ((a, b), faces) in &edges {
        let [fa, fb] = match faces.as_slice() {
            [x, y] => [*x, *y],
            _ => continue, // boundary or non-manifold edge: no well-defined curvature
        };
        let ((na, ca), (nb, cb)) = (face[fa], face[fb]);
        let dist = (cb - ca).length();
        if dist < 1e-12 || na == Vec3::ZERO || nb == Vec3::ZERO {
            continue;
        }
        let angle = na.dot(nb).clamp(-1.0, 1.0).acos();
        let sign = if (nb - na).dot(cb - ca) >= 0.0 { 1.0 } else { -1.0 };
        let k = sign * angle / dist;
        for v in [a, b] {
            let e = sum.entry(*v).or_insert((0.0, 0));
            e.0 += k;
            e.1 += 1;
        }
    }
    (0..positions.len())
        .map(|i| sum.get(&weld[i]).map_or(0.0, |(s, n)| s / *n as f32))
        .collect()
}

/// Connected piece per triangle (triangles sharing a welded vertex belong together).
fn pieces(tris: &[[u32; 3]], weld: &[u32]) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..weld.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for t in tris {
        let ids = t.map(|i| weld[i as usize]);
        let r0 = find(&mut parent, ids[0]);
        for id in &ids[1..] {
            let r = find(&mut parent, *id);
            if r != r0 {
                parent[r as usize] = r0;
            }
        }
    }
    tris.iter().map(|t| find(&mut parent, weld[t[0] as usize])).collect()
}

impl Geometry {
    /// Replace this part's curvature with the values of the whole model it belongs to, matched by
    /// position, so the seam between two materials has the curvature of the surface it really is
    /// (on its own, a part's border looks like a flat edge). Vertices the whole model lacks keep
    /// their own value.
    pub fn adopt_curvature(&mut self, whole: &Geometry) {
        let quant = |p: Vec3| -> (i64, i64, i64) {
            let q = (self.diagonal() * 1e-5).max(1e-9);
            ((p.x / q).round() as i64, (p.y / q).round() as i64, (p.z / q).round() as i64)
        };
        let mut by_pos: HashMap<(i64, i64, i64), f32> = HashMap::with_capacity(whole.positions.len());
        for (p, k) in whole.positions.iter().zip(&whole.curvature) {
            by_pos.entry(quant(*p)).or_insert(*k);
        }
        let found: Vec<Option<f32>> = self.positions.iter().map(|p| by_pos.get(&quant(*p)).copied()).collect();
        for (k, f) in self.curvature.iter_mut().zip(found) {
            if let Some(v) = f {
                *k = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_mesh::Vertex;

    pub fn quad(z: f32) -> Mesh {
        let v = |x: f32, y: f32| Vertex { position: [x, y, z], normal: [0.0, 0.0, 1.0], uv: [x, 1.0 - y] };
        Mesh { vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)], indices: vec![0, 1, 2, 0, 2, 3], ..Default::default() }
    }

    /// Two quads folded along a shared edge: alone each is flat, together the fold is convex.
    #[test]
    fn a_part_adopts_the_curvature_of_the_whole_model_at_its_seam() {
        let v = |p: [f32; 3], uv: [f32; 2]| op_mesh::Vertex { position: p, normal: [0.0, 1.0, 0.0], uv };
        let whole = Mesh {
            vertices: vec![
                v([0.0, 0.0, 0.0], [0.0, 0.0]),
                v([1.0, 0.0, 0.0], [1.0, 0.0]),
                v([1.0, 0.0, 1.0], [1.0, 1.0]),
                v([0.0, 0.0, 1.0], [0.0, 1.0]),
                v([1.0, -1.0, 0.0], [2.0, 0.0]),
                v([1.0, -1.0, 1.0], [2.0, 1.0]),
            ],
            // Top quad (y = 0) and a wall hanging down from its x = 1 edge.
            indices: vec![0, 3, 2, 0, 2, 1, 1, 2, 5, 1, 5, 4],
            ..Default::default()
        };
        let part = Mesh { vertices: whole.vertices.clone(), indices: whole.indices[..6].to_vec(), ..Default::default() };
        let mut alone = Geometry::new(&part);
        assert!(alone.curvature.iter().all(|k| k.abs() < 1e-6), "a single flat quad has no curvature");
        let all = Geometry::new(&whole);
        assert!(all.curvature[1] > 0.1 && all.curvature[2] > 0.1, "the fold is convex: {:?}", all.curvature);
        alone.adopt_curvature(&all);
        assert_eq!(alone.curvature[1], all.curvature[1]);
        assert_eq!(alone.curvature[2], all.curvature[2]);
        assert!(alone.curvature[0].abs() < 1e-6, "far from the seam it stays flat");
    }

    #[test]
    fn tangent_frame_points_along_increasing_u_and_up_the_image() {
        let g = Geometry::new(&quad(0.0));
        for i in 0..4 {
            assert!((g.tangents[i] - Vec3::X).length() < 1e-5, "tangent {:?}", g.tangents[i]);
            let n = g.normals[i];
            let bitangent = n.cross(g.tangents[i]) * g.handedness[i];
            // v = 1 - y, so the image "up" direction (decreasing v) is +Y.
            assert!((bitangent - Vec3::Y).length() < 1e-5, "bitangent {bitangent:?}");
        }
    }

    #[test]
    fn mirrored_uvs_flip_handedness_not_the_tangent_plane() {
        let mut m = quad(0.0);
        for v in &mut m.vertices {
            v.uv[1] = v.position[1]; // v = y: image up now points toward -Y
        }
        let g = Geometry::new(&m);
        let b = g.normals[0].cross(g.tangents[0]) * g.handedness[0];
        assert!((b - Vec3::NEG_Y).length() < 1e-5, "{b:?}");
    }

    #[test]
    fn curvature_is_zero_on_a_plane_positive_on_a_sphere_and_negative_inside_out() {
        let plane = Geometry::new(&quad(0.0));
        assert!(plane.curvature.iter().all(|k| k.abs() < 1e-6));

        let sphere = Mesh::uv_sphere(2.0, 48, 24);
        let g = Geometry::new(&sphere);
        // Away from the poles the estimate is close to 1 / radius.
        let mid: Vec<f32> = g.positions.iter().zip(&g.curvature).filter(|(p, _)| p.y.abs() < 1.2).map(|(_, k)| *k).collect();
        let mean = mid.iter().sum::<f32>() / mid.len() as f32;
        assert!(mean > 0.35 && mean < 0.65, "expected about 0.5, got {mean}");

        let mut inside_out = sphere.clone();
        for t in inside_out.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        let gi = Geometry::new(&inside_out);
        let mean_in = gi.curvature.iter().sum::<f32>() / gi.curvature.len() as f32;
        assert!(mean_in < -0.2, "inverted sphere must read concave, got {mean_in}");
    }

    #[test]
    fn cube_edges_register_as_convex_even_with_flat_shading() {
        // 12 triangles, flat normals, 24 vertices (split at every edge).
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let faces: [(Vec3, Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        ];
        for (n, u, v) in faces {
            let base = vertices.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = n + u * a + v * b;
                vertices.push(Vertex { position: p.into(), normal: n.into(), uv: [0.0, 0.0] });
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let g = Geometry::new(&Mesh { vertices, indices, ..Default::default() });
        assert!(g.curvature.iter().all(|k| *k > 0.0), "every cube corner/edge vertex is convex: {:?}", g.curvature);
        assert_eq!(g.piece.iter().collect::<std::collections::HashSet<_>>().len(), 1, "one connected piece");
    }

    #[test]
    fn disjoint_boxes_are_separate_pieces() {
        let a = quad(0.0);
        let mut b = quad(5.0);
        let mut all = a.clone();
        let off = all.vertices.len() as u32;
        all.vertices.append(&mut b.vertices);
        all.indices.extend(b.indices.iter().map(|i| i + off));
        let g = Geometry::new(&all);
        assert_eq!(g.piece[0], g.piece[1], "triangles of one quad share a piece");
        assert_ne!(g.piece[0], g.piece[2], "the second quad is another piece");
    }
}

#[cfg(test)]
pub(crate) use tests::quad as test_quad;
