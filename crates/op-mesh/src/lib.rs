//! Mesh data and loading for open-painter.

use glam::{Mat4, Vec2, Vec3};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("FBX import failed: {0}")]
    Fbx(String),
    #[error("glTF import failed: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("file contains no triangle geometry")]
    Empty,
    #[error("invalid mesh data: {0}")]
    Format(String),
}

mod fbx;
mod obj;
mod tiles;
pub use tiles::{Tile, TileRange, MAX_TILE_U, MAX_TILE_V};
mod triangulate;

const MESH_MAGIC_V1: &[u8; 8] = b"OPMESH01";
const MESH_MAGIC: &[u8; 8] = b"OPMESH02";
/// Names and counts in a stored mesh are limited so a corrupt file cannot ask for absurd allocations.
const MAX_SETS: usize = 4096;
const MAX_SET_NAME: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// How a fill decides which triangles belong together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Connect {
    /// Triangles that meet at a shared position *and* UV: one UV island.
    UvIsland,
    /// Triangles that meet at a shared position, across UV seams: one piece of the mesh.
    MeshPart,
    /// Triangles that share an edge and face (nearly) the same way: one flat face, however many
    /// triangles it is cut into. The tolerance is the largest turn between neighbours, in degrees.
    FlatFace(f32),
}

/// A run of triangles that share a material: one texture set. `first_index` and `index_count`
/// are in the mesh's index buffer and are multiples of three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetRange {
    pub name: String,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Material groups covering the index buffer in order. Empty means "one set, the whole mesh".
    pub sets: Vec<SetRange>,
}

impl Mesh {
    /// The texture sets of this mesh; a mesh without explicit sets is a single set.
    pub fn set_ranges(&self) -> Vec<SetRange> {
        if self.sets.is_empty() {
            vec![SetRange { name: "Default".into(), first_index: 0, index_count: self.indices.len() as u32 }]
        } else {
            self.sets.clone()
        }
    }

    /// The sets are in order, cover every index exactly once and start on triangle boundaries.
    pub fn sets_are_valid(&self) -> bool {
        if self.sets.is_empty() {
            return true;
        }
        let mut next = 0u32;
        for s in &self.sets {
            if s.first_index != next || s.index_count % 3 != 0 || s.name.len() > MAX_SET_NAME {
                return false;
            }
            let Some(end) = next.checked_add(s.index_count) else { return false };
            next = end;
        }
        next as usize == self.indices.len()
    }

    /// A mesh made of just texture set `set`: its triangles with the vertices they use, renumbered.
    /// Returns `None` if there is no such set.
    pub fn extract_set(&self, set: usize) -> Option<Mesh> {
        let ranges = self.set_ranges();
        let r = ranges.get(set)?;
        let tris = &self.indices[r.first_index as usize..(r.first_index + r.index_count) as usize];
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut vertices = Vec::new();
        let indices = tris
            .iter()
            .map(|&i| {
                let slot = &mut remap[i as usize];
                if *slot == u32::MAX {
                    *slot = vertices.len() as u32;
                    vertices.push(self.vertices[i as usize]);
                }
                *slot
            })
            .collect::<Vec<_>>();
        let count = indices.len() as u32;
        Some(Mesh { vertices, indices, sets: vec![SetRange { name: r.name.clone(), first_index: 0, index_count: count }] })
    }

    /// The nearest surface along a ray: the texture set it belongs to and the distance. Both sides
    /// of a triangle count. Brute force, which is fast enough for a click on a large model.
    pub fn raycast_set(&self, origin: Vec3, dir: Vec3) -> Option<(usize, f32)> {
        self.raycast_set_where(origin, dir, |_| true)
    }

    /// Like [`Mesh::raycast_set`], but only sets for which `visible` is true can be hit; the others are
    /// seen through.
    pub fn raycast_set_where(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool) -> Option<(usize, f32)> {
        self.raycast_uv_where(origin, dir, visible).map(|(set, dist, _)| (set, dist))
    }

    /// The nearest visible surface along a ray: its texture set, the distance and the UV coordinate hit.
    pub fn raycast_uv_where(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool) -> Option<(usize, f32, [f32; 2])> {
        self.raycast_uv_with(origin, dir, visible, |_, _| true)
    }

    /// Like [`Mesh::raycast_uv_where`], and a hit also has to be accepted: `accept` gets the index of the
    /// triangle's first corner and the point hit, and can refuse hidden or cut-away surfaces.
    pub fn raycast_uv_with(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool, accept: impl Fn(usize, Vec3) -> bool) -> Option<(usize, f32, [f32; 2])> {
        self.raycast_impl(origin, dir, visible, accept).map(|(set, _, dist, uv)| (set, dist, uv))
    }

    /// The nearest visible triangle along a ray: its texture set and the index of its first corner in
    /// `indices` (a multiple of three).
    pub fn raycast_triangle_where(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool) -> Option<(usize, usize)> {
        self.raycast_triangle_with(origin, dir, visible, |_, _| true)
    }

    /// Like [`Mesh::raycast_triangle_where`] with an `accept` test (see [`Mesh::raycast_uv_with`]).
    pub fn raycast_triangle_with(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool, accept: impl Fn(usize, Vec3) -> bool) -> Option<(usize, usize)> {
        self.raycast_impl(origin, dir, visible, accept).map(|(set, first, _, _)| (set, first))
    }

    fn raycast_impl(&self, origin: Vec3, dir: Vec3, visible: impl Fn(usize) -> bool, accept: impl Fn(usize, Vec3) -> bool) -> Option<(usize, usize, f32, [f32; 2])> {
        let pos = |i: u32| Vec3::from(self.vertices[i as usize].position);
        let mut best: Option<(usize, usize, f32, [f32; 2])> = None;
        for (set, r) in self.set_ranges().iter().enumerate() {
            if !visible(set) {
                continue;
            }
            for first in (r.first_index..r.first_index + r.index_count).step_by(3) {
                let t = &self.indices[first as usize..first as usize + 3];
                let (a, b, c) = (pos(t[0]), pos(t[1]), pos(t[2]));
                // Moller-Trumbore.
                let (e1, e2) = (b - a, c - a);
                let p = dir.cross(e2);
                let det = e1.dot(p);
                if det.abs() < 1e-12 {
                    continue;
                }
                let inv = 1.0 / det;
                let s = origin - a;
                let u = s.dot(p) * inv;
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let q = s.cross(e1);
                let v = dir.dot(q) * inv;
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let dist = e2.dot(q) * inv;
                if dist > 0.0 && best.is_none_or(|(_, _, d, _)| dist < d) && accept(first as usize, origin + dir * dist) {
                    let uv = |i: u32| Vec2::from(self.vertices[i as usize].uv);
                    let hit = uv(t[0]) * (1.0 - u - v) + uv(t[1]) * u + uv(t[2]) * v;
                    best = Some((set, first as usize, dist, hit.to_array()));
                }
            }
        }
        best
    }

    /// The triangles connected to the one starting at index `first`, as the first-corner indices of
    /// each (including the one given), staying inside its texture set. Triangles connect when they
    /// share a corner position — and, for `Connect::UvIsland`, the corner's UV too, so a hard UV seam
    /// stops the fill where a mesh-only seam would not.
    pub fn connected_triangles(&self, first: usize, how: Connect) -> Vec<usize> {
        let Some(range) = self.set_ranges().into_iter().find(|r| (r.first_index as usize..(r.first_index + r.index_count) as usize).contains(&first)) else {
            return Vec::new();
        };
        let tris: Vec<usize> = (range.first_index as usize..(range.first_index + range.index_count) as usize).step_by(3).collect();
        if let Connect::FlatFace(tolerance) = how {
            return self.flat_face(first, &tris, tolerance);
        }
        let key = |i: u32| {
            let v = &self.vertices[i as usize];
            let p = v.position.map(f32::to_bits);
            let uv = if how == Connect::UvIsland { v.uv.map(f32::to_bits) } else { [0, 0] };
            (p, uv)
        };
        // Which triangles touch each corner key.
        let mut by_key: std::collections::HashMap<([u32; 3], [u32; 2]), Vec<usize>> = std::collections::HashMap::new();
        for &t in &tris {
            for c in 0..3 {
                by_key.entry(key(self.indices[t + c])).or_default().push(t);
            }
        }
        let mut seen = std::collections::HashSet::new();
        let mut order = Vec::new();
        let mut stack = vec![first];
        seen.insert(first);
        while let Some(t) = stack.pop() {
            order.push(t);
            for c in 0..3 {
                for &n in by_key.get(&key(self.indices[t + c])).into_iter().flatten() {
                    if seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
        order.sort_unstable();
        order
    }

    /// Triangles reachable from `first` across shared edges without turning more than `tolerance`
    /// degrees from the triangle next to them.
    fn flat_face(&self, first: usize, tris: &[usize], tolerance: f32) -> Vec<usize> {
        let pos = |i: u32| Vec3::from(self.vertices[i as usize].position);
        let normal = |t: usize| {
            let (a, b, c) = (pos(self.indices[t]), pos(self.indices[t + 1]), pos(self.indices[t + 2]));
            (b - a).cross(c - a).normalize_or_zero()
        };
        let edge = |a: u32, b: u32| {
            let (pa, pb) = (self.vertices[a as usize].position.map(f32::to_bits), self.vertices[b as usize].position.map(f32::to_bits));
            if pa <= pb { (pa, pb) } else { (pb, pa) }
        };
        let mut by_edge: std::collections::HashMap<([u32; 3], [u32; 3]), Vec<usize>> = std::collections::HashMap::new();
        for &t in tris {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                by_edge.entry(edge(self.indices[t + a], self.indices[t + b])).or_default().push(t);
            }
        }
        let min_dot = tolerance.clamp(0.0, 180.0).to_radians().cos();
        let mut seen = std::collections::HashSet::new();
        let mut order = Vec::new();
        let mut stack = vec![first];
        seen.insert(first);
        while let Some(t) = stack.pop() {
            order.push(t);
            let n = normal(t);
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                for &other in by_edge.get(&edge(self.indices[t + a], self.indices[t + b])).into_iter().flatten() {
                    if !seen.contains(&other) && normal(other).dot(n) >= min_dot {
                        seen.insert(other);
                        stack.push(other);
                    }
                }
            }
        }
        order.sort_unstable();
        order
    }

    /// Axis-aligned bounds as (min, max). Returns zeros for an empty mesh.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut it = self.vertices.iter().map(|v| Vec3::from(v.position));
        let Some(first) = it.next() else {
            return (Vec3::ZERO, Vec3::ZERO);
        };
        it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)))
    }

    /// Recompute smooth, area-weighted vertex normals from the triangles. Vertices at the same
    /// position (split apart by UV seams, say) share their faces, so a seam does not show as a
    /// lighting edge; only faces within 60 degrees of each other are blended, so creases stay sharp.
    pub fn recompute_normals(&mut self) {
        const BLEND_COS: f32 = 0.5; // cos(60 degrees)
        let pos = |i: usize| Vec3::from(self.vertices[i].position);
        let tris: Vec<Vec3> = self
            .indices
            .chunks_exact(3)
            .map(|t| {
                let (pa, pb, pc) = (pos(t[0] as usize), pos(t[1] as usize), pos(t[2] as usize));
                (pb - pa).cross(pc - pa) // length = 2 * area
            })
            .collect();
        // What each vertex's own faces say, and the faces at each position grouped into smoothing
        // clusters: a face joins the first cluster whose first face is within 60 degrees of it. The
        // clusters belong to the position, so every seam copy of a vertex gets the same normal.
        let mut own = vec![Vec3::ZERO; self.vertices.len()];
        let mut clusters: std::collections::HashMap<[u32; 3], Vec<(Vec3, Vec3)>> = std::collections::HashMap::new();
        let mut seen: std::collections::HashSet<([u32; 3], u32)> = std::collections::HashSet::new();
        for (f, tri) in self.indices.chunks_exact(3).enumerate() {
            let Some(dir) = tris[f].try_normalize() else { continue };
            for &i in tri {
                own[i as usize] += tris[f];
                let key = self.vertices[i as usize].position.map(f32::to_bits);
                if !seen.insert((key, f as u32)) {
                    continue; // a degenerate face can list one position twice
                }
                let list = clusters.entry(key).or_default();
                match list.iter_mut().find(|(rep, _)| rep.dot(dir) > BLEND_COS) {
                    Some((_, sum)) => *sum += tris[f],
                    None => list.push((dir, tris[f])),
                }
            }
        }
        for (i, v) in self.vertices.iter_mut().enumerate() {
            let Some(reference) = own[i].try_normalize() else {
                v.normal = Vec3::Y.into();
                continue;
            };
            let best = clusters
                .get(&v.position.map(f32::to_bits))
                .and_then(|list| list.iter().max_by(|a, b| a.0.dot(reference).total_cmp(&b.0.dot(reference))));
            v.normal = best.and_then(|(_, sum)| sum.try_normalize()).unwrap_or(reference).into();
        }
    }

    /// UV sphere with `segments` around and `rings` top to bottom.
    pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Self {
        let (segments, rings) = (segments.max(3), rings.max(2));
        let mut vertices = Vec::with_capacity(((segments + 1) * (rings + 1)) as usize);
        for r in 0..=rings {
            let v = r as f32 / rings as f32;
            let phi = v * std::f32::consts::PI;
            for s in 0..=segments {
                let u = s as f32 / segments as f32;
                let theta = u * std::f32::consts::TAU;
                let n = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
                vertices.push(Vertex {
                    position: (n * radius).into(),
                    normal: n.into(),
                    uv: Vec2::new(u, v).into(),
                });
            }
        }
        let mut indices = Vec::new();
        let stride = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let i = r * stride + s;
                indices.extend_from_slice(&[i, i + 1, i + stride, i + 1, i + stride + 1, i + stride]);
            }
        }
        Self { vertices, indices, sets: Vec::new() }
    }

    /// Compact binary form used inside project files: magic, counts, vertices, indices.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.vertices.len() * 32 + self.indices.len() * 4);
        out.extend_from_slice(MESH_MAGIC);
        out.extend_from_slice(&(self.vertices.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        out.extend_from_slice(bytemuck::cast_slice(&self.indices));
        out.extend_from_slice(&(self.sets.len() as u32).to_le_bytes());
        for s in &self.sets {
            out.extend_from_slice(&(s.name.len() as u32).to_le_bytes());
            out.extend_from_slice(s.name.as_bytes());
            out.extend_from_slice(&s.first_index.to_le_bytes());
            out.extend_from_slice(&s.index_count.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Mesh::to_bytes`] (and of the older version without material sets). Rejects
    /// truncated data, trailing bytes, indices outside the vertex list and inconsistent sets.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MeshError> {
        let bad = |why: &str| MeshError::Format(why.to_string());
        if bytes.len() < 16 {
            return Err(bad("not a mesh blob"));
        }
        let v2 = match &bytes[..8] {
            m if m == MESH_MAGIC => true,
            m if m == MESH_MAGIC_V1 => false,
            _ => return Err(bad("not a mesh blob")),
        };
        let nv = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let ni = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let vbytes = nv.checked_mul(std::mem::size_of::<Vertex>()).ok_or_else(|| bad("vertex count overflow"))?;
        let ibytes = ni.checked_mul(4).ok_or_else(|| bad("index count overflow"))?;
        let body_end = 16usize.checked_add(vbytes).and_then(|x| x.checked_add(ibytes)).ok_or_else(|| bad("size overflow"))?;
        if bytes.len() < body_end {
            return Err(bad("length does not match header"));
        }
        let vertices: Vec<Vertex> = bytemuck::pod_collect_to_vec(&bytes[16..16 + vbytes]);
        let indices: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes[16 + vbytes..body_end]);
        if ni % 3 != 0 || indices.iter().any(|&i| i as usize >= nv) {
            return Err(bad("invalid indices"));
        }
        let mut sets = Vec::new();
        let mut rest = &bytes[body_end..];
        if v2 {
            let mut take = |n: usize| -> Result<&[u8], MeshError> {
                if rest.len() < n {
                    return Err(bad("truncated sets"));
                }
                let (a, b) = rest.split_at(n);
                rest = b;
                Ok(a)
            };
            let count = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            if count > MAX_SETS {
                return Err(bad("too many sets"));
            }
            for _ in 0..count {
                let len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
                if len > MAX_SET_NAME {
                    return Err(bad("set name too long"));
                }
                let name = String::from_utf8(take(len)?.to_vec()).map_err(|_| bad("set name is not text"))?;
                let first_index = u32::from_le_bytes(take(4)?.try_into().unwrap());
                let index_count = u32::from_le_bytes(take(4)?.try_into().unwrap());
                sets.push(SetRange { name, first_index, index_count });
            }
        }
        if !(if v2 { rest.is_empty() } else { bytes.len() == body_end }) {
            return Err(bad("length does not match header"));
        }
        let mesh = Self { vertices, indices, sets };
        if !mesh.sets_are_valid() {
            return Err(bad("inconsistent material sets"));
        }
        Ok(mesh)
    }

    /// Load a Wavefront OBJ; each `usemtl` material becomes a texture set. Its V axis points up; the
    /// mesh is converted to the image convention (V down) the rest of the app uses.
    pub fn load_obj(path: impl AsRef<Path>) -> Result<Self, MeshError> {
        let mut mesh = obj::load(path.as_ref())?;
        mesh.flip_v_per_tile();
        Ok(mesh)
    }

    /// Load every mesh in an FBX file (binary, version 7.x), with its node transforms applied and
    /// converted to Y-up. Triangles are grouped by material into [`SetRange`]s; materials with the
    /// same name on different objects share one set.
    pub fn load_fbx(path: impl AsRef<Path>) -> Result<Self, MeshError> {
        let mut mesh = fbx::load(path.as_ref())?;
        mesh.flip_v_per_tile();
        Ok(mesh)
    }

    /// Load every triangle primitive in the default scene of a glTF/GLB file,
    /// baked into world space and merged into one mesh.
    pub fn load_gltf(path: impl AsRef<Path>) -> Result<Self, MeshError> {
        let (doc, buffers, _images) = gltf::import(path)?;
        let scene = doc.default_scene().or_else(|| doc.scenes().next());
        let mut mesh = Mesh::default();
        if let Some(scene) = scene {
            for node in scene.nodes() {
                append_node(&node, Mat4::IDENTITY, &buffers, &mut mesh);
            }
        }
        if mesh.indices.is_empty() {
            return Err(MeshError::Empty);
        }
        Ok(mesh)
    }
}

fn append_node(node: &gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Mesh) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    let normal_mat = world.inverse().transpose();
    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(positions) = reader.read_positions() else { continue };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            let base = out.vertices.len() as u32;
            for (i, p) in positions.iter().enumerate() {
                out.vertices.push(Vertex {
                    position: world.transform_point3(Vec3::from(*p)).into(),
                    normal: normals
                        .as_ref()
                        .map(|n| normal_mat.transform_vector3(Vec3::from(n[i])).normalize_or_zero().into())
                        .unwrap_or([0.0, 1.0, 0.0]),
                    uv: uvs.as_ref().map(|t| t[i]).unwrap_or([0.0; 2]),
                });
            }
            let idx: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let tri_start = out.indices.len();
            out.indices.extend(idx.iter().map(|i| i + base));
            if normals.is_none() {
                // Smooth-normal fallback for just this primitive.
                let first = base as usize;
                let mut part = Mesh {
                    vertices: out.vertices[first..].to_vec(),
                    indices: out.indices[tri_start..].iter().map(|i| i - base).collect(),
                    ..Default::default()
                };
                part.recompute_normals();
                out.vertices[first..].copy_from_slice(&part.vertices);
            }
        }
    }
    for child in node.children() {
        append_node(&child, world, buffers, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raycast_reports_the_uv_under_the_ray() {
        let m = Mesh::uv_sphere(1.0, 96, 48);
        let target = Vec3::new(0.3, 0.5, 0.8).normalize();
        let (set, dist, uv) = m.raycast_uv_where(target * 3.0, -target, |_| true).expect("the ray hits the sphere");
        assert_eq!(set, 0);
        assert!((dist - 2.0).abs() < 0.01, "front surface is 2 away: {dist}");
        let v = target.y.acos() / std::f32::consts::PI;
        let u = target.z.atan2(target.x).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
        assert!((uv[0] - u).abs() < 0.01 && (uv[1] - v).abs() < 0.01, "uv {uv:?} vs {:?}", [u, v]);
        assert!(m.raycast_uv_where(Vec3::new(5.0, 5.0, 5.0), Vec3::X, |_| true).is_none(), "a miss reports nothing");
    }

    #[test]
    fn sphere_is_unit_and_indexed_in_range() {
        let m = Mesh::uv_sphere(1.0, 32, 16);
        assert_eq!(m.indices.len() % 3, 0);
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        let (lo, hi) = m.bounds();
        assert!((hi - Vec3::ONE).abs().max_element() < 1e-3);
        assert!((lo + Vec3::ONE).abs().max_element() < 1e-3);
    }

    #[test]
    fn recomputed_normals_agree_across_a_split_seam_but_keep_creases() {
        let v = |x: f32, y: f32, z: f32, u: f32| Vertex { position: [x, y, z], normal: [0.0; 3], uv: [u, 0.0] };
        // Two triangles of a gently bent strip sharing the edge (0,0,0)-(0,1,0), with the edge
        // vertices duplicated as a UV seam would, then a second pair folded 90 degrees.
        let mut m = Mesh {
            vertices: vec![
                v(-1.0, 0.0, 0.0, 0.0), v(0.0, 0.0, 0.0, 1.0), v(0.0, 1.0, 0.0, 1.0),
                v(0.0, 0.0, 0.0, 0.0), v(1.0, 0.0, 0.3, 1.0), v(0.0, 1.0, 0.0, 0.0),
                v(10.0, 0.0, 0.0, 0.0), v(11.0, 0.0, 0.0, 1.0), v(10.0, 1.0, 0.0, 0.0),
                v(10.0, 0.0, 0.0, 0.0), v(10.0, 0.0, 1.0, 1.0), v(10.0, 1.0, 0.0, 0.0),
            ],
            indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ..Default::default()
        };
        m.recompute_normals();
        let n = |i: usize| Vec3::from(m.vertices[i].normal);
        assert!((n(1) - n(3)).length() < 1e-6, "the seam copies must share one normal");
        assert!(n(6).dot(n(9)).abs() < 0.01, "a 90 degree crease stays sharp: {:?} {:?}", n(6), n(9));
    }

    #[test]
    fn recomputed_normals_point_outward() {
        let mut m = Mesh::uv_sphere(1.0, 64, 32);
        m.recompute_normals();
        // Skip pole vertices, which are degenerate fans. With outward winding the
        // recomputed normal must agree with the radial direction.
        for v in m.vertices.iter().filter(|v| v.position[1].abs() < 0.95) {
            assert!(Vec3::from(v.normal).dot(Vec3::from(v.position)) > 0.99);
        }
    }

    #[test]
    fn bytes_roundtrip_is_exact() {
        let m = Mesh::uv_sphere(2.0, 16, 8);
        let back = Mesh::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back.indices, m.indices);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&back.vertices), bytemuck::cast_slice::<_, u8>(&m.vertices));
    }

    #[test]
    fn from_bytes_rejects_malformed_input() {
        let good = Mesh::uv_sphere(1.0, 8, 4).to_bytes();
        assert!(Mesh::from_bytes(&[]).is_err());
        assert!(Mesh::from_bytes(b"NOTAMESHxxxxxxxxxxxx").is_err());
        assert!(Mesh::from_bytes(&good[..good.len() - 1]).is_err(), "truncated");
        let mut longer = good.clone();
        longer.push(0);
        assert!(Mesh::from_bytes(&longer).is_err(), "trailing byte");
        let mut huge = good.clone();
        huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Mesh::from_bytes(&huge).is_err(), "absurd vertex count");
        let mut oob = good.clone();
        let n = oob.len();
        oob[n - 8..n - 4].copy_from_slice(&u32::MAX.to_le_bytes()); // last index; 4 bytes of set count follow
        assert!(Mesh::from_bytes(&oob).is_err(), "index out of range");
    }

    fn two_sets() -> Mesh {
        let mut m = Mesh::uv_sphere(1.0, 8, 4);
        let half = (m.indices.len() / 6 * 3) as u32;
        let total = m.indices.len() as u32;
        m.sets = vec![
            SetRange { name: "Skin".into(), first_index: 0, index_count: half },
            SetRange { name: "Ünïcode ✓".into(), first_index: half, index_count: total - half },
        ];
        m
    }

    #[test]
    fn extracting_a_set_keeps_only_its_triangles_and_renumbers() {
        let m = two_sets();
        let (a, b) = (m.extract_set(0).unwrap(), m.extract_set(1).unwrap());
        assert_eq!(a.indices.len() + b.indices.len(), m.indices.len());
        assert_eq!(a.sets.len(), 1);
        assert!(a.sets_are_valid() && b.sets_are_valid());
        for part in [&a, &b] {
            assert!(part.indices.iter().all(|&i| (i as usize) < part.vertices.len()));
            assert!(part.vertices.len() <= m.vertices.len());
        }
        // The triangles are the same geometry: compare corner positions in order.
        let corners = |mesh: &Mesh, tris: &[u32]| -> Vec<[f32; 3]> { tris.iter().map(|&i| mesh.vertices[i as usize].position).collect() };
        let r1 = &m.sets[1];
        let want = corners(&m, &m.indices[r1.first_index as usize..(r1.first_index + r1.index_count) as usize]);
        assert_eq!(corners(&b, &b.indices), want);
        assert!(m.extract_set(2).is_none());
        // A mesh without explicit sets is its own single set.
        let plain = Mesh::uv_sphere(1.0, 8, 4);
        assert_eq!(plain.extract_set(0).unwrap().indices.len(), plain.indices.len());
    }

    #[test]
    fn a_ray_finds_the_nearest_surface_and_its_set() {
        let m = two_sets();
        // From far outside, aim at the sphere centre along each axis: the first surface hit is the
        // sphere's, and its set is the one whose triangles contain that point.
        let (set, d) = m.raycast_set(Vec3::new(5.0, 0.3, 0.2), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        assert!((d - 4.07).abs() < 0.3, "distance {d}");
        let hit = Vec3::new(5.0 - d, 0.3, 0.2);
        let north = hit.y > 0.0;
        assert_eq!(set, if north { 0 } else { 1 }, "hit {hit}");
        let (south_set, _) = m.raycast_set(Vec3::new(5.0, -0.6, 0.0), Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        assert_eq!(south_set, 1);
        assert!(m.raycast_set(Vec3::new(5.0, 5.0, 5.0), Vec3::new(1.0, 0.0, 0.0)).is_none(), "a miss");
        assert!(m.raycast_set(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 1.0)).is_none(), "pointing away");
        // A hidden set is seen through: aiming at the northern half with it hidden finds nothing.
        let north = (Vec3::new(5.0, 0.3, 0.2), Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(m.raycast_set_where(north.0, north.1, |_| true).map(|h| h.0), Some(0));
        let seen_through = m.raycast_set_where(north.0, north.1, |s| s != 0);
        assert!(seen_through.is_none_or(|(set, _)| set == 1), "{seen_through:?}");
    }

    #[test]
    fn sets_survive_a_round_trip_and_are_validated() {
        let m = two_sets();
        assert!(m.sets_are_valid());
        let back = Mesh::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back.sets, m.sets);
        // Gaps, overlaps, partial triangles and short coverage are all rejected.
        for edit in [
            |m: &mut Mesh| m.sets[1].first_index += 3,
            |m: &mut Mesh| m.sets[0].index_count += 3,
            |m: &mut Mesh| m.sets[0].index_count += 1,
            |m: &mut Mesh| m.sets[1].index_count -= 3,
            |m: &mut Mesh| m.sets[0].name = "x".repeat(300),
        ] {
            let mut bad = two_sets();
            edit(&mut bad);
            assert!(!bad.sets_are_valid());
            assert!(Mesh::from_bytes(&bad.to_bytes()).is_err());
        }
    }

    #[test]
    fn a_mesh_without_sets_is_one_set_and_old_blobs_still_load() {
        let m = Mesh::uv_sphere(1.0, 8, 4);
        let ranges = m.set_ranges();
        assert_eq!((ranges.len(), ranges[0].first_index, ranges[0].index_count), (1, 0, m.indices.len() as u32));
        // The previous format: old magic and no trailing set list.
        let mut v1 = m.to_bytes();
        v1.truncate(v1.len() - 4);
        v1[..8].copy_from_slice(b"OPMESH01");
        let back = Mesh::from_bytes(&v1).unwrap();
        assert_eq!(back.indices, m.indices);
        assert!(back.sets.is_empty());
        v1.push(0);
        assert!(Mesh::from_bytes(&v1).is_err(), "trailing byte after a v1 blob");
    }

    #[test]
    fn corrupt_set_tables_never_panic() {
        let good = two_sets().to_bytes();
        for cut in 0..good.len() {
            let _ = Mesh::from_bytes(&good[..cut]);
        }
        let mut huge = good.clone();
        let at = 16 + 32 * Mesh::uv_sphere(1.0, 8, 4).vertices.len() + 4 * Mesh::uv_sphere(1.0, 8, 4).indices.len();
        huge[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Mesh::from_bytes(&huge).is_err());
    }

    /// Two unit quads side by side that touch along an edge. In UV space they are separate islands;
    /// as geometry they are one piece.
    fn two_quads() -> Mesh {
        let v = |x: f32, u: f32, v: f32| Vertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [u, v] };
        // Quad A: x 0..1, UV 0..0.4. Quad B: x 1..2, UV 0.6..1 (its shared edge at x=1 has different UVs).
        let mut vertices = vec![v(0.0, 0.0, 0.0), v(1.0, 0.4, 0.0), v(1.0, 0.4, 1.0), v(0.0, 0.0, 1.0)];
        vertices[2].position[1] = 1.0;
        vertices[3].position[1] = 1.0;
        let mut b = vec![v(1.0, 0.6, 0.0), v(2.0, 1.0, 0.0), v(2.0, 1.0, 1.0), v(1.0, 0.6, 1.0)];
        b[2].position[1] = 1.0;
        b[3].position[1] = 1.0;
        vertices.extend(b);
        Mesh { vertices, indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], sets: Vec::new() }
    }

    #[test]
    fn a_fill_finds_the_uv_island_or_the_mesh_part() {
        let m = two_quads();
        assert_eq!(m.connected_triangles(0, Connect::UvIsland), vec![0, 3], "quad A is its own UV island");
        assert_eq!(m.connected_triangles(9, Connect::UvIsland), vec![6, 9]);
        assert_eq!(m.connected_triangles(0, Connect::MeshPart), vec![0, 3, 6, 9], "the quads meet in space");
        assert!(m.connected_triangles(100, Connect::UvIsland).is_empty(), "no such triangle");
    }

    #[test]
    fn a_fill_stays_inside_its_texture_set() {
        let mut m = two_quads();
        m.sets = vec![
            SetRange { name: "a".into(), first_index: 0, index_count: 6 },
            SetRange { name: "b".into(), first_index: 6, index_count: 6 },
        ];
        assert_eq!(m.connected_triangles(0, Connect::MeshPart), vec![0, 3]);
    }

    #[test]
    fn the_ray_finds_the_triangle_and_its_set() {
        let mut m = two_quads();
        m.sets = vec![
            SetRange { name: "a".into(), first_index: 0, index_count: 6 },
            SetRange { name: "b".into(), first_index: 6, index_count: 6 },
        ];
        let hit = m.raycast_triangle_where(Vec3::new(1.5, 0.2, 5.0), -Vec3::Z, |_| true).expect("hits quad B");
        assert_eq!(hit.0, 1);
        assert!(hit.1 == 6 || hit.1 == 9);
        assert!(m.raycast_triangle_where(Vec3::new(1.5, 0.2, 5.0), -Vec3::Z, |s| s == 0).is_none(), "set b is hidden");
    }

    #[test]
    fn a_flat_face_follows_edges_and_stops_at_creases() {
        // A unit square in the z=0 plane (two triangles) with a second square standing up from its
        // right edge, and a third square far away that touches nothing.
        let v = |x: f32, y: f32, z: f32| Vertex { position: [x, y, z], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let vertices = vec![
            v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 1.0, 0.0), v(0.0, 1.0, 0.0), // flat quad
            v(1.0, 0.0, 1.0), v(1.0, 1.0, 1.0),                                      // wall above the right edge
            v(5.0, 5.0, 0.0), v(6.0, 5.0, 0.0), v(6.0, 6.0, 0.0),                    // a lone triangle
        ];
        let indices = vec![0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2, 6, 7, 8];
        let m = Mesh { vertices, indices, sets: Vec::new() };
        assert_eq!(m.connected_triangles(0, Connect::FlatFace(5.0)), vec![0, 3], "the flat quad, not the wall");
        assert_eq!(m.connected_triangles(6, Connect::FlatFace(5.0)), vec![6, 9], "the wall is its own face");
        assert_eq!(m.connected_triangles(12, Connect::FlatFace(5.0)), vec![12]);
        // A generous tolerance walks over the crease.
        assert_eq!(m.connected_triangles(0, Connect::FlatFace(100.0)), vec![0, 3, 6, 9]);
    }

    #[test]
    fn a_ray_can_see_through_triangles_the_caller_refuses() {
        // Two parallel triangles, the one at z=1 in front of the one at z=0.
        let v = |z: f32| [Vertex { position: [-1.0, -1.0, z], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] }, Vertex { position: [1.0, -1.0, z], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0] }, Vertex { position: [0.0, 1.0, z], normal: [0.0, 0.0, 1.0], uv: [0.5, 1.0] }];
        let mut vertices = v(0.0).to_vec();
        vertices.extend(v(1.0));
        let m = Mesh { vertices, indices: vec![0, 1, 2, 3, 4, 5], sets: Vec::new() };
        let (o, d) = (Vec3::new(0.0, 0.0, 5.0), -Vec3::Z);
        assert_eq!(m.raycast_triangle_where(o, d, |_| true), Some((0, 3)), "the front triangle first");
        assert_eq!(m.raycast_triangle_with(o, d, |_| true, |first, _| first != 3), Some((0, 0)), "refusing it reveals the one behind");
        assert_eq!(m.raycast_triangle_with(o, d, |_| true, |_, hit| hit.z < 0.5), Some((0, 0)), "refusing by position too");
        assert!(m.raycast_uv_with(o, d, |_| true, |_, _| false).is_none(), "refusing everything hits nothing");
    }
}
