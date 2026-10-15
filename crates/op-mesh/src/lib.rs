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

    /// Axis-aligned bounds as (min, max). Returns zeros for an empty mesh.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut it = self.vertices.iter().map(|v| Vec3::from(v.position));
        let Some(first) = it.next() else {
            return (Vec3::ZERO, Vec3::ZERO);
        };
        it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)))
    }

    /// Recompute smooth, area-weighted vertex normals from the triangles.
    pub fn recompute_normals(&mut self) {
        let mut acc = vec![Vec3::ZERO; self.vertices.len()];
        for tri in self.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let (pa, pb, pc) = (
                Vec3::from(self.vertices[a].position),
                Vec3::from(self.vertices[b].position),
                Vec3::from(self.vertices[c].position),
            );
            let n = (pb - pa).cross(pc - pa); // length = 2 * area
            acc[a] += n;
            acc[b] += n;
            acc[c] += n;
        }
        for (v, n) in self.vertices.iter_mut().zip(acc) {
            v.normal = n.try_normalize().unwrap_or(Vec3::Y).into();
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

    /// Load every mesh in an FBX file (binary, version 7.x), with its node transforms applied and
    /// converted to Y-up. Triangles are grouped by material into [`SetRange`]s; materials with the
    /// same name on different objects share one set.
    pub fn load_fbx(path: impl AsRef<Path>) -> Result<Self, MeshError> {
        fbx::load(path.as_ref())
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
    fn sphere_is_unit_and_indexed_in_range() {
        let m = Mesh::uv_sphere(1.0, 32, 16);
        assert_eq!(m.indices.len() % 3, 0);
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        let (lo, hi) = m.bounds();
        assert!((hi - Vec3::ONE).abs().max_element() < 1e-3);
        assert!((lo + Vec3::ONE).abs().max_element() < 1e-3);
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
}
