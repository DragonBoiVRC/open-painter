//! Mesh data and loading for open-painter.

use glam::{Mat4, Vec2, Vec3};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("glTF import failed: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("file contains no triangle geometry")]
    Empty,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
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
        Self { vertices, indices }
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
}
