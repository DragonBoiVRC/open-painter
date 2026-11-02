//! STL import (binary and ASCII): triangles only, flat-shaded, one texture set. STL has no UVs.

use crate::{Mesh, MeshError, SetRange, Vertex};
use std::path::Path;

const MAX_FILE_BYTES: u64 = 1 << 30;

fn err(why: impl ToString) -> MeshError {
    MeshError::Format(format!("STL: {}", why.to_string()))
}

pub fn load(path: &Path) -> Result<Mesh, MeshError> {
    if std::fs::metadata(path).map_err(err)?.len() > MAX_FILE_BYTES {
        return Err(err("file is too large"));
    }
    parse(&std::fs::read(path).map_err(err)?)
}

fn mesh_from(tris: Vec<[[f32; 3]; 3]>) -> Result<Mesh, MeshError> {
    if tris.is_empty() {
        return Err(MeshError::Empty);
    }
    let mut mesh = Mesh::default();
    for t in &tris {
        let (a, b, c) = (glam::Vec3::from(t[0]), glam::Vec3::from(t[1]), glam::Vec3::from(t[2]));
        if t.iter().flatten().any(|v| !v.is_finite()) {
            return Err(err("a vertex is not a number"));
        }
        let n = (b - a).cross(c - a).normalize_or_zero().to_array();
        for p in t {
            mesh.indices.push(mesh.vertices.len() as u32);
            mesh.vertices.push(Vertex { position: *p, normal: n, uv: [0.0; 2] });
        }
    }
    mesh.sets.push(SetRange { name: "DefaultMaterial".into(), first_index: 0, index_count: mesh.indices.len() as u32 });
    Ok(mesh)
}

pub fn parse(bytes: &[u8]) -> Result<Mesh, MeshError> {
    // Binary files can start with "solid" too, so trust the size: 80-byte header, count, 50 bytes each.
    if bytes.len() >= 84 {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if n.checked_mul(50).and_then(|b| b.checked_add(84)) == Some(bytes.len()) {
            let f = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            let tris = (0..n)
                .map(|i| {
                    let o = 84 + i * 50 + 12;
                    [0, 1, 2].map(|k| [f(o + k * 12), f(o + k * 12 + 4), f(o + k * 12 + 8)])
                })
                .collect();
            return mesh_from(tris);
        }
    }
    let text = std::str::from_utf8(bytes).map_err(|_| err("neither binary nor ASCII STL"))?;
    let mut pts: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        let mut w = line.split_whitespace();
        if w.next() == Some("vertex") {
            let v: Vec<f32> = w.filter_map(|t| t.parse().ok()).collect();
            if v.len() != 3 {
                return Err(err("a vertex line is incomplete"));
            }
            pts.push([v[0], v[1], v[2]]);
        }
    }
    if pts.len() % 3 != 0 {
        return Err(err("the vertices do not make whole triangles"));
    }
    mesh_from(pts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_and_ascii_stl_read_the_same_triangle() {
        let mut bin = vec![0u8; 80];
        bin.extend(1u32.to_le_bytes());
        bin.extend([0u8; 12]);
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in v {
                bin.extend(c.to_le_bytes());
            }
        }
        bin.extend([0u8; 2]);
        let a = parse(&bin).unwrap();
        let b = parse(b"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid").unwrap();
        assert_eq!(a.indices.len(), 3);
        assert_eq!(a.vertices[1].position, b.vertices[1].position);
        assert_eq!(a.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert!(parse(b"solid\nvertex 0 0\n").is_err());
        assert!(parse(b"").is_err());
    }
}
