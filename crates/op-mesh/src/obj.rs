//! Wavefront OBJ import: positions, UVs, normals and `usemtl` groups (each material is a texture set).

use crate::triangulate::triangulate;
use crate::{Mesh, MeshError, SetRange, Vertex};
use glam::DVec3;
use std::collections::HashMap;
use std::path::Path;

const MAX_FILE_BYTES: u64 = 1 << 30;
const DEFAULT_SET: &str = "DefaultMaterial";

fn err(why: impl ToString) -> MeshError {
    MeshError::Format(format!("OBJ: {}", why.to_string()))
}

pub fn load(path: &Path) -> Result<Mesh, MeshError> {
    if std::fs::metadata(path).map_err(|e| err(e))?.len() > MAX_FILE_BYTES {
        return Err(err("file is too large"));
    }
    let text = std::fs::read_to_string(path).map_err(|e| err(e))?;
    parse(&text)
}

/// Resolve a 1-based (or negative, from-the-end) OBJ index.
fn resolve(token: &str, len: usize) -> Result<Option<usize>, MeshError> {
    if token.is_empty() {
        return Ok(None);
    }
    let i: i64 = token.parse().map_err(|_| err(format!("bad index '{token}'")))?;
    let at = if i > 0 { i - 1 } else { len as i64 + i };
    if at < 0 || at as usize >= len {
        return Err(err(format!("index {i} is out of range")));
    }
    Ok(Some(at as usize))
}

pub fn parse(text: &str) -> Result<Mesh, MeshError> {
    let (mut pos, mut uvs, mut nrm): (Vec<[f64; 3]>, Vec<[f32; 2]>, Vec<[f32; 3]>) = Default::default();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut dedup: HashMap<(usize, Option<usize>, Option<usize>), u32> = HashMap::new();
    let mut sets: Vec<(String, Vec<u32>)> = Vec::new();
    let mut current = None::<usize>;
    let mut missing_normals = false;
    let nums = |rest: &str, n: usize| -> Result<Vec<f64>, MeshError> {
        let v: Vec<f64> = rest.split_whitespace().take(n).map(|t| t.parse::<f64>().map_err(|_| err(format!("bad number '{t}'")))).collect::<Result<_, _>>()?;
        if v.len() < n || v.iter().any(|x| !x.is_finite()) {
            return Err(err("a vertex line is incomplete"));
        }
        Ok(v)
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let (key, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match key {
            "v" => {
                let v = nums(rest, 3)?;
                pos.push([v[0], v[1], v[2]]);
            }
            "vt" => {
                let v = nums(rest, 2)?;
                uvs.push([v[0] as f32, v[1] as f32]);
            }
            "vn" => {
                let v = nums(rest, 3)?;
                nrm.push([v[0] as f32, v[1] as f32, v[2] as f32]);
            }
            "usemtl" => {
                let name = if rest.trim().is_empty() { DEFAULT_SET } else { rest.trim() };
                current = Some(match sets.iter().position(|(n, _)| n == name) {
                    Some(i) => i,
                    None => {
                        sets.push((name.to_string(), Vec::new()));
                        sets.len() - 1
                    }
                });
            }
            "f" => {
                let mut corners = Vec::new();
                for tok in rest.split_whitespace() {
                    let mut it = tok.split('/');
                    let v = resolve(it.next().unwrap_or(""), pos.len())?.ok_or_else(|| err("a face corner has no position"))?;
                    let t = resolve(it.next().unwrap_or(""), uvs.len())?;
                    let n = resolve(it.next().unwrap_or(""), nrm.len())?;
                    missing_normals |= n.is_none();
                    let key = (v, t, n);
                    let index = *dedup.entry(key).or_insert_with(|| {
                        vertices.push(Vertex {
                            position: pos[v].map(|c| c as f32),
                            normal: n.map_or([0.0; 3], |i| nrm[i]),
                            uv: t.map_or([0.0; 2], |i| uvs[i]),
                        });
                        (vertices.len() - 1) as u32
                    });
                    corners.push((index, DVec3::from(pos[v])));
                }
                if corners.len() < 3 {
                    continue;
                }
                let set = *current.get_or_insert_with(|| {
                    match sets.iter().position(|(n, _)| n == DEFAULT_SET) {
                        Some(i) => i,
                        None => {
                            sets.push((DEFAULT_SET.to_string(), Vec::new()));
                            sets.len() - 1
                        }
                    }
                });
                let pts: Vec<DVec3> = corners.iter().map(|c| c.1).collect();
                for t in triangulate(&pts) {
                    sets[set].1.extend(t.map(|k| corners[k].0));
                }
            }
            _ => {}
        }
    }
    let mut mesh = Mesh { vertices, ..Default::default() };
    for (name, indices) in sets {
        if indices.is_empty() {
            continue;
        }
        mesh.sets.push(SetRange { name, first_index: mesh.indices.len() as u32, index_count: indices.len() as u32 });
        mesh.indices.extend(indices);
    }
    if mesh.indices.is_empty() {
        return Err(MeshError::Empty);
    }
    if missing_normals {
        // Corners without a stored normal get smooth normals from their triangles.
        let keep: Vec<([f32; 3], bool)> = mesh.vertices.iter().map(|v| (v.normal, v.normal != [0.0; 3])).collect();
        mesh.recompute_normals();
        for (v, (old, had)) in mesh.vertices.iter_mut().zip(keep) {
            if had {
                v.normal = old;
            }
        }
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBE: &str = "
# two materials: the top face is Lid, the rest Body
v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1
vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1
usemtl Body
f 1/1 4/4 3/3 2/2
f 1/1 2/2 6/3 5/4
f 2/1 3/2 7/3 6/4
f 3/1 4/2 8/3 7/4
f 4/1 1/2 5/3 8/4
usemtl Lid
f 5/1 6/2 7/3 8/4
";

    #[test]
    fn faces_become_triangles_grouped_by_material() {
        let m = parse(CUBE).unwrap();
        assert!(m.sets_are_valid());
        let counts: Vec<(&str, u32)> = m.sets.iter().map(|s| (s.name.as_str(), s.index_count / 3)).collect();
        assert_eq!(counts, [("Body", 10), ("Lid", 2)]);
        assert!(m.vertices.iter().all(|v| (Vec3Ext(v.normal)).unit()), "normals were generated and are unit length");
    }

    struct Vec3Ext([f32; 3]);
    impl Vec3Ext {
        fn unit(&self) -> bool {
            let l = (self.0[0].powi(2) + self.0[1].powi(2) + self.0[2].powi(2)).sqrt();
            (l - 1.0).abs() < 1e-3
        }
    }

    #[test]
    fn negative_indices_and_concave_faces_work() {
        let l = "v 0 0 0\nv 2 0 0\nv 2 1 0\nv 1 1 0\nv 1 2 0\nv 0 2 0\nf -6 -5 -4 -3 -2 -1\n";
        let m = parse(l).unwrap();
        assert_eq!(m.indices.len(), 12);
        assert_eq!(m.sets[0].name, "DefaultMaterial");
        let area: f32 = m
            .indices
            .chunks(3)
            .map(|t| {
                let p = |i: u32| m.vertices[i as usize].position;
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
            })
            .sum();
        assert!((area - 3.0).abs() < 1e-4, "{area}");
    }

    #[test]
    fn bad_files_are_errors_not_panics() {
        for bad in ["", "v 0 0\n", "v 0 0 0\nf 1 2 3\n", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n", "v a b c\n", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/x 2 3\n", "v nan 0 0\n"] {
            assert!(parse(bad).is_err(), "should reject {bad:?}");
        }
    }
}
