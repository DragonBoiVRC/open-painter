//! PLY import (ASCII and binary): vertices with optional normals and UVs, faces of any size.

use crate::triangulate::triangulate;
use crate::{Mesh, MeshError, SetRange, Vertex};
use glam::DVec3;
use std::path::Path;

const MAX_FILE_BYTES: u64 = 1 << 30;

fn err(why: impl ToString) -> MeshError {
    MeshError::Format(format!("PLY: {}", why.to_string()))
}

pub fn load(path: &Path) -> Result<Mesh, MeshError> {
    if std::fs::metadata(path).map_err(err)?.len() > MAX_FILE_BYTES {
        return Err(err("file is too large"));
    }
    parse(&std::fs::read(path).map_err(err)?)
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

struct Prop {
    name: String,
    ty: Ty,
    /// For a list: the type of its length.
    list: Option<Ty>,
}

struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

/// Reads values in the file's encoding.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    ascii: bool,
    big: bool,
}

impl Reader<'_> {
    fn value(&mut self, ty: Ty) -> Result<f64, MeshError> {
        if self.ascii {
            let rest = &self.data[self.at..];
            let skip = rest.iter().take_while(|b| b.is_ascii_whitespace()).count();
            let len = rest[skip..].iter().take_while(|b| !b.is_ascii_whitespace()).count();
            if len == 0 {
                return Err(err("the file ends early"));
            }
            let tok = std::str::from_utf8(&rest[skip..skip + len]).map_err(|_| err("bad text"))?;
            self.at += skip + len;
            return tok.parse::<f64>().map_err(|_| err(format!("bad number '{tok}'")));
        }
        let n = ty.size();
        let bytes = self.data.get(self.at..self.at + n).ok_or_else(|| err("the file ends early"))?;
        self.at += n;
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(bytes);
        if self.big {
            b[..n].reverse();
        }
        Ok(match ty {
            Ty::I8 => b[0] as i8 as f64,
            Ty::U8 => b[0] as f64,
            Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Ty::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Ty::F64 => f64::from_le_bytes(b),
        })
    }
}

pub fn parse(data: &[u8]) -> Result<Mesh, MeshError> {
    let end = data.windows(10).position(|w| w == b"end_header").ok_or_else(|| err("no header"))?;
    let header = std::str::from_utf8(&data[..end]).map_err(|_| err("bad header"))?;
    let mut body = end + "end_header".len();
    // The header's newline (maybe CRLF) belongs to it.
    if data.get(body) == Some(&b'\r') {
        body += 1;
    }
    if data.get(body) == Some(&b'\n') {
        body += 1;
    }
    let mut lines = header.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(err("not a PLY file"));
    }
    let (mut ascii, mut big) = (false, false);
    let mut elements: Vec<Element> = Vec::new();
    for line in lines {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["format", f, ..] => match *f {
                "ascii" => ascii = true,
                "binary_little_endian" => {}
                "binary_big_endian" => big = true,
                other => return Err(err(format!("unknown format {other}"))),
            },
            ["element", name, n] => elements.push(Element { name: name.to_string(), count: n.parse().map_err(|_| err("bad element count"))?, props: Vec::new() }),
            ["property", "list", ct, it, name] => {
                let (ct, it) = (Ty::parse(ct).ok_or_else(|| err("bad list type"))?, Ty::parse(it).ok_or_else(|| err("bad list type"))?);
                elements.last_mut().ok_or_else(|| err("property before element"))?.props.push(Prop { name: name.to_string(), ty: it, list: Some(ct) });
            }
            ["property", t, name] => {
                let ty = Ty::parse(t).ok_or_else(|| err(format!("bad type {t}")))?;
                elements.last_mut().ok_or_else(|| err("property before element"))?.props.push(Prop { name: name.to_string(), ty, list: None });
            }
            _ => {}
        }
    }
    let mut r = Reader { data, at: body, ascii, big };
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut tris: Vec<u32> = Vec::new();
    let mut has_normals = false;
    for el in &elements {
        if el.count > 50_000_000 {
            return Err(err("too many elements"));
        }
        for _ in 0..el.count {
            let mut v = Vertex { position: [0.0; 3], normal: [0.0; 3], uv: [0.0; 2] };
            for p in &el.props {
                if let Some(ct) = p.list {
                    let n = r.value(ct)? as usize;
                    if n > 1_000_000 {
                        return Err(err("a face is too large"));
                    }
                    let idx = (0..n).map(|_| r.value(p.ty)).collect::<Result<Vec<_>, _>>()?;
                    if el.name == "face" && (p.name == "vertex_indices" || p.name == "vertex_index") {
                        if idx.iter().any(|&i| i < 0.0 || i as usize >= vertices.len()) {
                            return Err(err("a face uses a vertex that does not exist"));
                        }
                        let corners: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
                        let pts: Vec<DVec3> = corners.iter().map(|&i| DVec3::from(vertices[i as usize].position.map(f64::from))).collect();
                        if corners.len() >= 3 {
                            for t in triangulate(&pts) {
                                tris.extend(t.map(|k| corners[k]));
                            }
                        }
                    }
                    continue;
                }
                let x = r.value(p.ty)? as f32;
                if el.name == "vertex" {
                    match p.name.as_str() {
                        "x" => v.position[0] = x,
                        "y" => v.position[1] = x,
                        "z" => v.position[2] = x,
                        "nx" => {
                            v.normal[0] = x;
                            has_normals = true;
                        }
                        "ny" => v.normal[1] = x,
                        "nz" => v.normal[2] = x,
                        "s" | "u" | "texture_u" | "texture_s" => v.uv[0] = x,
                        "t" | "v" | "texture_v" | "texture_t" => v.uv[1] = 1.0 - x, // V up in the file, down here
                        _ => {}
                    }
                }
            }
            if el.name == "vertex" {
                if v.position.iter().any(|c| !c.is_finite()) {
                    return Err(err("a vertex is not a number"));
                }
                vertices.push(v);
            }
        }
    }
    if tris.is_empty() {
        return Err(MeshError::Empty);
    }
    let mut mesh = Mesh { vertices, indices: tris, ..Default::default() };
    mesh.sets.push(SetRange { name: "DefaultMaterial".into(), first_index: 0, index_count: mesh.indices.len() as u32 });
    if !has_normals {
        mesh.recompute_normals();
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASCII: &str = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty float s\nproperty float t\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 0 0\n1 0 0 1 0\n1 1 0 1 1\n0 1 0 0 1\n4 0 1 2 3\n";

    #[test]
    fn an_ascii_quad_becomes_two_triangles_with_flipped_v() {
        let m = parse(ASCII.as_bytes()).unwrap();
        assert_eq!(m.indices.len(), 6);
        assert_eq!(m.vertices[3].uv, [0.0, 0.0], "t = 1 is V 0 in the image convention");
        assert!(m.vertices[0].normal[2].abs() > 0.9, "normals are computed when the file has none");
    }

    #[test]
    fn binary_big_endian_matches_ascii_and_bad_files_error() {
        let mut b = b"ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar uint vertex_indices\nend_header\n".to_vec();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in v {
                b.extend(c.to_be_bytes());
            }
        }
        b.push(3);
        for i in [0u32, 1, 2] {
            b.extend(i.to_be_bytes());
        }
        assert_eq!(parse(&b).unwrap().indices, vec![0, 1, 2]);
        assert!(parse(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n").is_err());
        assert!(parse(b"not a ply").is_err());
        let mut truncated = b.clone();
        truncated.truncate(truncated.len() - 6);
        assert!(parse(&truncated).is_err());
    }
}
