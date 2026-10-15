//! UV-space coverage: which texels of the texture a mesh actually uses.
//! Needed to pad (dilate) texture borders around UV islands on export.

use op_mesh::Mesh;

/// Rasterise every triangle's UV footprint into a `size` x `size` mask.
/// Texel (x, y) corresponds to uv = ((x + 0.5)/size, (y + 0.5)/size), the same
/// orientation the paint textures use (v grows downward). A texel counts as covered
/// if its center or any corner lies inside a triangle, which is slightly conservative
/// so that thin slivers are not lost.
pub fn uv_coverage(mesh: &Mesh, size: u32) -> Vec<bool> {
    let s = size as usize;
    let mut mask = vec![false; s * s];
    let sf = size as f32;
    for tri in mesh.indices.chunks_exact(3) {
        let p: Vec<[f32; 2]> = tri
            .iter()
            .map(|&i| {
                let uv = mesh.vertices[i as usize].uv;
                [uv[0] * sf, uv[1] * sf]
            })
            .collect();
        let area = edge(p[0], p[1], p[2]);
        if area.abs() < 1e-9 || p.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        let min_x = p.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let max_x = (p.iter().map(|q| q[0]).fold(f32::NEG_INFINITY, f32::max).ceil().max(0.0) as usize).min(s.saturating_sub(1));
        let min_y = p.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let max_y = (p.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max).ceil().max(0.0) as usize).min(s.saturating_sub(1));
        let sign = area.signum();
        let inside = |q: [f32; 2]| {
            let w0 = edge(p[1], p[2], q) * sign;
            let w1 = edge(p[2], p[0], q) * sign;
            let w2 = edge(p[0], p[1], q) * sign;
            w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
        };
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let (fx, fy) = (x as f32, y as f32);
                let samples = [[fx + 0.5, fy + 0.5], [fx, fy], [fx + 1.0, fy], [fx, fy + 1.0], [fx + 1.0, fy + 1.0]];
                if samples.iter().any(|&q| inside(q)) {
                    mask[y * s + x] = true;
                }
            }
        }
    }
    mask
}

fn edge(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
