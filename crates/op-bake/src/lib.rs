//! Mesh-map baking: ambient occlusion, curvature, thickness, position, world and
//! tangent-space normals, bent normals, height and piece ids, rendered into UV space.
//!
//! Every texel covered by the low-poly UV layout is shaded by casting rays against the
//! "target" geometry: the optional high-poly mesh, or the low-poly mesh itself. Sampling
//! patterns are seeded per texel, so a bake is reproducible regardless of thread scheduling.

mod bvh;
mod geom;

use bvh::{Bvh, Ray};
use geom::Geometry;
use glam::Vec3;
use op_core::{BakedMap, MeshMap};
use op_mesh::Mesh;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

pub const MAX_SIZE: u32 = 8192;

#[derive(Debug, Clone)]
pub struct BakeSettings {
    /// Output resolution (square).
    pub size: u32,
    pub maps: Vec<MeshMap>,
    /// Rays per texel for ambient occlusion, bent normals and thickness.
    pub samples: u32,
    /// Occluders farther than this (fraction of the mesh's bounding-box diagonal) are ignored.
    pub ao_distance: f32,
    /// Rays longer than this (fraction of the diagonal) count as maximally thick.
    pub thickness_distance: f32,
    /// Multiplier for the curvature map contrast.
    pub curvature_strength: f32,
    /// How far in front of / behind the low-poly surface the high-poly mesh is searched
    /// (fractions of the diagonal). Also sets the height range.
    pub cage_front: f32,
    pub cage_rear: f32,
    /// Texels of padding added around UV islands: the nearest shaded values are copied
    /// outward so filtering, blurring and mip-mapping never pull in empty space.
    pub padding: u32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            size: 1024,
            maps: MeshMap::ALL.to_vec(),
            samples: 32,
            ao_distance: 0.1,
            thickness_distance: 0.25,
            curvature_strength: 1.0,
            cage_front: 0.05,
            cage_rear: 0.05,
            padding: 16,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    #[error("{0}")]
    Invalid(String),
    #[error("the bake was cancelled")]
    Cancelled,
}

/// Shared between the baking thread and the UI.
#[derive(Debug, Default)]
pub struct Progress {
    rows_done: AtomicU32,
    rows_total: AtomicU32,
    cancel: AtomicBool,
}

impl Progress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// 0.0 ..= 1.0
    pub fn fraction(&self) -> f32 {
        let total = self.rows_total.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { self.rows_done.load(Ordering::Relaxed) as f32 / total as f32 }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

pub struct BakeResult {
    pub size: u32,
    pub maps: HashMap<MeshMap, BakedMap>,
    /// Texels that were shaded (covered by some triangle's UV footprint).
    pub covered: usize,
}

/// One UV texel's owner triangle and barycentric weights of its vertices 1 and 2.
#[derive(Clone, Copy)]
struct Sample {
    tri: u32,
    w1: f32,
    w2: f32,
}

const NONE: u32 = u32::MAX;

/// Rasterise every triangle's UV footprint. When footprints overlap the last triangle wins.
fn rasterize(geo: &Geometry, size: u32) -> Vec<Sample> {
    let s = size as usize;
    let sf = size as f32;
    let mut out = vec![Sample { tri: NONE, w1: 0.0, w2: 0.0 }; s * s];
    for (ti, tri) in geo.tris.iter().enumerate() {
        let p: [[f32; 2]; 3] = tri.map(|i| {
            let uv = geo.uvs[i as usize];
            [uv.x * sf, uv.y * sf]
        });
        if p.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        let edge = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let area = edge(p[0], p[1], p[2]);
        if area.abs() < 1e-9 {
            continue;
        }
        let lo = |k: usize| p.iter().map(|q| q[k]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let hi = |k: usize| (p.iter().map(|q| q[k]).fold(f32::NEG_INFINITY, f32::max).ceil().max(0.0) as usize).min(s - 1);
        let eps = -1e-4;
        for y in lo(1)..=hi(1) {
            for x in lo(0)..=hi(0) {
                let c = [x as f32 + 0.5, y as f32 + 0.5];
                let w0 = edge(p[1], p[2], c) / area;
                let w1 = edge(p[2], p[0], c) / area;
                let w2 = edge(p[0], p[1], c) / area;
                if w0 >= eps && w1 >= eps && w2 >= eps {
                    out[y * s + x] = Sample { tri: ti as u32, w1, w2 };
                }
            }
        }
    }
    out
}

/// Copy shaded values outward into uncovered texels, `iterations` texels deep. All maps are
/// filled together; each new texel is the average of the already-filled texels around it.
/// Work is proportional to the padded area, not to `iterations * size * size`.
fn pad_edges(bufs: &mut [Vec<u16>], channels: &[usize], covered: &[bool], size: usize, iterations: u32) {
    if iterations == 0 || bufs.is_empty() {
        return;
    }
    let mut filled = covered.to_vec();
    let neighbours = |i: usize| {
        let (x, y) = ((i % size) as i64, (i / size) as i64);
        (-1i64..=1).flat_map(move |dy| (-1i64..=1).map(move |dx| (dx, dy))).filter_map(move |(dx, dy)| {
            let (nx, ny) = (x + dx, y + dy);
            ((dx, dy) != (0, 0) && nx >= 0 && ny >= 0 && nx < size as i64 && ny < size as i64).then(|| ny as usize * size + nx as usize)
        })
    };
    // First frontier: empty texels that touch a filled one.
    let mut queued = vec![false; filled.len()];
    let mut frontier: Vec<usize> = Vec::new();
    for i in 0..filled.len() {
        if !filled[i] && neighbours(i).any(|n| filled[n]) {
            queued[i] = true;
            frontier.push(i);
        }
    }
    for _ in 0..iterations {
        if frontier.is_empty() {
            break;
        }
        // Compute the whole wave from the previous state, then write it.
        let wave: Vec<(usize, Vec<u16>)> = frontier
            .iter()
            .map(|&i| {
                let src: Vec<usize> = neighbours(i).filter(|&n| filled[n]).collect();
                let mut vals = Vec::new();
                for (buf, &c) in bufs.iter().zip(channels) {
                    for k in 0..c {
                        let sum: u32 = src.iter().map(|&n| buf[n * c + k] as u32).sum();
                        vals.push(((sum + src.len() as u32 / 2) / src.len() as u32) as u16);
                    }
                }
                (i, vals)
            })
            .collect();
        let mut next = Vec::new();
        for (i, vals) in wave {
            let mut at = 0;
            for (buf, &c) in bufs.iter_mut().zip(channels) {
                buf[i * c..(i + 1) * c].copy_from_slice(&vals[at..at + c]);
                at += c;
            }
            filled[i] = true;
            for n in neighbours(i) {
                if !filled[n] && !queued[n] {
                    queued[n] = true;
                    next.push(n);
                }
            }
        }
        frontier = next;
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn unit_float(x: u64) -> f32 {
    ((x >> 40) as f32) / (1u64 << 24) as f32
}

fn radical_inverse(mut i: u32) -> f32 {
    i = i.rotate_right(16);
    i = ((i & 0x5555_5555) << 1) | ((i & 0xaaaa_aaaa) >> 1);
    i = ((i & 0x3333_3333) << 2) | ((i & 0xcccc_cccc) >> 2);
    i = ((i & 0x0f0f_0f0f) << 4) | ((i & 0xf0f0_f0f0) >> 4);
    i = ((i & 0x00ff_00ff) << 8) | ((i & 0xff00_ff00) >> 8);
    i as f32 * 2.328_306_4e-10
}

/// Cosine-weighted direction in the local frame (z = along the normal).
fn cosine_dir(u1: f32, u2: f32) -> Vec3 {
    let r = u1.sqrt();
    let phi = std::f32::consts::TAU * u2;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).max(0.0).sqrt())
}

fn hue_color(h: f32) -> [f32; 3] {
    let (s, v) = (0.55, 0.95);
    let k = |n: f32| {
        let kk = (n + h * 6.0) % 6.0;
        v - v * s * kk.min(4.0 - kk).clamp(0.0, 1.0)
    };
    [k(5.0), k(3.0), k(1.0)]
}

/// Surface point and orthonormal frame at a texel.
struct Surface {
    p: Vec3,
    n: Vec3,
    t: Vec3,
    b: Vec3,
}

struct Context<'a> {
    geo: &'a Geometry,
    high: Option<&'a Geometry>,
    target: Bvh,
    settings: &'a BakeSettings,
    diag: f32,
    bias: f32,
    extent: Vec3,
    wants_rays: bool,
    wants_thickness: bool,
    wants_high: bool,
}

impl Context<'_> {
    fn surface(&self, s: Sample) -> Surface {
        let tri = self.geo.tris[s.tri as usize].map(|i| i as usize);
        let w = [1.0 - s.w1 - s.w2, s.w1, s.w2];
        let mix = |v: &[Vec3]| v[tri[0]] * w[0] + v[tri[1]] * w[1] + v[tri[2]] * w[2];
        let p = mix(&self.geo.positions);
        let face = self.geo.face_normal(s.tri as usize);
        let mut n = mix(&self.geo.normals).normalize_or_zero();
        if n == Vec3::ZERO {
            n = if face == Vec3::ZERO { Vec3::Z } else { face };
        }
        let tangent = mix(&self.geo.tangents);
        let mut t = (tangent - n * n.dot(tangent)).normalize_or_zero();
        if t == Vec3::ZERO {
            let helper = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            t = (helper - n * n.dot(helper)).normalize_or_zero();
        }
        let hand: f32 = tri.iter().map(|&i| self.geo.handedness[i]).sum();
        let b = n.cross(t) * if hand < 0.0 { -1.0 } else { 1.0 };
        Surface { p, n, t, b }
    }

    fn shade(&self, x: u32, y: u32, s: Sample, maps: &[MeshMap], out: &mut Vec<u16>) {
        let sf = self.surface(s);
        let seed = splitmix(((y as u64) << 32) | x as u64);
        let (rx, ry) = (unit_float(splitmix(seed ^ 1)), unit_float(splitmix(seed ^ 2)));
        let n_samples = self.settings.samples.max(1);

        // Hemisphere visibility: used by AO and bent normals.
        let (mut ao, mut bent) = (1.0, sf.n);
        if self.wants_rays {
            let max_d = self.settings.ao_distance * self.diag;
            let (mut free, mut acc) = (0u32, Vec3::ZERO);
            for i in 0..n_samples {
                let u1 = (((i as f32 + 0.5) / n_samples as f32) + rx).fract();
                let u2 = (radical_inverse(i) + ry).fract();
                let l = cosine_dir(u1, u2);
                let dir = sf.t * l.x + sf.b * l.y + sf.n * l.z;
                let ray = Ray { origin: sf.p + sf.n * self.bias, dir, tmin: 0.0, tmax: max_d };
                if !self.target.any_hit(&ray) {
                    free += 1;
                    acc += dir;
                }
            }
            ao = free as f32 / n_samples as f32;
            if acc.length_squared() > 1e-12 {
                bent = acc.normalize();
            }
        }

        let thickness = if self.wants_thickness {
            let max_d = self.settings.thickness_distance * self.diag;
            let mut sum = 0.0;
            for i in 0..n_samples {
                let u1 = (((i as f32 + 0.5) / n_samples as f32) + ry).fract();
                let u2 = (radical_inverse(i) + rx).fract();
                let l = cosine_dir(u1, u2);
                let dir = sf.t * l.x + sf.b * l.y - sf.n * l.z;
                let ray = Ray { origin: sf.p - sf.n * self.bias, dir, tmin: 0.0, tmax: max_d };
                sum += self.target.closest_hit(&ray).map_or(max_d, |h| h.t.min(max_d));
            }
            sum / n_samples as f32 / max_d
        } else {
            0.0
        };

        // High-poly projection: detail normal, height and coverage.
        let (mut detail_n, mut height, mut opacity) = (Vec3::Z, 0.5, 1.0);
        if self.wants_high {
            if let Some(high) = self.high {
                let front = self.settings.cage_front * self.diag;
                let rear = self.settings.cage_rear * self.diag;
                let ray = Ray { origin: sf.p + sf.n * front, dir: -sf.n, tmin: 0.0, tmax: front + rear };
                match self.target.closest_hit(&ray) {
                    Some(h) => {
                        let tri = high.tris[h.tri as usize].map(|i| i as usize);
                        let w = [1.0 - h.u - h.v, h.u, h.v];
                        let mut nh = (high.normals[tri[0]] * w[0] + high.normals[tri[1]] * w[1] + high.normals[tri[2]] * w[2])
                            .normalize_or_zero();
                        if nh == Vec3::ZERO {
                            nh = high.face_normal(h.tri as usize);
                        }
                        detail_n = Vec3::new(nh.dot(sf.t), nh.dot(sf.b), nh.dot(sf.n)).normalize_or_zero();
                        let range = front.max(rear).max(1e-9);
                        height = 0.5 + 0.5 * ((front - h.t) / range).clamp(-1.0, 1.0);
                    }
                    None => opacity = 0.0,
                }
            }
        }

        for map in maps {
            let v: [f32; 3] = match map {
                MeshMap::Position => {
                    let q = (sf.p - self.geo.min) / self.extent;
                    [q.x, q.y, q.z]
                }
                MeshMap::WorldSpaceNormal => [sf.n.x * 0.5 + 0.5, sf.n.y * 0.5 + 0.5, sf.n.z * 0.5 + 0.5],
                MeshMap::AmbientOcclusion => [ao; 3],
                MeshMap::Thickness => [thickness; 3],
                MeshMap::Curvature => {
                    let tri = self.geo.tris[s.tri as usize].map(|i| i as usize);
                    let w = [1.0 - s.w1 - s.w2, s.w1, s.w2];
                    let k: f32 = (0..3).map(|c| self.geo.curvature[tri[c]] * w[c]).sum();
                    let c = k * self.diag * 0.1 * self.settings.curvature_strength;
                    [0.5 + 0.5 * c.clamp(-1.0, 1.0); 3]
                }
                MeshMap::NormalBase => [detail_n.x * 0.5 + 0.5, detail_n.y * 0.5 + 0.5, detail_n.z * 0.5 + 0.5],
                MeshMap::HeightBase => [height; 3],
                MeshMap::OpacityBase => [opacity; 3],
                MeshMap::BentNormals => {
                    let l = Vec3::new(bent.dot(sf.t), bent.dot(sf.b), bent.dot(sf.n));
                    [l.x * 0.5 + 0.5, l.y * 0.5 + 0.5, l.z * 0.5 + 0.5]
                }
                MeshMap::Id => {
                    let h = unit_float(splitmix(self.geo.piece[s.tri as usize] as u64 ^ 0x1d));
                    hue_color(h)
                }
            };
            for c in 0..map.channels() as usize {
                out.push(BakedMap::quantize(v[c]));
            }
        }
    }
}

fn validate(low: &Mesh, high: Option<&Mesh>, settings: &BakeSettings) -> Result<(), BakeError> {
    let bad = |m: String| Err(BakeError::Invalid(m));
    if settings.size == 0 || settings.size > MAX_SIZE {
        return bad(format!("size must be between 1 and {MAX_SIZE}"));
    }
    if settings.maps.is_empty() {
        return bad("no maps selected".into());
    }
    if low.indices.len() < 3 || low.indices.len() % 3 != 0 {
        return bad("the mesh has no triangles".into());
    }
    if low.indices.iter().any(|&i| i as usize >= low.vertices.len()) {
        return bad("the mesh has out-of-range indices".into());
    }
    if let Some(h) = high {
        if h.indices.len() < 3 || h.indices.len() % 3 != 0 || h.indices.iter().any(|&i| i as usize >= h.vertices.len()) {
            return bad("the high-poly mesh is empty or invalid".into());
        }
    }
    let frac = |v: f32, name: &str| {
        if v.is_finite() && v > 0.0 { Ok(()) } else { Err(BakeError::Invalid(format!("{name} must be a positive number"))) }
    };
    frac(settings.ao_distance, "AO distance")?;
    frac(settings.thickness_distance, "thickness distance")?;
    frac(settings.cage_front, "front distance")?;
    frac(settings.cage_rear, "rear distance")?;
    if !settings.curvature_strength.is_finite() {
        return bad("curvature strength must be a number".into());
    }
    let rays = settings.maps.iter().any(|m| matches!(m, MeshMap::AmbientOcclusion | MeshMap::BentNormals | MeshMap::Thickness));
    if rays && settings.samples == 0 {
        return bad("at least one ray sample is needed".into());
    }
    Ok(())
}

/// Bake `settings.maps` for `low`. When `high` is given, normals, height and opacity are
/// projected from it, and rays for AO / thickness are cast against it.
pub fn bake(low: &Mesh, high: Option<&Mesh>, settings: &BakeSettings, progress: &Progress) -> Result<BakeResult, BakeError> {
    validate(low, high, settings)?;
    let geo = Geometry::new(low);
    let high_geo = high.map(Geometry::new);
    let target_geo = high_geo.as_ref().unwrap_or(&geo);
    let diag = geo.diagonal();
    let ctx = Context {
        geo: &geo,
        high: high_geo.as_ref(),
        target: Bvh::build(&target_geo.triangles()),
        settings,
        diag,
        bias: (diag * 1e-4).max(1e-7),
        extent: (geo.max - geo.min).max(Vec3::splat(1e-9)),
        wants_rays: settings.maps.iter().any(|m| matches!(m, MeshMap::AmbientOcclusion | MeshMap::BentNormals)),
        wants_thickness: settings.maps.contains(&MeshMap::Thickness),
        wants_high: settings
            .maps
            .iter()
            .any(|m| matches!(m, MeshMap::NormalBase | MeshMap::HeightBase | MeshMap::OpacityBase)),
    };

    let size = settings.size;
    let samples = rasterize(&geo, size);
    progress.rows_done.store(0, Ordering::Relaxed);
    progress.rows_total.store(size, Ordering::Relaxed);

    let maps = &settings.maps;
    let per_texel: usize = maps.iter().map(|m| m.channels() as usize).sum();
    let rows: Vec<Option<Vec<u16>>> = (0..size)
        .into_par_iter()
        .map(|y| {
            if progress.is_cancelled() {
                return None;
            }
            // Uncovered texels stay zero; the exporter pads them from their neighbours.
            let mut row = Vec::with_capacity(size as usize * per_texel);
            for x in 0..size {
                let s = samples[(y * size + x) as usize];
                if s.tri == NONE {
                    row.extend(std::iter::repeat_n(0u16, per_texel));
                } else {
                    ctx.shade(x, y, s, maps, &mut row);
                }
            }
            progress.rows_done.fetch_add(1, Ordering::Relaxed);
            Some(row)
        })
        .collect();
    if rows.iter().any(Option::is_none) || progress.is_cancelled() {
        return Err(BakeError::Cancelled);
    }

    // De-interleave into one buffer per map.
    let n = size as usize * size as usize;
    let mut bufs: Vec<Vec<u16>> = maps.iter().map(|m| Vec::with_capacity(n * m.channels() as usize)).collect();
    for row in rows.into_iter().flatten() {
        for texel in row.chunks_exact(per_texel) {
            let mut at = 0;
            for (buf, map) in bufs.iter_mut().zip(maps) {
                let c = map.channels() as usize;
                buf.extend_from_slice(&texel[at..at + c]);
                at += c;
            }
        }
    }
    let covered_mask: Vec<bool> = samples.iter().map(|s| s.tri != NONE).collect();
    let channels: Vec<usize> = maps.iter().map(|m| m.channels() as usize).collect();
    pad_edges(&mut bufs, &channels, &covered_mask, size as usize, settings.padding);
    let mut result = HashMap::new();
    for (map, data) in maps.iter().zip(bufs) {
        let baked = BakedMap::new(size, map.channels(), data).map_err(BakeError::Invalid)?;
        result.insert(*map, baked);
    }
    let covered = covered_mask.iter().filter(|c| **c).count();
    Ok(BakeResult { size, maps: result, covered })
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_mesh::Vertex;

    fn settings(size: u32, maps: &[MeshMap]) -> BakeSettings {
        BakeSettings { size, maps: maps.to_vec(), samples: 32, ..Default::default() }
    }

    fn run(low: &Mesh, high: Option<&Mesh>, s: &BakeSettings) -> BakeResult {
        bake(low, high, s, &Progress::default()).unwrap()
    }

    fn plane() -> Mesh {
        geom::test_quad(0.0)
    }

    /// Texel (x, y) of a baked map.
    fn at(r: &BakeResult, map: MeshMap, x: u32, y: u32) -> [f32; 3] {
        r.maps[&map].texel((y * r.size + x) as usize)
    }

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn rasterization_covers_the_uv_footprint_with_correct_barycentrics() {
        let g = Geometry::new(&plane());
        let s = rasterize(&g, 16);
        assert!(s.iter().all(|t| t.tri != NONE), "a full-square quad covers every texel");
        // Texel (3, 5): centre uv = (3.5/16, 5.5/16); position = (u, 1 - v).
        let t = s[5 * 16 + 3];
        let tri = g.tris[t.tri as usize].map(|i| i as usize);
        let w = [1.0 - t.w1 - t.w2, t.w1, t.w2];
        let p = g.positions[tri[0]] * w[0] + g.positions[tri[1]] * w[1] + g.positions[tri[2]] * w[2];
        assert!((p - Vec3::new(3.5 / 16.0, 1.0 - 5.5 / 16.0, 0.0)).length() < 1e-5, "{p}");
    }

    #[test]
    fn position_map_spans_the_bounding_box() {
        let r = run(&plane(), None, &settings(8, &[MeshMap::Position]));
        // Left/top texel: x ~ 0, y ~ 1; right/bottom: x ~ 1, y ~ 0 (v grows downward).
        assert!(close(at(&r, MeshMap::Position, 0, 0), [0.0625, 1.0 - 0.0625, 0.0], 1e-3), "{:?}", at(&r, MeshMap::Position, 0, 0));
        assert!(close(at(&r, MeshMap::Position, 7, 7), [1.0 - 0.0625, 0.0625, 0.0], 1e-3));
        assert_eq!(r.covered, 64);
    }

    #[test]
    fn world_normals_of_a_sphere_point_outward() {
        let sphere = Mesh::uv_sphere(1.0, 64, 32);
        let r = run(&sphere, None, &settings(64, &[MeshMap::WorldSpaceNormal, MeshMap::Position]));
        let mut checked = 0;
        for y in (8..56).step_by(5) {
            for x in (0..64).step_by(7) {
                let p = at(&r, MeshMap::Position, x, y);
                let world = Vec3::new(p[0], p[1], p[2]) * 2.0 - Vec3::ONE; // bbox is [-1, 1]
                let n = at(&r, MeshMap::WorldSpaceNormal, x, y);
                let n = Vec3::new(n[0], n[1], n[2]) * 2.0 - Vec3::ONE;
                assert!(n.normalize().dot(world.normalize()) > 0.97, "texel ({x},{y}): normal {n} vs radial {world}");
                checked += 1;
            }
        }
        assert!(checked > 40);
    }

    #[test]
    fn ambient_occlusion_is_open_on_a_convex_sphere_and_dark_in_a_corner() {
        let sphere = Mesh::uv_sphere(1.0, 48, 24);
        let r = run(&sphere, None, &settings(64, &[MeshMap::AmbientOcclusion]));
        // Known artifact: the sampling hemisphere follows the smoothed normal, so on a faceted
        // surface an occasional grazing ray dips below a facet and hits its neighbour. It must stay rare.
        let (mut worst, mut sum, mut count): (f32, f32, u32) = (1.0, 0.0, 0);
        for y in 6..58 {
            for x in 0..64 {
                let v = at(&r, MeshMap::AmbientOcclusion, x, y)[0];
                worst = worst.min(v);
                sum += v;
                count += 1;
            }
        }
        assert!(worst > 0.9, "no texel of a convex sphere may be noticeably occluded, darkest {worst}");
        assert!(sum / count as f32 > 0.995, "mean AO should be essentially open, got {}", sum / count as f32);

        // Open-box corner: a floor plus a wall rising from its x = 0 edge. Texels beside the
        // wall must be darker than ones far from it.
        let v = |p: [f32; 3], uv: [f32; 2], n: [f32; 3]| Vertex { position: p, normal: n, uv };
        let floor = [
            v([0.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]),
            v([1.0, 0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 1.0]),
            v([1.0, 1.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0]),
            v([0.0, 1.0, 0.0], [0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        // The wall's UVs lie outside the floor's footprint: it only has to exist as geometry.
        let wall = [
            v([0.0, 0.0, 0.0], [5.0, 5.0], [1.0, 0.0, 0.0]),
            v([0.0, 1.0, 0.0], [5.0, 5.0], [1.0, 0.0, 0.0]),
            v([0.0, 1.0, 1.0], [5.0, 5.0], [1.0, 0.0, 0.0]),
            v([0.0, 0.0, 1.0], [5.0, 5.0], [1.0, 0.0, 0.0]),
        ];
        let mut vertices = floor.to_vec();
        vertices.extend(wall);
        let indices = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        let corner = Mesh { vertices, indices, ..Default::default() };
        let r = run(&corner, None, &BakeSettings { size: 32, maps: vec![MeshMap::AmbientOcclusion], samples: 64, ao_distance: 0.5, ..Default::default() });
        let near = at(&r, MeshMap::AmbientOcclusion, 1, 16)[0];
        let far = at(&r, MeshMap::AmbientOcclusion, 30, 16)[0];
        assert!(near < far - 0.1, "texels beside the wall must be darker: near {near}, far {far}");
        assert!(far > 0.85, "far from the wall the floor is mostly open: {far}");
    }

    #[test]
    fn inside_of_a_closed_shell_is_fully_occluded() {
        let sphere = {
            let mut m = Mesh::uv_sphere(1.0, 24, 12);
            for t in m.indices.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
            for v in &mut m.vertices {
                v.normal = [-v.normal[0], -v.normal[1], -v.normal[2]];
            }
            m
        };
        let r = run(&sphere, None, &BakeSettings { size: 32, maps: vec![MeshMap::AmbientOcclusion], samples: 32, ao_distance: 4.0, ..Default::default() });
        let mean = (8..24).map(|y| at(&r, MeshMap::AmbientOcclusion, 16, y)[0]).sum::<f32>() / 16.0;
        assert!(mean < 0.1, "rays inside a closed shell always hit it: {mean}");
    }

    #[test]
    fn thickness_grows_with_the_slab_and_stays_in_range() {
        let slab = |t: f32| {
            // Box [0,1]x[0,1]x[0,t]; only the top face is UV-mapped (others sit at uv 5,5, outside).
            let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
            let quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3, n: Vec3, uv: bool| {
                let u = |x: f32, y: f32| if uv { [x, 1.0 - y] } else { [5.0, 5.0] };
                [a, b, c, d].map(|q| Vertex { position: q.into(), normal: n.into(), uv: u(q.x, q.y) })
            };
            let faces = [
                quad(p(0., 0., t), p(1., 0., t), p(1., 1., t), p(0., 1., t), Vec3::Z, true),
                quad(p(0., 1., 0.), p(1., 1., 0.), p(1., 0., 0.), p(0., 0., 0.), Vec3::NEG_Z, false),
                quad(p(0., 0., 0.), p(1., 0., 0.), p(1., 0., t), p(0., 0., t), Vec3::NEG_Y, false),
                quad(p(1., 1., 0.), p(0., 1., 0.), p(0., 1., t), p(1., 1., t), Vec3::Y, false),
                quad(p(0., 1., 0.), p(0., 0., 0.), p(0., 0., t), p(0., 1., t), Vec3::NEG_X, false),
                quad(p(1., 0., 0.), p(1., 1., 0.), p(1., 1., t), p(1., 0., t), Vec3::X, false),
            ];
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            for f in faces {
                let b = vertices.len() as u32;
                vertices.extend(f);
                indices.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
            }
            Mesh { vertices, indices, ..Default::default() }
        };
        let s = BakeSettings { size: 16, maps: vec![MeshMap::Thickness], samples: 64, thickness_distance: 0.5, ..Default::default() };
        let (thin, thick) = (run(&slab(0.1), None, &s), run(&slab(0.3), None, &s));
        let (a, b) = (at(&thin, MeshMap::Thickness, 8, 8)[0], at(&thick, MeshMap::Thickness, 8, 8)[0]);
        assert!(a < b, "thicker slab must read thicker: {a} vs {b}");
        assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b));
        // Rays through a slab of thickness t are at least t long: the mean can't be below t / max distance.
        let diag = Vec3::new(1.0, 1.0, 0.1).length();
        assert!(a >= 0.1 / (0.5 * diag) - 0.02, "thin slab lower bound: {a}");
    }

    #[test]
    fn curvature_is_neutral_on_a_plane_convex_on_a_sphere_and_concave_inside_out() {
        let r = run(&plane(), None, &settings(8, &[MeshMap::Curvature]));
        assert!(close(at(&r, MeshMap::Curvature, 4, 4), [0.5; 3], 1e-3));
        let sphere = Mesh::uv_sphere(1.0, 48, 24);
        let r = run(&sphere, None, &settings(48, &[MeshMap::Curvature]));
        assert!(at(&r, MeshMap::Curvature, 24, 24)[0] > 0.55, "convex reads brighter than 0.5");
        let mut inside_out = sphere;
        for t in inside_out.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        let r = run(&inside_out, None, &settings(48, &[MeshMap::Curvature]));
        assert!(at(&r, MeshMap::Curvature, 24, 24)[0] < 0.45, "concave reads darker than 0.5");
    }

    #[test]
    fn normal_map_is_flat_without_a_high_poly_and_for_identical_meshes() {
        let flat = run(&plane(), None, &settings(8, &[MeshMap::NormalBase, MeshMap::HeightBase, MeshMap::OpacityBase]));
        assert!(close(at(&flat, MeshMap::NormalBase, 3, 3), [0.5, 0.5, 1.0], 1e-3));
        assert!(close(at(&flat, MeshMap::HeightBase, 3, 3), [0.5; 3], 1e-3));
        // Baking a mesh onto itself must give a flat result too (ray hits, identity transform).
        let sphere = Mesh::uv_sphere(1.0, 48, 24);
        let r = run(&sphere, Some(&sphere), &settings(48, &[MeshMap::NormalBase]));
        let mut worst: f32 = 0.0;
        for y in 6..42 {
            for x in 0..48 {
                let n = at(&r, MeshMap::NormalBase, x, y);
                worst = worst.max((n[0] - 0.5).abs()).max((n[1] - 0.5).abs()).max(1.0 - n[2]);
            }
        }
        assert!(worst < 0.03, "self-bake should be flat, worst deviation {worst}");
    }

    /// A bump on a high-poly plane must tilt the baked normal away from the bump's crest,
    /// with the sign convention of OpenGL normal maps (red = +u / right, green = image up).
    #[test]
    fn high_poly_bump_tilts_normals_in_the_expected_directions() {
        let n = 40;
        let h = |x: f32, y: f32| {
            let (dx, dy) = (x - 0.5, y - 0.5);
            0.08 * (-(dx * dx + dy * dy) / 0.02).exp()
        };
        let mut vertices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
                vertices.push(Vertex { position: [x, y, h(x, y)], normal: [0.0, 0.0, 1.0], uv: [x, 1.0 - y] });
            }
        }
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = (j * (n + 1) + i) as u32;
                let (b, c, d) = (a + 1, a + n as u32 + 2, a + n as u32 + 1);
                indices.extend([a, b, c, a, c, d]);
            }
        }
        let mut high = Mesh { vertices, indices, ..Default::default() };
        high.recompute_normals();
        let s = BakeSettings { size: 32, maps: vec![MeshMap::NormalBase, MeshMap::HeightBase, MeshMap::OpacityBase], cage_front: 0.2, cage_rear: 0.2, ..Default::default() };
        let r = run(&plane(), Some(&high), &s);
        // Texel columns: x grows with u (world +X). Rows: y grows downward (world -Y).
        let right_flank = at(&r, MeshMap::NormalBase, 20, 16); // x ~ 0.64: normal leans +X
        let left_flank = at(&r, MeshMap::NormalBase, 11, 16); //  x ~ 0.36: leans -X
        let upper_flank = at(&r, MeshMap::NormalBase, 16, 11); // top of image = +Y: leans +Y -> green > 0.5
        let lower_flank = at(&r, MeshMap::NormalBase, 16, 20);
        assert!(right_flank[0] > 0.55 && left_flank[0] < 0.45, "red follows +X: {right_flank:?} / {left_flank:?}");
        assert!(upper_flank[1] > 0.55 && lower_flank[1] < 0.45, "green follows image-up (+Y): {upper_flank:?} / {lower_flank:?}");
        assert!(right_flank[2] > 0.5, "z stays positive");
        // Height: the crest is above the low-poly plane, the corners are on it.
        assert!(at(&r, MeshMap::HeightBase, 16, 16)[0] > at(&r, MeshMap::HeightBase, 0, 0)[0] + 0.1);
        assert!(close(at(&r, MeshMap::OpacityBase, 16, 16), [1.0; 3], 1e-3), "high poly found under the crest");
    }

    #[test]
    fn height_reflects_the_signed_offset_of_the_high_poly() {
        let mut high = plane();
        for v in &mut high.vertices {
            v.position[2] = 0.1; // 0.1 above the low-poly plane
        }
        let s = BakeSettings { size: 8, maps: vec![MeshMap::HeightBase], cage_front: 0.2, cage_rear: 0.1, ..Default::default() };
        let r = run(&plane(), Some(&high), &s);
        // The low plane's bounding box diagonal is sqrt(2); the height range is max(front, rear) * diag.
        let range = 0.2f32 * 2f32.sqrt();
        let want = 0.5 + 0.5 * (0.1 / range);
        assert!((at(&r, MeshMap::HeightBase, 4, 4)[0] - want).abs() < 0.01, "got {} want {want}", at(&r, MeshMap::HeightBase, 4, 4)[0]);
    }

    #[test]
    fn missing_high_poly_surface_reads_as_transparent() {
        let mut far = plane();
        for v in &mut far.vertices {
            v.position[2] = 10.0; // far outside the cage
        }
        let s = BakeSettings { size: 8, maps: vec![MeshMap::OpacityBase, MeshMap::NormalBase], ..Default::default() };
        let r = run(&plane(), Some(&far), &s);
        assert!(close(at(&r, MeshMap::OpacityBase, 4, 4), [0.0; 3], 1e-3));
        assert!(close(at(&r, MeshMap::NormalBase, 4, 4), [0.5, 0.5, 1.0], 1e-3), "no hit: flat normal");
    }

    #[test]
    fn ids_distinguish_pieces() {
        let mut first = plane();
        for v in &mut first.vertices {
            v.uv[0] *= 0.5; // left half of UV space
        }
        let mut second = plane();
        for v in &mut second.vertices {
            v.position[0] += 3.0;
            v.uv[0] = v.uv[0] * 0.5 + 0.5; // right half
        }
        let mut all = first;
        let off = all.vertices.len() as u32;
        all.vertices.extend(second.vertices.clone());
        all.indices.extend(second.indices.iter().map(|i| i + off));
        let r = run(&all, None, &settings(16, &[MeshMap::Id]));
        let (left, right) = (at(&r, MeshMap::Id, 3, 8), at(&r, MeshMap::Id, 12, 8));
        assert!(!close(left, right, 0.02), "pieces share a color: {left:?} vs {right:?}");
        assert!(close(at(&r, MeshMap::Id, 3, 8), at(&r, MeshMap::Id, 4, 3), 1e-4), "one piece, one color");
    }

    #[test]
    fn bent_normals_of_an_open_plane_point_along_the_normal() {
        let flat = run(&plane(), None, &settings(8, &[MeshMap::BentNormals]));
        let n = at(&flat, MeshMap::BentNormals, 4, 4);
        // Open flat plane: the average free direction is the normal, so (0.5, 0.5, 1) up to sampling noise.
        assert!((n[0] - 0.5).abs() < 0.08 && (n[1] - 0.5).abs() < 0.08 && n[2] > 0.9, "{n:?}");
    }

    #[test]
    fn cancel_and_progress_and_determinism() {
        let sphere = Mesh::uv_sphere(1.0, 24, 12);
        let s = settings(24, &[MeshMap::AmbientOcclusion, MeshMap::Thickness]);
        let p = Progress::new();
        p.cancel();
        assert!(matches!(bake(&sphere, None, &s, &p), Err(BakeError::Cancelled)));

        let p = Progress::new();
        let a = bake(&sphere, None, &s, &p).unwrap();
        assert!((p.fraction() - 1.0).abs() < 1e-6, "progress reaches 100%");
        let b = bake(&sphere, None, &s, &Progress::default()).unwrap();
        for m in &s.maps {
            assert_eq!(a.maps[m], b.maps[m], "{m:?} differs between runs");
        }
    }

    #[test]
    fn padding_copies_edge_values_outward_by_exactly_the_requested_width() {
        // One scalar map and one 3-channel map on a 9x9 grid with a single covered texel.
        let size = 9;
        let mut covered = vec![false; 81];
        let centre = 4 * 9 + 4;
        covered[centre] = true;
        let mut one = vec![0u16; 81];
        one[centre] = 40000;
        let mut three = vec![0u16; 81 * 3];
        three[centre * 3..centre * 3 + 3].copy_from_slice(&[100, 200, 300]);
        let mut bufs = vec![one, three];
        pad_edges(&mut bufs, &[1, 3], &covered, size, 3);
        for y in 0..9usize {
            for x in 0..9usize {
                let d = (x as i32 - 4).abs().max((y as i32 - 4).abs());
                let i = y * 9 + x;
                if d <= 3 {
                    assert_eq!(bufs[0][i], 40000, "({x},{y}) distance {d} should be padded");
                    assert_eq!(bufs[1][i * 3..i * 3 + 3], [100, 200, 300]);
                } else {
                    assert_eq!(bufs[0][i], 0, "({x},{y}) distance {d} is beyond the padding");
                }
            }
        }
    }

    #[test]
    fn padding_averages_between_islands_and_never_changes_covered_texels() {
        let size = 7;
        let mut covered = vec![false; 49];
        let mut buf = vec![0u16; 49];
        // Two covered texels in one row, 2 apart, with different values; the texel between them is empty.
        covered[3 * 7 + 1] = true;
        covered[3 * 7 + 3] = true;
        buf[3 * 7 + 1] = 1000;
        buf[3 * 7 + 3] = 3000;
        let mut bufs = vec![buf];
        pad_edges(&mut bufs, &[1], &covered, size, 1);
        assert_eq!(bufs[0][3 * 7 + 1], 1000, "covered texels are untouched");
        assert_eq!(bufs[0][3 * 7 + 3], 3000);
        assert_eq!(bufs[0][3 * 7 + 2], 2000, "the texel between is the average of both neighbours");
    }

    #[test]
    fn padding_with_zero_iterations_or_full_coverage_is_a_no_op() {
        let mut bufs = vec![vec![5u16; 16]];
        let before = bufs.clone();
        pad_edges(&mut bufs, &[1], &[true; 16], 4, 8);
        assert_eq!(bufs, before);
        pad_edges(&mut bufs, &[1], &[false; 16], 4, 0);
        assert_eq!(bufs, before);
        pad_edges(&mut bufs, &[1], &[false; 16], 4, 8); // nothing to copy from: must not loop or panic
        assert_eq!(bufs, before);
    }

    #[test]
    fn baked_maps_are_padded_beyond_the_uv_footprint() {
        // The plane only fills the left half of UV space.
        let mut half = plane();
        for v in &mut half.vertices {
            v.uv[0] *= 0.5;
        }
        let padded = run(&half, None, &BakeSettings { size: 16, maps: vec![MeshMap::Position], padding: 3, ..Default::default() });
        let unpadded = run(&half, None, &BakeSettings { size: 16, maps: vec![MeshMap::Position], padding: 0, ..Default::default() });
        // Column 7 is the last covered column; 8..=10 are padding, 11 is beyond it.
        assert_eq!(at(&unpadded, MeshMap::Position, 9, 8), [0.0; 3]);
        assert_eq!(at(&padded, MeshMap::Position, 9, 8), at(&padded, MeshMap::Position, 7, 8), "padding copies the edge");
        assert_eq!(at(&padded, MeshMap::Position, 11, 8), [0.0; 3], "no more than 3 texels");
        assert_eq!(at(&padded, MeshMap::Position, 3, 8), at(&unpadded, MeshMap::Position, 3, 8), "covered texels unchanged");
    }

    #[test]
    fn invalid_settings_and_meshes_are_rejected_not_crashed_on() {
        let p = Progress::default();
        let ok = settings(8, &[MeshMap::Position]);
        assert!(bake(&plane(), None, &BakeSettings { size: 0, ..ok.clone() }, &p).is_err());
        assert!(bake(&plane(), None, &BakeSettings { size: 99_999, ..ok.clone() }, &p).is_err());
        assert!(bake(&plane(), None, &BakeSettings { maps: vec![], ..ok.clone() }, &p).is_err());
        assert!(bake(&plane(), None, &BakeSettings { maps: vec![MeshMap::AmbientOcclusion], samples: 0, ..ok.clone() }, &p).is_err());
        assert!(bake(&plane(), None, &BakeSettings { ao_distance: f32::NAN, ..ok.clone() }, &p).is_err());
        assert!(bake(&Mesh::default(), None, &ok, &p).is_err(), "empty mesh");
        let mut bad = plane();
        bad.indices[0] = 99;
        assert!(bake(&bad, None, &ok, &p).is_err(), "out-of-range index");
        assert!(bake(&plane(), Some(&Mesh::default()), &ok, &p).is_err(), "empty high-poly");
        // Non-finite and out-of-range UVs must not panic.
        let mut weird = plane();
        weird.vertices[0].uv = [f32::NAN, 0.5];
        weird.vertices[2].uv = [40.0, -7.0];
        let _ = bake(&weird, None, &ok, &p);
    }
}
