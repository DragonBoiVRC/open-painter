//! Graph evaluation on the CPU.
//!
//! Every node produces one `size * size` grayscale image (row-major, v downward, matching the
//! texture layout). Results of each node are clamped to 0..1, except constants. Nodes are
//! evaluated in dependency order and an image is freed as soon as its last consumer has run.

use crate::noise;
use op_core::graph::{BlendOp, Graph, Op};
use op_core::{BakedMap, MeshMap};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

pub const MAX_SIZE: u32 = 8192;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProcError {
    #[error("{0}")]
    Invalid(String),
    #[error("evaluation was cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    pub size: u32,
    /// `size * size` values in 0..=1.
    pub values: Vec<f32>,
    /// Inputs that were not available and were replaced by neutral values.
    pub warnings: Vec<String>,
}

type Image = Vec<f32>;

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One channel of a baked map at `size` x `size`, resampled bilinearly if it was baked at another size.
fn fetch(map: &BakedMap, size: u32, comp: usize) -> Image {
    let comp = comp.min(map.channels as usize - 1);
    let ch = map.channels as usize;
    let at = |x: usize, y: usize| map.data[(y * map.size as usize + x) * ch + comp] as f32 / 65535.0;
    let n = size as usize;
    if map.size == size {
        return (0..n * n).into_par_iter().map(|i| at(i % n, i / n)).collect();
    }
    let ms = map.size as usize;
    let scale = ms as f32 / size as f32;
    (0..n * n)
        .into_par_iter()
        .map(|i| {
            // Align texel centres, then interpolate between the four surrounding source texels.
            let sx = ((i % n) as f32 + 0.5) * scale - 0.5;
            let sy = ((i / n) as f32 + 0.5) * scale - 0.5;
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let c = |v: f32| (v.max(0.0) as usize).min(ms - 1);
            let (xa, xb, ya, yb) = (c(x0), c(x0 + 1.0), c(y0), c(y0 + 1.0));
            let top = at(xa, ya) * (1.0 - fx) + at(xb, ya) * fx;
            let bottom = at(xa, yb) * (1.0 - fx) + at(xb, yb) * fx;
            top * (1.0 - fy) + bottom * fy
        })
        .collect()
}

fn blend(op: BlendOp, a: f32, b: f32) -> f32 {
    match op {
        BlendOp::Add => a + b,
        BlendOp::Multiply => a * b,
        BlendOp::Subtract => a - b,
        BlendOp::Min => a.min(b),
        BlendOp::Max => a.max(b),
        BlendOp::Screen => 1.0 - (1.0 - a) * (1.0 - b),
        BlendOp::Overlay => {
            if a < 0.5 { 2.0 * a * b } else { 1.0 - 2.0 * (1.0 - a) * (1.0 - b) }
        }
        BlendOp::Difference => (a - b).abs(),
    }
}

/// Sliding-window box blur of every row, clamping at the edges.
fn box_rows(img: &mut [f32], n: usize, w: usize) {
    let norm = 1.0 / (2 * w + 1) as f32;
    img.par_chunks_mut(n).for_each(|row| {
        let src = row.to_vec();
        let at = |k: i64| src[k.clamp(0, n as i64 - 1) as usize];
        let mut sum: f32 = (-(w as i64)..=w as i64).map(at).sum();
        for x in 0..n as i64 {
            row[x as usize] = sum * norm;
            sum += at(x + 1 + w as i64) - at(x - w as i64);
        }
    });
}

fn transpose(img: &[f32], n: usize) -> Image {
    let mut out = vec![0.0; img.len()];
    out.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = img[x * n + y];
        }
    });
    out
}

/// Approximate Gaussian blur: three box blurs per axis. `radius` is a fraction of the texture size.
fn blur(img: &[f32], n: usize, radius: f32) -> Image {
    let r_px = radius * n as f32;
    if r_px < 0.5 {
        return img.to_vec();
    }
    let w = ((r_px / 1.5).round() as usize).max(1);
    let mut cur = img.to_vec();
    for _ in 0..3 {
        box_rows(&mut cur, n, w);
        let mut t = transpose(&cur, n);
        box_rows(&mut t, n, w);
        cur = transpose(&t, n);
    }
    cur
}

struct Eval<'a> {
    size: u32,
    baked: &'a HashMap<MeshMap, BakedMap>,
    warnings: Vec<String>,
    /// Object-space positions (x, y, z images), fetched on first use.
    positions: Option<Option<[Image; 3]>>,
}

impl Eval<'_> {
    fn warn(&mut self, msg: String) {
        if !self.warnings.contains(&msg) {
            self.warnings.push(msg);
        }
    }

    fn baked(&mut self, map: MeshMap, comp: u8) -> Image {
        let n = self.size as usize;
        match self.baked.get(&map) {
            Some(b) => fetch(b, self.size, comp as usize),
            None => {
                self.warn(format!("{} has not been baked: a neutral value is used.", map.label()));
                vec![map.default_value()[(comp as usize).min(2)]; n * n]
            }
        }
    }

    fn positions(&mut self) -> Option<&[Image; 3]> {
        if self.positions.is_none() {
            let found = self.baked.get(&MeshMap::Position).map(|p| [0, 1, 2].map(|c| fetch(p, self.size, c)));
            if found.is_none() {
                self.warn("Position has not been baked: noise falls back to UV space.".into());
            }
            self.positions = Some(found);
        }
        self.positions.as_ref().and_then(|p| p.as_ref())
    }

    fn noise(&mut self, op: &Op) -> Image {
        let Op::Noise { kind, scale, octaves, persistence, seed, uv_space } = *op else { unreachable!() };
        let n = self.size as usize;
        let inv = 1.0 / n as f32;
        let positions = if uv_space { None } else { self.positions() };
        match positions {
            Some([px, py, pz]) => (0..n * n)
                .into_par_iter()
                .map(|i| noise::fractal(kind, [px[i] * scale, py[i] * scale, pz[i] * scale], octaves, persistence, seed))
                .collect(),
            None => (0..n * n)
                .into_par_iter()
                .map(|i| {
                    let p = [((i % n) as f32 + 0.5) * inv * scale, ((i / n) as f32 + 0.5) * inv * scale, 0.0];
                    noise::fractal(kind, p, octaves, persistence, seed)
                })
                .collect(),
        }
    }
}

fn map1(a: &[f32], f: impl Fn(f32) -> f32 + Sync) -> Image {
    a.par_iter().map(|v| f(*v).clamp(0.0, 1.0)).collect()
}

fn apply(ev: &mut Eval, op: &Op, inputs: &[&[f32]]) -> Image {
    let n = ev.size as usize;
    match op {
        Op::Const { value } => vec![*value; n * n],
        Op::Baked { map, comp } => ev.baked(*map, *comp),
        Op::Noise { .. } => ev.noise(op),
        Op::Uv { axis } => {
            let axis = *axis;
            (0..n * n)
                .into_par_iter()
                .map(|i| {
                    let k = if axis == 0 { i % n } else { i / n };
                    (k as f32 + 0.5) / n as f32
                })
                .collect()
        }
        Op::Invert => map1(inputs[0], |v| 1.0 - v),
        Op::Levels { in_lo, in_hi, gamma, out_lo, out_hi } => {
            // A zero-width input range would divide by zero; treat it as a tiny one.
            let d = in_hi - in_lo;
            let span = if d.abs() < 1e-6 { 1e-6 } else { d };
            let inv_gamma = 1.0 / gamma;
            map1(inputs[0], |v| {
                let t = ((v - in_lo) / span).clamp(0.0, 1.0).powf(inv_gamma);
                out_lo + t * (out_hi - out_lo)
            })
        }
        Op::Contrast { amount } => {
            let k = 2f32.powf(amount * 4.0);
            map1(inputs[0], |v| 0.5 + (v - 0.5) * k)
        }
        Op::Threshold { level, softness } => {
            let (level, softness) = (*level, *softness);
            map1(inputs[0], |v| {
                if softness <= 1e-6 {
                    if v >= level { 1.0 } else { 0.0 }
                } else {
                    smoothstep01((v - (level - softness * 0.5)) / softness)
                }
            })
        }
        Op::Blur { radius } => blur(inputs[0], n, *radius).into_iter().map(|v| v.clamp(0.0, 1.0)).collect(),
        Op::Blend { op, opacity } => inputs[0]
            .par_iter()
            .zip(inputs[1].par_iter())
            .map(|(a, b)| (a + (blend(*op, *a, *b) - a) * opacity).clamp(0.0, 1.0))
            .collect(),
        Op::Lerp => (0..n * n).into_par_iter().map(|i| (inputs[0][i] + (inputs[1][i] - inputs[0][i]) * inputs[2][i]).clamp(0.0, 1.0)).collect(),
    }
}

/// Evaluate `graph` at `size` x `size`.
pub fn evaluate(
    graph: &Graph,
    size: u32,
    baked: &HashMap<MeshMap, BakedMap>,
    cancel: &AtomicBool,
) -> Result<Evaluation, ProcError> {
    if size == 0 || size > MAX_SIZE {
        return Err(ProcError::Invalid(format!("size must be between 1 and {MAX_SIZE}")));
    }
    graph.validate().map_err(ProcError::Invalid)?;
    let order = graph.order().map_err(ProcError::Invalid)?;
    let index_of = |id| graph.nodes.iter().position(|n| n.id == id).expect("validated");

    // How many times each node's image is still needed.
    let mut uses = vec![0usize; graph.nodes.len()];
    for &i in &order {
        for inp in &graph.nodes[i].inputs {
            uses[index_of(*inp)] += 1;
        }
    }
    let out_idx = index_of(graph.output);
    uses[out_idx] += 1;

    let mut ev = Eval { size, baked, warnings: Vec::new(), positions: None };
    let mut images: Vec<Option<Image>> = vec![None; graph.nodes.len()];
    for &i in &order {
        if cancel.load(Ordering::Relaxed) {
            return Err(ProcError::Cancelled);
        }
        let node = &graph.nodes[i];
        let idxs: Vec<usize> = node.inputs.iter().map(|id| index_of(*id)).collect();
        let result = {
            let ins: Vec<&[f32]> = idxs.iter().map(|&j| images[j].as_deref().expect("inputs are evaluated first")).collect();
            apply(&mut ev, &node.op, &ins)
        };
        for j in idxs {
            uses[j] -= 1;
            if uses[j] == 0 {
                images[j] = None; // last consumer ran: free the memory
            }
        }
        images[i] = Some(result);
    }
    let values = images[out_idx].take().expect("output evaluated").into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    Ok(Evaluation { size, values, warnings: ev.warnings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::graph::{GraphBuilder, NoiseKind};

    pub fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    pub fn run(g: &Graph, size: u32) -> Evaluation {
        evaluate(g, size, &HashMap::new(), &no_cancel()).unwrap()
    }

    fn unary(op: Op, source: Op) -> Graph {
        let mut b = GraphBuilder::new();
        let s = b.add(source, &[]);
        let o = b.add(op, &[s]);
        b.finish(o)
    }

    #[test]
    fn uv_sources_follow_texel_centres() {
        let size = 8;
        let u = run(&unary(Op::Invert, Op::Uv { axis: 0 }), size);
        let v = run(&unary(Op::Blur { radius: 0.0 }, Op::Uv { axis: 1 }), size);
        for y in 0..8usize {
            for x in 0..8usize {
                assert!((u.values[y * 8 + x] - (1.0 - (x as f32 + 0.5) / 8.0)).abs() < 1e-6);
                assert!((v.values[y * 8 + x] - (y as f32 + 0.5) / 8.0).abs() < 1e-6, "v grows downward");
            }
        }
    }

    #[test]
    fn levels_remap_clamp_and_apply_gamma() {
        let at = |lo, hi, gamma, olo, ohi, x: f32| {
            let e = run(&unary(Op::Levels { in_lo: lo, in_hi: hi, gamma, out_lo: olo, out_hi: ohi }, Op::Const { value: x }), 2);
            e.values[0]
        };
        assert!((at(0.2, 0.6, 1.0, 0.0, 1.0, 0.4) - 0.5).abs() < 1e-6, "midpoint");
        assert_eq!(at(0.2, 0.6, 1.0, 0.0, 1.0, 0.1), 0.0, "below the input range clamps");
        assert_eq!(at(0.2, 0.6, 1.0, 0.0, 1.0, 0.9), 1.0, "above clamps");
        assert!((at(0.0, 1.0, 2.0, 0.0, 1.0, 0.25) - 0.5).abs() < 1e-6, "gamma 2 lifts 0.25 to 0.5");
        assert!((at(0.0, 1.0, 1.0, 0.2, 0.8, 0.5) - 0.5).abs() < 1e-6, "output range");
        assert!((at(0.0, 1.0, 1.0, 1.0, 0.0, 0.25) - 0.75).abs() < 1e-6, "reversed output range inverts");
        // A degenerate input range must not produce NaN.
        let d = at(0.5, 0.5, 1.0, 0.0, 1.0, 0.7);
        assert!(d.is_finite() && (0.0..=1.0).contains(&d));
    }

    #[test]
    fn threshold_is_hard_at_zero_softness_and_smooth_otherwise() {
        let at = |level, softness, x: f32| run(&unary(Op::Threshold { level, softness }, Op::Const { value: x }), 2).values[0];
        assert_eq!((at(0.5, 0.0, 0.49), at(0.5, 0.0, 0.5), at(0.5, 0.0, 0.9)), (0.0, 1.0, 1.0));
        assert_eq!(at(0.5, 0.2, 0.3), 0.0);
        assert_eq!(at(0.5, 0.2, 0.7), 1.0);
        assert!((at(0.5, 0.2, 0.5) - 0.5).abs() < 1e-6, "centre of the ramp");
        assert!(at(0.5, 0.2, 0.55) > at(0.5, 0.2, 0.45), "monotonic");
    }

    #[test]
    fn contrast_pivots_about_one_half() {
        let at = |amount, x: f32| run(&unary(Op::Contrast { amount }, Op::Const { value: x }), 2).values[0];
        assert!((at(0.0, 0.3) - 0.3).abs() < 1e-6);
        assert!((at(0.5, 0.5) - 0.5).abs() < 1e-6, "0.5 is the pivot");
        assert!(at(0.5, 0.6) > 0.6 && at(0.5, 0.4) < 0.4, "positive contrast spreads values");
        assert!(at(-0.5, 0.6) < 0.6 && at(-0.5, 0.6) > 0.5, "negative contrast compresses them");
    }

    #[test]
    fn blends_and_lerp_compute_their_formulas() {
        let two = |op: BlendOp, opacity: f32, a: f32, b: f32| {
            let mut g = GraphBuilder::new();
            let x = g.add(Op::Const { value: a }, &[]);
            let y = g.add(Op::Const { value: b }, &[]);
            let o = g.add(Op::Blend { op, opacity }, &[x, y]);
            run(&g.finish(o), 2).values[0]
        };
        assert!((two(BlendOp::Add, 1.0, 0.3, 0.4) - 0.7).abs() < 1e-6);
        assert!((two(BlendOp::Multiply, 1.0, 0.5, 0.6) - 0.3).abs() < 1e-6);
        assert!((two(BlendOp::Subtract, 1.0, 0.5, 0.2) - 0.3).abs() < 1e-6);
        assert_eq!(two(BlendOp::Subtract, 1.0, 0.2, 0.5), 0.0, "clamped at zero");
        assert_eq!((two(BlendOp::Min, 1.0, 0.3, 0.8), two(BlendOp::Max, 1.0, 0.3, 0.8)), (0.3, 0.8));
        assert!((two(BlendOp::Screen, 1.0, 0.5, 0.5) - 0.75).abs() < 1e-6);
        assert!((two(BlendOp::Overlay, 1.0, 0.25, 0.5) - 0.25).abs() < 1e-6, "dark base multiplies: 2*0.25*0.5");
        assert!((two(BlendOp::Overlay, 1.0, 0.75, 0.5) - 0.75).abs() < 1e-6, "light base screens: 1-2*0.25*0.5");
        assert!((two(BlendOp::Difference, 1.0, 0.2, 0.9) - 0.7).abs() < 1e-6);
        assert!((two(BlendOp::Multiply, 0.5, 0.5, 0.6) - 0.4).abs() < 1e-6, "opacity 0.5 blends halfway: 0.5 + (0.3 - 0.5) * 0.5");
        assert!((two(BlendOp::Multiply, 0.0, 0.5, 0.6) - 0.5).abs() < 1e-6, "opacity 0 leaves a untouched");

        let mut g = GraphBuilder::new();
        let a = g.add(Op::Const { value: 0.2 }, &[]);
        let b = g.add(Op::Const { value: 0.8 }, &[]);
        let t = g.add(Op::Uv { axis: 0 }, &[]);
        let o = g.add(Op::Lerp, &[a, b, t]);
        let e = run(&g.finish(o), 4);
        for x in 0..4 {
            let tt = (x as f32 + 0.5) / 4.0;
            assert!((e.values[x] - (0.2 + 0.6 * tt)).abs() < 1e-6);
        }
    }

    /// An image with one bright texel in the middle, for blur tests.
    fn impulse_map(size: u32) -> HashMap<MeshMap, BakedMap> {
        let mut m = BakedMap::filled(size, 1, [0.0; 3]);
        let c = (size / 2) as usize * size as usize + (size / 2) as usize;
        m.data[c] = 65535;
        HashMap::from([(MeshMap::AmbientOcclusion, m)])
    }

    fn baked_graph(map: MeshMap, extra: Option<Op>) -> Graph {
        let mut b = GraphBuilder::new();
        let s = b.add(Op::Baked { map, comp: 0 }, &[]);
        let o = match extra {
            Some(op) => b.add(op, &[s]),
            None => s,
        };
        b.finish(o)
    }

    #[test]
    fn blur_spreads_energy_symmetrically_and_conserves_the_total_away_from_edges() {
        let size = 33;
        let maps = impulse_map(size);
        let sum = |e: &Evaluation| e.values.iter().sum::<f32>();
        let flat = evaluate(&baked_graph(MeshMap::AmbientOcclusion, None), size, &maps, &no_cancel()).unwrap();
        let soft = evaluate(&baked_graph(MeshMap::AmbientOcclusion, Some(Op::Blur { radius: 0.1 })), size, &maps, &no_cancel()).unwrap();
        let wide = evaluate(&baked_graph(MeshMap::AmbientOcclusion, Some(Op::Blur { radius: 0.2 })), size, &maps, &no_cancel()).unwrap();
        let n = size as usize;
        let peak = |e: &Evaluation| e.values.iter().cloned().fold(0.0f32, f32::max);
        assert!((sum(&flat) - 1.0).abs() < 1e-3);
        assert!((sum(&soft) - 1.0).abs() < 0.02, "energy is conserved when the blur stays inside the image: {}", sum(&soft));
        assert!(peak(&soft) < peak(&flat) && peak(&wide) < peak(&soft), "larger radius, lower peak");
        let c = n / 2;
        let at = |e: &Evaluation, dx: i32, dy: i32| e.values[((c as i32 + dy) as usize) * n + (c as i32 + dx) as usize];
        for (dx, dy) in [(3, 0), (0, 3), (2, 2)] {
            assert!((at(&soft, dx, dy) - at(&soft, -dx, -dy)).abs() < 1e-6, "symmetric about the impulse");
        }
        assert!((at(&soft, 3, 0) - at(&soft, 0, 3)).abs() < 1e-6, "same in x and y");
    }

    #[test]
    fn blur_of_a_constant_stays_constant_and_zero_radius_is_the_identity() {
        let c = run(&unary(Op::Blur { radius: 0.2 }, Op::Const { value: 0.37 }), 16);
        assert!(c.values.iter().all(|v| (v - 0.37).abs() < 1e-5), "edges are clamped, not darkened");
        let maps = impulse_map(9);
        let id = evaluate(&baked_graph(MeshMap::AmbientOcclusion, Some(Op::Blur { radius: 0.0 })), 9, &maps, &no_cancel()).unwrap();
        let raw = evaluate(&baked_graph(MeshMap::AmbientOcclusion, None), 9, &maps, &no_cancel()).unwrap();
        assert_eq!(id.values, raw.values);
    }

    #[test]
    fn missing_baked_maps_use_a_neutral_value_and_warn() {
        let e = evaluate(&baked_graph(MeshMap::Curvature, None), 4, &HashMap::new(), &no_cancel()).unwrap();
        assert!(e.values.iter().all(|v| (*v - 0.5).abs() < 1e-6), "curvature defaults to flat");
        assert_eq!(e.warnings.len(), 1);
        assert!(e.warnings[0].contains("Curvature"), "{:?}", e.warnings);
        // The same missing map used twice warns once.
        let mut b = GraphBuilder::new();
        let a = b.add(Op::Baked { map: MeshMap::Curvature, comp: 0 }, &[]);
        let c = b.add(Op::Baked { map: MeshMap::Curvature, comp: 0 }, &[]);
        let o = b.add(Op::Blend { op: BlendOp::Add, opacity: 1.0 }, &[a, c]);
        assert_eq!(evaluate(&b.finish(o), 4, &HashMap::new(), &no_cancel()).unwrap().warnings.len(), 1);
    }

    #[test]
    fn vector_maps_expose_each_component() {
        let mut m = BakedMap::filled(4, 3, [0.0; 3]);
        for i in 0..16 {
            m.set_texel(i, [0.25, 0.5, 0.75]);
        }
        let maps = HashMap::from([(MeshMap::WorldSpaceNormal, m)]);
        for (comp, want) in [(0u8, 0.25), (1, 0.5), (2, 0.75)] {
            let mut b = GraphBuilder::new();
            let o = b.add(Op::Baked { map: MeshMap::WorldSpaceNormal, comp }, &[]);
            let e = evaluate(&b.finish(o), 4, &maps, &no_cancel()).unwrap();
            assert!((e.values[5] - want).abs() < 1e-4, "component {comp}");
        }
    }

    #[test]
    fn baked_maps_at_another_resolution_are_resampled() {
        // A horizontal ramp baked at 4x4, evaluated at 8x8 and at 2x2.
        let mut m = BakedMap::filled(4, 1, [0.0; 3]);
        for y in 0..4 {
            for x in 0..4 {
                m.data[y * 4 + x] = BakedMap::quantize(x as f32 / 3.0);
            }
        }
        let maps = HashMap::from([(MeshMap::HeightBase, m)]);
        let g = baked_graph(MeshMap::HeightBase, None);
        let up = evaluate(&g, 8, &maps, &no_cancel()).unwrap();
        let row = &up.values[3 * 8..4 * 8];
        assert!(row.windows(2).all(|w| w[1] >= w[0] - 1e-6), "stays monotonic: {row:?}");
        assert!(row[0] < 0.1 && row[7] > 0.9, "keeps the range");
        let down = evaluate(&g, 2, &maps, &no_cancel()).unwrap();
        assert!(down.values[0] < down.values[1], "downsampled ramp still rises");
        // A constant map is unchanged by resampling.
        let flat = HashMap::from([(MeshMap::HeightBase, BakedMap::filled(4, 1, [0.6; 3]))]);
        let e = evaluate(&g, 7, &flat, &no_cancel()).unwrap();
        assert!(e.values.iter().all(|v| (v - 0.6).abs() < 1e-3));
    }

    #[test]
    fn object_space_noise_depends_only_on_the_baked_position() {
        // Two texels with the same object-space position must get the same noise value even
        // though they sit at different UV coordinates (as on either side of a UV seam).
        let size = 8u32;
        let mut pos = BakedMap::filled(size, 3, [0.0; 3]);
        for i in 0..64usize {
            let x = (i % 8) as f32 / 8.0;
            pos.set_texel(i, [x, 0.3, 0.6]);
        }
        // Make texel (7, 3) coincide with texel (0, 5).
        let (a, b) = (3 * 8 + 7, 5 * 8);
        let p = pos.texel(b);
        pos.set_texel(a, p);
        let maps = HashMap::from([(MeshMap::Position, pos)]);
        let mut g = GraphBuilder::new();
        let o = g.add(Op::Noise { kind: NoiseKind::Perlin, scale: 5.0, octaves: 3, persistence: 0.5, seed: 4, uv_space: false }, &[]);
        let e = evaluate(&g.finish(o), size, &maps, &no_cancel()).unwrap();
        assert_eq!(e.values[a], e.values[b], "same position, same noise");
        assert!(e.warnings.is_empty());
        // In UV space the same two texels differ.
        let mut g = GraphBuilder::new();
        let o = g.add(Op::Noise { kind: NoiseKind::Perlin, scale: 5.0, octaves: 3, persistence: 0.5, seed: 4, uv_space: true }, &[]);
        let uv = evaluate(&g.finish(o), size, &maps, &no_cancel()).unwrap();
        assert_ne!(uv.values[a], uv.values[b]);
    }

    #[test]
    fn noise_without_a_position_bake_falls_back_to_uv_with_a_warning() {
        let mut g = GraphBuilder::new();
        let o = g.add(Op::Noise { kind: NoiseKind::Value, scale: 4.0, octaves: 2, persistence: 0.5, seed: 1, uv_space: false }, &[]);
        let e = evaluate(&g.finish(o), 8, &HashMap::new(), &no_cancel()).unwrap();
        assert!(e.warnings.iter().any(|w| w.contains("Position")), "{:?}", e.warnings);
        assert!(e.values.iter().any(|v| (v - e.values[0]).abs() > 1e-3), "still produces a pattern");
    }

    #[test]
    fn results_are_deterministic_and_independent_of_thread_count() {
        let mut g = GraphBuilder::new();
        let n = g.add(Op::Noise { kind: NoiseKind::Cellular, scale: 6.0, octaves: 3, persistence: 0.5, seed: 8, uv_space: true }, &[]);
        let b = g.add(Op::Blur { radius: 0.05 }, &[n]);
        let o = g.add(Op::Levels { in_lo: 0.2, in_hi: 0.7, gamma: 1.3, out_lo: 0.0, out_hi: 1.0 }, &[b]);
        let graph = g.finish(o);
        let many = run(&graph, 48);
        assert_eq!(many, run(&graph, 48));
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| run(&graph, 48));
        assert_eq!(many.values, one.values, "serial and parallel agree bit for bit");
    }

    #[test]
    fn cancellation_and_invalid_input_are_errors_not_panics() {
        let g = unary(Op::Invert, Op::Const { value: 0.5 });
        let flag = AtomicBool::new(true);
        assert_eq!(evaluate(&g, 8, &HashMap::new(), &flag), Err(ProcError::Cancelled));
        assert!(matches!(evaluate(&g, 0, &HashMap::new(), &no_cancel()), Err(ProcError::Invalid(_))));
        assert!(matches!(evaluate(&g, MAX_SIZE + 1, &HashMap::new(), &no_cancel()), Err(ProcError::Invalid(_))));
        let cyc = Graph {
            nodes: vec![op_core::graph::GraphNode { id: 0, op: Op::Invert, inputs: vec![0] }],
            output: 0,
        };
        assert!(matches!(evaluate(&cyc, 8, &HashMap::new(), &no_cancel()), Err(ProcError::Invalid(_))));
    }

    #[test]
    fn a_maximum_length_chain_evaluates_and_frees_as_it_goes() {
        let mut b = GraphBuilder::new();
        let mut prev = b.add(Op::Const { value: 0.25 }, &[]);
        for _ in 0..op_core::graph::MAX_GRAPH_NODES - 1 {
            prev = b.add(Op::Invert, &[prev]);
        }
        // 255 inversions of 0.25: odd count -> 0.75
        let e = run(&b.finish(prev), 16);
        assert!(e.values.iter().all(|v| (v - 0.75).abs() < 1e-5));
    }

    #[test]
    fn a_shared_input_feeds_both_branches() {
        let mut b = GraphBuilder::new();
        let x = b.add(Op::Uv { axis: 0 }, &[]);
        let inv = b.add(Op::Invert, &[x]);
        let o = b.add(Op::Blend { op: BlendOp::Add, opacity: 1.0 }, &[x, inv]);
        let e = run(&b.finish(o), 8);
        assert!(e.values.iter().all(|v| (v - 1.0).abs() < 1e-6), "x + (1 - x) = 1");
    }
}
