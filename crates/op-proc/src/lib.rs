//! Procedural mask generation: evaluate a graph, or one of the built-in generators, into a
//! grayscale image in UV space, using the baked mesh maps as inputs.
//!
//! Evaluation runs on the CPU (rayon) and is deterministic: the same inputs give the same
//! bits whatever the thread count.

mod eval;
mod noise;
mod templates;

pub use eval::{evaluate, Evaluation, ProcError, MAX_SIZE};
pub use templates::{build_graph, label, param_specs, required_maps, value, ParamSpec};

use op_core::graph::Graph;
use op_core::{BakedMap, Generator, GeneratorKind, MeshMap};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

/// Evaluate a mask generator at `size` x `size`.
pub fn generate(
    generator: &Generator,
    size: u32,
    baked: &HashMap<MeshMap, BakedMap>,
    cancel: &AtomicBool,
) -> Result<Evaluation, ProcError> {
    let graph: Graph = build_graph(generator).map_err(ProcError::Invalid)?;
    evaluate(&graph, size, baked, cancel)
}

/// 8-bit mask texels from evaluated values.
pub fn to_mask_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect()
}

/// A custom generator from a JSON file: `{"name": "...", "graph": { ... }}`.
pub fn custom_generator_from_json(text: &str) -> Result<Generator, String> {
    #[derive(serde_json_shim::Deserialize)]
    struct File {
        name: String,
        graph: Graph,
    }
    let f: File = serde_json_shim::from_str(text).map_err(|e| e.to_string())?;
    let g = Generator::new(GeneratorKind::Custom { name: f.name, graph: f.graph });
    g.validate()?;
    Ok(g)
}

mod serde_json_shim {
    pub use serde::Deserialize;
    pub use serde_json::from_str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::graph::{GraphBuilder, Op};

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    /// A baked map built texel by texel.
    fn map(size: u32, channels: u8, f: impl Fn(usize, usize) -> [f32; 3]) -> BakedMap {
        let mut m = BakedMap::filled(size, channels, [0.0; 3]);
        for y in 0..size as usize {
            for x in 0..size as usize {
                m.set_texel(y * size as usize + x, f(x, y));
            }
        }
        m
    }

    fn eval(g: &Generator, size: u32, baked: &HashMap<MeshMap, BakedMap>) -> Evaluation {
        generate(g, size, baked, &no_cancel()).unwrap()
    }

    fn total(e: &Evaluation) -> f32 {
        e.values.iter().sum()
    }

    fn no_breakup(kind: GeneratorKind) -> Generator {
        Generator::new(kind).with("breakup", 0.0)
    }

    /// Curvature: flat (0.5) everywhere except a bright ridge down the middle columns.
    fn ridge(size: u32) -> HashMap<MeshMap, BakedMap> {
        let mid = size as usize / 2;
        HashMap::from([(
            MeshMap::Curvature,
            map(size, 1, |x, _| if (x as i32 - mid as i32).abs() <= 1 { [0.95; 3] } else { [0.5; 3] }),
        )])
    }

    #[test]
    fn edge_wear_marks_the_ridge_and_leaves_flat_areas_alone() {
        let size = 64;
        let e = eval(&no_breakup(GeneratorKind::EdgeWear).with("width", 0.0), size, &ridge(size));
        let n = size as usize;
        let ridge_value = e.values[10 * n + n / 2];
        let flat_value = e.values[10 * n + 5];
        assert!(ridge_value > 0.9, "ridge {ridge_value}");
        assert!(flat_value < 0.02, "flat area {flat_value}");
        assert!(e.values.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn edge_wear_amount_grows_the_worn_area() {
        // A smooth bump profile so more amount reaches further down its flanks.
        let size = 64;
        let bump = HashMap::from([(
            MeshMap::Curvature,
            map(size, 1, |x, _| {
                let d = (x as f32 - 32.0).abs() / 32.0;
                [0.5 + 0.5 * (1.0 - d).powi(2); 3]
            }),
        )]);
        let area = |amount: f32| {
            let e = eval(&no_breakup(GeneratorKind::EdgeWear).with("amount", amount).with("width", 0.0), size, &bump);
            e.values.iter().filter(|v| **v > 0.5).count()
        };
        let (low, mid, high) = (area(0.2), area(0.5), area(0.9));
        assert!(low < mid && mid < high, "areas {low} {mid} {high}");
    }

    #[test]
    fn edge_wear_softness_widens_the_transition() {
        let size = 64;
        let spread = |width: f32| {
            let e = eval(&no_breakup(GeneratorKind::EdgeWear).with("width", width), size, &ridge(size));
            e.values.iter().filter(|v| **v > 0.05).count()
        };
        assert!(spread(1.0) > spread(0.0), "blurred edges spill over a larger area");
    }

    #[test]
    fn invert_flips_every_generator_exactly() {
        let size = 32;
        let maps = ridge(size);
        for kind in [GeneratorKind::EdgeWear, GeneratorKind::CreviceDirt, GeneratorKind::TopDust, GeneratorKind::HeightGradient, GeneratorKind::Noise] {
            let plain = eval(&Generator::new(kind.clone()), size, &maps);
            let flipped = eval(&Generator::new(kind.clone()).inverted(), size, &maps);
            for (a, b) in plain.values.iter().zip(&flipped.values) {
                assert!((a + b - 1.0).abs() < 1e-5, "{}: {a} + {b} != 1", label(&kind));
            }
        }
    }

    #[test]
    fn breakup_changes_the_mask_with_the_seed_and_stays_reproducible() {
        let size = 48;
        let maps = HashMap::from([
            (MeshMap::Curvature, map(size, 1, |_, _| [0.95; 3])), // everything is an edge
            (MeshMap::Position, map(size, 3, |x, y| [x as f32 / 48.0, y as f32 / 48.0, 0.5])),
        ]);
        let with = |seed: f32, breakup: f32| eval(&Generator::new(GeneratorKind::EdgeWear).with("seed", seed).with("breakup", breakup), size, &maps);
        let a = with(1.0, 0.8);
        assert_eq!(a, with(1.0, 0.8), "same seed, same result");
        assert_ne!(a.values, with(2.0, 0.8).values, "another seed gives another pattern");
        let solid = with(1.0, 0.0);
        assert!(solid.values.iter().all(|v| *v > 0.99), "without breakup everything is worn");
        assert!(total(&a) < total(&solid) * 0.95, "breakup removes some of the wear");
        assert!(a.values.iter().any(|v| *v > 0.9) && a.values.iter().any(|v| *v < 0.3), "pattern has both worn and intact areas");
    }

    #[test]
    fn crevice_dirt_follows_baked_occlusion_and_concave_curvature() {
        let size = 32;
        // A dark pit in AO.
        let ao = HashMap::from([(
            MeshMap::AmbientOcclusion,
            map(size, 1, |x, y| if (8..16).contains(&x) && (8..16).contains(&y) { [0.1; 3] } else { [1.0; 3] }),
        )]);
        let e = eval(&no_breakup(GeneratorKind::CreviceDirt), size, &ao);
        let n = size as usize;
        assert!(e.values[12 * n + 12] > 0.8, "inside the pit: {}", e.values[12 * n + 12]);
        assert!(e.values[28 * n + 28] < 0.02, "open area: {}", e.values[28 * n + 28]);
        // Concave curvature alone (no AO baked) also produces dirt, with a warning about AO.
        let concave = HashMap::from([(MeshMap::Curvature, map(size, 1, |x, _| if x < 8 { [0.05; 3] } else { [0.5; 3] }))]);
        let e = eval(&no_breakup(GeneratorKind::CreviceDirt), size, &concave);
        assert!(e.values[5 * n + 2] > 0.8 && e.values[5 * n + 20] < 0.02);
        assert!(e.warnings.iter().any(|w| w.contains("Ambient occlusion")), "{:?}", e.warnings);
        // Convex edges are not dirty.
        let convex = HashMap::from([(MeshMap::Curvature, map(size, 1, |_, _| [0.95; 3]))]);
        assert!(eval(&no_breakup(GeneratorKind::CreviceDirt), size, &convex).values.iter().all(|v| *v < 0.02));
    }

    #[test]
    fn crevice_dirt_amount_grows_the_dirty_area() {
        let size = 32;
        let ao = HashMap::from([(MeshMap::AmbientOcclusion, map(size, 1, |x, _| [x as f32 / 31.0; 3]))]);
        let area = |a: f32| eval(&no_breakup(GeneratorKind::CreviceDirt).with("amount", a), size, &ao).values.iter().filter(|v| **v > 0.5).count();
        assert!(area(0.2) < area(0.5) && area(0.5) < area(0.9));
    }

    #[test]
    fn top_dust_covers_up_facing_surfaces_only() {
        let size = 32;
        // World-normal y encoded as 0.5 + 0.5 * n.y: left third faces down, middle sideways, right up.
        let normals = HashMap::from([(
            MeshMap::WorldSpaceNormal,
            map(size, 3, |x, _| {
                let ny = if x < 10 { -1.0 } else if x < 22 { 0.0 } else { 1.0 };
                [0.5, 0.5 + 0.5 * ny, 0.5]
            }),
        )]);
        let e = eval(&no_breakup(GeneratorKind::TopDust), size, &normals);
        let n = size as usize;
        assert!(e.values[4 * n + 28] > 0.95, "facing up: {}", e.values[4 * n + 28]);
        assert!(e.values[4 * n + 15] < 0.2, "facing sideways: {}", e.values[4 * n + 15]);
        assert_eq!(e.values[4 * n + 3], 0.0, "facing down");
        // More amount lets steeper surfaces collect dust.
        let sloped = HashMap::from([(MeshMap::WorldSpaceNormal, map(size, 3, |x, _| [0.5, 0.5 + 0.5 * (x as f32 / 31.0), 0.5]))]);
        let area = |a: f32| eval(&no_breakup(GeneratorKind::TopDust).with("amount", a), size, &sloped).values.iter().filter(|v| **v > 0.5).count();
        assert!(area(0.2) < area(0.6) && area(0.6) < area(1.0));
    }

    #[test]
    fn height_gradient_follows_the_baked_position() {
        let size = 32;
        let pos = HashMap::from([(MeshMap::Position, map(size, 3, |_, y| [0.5, 1.0 - y as f32 / 31.0, 0.5]))]);
        let e = eval(&no_breakup(GeneratorKind::HeightGradient).with("low", 0.25).with("high", 0.75), size, &pos);
        let n = size as usize;
        let col: Vec<f32> = (0..n).map(|y| e.values[y * n + 7]).collect();
        assert!(col.windows(2).all(|w| w[1] <= w[0] + 1e-6), "taller (upper rows) is brighter: {col:?}");
        assert_eq!(col[0], 1.0, "above the end height");
        assert_eq!(col[n - 1], 0.0, "below the start height");
        // Swapping low and high reverses the gradient.
        let rev = eval(&no_breakup(GeneratorKind::HeightGradient).with("low", 0.75).with("high", 0.25), size, &pos);
        assert!(rev.values[0] < 0.01 && rev.values[(n - 1) * n] > 0.99);
    }

    #[test]
    fn noise_generator_has_the_requested_character() {
        let size = 64;
        let stats = |g: &Generator| {
            let e = eval(g, size, &HashMap::new()); // no Position bake: UV fallback
            let mean = total(&e) / e.values.len() as f32;
            let sd = (e.values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / e.values.len() as f32).sqrt();
            (mean, sd, e)
        };
        let (mean, sd, base) = stats(&Generator::new(GeneratorKind::Noise));
        assert!((mean - 0.5).abs() < 0.2 && sd > 0.05, "mean {mean} sd {sd}");
        assert!(base.warnings.iter().any(|w| w.contains("Position")));
        let (_, sd_hard, _) = stats(&Generator::new(GeneratorKind::Noise).with("contrast", 1.0));
        let (_, sd_soft, _) = stats(&Generator::new(GeneratorKind::Noise).with("contrast", 0.0));
        assert!(sd_hard > sd_soft, "contrast spreads the values: {sd_hard} vs {sd_soft}");
        let (_, _, a) = stats(&Generator::new(GeneratorKind::Noise).with("seed", 1.0));
        let (_, _, b) = stats(&Generator::new(GeneratorKind::Noise).with("seed", 2.0));
        assert_ne!(a.values, b.values);
        // A larger scale means more variation per texel.
        let detail = |scale: f32| {
            let e = eval(&Generator::new(GeneratorKind::Noise).with("scale", scale), size, &HashMap::new());
            e.values.windows(2).map(|w| (w[0] - w[1]).abs()).sum::<f32>()
        };
        assert!(detail(24.0) > detail(4.0));
    }

    #[test]
    fn mask_bytes_round_and_clamp() {
        assert_eq!(to_mask_bytes(&[0.0, 1.0, 0.5, -3.0, 9.0, 0.002]), [0, 255, 128, 0, 255, 1]);
    }

    #[test]
    fn custom_generators_load_from_json_and_run() {
        let mut b = GraphBuilder::new();
        let u = b.add(Op::Uv { axis: 0 }, &[]);
        let graph = b.finish(u);
        let json = format!(r#"{{"name":"left to right","graph":{}}}"#, serde_json::to_string(&graph).unwrap());
        let g = custom_generator_from_json(&json).unwrap();
        assert_eq!(label(&g.kind), "left to right");
        let e = eval(&g, 8, &HashMap::new());
        assert!(e.values[0] < e.values[7], "u increases to the right");
        // Hostile files fail cleanly.
        assert!(custom_generator_from_json("not json").is_err());
        assert!(custom_generator_from_json(r#"{"name":"x","graph":{"nodes":[],"output":0}}"#).is_err());
        assert!(custom_generator_from_json(r#"{"name":"x","graph":{"nodes":[{"id":0,"op":{"Invert":null},"inputs":[0]}],"output":0}}"#).is_err(), "cycle");
    }

    // ---- end to end with the real baker --------------------------------------------------

    use op_bake::{bake, BakeSettings, Progress};
    use op_mesh::{Mesh, Vertex};

    /// A cube whose six faces are subdivided `n` x `n` and laid out in a 3 x 2 UV atlas, so that
    /// every face's border in UV space is a real cube edge.
    fn cube(n: usize) -> Mesh {
        let faces: [(glam_like::V3, glam_like::V3, glam_like::V3); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (f, (nrm, u, v)) in faces.iter().enumerate() {
            let (tx, ty) = ((f % 3) as f32 / 3.0, (f / 3) as f32 / 2.0);
            let base = vertices.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let (a, b) = (i as f32 / n as f32, j as f32 / n as f32);
                    let (s, t) = (a * 2.0 - 1.0, b * 2.0 - 1.0);
                    let p = [nrm[0] + u[0] * s + v[0] * t, nrm[1] + u[1] * s + v[1] * t, nrm[2] + u[2] * s + v[2] * t];
                    vertices.push(Vertex { position: p, normal: *nrm, uv: [tx + a / 3.0, ty + b / 2.0] });
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let k = base + (j * (n + 1) + i) as u32;
                    let (b, c, d) = (k + 1, k + n as u32 + 2, k + n as u32 + 1);
                    indices.extend([k, b, c, k, c, d]);
                }
            }
        }
        // Orient every triangle outward.
        for tri in indices.chunks_exact_mut(3) {
            let p = |i: u32| glam_like::from(vertices[i as usize].position);
            let (a, b, c) = (p(tri[0]), p(tri[1]), p(tri[2]));
            let n = glam_like::cross(glam_like::sub(b, a), glam_like::sub(c, a));
            let centre = glam_like::scale(glam_like::add(glam_like::add(a, b), c), 1.0 / 3.0);
            if glam_like::dot(n, centre) < 0.0 {
                tri.swap(1, 2);
            }
        }
        Mesh { vertices, indices }
    }

    /// Tiny vector helpers so the test does not need glam as a dependency of this crate.
    mod glam_like {
        pub type V3 = [f32; 3];
        pub fn from(v: [f32; 3]) -> V3 {
            v
        }
        pub fn sub(a: V3, b: V3) -> V3 {
            [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        pub fn add(a: V3, b: V3) -> V3 {
            [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
        }
        pub fn scale(a: V3, s: f32) -> V3 {
            [a[0] * s, a[1] * s, a[2] * s]
        }
        pub fn dot(a: V3, b: V3) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }
        pub fn cross(a: V3, b: V3) -> V3 {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        }
    }

    fn baked_cube(size: u32) -> HashMap<MeshMap, BakedMap> {
        let settings = BakeSettings {
            size,
            maps: vec![MeshMap::Curvature, MeshMap::AmbientOcclusion, MeshMap::WorldSpaceNormal, MeshMap::Position],
            samples: 16,
            ..Default::default()
        };
        bake(&cube(8), None, &settings, &Progress::default()).unwrap().maps
    }

    /// Mean of `values` over texels inside tile `(tile_x, tile_y)` whose distance to the tile border
    /// is in `[lo, hi)` texels.
    fn tile_ring_mean(e: &Evaluation, tile: (usize, usize), lo: usize, hi: usize) -> f32 {
        let (tw, th) = (e.size as usize / 3, e.size as usize / 2);
        let (mut sum, mut count) = (0.0, 0);
        for ty in 0..th {
            for tx in 0..tw {
                let d = tx.min(tw - 1 - tx).min(ty).min(th - 1 - ty);
                if d >= lo && d < hi {
                    sum += e.values[(tile.1 * th + ty) * e.size as usize + tile.0 * tw + tx];
                    count += 1;
                }
            }
        }
        sum / count as f32
    }

    #[test]
    fn real_bake_edge_wear_hugs_the_cubes_edges() {
        let size = 96;
        let maps = baked_cube(size);
        let e = eval(&Generator::new(GeneratorKind::EdgeWear).with("breakup", 0.0), size, &maps);
        for tile in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)] {
            let edge = tile_ring_mean(&e, tile, 0, 3);
            let middle = tile_ring_mean(&e, tile, 10, 100);
            assert!(edge > 0.3, "tile {tile:?}: edge wear {edge}");
            assert!(middle < 0.05, "tile {tile:?}: centre of the face {middle}");
            assert!(edge > middle * 8.0, "tile {tile:?}: edges {edge} vs centre {middle}");
        }
    }

    #[test]
    fn real_bake_top_dust_lands_on_the_top_face_only() {
        let size = 96;
        let maps = baked_cube(size);
        let e = eval(&no_breakup(GeneratorKind::TopDust), size, &maps);
        // Face order: +x, -x, +y, -y, +z, -z in tiles (0,0) (1,0) (2,0) (0,1) (1,1) (2,1): +y is tile (2, 0).
        let top = tile_ring_mean(&e, (2, 0), 6, 100);
        assert!(top > 0.95, "top face {top}");
        for tile in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)] {
            assert!(tile_ring_mean(&e, tile, 6, 100) < 0.05, "tile {tile:?} should stay clean");
        }
        assert!(e.warnings.is_empty(), "{:?}", e.warnings);
    }

    #[test]
    fn real_bake_crevice_dirt_is_empty_on_a_convex_cube() {
        let size = 96;
        let maps = baked_cube(size);
        let e = eval(&no_breakup(GeneratorKind::CreviceDirt), size, &maps);
        assert!(total(&e) / e.values.len() as f32 > -1.0);
        let mean = total(&e) / e.values.len() as f32;
        assert!(mean < 0.03, "a convex cube has no crevices, mean mask {mean}");
    }
}
