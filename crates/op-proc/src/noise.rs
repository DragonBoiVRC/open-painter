//! Deterministic 3D noise: value, gradient (Perlin) and cellular, plus fractal sums.
//! Every function is a pure function of its inputs, so results never depend on threading.

use op_core::graph::NoiseKind;

fn hash(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x9e37_79b9;
    for v in [x as u32, y as u32, z as u32] {
        h = (h ^ v.wrapping_mul(0x85eb_ca6b)).rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn unit(h: u32) -> f32 {
    h as f32 / u32::MAX as f32
}

/// Quintic fade: zero first and second derivative at the lattice points.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn split(v: f32) -> (i32, f32) {
    let f = v.floor();
    (f as i32, v - f)
}

/// Lattice value noise, 0..1.
pub fn value3(p: [f32; 3], seed: u32) -> f32 {
    let ((ix, fx), (iy, fy), (iz, fz)) = (split(p[0]), split(p[1]), split(p[2]));
    let c = |dx, dy, dz| unit(hash(ix + dx, iy + dy, iz + dz, seed));
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), u);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), u);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), u);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

fn grad(h: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = h & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// Gradient (Perlin) noise, 0..1 (zero-mean signal remapped).
pub fn perlin3(p: [f32; 3], seed: u32) -> f32 {
    let ((ix, fx), (iy, fy), (iz, fz)) = (split(p[0]), split(p[1]), split(p[2]));
    let g = |dx: i32, dy: i32, dz: i32| grad(hash(ix + dx, iy + dy, iz + dz, seed), fx - dx as f32, fy - dy as f32, fz - dz as f32);
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));
    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    let n = lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
    (n * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// Distance to the nearest of one jittered feature point per lattice cell, clamped to 0..1.
pub fn cellular3(p: [f32; 3], seed: u32) -> f32 {
    let ((ix, _), (iy, _), (iz, _)) = (split(p[0]), split(p[1]), split(p[2]));
    let mut best = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let jitter = |k: u32| unit(hash(cx, cy, cz, seed ^ (0xa5a5_0000 + k)));
                let f = [cx as f32 + jitter(0), cy as f32 + jitter(1), cz as f32 + jitter(2)];
                let d = ((f[0] - p[0]).powi(2) + (f[1] - p[1]).powi(2) + (f[2] - p[2]).powi(2)).sqrt();
                best = best.min(d);
            }
        }
    }
    best.min(1.0)
}

/// Fractal sum of `octaves` octaves (lacunarity 2), normalised to 0..1.
pub fn fractal(kind: NoiseKind, p: [f32; 3], octaves: u32, persistence: f32, seed: u32) -> f32 {
    let (mut sum, mut total, mut amp, mut freq) = (0.0, 0.0, 1.0, 1.0);
    for o in 0..octaves {
        let q = [p[0] * freq, p[1] * freq, p[2] * freq];
        let s = seed.wrapping_add(o.wrapping_mul(1013));
        let n = match kind {
            NoiseKind::Value => value3(q, s),
            NoiseKind::Perlin => perlin3(q, s),
            NoiseKind::Cellular => cellular3(q, s),
        };
        sum += n * amp;
        total += amp;
        amp *= persistence;
        freq *= 2.0;
    }
    if total > 0.0 { (sum / total).clamp(0.0, 1.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed lattice of sample points, deliberately off the integer grid.
    fn samples() -> impl Iterator<Item = [f32; 3]> {
        (0..40).flat_map(|i| (0..40).map(move |j| [i as f32 * 0.173 + 0.07, j as f32 * 0.191 + 0.11, (i * j) as f32 * 0.013]))
    }

    fn stats(f: impl Fn([f32; 3]) -> f32) -> (f32, f32, f32, f32) {
        let v: Vec<f32> = samples().map(f).collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        (mean, var.sqrt(), v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
    }

    #[test]
    fn every_kind_stays_in_unit_range_with_sensible_statistics() {
        for (name, f) in [
            ("value", Box::new(|p| value3(p, 3)) as Box<dyn Fn([f32; 3]) -> f32>),
            ("perlin", Box::new(|p| perlin3(p, 3))),
            ("cellular", Box::new(|p| cellular3(p, 3))),
        ] {
            let (mean, sd, lo, hi) = stats(&*f);
            assert!(lo >= 0.0 && hi <= 1.0, "{name}: range {lo}..{hi}");
            assert!(sd > 0.05, "{name}: output is nearly constant (sd {sd})");
            assert!(mean > 0.15 && mean < 0.85, "{name}: mean {mean}");
        }
        let (mean, ..) = stats(|p| perlin3(p, 11));
        assert!((mean - 0.5).abs() < 0.08, "perlin is zero-mean before remapping: {mean}");
    }

    #[test]
    fn noise_is_deterministic_and_the_seed_matters() {
        for kind in [NoiseKind::Value, NoiseKind::Perlin, NoiseKind::Cellular] {
            let p = [1.37, 2.91, 0.55];
            assert_eq!(fractal(kind, p, 4, 0.5, 7), fractal(kind, p, 4, 0.5, 7));
            let differing = samples().filter(|q| fractal(kind, *q, 3, 0.5, 1) != fractal(kind, *q, 3, 0.5, 2)).count();
            assert!(differing > 1000, "{kind:?}: seeds 1 and 2 agree almost everywhere ({differing} differ)");
        }
    }

    #[test]
    fn value_and_perlin_are_continuous() {
        // A tiny step in position must only change the value a little.
        for f in [|p| value3(p, 5), |p| perlin3(p, 5)] {
            let worst = samples()
                .map(|p| (f(p) - f([p[0] + 1e-3, p[1], p[2]])).abs().max((f(p) - f([p[0], p[1] + 1e-3, p[2]])).abs()))
                .fold(0.0f32, f32::max);
            assert!(worst < 0.02, "noise jumps by {worst} over a 0.001 step");
        }
    }

    #[test]
    fn value_noise_hits_the_lattice_value_exactly() {
        // At integer coordinates the interpolation weights are zero for the far corners.
        let p = [3.0, -2.0, 5.0];
        assert!((value3(p, 9) - unit(hash(3, -2, 5, 9))).abs() < 1e-6);
        // Negative coordinates work (floor, not truncation).
        assert!(value3([-0.25, -0.5, -0.75], 9).is_finite());
    }

    #[test]
    fn perlin_is_zero_at_lattice_points() {
        // Gradient noise is exactly the middle gray (0.5) on the integer lattice.
        for p in [[0.0, 0.0, 0.0], [4.0, 7.0, -3.0], [-2.0, 1.0, 9.0]] {
            assert!((perlin3(p, 4) - 0.5).abs() < 1e-6, "{p:?} -> {}", perlin3(p, 4));
        }
    }

    #[test]
    fn cellular_is_zero_at_a_feature_point_and_grows_away_from_it() {
        // Find the feature point of cell (0,0,0) the same way the noise does.
        let jitter = |k: u32| unit(hash(0, 0, 0, 5 ^ (0xa5a5_0000 + k)));
        let feature = [jitter(0), jitter(1), jitter(2)];
        assert!(cellular3(feature, 5) < 1e-5);
        let away = [feature[0] + 0.2, feature[1], feature[2]];
        assert!(cellular3(away, 5) > 0.0);
    }

    #[test]
    fn more_octaves_add_detail_and_persistence_controls_it() {
        let rough = |octaves, persistence| {
            // Mean absolute difference between neighbouring samples: high-frequency content.
            let d: f32 = samples().map(|p| (fractal(NoiseKind::Perlin, p, octaves, persistence, 1) - fractal(NoiseKind::Perlin, [p[0] + 0.02, p[1], p[2]], octaves, persistence, 1)).abs()).sum();
            d / 1600.0
        };
        assert!(rough(6, 0.7) > rough(1, 0.7) * 1.3, "extra octaves must add fine detail");
        assert!(rough(6, 0.7) > rough(6, 0.2), "higher persistence keeps more fine detail");
    }

    #[test]
    fn fractal_with_zero_octaves_is_defined() {
        assert_eq!(fractal(NoiseKind::Value, [1.0, 2.0, 3.0], 0, 0.5, 1), 0.0);
    }

    #[test]
    fn hash_is_well_distributed() {
        // Neighbouring lattice points should not produce correlated hashes.
        let mut buckets = [0u32; 16];
        for x in 0..64 {
            for y in 0..64 {
                buckets[(hash(x, y, 0, 1) >> 28) as usize] += 1;
            }
        }
        let expected = 64 * 64 / 16;
        assert!(buckets.iter().all(|&b| (b as i32 - expected as i32).abs() < expected as i32 / 3), "{buckets:?}");
    }
}
