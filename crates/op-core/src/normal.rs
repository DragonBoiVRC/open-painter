//! Normal maps from painted height.
//!
//! The composited height channel (0..1, 0.5 flat) becomes a tangent-space normal map in the
//! "OpenGL" convention: +X to the right of the image, +Y towards the top of the image, +Z out of
//! the surface. Detail from a baked normal map (from a high-poly mesh) is combined on top by
//! adding slopes, so a dent painted on a baked surface keeps the baked detail.
//!
//! The GPU shader (`normal.wgsl` in op-app) implements the same math; a test compares them.

/// Height at the texel after clamping coordinates to the image.
fn at(h: &[f32], size: usize, x: i64, y: i64) -> f32 {
    let cx = x.clamp(0, size as i64 - 1) as usize;
    let cy = y.clamp(0, size as i64 - 1) as usize;
    h[cy * size + cx]
}

/// Slope-space scale: how much one unit of height gradient (per UV unit) bends the normal at
/// `strength` 1.
pub const SLOPE_PER_UV: f32 = 0.01;

/// The normal for one texel, from a central-difference gradient of `height` (`size` x `size`,
/// row 0 at the top) and the optional baked normal at that texel.
pub fn normal_at(height: &[f32], size: u32, x: u32, y: u32, strength: f32, base: Option<[f32; 3]>) -> [f32; 3] {
    let n = size as usize;
    let (xi, yi) = (x as i64, y as i64);
    // Per-texel differences, scaled to per-UV so the look does not depend on the resolution.
    let dx = (at(height, n, xi + 1, yi) - at(height, n, xi - 1, yi)) * 0.5 * size as f32;
    let drow = (at(height, n, xi, yi + 1) - at(height, n, xi, yi - 1)) * 0.5 * size as f32;
    // Image-up is decreasing row, so the slope along +Y is minus the row gradient.
    let sx = -dx * strength * SLOPE_PER_UV;
    let sy = drow * strength * SLOPE_PER_UV;
    let (bx, by, bz) = match base {
        Some(b) => (b[0], b[1], b[2].max(1e-3)),
        None => (0.0, 0.0, 1.0),
    };
    let v = [bx + sx, by + sy, bz];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

/// The whole normal map. `base` holds a baked tangent-space normal per texel (unit vectors, not
/// packed) and must be `size * size` long if given.
pub fn height_to_normal(height: &[f32], size: u32, strength: f32, base: Option<&[[f32; 3]]>) -> Vec<[f32; 3]> {
    assert_eq!(height.len(), (size as usize).pow(2), "height map must be size x size");
    if let Some(b) = base {
        assert_eq!(b.len(), height.len(), "baked normal must be size x size");
    }
    let mut out = Vec::with_capacity(height.len());
    for y in 0..size {
        for x in 0..size {
            let b = base.map(|b| b[(y * size + x) as usize]);
            out.push(normal_at(height, size, x, y, strength, b));
        }
    }
    out
}

/// Pack a unit normal into 0..1 colour channels.
pub fn pack(n: [f32; 3]) -> [f32; 3] {
    n.map(|c| c * 0.5 + 0.5)
}

/// Inverse of [`pack`], renormalised.
pub fn unpack(c: [f32; 3]) -> [f32; 3] {
    let v = c.map(|x| x * 2.0 - 1.0);
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    v.map(|x| x / len)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: u32 = 16;

    fn ramp(f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
        (0..N * N).map(|i| f(i % N, i / N)).collect()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn flat_height_gives_the_flat_normal() {
        let h = vec![0.5; (N * N) as usize];
        for n in height_to_normal(&h, N, 1.0, None) {
            assert!(close(n, [0.0, 0.0, 1.0]));
        }
    }

    #[test]
    fn a_slope_tilts_the_normal_away_from_the_rise() {
        // Height rising to the right: the surface faces left, so x is negative.
        let right = ramp(|x, _| x as f32 / N as f32);
        let n = normal_at(&right, N, 8, 8, 1.0, None);
        assert!(n[0] < -0.005 && n[1].abs() < 1e-5 && n[2] > 0.0, "{n:?}");
        // Height rising towards the top of the image: the surface faces down the image, so y < 0.
        let up = ramp(|_, y| 1.0 - y as f32 / N as f32);
        let n = normal_at(&up, N, 8, 8, 1.0, None);
        assert!(n[1] < -0.005 && n[0].abs() < 1e-5, "{n:?}");
        // Both are unit length.
        for n in [normal_at(&right, N, 8, 8, 1.0, None), n] {
            assert!((n.iter().map(|c| c * c).sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn strength_scales_the_tilt_and_resolution_does_not_change_the_look() {
        let h = ramp(|x, _| x as f32 / N as f32);
        let weak = normal_at(&h, N, 8, 8, 1.0, None)[0];
        let strong = normal_at(&h, N, 8, 8, 4.0, None)[0];
        assert!(strong < weak * 3.0, "four times the strength tilts about four times as much: {weak} {strong}");
        assert_eq!(normal_at(&h, N, 8, 8, 0.0, None), [0.0, 0.0, 1.0]);
        // The same ramp at twice the resolution gives the same normal.
        let big: Vec<f32> = (0..4 * N * N).map(|i| ((i % (2 * N)) as f32 + 0.5) / (2 * N) as f32).collect();
        let a = normal_at(&h, N, 8, 8, 1.0, None)[0];
        let b = normal_at(&big, 2 * N, 16, 16, 1.0, None)[0];
        assert!((a - b).abs() < 5e-3, "{a} vs {b}");
    }

    #[test]
    fn baked_detail_is_kept_under_painted_height() {
        let flat = vec![0.5; (N * N) as usize];
        let tilted = [0.3, -0.2, 0.9327];
        let n = normal_at(&flat, N, 4, 4, 1.0, Some(tilted));
        let len = (tilted.iter().map(|c| c * c).sum::<f32>()).sqrt();
        assert!(close(n, tilted.map(|c| c / len)), "{n:?}");
        // A painted slope adds to it.
        let h = ramp(|x, _| x as f32 / N as f32);
        let with = normal_at(&h, N, 8, 8, 1.0, Some(tilted));
        assert!(with[0] < n[0], "painted rise pushes x further negative");
    }

    #[test]
    fn edges_clamp_and_packing_round_trips() {
        let h = ramp(|x, y| ((x + y) % 3) as f32 / 3.0);
        for (x, y) in [(0, 0), (N - 1, 0), (0, N - 1), (N - 1, N - 1)] {
            let n = normal_at(&h, N, x, y, 2.0, None);
            assert!(n.iter().all(|c| c.is_finite()));
        }
        let n = [0.3, -0.4, (1.0f32 - 0.25).sqrt()];
        assert!(close(unpack(pack(n)), n));
    }
}
