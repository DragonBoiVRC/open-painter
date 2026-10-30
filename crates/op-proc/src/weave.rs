//! Textile patterns for [`op_core::graph::Op::Weave`]. Coordinates are in cells (one thread
//! crossing per cell); the result is the surface height in 0..1, low in the gaps between threads.

use op_core::graph::WeaveKind;
use std::f32::consts::PI;

/// 0 at a cell edge, 1 in the middle.
fn hump(f: f32) -> f32 {
    (PI * f).sin()
}

pub fn height(kind: WeaveKind, x: f32, y: f32) -> f32 {
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let h = match kind {
        WeaveKind::Plain | WeaveKind::Twill | WeaveKind::Basket => {
            let warp_top = match kind {
                WeaveKind::Plain => (ix + iy).rem_euclid(2) == 0,
                WeaveKind::Twill => (ix + iy).rem_euclid(4) != 0,
                _ => ((ix >> 1) + (iy >> 1)).rem_euclid(2) == 0,
            };
            // The thread on top arches across its width and rises toward the middle of the cell.
            let (across, along) = if warp_top { (fx, fy) } else { (fy, fx) };
            0.3 + 0.7 * hump(across).powf(0.7) * (0.65 + 0.35 * hump(along))
        }
        WeaveKind::Knit => {
            // Two slanted legs per stitch, meeting in a V.
            let leg = |centre: f32, lean: f32| (1.0 - (fx - (centre + lean * (fy - 0.5))).abs() / 0.26).max(0.0).powf(0.6);
            0.2 + 0.8 * leg(0.27, 0.36).max(leg(0.73, -0.36)) * (0.7 + 0.3 * hump(fy))
        }
        WeaveKind::Ribs => 0.15 + 0.85 * hump(fx).powf(0.8),
    };
    h.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [WeaveKind; 5] = [WeaveKind::Plain, WeaveKind::Twill, WeaveKind::Basket, WeaveKind::Knit, WeaveKind::Ribs];

    #[test]
    fn patterns_stay_in_range_repeat_and_have_relief() {
        for kind in ALL {
            let (mut lo, mut hi) = (1.0f32, 0.0f32);
            for i in 0..64 {
                for j in 0..64 {
                    let (x, y) = (i as f32 / 8.0 + 0.03, j as f32 / 8.0 + 0.07);
                    let v = height(kind, x, y);
                    assert!((0.0..=1.0).contains(&v), "{kind:?} out of range");
                    lo = lo.min(v);
                    hi = hi.max(v);
                    // Repeats every 4 cells in both directions (the longest repeat, basket and twill).
                    assert!((v - height(kind, x + 4.0, y + 4.0)).abs() < 1e-3, "{kind:?} does not tile");
                }
            }
            assert!(hi - lo > 0.4, "{kind:?} is nearly flat");
        }
    }
}
