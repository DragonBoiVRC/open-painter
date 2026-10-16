//! CPU reference implementation of the compositing math.
//!
//! The GPU compositor (`composite.wgsl`) implements the same formulas; a test in
//! op-app renders on the GPU and compares against this module.
//!
//! Colors are premultiplied: `(r*a, g*a, b*a, a)`. Blending follows the W3C
//! Compositing and Blending model with separable and non-separable blend functions.

use crate::{srgb_to_linear, BlendMode, Document, FillParams, Node, NodeKind};

/// Separable blend function B(cb, cs) on straight (un-premultiplied) values.
pub fn blend(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::Overlay => {
            if cb <= 0.5 { 2.0 * cb * cs } else { 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs) }
        }
        BlendMode::Add => (cb + cs).min(1.0),
        BlendMode::Subtract => (cb - cs).max(0.0),
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::SoftLight => {
            if cs <= 0.5 {
                cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
            } else {
                let d = if cb <= 0.25 { ((16.0 * cb - 12.0) * cb + 4.0) * cb } else { cb.sqrt() };
                cb + (2.0 * cs - 1.0) * (d - cb)
            }
        }
        BlendMode::HardLight => {
            if cs <= 0.5 { 2.0 * cb * cs } else { cb + (2.0 * cs - 1.0) - cb * (2.0 * cs - 1.0) }
        }
        BlendMode::ColorDodge => {
            if cb <= 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if cb >= 1.0 {
                1.0
            } else if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        BlendMode::LinearBurn => (cb + cs - 1.0).max(0.0),
        BlendMode::Exclusion => cb + cs - 2.0 * cb * cs,
        BlendMode::Divide => {
            if cs <= 0.0 { 1.0 } else { (cb / cs).min(1.0) }
        }
        // Non-separable modes need all three channels; see `blend_rgb`.
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => cs,
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut c = c;
    if n < 0.0 {
        c = c.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 {
        c = c.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color(c.map(|v| v + d))
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    if mx > mn {
        c.map(|v| (v - mn) * s / (mx - mn))
    } else {
        [0.0; 3]
    }
}

/// Blend function on whole colors: the separable modes per channel, plus the
/// non-separable hue / saturation / color / luminosity modes.
pub fn blend_rgb(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    match mode {
        BlendMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
        BlendMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
        BlendMode::Color => set_lum(cs, lum(cb)),
        BlendMode::Luminosity => set_lum(cb, lum(cs)),
        _ => [0, 1, 2].map(|i| blend(mode, cb[i], cs[i])),
    }
}

/// Composite a source with straight color `cs` and effective alpha `a_s` over a
/// premultiplied backdrop (rgb premultiplied, alpha last).
pub fn over(mode: BlendMode, backdrop: [f32; 4], cs: [f32; 3], a_s: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let mut out = [0.0; 4];
    let cb = if ab > 0.0 { [backdrop[0] / ab, backdrop[1] / ab, backdrop[2] / ab] } else { [0.0; 3] };
    let blended = blend_rgb(mode, cb, cs);
    for i in 0..3 {
        let b = blended[i];
        out[i] = (1.0 - a_s) * backdrop[i] + a_s * ((1.0 - ab) * cs[i] + ab * b);
    }
    out[3] = a_s + ab * (1.0 - a_s);
    out
}

/// One pixel of every channel, premultiplied.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PixelSet {
    pub base: [f32; 4],
    /// (value * coverage, coverage)
    pub metal: [f32; 2],
    pub rough: [f32; 2],
    /// (value * coverage, coverage); the value is a height in 0..1 where 0.5 is flat.
    pub height: [f32; 2],
    /// Premultiplied linear colour, like `base`.
    pub emissive: [f32; 4],
}

fn over_scalar(mode: BlendMode, backdrop: [f32; 2], value: f32, a_s: f32) -> [f32; 2] {
    let o = over(mode, [backdrop[0], backdrop[0], backdrop[0], backdrop[1]], [value; 3], a_s);
    [o[0], o[3]]
}

fn apply_layer(node: &Node, bk: PixelSet, src: PixelSet) -> PixelSet {
    let (mode, k) = (node.blend, node.opacity);
    let [mb, mm, mr, mh, me] = node.channels.blended().map(|c| (c.mode(mode), k * c.factor()));
    // Un-premultiply the source, then composite each channel with its own coverage.
    let straight = |v: f32, a: f32| if a > 0.0 { v / a } else { 0.0 };
    let base_c = [straight(src.base[0], src.base[3]), straight(src.base[1], src.base[3]), straight(src.base[2], src.base[3])];
    let emissive_c = [straight(src.emissive[0], src.emissive[3]), straight(src.emissive[1], src.emissive[3]), straight(src.emissive[2], src.emissive[3])];
    PixelSet {
        base: over(mb.0, bk.base, base_c, src.base[3] * mb.1),
        metal: over_scalar(mm.0, bk.metal, straight(src.metal[0], src.metal[1]), src.metal[1] * mm.1),
        rough: over_scalar(mr.0, bk.rough, straight(src.rough[0], src.rough[1]), src.rough[1] * mr.1),
        height: over_scalar(mh.0, bk.height, straight(src.height[0], src.height[1]), src.height[1] * mh.1),
        emissive: over(me.0, bk.emissive, emissive_c, src.emissive[3] * me.1),
    }
}

fn fill_pixel(f: &FillParams) -> PixelSet {
    let mut p = PixelSet::default();
    if let Some(c) = f.base {
        let l = c.map(srgb_to_linear);
        p.base = [l[0], l[1], l[2], 1.0];
    }
    if let Some(m) = f.metallic {
        p.metal = [m, 1.0];
    }
    if let Some(r) = f.roughness {
        p.rough = [r, 1.0];
    }
    if let Some(h) = f.height {
        p.height = [h, 1.0];
    }
    if let Some(c) = f.emissive {
        let l = c.map(srgb_to_linear);
        p.emissive = [l[0], l[1], l[2], 1.0];
    }
    p
}

fn eval_nodes(nodes: &[Node]) -> PixelSet {
    let mut acc = PixelSet::default();
    for n in nodes.iter().filter(|n| n.visible) {
        // Paint layers are empty here (no texture data); masks count as fully revealing.
        let src = match &n.kind {
            NodeKind::Paint => PixelSet::default(),
            NodeKind::Fill(f) => fill_pixel(f),
            NodeKind::Group { children } => eval_nodes(children),
        };
        acc = apply_layer(n, acc, src);
    }
    acc
}

/// Evaluate a document made only of fill layers and groups at a single pixel.
pub fn eval_fill_only(doc: &Document) -> PixelSet {
    eval_nodes(&doc.nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeKind;

    fn fill_m(v: f32) -> NodeKind {
        NodeKind::Fill(FillParams { metallic: Some(v), ..Default::default() })
    }

    #[test]
    fn normal_opaque_replaces_backdrop() {
        let o = over(BlendMode::Normal, [0.2, 0.2, 0.2, 1.0], [0.9, 0.5, 0.1], 1.0);
        assert_eq!(o, [0.9, 0.5, 0.1, 1.0]);
    }

    #[test]
    fn multiply_half_opacity_matches_hand_calculation() {
        // backdrop 0.8, source 0.5 multiply at 50%: 0.5*0.8 + 0.5*(0.8*0.5) = 0.6
        let o = over(BlendMode::Multiply, [0.8, 0.8, 0.8, 1.0], [0.5; 3], 0.5);
        assert!((o[0] - 0.6).abs() < 1e-6 && (o[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn over_empty_backdrop_keeps_source_regardless_of_mode() {
        for m in BlendMode::ALL {
            let o = over(m, [0.0; 4], [0.3, 0.6, 0.9], 0.5);
            assert!((o[0] - 0.15).abs() < 1e-6 && (o[3] - 0.5).abs() < 1e-6, "{m:?}");
        }
    }

    #[test]
    fn alpha_accumulates_like_porter_duff_over() {
        let o = over(BlendMode::Normal, [0.0, 0.0, 0.0, 0.5], [0.0; 3], 0.5);
        assert!((o[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn group_composites_children_in_isolation() {
        // Multiply inside a group multiplies against the group's *own* backdrop (empty),
        // so it reads as plain 0.5; the group then multiplies that over 0.8.
        let mut d = Document::default();
        d.add(None, None, "bg", fill_m(0.8));
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        let inner = d.add(Some(g), None, "inner", fill_m(0.5));
        d.find_mut(inner).unwrap().blend = BlendMode::Multiply;
        d.find_mut(g).unwrap().blend = BlendMode::Multiply;
        let p = eval_fill_only(&d);
        assert!((p.metal[0] - 0.4).abs() < 1e-6, "got {}", p.metal[0]);
    }

    #[test]
    fn hidden_layers_are_skipped() {
        let mut d = Document::default();
        d.add(None, None, "a", fill_m(0.3));
        let b = d.add(None, None, "b", fill_m(0.9));
        d.find_mut(b).unwrap().visible = false;
        assert!((eval_fill_only(&d).metal[0] - 0.3).abs() < 1e-6);
    }
}
