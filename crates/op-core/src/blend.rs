//! CPU reference implementation of the compositing math.
//!
//! The GPU compositor (`composite.wgsl`) implements the same formulas; a test in
//! op-app renders on the GPU and compares against this module.
//!
//! Colors are premultiplied: `(r*a, g*a, b*a, a)`. Blending follows the W3C
//! Compositing and Blending model with separable blend functions.

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
    }
}

/// Composite a source with straight color `cs` and effective alpha `a_s` over a
/// premultiplied backdrop (rgb premultiplied, alpha last).
pub fn over(mode: BlendMode, backdrop: [f32; 4], cs: [f32; 3], a_s: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let mut out = [0.0; 4];
    for i in 0..3 {
        let cb = if ab > 0.0 { backdrop[i] / ab } else { 0.0 };
        let b = blend(mode, cb, cs[i]);
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
}

fn over_scalar(mode: BlendMode, backdrop: [f32; 2], value: f32, a_s: f32) -> [f32; 2] {
    let o = over(mode, [backdrop[0], backdrop[0], backdrop[0], backdrop[1]], [value; 3], a_s);
    [o[0], o[3]]
}

fn apply_layer(mode: BlendMode, k: f32, bk: PixelSet, src: PixelSet) -> PixelSet {
    // Un-premultiply the source, then composite each channel with its own coverage.
    let straight = |v: f32, a: f32| if a > 0.0 { v / a } else { 0.0 };
    let base_c = [straight(src.base[0], src.base[3]), straight(src.base[1], src.base[3]), straight(src.base[2], src.base[3])];
    PixelSet {
        base: over(mode, bk.base, base_c, src.base[3] * k),
        metal: over_scalar(mode, bk.metal, straight(src.metal[0], src.metal[1]), src.metal[1] * k),
        rough: over_scalar(mode, bk.rough, straight(src.rough[0], src.rough[1]), src.rough[1] * k),
        height: over_scalar(mode, bk.height, straight(src.height[0], src.height[1]), src.height[1] * k),
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
        acc = apply_layer(n.blend, n.opacity, acc, src);
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
