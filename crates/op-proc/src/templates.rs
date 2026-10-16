//! The built-in mask generators: each is a recipe that builds a [`Graph`] from a few named
//! parameters. The recipes are original; they only use the baked maps and generic operations.

use op_core::graph::{BlendOp, Graph, GraphBuilder, GraphNodeId, NoiseKind, Op};
use op_core::{Generator, GeneratorKind, MeshMap};

/// A parameter a template exposes: what the UI shows and what values are allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    /// Whole numbers only (seed, octaves).
    pub integer: bool,
}

const fn p(key: &'static str, label: &'static str, min: f32, max: f32, default: f32) -> ParamSpec {
    ParamSpec { key, label, min, max, default, integer: false }
}

const fn int(key: &'static str, label: &'static str, min: f32, max: f32, default: f32) -> ParamSpec {
    ParamSpec { key, label, min, max, default, integer: true }
}

const EDGE_WEAR: [ParamSpec; 5] = [
    p("amount", "Amount", 0.0, 1.0, 0.5),
    p("width", "Edge softness", 0.0, 1.0, 0.35),
    p("breakup", "Breakup", 0.0, 1.0, 0.6),
    p("scale", "Noise scale", 1.0, 64.0, 8.0),
    int("seed", "Seed", 0.0, 9999.0, 1.0),
];
const CREVICE_DIRT: [ParamSpec; 5] = [
    p("amount", "Amount", 0.0, 1.0, 0.5),
    p("softness", "Softness", 0.0, 1.0, 0.4),
    p("breakup", "Breakup", 0.0, 1.0, 0.5),
    p("scale", "Noise scale", 1.0, 64.0, 6.0),
    int("seed", "Seed", 0.0, 9999.0, 2.0),
];
const TOP_DUST: [ParamSpec; 5] = [
    p("amount", "Amount", 0.0, 1.0, 0.5),
    p("softness", "Softness", 0.0, 1.0, 0.4),
    p("breakup", "Breakup", 0.0, 1.0, 0.5),
    p("scale", "Noise scale", 1.0, 64.0, 10.0),
    int("seed", "Seed", 0.0, 9999.0, 3.0),
];
const HEIGHT_GRADIENT: [ParamSpec; 5] = [
    p("low", "Start height", 0.0, 1.0, 0.3),
    p("high", "End height", 0.0, 1.0, 0.7),
    p("breakup", "Breakup", 0.0, 1.0, 0.2),
    p("scale", "Noise scale", 1.0, 64.0, 6.0),
    int("seed", "Seed", 0.0, 9999.0, 4.0),
];
const NOISE: [ParamSpec; 4] = [
    p("scale", "Scale", 1.0, 64.0, 8.0),
    int("octaves", "Octaves", 1.0, 8.0, 5.0),
    p("contrast", "Contrast", 0.0, 1.0, 0.5),
    int("seed", "Seed", 0.0, 9999.0, 5.0),
];

const COLOR_SELECT: [ParamSpec; 4] = [
    p("r", "ID red", 0.0, 1.0, 1.0),
    p("g", "ID green", 0.0, 1.0, 0.0),
    p("b", "ID blue", 0.0, 1.0, 0.0),
    p("tolerance", "Tolerance", 0.0, 0.5, 0.04),
];

pub fn param_specs(kind: &GeneratorKind) -> &'static [ParamSpec] {
    match kind {
        GeneratorKind::EdgeWear => &EDGE_WEAR,
        GeneratorKind::CreviceDirt => &CREVICE_DIRT,
        GeneratorKind::TopDust => &TOP_DUST,
        GeneratorKind::HeightGradient => &HEIGHT_GRADIENT,
        GeneratorKind::Noise => &NOISE,
        GeneratorKind::ColorSelect => &COLOR_SELECT,
        GeneratorKind::Custom { .. } => &[],
    }
}

pub fn label(kind: &GeneratorKind) -> String {
    match kind {
        GeneratorKind::EdgeWear => "Edge wear".into(),
        GeneratorKind::CreviceDirt => "Crevice dirt".into(),
        GeneratorKind::TopDust => "Top dust".into(),
        GeneratorKind::HeightGradient => "Height gradient".into(),
        GeneratorKind::Noise => "Noise".into(),
        GeneratorKind::ColorSelect => "Color selection".into(),
        GeneratorKind::Custom { name, .. } => name.clone(),
    }
}

/// The maps a generator reads, so the UI can say what needs baking.
pub fn required_maps(kind: &GeneratorKind) -> Vec<MeshMap> {
    match kind {
        GeneratorKind::EdgeWear => vec![MeshMap::Curvature, MeshMap::Position],
        GeneratorKind::CreviceDirt => vec![MeshMap::AmbientOcclusion, MeshMap::Curvature, MeshMap::Position],
        GeneratorKind::TopDust => vec![MeshMap::WorldSpaceNormal, MeshMap::Position],
        GeneratorKind::HeightGradient => vec![MeshMap::Position],
        GeneratorKind::Noise => vec![MeshMap::Position],
        GeneratorKind::ColorSelect => vec![MeshMap::Id],
        GeneratorKind::Custom { .. } => vec![],
    }
}

/// The value of one parameter: the generator's own if set (clamped to the allowed range), else the default.
pub fn value(gen: &Generator, key: &str) -> f32 {
    let spec = param_specs(&gen.kind).iter().find(|s| s.key == key).unwrap_or_else(|| panic!("template has no parameter '{key}'"));
    let v = gen.params.get(key).copied().unwrap_or(spec.default).clamp(spec.min, spec.max);
    if spec.integer { v.round() } else { v }
}

fn levels(b: &mut GraphBuilder, input: GraphNodeId, lo: f32, hi: f32) -> GraphNodeId {
    b.add(Op::Levels { in_lo: lo, in_hi: hi, gamma: 1.0, out_lo: 0.0, out_hi: 1.0 }, &[input])
}

fn baked(b: &mut GraphBuilder, map: MeshMap, comp: u8) -> GraphNodeId {
    b.add(Op::Baked { map, comp }, &[])
}

fn blur(b: &mut GraphBuilder, input: GraphNodeId, radius: f32) -> GraphNodeId {
    if radius < 0.0005 { input } else { b.add(Op::Blur { radius }, &[input]) }
}

/// Break a mask up with noise: `base * (1 - strength + strength * noise)`. Strength 0 leaves
/// the mask exactly as it was.
fn breakup(b: &mut GraphBuilder, base: GraphNodeId, strength: f32, scale: f32, seed: u32) -> GraphNodeId {
    if strength <= 0.0 {
        return base;
    }
    let n = b.add(Op::Noise { kind: NoiseKind::Perlin, scale, octaves: 5, persistence: 0.55, seed, uv_space: false }, &[]);
    let nl = levels(b, n, 0.3, 0.7);
    let product = b.add(Op::Blend { op: BlendOp::Multiply, opacity: 1.0 }, &[base, nl]);
    let t = b.add(Op::Const { value: strength }, &[]);
    b.add(Op::Lerp, &[base, product, t])
}

/// Build the graph for a generator. The result is validated.
pub fn build_graph(gen: &Generator) -> Result<Graph, String> {
    gen.validate()?;
    let mut b = GraphBuilder::new();
    let v = |key: &str| value(gen, key);
    let out = match &gen.kind {
        GeneratorKind::Custom { graph, .. } => {
            let mut g = graph.clone();
            if gen.invert {
                let id = g.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
                g.nodes.push(op_core::graph::GraphNode { id, op: Op::Invert, inputs: vec![g.output] });
                g.output = id;
            }
            g.validate()?;
            return Ok(g);
        }
        GeneratorKind::EdgeWear => {
            let curvature = baked(&mut b, MeshMap::Curvature, 0);
            let lo = 0.52 + (1.0 - v("amount")) * 0.38;
            let edge = levels(&mut b, curvature, lo, lo + 0.08);
            let wide = blur(&mut b, edge, v("width") * 0.03);
            let broken = breakup(&mut b, wide, v("breakup"), v("scale"), v("seed") as u32);
            levels(&mut b, broken, 0.15, 0.55)
        }
        GeneratorKind::CreviceDirt => {
            let (amount, softness) = (v("amount"), v("softness"));
            let ao = baked(&mut b, MeshMap::AmbientOcclusion, 0);
            let occluded = b.add(Op::Invert, &[ao]);
            let lo = (1.0 - amount) * 0.5;
            let from_ao = levels(&mut b, occluded, lo, lo + 0.1 + softness * 0.5);
            let curvature = baked(&mut b, MeshMap::Curvature, 0);
            let concave = b.add(Op::Invert, &[curvature]);
            let clo = 0.5 + (1.0 - amount) * 0.3;
            let from_curvature = levels(&mut b, concave, clo, clo + 0.1 + softness * 0.3);
            let both = b.add(Op::Blend { op: BlendOp::Max, opacity: 1.0 }, &[from_ao, from_curvature]);
            let soft = blur(&mut b, both, softness * 0.01);
            breakup(&mut b, soft, v("breakup"), v("scale"), v("seed") as u32)
        }
        GeneratorKind::TopDust => {
            let (amount, softness) = (v("amount"), v("softness"));
            // The baked world normal stores n.y as 0.5 + 0.5 * n.y.
            let up = baked(&mut b, MeshMap::WorldSpaceNormal, 1);
            let lo = (0.5 + (1.0 - amount) * 0.45).min(0.97);
            let hi = (lo + 0.05 + softness * 0.4).min(1.0);
            let facing = levels(&mut b, up, lo, hi);
            breakup(&mut b, facing, v("breakup"), v("scale"), v("seed") as u32)
        }
        GeneratorKind::HeightGradient => {
            let y = baked(&mut b, MeshMap::Position, 1);
            let ramp = levels(&mut b, y, v("low"), v("high"));
            breakup(&mut b, ramp, v("breakup"), v("scale"), v("seed") as u32)
        }
        GeneratorKind::ColorSelect => {
            // How far each ID channel is from the chosen colour; the worst one decides.
            let mut worst: Option<GraphNodeId> = None;
            for (comp, key) in ["r", "g", "b"].into_iter().enumerate() {
                let id = baked(&mut b, MeshMap::Id, comp as u8);
                let target = b.add(Op::Const { value: v(key) }, &[]);
                let d = b.add(Op::Blend { op: BlendOp::Difference, opacity: 1.0 }, &[id, target]);
                worst = Some(match worst {
                    None => d,
                    Some(w) => b.add(Op::Blend { op: BlendOp::Max, opacity: 1.0 }, &[w, d]),
                });
            }
            let tolerance = v("tolerance");
            let far = b.add(Op::Threshold { level: tolerance + 0.004, softness: 0.008 }, &[worst.expect("three channels")]);
            b.add(Op::Invert, &[far])
        }
        GeneratorKind::Noise => {
            let n = b.add(
                Op::Noise { kind: NoiseKind::Perlin, scale: v("scale"), octaves: v("octaves") as u32, persistence: 0.5, seed: v("seed") as u32, uv_space: false },
                &[],
            );
            let k = 0.45 * v("contrast");
            levels(&mut b, n, k, 1.0 - k)
        }
    };
    let out = if gen.invert { b.add(Op::Invert, &[out]) } else { out };
    let graph = b.finish(out);
    graph.validate()?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [GeneratorKind; 6] = [
        GeneratorKind::EdgeWear,
        GeneratorKind::CreviceDirt,
        GeneratorKind::TopDust,
        GeneratorKind::HeightGradient,
        GeneratorKind::Noise,
        GeneratorKind::ColorSelect,
    ];

    #[test]
    fn every_template_builds_a_valid_graph_with_defaults_and_at_the_extremes() {
        for kind in ALL {
            for invert in [false, true] {
                // defaults
                let mut g = Generator::new(kind.clone());
                g.invert = invert;
                build_graph(&g).unwrap_or_else(|e| panic!("{} defaults: {e}", label(&kind)));
                // every parameter at its minimum and at its maximum
                for pick_max in [false, true] {
                    let mut g = Generator::new(kind.clone());
                    g.invert = invert;
                    for s in param_specs(&kind) {
                        g.params.insert(s.key.into(), if pick_max { s.max } else { s.min });
                    }
                    build_graph(&g).unwrap_or_else(|e| panic!("{} extremes (max={pick_max}): {e}", label(&kind)));
                }
            }
        }
    }

    #[test]
    fn parameter_values_are_clamped_and_integers_rounded() {
        let g = Generator::new(GeneratorKind::EdgeWear).with("amount", 7.0).with("seed", 12.7).with("scale", -3.0);
        assert_eq!(value(&g, "amount"), 1.0);
        assert_eq!(value(&g, "seed"), 13.0);
        assert_eq!(value(&g, "scale"), 1.0);
        assert_eq!(value(&Generator::new(GeneratorKind::EdgeWear), "amount"), 0.5, "default");
        // Out-of-range or unknown keys can never make graph construction fail or panic.
        let wild = Generator::new(GeneratorKind::Noise).with("octaves", 999.0).with("unknown", 5.0);
        build_graph(&wild).unwrap();
        assert_eq!(value(&wild, "octaves"), 8.0);
    }

    #[test]
    fn specs_are_sane() {
        for kind in ALL {
            for s in param_specs(&kind) {
                assert!(s.min < s.max, "{}: empty range", s.key);
                assert!((s.min..=s.max).contains(&s.default), "{}: default outside range", s.key);
                assert!(!s.label.is_empty());
            }
            let keys: std::collections::HashSet<_> = param_specs(&kind).iter().map(|s| s.key).collect();
            assert_eq!(keys.len(), param_specs(&kind).len(), "duplicate parameter keys");
            assert!(!required_maps(&kind).is_empty());
        }
    }

    #[test]
    fn a_custom_generator_is_used_as_is_and_can_be_inverted() {
        let mut b = GraphBuilder::new();
        let c = b.add(Op::Const { value: 0.25 }, &[]);
        let graph = b.finish(c);
        let g = Generator::new(GeneratorKind::Custom { name: "mine".into(), graph: graph.clone() });
        assert_eq!(build_graph(&g).unwrap(), graph);
        let inv = build_graph(&g.clone().inverted()).unwrap();
        assert_eq!(inv.nodes.len(), 2);
        assert_eq!(inv.node(inv.output).unwrap().op, Op::Invert);
        // A broken custom graph is rejected, not built.
        let broken = Generator::new(GeneratorKind::Custom {
            name: "x".into(),
            graph: Graph { nodes: vec![], output: 0 },
        });
        assert!(build_graph(&broken).is_err());
        assert_eq!(label(&g.kind), "mine");
    }
}
