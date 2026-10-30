//! Smart materials: a saved layer subtree (fills, groups, masks with generators) that can be
//! dropped onto any document. Everything in one is procedural, so no texture data is stored.
//!
//! The built-in ones below are original; users can add their own as JSON files.

use crate::graph::{BlendOp, GraphBuilder, NoiseKind, Op, WeaveKind};
use crate::{BlendMode, FillParams, Generator, GeneratorKind, Mask, Node, NodeKind};
use serde::{Deserialize, Serialize};

pub const MAX_SMART_NODES: usize = 200;
pub const MAX_SMART_DEPTH: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SmartMaterial {
    pub name: String,
    /// Bottom to top. Node ids are placeholders; they are replaced on insertion.
    pub nodes: Vec<Node>,
}

fn check_node(n: &Node, depth: usize, count: &mut usize) -> Result<(), String> {
    *count += 1;
    if *count > MAX_SMART_NODES {
        return Err(format!("more than {MAX_SMART_NODES} layers"));
    }
    if depth > MAX_SMART_DEPTH {
        return Err("layers are nested too deeply".into());
    }
    if n.name.len() > 128 {
        return Err("a layer name is too long".into());
    }
    if !(n.opacity.is_finite() && (0.0..=1.0).contains(&n.opacity)) {
        return Err(format!("layer '{}' has an opacity outside 0..=1", n.name));
    }
    if let Some(m) = &n.mask {
        if let Some(g) = &m.generator {
            g.validate().map_err(|e| format!("layer '{}': {e}", n.name))?;
        }
    }
    match &n.kind {
        NodeKind::Paint => {}
        NodeKind::Fill(f) => {
            let unit = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
            let ok = f.base.is_none_or(|c| c.iter().all(|v| unit(*v)))
                && f.metallic.is_none_or(unit)
                && f.roughness.is_none_or(unit)
                && f.height.is_none_or(unit);
            if !ok {
                return Err(format!("layer '{}' has fill values outside 0..=1", n.name));
            }
            if f.maps.iter().any(|(_, t)| !crate::is_valid_texture_id(&t.id)) {
                return Err(format!("layer '{}' names a texture in an unknown form", n.name));
            }
            if !f.uv.is_sane() {
                return Err(format!("layer '{}' has an unusable texture placement", n.name));
            }
        }
        NodeKind::Group { children } => {
            for c in children {
                check_node(c, depth + 1, count)?;
            }
        }
    }
    Ok(())
}

impl SmartMaterial {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.name.len() > 128 {
            return Err("a smart material needs a name of up to 128 characters".into());
        }
        if self.nodes.is_empty() {
            return Err("a smart material needs at least one layer".into());
        }
        let mut count = 0;
        self.nodes.iter().try_for_each(|n| check_node(n, 0, &mut count))
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let m: SmartMaterial = serde_json::from_str(text).map_err(|e| e.to_string())?;
        m.validate()?;
        Ok(m)
    }

    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

fn layer(name: &str, base: [f32; 3], metallic: Option<f32>, roughness: Option<f32>) -> Node {
    Node {
        id: 0,
        name: name.to_string(),
        visible: true,
        opacity: 1.0,
        blend: BlendMode::Normal,
        mask: None,
        adjust: Default::default(),
        filters: Vec::new(),
        channels: Default::default(),
        kind: NodeKind::Fill(FillParams { base: Some(base), metallic, roughness, ..Default::default() }),
    }
}

impl Node {
    fn with_mask(mut self, g: Generator) -> Self {
        self.mask = Some(Mask::generated(g));
        self
    }

    /// Raises (above 0.5) or sinks (below) this fill's height where it shows.
    fn with_height(mut self, h: f32) -> Self {
        if let NodeKind::Fill(f) = &mut self.kind {
            f.height = Some(h);
        }
        self
    }

    fn blended(mut self, blend: BlendMode, opacity: f32) -> Self {
        self.blend = blend;
        self.opacity = opacity;
        self
    }
}

/// A layer that only sets some channels (height, roughness), masked by a generator.
fn detail(name: &str, fill: FillParams, g: Generator) -> Node {
    let mut n = layer(name, [0.0; 3], None, None);
    n.kind = NodeKind::Fill(fill);
    n.with_mask(g)
}

/// Surface relief: noise-driven height, from which the normal map is derived.
fn relief(name: &str, height: f32, scale: f32) -> Node {
    let noise = Generator::new(GeneratorKind::Noise).with("scale", scale).with("contrast", 0.7);
    detail(name, FillParams { height: Some(height), ..Default::default() }, noise)
}

/// A fabric: dyed base, woven height, darker gaps between the threads, and some mottling.
/// `count` is how many threads fit across the UV layout; edit the material's JSON to change it.
fn fabric(name: &str, colour: [f32; 3], roughness: f32, kind: WeaveKind, count: f32, bump: f32, fuzz: f32) -> SmartMaterial {
    // Irregular thread thickness: a little UV-space noise added to the pattern.
    let weave = || {
        let mut b = GraphBuilder::new();
        let w = b.add(Op::Weave { kind, count }, &[]);
        let out = if fuzz > 0.0 {
            let n = b.add(Op::Noise { kind: NoiseKind::Perlin, scale: count / 2.0, octaves: 3, persistence: 0.5, seed: 7, uv_space: true }, &[]);
            b.add(Op::Blend { op: BlendOp::Add, opacity: fuzz }, &[w, n])
        } else {
            w
        };
        Generator::new(GeneratorKind::Custom { name: "Weave".into(), graph: b.finish(out) })
    };
    let dark = colour.map(|c| c * 0.35);
    SmartMaterial {
        name: name.into(),
        nodes: vec![group(
            name,
            vec![
                layer("Fabric", colour, Some(0.0), Some(roughness)),
                detail("Weave height", FillParams { height: Some(bump), ..Default::default() }, weave()),
                layer("Gaps", dark, None, None).blended(BlendMode::Multiply, 0.8).with_mask(weave().inverted()),
                layer("Mottling", dark, None, None)
                    .blended(BlendMode::Multiply, 0.3)
                    .with_mask(Generator::new(GeneratorKind::Noise).with("scale", 20.0)),
            ],
        )],
    }
}

/// Fabrics, built on the weave pattern op.
pub fn builtin_fabrics() -> Vec<SmartMaterial> {
    use WeaveKind::*;
    vec![
        fabric("Canvas", [0.78, 0.72, 0.58], 0.9, Plain, 70.0, 0.72, 0.2),
        fabric("Linen", [0.82, 0.76, 0.64], 0.85, Plain, 60.0, 0.68, 0.5),
        fabric("Denim", [0.12, 0.20, 0.40], 0.9, Twill, 80.0, 0.72, 0.25),
        fabric("Burlap", [0.55, 0.42, 0.25], 0.95, Basket, 40.0, 0.8, 0.4),
        fabric("Wool knit", [0.62, 0.55, 0.45], 0.95, Knit, 36.0, 0.8, 0.25),
        fabric("Corduroy", [0.40, 0.22, 0.14], 0.9, Ribs, 40.0, 0.75, 0.1),
    ]
}

fn group(name: &str, children: Vec<Node>) -> Node {
    Node {
        id: 0,
        name: name.to_string(),
        visible: true,
        opacity: 1.0,
        blend: BlendMode::Normal,
        mask: None,
        adjust: Default::default(),
        filters: Vec::new(),
        channels: Default::default(),
        kind: NodeKind::Group { children },
    }
}

/// The smart materials that ship with the application.
pub fn builtin_smart_materials() -> Vec<SmartMaterial> {
    use GeneratorKind::*;
    let mut all = vec![
        SmartMaterial {
            name: "Worn painted metal".into(),
            nodes: vec![group(
                "Worn painted metal",
                vec![
                    layer("Bare metal", [0.62, 0.62, 0.64], Some(1.0), Some(0.35)),
                    // The paint chips away where the edge wear is strong.
                    layer("Paint", [0.70, 0.12, 0.10], Some(0.0), Some(0.5))
                        .with_height(0.56)
                        .with_mask(Generator::new(EdgeWear).inverted()),
                    layer("Grime", [0.05, 0.045, 0.04], None, Some(0.9))
                        .blended(BlendMode::Multiply, 0.7)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
        SmartMaterial {
            name: "Rusty iron".into(),
            nodes: vec![group(
                "Rusty iron",
                vec![
                    layer("Steel", [0.45, 0.45, 0.47], Some(1.0), Some(0.4)),
                    layer("Rust", [0.42, 0.17, 0.07], Some(0.0), Some(0.85))
                        .with_height(0.4)
                        .with_mask(Generator::new(Noise).with("scale", 6.0).with("contrast", 0.75)),
                    layer("Rust (edges)", [0.25, 0.10, 0.05], Some(0.0), Some(0.9))
                        .with_height(0.38)
                        .with_mask(Generator::new(EdgeWear).with("amount", 0.6)),
                ],
            )],
        },
        SmartMaterial {
            name: "Dusty stone".into(),
            nodes: vec![group(
                "Dusty stone",
                vec![
                    layer("Stone", [0.45, 0.43, 0.40], Some(0.0), Some(0.9)),
                    relief("Stone relief", 0.68, 14.0),
                    layer("Dust", [0.62, 0.57, 0.48], None, Some(1.0)).with_height(0.56).with_mask(Generator::new(TopDust)),
                    layer("Dark crevices", [0.12, 0.11, 0.10], None, None)
                        .blended(BlendMode::Multiply, 0.8)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
        SmartMaterial {
            name: "Dirty steel".into(),
            nodes: vec![group(
                "Dirty steel",
                vec![
                    layer("Steel", [0.56, 0.57, 0.58], Some(1.0), Some(0.3)),
                    layer("Smudges", [0.30, 0.30, 0.31], None, Some(0.65))
                        .blended(BlendMode::Multiply, 0.6)
                        .with_mask(Generator::new(Noise).with("scale", 10.0).with("contrast", 0.6)),
                    layer("Grime", [0.06, 0.055, 0.05], None, Some(0.9))
                        .blended(BlendMode::Multiply, 0.8)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
        SmartMaterial {
            name: "Worn gold".into(),
            nodes: vec![group(
                "Worn gold",
                vec![
                    layer("Gold", [1.0, 0.77, 0.34], Some(1.0), Some(0.25)),
                    layer("Rubbed edges", [0.78, 0.60, 0.30], Some(1.0), Some(0.5))
                        .with_height(0.46)
                        .with_mask(Generator::new(EdgeWear)),
                    layer("Tarnish", [0.20, 0.13, 0.05], None, Some(0.8))
                        .blended(BlendMode::Multiply, 0.7)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
        SmartMaterial {
            name: "Weathered copper".into(),
            nodes: vec![group(
                "Weathered copper",
                vec![
                    layer("Copper", [0.95, 0.64, 0.54], Some(1.0), Some(0.3)),
                    layer("Patina", [0.28, 0.62, 0.52], Some(0.0), Some(0.7))
                        .with_height(0.56)
                        .with_mask(Generator::new(CreviceDirt).with("amount", 0.7)),
                    layer("Patina (blotches)", [0.35, 0.68, 0.58], Some(0.0), Some(0.75))
                        .with_mask(Generator::new(Noise).with("scale", 7.0).with("contrast", 0.8)),
                ],
            )],
        },
        SmartMaterial {
            name: "Chipped paint".into(),
            nodes: vec![group(
                "Chipped paint",
                vec![
                    layer("Plaster", [0.55, 0.53, 0.50], Some(0.0), Some(0.95)),
                    relief("Plaster relief", 0.62, 30.0),
                    layer("Paint", [0.80, 0.78, 0.70], Some(0.0), Some(0.6))
                        .with_height(0.58)
                        .with_mask(Generator::new(EdgeWear).inverted()),
                    layer("Stains", [0.20, 0.17, 0.12], None, Some(0.9))
                        .blended(BlendMode::Multiply, 0.6)
                        .with_mask(Generator::new(Noise).with("scale", 5.0).with("contrast", 0.7)),
                ],
            )],
        },
        SmartMaterial {
            name: "Concrete".into(),
            nodes: vec![group(
                "Concrete",
                vec![
                    layer("Concrete", [0.50, 0.49, 0.47], Some(0.0), Some(0.92)),
                    relief("Concrete relief", 0.66, 22.0),
                    layer("Mottling", [0.34, 0.33, 0.32], None, Some(0.95))
                        .blended(BlendMode::Multiply, 0.7)
                        .with_mask(Generator::new(Noise).with("scale", 14.0).with("contrast", 0.5)),
                    layer("Damp stains", [0.12, 0.12, 0.11], None, Some(0.8))
                        .blended(BlendMode::Multiply, 0.8)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
        SmartMaterial {
            name: "Mossy stone".into(),
            nodes: vec![group(
                "Mossy stone",
                vec![
                    layer("Stone", [0.36, 0.35, 0.33], Some(0.0), Some(0.9)),
                    relief("Stone relief", 0.68, 14.0),
                    layer("Moss", [0.16, 0.30, 0.08], Some(0.0), Some(1.0))
                        .with_height(0.6)
                        .with_mask(Generator::new(CreviceDirt).with("amount", 0.65)),
                    layer("Moss (top)", [0.20, 0.36, 0.10], Some(0.0), Some(1.0))
                        .with_mask(Generator::new(TopDust).with("amount", 0.4)),
                ],
            )],
        },
        SmartMaterial {
            name: "Snow".into(),
            nodes: vec![group(
                "Snow",
                vec![
                    layer("Rock", [0.28, 0.27, 0.26], Some(0.0), Some(0.9)),
                    relief("Rock relief", 0.7, 12.0),
                    layer("Snow", [0.93, 0.95, 0.98], Some(0.0), Some(0.55))
                        .with_height(0.62)
                        .with_mask(Generator::new(TopDust).with("amount", 0.7)),
                    layer("Drifts", [0.97, 0.98, 1.0], Some(0.0), Some(0.5))
                        .with_height(0.68)
                        .with_mask(Generator::new(Noise).with("scale", 6.0).with("contrast", 0.6)),
                ],
            )],
        },
        SmartMaterial {
            name: "Scuffed plastic".into(),
            nodes: vec![group(
                "Scuffed plastic",
                vec![
                    layer("Plastic", [0.10, 0.12, 0.45], Some(0.0), Some(0.35)),
                    layer("Scuffs", [0.35, 0.37, 0.65], Some(0.0), Some(0.7))
                        .with_height(0.46)
                        .with_mask(Generator::new(EdgeWear).with("amount", 0.4)),
                    layer("Dirt", [0.08, 0.07, 0.06], None, Some(0.85))
                        .blended(BlendMode::Multiply, 0.6)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
    ];
    all.extend(builtin_fabrics());
    all.extend(builtin_base_materials());
    all
}

/// Plain single-layer PBR materials (metals, plastics, rubber and so on), as in the default
/// material library of other painters. Values are common sRGB reference colours.
pub fn builtin_base_materials() -> Vec<SmartMaterial> {
    // name, sRGB colour, metallic, roughness, then the surface: noise scale (0 = perfectly
    // flat), height of the bumps and how much the roughness varies.
    const TABLE: &[(&str, [f32; 3], f32, f32, f32, f32, f32)] = &[
        ("Aluminium", [0.96, 0.96, 0.97], 1.0, 0.35, 50.0, 0.53, 0.15),
        ("Brass", [0.96, 0.82, 0.55], 1.0, 0.3, 50.0, 0.53, 0.12),
        ("Bronze", [0.80, 0.55, 0.33], 1.0, 0.35, 40.0, 0.54, 0.15),
        ("Chromium", [0.55, 0.56, 0.55], 1.0, 0.05, 0.0, 0.5, 0.0),
        ("Copper", [0.97, 0.74, 0.63], 1.0, 0.25, 45.0, 0.53, 0.12),
        ("Gold", [1.0, 0.77, 0.34], 1.0, 0.2, 50.0, 0.52, 0.1),
        ("Iron", [0.77, 0.78, 0.78], 1.0, 0.45, 30.0, 0.56, 0.2),
        ("Silver", [0.98, 0.97, 0.95], 1.0, 0.15, 50.0, 0.52, 0.1),
        ("Steel", [0.75, 0.76, 0.77], 1.0, 0.3, 50.0, 0.53, 0.15),
        ("Titanium", [0.76, 0.73, 0.69], 1.0, 0.4, 45.0, 0.53, 0.12),
        ("Plastic (glossy white)", [0.90, 0.90, 0.90], 0.0, 0.2, 0.0, 0.5, 0.0),
        ("Plastic (matte black)", [0.03, 0.03, 0.03], 0.0, 0.6, 40.0, 0.54, 0.1),
        ("Rubber", [0.05, 0.05, 0.05], 0.0, 0.85, 36.0, 0.6, 0.1),
        ("Leather", [0.25, 0.14, 0.09], 0.0, 0.6, 44.0, 0.64, 0.15),
        ("Ceramic", [0.92, 0.91, 0.88], 0.0, 0.1, 0.0, 0.5, 0.0),
        ("Marble", [0.85, 0.84, 0.82], 0.0, 0.25, 12.0, 0.54, 0.15),
        ("Wood (oak)", [0.55, 0.38, 0.21], 0.0, 0.65, 30.0, 0.6, 0.1),
        ("Skin", [0.80, 0.58, 0.47], 0.0, 0.5, 56.0, 0.57, 0.1),
    ];
    TABLE
        .iter()
        .map(|&(n, c, m, r, scale, bump, vary)| {
            let base = layer(n, c, Some(m), Some(r));
            let nodes = if scale == 0.0 {
                vec![base]
            } else {
                let rough = FillParams { roughness: Some((r + vary).min(1.0)), ..Default::default() };
                let breakup = Generator::new(GeneratorKind::Noise).with("scale", scale / 4.0).with("contrast", 0.5);
                vec![group(n, vec![base, relief("Relief", bump, scale), detail("Roughness variation", rough, breakup)])]
            };
            SmartMaterial { name: n.into(), nodes }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphBuilder, Op};
    use crate::Document;
    use std::collections::HashSet;

    fn ids(nodes: &[Node], out: &mut Vec<u64>) {
        for n in nodes {
            out.push(n.id);
            if let NodeKind::Group { children } = &n.kind {
                ids(children, out);
            }
        }
    }

    #[test]
    fn builtins_are_valid_and_roundtrip_through_json() {
        let all = builtin_smart_materials();
        assert!(all.len() >= 3);
        let mut names = HashSet::new();
        for m in &all {
            m.validate().unwrap_or_else(|e| panic!("{}: {e}", m.name));
            assert!(names.insert(m.name.clone()), "duplicate name {}", m.name);
            assert_eq!(&SmartMaterial::from_json(&m.to_json().unwrap()).unwrap(), m);
        }
    }

    #[test]
    fn every_builtin_uses_generated_masks_so_no_paint_data_is_needed() {
        for m in builtin_smart_materials() {
            if matches!(m.nodes[..], [Node { kind: NodeKind::Fill(_), .. }]) {
                continue; // plain single-layer materials have nothing to generate
            }
            let mut stack: Vec<&Node> = m.nodes.iter().collect();
            let mut generated = 0;
            while let Some(n) = stack.pop() {
                assert!(!matches!(n.kind, NodeKind::Paint), "{} contains a paint layer", m.name);
                generated += n.mask.as_ref().is_some_and(|k| k.generator.is_some()) as u32;
                if let NodeKind::Group { children } = &n.kind {
                    stack.extend(children.iter());
                }
            }
            assert!(generated >= 2, "{} should use generated masks", m.name);
        }
    }

    #[test]
    fn insertion_assigns_fresh_unique_ids_and_keeps_structure() {
        let mut doc = Document::with_default_material();
        let existing = doc.ids();
        let m = &builtin_smart_materials()[0];
        let top = doc.insert_nodes(None, None, m.nodes.clone());
        assert_eq!(top.len(), 1);
        let all = doc.ids();
        let unique: HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "ids must be unique");
        assert!(existing.iter().all(|e| all.contains(e)), "existing layers untouched");
        let group = doc.find(top[0]).unwrap();
        assert!(matches!(&group.kind, NodeKind::Group { children } if children.len() == 3));
        // Inserting the same material twice never collides.
        doc.insert_nodes(None, None, m.nodes.clone());
        let again: HashSet<_> = doc.ids().into_iter().collect();
        assert_eq!(again.len(), doc.ids().len());
    }

    #[test]
    fn insertion_respects_parent_and_index() {
        let mut doc = Document::default();
        let a = doc.add(None, None, "a", NodeKind::Paint);
        let g = doc.add(None, None, "g", NodeKind::Group { children: vec![] });
        let inner = doc.add(Some(g), None, "inner", NodeKind::Paint);
        let m = &builtin_smart_materials()[2];
        let ids = doc.insert_nodes(Some(g), Some(0), m.nodes.clone());
        assert_eq!(doc.locate(ids[0]), Some((Some(g), 0)), "inserted at the bottom of the group");
        assert_eq!(doc.locate(inner), Some((Some(g), 1)));
        assert_eq!(doc.locate(a), Some((None, 0)));
        // A non-group parent falls back to the top level instead of failing.
        let at_top = doc.insert_nodes(Some(a), None, m.nodes.clone());
        assert_eq!(doc.locate(at_top[0]).unwrap().0, None);
        let mut all = Vec::new();
        ids_of(&doc, &mut all);
        assert_eq!(all.len(), all.iter().collect::<HashSet<_>>().len());
    }

    fn ids_of(doc: &Document, out: &mut Vec<u64>) {
        ids(&doc.nodes, out);
    }

    #[test]
    fn a_subtree_can_be_extracted_and_reinserted_as_a_copy() {
        let mut doc = Document::default();
        let m = &builtin_smart_materials()[0];
        let id = doc.insert_nodes(None, None, m.nodes.clone())[0];
        let copy = doc.extract(id).unwrap();
        let saved = SmartMaterial { name: "copy".into(), nodes: vec![copy] };
        saved.validate().unwrap();
        let before = doc.ids().len();
        doc.insert_nodes(None, None, saved.nodes);
        assert_eq!(doc.ids().len(), before * 2);
        assert!(doc.extract(9999).is_none());
    }

    #[test]
    fn hostile_files_are_rejected() {
        let ok = builtin_smart_materials().remove(0);
        // not json / wrong shape
        assert!(SmartMaterial::from_json("nope").is_err());
        assert!(SmartMaterial::from_json("{}").is_err());
        // empty name, no layers
        assert!(SmartMaterial { name: " ".into(), ..ok.clone() }.validate().is_err());
        assert!(SmartMaterial { nodes: vec![], ..ok.clone() }.validate().is_err());
        // out-of-range opacity / fill values
        let mut bad = ok.clone();
        if let NodeKind::Group { children } = &mut bad.nodes[0].kind {
            children[0].opacity = 7.0;
        }
        assert!(bad.validate().unwrap_err().contains("opacity"));
        let mut bad = ok.clone();
        if let NodeKind::Group { children } = &mut bad.nodes[0].kind {
            children[0].kind = NodeKind::Fill(FillParams { base: Some([2.0, 0.0, 0.0]), ..Default::default() });
        }
        assert!(bad.validate().unwrap_err().contains("fill values"));
        let mut bad = ok.clone();
        if let NodeKind::Group { children } = &mut bad.nodes[0].kind {
            children[0].kind = NodeKind::Fill(FillParams { metallic: Some(f32::NAN), ..Default::default() });
        }
        assert!(bad.validate().is_err());
        // nesting depth and node count
        let mut deep = layer("leaf", [0.5; 3], None, None);
        for _ in 0..MAX_SMART_DEPTH + 2 {
            deep = group("g", vec![deep]);
        }
        assert!(SmartMaterial { name: "deep".into(), nodes: vec![deep] }.validate().unwrap_err().contains("deeply"));
        let many = (0..MAX_SMART_NODES + 1).map(|_| layer("l", [0.5; 3], None, None)).collect();
        assert!(SmartMaterial { name: "many".into(), nodes: many }.validate().unwrap_err().contains("more than"));
    }

    #[test]
    fn a_custom_generator_with_a_broken_graph_is_rejected() {
        let mut b = GraphBuilder::new();
        let x = b.add(Op::Invert, &[7]); // reads a node that does not exist
        let bad_graph = b.finish(x);
        let g = Generator::new(GeneratorKind::Custom { name: "broken".into(), graph: bad_graph });
        let m = SmartMaterial {
            name: "custom".into(),
            nodes: vec![layer("l", [0.5; 3], None, None).with_mask(g)],
        };
        assert!(m.validate().unwrap_err().contains("does not exist"));
        // And a generator with a NaN parameter.
        let nan = Generator::new(GeneratorKind::Noise).with("scale", f32::NAN);
        assert!(nan.validate().is_err());
        assert!(Generator::new(GeneratorKind::Noise).with("scale", 4.0).validate().is_ok());
    }

    #[test]
    fn old_masks_without_a_generator_still_deserialize() {
        let json = r#"{"enabled":true,"init_white":false}"#;
        let m: Mask = serde_json::from_str(json).unwrap();
        assert_eq!(m, Mask::new(false));
        assert!(m.generator.is_none());
    }
}
