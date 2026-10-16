//! Smart materials: a saved layer subtree (fills, groups, masks with generators) that can be
//! dropped onto any document. Everything in one is procedural, so no texture data is stored.
//!
//! The built-in ones below are original; users can add their own as JSON files.

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
        kind: NodeKind::Fill(FillParams { base: Some(base), metallic, roughness, ..Default::default() }),
    }
}

impl Node {
    fn with_mask(mut self, g: Generator) -> Self {
        self.mask = Some(Mask::generated(g));
        self
    }

    fn blended(mut self, blend: BlendMode, opacity: f32) -> Self {
        self.blend = blend;
        self.opacity = opacity;
        self
    }
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
        kind: NodeKind::Group { children },
    }
}

/// The smart materials that ship with the application.
pub fn builtin_smart_materials() -> Vec<SmartMaterial> {
    use GeneratorKind::*;
    vec![
        SmartMaterial {
            name: "Worn painted metal".into(),
            nodes: vec![group(
                "Worn painted metal",
                vec![
                    layer("Bare metal", [0.62, 0.62, 0.64], Some(1.0), Some(0.35)),
                    // The paint chips away where the edge wear is strong.
                    layer("Paint", [0.70, 0.12, 0.10], Some(0.0), Some(0.5))
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
                        .with_mask(Generator::new(Noise).with("scale", 6.0).with("contrast", 0.75)),
                    layer("Rust (edges)", [0.25, 0.10, 0.05], Some(0.0), Some(0.9))
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
                    layer("Dust", [0.62, 0.57, 0.48], None, Some(1.0)).with_mask(Generator::new(TopDust)),
                    layer("Dark crevices", [0.12, 0.11, 0.10], None, None)
                        .blended(BlendMode::Multiply, 0.8)
                        .with_mask(Generator::new(CreviceDirt)),
                ],
            )],
        },
    ]
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
