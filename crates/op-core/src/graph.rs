//! Procedural graph: a validated DAG of scalar-field nodes (noise, baked maps, levels, blur,
//! blends ...) that produces one grayscale image in UV space. This is the data model; the
//! evaluator lives in `op-proc`.
//!
//! Graphs can come from files, so everything is validated before use: arity, references,
//! cycles, parameter ranges and size limits.

use crate::MeshMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub type GraphNodeId = u32;

pub const MAX_GRAPH_NODES: usize = 256;
pub const MAX_NOISE_OCTAVES: u32 = 10;
pub const MAX_BLUR: f32 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseKind {
    Value,
    Perlin,
    /// Distance to the nearest feature point.
    Cellular,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendOp {
    Add,
    Multiply,
    Subtract,
    Min,
    Max,
    Screen,
    Overlay,
    Difference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // --- sources (no inputs)
    Const { value: f32 },
    /// A baked mesh map. For 3-channel maps `comp` (0..=2) picks the channel; scalar maps ignore it.
    Baked { map: MeshMap, comp: u8 },
    /// Fractal noise in 0..1. Evaluated in object space (from the baked position map) unless
    /// `uv_space` is set, so patterns do not stretch or break at UV seams.
    Noise { kind: NoiseKind, scale: f32, octaves: u32, persistence: f32, seed: u32, uv_space: bool },
    /// The u (axis 0) or v (axis 1) texture coordinate.
    Uv { axis: u8 },
    // --- one input
    Invert,
    Levels { in_lo: f32, in_hi: f32, gamma: f32, out_lo: f32, out_hi: f32 },
    /// Scale contrast about 0.5. `amount` in -1..=1 (0 = unchanged).
    Contrast { amount: f32 },
    /// Smooth step around `level`; `softness` is the transition width.
    Threshold { level: f32, softness: f32 },
    /// Box-blur approximation of a Gaussian. `radius` is a fraction of the texture size.
    Blur { radius: f32 },
    // --- two inputs
    /// `lerp(a, op(a, b), opacity)`
    Blend { op: BlendOp, opacity: f32 },
    // --- three inputs
    /// `lerp(a, b, t)` with `t` taken per texel from the third input.
    Lerp,
}

impl Op {
    pub fn arity(&self) -> usize {
        match self {
            Op::Const { .. } | Op::Baked { .. } | Op::Noise { .. } | Op::Uv { .. } => 0,
            Op::Invert | Op::Levels { .. } | Op::Contrast { .. } | Op::Threshold { .. } | Op::Blur { .. } => 1,
            Op::Blend { .. } => 2,
            Op::Lerp => 3,
        }
    }

    fn validate(&self) -> Result<(), String> {
        let finite = |v: f32, what: &str| if v.is_finite() { Ok(()) } else { Err(format!("{what} is not a finite number")) };
        match self {
            Op::Const { value } => finite(*value, "constant"),
            Op::Baked { comp, .. } if *comp > 2 => Err("baked map component must be 0, 1 or 2".into()),
            Op::Baked { .. } => Ok(()),
            Op::Noise { scale, octaves, persistence, .. } => {
                finite(*scale, "noise scale")?;
                finite(*persistence, "noise persistence")?;
                if !(*scale > 0.0 && *scale <= 1000.0) {
                    return Err("noise scale must be in (0, 1000]".into());
                }
                if !(1..=MAX_NOISE_OCTAVES).contains(octaves) {
                    return Err(format!("noise octaves must be 1..={MAX_NOISE_OCTAVES}"));
                }
                if !(0.0..=1.0).contains(persistence) {
                    return Err("noise persistence must be in 0..=1".into());
                }
                Ok(())
            }
            Op::Uv { axis } if *axis > 1 => Err("uv axis must be 0 or 1".into()),
            Op::Uv { .. } | Op::Invert | Op::Lerp => Ok(()),
            Op::Levels { in_lo, in_hi, gamma, out_lo, out_hi } => {
                for (v, n) in [(in_lo, "in_lo"), (in_hi, "in_hi"), (out_lo, "out_lo"), (out_hi, "out_hi")] {
                    finite(*v, n)?;
                }
                finite(*gamma, "gamma")?;
                if !(*gamma > 0.0 && *gamma <= 10.0) {
                    return Err("gamma must be in (0, 10]".into());
                }
                Ok(())
            }
            Op::Contrast { amount } => {
                finite(*amount, "contrast")?;
                if (-1.0..=1.0).contains(amount) { Ok(()) } else { Err("contrast must be in -1..=1".into()) }
            }
            Op::Threshold { level, softness } => {
                finite(*level, "threshold level")?;
                finite(*softness, "threshold softness")?;
                if *softness >= 0.0 { Ok(()) } else { Err("threshold softness must not be negative".into()) }
            }
            Op::Blur { radius } => {
                finite(*radius, "blur radius")?;
                if (0.0..=MAX_BLUR).contains(radius) { Ok(()) } else { Err(format!("blur radius must be in 0..={MAX_BLUR}")) }
            }
            Op::Blend { opacity, .. } => {
                finite(*opacity, "blend opacity")?;
                if (0.0..=1.0).contains(opacity) { Ok(()) } else { Err("blend opacity must be in 0..=1".into()) }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: GraphNodeId,
    pub op: Op,
    /// Ids of the nodes feeding this one, in operand order.
    pub inputs: Vec<GraphNodeId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub output: GraphNodeId,
}

impl Graph {
    pub fn node(&self, id: GraphNodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Check the whole graph. Must pass before the graph is evaluated.
    pub fn validate(&self) -> Result<(), String> {
        if self.nodes.is_empty() || self.nodes.len() > MAX_GRAPH_NODES {
            return Err(format!("a graph needs 1 to {MAX_GRAPH_NODES} nodes, this has {}", self.nodes.len()));
        }
        let mut ids = HashSet::new();
        for n in &self.nodes {
            if !ids.insert(n.id) {
                return Err(format!("node id {} is used twice", n.id));
            }
        }
        for n in &self.nodes {
            if n.inputs.len() != n.op.arity() {
                return Err(format!("node {} ({:?}) needs {} input(s), has {}", n.id, n.op, n.op.arity(), n.inputs.len()));
            }
            if let Some(missing) = n.inputs.iter().find(|i| !ids.contains(i)) {
                return Err(format!("node {} reads node {missing}, which does not exist", n.id));
            }
            n.op.validate().map_err(|e| format!("node {}: {e}", n.id))?;
        }
        if !ids.contains(&self.output) {
            return Err(format!("the output node {} does not exist", self.output));
        }
        self.order().map(|_| ())
    }

    /// Node indices in evaluation order (inputs before the nodes that use them), restricted to
    /// nodes the output depends on. Fails on cycles.
    pub fn order(&self) -> Result<Vec<usize>, String> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Visiting,
            Done,
        }
        let index = |id: GraphNodeId| self.nodes.iter().position(|n| n.id == id);
        let mut marks = vec![None; self.nodes.len()];
        let mut order = Vec::new();
        // Iterative DFS: graphs may be adversarial, so no recursion.
        let root = index(self.output).ok_or_else(|| format!("the output node {} does not exist", self.output))?;
        let mut stack = vec![(root, 0usize)];
        marks[root] = Some(Mark::Visiting);
        while let Some((i, next)) = stack.pop() {
            let inputs = &self.nodes[i].inputs;
            if next < inputs.len() {
                stack.push((i, next + 1));
                let j = index(inputs[next]).ok_or_else(|| format!("node {} reads a missing node", self.nodes[i].id))?;
                match marks[j] {
                    Some(Mark::Visiting) => return Err(format!("the graph has a cycle through node {}", self.nodes[j].id)),
                    Some(Mark::Done) => {}
                    None => {
                        marks[j] = Some(Mark::Visiting);
                        stack.push((j, 0));
                    }
                }
            } else {
                marks[i] = Some(Mark::Done);
                order.push(i);
            }
        }
        Ok(order)
    }
}

/// Small helper to assemble graphs in code (templates, tests).
#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<GraphNode>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, op: Op, inputs: &[GraphNodeId]) -> GraphNodeId {
        let id = self.nodes.len() as GraphNodeId;
        self.nodes.push(GraphNode { id, op, inputs: inputs.to_vec() });
        id
    }

    pub fn finish(self, output: GraphNodeId) -> Graph {
        Graph { nodes: self.nodes, output }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(v: f32) -> Op {
        Op::Const { value: v }
    }

    #[test]
    fn a_wellformed_graph_validates_and_orders_inputs_first() {
        let mut b = GraphBuilder::new();
        let x = b.add(c(0.2), &[]);
        let y = b.add(c(0.7), &[]);
        let sum = b.add(Op::Blend { op: BlendOp::Add, opacity: 1.0 }, &[x, y]);
        let out = b.add(Op::Invert, &[sum]);
        let g = b.finish(out);
        g.validate().unwrap();
        let order = g.order().unwrap();
        let pos = |id| order.iter().position(|&i| g.nodes[i].id == id).unwrap();
        assert!(pos(x) < pos(sum) && pos(y) < pos(sum) && pos(sum) < pos(out));
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn nodes_the_output_does_not_use_are_not_evaluated() {
        let mut b = GraphBuilder::new();
        let used = b.add(c(1.0), &[]);
        let _unused = b.add(Op::Noise { kind: NoiseKind::Perlin, scale: 4.0, octaves: 3, persistence: 0.5, seed: 1, uv_space: false }, &[]);
        let g = b.finish(used);
        assert_eq!(g.order().unwrap().len(), 1);
    }

    #[test]
    fn structural_errors_are_rejected_with_a_reason() {
        let bad = |g: Graph| g.validate().unwrap_err();
        // wrong arity
        let g = Graph { nodes: vec![GraphNode { id: 0, op: Op::Invert, inputs: vec![] }], output: 0 };
        assert!(bad(g).contains("needs 1 input"));
        // missing input
        let g = Graph { nodes: vec![GraphNode { id: 0, op: Op::Invert, inputs: vec![9] }], output: 0 };
        assert!(bad(g).contains("does not exist"));
        // duplicate ids
        let g = Graph { nodes: vec![GraphNode { id: 1, op: c(0.0), inputs: vec![] }, GraphNode { id: 1, op: c(1.0), inputs: vec![] }], output: 1 };
        assert!(bad(g).contains("twice"));
        // missing output
        let g = Graph { nodes: vec![GraphNode { id: 0, op: c(0.0), inputs: vec![] }], output: 5 };
        assert!(bad(g).contains("output"));
        // empty and oversized
        assert!(Graph { nodes: vec![], output: 0 }.validate().is_err());
        let many: Vec<_> = (0..MAX_GRAPH_NODES as u32 + 1).map(|id| GraphNode { id, op: c(0.0), inputs: vec![] }).collect();
        assert!(Graph { nodes: many, output: 0 }.validate().is_err());
    }

    #[test]
    fn cycles_are_detected_including_self_loops() {
        let two = Graph {
            nodes: vec![
                GraphNode { id: 0, op: Op::Invert, inputs: vec![1] },
                GraphNode { id: 1, op: Op::Invert, inputs: vec![0] },
            ],
            output: 0,
        };
        assert!(two.validate().unwrap_err().contains("cycle"));
        let own = Graph { nodes: vec![GraphNode { id: 0, op: Op::Invert, inputs: vec![0] }], output: 0 };
        assert!(own.validate().unwrap_err().contains("cycle"));
    }

    #[test]
    fn shared_subgraphs_are_not_mistaken_for_cycles() {
        // diamond: x feeds both arms of a blend
        let mut b = GraphBuilder::new();
        let x = b.add(c(0.5), &[]);
        let a = b.add(Op::Invert, &[x]);
        let m = b.add(Op::Blur { radius: 0.0 }, &[x]);
        let out = b.add(Op::Blend { op: BlendOp::Max, opacity: 1.0 }, &[a, m]);
        let g = b.finish(out);
        g.validate().unwrap();
        assert_eq!(g.order().unwrap().len(), 4, "x is listed once");
    }

    #[test]
    fn parameter_ranges_are_enforced() {
        // A constant feeding every input of the op under test.
        let single = |op: Op| {
            let arity = op.arity();
            let nodes = vec![
                GraphNode { id: 0, op: c(0.0), inputs: vec![] },
                GraphNode { id: 1, op, inputs: vec![0; arity] },
            ];
            Graph { nodes, output: 1 }
        };
        let noise = |scale, octaves, persistence| Op::Noise { kind: NoiseKind::Value, scale, octaves, persistence, seed: 0, uv_space: true };
        assert!(single(noise(4.0, 4, 0.5)).validate().is_ok());
        assert!(single(noise(0.0, 4, 0.5)).validate().is_err(), "zero scale");
        assert!(single(noise(f32::NAN, 4, 0.5)).validate().is_err());
        assert!(single(noise(4.0, 0, 0.5)).validate().is_err());
        assert!(single(noise(4.0, 99, 0.5)).validate().is_err());
        assert!(single(noise(4.0, 4, 1.5)).validate().is_err());
        assert!(single(Op::Blur { radius: 0.5 }).validate().is_err());
        assert!(single(Op::Blur { radius: -0.1 }).validate().is_err());
        assert!(single(Op::Contrast { amount: 2.0 }).validate().is_err());
        assert!(single(Op::Threshold { level: 0.5, softness: -1.0 }).validate().is_err());
        assert!(single(Op::Levels { in_lo: 0.0, in_hi: 1.0, gamma: 0.0, out_lo: 0.0, out_hi: 1.0 }).validate().is_err());
        assert!(single(Op::Levels { in_lo: f32::INFINITY, in_hi: 1.0, gamma: 1.0, out_lo: 0.0, out_hi: 1.0 }).validate().is_err());
        assert!(single(Op::Baked { map: MeshMap::Curvature, comp: 3 }).validate().is_err());
        assert!(single(Op::Uv { axis: 2 }).validate().is_err());
        assert!(single(Op::Blend { op: BlendOp::Add, opacity: 1.5 }).validate().is_err());
        assert!(single(Op::Blend { op: BlendOp::Add, opacity: 0.5 }).validate().is_ok());
        assert!(single(Op::Lerp).validate().is_ok(), "three-input op with a shared input");
    }

    #[test]
    fn graphs_roundtrip_through_json() {
        let mut b = GraphBuilder::new();
        let n = b.add(Op::Noise { kind: NoiseKind::Cellular, scale: 7.0, octaves: 3, persistence: 0.4, seed: 9, uv_space: false }, &[]);
        let out = b.add(Op::Levels { in_lo: 0.1, in_hi: 0.9, gamma: 1.2, out_lo: 0.0, out_hi: 1.0 }, &[n]);
        let g = b.finish(out);
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Graph>(&json).unwrap(), g);
    }

    #[test]
    fn a_long_chain_validates_without_recursion_trouble() {
        let mut b = GraphBuilder::new();
        let mut prev = b.add(c(0.5), &[]);
        for _ in 0..MAX_GRAPH_NODES - 1 {
            prev = b.add(Op::Invert, &[prev]);
        }
        let g = b.finish(prev);
        g.validate().unwrap();
    }
}
