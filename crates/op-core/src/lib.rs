//! Document model for open-painter: the layer stack. No GPU, no UI.
//!
//! `nodes` lists are ordered **bottom to top** (index 0 is composited first).

pub mod blend;

pub type NodeId = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Add,
    Subtract,
    Darken,
    Lighten,
    Difference,
}

impl BlendMode {
    pub const ALL: [BlendMode; 9] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Difference,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Add => "Add",
            BlendMode::Subtract => "Subtract",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::Difference => "Difference",
        }
    }

    /// Index shared with the compositor shader.
    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|m| *m == self).unwrap() as u32
    }
}

/// A fill layer writes constants. `None` leaves that channel untouched.
#[derive(Clone, Debug, PartialEq)]
pub struct FillParams {
    /// sRGB color.
    pub base: Option<[f32; 3]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub enabled: bool,
    /// Initial content: white reveals everything, black hides everything.
    pub init_white: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Owns paintable base/metal/rough textures.
    Paint,
    Fill(FillParams),
    /// Children are composited in isolation, then blended as one layer.
    Group { children: Vec<Node> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub mask: Option<Mask>,
    pub kind: NodeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub nodes: Vec<Node>,
    next_id: NodeId,
}

impl Default for Document {
    fn default() -> Self {
        Self { nodes: Vec::new(), next_id: 1 }
    }
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn find_in(nodes: &[Node], id: NodeId) -> Option<&Node> {
    for n in nodes {
        if n.id == id {
            return Some(n);
        }
        if let NodeKind::Group { children } = &n.kind {
            if let Some(f) = find_in(children, id) {
                return Some(f);
            }
        }
    }
    None
}

fn find_in_mut(nodes: &mut [Node], id: NodeId) -> Option<&mut Node> {
    for n in nodes {
        if n.id == id {
            return Some(n);
        }
        if let NodeKind::Group { children } = &mut n.kind {
            if let Some(f) = find_in_mut(children, id) {
                return Some(f);
            }
        }
    }
    None
}

/// The sibling list that contains `id`, plus its index in it.
fn list_of_mut(nodes: &mut Vec<Node>, id: NodeId) -> Option<(&mut Vec<Node>, usize)> {
    if let Some(i) = nodes.iter().position(|n| n.id == id) {
        return Some((nodes, i));
    }
    for n in nodes.iter_mut() {
        if let NodeKind::Group { children } = &mut n.kind {
            if let Some(r) = list_of_mut(children, id) {
                return Some(r);
            }
        }
    }
    None
}

fn collect_ids(nodes: &[Node], out: &mut Vec<NodeId>) {
    for n in nodes {
        out.push(n.id);
        if let NodeKind::Group { children } = &n.kind {
            collect_ids(children, out);
        }
    }
}

impl Document {
    /// A fresh document: one orange fill layer, like a default material.
    pub fn with_default_material() -> Self {
        let mut d = Self::default();
        d.add(
            None,
            None,
            "Base material",
            NodeKind::Fill(FillParams { base: Some([0.80, 0.55, 0.30]), metallic: Some(0.0), roughness: Some(0.45) }),
        );
        d
    }

    pub fn find(&self, id: NodeId) -> Option<&Node> {
        find_in(&self.nodes, id)
    }

    pub fn find_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        find_in_mut(&mut self.nodes, id)
    }

    pub fn ids(&self) -> Vec<NodeId> {
        let mut v = Vec::new();
        collect_ids(&self.nodes, &mut v);
        v
    }

    /// Parent group (None = top level) and index within its sibling list.
    pub fn locate(&self, id: NodeId) -> Option<(Option<NodeId>, usize)> {
        fn go(nodes: &[Node], parent: Option<NodeId>, id: NodeId) -> Option<(Option<NodeId>, usize)> {
            for (i, n) in nodes.iter().enumerate() {
                if n.id == id {
                    return Some((parent, i));
                }
                if let NodeKind::Group { children } = &n.kind {
                    if let Some(r) = go(children, Some(n.id), id) {
                        return Some(r);
                    }
                }
            }
            None
        }
        go(&self.nodes, None, id)
    }

    /// Add a node into `parent` (None = top level) at `index` (None = on top).
    pub fn add(&mut self, parent: Option<NodeId>, index: Option<usize>, name: &str, kind: NodeKind) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = Node {
            id,
            name: name.to_string(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            mask: None,
            kind,
        };
        let list = match parent.and_then(|p| self.find_mut(p)) {
            Some(Node { kind: NodeKind::Group { children }, .. }) => children,
            _ => &mut self.nodes,
        };
        let at = index.unwrap_or(list.len()).min(list.len());
        list.insert(at, node);
        id
    }

    pub fn remove(&mut self, id: NodeId) -> Option<Node> {
        let (list, i) = list_of_mut(&mut self.nodes, id)?;
        Some(list.remove(i))
    }

    /// Move one step toward the top of its sibling list.
    pub fn move_up(&mut self, id: NodeId) -> bool {
        match list_of_mut(&mut self.nodes, id) {
            Some((list, i)) if i + 1 < list.len() => {
                list.swap(i, i + 1);
                true
            }
            _ => false,
        }
    }

    pub fn move_down(&mut self, id: NodeId) -> bool {
        match list_of_mut(&mut self.nodes, id) {
            Some((list, i)) if i > 0 => {
                list.swap(i, i - 1);
                true
            }
            _ => false,
        }
    }

    /// Move into the group directly above it in the sibling list (as that group's bottom child).
    pub fn indent(&mut self, id: NodeId) -> bool {
        let Some((list, i)) = list_of_mut(&mut self.nodes, id) else { return false };
        if i + 1 >= list.len() || !matches!(list[i + 1].kind, NodeKind::Group { .. }) {
            return false;
        }
        let node = list.remove(i);
        if let NodeKind::Group { children } = &mut list[i].kind {
            children.insert(0, node);
        }
        true
    }

    /// Move out of its parent group, to just below that group.
    pub fn outdent(&mut self, id: NodeId) -> bool {
        let Some((Some(parent), _)) = self.locate(id) else { return false };
        let Some(node) = self.remove(id) else { return false };
        let (list, pi) = list_of_mut(&mut self.nodes, parent).expect("parent exists");
        list.insert(pi, node);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(nodes: &[Node]) -> Vec<&str> {
        nodes.iter().map(|n| n.name.as_str()).collect()
    }

    fn doc_abc() -> (Document, NodeId, NodeId, NodeId) {
        let mut d = Document::default();
        let a = d.add(None, None, "a", NodeKind::Paint);
        let b = d.add(None, None, "b", NodeKind::Paint);
        let c = d.add(None, None, "c", NodeKind::Group { children: vec![] });
        (d, a, b, c)
    }

    #[test]
    fn add_orders_bottom_to_top_and_ids_are_unique() {
        let (d, a, b, c) = doc_abc();
        assert_eq!(names(&d.nodes), ["a", "b", "c"]);
        assert!(a != b && b != c && a != c);
        assert_eq!(d.ids().len(), 3);
    }

    #[test]
    fn move_up_down_stay_within_bounds() {
        let (mut d, a, _b, c) = doc_abc();
        assert!(d.move_up(a));
        assert_eq!(names(&d.nodes), ["b", "a", "c"]);
        assert!(!d.move_up(c), "already on top");
        assert!(d.move_down(c));
        assert_eq!(names(&d.nodes), ["b", "c", "a"]);
        assert!(d.move_down(a) && !d.move_down(d.nodes[0].id));
    }

    #[test]
    fn indent_requires_group_above_and_outdent_reverses_it() {
        let (mut d, a, b, c) = doc_abc();
        assert!(!d.indent(a), "b is not a group");
        assert!(d.indent(b), "c (group) is directly above b");
        assert_eq!(names(&d.nodes), ["a", "c"]);
        assert_eq!(d.locate(b), Some((Some(c), 0)));
        assert!(d.outdent(b));
        assert_eq!(names(&d.nodes), ["a", "b", "c"]);
        assert!(!d.outdent(b), "already top level");
    }

    #[test]
    fn nested_find_remove_and_locate() {
        let (mut d, _a, b, c) = doc_abc();
        d.indent(b);
        let inner = d.add(Some(c), None, "inner", NodeKind::Paint);
        assert_eq!(d.find(inner).unwrap().name, "inner");
        assert_eq!(d.locate(inner), Some((Some(c), 1)));
        assert_eq!(d.remove(inner).unwrap().id, inner);
        assert!(d.find(inner).is_none());
        assert_eq!(d.ids().len(), 3);
    }

    #[test]
    fn clone_equality_detects_edits() {
        let (mut d, a, ..) = doc_abc();
        let before = d.clone();
        assert_eq!(d, before);
        d.find_mut(a).unwrap().opacity = 0.5;
        assert_ne!(d, before);
    }
}
