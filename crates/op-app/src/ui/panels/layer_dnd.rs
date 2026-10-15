//! Where a dragged layer lands: a pure function from the row under the pointer to a document slot.

use op_core::{Document, Node, NodeId, NodeKind};

/// One visible row of the layers panel, top of the stack first.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: NodeId,
    pub depth: usize,
    pub parent: Option<NodeId>,
    /// Position among its siblings (0 = bottom of that list).
    pub index: usize,
    pub is_group: bool,
    pub expanded: bool,
    pub child_count: usize,
}

/// Flatten the document into the rows that are on screen, honouring collapsed groups.
pub fn rows(nodes: &[Node], collapsed: &dyn Fn(NodeId) -> bool) -> Vec<Row> {
    fn walk(nodes: &[Node], parent: Option<NodeId>, depth: usize, collapsed: &dyn Fn(NodeId) -> bool, out: &mut Vec<Row>) {
        for (index, n) in nodes.iter().enumerate().rev() {
            let (is_group, child_count) = match &n.kind {
                NodeKind::Group { children } => (true, children.len()),
                _ => (false, 0),
            };
            let expanded = is_group && !collapsed(n.id);
            out.push(Row {
                id: n.id,
                depth,
                parent,
                index,
                is_group,
                expanded,
                child_count,
            });
            if expanded {
                if let NodeKind::Group { children } = &n.kind {
                    walk(children, Some(n.id), depth + 1, collapsed, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(nodes, None, 0, collapsed, &mut out);
    out
}

/// A drop destination: the parent list (`None` = top level) and the slot in it, in the terms of
/// [`Document::move_to`], plus the row edge the indicator is drawn at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drop {
    pub parent: Option<NodeId>,
    pub index: usize,
    pub indicator: Indicator,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Indicator {
    /// A line along the top or bottom edge of the hovered row, indented to `depth`.
    Above(usize),
    Below(usize),
    /// The hovered group row itself is the target.
    Into,
}

/// Decide where dropping on `row` lands. `frac` is the pointer's vertical position within the row
/// (0 = top edge, 1 = bottom edge). Group rows accept drops in their middle half, which puts the
/// layer on top of the group's contents; otherwise the top half is above the row and the bottom
/// half below it. Rows are drawn top-of-stack first, so "above" means a higher index.
/// Returns `None` when the drop is impossible (onto itself or into its own descendants) or would
/// not change anything.
pub fn drop_target(doc: &Document, dragged: NodeId, row: &Row, frac: f32) -> Option<Drop> {
    if doc.is_within(dragged, row.id) {
        return None;
    }
    let into = row.is_group && (0.25..0.75).contains(&frac);
    let target = if into {
        Drop {
            parent: Some(row.id),
            index: row.child_count,
            indicator: Indicator::Into,
        }
    } else if frac < 0.5 {
        // An expanded group's top edge is above the group, not above its first child.
        Drop {
            parent: row.parent,
            index: row.index + 1,
            indicator: Indicator::Above(row.depth),
        }
    } else if row.is_group && row.expanded {
        // Just under an open group's header is the top of its contents.
        Drop {
            parent: Some(row.id),
            index: row.child_count,
            indicator: Indicator::Below(row.depth + 1),
        }
    } else {
        Drop {
            parent: row.parent,
            index: row.index,
            indicator: Indicator::Below(row.depth),
        }
    };
    // Dropping a layer where it already is: the slot directly before or after itself.
    if let Some((p, i)) = doc.locate(dragged) {
        if p == target.parent && (target.index == i || target.index == i + 1) {
            return None;
        }
    }
    Some(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Top level, bottom to top: a, g{x, y}, b.   Rows on screen: b, g, y, x, a.
    fn fixture() -> (Document, [NodeId; 5]) {
        let mut d = Document::default();
        let a = d.add(None, None, "a", NodeKind::Paint);
        let g = d.add(None, None, "g", NodeKind::Group { children: vec![] });
        let x = d.add(Some(g), None, "x", NodeKind::Paint);
        let y = d.add(Some(g), None, "y", NodeKind::Paint);
        let b = d.add(None, None, "b", NodeKind::Paint);
        (d, [a, g, x, y, b])
    }

    fn row_of(d: &Document, id: NodeId) -> Row {
        rows(&d.nodes, &|_| false).into_iter().find(|r| r.id == id).unwrap()
    }

    #[test]
    fn rows_are_listed_top_first_and_respect_collapsing() {
        let (d, [a, g, x, y, b]) = fixture();
        let open: Vec<_> = rows(&d.nodes, &|_| false).iter().map(|r| (r.id, r.depth)).collect();
        assert_eq!(open, [(b, 0), (g, 0), (y, 1), (x, 1), (a, 0)]);
        let shut: Vec<_> = rows(&d.nodes, &|id| id == g).iter().map(|r| r.id).collect();
        assert_eq!(shut, [b, g, a]);
    }

    #[test]
    fn dropping_on_the_edges_of_a_plain_row() {
        let (d, [a, _g, _x, _y, b]) = fixture();
        // Dragging b (top) onto the upper half of a puts it above a; the lower half, below a.
        let ra = row_of(&d, a);
        assert_eq!(drop_target(&d, b, &ra, 0.2).map(|t| (t.parent, t.index)), Some((None, 1)));
        assert_eq!(drop_target(&d, b, &ra, 0.8).map(|t| (t.parent, t.index)), Some((None, 0)));
    }

    #[test]
    fn group_rows_accept_a_drop_into_their_middle() {
        let (d, [a, g, ..]) = fixture();
        let rg = row_of(&d, g);
        let t = drop_target(&d, a, &rg, 0.5).unwrap();
        assert_eq!((t.parent, t.index, t.indicator), (Some(g), 2, Indicator::Into));
        // Above an expanded group is above the group; below its header is inside it, on top.
        let t = drop_target(&d, a, &rg, 0.1).unwrap();
        assert_eq!((t.parent, t.index), (None, 2));
        let t = drop_target(&d, a, &rg, 0.9).unwrap();
        assert_eq!((t.parent, t.index, t.indicator), (Some(g), 2, Indicator::Below(1)));
    }

    #[test]
    fn a_drop_that_changes_nothing_or_cannot_happen_is_refused() {
        let (d, [a, g, x, _y, b]) = fixture();
        // Onto itself, and onto the slots directly beside itself.
        assert_eq!(drop_target(&d, a, &row_of(&d, a), 0.2), None);
        assert_eq!(drop_target(&d, a, &row_of(&d, a), 0.8), None);
        assert_eq!(drop_target(&d, b, &row_of(&d, g), 0.1), None, "b is already directly above g");
        // A group into its own contents.
        assert_eq!(drop_target(&d, g, &row_of(&d, x), 0.5), None);
        assert_eq!(drop_target(&d, g, &row_of(&d, g), 0.5), None);
    }

    /// Every drop the function offers must be one `move_to` accepts and that lands the layer
    /// exactly where the indicator says, for every dragged layer, row and pointer position.
    #[test]
    fn every_offered_drop_is_executable_and_lands_in_place() {
        let (d, ids) = fixture();
        for dragged in ids {
            for row in rows(&d.nodes, &|_| false) {
                for frac in [0.05, 0.3, 0.5, 0.7, 0.95] {
                    let Some(t) = drop_target(&d, dragged, &row, frac) else { continue };
                    let mut e = d.clone();
                    assert!(e.move_to(dragged, t.parent, t.index), "{dragged} -> {t:?}");
                    assert_eq!(e.ids().len(), d.ids().len());
                    assert_ne!(e, d, "an offered drop must change something");
                    let (p, i) = e.locate(dragged).unwrap();
                    assert_eq!(p, t.parent);
                    // The layer ends up adjacent to the row it was dropped on, on the drawn side.
                    let after = rows(&e.nodes, &|_| false);
                    let pos = |id| after.iter().position(|r| r.id == id).unwrap();
                    match t.indicator {
                        Indicator::Above(_) => assert_eq!(pos(dragged) + 1 + subtree_rows(&after, dragged), pos(row.id), "{dragged} above {}", row.id),
                        Indicator::Into => assert_eq!(pos(dragged), pos(row.id) + 1, "first child of {}", row.id),
                        Indicator::Below(_) if t.parent == Some(row.id) => assert_eq!(pos(dragged), pos(row.id) + 1),
                        Indicator::Below(_) => assert_eq!(pos(dragged), pos(row.id) + 1 + subtree_rows(&after, row.id)),
                    }
                    let _ = i;
                }
            }
        }
    }

    /// Rows belonging to a collapsed-or-open group below its header (0 for a plain layer).
    fn subtree_rows(rows: &[Row], id: NodeId) -> usize {
        let at = rows.iter().position(|r| r.id == id).unwrap();
        let depth = rows[at].depth;
        rows[at + 1..].iter().take_while(|r| r.depth > depth).count()
    }
}
