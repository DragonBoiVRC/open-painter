//! UDIM tiles: UV space is a grid of unit squares, numbered 1001 + u + 10 * v. A texture set whose
//! UVs cover several tiles has one texture per tile. A triangle belongs to the tile that contains
//! its UV centroid, so a triangle that straddles a border is drawn in one tile only.

use crate::{Mesh, SetRange, Vertex};

/// Largest tile column and row (UDIM numbers run 1001 to 1099 in rows of ten).
pub const MAX_TILE_U: u32 = 9;
pub const MAX_TILE_V: u32 = 99;

/// One UV tile. Ordered by UDIM number (row first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub u: u32,
    pub v: u32,
}

impl Ord for Tile {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.v, self.u).cmp(&(other.v, other.u))
    }
}

impl PartialOrd for Tile {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Tile {
    pub const FIRST: Tile = Tile { u: 0, v: 0 };

    /// The UDIM number: 1001 for the first tile.
    pub fn udim(self) -> u32 {
        1001 + self.u + 10 * self.v
    }

    pub fn from_udim(n: u32) -> Option<Tile> {
        let i = n.checked_sub(1001)?;
        let t = Tile { u: i % 10, v: i / 10 };
        (t.v <= MAX_TILE_V).then_some(t)
    }

    /// The tile containing (u, v). Coordinates below zero or beyond the grid land in the nearest tile.
    pub fn containing(u: f32, v: f32) -> Tile {
        let cell = |x: f32, max: u32| if x.is_finite() { (x.floor().max(0.0) as u32).min(max) } else { 0 };
        Tile { u: cell(u, MAX_TILE_U), v: cell(v, MAX_TILE_V) }
    }
}

/// A run of triangles of one set that all lie in one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRange {
    pub set: usize,
    pub tile: Tile,
    pub first_index: u32,
    pub index_count: u32,
}

impl Mesh {
    fn tile_of(&self, tri: &[u32]) -> Tile {
        let uv = |i: u32| self.vertices[i as usize].uv;
        let (a, b, c) = (uv(tri[0]), uv(tri[1]), uv(tri[2]));
        Tile::containing((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
    }

    /// The tiles a set's triangles occupy, in UDIM order. A set with no triangles has the first tile.
    pub fn set_tiles(&self, set: usize) -> Vec<Tile> {
        let ranges = self.set_ranges();
        let Some(r) = ranges.get(set) else { return vec![Tile::FIRST] };
        let tris = &self.indices[r.first_index as usize..(r.first_index + r.index_count) as usize];
        let mut tiles: Vec<Tile> = tris.chunks_exact(3).map(|t| self.tile_of(t)).collect();
        tiles.sort_unstable();
        tiles.dedup();
        if tiles.is_empty() {
            tiles.push(Tile::FIRST);
        }
        tiles
    }

    /// Convert UVs from V-up (OBJ, FBX, Substance, Unity) to V-down (image rows, glTF): inside each
    /// triangle's own tile row, `v` becomes `2k + 1 - v`, so UDIM tile numbers do not change. A vertex
    /// shared by triangles in different tile rows is copied, one per row.
    pub fn flip_v_per_tile(&mut self) {
        let mut copies: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        let mut vertices = std::mem::take(&mut self.vertices);
        for t in 0..self.indices.len() / 3 {
            let tri = [self.indices[t * 3], self.indices[t * 3 + 1], self.indices[t * 3 + 2]];
            let centroid = tri.iter().map(|i| vertices[*i as usize].uv[1]).sum::<f32>() / 3.0;
            let row = Tile::containing(0.0, centroid).v;
            for (k, i) in tri.into_iter().enumerate() {
                let new = *copies.entry((i, row)).or_insert_with(|| {
                    let mut v = vertices[i as usize];
                    v.uv[1] = 2.0 * row as f32 + 1.0 - v.uv[1];
                    vertices.push(v);
                    (vertices.len() - 1) as u32
                });
                self.indices[t * 3 + k] = new;
            }
        }
        // Indices now point at the flipped copies only; keep just those (in first-use order).
        let mut remap = vec![u32::MAX; vertices.len()];
        let mut kept = Vec::with_capacity(vertices.len());
        for idx in self.indices.iter_mut() {
            let r = &mut remap[*idx as usize];
            if *r == u32::MAX {
                *r = kept.len() as u32;
                kept.push(vertices[*idx as usize]);
            }
            *idx = *r;
        }
        self.vertices = kept;
    }

    /// Order each set's triangles by tile (keeping the order inside a tile), so every tile of a set is
    /// one contiguous run of the index buffer. Sets stay where they are.
    pub fn partition_tiles(&mut self) {
        for r in self.set_ranges() {
            let (a, b) = (r.first_index as usize, (r.first_index + r.index_count) as usize);
            let mut tris: Vec<([u32; 3], Tile)> = self.indices[a..b].chunks_exact(3).map(|t| ([t[0], t[1], t[2]], self.tile_of(t))).collect();
            tris.sort_by_key(|(_, tile)| *tile); // stable
            for (slot, (t, _)) in self.indices[a..b].chunks_exact_mut(3).zip(tris) {
                slot.copy_from_slice(&t);
            }
        }
    }

    /// The tile runs of every set, in set order then UDIM order. The mesh must have been through
    /// [`Mesh::partition_tiles`]; otherwise a tile can appear in several runs.
    pub fn tile_ranges(&self) -> Vec<TileRange> {
        let mut out = Vec::new();
        for (set, r) in self.set_ranges().iter().enumerate() {
            let mut cur: Option<TileRange> = None;
            for k in 0..(r.index_count / 3) {
                let first = r.first_index + k * 3;
                let tile = self.tile_of(&self.indices[first as usize..first as usize + 3]);
                match &mut cur {
                    Some(c) if c.tile == tile => c.index_count += 3,
                    _ => {
                        out.extend(cur.take());
                        cur = Some(TileRange { set, tile, first_index: first, index_count: 3 });
                    }
                }
            }
            out.extend(cur);
            if !out.iter().any(|t| t.set == set) {
                out.push(TileRange { set, tile: Tile::FIRST, first_index: r.first_index, index_count: 0 });
            }
        }
        out
    }

    /// The paint units of the mesh: with `udim` one per tile of each set (see [`Mesh::tile_ranges`]),
    /// without it one per set covering all its triangles in the first tile.
    pub fn unit_ranges(&self, udim: bool) -> Vec<TileRange> {
        if udim {
            return self.tile_ranges();
        }
        self.set_ranges()
            .iter()
            .enumerate()
            .map(|(set, r)| TileRange { set, tile: Tile::FIRST, first_index: r.first_index, index_count: r.index_count })
            .collect()
    }

    /// The triangles of one set that lie in `tile`, with vertices renumbered and UVs shifted so the
    /// tile becomes the unit square. This is the geometry a per-tile bake or export works on.
    pub fn extract_tile(&self, set: usize, tile: Tile) -> Option<Mesh> {
        let ranges = self.set_ranges();
        let r = ranges.get(set)?;
        let tris = &self.indices[r.first_index as usize..(r.first_index + r.index_count) as usize];
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices = Vec::new();
        for t in tris.chunks_exact(3).filter(|t| self.tile_of(t) == tile) {
            for &i in t {
                let slot = &mut remap[i as usize];
                if *slot == u32::MAX {
                    *slot = vertices.len() as u32;
                    let mut v = self.vertices[i as usize];
                    v.uv = [v.uv[0] - tile.u as f32, v.uv[1] - tile.v as f32];
                    vertices.push(v);
                }
                indices.push(*slot);
            }
        }
        let count = indices.len() as u32;
        Some(Mesh { vertices, indices, sets: vec![SetRange { name: r.name.clone(), first_index: 0, index_count: count }] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flipping_v_keeps_each_triangle_in_its_tile_and_copies_shared_vertices() {
        let v = |u: f32, w: f32| Vertex { position: [0.0; 3], normal: [0.0, 0.0, 1.0], uv: [u, w] };
        // Triangle A sits in row 0; triangle B in row 1; they share vertex 1, which lies on v = 1.0.
        let mut m = Mesh {
            vertices: vec![v(0.1, 0.2), v(0.5, 1.0), v(0.9, 0.2), v(0.1, 1.8), v(0.9, 1.8)],
            indices: vec![0, 1, 2, 1, 3, 4],
            ..Default::default()
        };
        m.flip_v_per_tile();
        let uv = |k: usize| m.vertices[m.indices[k] as usize].uv;
        let near = |a: f32, b: f32| (a - b).abs() < 1e-6;
        // Row 0: v -> 1 - v. Row 1: v -> 3 - v (so 1.8 -> 1.2 and the shared 1.0 -> 2.0).
        assert!(near(uv(0)[1], 0.8) && near(uv(1)[1], 0.0) && near(uv(2)[1], 0.8));
        assert!(near(uv(3)[1], 2.0) && near(uv(4)[1], 1.2) && near(uv(5)[1], 1.2));
        assert_eq!(m.tile_of(&m.indices[0..3]).udim(), 1001);
        assert_eq!(m.tile_of(&m.indices[3..6]).udim(), 1011);
        assert_eq!(m.vertices.len(), 6, "the shared vertex exists once per row");
        // Flipping an ordinary single-tile mesh is just 1 - v with no extra vertices.
        let mut n = Mesh { vertices: vec![v(0.0, 0.25), v(1.0, 0.75), v(0.5, 0.5)], indices: vec![0, 1, 2], ..Default::default() };
        n.flip_v_per_tile();
        assert_eq!(n.vertices.len(), 3);
        assert_eq!(n.vertices.iter().map(|x| x.uv[1]).collect::<Vec<_>>(), vec![0.75, 0.25, 0.5]);
    }

    /// Two sets; set 0 has quads in tiles 1001, 1002 and 1011, set 1 only in 1001.
    fn mesh() -> Mesh {
        let quad = |u0: f32, v0: f32, verts: &mut Vec<Vertex>, idx: &mut Vec<u32>| {
            let base = verts.len() as u32;
            for (u, v) in [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)] {
                verts.push(Vertex { position: [u, v, 0.0], normal: [0.0, 0.0, 1.0], uv: [u0 + u, v0 + v] });
            }
            idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        };
        let (mut v, mut i) = (Vec::new(), Vec::new());
        // Deliberately out of tile order: 1011, 1001, 1002, 1001.
        quad(0.0, 1.0, &mut v, &mut i);
        quad(0.0, 0.0, &mut v, &mut i);
        quad(1.0, 0.0, &mut v, &mut i);
        quad(0.0, 0.0, &mut v, &mut i);
        let split = i.len() as u32;
        quad(0.0, 0.0, &mut v, &mut i);
        let total = i.len() as u32;
        Mesh {
            vertices: v,
            indices: i,
            sets: vec![
                SetRange { name: "A".into(), first_index: 0, index_count: split },
                SetRange { name: "B".into(), first_index: split, index_count: total - split },
            ],
        }
    }

    #[test]
    fn without_udim_every_set_is_one_unit_in_the_first_tile() {
        let m = mesh();
        let units = m.unit_ranges(false);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.tile == Tile::FIRST));
        assert_eq!(units[0].index_count, m.set_ranges()[0].index_count);
        let mut m = m;
        m.partition_tiles();
        assert_eq!(m.unit_ranges(true).len(), 4, "with udim: three tiles in set 0 plus one in set 1");
    }

    #[test]
    fn udim_numbers_round_trip() {
        assert_eq!(Tile::FIRST.udim(), 1001);
        assert_eq!(Tile { u: 1, v: 0 }.udim(), 1002);
        assert_eq!(Tile { u: 0, v: 1 }.udim(), 1011);
        assert_eq!(Tile::from_udim(1024), Some(Tile { u: 3, v: 2 }));
        assert_eq!(Tile::from_udim(1000), None);
        assert_eq!(Tile::from_udim(99999), None);
        for n in [1001, 1009, 1010, 1011, 1099] {
            assert_eq!(Tile::from_udim(n).map(Tile::udim), Some(n));
        }
        assert!(Tile { u: 9, v: 0 } < Tile { u: 0, v: 1 }, "ordered by row first, like the numbers");
    }

    #[test]
    fn coordinates_outside_the_grid_clamp_to_the_nearest_tile() {
        assert_eq!(Tile::containing(0.5, 0.5), Tile::FIRST);
        assert_eq!(Tile::containing(2.9, 1.0), Tile { u: 2, v: 1 });
        assert_eq!(Tile::containing(-3.0, -0.1), Tile::FIRST);
        assert_eq!(Tile::containing(50.0, 500.0), Tile { u: MAX_TILE_U, v: MAX_TILE_V });
        assert_eq!(Tile::containing(f32::NAN, 0.5), Tile::FIRST);
    }

    #[test]
    fn a_sets_tiles_are_found_sorted_and_unique() {
        let m = mesh();
        assert_eq!(m.set_tiles(0), [Tile { u: 0, v: 0 }, Tile { u: 1, v: 0 }, Tile { u: 0, v: 1 }]);
        assert_eq!(m.set_tiles(1), [Tile::FIRST]);
        assert_eq!(m.set_tiles(7), [Tile::FIRST], "unknown set: the first tile");
        assert_eq!(Mesh::default().set_tiles(0), [Tile::FIRST]);
    }

    #[test]
    fn partitioning_makes_each_tile_one_run_and_keeps_every_triangle() {
        let mut m = mesh();
        let before: std::collections::BTreeSet<[u32; 3]> = m.indices.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
        m.partition_tiles();
        let after: std::collections::BTreeSet<[u32; 3]> = m.indices.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
        assert_eq!(before, after, "the same triangles");
        assert!(m.sets_are_valid());
        let runs = m.tile_ranges();
        let of = |set: usize| runs.iter().filter(|r| r.set == set).map(|r| (r.tile.udim(), r.index_count / 3)).collect::<Vec<_>>();
        assert_eq!(of(0), [(1001, 4), (1002, 2), (1011, 2)], "two quads in 1001 are one run now");
        assert_eq!(of(1), [(1001, 2)]);
        // Runs tile the set exactly.
        for set in 0..2 {
            let mut at = m.set_ranges()[set].first_index;
            for r in runs.iter().filter(|r| r.set == set) {
                assert_eq!(r.first_index, at);
                at += r.index_count;
            }
            assert_eq!(at, m.set_ranges()[set].first_index + m.set_ranges()[set].index_count);
        }
        // Partitioning twice changes nothing.
        let once = m.indices.clone();
        m.partition_tiles();
        assert_eq!(m.indices, once);
    }

    #[test]
    fn extracting_a_tile_shifts_its_uvs_into_the_unit_square() {
        let m = mesh();
        let t = m.extract_tile(0, Tile { u: 1, v: 0 }).unwrap();
        assert_eq!(t.indices.len(), 6);
        assert!(t.vertices.iter().all(|v| (0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1])), "{:?}", t.vertices);
        let c = m.extract_tile(0, Tile { u: 0, v: 1 }).unwrap();
        assert!(c.vertices.iter().all(|v| v.uv[1] < 1.0 && v.uv[1] > 0.0));
        assert_eq!(m.extract_tile(0, Tile { u: 5, v: 5 }).unwrap().indices.len(), 0, "a tile the set does not use is empty");
        assert!(m.extract_tile(9, Tile::FIRST).is_none());
    }

    #[test]
    fn degenerate_triangles_and_sets_without_triangles_are_fine() {
        let mut m = Mesh::default();
        assert!(m.tile_ranges().iter().all(|r| r.index_count == 0));
        m.partition_tiles();
        let sphere = Mesh::uv_sphere(1.0, 8, 4);
        assert_eq!(sphere.set_tiles(0), [Tile::FIRST]);
        let runs = sphere.tile_ranges();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].index_count as usize, sphere.indices.len());
    }
}
