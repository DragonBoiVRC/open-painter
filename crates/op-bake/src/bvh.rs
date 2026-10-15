//! Bounding-volume hierarchy for ray queries against a triangle soup.
//!
//! Median-split build, two-sided Moeller-Trumbore intersection. Queries do not allocate.

use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    /// Need not be normalised, but `tmin`/`tmax`/hit distances are in units of `dir`.
    pub dir: Vec3,
    pub tmin: f32,
    pub tmax: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    /// Index into the triangle list given to [`Bvh::build`].
    pub tri: u32,
    /// Barycentric weights of vertices 1 and 2 (vertex 0 gets `1 - u - v`).
    pub u: f32,
    pub v: f32,
}

#[derive(Clone, Copy)]
struct Tri {
    p0: Vec3,
    e1: Vec3,
    e2: Vec3,
}

/// Two-sided ray/triangle test. Returns `(t, u, v)`.
fn intersect(tri: &Tri, ray: &Ray) -> Option<(f32, f32, f32)> {
    let pvec = ray.dir.cross(tri.e2);
    let det = tri.e1.dot(pvec);
    if det.abs() < 1e-12 {
        return None; // parallel to the triangle plane (or degenerate)
    }
    let inv = 1.0 / det;
    let tvec = ray.origin - tri.p0;
    let u = tvec.dot(pvec) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(tri.e1);
    let v = ray.dir.dot(qvec) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = tri.e2.dot(qvec) * inv;
    (t >= ray.tmin && t <= ray.tmax).then_some((t, u, v))
}

#[derive(Clone, Copy)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first index into `order`. Internal: index of the right child (left is `self + 1`).
    first: u32,
    /// 0 for internal nodes.
    count: u32,
    axis: u8,
}

pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
    tris: Vec<Tri>,
}

const LEAF_SIZE: usize = 4;

impl Bvh {
    pub fn build(triangles: &[[Vec3; 3]]) -> Self {
        let tris: Vec<Tri> = triangles.iter().map(|t| Tri { p0: t[0], e1: t[1] - t[0], e2: t[2] - t[0] }).collect();
        let centroids: Vec<Vec3> = triangles.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
        let mut order: Vec<u32> = (0..triangles.len() as u32).collect();
        let mut nodes = Vec::with_capacity(triangles.len() / 2 + 1);
        if !triangles.is_empty() {
            Self::build_node(triangles, &centroids, &mut order, 0, triangles.len(), &mut nodes);
        }
        Self { nodes, order, tris }
    }

    fn build_node(
        triangles: &[[Vec3; 3]],
        centroids: &[Vec3],
        order: &mut [u32],
        start: usize,
        end: usize,
        nodes: &mut Vec<Node>,
    ) -> usize {
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let (mut cmin, mut cmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for &i in &order[start..end] {
            for p in triangles[i as usize] {
                min = min.min(p);
                max = max.max(p);
            }
            cmin = cmin.min(centroids[i as usize]);
            cmax = cmax.max(centroids[i as usize]);
        }
        let me = nodes.len();
        nodes.push(Node { min, max, first: start as u32, count: (end - start) as u32, axis: 0 });
        let extent = cmax - cmin;
        let axis = if extent.x >= extent.y && extent.x >= extent.z { 0 } else if extent.y >= extent.z { 1 } else { 2 };
        if end - start <= LEAF_SIZE || extent[axis] <= 0.0 {
            return me;
        }
        let mid = start + (end - start) / 2;
        order[start..end].select_nth_unstable_by(mid - start, |a, b| {
            centroids[*a as usize][axis].total_cmp(&centroids[*b as usize][axis])
        });
        Self::build_node(triangles, centroids, order, start, mid, nodes);
        let right = Self::build_node(triangles, centroids, order, mid, end, nodes);
        nodes[me].count = 0;
        nodes[me].first = right as u32;
        nodes[me].axis = axis as u8;
        me
    }

    #[cfg(test)]
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    fn box_hit(node: &Node, ray: &Ray, inv: Vec3, tmax: f32) -> bool {
        let t1 = (node.min - ray.origin) * inv;
        let t2 = (node.max - ray.origin) * inv;
        // f32::min/max ignore NaN (a ray lying in a slab plane), which keeps the test conservative.
        let tn = t1.min(t2);
        let tf = t1.max(t2);
        let near = tn.x.max(tn.y).max(tn.z).max(ray.tmin);
        let far = tf.x.min(tf.y).min(tf.z).min(tmax);
        near <= far
    }

    /// Nearest intersection along the ray.
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = Vec3::ONE / ray.dir;
        let mut best: Option<Hit> = None;
        let mut tmax = ray.tmax;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !Self::box_hit(node, ray, inv, tmax) {
                continue;
            }
            if node.count > 0 {
                for k in node.first..node.first + node.count {
                    let ti = self.order[k as usize];
                    let limited = Ray { tmax, ..*ray };
                    if let Some((t, u, v)) = intersect(&self.tris[ti as usize], &limited) {
                        tmax = t;
                        best = Some(Hit { t, tri: ti, u, v });
                    }
                }
            } else {
                let me = stack[sp];
                let (left, right) = (me + 1, node.first);
                // Visit the child on the ray's side of the split first.
                let (near, far) = if ray.dir[node.axis as usize] < 0.0 { (right, left) } else { (left, right) };
                stack[sp] = far;
                stack[sp + 1] = near;
                sp += 2;
            }
        }
        best
    }

    /// True if anything intersects the ray (early exit).
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = Vec3::ONE / ray.dir;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if !Self::box_hit(node, ray, inv, ray.tmax) {
                continue;
            }
            if node.count > 0 {
                for k in node.first..node.first + node.count {
                    if intersect(&self.tris[self.order[k as usize] as usize], ray).is_some() {
                        return true;
                    }
                }
            } else {
                let me = stack[sp];
                stack[sp] = node.first;
                stack[sp + 1] = me + 1;
                sp += 2;
            }
        }
        false
    }

    #[cfg(test)]
    fn brute_closest(&self, ray: &Ray) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, t) in self.tris.iter().enumerate() {
            let limit = best.map_or(ray.tmax, |h| h.t);
            if let Some((t, u, v)) = intersect(t, &Ray { tmax: limit, ..*ray }) {
                best = Some(Hit { t, tri: i as u32, u, v });
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator so failures are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (1u64 << 31) as f32
        }
        fn vec(&mut self, lo: f32, hi: f32) -> Vec3 {
            Vec3::new(lo + (hi - lo) * self.next(), lo + (hi - lo) * self.next(), lo + (hi - lo) * self.next())
        }
    }

    fn soup(n: usize, seed: u64) -> Vec<[Vec3; 3]> {
        let mut r = Rng(seed);
        (0..n)
            .map(|_| {
                let c = r.vec(-5.0, 5.0);
                [c + r.vec(-0.8, 0.8), c + r.vec(-0.8, 0.8), c + r.vec(-0.8, 0.8)]
            })
            .collect()
    }

    #[test]
    fn closest_and_any_hit_agree_with_brute_force() {
        for (n, seed) in [(1, 1), (5, 2), (200, 3), (3000, 4)] {
            let tris = soup(n, seed);
            let bvh = Bvh::build(&tris);
            let mut r = Rng(seed ^ 0xabcdef);
            let mut hits = 0;
            for _ in 0..1500 {
                let ray = Ray {
                    origin: r.vec(-7.0, 7.0),
                    dir: (r.vec(-1.0, 1.0)).normalize_or_zero().max(Vec3::splat(-1.0)),
                    tmin: 1e-4,
                    tmax: if r.next() < 0.5 { f32::INFINITY } else { 6.0 },
                };
                if ray.dir == Vec3::ZERO {
                    continue;
                }
                let (fast, slow) = (bvh.closest_hit(&ray), bvh.brute_closest(&ray));
                match (fast, slow) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        hits += 1;
                        assert!((a.t - b.t).abs() < 1e-4, "n={n}: t {} vs {}", a.t, b.t);
                    }
                    other => panic!("n={n}: closest hit mismatch {other:?} for {ray:?}"),
                }
                assert_eq!(bvh.any_hit(&ray), slow.is_some(), "n={n}: any_hit disagrees for {ray:?}");
            }
            assert!(n < 200 || hits > 50, "scenario too sparse to be a meaningful test ({hits} hits)");
        }
    }

    #[test]
    fn barycentrics_reconstruct_the_hit_point() {
        let tri = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0)];
        let bvh = Bvh::build(&[tri]);
        let ray = Ray { origin: Vec3::new(0.5, 0.25, 3.0), dir: Vec3::NEG_Z, tmin: 0.0, tmax: f32::INFINITY };
        let h = bvh.closest_hit(&ray).unwrap();
        assert!((h.t - 3.0).abs() < 1e-6);
        let p = tri[0] * (1.0 - h.u - h.v) + tri[1] * h.u + tri[2] * h.v;
        assert!((p - Vec3::new(0.5, 0.25, 0.0)).length() < 1e-6, "{p}");
    }

    #[test]
    fn range_limits_misses_and_parallel_rays() {
        let tri = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let bvh = Bvh::build(&[tri]);
        let down = |o: Vec3| Ray { origin: o, dir: Vec3::NEG_Z, tmin: 0.0, tmax: f32::INFINITY };
        assert!(bvh.closest_hit(&down(Vec3::new(0.0, 0.0, 1.0))).is_some());
        assert!(bvh.closest_hit(&down(Vec3::new(5.0, 0.0, 1.0))).is_none(), "outside");
        // Two-sided: from below, going up.
        let up = Ray { origin: Vec3::new(0.0, 0.0, -1.0), dir: Vec3::Z, tmin: 0.0, tmax: f32::INFINITY };
        assert!(bvh.closest_hit(&up).is_some(), "back faces count");
        // tmax before the surface, tmin after it.
        assert!(bvh.closest_hit(&Ray { tmax: 0.5, ..down(Vec3::new(0.0, 0.0, 1.0)) }).is_none());
        assert!(bvh.closest_hit(&Ray { tmin: 1.5, ..down(Vec3::new(0.0, 0.0, 1.0)) }).is_none());
        // Parallel to the plane, inside it.
        let flat = Ray { origin: Vec3::new(-3.0, 0.0, 0.0), dir: Vec3::X, tmin: 0.0, tmax: f32::INFINITY };
        let _ = bvh.closest_hit(&flat); // must not panic or produce NaN hits
        assert!(!bvh.any_hit(&Ray { origin: Vec3::new(-3.0, 0.0, 1.0), dir: Vec3::X, tmin: 0.0, tmax: 100.0 }));
    }

    #[test]
    fn empty_and_degenerate_inputs_are_safe() {
        let empty = Bvh::build(&[]);
        let ray = Ray { origin: Vec3::ZERO, dir: Vec3::X, tmin: 0.0, tmax: 10.0 };
        assert!(empty.closest_hit(&ray).is_none() && !empty.any_hit(&ray));
        let p = Vec3::splat(1.0);
        let degenerate = Bvh::build(&[[p, p, p]; 9]); // identical centroids: cannot be split
        assert!(degenerate.closest_hit(&ray).is_none());
        assert_eq!(degenerate.triangle_count(), 9);
    }
}
