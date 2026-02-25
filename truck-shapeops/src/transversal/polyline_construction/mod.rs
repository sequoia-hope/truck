// Order-sensitive: graph walk starts at FxHashMap entry (arbitrary order),
// but polyline direction is canonicalized post-construction (Sprint 26) so
// final output is deterministic regardless of graph traversal order.
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;
use truck_base::{cgmath64::*, tolerance::*};
use truck_meshalgo::prelude::PolylineCurve;

pub fn construct_polylines(lines: &[(Point3, Point3)], tol: f64) -> Vec<PolylineCurve<Point3>> {
    // Compute adaptive grid spacing from the minimum non-degenerate segment length.
    // This prevents distinct intersection points from collapsing into the same cell
    // at corner intersections where points are closer together than 2*tol.
    let min_seg_len = lines
        .iter()
        .map(|(a, b)| {
            let d = b - a;
            (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
        })
        .filter(|&len| len > tol)
        .fold(f64::MAX, f64::min);
    let spacing = if min_seg_len < f64::MAX {
        // Use half the minimum segment length, but never smaller than 2*tol
        (min_seg_len * 0.5).max(2.0 * tol)
    } else {
        2.0 * tol
    };

    let mut graph = Graph::with_spacing(spacing);
    for line in lines {
        graph.add_edge(*line);
    }
    let mut res = Vec::new();
    while !graph.is_empty() {
        let (mut idx, node) = graph.get_one();
        let mut wire: VecDeque<_> = vec![node.coord].into();
        while let Some((idx0, pt)) = graph.get_a_next_node(idx) {
            idx = idx0;
            wire.push_back(pt);
        }
        let mut idx = graph.make_index(wire[0]);
        while let Some((idx0, pt)) = graph.get_a_next_node(idx) {
            idx = idx0;
            wire.push_front(pt);
        }
        res.push(PolylineCurve(wire.into()));
    }
    // Canonicalize polyline direction for deterministic boolean results.
    // The graph walk starts at an arbitrary FxHashMap entry, so polyline
    // direction depends on hash iteration order. from_is_curve uses
    // leader().der(t) — the polyline tangent — to decide And vs Or status.
    // Flipping the polyline flips der(t), which flips the status.
    for poly in &mut res {
        if let (Some(&f), Some(&b)) = (poly.first(), poly.last()) {
            if f.near(&b) {
                // CLOSED polyline: canonical direction via geometric criteria.
                // 1. Find vertex with lexicographically smallest (x,y,z)
                // 2. Rotate so that vertex is first
                // 3. Pick direction where second vertex is lex-smaller
                // This selects a unique representative from 2N equivalent forms
                // (N rotations × 2 directions).
                let n = poly.len() - 1; // unique vertex count (last == first)
                if n >= 3 {
                    let min_idx = (0..n)
                        .min_by(|&a, &b| {
                            let pa = poly[a];
                            let pb = poly[b];
                            (pa.x, pa.y, pa.z)
                                .partial_cmp(&(pb.x, pb.y, pb.z))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    if min_idx > 0 {
                        let mut unique: Vec<_> = poly.0[..n].to_vec();
                        unique.rotate_left(min_idx);
                        unique.push(unique[0]);
                        poly.0 = unique;
                    }
                    // Pick canonical direction: second vertex should be lex-smaller
                    // than last unique vertex (the "backward" neighbor).
                    let fwd = poly[1];
                    let bwd = poly[n - 1];
                    if (bwd.x, bwd.y, bwd.z) < (fwd.x, fwd.y, fwd.z) {
                        let mut unique: Vec<_> = poly.0[..n].to_vec();
                        unique[1..].reverse();
                        unique.push(unique[0]);
                        poly.0 = unique;
                    }
                }
            } else {
                // OPEN polyline: lex-smallest endpoint comes first.
                if (b.x, b.y, b.z) < (f.x, f.y, f.z) {
                    poly.reverse();
                }
            }
        }
    }
    res
}

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct PointIndex([i64; 3]);

fn quantize(pt: Point3, spacing: f64) -> PointIndex {
    let idx = pt.add_element_wise(spacing * 0.5) / spacing;
    PointIndex(idx.cast::<i64>().unwrap().into())
}

struct Node {
    coord: Point3,
    adjacency: HashSet<PointIndex>,
}

impl Node {
    #[inline(always)]
    fn new(coord: Point3, adjacency: HashSet<PointIndex>) -> Node {
        Node { coord, adjacency }
    }

    fn pop_one_adjacency(&mut self) -> PointIndex {
        let idx = *self.adjacency.iter().min().unwrap();
        self.adjacency.remove(&idx);
        idx
    }
}

struct Graph {
    map: HashMap<PointIndex, Node>,
    spacing: f64,
}

impl std::ops::Deref for Graph {
    type Target = HashMap<PointIndex, Node>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl std::ops::DerefMut for Graph {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

impl Graph {
    fn with_spacing(spacing: f64) -> Self {
        Graph {
            map: HashMap::default(),
            spacing,
        }
    }

    #[inline(always)]
    fn make_index(&self, pt: Point3) -> PointIndex {
        quantize(pt, self.spacing)
    }

    fn add_half_edge(&mut self, pt0: Point3, pt1: Point3) {
        let idx0 = self.make_index(pt0);
        let idx1 = self.make_index(pt1);
        if let Some(node) = self.get_mut(&idx0) {
            node.adjacency.insert(idx1);
        } else {
            let mut set = HashSet::default();
            set.insert(idx1);
            self.insert(idx0, Node::new(pt0, set));
        }
    }

    fn add_edge(&mut self, line: (Point3, Point3)) {
        if !line.0.near(&line.1) {
            self.add_half_edge(line.0, line.1);
            self.add_half_edge(line.1, line.0);
        }
    }

    fn get_one(&self) -> (PointIndex, &Node) {
        self.iter()
            .min_by_key(|(idx, _)| **idx)
            .map(|(idx, node)| (*idx, node))
            .unwrap()
    }

    fn get_a_next_node(&mut self, idx: PointIndex) -> Option<(PointIndex, Point3)> {
        let node = self.get_mut(&idx)?;
        let idx0 = node.pop_one_adjacency();
        if node.adjacency.is_empty() {
            self.remove(&idx);
        }
        let node = self.get_mut(&idx0)?;
        node.adjacency.remove(&idx);
        let pt = node.coord;
        if node.adjacency.is_empty() {
            self.remove(&idx0);
        }
        Some((idx0, pt))
    }
}

#[cfg(test)]
mod tests;
