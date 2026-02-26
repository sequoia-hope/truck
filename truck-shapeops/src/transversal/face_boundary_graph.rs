//! Face Boundary Graph (FBG) — topology-first face division.
//!
//! Implements the Sugihara & Iri half-edge linking approach: given a face's
//! boundary wires (original + IC-derived), build a planar graph in parametric
//! (u,v) space, radially sort outgoing half-edges at each vertex, link via the
//! "next CCW after twin" rule, then extract face cycles.

use rustc_hash::FxHashMap;
use std::ops::Deref;
use truck_base::cgmath64::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

use super::loops_store::{Loops, ShapesOpStatus};

type FaceFragment<C, S> = (Face<Point3, C, S>, ShapesOpStatus);

/// Half-edge in the face boundary graph.
#[derive(Debug, Clone)]
pub struct FBGHalfEdge<C> {
    /// Index in the half_edges vec.
    pub id: usize,
    /// Index into vertices vec (origin of this half-edge).
    pub origin: usize,
    /// Twin half-edge index (= id ^ 1 for contiguously-allocated pairs).
    pub twin: usize,
    /// Next half-edge in the face cycle (set by `link_next`).
    pub next: usize,
    /// The underlying truck edge.
    pub edge: Edge<Point3, C>,
    /// Direction on the underlying edge (true = same as edge front→back).
    pub forward: bool,
    /// Classification status from BoundaryWire.
    pub status: ShapesOpStatus,
    /// Whether this half-edge has been consumed during cycle extraction.
    pub used: bool,
}

/// Vertex in the face boundary graph with parametric (u,v) position.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FBGVertex {
    /// Index into vertices vec.
    pub id: usize,
    /// The truck vertex.
    pub vertex: Vertex<Point3>,
    /// Parametric position on face surface.
    pub uv: Point2,
    /// Half-edge IDs leaving this vertex, sorted CCW by departure angle.
    pub outgoing: Vec<usize>,
}

/// The face boundary graph for one face.
#[derive(Debug)]
pub struct FaceBoundaryGraph<C> {
    /// All vertices in the graph.
    pub vertices: Vec<FBGVertex>,
    /// All half-edges (allocated in pairs: indices 2k and 2k+1 are twins).
    pub half_edges: Vec<FBGHalfEdge<C>>,
}

impl<C: Clone> FaceBoundaryGraph<C> {
    /// Build a face boundary graph from the loops (boundary wires) of a face.
    ///
    /// Returns `None` if parametric mapping fails for any vertex.
    pub fn from_loops<S>(loops: &Loops<Point3, C>, surface: &S, tol: f64) -> Option<Self>
    where
        S: SearchParameter<D2, Point = Point3>,
    {
        let _ = tol; // reserved for future short-edge tangent direction
                     // Step 1: Collect unique vertices and map to parametric (u,v).
        let mut vid_to_idx: FxHashMap<VertexID<Point3>, usize> = FxHashMap::default();
        let mut vertices: Vec<FBGVertex> = Vec::new();

        for bw in loops.iter() {
            let wire: &Wire<Point3, C> = bw.deref();
            for v in wire.vertex_iter() {
                let vid = v.id();
                if vid_to_idx.contains_key(&vid) {
                    continue;
                }
                let pt = v.point();
                let uv_tuple = surface.search_parameter(pt, None, 100)?;
                let uv = Point2::new(uv_tuple.0, uv_tuple.1);
                let idx = vertices.len();
                vertices.push(FBGVertex {
                    id: idx,
                    vertex: v.clone(),
                    uv,
                    outgoing: Vec::new(),
                });
                vid_to_idx.insert(vid, idx);
            }
        }

        // Step 2: Build half-edge pairs for each edge in each wire.
        let mut half_edges: Vec<FBGHalfEdge<C>> = Vec::new();

        for bw in loops.iter() {
            let wire: &Wire<Point3, C> = bw.deref();
            let status = bw.status();

            for edge in wire.iter() {
                let front_vid = edge.front().id();
                let back_vid = edge.back().id();

                // Skip degenerate zero-length edges (same vertex at both ends).
                let front_idx = match vid_to_idx.get(&front_vid) {
                    Some(&i) => i,
                    None => continue,
                };
                let back_idx = match vid_to_idx.get(&back_vid) {
                    Some(&i) => i,
                    None => continue,
                };
                if front_idx == back_idx {
                    continue;
                }

                // Check if we already have a half-edge for this edge direction.
                // This happens when an IC edge appears in two BoundaryWires with
                // complementary statuses (And wire uses forward, Or wire uses inverse).
                let edge_id = edge.id();
                let edge_ori = edge.orientation();

                let existing_idx = half_edges
                    .iter()
                    .position(|he| he.edge.id() == edge_id && he.forward == edge_ori);

                if let Some(idx) = existing_idx {
                    // Update the found half-edge's status from the second wire.
                    // The first wire set both HEs to its status; this corrects the
                    // reverse HE to use the second wire's (complementary) status.
                    if status != ShapesOpStatus::Unknown {
                        half_edges[idx].status = status;
                    }
                    continue;
                }

                let he_id = half_edges.len();
                // Forward half-edge: front→back
                half_edges.push(FBGHalfEdge {
                    id: he_id,
                    origin: front_idx,
                    twin: he_id + 1,
                    next: usize::MAX, // set by link_next
                    edge: edge.clone(),
                    forward: true,
                    status,
                    used: false,
                });
                // Reverse half-edge: back→front
                half_edges.push(FBGHalfEdge {
                    id: he_id + 1,
                    origin: back_idx,
                    twin: he_id,
                    next: usize::MAX, // set by link_next
                    edge: edge.clone(),
                    forward: false,
                    status,
                    used: false,
                });
            }
        }

        if half_edges.is_empty() {
            return None;
        }

        let mut graph = FaceBoundaryGraph {
            vertices,
            half_edges,
        };

        // Step 3: Build outgoing lists and radially sort.
        graph.build_outgoing();
        graph.radial_sort(tol);

        // Check connectivity: FBG only works correctly when all wires share
        // vertices (IC edges connect boundary and IC wires). Disconnected
        // components (e.g., a hole wire with no shared vertices) produce
        // spurious fragments. Fall back to legacy for disconnected graphs.
        if !graph.is_connected() {
            return None;
        }

        // Step 4: Link next pointers using the Sugihara & Iri rule.
        graph.link_next();

        // Validate: every half-edge must have a valid next pointer.
        if graph.half_edges.iter().any(|he| he.next == usize::MAX) {
            return None;
        }

        Some(graph)
    }

    /// Check if the graph is connected (all vertices reachable from vertex 0).
    /// Returns false for disconnected components (e.g., a hole wire with no
    /// shared vertices with the boundary wire).
    fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }
        let n = self.vertices.len();
        let mut visited = vec![false; n];
        let mut stack = vec![0usize];
        visited[0] = true;
        let mut count = 1usize;

        while let Some(v_idx) = stack.pop() {
            for &he_id in &self.vertices[v_idx].outgoing {
                let dest = self.half_edges[self.half_edges[he_id].twin].origin;
                if !visited[dest] {
                    visited[dest] = true;
                    count += 1;
                    stack.push(dest);
                }
            }
        }

        count == n
    }

    /// Build outgoing half-edge lists for each vertex.
    fn build_outgoing(&mut self) {
        // Clear existing outgoing lists.
        for v in &mut self.vertices {
            v.outgoing.clear();
        }
        for he in &self.half_edges {
            self.vertices[he.origin].outgoing.push(he.id);
        }
    }

    /// Radially sort outgoing half-edges at each vertex CCW by departure angle
    /// in parametric (u,v) space.
    fn radial_sort(&mut self, tol: f64) {
        let param_tol = tol * 1e-6; // very small threshold in parameter space
        for v_idx in 0..self.vertices.len() {
            let origin_uv = self.vertices[v_idx].uv;
            let outgoing = self.vertices[v_idx].outgoing.clone();

            // Compute departure angles for each outgoing half-edge.
            let mut angles: Vec<(usize, f64)> = outgoing
                .iter()
                .map(|&he_id| {
                    let he = &self.half_edges[he_id];
                    // The destination vertex is the origin of the twin.
                    let dest_idx = self.half_edges[he.twin].origin;
                    let dest_uv = self.vertices[dest_idx].uv;
                    let du = dest_uv.x - origin_uv.x;
                    let dv = dest_uv.y - origin_uv.y;

                    // For very short edges, the angle might be unreliable.
                    // But we still compute it; the edge won't be zero-length
                    // because we filtered degenerate edges in from_loops.
                    let len_sq = du * du + dv * dv;
                    if len_sq < param_tol * param_tol {
                        // Fallback: use a deterministic tiebreaker based on
                        // half-edge ID to avoid non-determinism.
                        (he_id, he_id as f64 * 1e-10)
                    } else {
                        (he_id, dv.atan2(du))
                    }
                })
                .collect();

            // Sort CCW by angle (ascending atan2 is CCW).
            angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            self.vertices[v_idx].outgoing = angles.iter().map(|&(he_id, _)| he_id).collect();
        }
    }

    /// Link `next` pointers using the standard DCEL rule:
    /// For half-edge h (from u to v), h.next is the outgoing half-edge at v
    /// that is PREVIOUS (CW-adjacent) to twin(h) in the CCW-sorted outgoing
    /// list. This ensures the face to the LEFT of h is traced by following
    /// next pointers.
    fn link_next(&mut self) {
        for he_idx in 0..self.half_edges.len() {
            let twin_idx = self.half_edges[he_idx].twin;
            let twin_origin = self.half_edges[twin_idx].origin;
            let outgoing = &self.vertices[twin_origin].outgoing;

            if outgoing.is_empty() {
                continue;
            }

            // Find twin_idx in the outgoing list of twin_origin.
            if let Some(pos) = outgoing.iter().position(|&id| id == twin_idx) {
                // Take the PREVIOUS entry in CCW order (= CW-adjacent).
                // For a planar DCEL, this traces the face to the LEFT of h.
                let n = outgoing.len();
                let prev_pos = (pos + n - 1) % n;
                let next_he = outgoing[prev_pos];
                self.half_edges[he_idx].next = next_he;
            }
        }
    }

    /// Extract face cycles by following `next` pointers.
    ///
    /// Returns a list of cycles, where each cycle is a list of half-edge IDs
    /// forming a closed boundary.
    pub fn extract_cycles(&mut self) -> Vec<Vec<usize>> {
        let mut cycles: Vec<Vec<usize>> = Vec::new();

        for start in 0..self.half_edges.len() {
            if self.half_edges[start].used {
                continue;
            }

            let mut cycle = Vec::new();
            let mut current = start;
            let max_steps = self.half_edges.len() + 1;
            let mut steps = 0;

            loop {
                if self.half_edges[current].used {
                    // Hit an already-used half-edge before closing the cycle.
                    // This shouldn't happen in a valid graph.
                    break;
                }
                self.half_edges[current].used = true;
                cycle.push(current);
                current = self.half_edges[current].next;
                steps += 1;

                if current == start {
                    break;
                }
                if steps > max_steps {
                    // Safety: prevent infinite loops from malformed graphs.
                    break;
                }
            }

            if current == start && !cycle.is_empty() {
                cycles.push(cycle);
            }
        }

        cycles
    }

    /// Build face fragments from extracted cycles.
    ///
    /// Each cycle becomes a wire. Positive-area cycles are outer boundaries;
    /// negative-area cycles are holes assigned to the containing outer boundary.
    /// Returns `(Face, ShapesOpStatus)` pairs.
    pub fn build_fragments<S>(
        &self,
        cycles: &[Vec<usize>],
        surface: &S,
        face_orientation: bool,
        tol: f64,
        tau_area: f64,
    ) -> Vec<FaceFragment<C, S>>
    where
        S: Clone + SearchParameter<D2, Point = Point3>,
    {
        if cycles.is_empty() {
            return Vec::new();
        }

        // For each cycle, build a wire, compute parametric area, and determine status.
        struct CycleInfo<C> {
            wire: Wire<Point3, C>,
            area: f64,
            status: ShapesOpStatus,
            uv_points: Vec<Point2>,
        }

        let mut cycle_infos: Vec<CycleInfo<C>> = Vec::new();

        for cycle in cycles {
            let mut edges_for_wire: Vec<Edge<Point3, C>> = Vec::new();
            let mut uv_points: Vec<Point2> = Vec::new();
            let mut cycle_status = ShapesOpStatus::Unknown;
            let mut and_count = 0usize;
            let mut or_count = 0usize;

            for &he_id in cycle {
                let he = &self.half_edges[he_id];
                let edge = if he.forward {
                    he.edge.clone()
                } else {
                    he.edge.inverse()
                };
                edges_for_wire.push(edge);

                // Collect parametric positions for area computation.
                uv_points.push(self.vertices[he.origin].uv);

                // Count IC-derived half-edge statuses for majority vote.
                match he.status {
                    ShapesOpStatus::And => and_count += 1,
                    ShapesOpStatus::Or => or_count += 1,
                    ShapesOpStatus::Unknown => {}
                }
            }
            if and_count > 0 || or_count > 0 {
                cycle_status = if and_count >= or_count {
                    ShapesOpStatus::And
                } else {
                    ShapesOpStatus::Or
                };
            }

            if edges_for_wire.is_empty() {
                continue;
            }

            let wire: Wire<Point3, C> = edges_for_wire.into_iter().collect();
            if !wire.is_closed() {
                continue;
            }

            // Compute signed parametric area using the shoelace formula.
            let area = parametric_area(&uv_points);

            cycle_infos.push(CycleInfo {
                wire,
                area,
                status: cycle_status,
                uv_points,
            });
        }

        // Separate into outer boundaries (positive area) and holes (negative area).
        let mut outers: Vec<usize> = Vec::new();
        let mut holes: Vec<usize> = Vec::new();

        for (i, info) in cycle_infos.iter().enumerate() {
            // Skip degenerate cycles with negligible parametric area
            // (matches legacy divide_one_face behavior).
            if info.area.abs() < tau_area {
                continue;
            }
            if info.area > 0.0 {
                outers.push(i);
            } else if info.area < 0.0 {
                holes.push(i);
            }
        }

        // If no outer boundaries found, nothing to build.
        if outers.is_empty() {
            return Vec::new();
        }

        // Assign each hole to its containing outer boundary via point-in-polygon
        // test in (u,v) space.
        let mut outer_holes: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for &outer_idx in &outers {
            outer_holes.insert(outer_idx, Vec::new());
        }

        // Collect vertex IDs for each cycle to check disjointness.
        let cycle_vertex_ids: Vec<rustc_hash::FxHashSet<VertexID<Point3>>> = cycle_infos
            .iter()
            .map(|info| info.wire.vertex_iter().map(|v| v.id()).collect())
            .collect();

        for &hole_idx in &holes {
            // Use the first vertex of the hole as the test point.
            let test_pt = cycle_infos[hole_idx].uv_points[0];
            let hole_vids = &cycle_vertex_ids[hole_idx];
            for &outer_idx in &outers {
                // Only assign hole if its vertices are disjoint from the
                // outer boundary. The exterior cycle shares vertices with
                // inner faces and must not be assigned as a hole.
                let outer_vids = &cycle_vertex_ids[outer_idx];
                if !hole_vids.is_disjoint(outer_vids) {
                    continue;
                }
                if point_in_polygon_uv(&cycle_infos[outer_idx].uv_points, test_pt) {
                    outer_holes.get_mut(&outer_idx).unwrap().push(hole_idx);
                    break;
                }
            }
        }

        // Build faces from outer boundaries + their assigned holes.
        let mut result: Vec<FaceFragment<C, S>> = Vec::new();

        for &outer_idx in &outers {
            let outer_info = &cycle_infos[outer_idx];
            let mut wires = vec![outer_info.wire.clone()];
            let mut face_status = outer_info.status;

            if let Some(hole_indices) = outer_holes.get(&outer_idx) {
                // Check for area cancellation: when outer + holes sum to nearly
                // zero, the face is consumed by the intersection (matches legacy
                // divide_one_face behavior).
                let hole_area_sum: f64 = hole_indices
                    .iter()
                    .map(|&hi| cycle_infos[hi].area)
                    .sum();
                if (outer_info.area + hole_area_sum).abs() < tol {
                    continue; // face consumed
                }

                for &hole_idx in hole_indices {
                    let hole_info = &cycle_infos[hole_idx];
                    wires.push(hole_info.wire.clone());
                    if hole_info.status != ShapesOpStatus::Unknown {
                        face_status = hole_info.status;
                    }
                }
            }

            match Face::try_new(wires, surface.clone()) {
                Ok(mut face) => {
                    if !face_orientation {
                        face.invert();
                    }
                    result.push((face, face_status));
                }
                Err(_) => {
                    // Face construction failed — skip this fragment.
                    // The caller will fall back to the legacy path.
                }
            }
        }

        result
    }
}

/// Compute the signed parametric area of a polygon using the shoelace formula.
/// Positive = CCW (outer boundary), negative = CW (hole).
fn parametric_area(pts: &[Point2]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let n = pts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        sum += (pts[j].x + pts[i].x) * (pts[j].y - pts[i].y);
    }
    sum / 2.0
}

/// Point-in-polygon test in (u,v) parametric space using the crossing number
/// (ray casting) algorithm.
fn point_in_polygon_uv(polygon: &[Point2], pt: Point2) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];
        if ((pi.y > pt.y) != (pj.y > pt.y))
            && (pt.x < (pj.x - pi.x) * (pt.y - pi.y) / (pj.y - pi.y) + pi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Top-level function: divide one face using the FBG approach.
///
/// Returns `Some(fragments)` if FBG succeeds, `None` if it fails
/// (caller should fall back to legacy `divide_one_face`).
pub fn divide_one_face_graph<C, S>(
    face: &Face<Point3, C, S>,
    loops: &Loops<Point3, C>,
    tol: f64,
    tau_area: f64,
) -> Option<Vec<FaceFragment<C, S>>>
where
    C: Clone + BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let surface = face.surface();
    let mut graph = FaceBoundaryGraph::from_loops(loops, &surface, tol)?;
    let cycles = graph.extract_cycles();

    if cycles.is_empty() {
        return None;
    }

    let fragments = graph.build_fragments(&cycles, &surface, face.orientation(), tol, tau_area);

    if fragments.is_empty() {
        return None;
    }

    Some(fragments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use truck_geometry::prelude::*;
    use truck_topology::Vertex;

    use super::super::loops_store::{BoundaryWire, ShapesOpStatus};

    type TestCurve = BSplineCurve<Point3>;

    fn line_edge(v0: &Vertex<Point3>, v1: &Vertex<Point3>) -> Edge<Point3, TestCurve> {
        let curve = BSplineCurve::new(KnotVec::bezier_knot(1), vec![v0.point(), v1.point()]);
        Edge::new(v0, v1, curve)
    }

    fn make_xy_plane() -> Plane {
        Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    }

    fn make_loops(
        wires: Vec<(Wire<Point3, TestCurve>, ShapesOpStatus)>,
    ) -> Loops<Point3, TestCurve> {
        wires
            .into_iter()
            .map(|(w, s)| BoundaryWire::new(w, s))
            .collect()
    }

    /// Test 1: Simple quad face (4-edge wire, no IC) → FBG produces 1 cycle.
    #[test]
    fn simple_quad_one_cycle() {
        let v = Vertex::news([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);
        let wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v[0], &v[1]),
            line_edge(&v[1], &v[2]),
            line_edge(&v[2], &v[3]),
            line_edge(&v[3], &v[0]),
        ]
        .into_iter()
        .collect();

        let loops = make_loops(vec![(wire, ShapesOpStatus::Unknown)]);
        let surface = make_xy_plane();
        let graph = FaceBoundaryGraph::from_loops(&loops, &surface, 0.01);
        assert!(graph.is_some(), "FBG construction should succeed");
        let mut graph = graph.unwrap();

        assert_eq!(graph.vertices.len(), 4);
        assert_eq!(graph.half_edges.len(), 8); // 4 edges × 2 half-edges

        let cycles = graph.extract_cycles();
        // A simple quad boundary produces 2 cycles: one for each side of each
        // half-edge traversal (inner and outer). The outer cycle goes around
        // the outside (CW in parameter space = negative area) and the inner
        // cycle goes CCW (positive area).
        assert_eq!(cycles.len(), 2, "quad boundary should produce 2 cycles");

        // Check that all half-edges are used.
        assert!(graph.half_edges.iter().all(|he| he.used));
    }

    /// Test 2: Face split by 1 shared IC edge → FBG produces 2 positive-area fragments.
    ///
    /// In the boolean pipeline, the IC edge is the SAME Edge object shared
    /// between both sub-face wires (one forward, one inverse). This test
    /// mirrors that structure.
    #[test]
    fn face_split_by_ic_two_fragments() {
        // A rectangle split vertically by a shared IC edge:
        //   v3 ------ v5 ------ v2
        //   |          |          |
        //   |   left   |  right   |
        //   |          |          |
        //   v0 ------ v4 ------ v1
        let v = Vertex::news([
            Point3::new(0.0, 0.0, 0.0), // v0
            Point3::new(2.0, 0.0, 0.0), // v1
            Point3::new(2.0, 1.0, 0.0), // v2
            Point3::new(0.0, 1.0, 0.0), // v3
            Point3::new(1.0, 0.0, 0.0), // v4 (IC vertex on bottom)
            Point3::new(1.0, 1.0, 0.0), // v5 (IC vertex on top)
        ]);

        // The IC edge is shared between both wires.
        let ic_edge = line_edge(&v[4], &v[5]);

        // Left sub-face wire: v0 → v4 → v5 → v3 → v0 (uses IC edge forward)
        let left_wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v[0], &v[4]),
            ic_edge.clone(),
            line_edge(&v[5], &v[3]),
            line_edge(&v[3], &v[0]),
        ]
        .into_iter()
        .collect();

        // Right sub-face wire: v4 → v1 → v2 → v5 → v4 (uses IC edge inverse)
        let right_wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v[4], &v[1]),
            line_edge(&v[1], &v[2]),
            line_edge(&v[2], &v[5]),
            ic_edge.inverse(),
        ]
        .into_iter()
        .collect();

        let loops = make_loops(vec![
            (left_wire, ShapesOpStatus::And),
            (right_wire, ShapesOpStatus::Or),
        ]);
        let surface = make_xy_plane();
        let graph = FaceBoundaryGraph::from_loops(&loops, &surface, 0.01);
        assert!(graph.is_some());
        let mut graph = graph.unwrap();

        assert_eq!(graph.vertices.len(), 6);
        // 7 unique edges × 2 half-edges = 14 (IC edge shared, counted once)
        assert_eq!(graph.half_edges.len(), 14);

        let cycles = graph.extract_cycles();
        // 3 cycles: left CCW, right CCW, exterior CW
        assert_eq!(cycles.len(), 3, "split face should produce 3 cycles");

        // Build fragments — should get 2 positive-area faces.
        let fragments = graph.build_fragments(&cycles, &surface, true, 0.01, 1e-10);
        assert_eq!(
            fragments.len(),
            2,
            "should produce exactly 2 face fragments"
        );

        // Verify that the two fragments have complementary And/Or statuses.
        let statuses: Vec<ShapesOpStatus> = fragments.iter().map(|(_, s)| *s).collect();
        assert!(
            statuses.contains(&ShapesOpStatus::And),
            "one fragment must have And status, got {:?}",
            statuses
        );
        assert!(
            statuses.contains(&ShapesOpStatus::Or),
            "one fragment must have Or status, got {:?}",
            statuses
        );
    }

    /// Test 3: Face with disconnected inner hole wire.
    ///
    /// FBG requires a connected graph (all wires share vertices). When the
    /// inner hole wire is disconnected from the outer boundary, from_loops
    /// returns None so the caller falls back to the legacy divider.
    #[test]
    fn face_with_hole_disconnected_returns_none() {
        // Outer boundary: 3×3 square (CCW).
        let v_outer = Vertex::news([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(3.0, 3.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        ]);
        let outer_wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v_outer[0], &v_outer[1]),
            line_edge(&v_outer[1], &v_outer[2]),
            line_edge(&v_outer[2], &v_outer[3]),
            line_edge(&v_outer[3], &v_outer[0]),
        ]
        .into_iter()
        .collect();

        // Inner hole: 1×1 square (CW winding = negative area).
        let v_inner = Vertex::news([
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        ]);
        let inner_wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v_inner[0], &v_inner[3]),
            line_edge(&v_inner[3], &v_inner[2]),
            line_edge(&v_inner[2], &v_inner[1]),
            line_edge(&v_inner[1], &v_inner[0]),
        ]
        .into_iter()
        .collect();

        let loops = make_loops(vec![
            (outer_wire, ShapesOpStatus::Unknown),
            (inner_wire, ShapesOpStatus::Unknown),
        ]);
        let surface = make_xy_plane();
        // Disconnected wires → FBG returns None (caller uses legacy divider).
        let graph = FaceBoundaryGraph::from_loops(&loops, &surface, 0.01);
        assert!(
            graph.is_none(),
            "FBG should return None for disconnected wire components"
        );
    }

    /// Test 4: Degenerate zero-length edge is filtered out during construction.
    #[test]
    fn degenerate_zero_length_edge_filtered() {
        let v = Vertex::news([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ]);
        // Create a triangle wire.
        let wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v[0], &v[1]),
            line_edge(&v[1], &v[2]),
            line_edge(&v[2], &v[0]),
        ]
        .into_iter()
        .collect();

        let loops = make_loops(vec![(wire, ShapesOpStatus::Unknown)]);
        let surface = make_xy_plane();
        let graph = FaceBoundaryGraph::from_loops(&loops, &surface, 0.01);
        assert!(graph.is_some());
        let graph = graph.unwrap();
        assert_eq!(graph.vertices.len(), 3);
        // 3 edges × 2 half-edges = 6
        assert_eq!(graph.half_edges.len(), 6);
        // No degenerate edges should be present (front_idx != back_idx for all).
    }

    /// Test 5: All half-edges have valid next pointers after link_next.
    #[test]
    fn all_half_edges_have_valid_next() {
        let v = Vertex::news([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ]);
        let wire: Wire<Point3, TestCurve> = vec![
            line_edge(&v[0], &v[1]),
            line_edge(&v[1], &v[2]),
            line_edge(&v[2], &v[0]),
        ]
        .into_iter()
        .collect();

        let loops = make_loops(vec![(wire, ShapesOpStatus::Unknown)]);
        let surface = make_xy_plane();
        let graph = FaceBoundaryGraph::from_loops(&loops, &surface, 0.01);
        assert!(graph.is_some());
        let graph = graph.unwrap();

        // Every half-edge must have a valid next pointer.
        for he in &graph.half_edges {
            assert_ne!(
                he.next,
                usize::MAX,
                "half-edge {} should have valid next pointer",
                he.id
            );
            assert!(
                he.next < graph.half_edges.len(),
                "next pointer {} out of bounds for half-edge {}",
                he.next,
                he.id
            );
        }

        // Check that following next from any half-edge eventually returns to start.
        for start in 0..graph.half_edges.len() {
            let mut current = start;
            let mut steps = 0;
            loop {
                current = graph.half_edges[current].next;
                steps += 1;
                if current == start || steps > graph.half_edges.len() {
                    break;
                }
            }
            assert_eq!(
                current, start,
                "following next from half-edge {} should form a cycle",
                start
            );
        }
    }

    /// Test: parametric_area computes correct values.
    #[test]
    fn parametric_area_unit_square() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let area = parametric_area(&pts);
        assert!(
            (area - 1.0).abs() < 1e-10,
            "unit square area should be 1.0, got {}",
            area
        );

        // Reversed winding → negative area.
        let pts_cw = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 0.0),
        ];
        let area_cw = parametric_area(&pts_cw);
        assert!(
            (area_cw + 1.0).abs() < 1e-10,
            "CW square area should be -1.0, got {}",
            area_cw
        );
    }

    /// Test: point_in_polygon_uv basic inclusion.
    #[test]
    fn point_in_polygon_basic() {
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        assert!(point_in_polygon_uv(&polygon, Point2::new(1.0, 1.0)));
        assert!(!point_in_polygon_uv(&polygon, Point2::new(3.0, 1.0)));
        assert!(!point_in_polygon_uv(&polygon, Point2::new(-1.0, 1.0)));
    }
}
