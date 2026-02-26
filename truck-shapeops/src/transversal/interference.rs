//! IC-edge crossing computation and pave block assembly.
//!
//! This module computes explicit intersection records between IC polylines
//! and face boundary edges, replacing the tolerance-based endpoint projection
//! in `loops_store`. The key fix is corner-vertex handling (MV3 bug): when an
//! IC endpoint lands near a boundary vertex, we reuse that vertex directly
//! instead of splitting the edge, preventing figure-8 wires.

use std::collections::BTreeMap;
use truck_base::cgmath64::*;
use truck_topology::*;

use super::loops_store::ShapesOpStatus;
use super::pave_block::*;

type PolylineCurve = truck_meshalgo::prelude::PolylineCurve<Point3>;

// ---------------------------------------------------------------------------
// Segment-segment closest approach in 3D
// ---------------------------------------------------------------------------

/// Result of a segment-segment closest-approach test.
#[derive(Debug, Clone, Copy)]
struct SegmentIntersection {
    /// Parameter on the first segment [0,1].
    s: f64,
    /// Parameter on the second segment [0,1].
    t: f64,
    /// Distance between the closest points.
    distance: f64,
}

/// Compute the closest approach between two 3D line segments.
///
/// Given segment A: P1 + s*(P2-P1), s ∈ [0,1]
///   and segment B: P3 + t*(P4-P3), t ∈ [0,1]
///
/// Returns the parameters (s, t) and the distance at closest approach.
fn segment_segment_closest(p1: Point3, p2: Point3, p3: Point3, p4: Point3) -> SegmentIntersection {
    let d1 = p2 - p1; // direction of segment A
    let d2 = p4 - p3; // direction of segment B
    let r = p1 - p3;

    let a = d1.dot(d1); // |d1|^2
    let e = d2.dot(d2); // |d2|^2
    let f = d2.dot(r);

    let eps = 1e-30;

    // Both segments degenerate to points
    if a <= eps && e <= eps {
        let dist = (p1 - p3).magnitude();
        return SegmentIntersection {
            s: 0.0,
            t: 0.0,
            distance: dist,
        };
    }

    let (mut s, t);

    if a <= eps {
        // First segment degenerates to a point
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= eps {
            // Second segment degenerates to a point
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case
            let b = d1.dot(d2);
            let denom = a * e - b * b; // always >= 0

            // If segments are not parallel, compute closest point on line A
            // to line B, and clamp to segment A.
            if denom.abs() > eps {
                s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
            } else {
                s = 0.0;
            }

            // Compute point on line B closest to S on segment A
            let t_num = b * s + f;
            if t_num < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t_num > e {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t_num / e;
            }
        }
    }

    let closest_a = p1 + s * d1;
    let closest_b = p3 + t * d2;
    let distance = (closest_a - closest_b).magnitude();

    SegmentIntersection { s, t, distance }
}

// ---------------------------------------------------------------------------
// Polyline utilities
// ---------------------------------------------------------------------------

/// Compute cumulative arc-length fractions for a polyline.
///
/// Returns a vector of length N (same as polyline points) where
/// `result[0] = 0.0` and `result[N-1] = 1.0`.
fn polyline_arc_fractions(points: &[Point3]) -> Vec<f64> {
    if points.len() <= 1 {
        return vec![0.0];
    }
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0);
    for i in 1..points.len() {
        let seg_len = (points[i] - points[i - 1]).magnitude();
        cumulative.push(cumulative[i - 1] + seg_len);
    }
    let total = *cumulative.last().unwrap();
    if total > 1e-30 {
        for v in &mut cumulative {
            *v /= total;
        }
    }
    cumulative
}

/// Convert a segment index + local parameter to a global arc-length fraction.
fn segment_to_arc_param(fractions: &[f64], seg_idx: usize, local_t: f64) -> f64 {
    if seg_idx + 1 >= fractions.len() {
        return 1.0;
    }
    let f0 = fractions[seg_idx];
    let f1 = fractions[seg_idx + 1];
    f0 + local_t * (f1 - f0)
}

/// Tessellate a boundary edge into a polyline of 3D points.
///
/// For now this creates a simple 2-point polyline from front to back vertex
/// positions. A future version can use the edge curve's parameter division
/// for higher fidelity.
fn edge_to_polyline<C>(edge: &Edge<Point3, C>) -> Vec<Point3> {
    vec![edge.front().point(), edge.back().point()]
}

// ---------------------------------------------------------------------------
// Core crossing computation
// ---------------------------------------------------------------------------

/// Compute IC-edge crossings for a single face.
///
/// For each IC polyline, finds where it crosses each boundary edge by
/// pairwise segment-segment closest-approach tests. Corner vertices
/// (within `tol` of a wire vertex) are flagged as `is_corner_touch`.
///
/// # Arguments
/// * `ic_polyline` — The IC polyline points.
/// * `ic_index` — Index of this IC in the face's IC list.
/// * `face_wire` — The face's boundary wire.
/// * `tol` — Geometric tolerance.
///
/// # Returns
/// A list of `IcVertex` crossings, unsorted.
pub fn compute_ic_edge_crossings<C: Clone>(
    ic_polyline: &PolylineCurve,
    ic_index: usize,
    face_wire: &Wire<Point3, C>,
    tol: f64,
) -> Vec<IcVertex> {
    let ic_points: &[Point3] = ic_polyline.as_ref();
    if ic_points.len() < 2 {
        return Vec::new();
    }

    let ic_fractions = polyline_arc_fractions(ic_points);
    let mut crossings = Vec::new();

    // Collect boundary vertices for corner-touch detection
    let boundary_vertices: Vec<(Vertex<Point3>, Point3)> = face_wire
        .iter()
        .map(|edge| {
            let v = edge.front().clone();
            let p = v.point();
            (v, p)
        })
        .collect();

    // Check IC endpoints against boundary vertices (corner touch)
    for (endpoint_idx, ic_pt) in [
        (0usize, ic_points[0]),
        (ic_points.len() - 1, *ic_points.last().unwrap()),
    ] {
        let ic_param = if endpoint_idx == 0 { 0.0 } else { 1.0 };

        for (bv, bv_pos) in &boundary_vertices {
            let dist = (ic_pt - *bv_pos).magnitude();
            if dist < tol {
                // IC endpoint touches a boundary vertex — corner touch.
                // Find which edge(s) this vertex belongs to and compute
                // the edge_param (0.0 for front, 1.0 for back).
                for edge in face_wire.iter() {
                    if edge.front() == bv {
                        crossings.push(IcVertex {
                            vertex: bv.clone(),
                            edge_param: 0.0,
                            ic_index,
                            ic_param,
                            is_corner_touch: true,
                        });
                        break;
                    }
                }
                break; // Only one boundary vertex match per IC endpoint
            }
        }
    }

    // Pairwise segment-segment intersection between IC segments and edge segments
    for (edge_idx, edge) in face_wire.iter().enumerate() {
        let edge_poly = edge_to_polyline(edge);
        let edge_fractions = polyline_arc_fractions(&edge_poly);

        for edge_seg in 0..edge_poly.len().saturating_sub(1) {
            let ep1 = edge_poly[edge_seg];
            let ep2 = edge_poly[edge_seg + 1];

            for ic_seg in 0..ic_points.len().saturating_sub(1) {
                let ip1 = ic_points[ic_seg];
                let ip2 = ic_points[ic_seg + 1];

                let result = segment_segment_closest(ip1, ip2, ep1, ep2);

                if result.distance < tol {
                    // Compute global parameters
                    let edge_param_global =
                        segment_to_arc_param(&edge_fractions, edge_seg, result.t);
                    let ic_param_global = segment_to_arc_param(&ic_fractions, ic_seg, result.s);

                    // Skip if this is a corner touch we already recorded
                    let near_edge_endpoint =
                        edge_param_global < tol || edge_param_global > 1.0 - tol;
                    let is_already_corner = near_edge_endpoint
                        && crossings.iter().any(|c: &IcVertex| {
                            c.is_corner_touch
                                && c.ic_index == ic_index
                                && (c.ic_param - ic_param_global).abs() < tol
                        });

                    if is_already_corner {
                        continue;
                    }

                    // Check if this crossing is at an edge endpoint (boundary vertex)
                    if edge_param_global < tol {
                        // Near front vertex of this edge
                        let front = edge.front().clone();
                        crossings.push(IcVertex {
                            vertex: front,
                            edge_param: 0.0,
                            ic_index,
                            ic_param: ic_param_global,
                            is_corner_touch: true,
                        });
                    } else if edge_param_global > 1.0 - tol {
                        // Near back vertex of this edge — this will also be
                        // the front vertex of the next edge. Record on the
                        // next edge as a corner touch if applicable; skip
                        // here to avoid duplicates. The next edge's front
                        // endpoint check will catch it.
                        let _ = edge_idx; // suppress unused warning
                    } else {
                        // Interior crossing — create a new vertex
                        let crossing_pt = ep1 + result.t * (ep2 - ep1);
                        let vertex = Vertex::new(crossing_pt);
                        crossings.push(IcVertex {
                            vertex,
                            edge_param: edge_param_global,
                            ic_index,
                            ic_param: ic_param_global,
                            is_corner_touch: false,
                        });
                    }
                }
            }
        }
    }

    crossings
}

/// Deduplicate crossings that are within `tol` of each other on the same edge.
///
/// When two crossings have `edge_param` values within `tol` and are from
/// the same IC, keep only the first one.
pub fn deduplicate_crossings(crossings: &mut Vec<IcVertex>, tol: f64) {
    if crossings.len() <= 1 {
        return;
    }

    // Sort by edge_param for dedup
    crossings.sort_by(|a, b| {
        a.edge_param
            .partial_cmp(&b.edge_param)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut i = 0;
    while i + 1 < crossings.len() {
        if (crossings[i].edge_param - crossings[i + 1].edge_param).abs() < tol
            && crossings[i].ic_index == crossings[i + 1].ic_index
        {
            // Keep the one with is_corner_touch=true if either is
            if crossings[i + 1].is_corner_touch && !crossings[i].is_corner_touch {
                crossings.swap(i, i + 1);
            }
            crossings.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Pave block assembly
// ---------------------------------------------------------------------------

/// Assemble pave blocks for all edges in a face's boundary wire.
///
/// For each boundary edge:
/// 1. Collect all IC crossings on that edge (filtering by proximity to
///    edge vertices).
/// 2. Sort by `edge_param`.
/// 3. Deduplicate within `tol`.
/// 4. Create N+1 pave blocks for N crossings.
///
/// Returns a map from EdgeID to pave blocks.
pub fn assemble_pave_blocks<C: Clone>(
    face_wire: &Wire<Point3, C>,
    all_crossings: &[IcVertex],
    tol: f64,
) -> BTreeMap<EdgeID<C>, Vec<PaveBlock<C>>> {
    let mut result = BTreeMap::new();

    for edge in face_wire.iter() {
        let edge_id = edge.id();
        let front_pt = edge.front().point();
        let back_pt = edge.back().point();

        // Collect crossings for this edge: those that are interior
        // (not corner touches at endpoints)
        let mut edge_crossings: Vec<IcVertex> = all_crossings
            .iter()
            .filter(|c| {
                if c.is_corner_touch {
                    // Corner touches are recorded at the edge front/back.
                    // Check if this crossing's vertex matches this edge's
                    // front or back.
                    let cv_pt = c.vertex.point();
                    let dist_front = (cv_pt - front_pt).magnitude();
                    let dist_back = (cv_pt - back_pt).magnitude();
                    dist_front < tol || dist_back < tol
                } else {
                    // Interior crossings: check if the crossing point is
                    // close to this edge's line segment.
                    let cv_pt = c.vertex.point();
                    let edge_dir = back_pt - front_pt;
                    let edge_len = edge_dir.magnitude();
                    if edge_len < 1e-30 {
                        return false;
                    }
                    let t = (cv_pt - front_pt).dot(edge_dir) / (edge_len * edge_len);
                    if t < -tol || t > 1.0 + tol {
                        return false;
                    }
                    let proj = front_pt + t.clamp(0.0, 1.0) * edge_dir;
                    (cv_pt - proj).magnitude() < tol
                }
            })
            .filter(|c| {
                // Exclude pure corner touches at endpoints — they don't
                // split the edge.
                if c.is_corner_touch {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        deduplicate_crossings(&mut edge_crossings, tol);
        let blocks = FaceInterference::build_edge_pave_blocks(edge, &edge_crossings);
        result.insert(edge_id, blocks);
    }

    result
}

// ---------------------------------------------------------------------------
// Interference table construction
// ---------------------------------------------------------------------------

/// Build the interference table for a pair of shells.
///
/// For each face pair that may interfere, computes IC-edge crossings and
/// assembles pave blocks. Currently a skeleton that processes crossings
/// face-by-face.
///
/// # Arguments
/// * `shell0_face_count` — number of faces in shell 0
/// * `shell1_face_count` — number of faces in shell 1
pub fn build_interference_table<C: Clone>(
    shell0_face_count: usize,
    shell1_face_count: usize,
) -> InterferenceTable<C> {
    // Phase 1: Create empty table. Face-pair crossing computation will be
    // wired in when we integrate with the main classification pipeline.
    InterferenceTable::new(shell0_face_count, shell1_face_count)
}

/// Populate a single face's interference from pre-computed IC polylines.
///
/// # Arguments
/// * `face_interference` — the face interference to populate
/// * `face_wire` — the face's boundary wire
/// * `ic_polylines` — list of IC polylines crossing this face
/// * `tol` — geometric tolerance
pub fn populate_face_interference<C: Clone>(
    face_interference: &mut FaceInterference<C>,
    face_wire: &Wire<Point3, C>,
    ic_polylines: &[(usize, PolylineCurve)],
    tol: f64,
) {
    // Compute crossings for each IC
    let mut all_crossings = Vec::new();
    for (ic_index, polyline) in ic_polylines {
        let crossings = compute_ic_edge_crossings(polyline, *ic_index, face_wire, tol);
        all_crossings.extend(crossings);
    }

    // Record crossings
    for crossing in &all_crossings {
        face_interference.add_crossing(crossing.clone());
    }

    // Assemble pave blocks
    face_interference.edge_pave_blocks = assemble_pave_blocks(face_wire, &all_crossings, tol);
}

// ---------------------------------------------------------------------------
// Adapter: interference table → loops store format
// ---------------------------------------------------------------------------

/// Convert pave blocks and IC segments for one face back into the
/// `BoundaryWire` format that `divide_one_face` expects.
///
/// This is a temporary bridge until Phase 3 replaces `divide_one_face`.
/// The conversion creates wires from pave block edges and IC segment
/// edges in the correct topological order.
///
/// # Arguments
/// * `face_interference` — the computed interference for this face
/// * `original_wire` — the original face boundary wire
///
/// # Returns
/// A list of `BoundaryWire`s representing the face's new boundaries.
pub fn interference_to_boundary_wires<C: Clone>(
    face_interference: &FaceInterference<C>,
    original_wire: &Wire<Point3, C>,
) -> Vec<super::loops_store::BoundaryWire<Point3, C>> {
    // If no crossings, return the original wire unchanged
    if face_interference.ic_vertices.is_empty() {
        return vec![super::loops_store::BoundaryWire::new(
            original_wire.clone(),
            ShapesOpStatus::Unknown,
        )];
    }

    // For Phase 1, fall through to original wire. The actual wire
    // reconstruction from pave blocks will be implemented in Phase 3
    // (face_boundary_graph.rs).
    vec![super::loops_store::BoundaryWire::new(
        original_wire.clone(),
        ShapesOpStatus::Unknown,
    )]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
