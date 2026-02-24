pub(crate) mod bvh;
pub(crate) mod coplanar;
#[allow(dead_code, unused_imports)]
pub(crate) mod coplanar_overlay;
mod coplanar_splitting;
pub mod diagnostics;
mod divide_face;
mod faces_classification;
mod integrate;
mod intersection_curve;
mod loops_store;
mod polyline_construction;
pub(crate) mod robust_classify;
pub use diagnostics::BooleanDiagnostics;
pub use integrate::{
    and, and_result, and_result_with_tol, and_with_tol, diagnose_open_edges, difference,
    difference_result, difference_result_with_tol, difference_with_tol, find_non_simple_wires,
    heal_shell_vertices, or, or_result, or_result_with_tol, or_with_tol,
    validate_euler_characteristic, BooleanStageError, BooleanTolerance, OpenEdgeInfo,
    ShapeOpsCurve, ShapeOpsSurface,
};

use truck_geometry::prelude::*;
use truck_topology::*;

/// Recursively split a non-simple wire at repeated vertices into simple sub-wires.
///
/// When `weld_coincident_edges` unifies vertices, a wire may end up with multiple
/// vertices appearing twice (forming chained figure-8s). This recursively splits
/// the wire at each repeated vertex until all sub-wires are simple and closed.
///
/// Returns `true` if the wire (and all its sub-wires) are successfully split into
/// simple closed wires in `output`. Returns `false` on failure.
pub(crate) fn split_wire_recursive<C: Clone>(
    wire: &Wire<Point3, C>,
    output: &mut Vec<Wire<Point3, C>>,
    depth: usize,
) -> bool {
    use rustc_hash::FxHashMap;
    type Vid = VertexID<Point3>;

    // Base case: wire is already simple and closed
    if wire.is_simple() && wire.is_closed() {
        output.push(wire.clone());
        return true;
    }

    // Guard against infinite recursion
    if depth >= 10 {
        return false;
    }

    let edges: Vec<_> = wire.iter().cloned().collect();

    // Collect all repeated vertices and try each one
    let mut seen: FxHashMap<Vid, Vec<usize>> = FxHashMap::default();
    for (i, edge) in edges.iter().enumerate() {
        let vid = edge.front().id();
        seen.entry(vid).or_default().push(i);
    }

    // Collect repeated vertices, sorted by first occurrence for deterministic iteration.
    let mut repeated: Vec<_> = seen.iter().filter(|(_, v)| v.len() >= 2).collect();
    repeated.sort_by_key(|(_, positions)| positions[0]);

    if repeated.is_empty() {
        // Wire is not simple but has no repeated front vertices — this means
        // the non-simplicity comes from position-based coincidence (different
        // vertex IDs at the same position). We can't split by vertex ID.
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "[split_wire] depth={}: {} edges, not simple, but no repeated vertex IDs (position-based coincidence?)",
                depth,
                edges.len()
            );
            for (i, e) in edges.iter().enumerate() {
                let a = e.absolute_clone();
                let fp = a.front().point();
                let bp = a.back().point();
                eprintln!(
                    "  e[{}]: fid={:?} bid={:?} f=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3}) ori={}",
                    i,
                    a.front().id(),
                    a.back().id(),
                    fp.x,
                    fp.y,
                    fp.z,
                    bp.x,
                    bp.y,
                    bp.z,
                    e.orientation(),
                );
            }
        }
        return false;
    }

    // Try splitting at each repeated vertex (deterministic order)
    for (_, positions) in &repeated {
        // Try every pair of occurrences of this vertex
        for pi in 0..positions.len() {
            for pj in (pi + 1)..positions.len() {
                let first_idx = positions[pi];
                let i = positions[pj];

                let inner_edges: Vec<_> = edges[first_idx..i].to_vec();
                let outer_edges: Vec<_> = edges[i..]
                    .iter()
                    .chain(edges[..first_idx].iter())
                    .cloned()
                    .collect();

                if inner_edges.is_empty() || outer_edges.is_empty() {
                    continue;
                }

                let inner_wire: Wire<Point3, C> = inner_edges.into_iter().collect();
                let outer_wire: Wire<Point3, C> = outer_edges.into_iter().collect();

                if !inner_wire.is_closed() || !outer_wire.is_closed() {
                    continue;
                }

                // Recursively split both sub-wires
                let mut candidate: Vec<Wire<Point3, C>> = Vec::new();
                if split_wire_recursive(&inner_wire, &mut candidate, depth + 1)
                    && split_wire_recursive(&outer_wire, &mut candidate, depth + 1)
                {
                    output.extend(candidate);
                    return true;
                }
                // This split point didn't work, try next
            }
        }
    }

    #[cfg(debug_assertions)]
    {
        eprintln!(
            "[split_wire] depth={}: FAILED {} edges, {} repeated verts, closed={}",
            depth,
            edges.len(),
            repeated.len(),
            wire.is_closed()
        );
        for (i, e) in edges.iter().enumerate() {
            let a = e.absolute_clone();
            let fp = a.front().point();
            let bp = a.back().point();
            eprintln!(
                "  e[{}]: fid={:?} bid={:?} f=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3}) ori={}",
                i,
                a.front().id(),
                a.back().id(),
                fp.x,
                fp.y,
                fp.z,
                bp.x,
                bp.y,
                bp.z,
                e.orientation(),
            );
        }
    }

    false
}
