#![allow(clippy::many_single_char_names)]

use super::faces_classification::FacesClassification;
use super::loops_store::*;
// Order-insensitive: HashMap is used for polyline caching (EdgeID -> PolylineCurve)
// and adjacency lookup in rebuild_connected_wires (dead code). Face division iterates
// over shells (Vec) in index order.
use rustc_hash::FxHashMap as HashMap;
use std::ops::Deref;
use truck_meshalgo::prelude::*;
use truck_topology::*;

/// Try to rebuild connected closed wires from a pool of edges.
///
/// When loops_store produces wires with connectivity issues (gaps, wrong
/// orientations, or mixed-up edges), this function collects all edges
/// and reconstructs proper closed wires by graph traversal. Each edge
/// can be traversed in either direction. Matching uses vertex IDs first,
/// then falls back to position proximity.
#[allow(dead_code)]
fn rebuild_connected_wires<C>(wires: &[Wire<Point3, C>], tol: f64) -> Option<Vec<Wire<Point3, C>>>
where
    C: Clone + BoundedCurve<Point = Point3>,
{
    // Collect all edges into a pool with both possible directions
    type Vid = VertexID<Point3>;
    let mut all_edges: Vec<(Edge<Point3, C>, bool)> = Vec::new(); // (abs_edge, used)
    for wire in wires {
        for edge in wire.iter() {
            all_edges.push((edge.absolute_clone(), false));
        }
    }

    if all_edges.is_empty() {
        return None;
    }

    // Build adjacency: vertex_id -> list of (edge_index, is_forward, other_vertex_id)
    let mut adjacency: HashMap<Vid, Vec<(usize, bool, Vid)>> = HashMap::default();
    for (i, (edge, _)) in all_edges.iter().enumerate() {
        let fid = edge.front().id();
        let bid = edge.back().id();
        adjacency.entry(fid).or_default().push((i, true, bid)); // forward: fid → bid
        adjacency.entry(bid).or_default().push((i, false, fid)); // backward: bid → fid
    }

    let mut result_wires: Vec<Wire<Point3, C>> = Vec::new();
    let n = all_edges.len();

    // Try to build closed wires
    while let Some(start_idx) = all_edges.iter().position(|(_, used)| !*used) {
        all_edges[start_idx].1 = true;
        let start_edge = &all_edges[start_idx].0;
        let start_vid = start_edge.front().id();
        let mut current_vid = start_edge.back().id();
        let mut wire_edges: Vec<Edge<Point3, C>> = vec![start_edge.clone()];

        // Follow the chain until we close the loop or get stuck
        let mut stuck = false;
        while current_vid != start_vid {
            if wire_edges.len() > n {
                stuck = true;
                break;
            }

            // Find an unused edge starting at current_vid
            let next = adjacency
                .get(&current_vid)
                .and_then(|adj| adj.iter().find(|(idx, _, _)| !all_edges[*idx].1))
                .copied();

            match next {
                Some((idx, forward, next_vid)) => {
                    all_edges[idx].1 = true;
                    let edge = &all_edges[idx].0;
                    if forward {
                        wire_edges.push(edge.clone());
                    } else {
                        wire_edges.push(edge.inverse());
                    }
                    current_vid = next_vid;
                }
                None => {
                    // Try position-based fallback: find an unused edge with
                    // a vertex near current position
                    let current_pos = {
                        // Find the point for current_vid
                        let last_edge = wire_edges.last().unwrap();
                        last_edge.back().point()
                    };
                    let found = all_edges
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, used))| !*used)
                        .find_map(|(idx, (edge, _))| {
                            let fp = edge.front().point();
                            let bp = edge.back().point();
                            if (fp - current_pos).magnitude() < tol {
                                Some((idx, true, edge.back().id()))
                            } else if (bp - current_pos).magnitude() < tol {
                                Some((idx, false, edge.front().id()))
                            } else {
                                None
                            }
                        });
                    match found {
                        Some((idx, forward, next_vid)) => {
                            all_edges[idx].1 = true;
                            let edge = &all_edges[idx].0;
                            if forward {
                                wire_edges.push(edge.clone());
                            } else {
                                wire_edges.push(edge.inverse());
                            }
                            current_vid = next_vid;
                        }
                        None => {
                            stuck = true;
                            break;
                        }
                    }
                }
            }
        }

        if stuck {
            // Couldn't build a closed wire — bail out
            return None;
        }

        let wire: Wire<Point3, C> = wire_edges.into_iter().collect();
        if wire.is_closed() {
            result_wires.push(wire);
        } else {
            return None;
        }
    }

    if result_wires.is_empty() {
        return None;
    }

    Some(result_wires)
}

fn create_parameter_boundary<P, C, S>(
    face: &Face<P, C, S>,
    wire: &Wire<P, C>,
    polys: &mut HashMap<EdgeID<C>, PolylineCurve<P>>,
    tol: f64,
) -> Option<PolylineCurve<Point2>>
where
    P: Copy,
    C: BoundedCurve<Point = P> + ParameterDivision1D<Point = P>,
    S: Clone + SearchParameter<D2, Point = P>,
{
    let surface = face.surface();
    let pt = wire.front_vertex().unwrap().point();
    let p: Point2 = surface.search_parameter(pt, None, 100)?.into();
    let vec = wire.edge_iter().try_fold(vec![p], |mut vec, edge| {
        let poly = polys.entry(edge.id()).or_insert_with(|| {
            let curve = edge.curve();
            let div = curve.parameter_division(curve.range_tuple(), tol).1;
            PolylineCurve(div)
        });
        let mut p = *vec.last().unwrap();
        let closure = |q: &P| -> Option<Point2> {
            p = surface.search_parameter(*q, Some(p.into()), 100)?.into();
            Some(p)
        };
        let add: Option<Vec<Point2>> = match edge.orientation() {
            true => poly.iter().skip(1).map(closure).collect(),
            false => poly.iter().rev().skip(1).map(closure).collect(),
        };
        vec.append(&mut add?);
        Some(vec)
    })?;
    Some(PolylineCurve(vec))
}

#[derive(Clone, Debug)]
struct WireChunk<'a, C> {
    poly: PolylineCurve<Point2>,
    wire: &'a BoundaryWire<Point3, C>,
}

type FaceWithShapesOpStatus<C, S> = (Face<Point3, C, S>, ShapesOpStatus);
fn divide_one_face<C, S>(
    face: &Face<Point3, C, S>,
    loops: &Loops<Point3, C>,
    tol: f64,
    tau_area: f64,
) -> Option<Vec<FaceWithShapesOpStatus<C, S>>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let (mut pre_faces, mut negative_wires) = (Vec::new(), Vec::new());
    let mut map = HashMap::default();
    loops.iter().try_for_each(|wire| {
        let poly = create_parameter_boundary(face, wire, &mut map, tol)?;
        let area = poly.area();
        // Skip degenerate loops with negligible parametric area. Uses
        // `tau_area` (typically `tau_model^2`) to avoid skipping real IC-derived
        // face fragments whose parametric area is small due to surface
        // parameterization compression (e.g., a 0.5×0.5 world-space corner
        // maps to area 0.028 in parametric space on a 3×3 face).
        if area.abs() < tau_area {
            return Some(());
        }
        match area > 0.0 {
            true => pre_faces.push(vec![WireChunk { poly, wire }]),
            false => negative_wires.push(WireChunk { poly, wire }),
        }
        Some(())
    })?;
    negative_wires.into_iter().try_for_each(|chunk| {
        let pt = chunk.poly.front();
        let idx = pre_faces.iter().position(|face| face[0].poly.include(pt));
        if let Some(i) = idx {
            let outer_area = pre_faces[i][0].poly.area();
            let chunk_area = chunk.poly.area();
            // When inner loop exactly matches outer boundary (areas cancel),
            // the face is consumed by the intersection — remove it.
            if (outer_area + chunk_area).abs() < tol {
                pre_faces[i].clear();
            } else {
                pre_faces[i].push(chunk);
            }
        }
        Some(())
    })?;
    let vec: Vec<_> = pre_faces
        .into_iter()
        .filter(|pre_face| !pre_face.is_empty())
        .filter_map(|pre_face| {
            let surface = face.surface();
            let op = pre_face
                .iter()
                .find(|chunk| chunk.wire.status() != ShapesOpStatus::Unknown);
            let status = match op {
                Some(chunk) => chunk.wire.status(),
                None => ShapesOpStatus::Unknown,
            };
            let wires: Vec<Wire<Point3, C>> = pre_face
                .into_iter()
                .map(|chunk| chunk.wire.deref().clone())
                .collect();
            match Face::try_new(wires.clone(), surface.clone()) {
                Ok(mut new_face) => {
                    if !face.orientation() {
                        new_face.invert();
                    }
                    Some((new_face, status))
                }
                Err(_e_outer) => {
                    // Try recursive wire splitting for non-simple wires.
                    // Do NOT use Face::new_unchecked here — non-simple faces
                    // from divide_one_face cause edge over-sharing (3+ refs)
                    // that breaks the Closed shell invariant in weld_coincident_edges.
                    let ori = face.orientation();
                    let mut split_wires: Vec<Wire<Point3, C>> = Vec::new();
                    let mut any_split = false;
                    for w in &wires {
                        if w.is_simple() {
                            split_wires.push(w.clone());
                            continue;
                        }
                        let mut split_result: Vec<Wire<Point3, C>> = Vec::new();
                        if super::split_wire_recursive(w, &mut split_result, 0) {
                            split_wires.extend(split_result);
                            any_split = true;
                        } else {
                            split_wires.push(w.clone());
                        }
                    }
                    if any_split {
                        if let Ok(mut new_face) =
                            Face::try_new(split_wires.clone(), surface.clone())
                        {
                            if !ori {
                                new_face.invert();
                            }
                            return Some((new_face, status));
                        }
                    }
                    #[cfg(debug_assertions)]
                    {
                        let check_wires = if any_split { &split_wires } else { &wires };
                        eprintln!(
                            "[boolean] Face::try_new failed in divide_one_face: {:?}",
                            _e_outer
                        );
                        for (wi, w) in check_wires.iter().enumerate() {
                            let edges: Vec<_> = w.iter().collect();
                            eprintln!(
                                "  wire[{}]: {} edges, closed={}, simple={}",
                                wi,
                                edges.len(),
                                w.is_closed(),
                                w.is_simple(),
                            );
                            for (ei, e) in edges.iter().enumerate() {
                                let a = e.absolute_clone();
                                let fp = a.front().point();
                                let bp = a.back().point();
                                eprintln!(
                                    "    edge[{}]: fid={:?} bid={:?} fp=({:.4},{:.4},{:.4}) bp=({:.4},{:.4},{:.4}) ori={}",
                                    ei, a.front().id(), a.back().id(),
                                    fp.x, fp.y, fp.z, bp.x, bp.y, bp.z,
                                    e.orientation(),
                                );
                            }
                        }
                    }
                    None
                }
            }
        })
        .collect();
    Some(vec)
}

/// Divide faces and track fragments from coplanar faces.
/// Returns (classification, coplanar_fragment_face_ids) so the caller can re-force
/// coplanar fragments to Unknown after `integrate_by_component`.
#[allow(clippy::type_complexity)]
pub fn divide_faces_with_coplanar<C, S>(
    shell: &Shell<Point3, C, S>,
    loops_store: &LoopsStore<Point3, C>,
    tol: f64,
    coplanar_faces: &rustc_hash::FxHashSet<usize>,
    tau_area: f64,
) -> Option<(FacesClassification<Point3, C, S>, Vec<FaceID<S>>)>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let mut res = FacesClassification::<Point3, C, S>::default();
    let mut coplanar_fragment_ids = Vec::new();
    shell
        .iter()
        .zip(loops_store)
        .enumerate()
        .try_for_each(|(idx, (face, loops))| {
            let is_coplanar = coplanar_faces.contains(&idx);
            if loops
                .iter()
                .all(|wire| wire.status() == ShapesOpStatus::Unknown)
            {
                // Rebuild from loops_store wires (not face.clone()) to preserve
                // vertex substitutions from add_polygon_vertex. This is needed
                // for weld_coincident_edges to find shared Vertex objects.
                let wires: Vec<Wire<Point3, C>> =
                    loops.iter().map(|bw| bw.deref().clone()).collect();
                let rebuilt = match Face::try_new(wires, face.surface()) {
                    Ok(mut f) => {
                        if !face.orientation() {
                            f.invert();
                        }
                        f
                    }
                    Err(_) => face.clone(),
                };
                if is_coplanar {
                    coplanar_fragment_ids.push(rebuilt.id());
                }
                res.push(rebuilt, ShapesOpStatus::Unknown);
            } else {
                // Pre-validate: check for degenerate loops that would cause
                // panics in parameter_division. If any wire has near-zero
                // spatial extent, skip division and use the undivided face.
                let has_degenerate_loop = loops.iter().any(|wire| {
                    let pts: Vec<Point3> = wire.vertex_iter().map(|v| v.point()).collect();
                    if pts.len() < 2 {
                        return true;
                    }
                    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
                    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
                    let (mut min_z, mut max_z) = (f64::MAX, f64::MIN);
                    for p in &pts {
                        min_x = min_x.min(p.x);
                        max_x = max_x.max(p.x);
                        min_y = min_y.min(p.y);
                        max_y = max_y.max(p.y);
                        min_z = min_z.min(p.z);
                        max_z = max_z.max(p.z);
                    }
                    let extent = (max_x - min_x).max(max_y - min_y).max(max_z - min_z);
                    extent < tol * 0.1
                });

                if has_degenerate_loop {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[boolean] Skipping degenerate loop in divide_face \
                         — using undivided face"
                    );
                    if is_coplanar {
                        coplanar_fragment_ids.push(face.id());
                    }
                    res.push(face.clone(), ShapesOpStatus::Unknown);
                } else {
                    match divide_one_face(face, loops, tol, tau_area) {
                        Some(vec) => {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[divide_face] face {} produced {} fragments",
                                idx,
                                vec.len(),
                            );
                            vec.into_iter().for_each(|(face, status)| {
                                if is_coplanar {
                                    coplanar_fragment_ids.push(face.id());
                                }
                                res.push(face, status);
                            });
                        }
                        None => return None,
                    }
                }
            }
            Some(())
        })?;
    Some((res, coplanar_fragment_ids))
}

#[cfg(test)]
mod tests;
