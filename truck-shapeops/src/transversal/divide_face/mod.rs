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

/// Merge wires that share vertex IDs into composite figure-8 wires, then split
/// into simple vertex-disjoint wires. Used as a last-resort recovery when
/// Face::try_new fails due to `disjoint_wires` being false.
///
/// Algorithm: splice wires at shared vertices to form figure-8 chains,
/// then use `split_wire_recursive` to decompose back into simple wires.
fn merge_splice_wires<C: Clone>(wires: &[Wire<Point3, C>]) -> Vec<Wire<Point3, C>> {
    use rustc_hash::FxHashMap;
    type Vid = VertexID<Point3>;

    if wires.len() < 2 {
        return wires.to_vec();
    }

    // Build vertex→wire_indices map to find sharing.
    let mut vid_to_wires: FxHashMap<Vid, Vec<usize>> = FxHashMap::default();
    for (wi, w) in wires.iter().enumerate() {
        for v in w.vertex_iter() {
            vid_to_wires.entry(v.id()).or_default().push(wi);
        }
    }

    // Union-find to group wires into connected components via shared vertices.
    let n = wires.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for wire_list in vid_to_wires.values() {
        let mut unique: Vec<usize> = wire_list.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() >= 2 {
            for &wi in &unique[1..] {
                union(&mut parent, unique[0], wi);
            }
        }
    }

    // Group wires by component.
    let mut components: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }

    let mut result: Vec<Wire<Point3, C>> = Vec::new();

    for comp in components.values() {
        if comp.len() == 1 {
            // Single wire — pass through unchanged.
            result.push(wires[comp[0]].clone());
            continue;
        }

        // Splice component wires at shared vertices into a composite figure-8.
        // Start with the first wire's edges, then splice each subsequent wire
        // by rotating both to start at a shared vertex.
        let mut composite: Vec<Edge<Point3, C>> = wires[comp[0]].iter().cloned().collect();

        let mut splice_ok = true;
        for &wi in &comp[1..] {
            let wire_edges: Vec<Edge<Point3, C>> = wires[wi].iter().cloned().collect();

            // Find vertex IDs in the composite.
            let comp_vids: std::collections::HashSet<Vid> =
                composite.iter().map(|e| e.front().id()).collect();

            // Find a shared vertex in the new wire.
            let splice_pos = wire_edges
                .iter()
                .position(|e| comp_vids.contains(&e.front().id()));

            if let Some(pos) = splice_pos {
                let shared_vid = wire_edges[pos].front().id();
                let comp_pos = composite.iter().position(|e| e.front().id() == shared_vid);

                if let Some(cp) = comp_pos {
                    // Splice: rotate composite to start at shared vertex,
                    // then append wire rotated to start at same vertex.
                    let mut spliced = Vec::with_capacity(composite.len() + wire_edges.len());
                    spliced.extend_from_slice(&composite[cp..]);
                    spliced.extend_from_slice(&composite[..cp]);
                    spliced.extend_from_slice(&wire_edges[pos..]);
                    spliced.extend_from_slice(&wire_edges[..pos]);
                    composite = spliced;
                } else {
                    splice_ok = false;
                    break;
                }
            } else {
                splice_ok = false;
                break;
            }
        }

        if !splice_ok {
            // Couldn't splice — return originals for this component.
            for &wi in comp {
                result.push(wires[wi].clone());
            }
            continue;
        }

        let composite_wire: Wire<Point3, C> = composite.into_iter().collect();

        if !composite_wire.is_closed() {
            // Composite not closed — return originals.
            for &wi in comp {
                result.push(wires[wi].clone());
            }
            continue;
        }

        // Split the composite at repeated vertices.
        let mut split_result: Vec<Wire<Point3, C>> = Vec::new();
        if super::split_wire_recursive(&composite_wire, &mut split_result, 0) {
            for w in split_result {
                if w.len() >= 3 && w.is_closed() && !is_biangle_wire(&w) {
                    result.push(w);
                }
            }
        } else {
            // Split failed — return originals.
            for &wi in comp {
                result.push(wires[wi].clone());
            }
        }
    }

    result
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
                .filter(|w| !is_biangle_wire(w))
                .collect();
            if wires.is_empty() {
                return None;
            }
            // Proactively split non-simple wires before Face::try_new.
            // This handles the k8 case where IC vertex insertion creates
            // 10-edge wires visiting the same vertex 3 times.
            let wires: Vec<Wire<Point3, C>> = {
                let mut split_wires = Vec::new();
                for w in wires {
                    if w.is_simple() {
                        split_wires.push(w);
                    } else {
                        let mut sub = Vec::new();
                        if super::split_wire_recursive(&w, &mut sub, 0) {
                            for sw in sub {
                                if sw.is_closed() && !is_biangle_wire(&sw) {
                                    split_wires.push(sw);
                                }
                            }
                        } else {
                            split_wires.push(w);
                        }
                    }
                }
                split_wires
            };
            if wires.is_empty() {
                return None;
            }
            match Face::try_new(wires.clone(), surface.clone()) {
                Ok(mut new_face) => {
                    if !face.orientation() {
                        new_face.invert();
                    }
                    Some((new_face, status))
                }
                Err(_e_outer) => {
                    // Diagnose the failure: individual wire simplicity vs.
                    // inter-wire vertex sharing (disjoint_wires check).
                    #[cfg(debug_assertions)]
                    {
                        let all_simple = wires.iter().all(|w| w.is_simple());
                        let all_closed = wires.iter().all(|w| w.is_closed());
                        let disjoint = Wire::disjoint_wires(&wires);
                        eprintln!(
                            "[boolean] Face::try_new failed: {:?} \
                             (wires={}, all_simple={}, all_closed={}, disjoint={})",
                            _e_outer,
                            wires.len(),
                            all_simple,
                            all_closed,
                            disjoint,
                        );
                        if !disjoint {
                            // Identify which vertex IDs are shared between wires
                            let mut seen = std::collections::HashMap::<VertexID<Point3>, Vec<usize>>::new();
                            for (wi, w) in wires.iter().enumerate() {
                                for v in w.vertex_iter() {
                                    seen.entry(v.id()).or_default().push(wi);
                                }
                            }
                            for (vid, wire_indices) in &seen {
                                if wire_indices.len() > 1 {
                                    let deduped: Vec<usize> = {
                                        let mut d = wire_indices.clone();
                                        d.dedup();
                                        d
                                    };
                                    if deduped.len() > 1 {
                                        eprintln!(
                                            "  shared vertex {:?} in wires {:?}",
                                            vid, deduped,
                                        );
                                    }
                                }
                            }
                        }
                        for (wi, w) in wires.iter().enumerate() {
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
                                    "    edge[{}]: fid={:?} bid={:?} fp=({:.6},{:.6},{:.6}) bp=({:.6},{:.6},{:.6}) ori={}",
                                    ei, a.front().id(), a.back().id(),
                                    fp.x, fp.y, fp.z, bp.x, bp.y, bp.z,
                                    e.orientation(),
                                );
                            }
                        }
                    }
                    let ori = face.orientation();
                    // Fix inter-wire shared vertices (disjoint_wires=false).
                    // When IC vertex insertion maps different IC endpoints to
                    // the same boundary vertex, multiple wires share vertex
                    // IDs. Remove wires whose vertex set is a proper subset
                    // of another wire — these are degenerate artifacts from
                    // IC splitting at existing boundary vertices.
                    if !Wire::disjoint_wires(&wires)
                        && wires.iter().all(|w| w.is_simple() && w.is_closed())
                    {
                        let vertex_sets: Vec<
                            std::collections::HashSet<VertexID<Point3>>,
                        > = wires
                            .iter()
                            .map(|w| w.vertex_iter().map(|v| v.id()).collect())
                            .collect();
                        // Only apply embedded-wire removal for simple
                        // pairwise sharing. Skip when any vertex appears
                        // in 3+ wires (complex multi-way sharing indicates
                        // legitimate hole topology, not degenerate artifacts).
                        let has_multiway = {
                            let mut vid_count =
                                std::collections::HashMap::<
                                    VertexID<Point3>,
                                    usize,
                                >::new();
                            for w in &wires {
                                for v in w.vertex_iter() {
                                    *vid_count.entry(v.id()).or_default() +=
                                        1;
                                }
                            }
                            vid_count.values().any(|&c| c >= 3)
                        };
                        if !has_multiway {
                            let mut fixed_wires: Vec<Wire<Point3, C>> =
                                Vec::new();
                            for (i, w) in wires.iter().enumerate() {
                                let is_embedded = vertex_sets
                                    .iter()
                                    .enumerate()
                                    .any(|(j, vj)| {
                                        i != j
                                            && vertex_sets[i].len()
                                                < vj.len()
                                            && vertex_sets[i].is_subset(vj)
                                    });
                                if !is_embedded {
                                    fixed_wires.push(w.clone());
                                }
                            }
                            if !fixed_wires.is_empty()
                                && fixed_wires.len() < wires.len()
                            {
                                #[cfg(debug_assertions)]
                                eprintln!(
                                    "[boolean] Removed {} fully-embedded \
                                     wires, retrying Face::try_new",
                                    wires.len() - fixed_wires.len(),
                                );
                                if let Ok(mut new_face) = Face::try_new(
                                    fixed_wires,
                                    surface.clone(),
                                ) {
                                    if !ori {
                                        new_face.invert();
                                    }
                                    return Some((new_face, status));
                                }
                            }
                            // Remaining pairwise sharing without
                            // embedded wires: fall through to Sprint D
                            // preservation (face kept as Unknown).
                        }
                    }
                    if Wire::disjoint_wires(&wires) {
                        // Individual wires are non-simple — try splitting
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
                                Face::try_new(split_wires, surface.clone())
                            {
                                if !ori {
                                    new_face.invert();
                                }
                                return Some((new_face, status));
                            }
                        }
                    }
                    // Last resort: merge wires sharing vertex IDs into a
                    // composite figure-8, then re-split into simple disjoint
                    // wires. This handles cases where IC vertex insertion maps
                    // different IC endpoints to the same boundary vertex.
                    // Only apply when sharing is pairwise — skip when any
                    // vertex appears in 3+ different wires (indicates
                    // legitimate hole topology, not IC artifacts).
                    if !Wire::disjoint_wires(&wires)
                        && wires.iter().all(|w| w.is_simple() && w.is_closed())
                        && wires.len() >= 2
                        && !{
                            let mut vid_wire_count =
                                std::collections::HashMap::<
                                    VertexID<Point3>,
                                    std::collections::HashSet<usize>,
                                >::new();
                            for (wi, w) in wires.iter().enumerate() {
                                for v in w.vertex_iter() {
                                    vid_wire_count
                                        .entry(v.id())
                                        .or_default()
                                        .insert(wi);
                                }
                            }
                            vid_wire_count.values().any(|s| s.len() >= 3)
                        }
                    {
                        let merged = merge_splice_wires(&wires);
                        if !merged.is_empty() && Wire::disjoint_wires(&merged) {
                            if let Ok(mut new_face) =
                                Face::try_new(merged, surface.clone())
                            {
                                #[cfg(debug_assertions)]
                                eprintln!(
                                    "[boolean] merge+splice recovered face \
                                     ({} wires → {} wires)",
                                    wires.len(),
                                    new_face.absolute_boundaries().len(),
                                );
                                if !ori {
                                    new_face.invert();
                                }
                                return Some((new_face, status));
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
                        Some(vec) if vec.is_empty() => {
                            // Zero fragments — preserve original face as Unknown
                            // so downstream classification (overlay → coplanar →
                            // ray-cast) can determine its status.
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[divide_face] face {} produced 0 fragments \
                                 — preserving as Unknown",
                                idx,
                            );
                            if is_coplanar {
                                coplanar_fragment_ids.push(face.id());
                            }
                            res.push(face.clone(), ShapesOpStatus::Unknown);
                        }
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
                        None => {
                            // Face division failed — preserve the original face
                            // as Unknown rather than aborting the entire boolean.
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[divide_face] face {} division failed \
                                 — preserving as Unknown",
                                idx,
                            );
                            if is_coplanar {
                                coplanar_fragment_ids.push(face.id());
                            }
                            res.push(face.clone(), ShapesOpStatus::Unknown);
                        }
                    }
                }
            }
            Some(())
        })?;
    Some((res, coplanar_fragment_ids))
}

#[cfg(test)]
mod tests;
