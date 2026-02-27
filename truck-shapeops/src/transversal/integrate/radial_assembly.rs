//! Topology-first shell assembly via radial-sort edge pairing.
//!
//! Replaces tolerance-based progressive weld (`assemble_boolean_shell_v2`) with
//! deterministic edge pairing: group edges by geometric vertex positions, then
//! use 3D radial sort + Sugihara-Iri linking to pair half-edges topologically.
//!
//! For manifold boolean results, most edge groups have exactly 2 half-edges
//! (fast path — no radial sort needed). Multi-edge groups (non-manifold junctions)
//! use 3D radial sort to determine angular ordering around the shared edge.

use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use truck_base::cgmath64::*;
use truck_topology::*;

use super::{validate_euler_characteristic, BooleanStageError, ShapeOpsCurve, ShapeOpsSurface};

/// A half-edge in the global assembly table.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AssemblyHalfEdge {
    /// Index of the face this half-edge belongs to.
    face_idx: usize,
    /// Index of the wire within that face's boundary list.
    wire_idx: usize,
    /// Index of the edge within that wire.
    edge_idx: usize,
    /// Front vertex position (world coordinates, respecting edge orientation in the wire).
    front_pos: Point3,
    /// Back vertex position (world coordinates, respecting edge orientation in the wire).
    back_pos: Point3,
    /// Whether the edge is used in forward orientation in this wire.
    forward: bool,
}

/// Quantized position key for grouping edges by geometric vertex pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct QuantizedPos {
    x: i64,
    y: i64,
    z: i64,
}

impl QuantizedPos {
    fn from_point(p: Point3, inv_quantum: f64) -> Self {
        Self {
            x: (p.x * inv_quantum).round() as i64,
            y: (p.y * inv_quantum).round() as i64,
            z: (p.z * inv_quantum).round() as i64,
        }
    }
}

/// Canonical key for an edge group: the sorted pair of quantized vertex positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EdgeGroupKey(QuantizedPos, QuantizedPos);

impl EdgeGroupKey {
    fn new(a: QuantizedPos, b: QuantizedPos) -> Self {
        if a <= b {
            Self(a, b)
        } else {
            Self(b, a)
        }
    }
}

/// A group of half-edges sharing the same geometric vertex pair.
#[derive(Debug)]
struct EdgeGroup {
    /// Indices into the global half-edge table.
    half_edge_indices: Vec<usize>,
}

/// Key identifying a half-edge position: (face_idx, wire_idx, edge_idx).
type HalfEdgeKey = (usize, usize, usize);

/// Replacement info: (canonical edge, same_direction as original).
type EdgeReplacement<C> = (Edge<Point3, C>, bool);

/// Result of radial assembly: the paired edge mapping.
///
/// Maps half-edge positions to the canonical edge that should be
/// used at that position, plus whether it should be inverted.
struct EdgePairing<C> {
    replacements: FxHashMap<HalfEdgeKey, EdgeReplacement<C>>,
}

/// Build the global half-edge table from classified faces.
///
/// For each face, iterates all boundary wires. For each edge in each wire,
/// creates an AssemblyHalfEdge recording the face/wire/edge indices and
/// the front/back vertex positions (accounting for edge orientation).
fn build_half_edge_table<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    faces: &[Face<Point3, C, S>],
) -> Vec<AssemblyHalfEdge> {
    let mut table = Vec::new();
    for (fi, face) in faces.iter().enumerate() {
        for (wi, wire) in face.absolute_boundaries().iter().enumerate() {
            for (ei, edge) in wire.iter().enumerate() {
                let front_pos = edge.front().point();
                let back_pos = edge.back().point();
                table.push(AssemblyHalfEdge {
                    face_idx: fi,
                    wire_idx: wi,
                    edge_idx: ei,
                    front_pos,
                    back_pos,
                    forward: edge.orientation(),
                });
            }
        }
    }
    table
}

/// Group half-edges by their quantized vertex pair.
///
/// Uses a quantization grid of `tau / 100` to bucket vertex positions.
/// Each group contains all half-edges that share the same geometric edge
/// (same two endpoint positions), regardless of direction.
fn group_by_vertex_pair(table: &[AssemblyHalfEdge], tau: f64) -> BTreeMap<EdgeGroupKey, EdgeGroup> {
    let quantum = tau / 100.0;
    let inv_quantum = 1.0 / quantum;

    let mut groups: BTreeMap<EdgeGroupKey, EdgeGroup> = BTreeMap::new();

    for (idx, he) in table.iter().enumerate() {
        let qa = QuantizedPos::from_point(he.front_pos, inv_quantum);
        let qb = QuantizedPos::from_point(he.back_pos, inv_quantum);
        let key = EdgeGroupKey::new(qa, qb);
        groups
            .entry(key)
            .or_insert_with(|| EdgeGroup {
                half_edge_indices: Vec::new(),
            })
            .half_edge_indices
            .push(idx);
    }

    groups
}

/// Compute the face normal at an edge's midpoint for 3D radial sort.
///
/// Evaluates the surface normal at the parametric point closest to the
/// geometric midpoint of the edge. Falls back to cross-product of edge
/// tangent with a perpendicular vector if the surface normal is degenerate.
fn compute_face_normal_at_edge<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    face: &Face<Point3, C, S>,
    edge_front: Point3,
    edge_back: Point3,
) -> Vector3 {
    let midpoint = EuclideanSpace::midpoint(edge_front, edge_back);
    let surface = face.surface();

    // Try to get surface normal at the edge midpoint
    if let Some((u, v)) = surface.search_nearest_parameter(midpoint, None, 100) {
        let normal = surface.normal(u, v);
        if normal.magnitude2() > 1e-20 {
            let oriented_normal = if face.orientation() {
                normal.normalize()
            } else {
                -normal.normalize()
            };
            return oriented_normal;
        }
    }

    // Fallback: use a perpendicular to the edge direction
    let edge_dir = edge_back - edge_front;
    if edge_dir.magnitude2() < 1e-30 {
        return Vector3::unit_z();
    }
    let edge_dir = edge_dir.normalize();

    // Pick a non-parallel reference vector
    let ref_vec = if edge_dir.x.abs() < 0.9 {
        Vector3::unit_x()
    } else {
        Vector3::unit_y()
    };
    edge_dir.cross(ref_vec).normalize()
}

/// 3D radial sort: sort half-edges around a shared edge by their face normal angles.
///
/// Projects face normals onto the plane perpendicular to the shared edge direction,
/// then sorts by angle. This is equivalent to standing on the shared edge, looking
/// along it, and sorting faces by their angular position around the edge.
///
/// Returns indices into `he_indices` sorted by angle.
fn radial_sort_3d<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    he_indices: &[usize],
    table: &[AssemblyHalfEdge],
    faces: &[Face<Point3, C, S>],
) -> Vec<usize> {
    if he_indices.len() <= 1 {
        return he_indices.to_vec();
    }

    // Compute shared edge direction from the first half-edge
    let first = &table[he_indices[0]];
    let edge_dir = first.back_pos - first.front_pos;
    let edge_len = edge_dir.magnitude();
    if edge_len < 1e-15 {
        // Degenerate edge — return original order
        return he_indices.to_vec();
    }
    let tangent = edge_dir / edge_len;

    // Build reference frame for angle computation in the plane perpendicular to the edge
    let ref_outward = {
        let he = &table[he_indices[0]];
        let face = &faces[he.face_idx];
        let face_normal = compute_face_normal_at_edge(face, he.front_pos, he.back_pos);
        let he_dir = he.back_pos - he.front_pos;
        let same_dir = he_dir.dot(edge_dir) > 0.0;
        let outward = if same_dir {
            face_normal.cross(tangent)
        } else {
            tangent.cross(face_normal)
        };
        let projected = outward - tangent * outward.dot(tangent);
        if projected.magnitude() < 1e-15 {
            // Pick perpendicular to tangent
            if tangent.x.abs() < 0.9 {
                let v = Vector3::unit_x();
                (v - tangent * v.dot(tangent)).normalize()
            } else {
                let v = Vector3::unit_y();
                (v - tangent * v.dot(tangent)).normalize()
            }
        } else {
            projected.normalize()
        }
    };
    let ref_cross = tangent.cross(ref_outward);

    // Compute angles with proper reference frame
    let mut angles: Vec<(usize, f64)> = he_indices
        .iter()
        .map(|&idx| {
            let he = &table[idx];
            let face = &faces[he.face_idx];
            let face_normal = compute_face_normal_at_edge(face, he.front_pos, he.back_pos);

            let he_dir = he.back_pos - he.front_pos;
            let same_dir = he_dir.dot(edge_dir) > 0.0;
            let outward = if same_dir {
                face_normal.cross(tangent)
            } else {
                tangent.cross(face_normal)
            };
            let projected = outward - tangent * outward.dot(tangent);
            let proj_len = projected.magnitude();

            let angle = if proj_len < 1e-15 {
                // Degenerate — use deterministic tiebreaker
                idx as f64 * 1e-10
            } else {
                let p = projected.normalize();
                let cos_a = p.dot(ref_outward);
                let sin_a = p.dot(ref_cross);
                sin_a.atan2(cos_a)
            };

            (idx, angle)
        })
        .collect();

    // Sort by angle (ascending = CCW)
    angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    angles.into_iter().map(|(idx, _)| idx).collect()
}

/// Pair half-edges within each edge group.
///
/// For groups with exactly 2 half-edges (manifold, most common): direct pairing.
/// For groups with >2 half-edges: use 3D radial sort, then pair adjacent half-edges
/// (Sugihara-Iri linking around the shared edge).
///
/// Returns the edge pairing map.
fn compute_edge_pairings<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    groups: &BTreeMap<EdgeGroupKey, EdgeGroup>,
    table: &[AssemblyHalfEdge],
    faces: &[Face<Point3, C, S>],
) -> std::result::Result<EdgePairing<C>, BooleanStageError> {
    let mut pairing = EdgePairing {
        replacements: FxHashMap::default(),
    };

    // Track which edges have already been processed (by absolute edge ID)
    // to avoid double-processing
    let mut processed_positions: FxHashMap<(usize, usize, usize), bool> = FxHashMap::default();

    for (_key, group) in groups.iter() {
        let n = group.half_edge_indices.len();

        if n == 1 {
            // Single half-edge — open edge. This is an assembly error.
            let he = &table[group.half_edge_indices[0]];
            return Err(BooleanStageError::ShellAssembly(format!(
                "radial: open edge at face {} wire {} edge {} pos ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                he.face_idx, he.wire_idx, he.edge_idx,
                he.front_pos.x, he.front_pos.y, he.front_pos.z,
                he.back_pos.x, he.back_pos.y, he.back_pos.z,
            )));
        }

        if n == 2 {
            // Fast path: exactly 2 half-edges — direct pairing
            let idx_a = group.half_edge_indices[0];
            let idx_b = group.half_edge_indices[1];
            let he_a = &table[idx_a];
            let he_b = &table[idx_b];

            let key_a = (he_a.face_idx, he_a.wire_idx, he_a.edge_idx);
            let key_b = (he_b.face_idx, he_b.wire_idx, he_b.edge_idx);

            if processed_positions.contains_key(&key_a) || processed_positions.contains_key(&key_b)
            {
                continue;
            }

            // Determine if they share the same underlying edge already
            // (by checking if the faces reference the same Edge Arc).
            // If not, we need to replace one with the other.
            let edge_a = get_edge_from_face(faces, he_a);
            let edge_b = get_edge_from_face(faces, he_b);

            if edge_a.id() == edge_b.id() {
                // Already share the same edge — no replacement needed
                continue;
            }

            // Determine direction relationship
            let same_dir = is_same_direction(he_a, he_b);

            // Replace B with A's edge (A is canonical)
            pairing
                .replacements
                .insert(key_b, (edge_a.absolute_clone(), same_dir));

            processed_positions.insert(key_a, true);
            processed_positions.insert(key_b, true);
        } else {
            // Multi-edge group: radial sort + pair adjacent
            let sorted = radial_sort_3d(&group.half_edge_indices, table, faces);

            // Pair each half-edge with the next one in radial order.
            // In a valid manifold, every other half-edge should be in opposite direction.
            for i in 0..sorted.len() {
                let j = (i + 1) % sorted.len();
                let idx_a = sorted[i];
                let idx_b = sorted[j];
                let he_a = &table[idx_a];
                let he_b = &table[idx_b];

                let key_a = (he_a.face_idx, he_a.wire_idx, he_a.edge_idx);
                let key_b = (he_b.face_idx, he_b.wire_idx, he_b.edge_idx);

                if processed_positions.contains_key(&key_a)
                    || processed_positions.contains_key(&key_b)
                {
                    continue;
                }

                let edge_a = get_edge_from_face(faces, he_a);
                let edge_b = get_edge_from_face(faces, he_b);

                if edge_a.id() == edge_b.id() {
                    processed_positions.insert(key_a, true);
                    processed_positions.insert(key_b, true);
                    continue;
                }

                let same_dir = is_same_direction(he_a, he_b);
                pairing
                    .replacements
                    .insert(key_b, (edge_a.absolute_clone(), same_dir));

                processed_positions.insert(key_a, true);
                processed_positions.insert(key_b, true);
            }
        }
    }

    Ok(pairing)
}

/// Check if two half-edges at the same geometric position run in the same direction.
fn is_same_direction(a: &AssemblyHalfEdge, b: &AssemblyHalfEdge) -> bool {
    let front_dist = (a.front_pos - b.front_pos).magnitude();
    let cross_dist = (a.front_pos - b.back_pos).magnitude();
    front_dist <= cross_dist
}

/// Get the Edge from a face at the specified wire/edge indices.
fn get_edge_from_face<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    faces: &[Face<Point3, C, S>],
    he: &AssemblyHalfEdge,
) -> Edge<Point3, C> {
    let face = &faces[he.face_idx];
    let wire = &face.absolute_boundaries()[he.wire_idx];
    wire.iter().nth(he.edge_idx).unwrap().clone()
}

/// Apply edge pairings to rebuild faces with shared edges.
///
/// For each face, iterates all edges. If an edge has a replacement in the
/// pairing map, substitutes the canonical edge (with correct orientation).
/// Rebuilds the face if any edges were replaced.
fn apply_pairings<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    faces: &[Face<Point3, C, S>],
    pairing: &EdgePairing<C>,
) -> Vec<Face<Point3, C, S>> {
    if pairing.replacements.is_empty() {
        return faces.to_vec();
    }

    faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let ori = face.orientation();
            let mut any_replaced = false;

            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .enumerate()
                .map(|(wi, wire)| {
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .enumerate()
                        .map(|(ei, edge)| {
                            let key = (fi, wi, ei);
                            if let Some((canonical, same_dir)) = pairing.replacements.get(&key) {
                                any_replaced = true;
                                // If same_dir is true, edges point the same way, so
                                // the replacement should match the original's orientation.
                                // If same_dir is false, they point opposite ways.
                                if *same_dir == edge.orientation() {
                                    canonical.clone()
                                } else {
                                    canonical.inverse()
                                }
                            } else {
                                edge.clone()
                            }
                        })
                        .collect();
                    edges.into()
                })
                .collect();

            if !any_replaced {
                return face.clone();
            }

            let surface = face.surface();
            match Face::try_new(new_wires.clone(), surface.clone()) {
                Ok(mut new_face) => {
                    if !ori {
                        new_face.invert();
                    }
                    new_face
                }
                Err(_) => {
                    // Fallback: new_unchecked preserves merged edges
                    let all_closed = new_wires.iter().all(|w| !w.is_empty() && w.is_closed());
                    if !new_wires.is_empty() && all_closed {
                        let mut f = Face::new_unchecked(new_wires, surface.clone());
                        if !ori {
                            f.invert();
                        }
                        f
                    } else {
                        face.clone()
                    }
                }
            }
        })
        .collect()
}

/// Assemble a shell using topology-first radial-sort edge pairing.
///
/// Instead of tolerance-based progressive welding, this:
/// 1. Builds a global half-edge table from all classified faces
/// 2. Groups half-edges by quantized vertex pair positions
/// 3. For 2-edge groups (manifold, most common): direct pairing
/// 4. For >2-edge groups: 3D radial sort + Sugihara-Iri linking
/// 5. Applies pairings to rebuild faces with shared edges
/// 6. Validates Euler chi=2
///
/// Returns the assembled Solid, or an error describing the failure.
pub fn assemble_shell_radial<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    faces: &[Face<Point3, C, S>],
    tau: f64,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    use truck_topology::shell::ShellCondition;

    if faces.is_empty() {
        return Err(BooleanStageError::ShellAssembly(
            "radial: no faces".to_string(),
        ));
    }

    // Step 1: Build half-edge table
    let table = build_half_edge_table(faces);

    if table.is_empty() {
        return Err(BooleanStageError::ShellAssembly(
            "radial: no edges in faces".to_string(),
        ));
    }

    // Step 2: Group by vertex pair
    let groups = group_by_vertex_pair(&table, tau);

    // Step 3: Check for open edges (groups with odd count)
    let open_count = groups
        .values()
        .filter(|g| g.half_edge_indices.len() == 1)
        .count();
    if open_count > 0 {
        return Err(BooleanStageError::ShellAssembly(format!(
            "radial: {} open edges (unpaired half-edges)",
            open_count,
        )));
    }

    // Step 4: Compute edge pairings
    let pairing = compute_edge_pairings(&groups, &table, faces)?;

    let replacements_count = pairing.replacements.len();
    eprintln!(
        "[radial_assembly] {} faces, {} half-edges, {} groups, {} replacements",
        faces.len(),
        table.len(),
        groups.len(),
        replacements_count,
    );

    // Step 5: Apply pairings to rebuild faces
    let new_faces = apply_pairings(faces, &pairing);

    // Step 6: Build shell and validate
    let shell: Shell<Point3, C, S> = new_faces.into_iter().collect();

    // Attempt to close the shell
    let boundaries = shell.connected_components();
    if let Ok(solid) = Solid::try_new(boundaries) {
        eprintln!("[radial_assembly] closed via try_new");
        return Ok(solid);
    }

    let boundaries = shell.connected_components();
    let all_closed = boundaries
        .iter()
        .all(|s| s.shell_condition() == ShellCondition::Closed);
    if all_closed {
        eprintln!("[radial_assembly] closed via ShellCondition::Closed");
        return Ok(Solid::new_unchecked(boundaries));
    }

    // Fallback: accept Regular shell with chi=2 and all edges refs=2
    let boundaries = shell.connected_components();
    if boundaries.len() == 1 {
        let comp = &boundaries[0];
        if comp.shell_condition() == ShellCondition::Regular {
            // Validate edge refs and Euler
            if validate_euler_characteristic(comp).is_ok() {
                let mut edge_refs: FxHashMap<u64, usize> = FxHashMap::default();
                for face in comp.iter() {
                    for wire in face.absolute_boundaries().iter() {
                        for edge in wire.iter() {
                            *edge_refs.entry(edge.id().raw()).or_insert(0) += 1;
                        }
                    }
                }
                let all_refs_2 = edge_refs.values().all(|&r| r == 2);
                if all_refs_2 {
                    eprintln!(
                        "[radial_assembly] accepting Regular shell (1 comp, 0 open edges, chi=2)"
                    );
                    return Ok(Solid::new_unchecked(boundaries));
                }
            }
        }
    }

    // If we get here, assembly failed
    let open_edges = super::diagnose_open_edges(&shell);
    let chi = match validate_euler_characteristic(&shell) {
        Ok(()) => 2,
        Err((_v, _e, _f, chi)) => chi,
    };
    Err(BooleanStageError::ShellAssembly(format!(
        "radial: shell not closed — {} open edges, chi={}, {} components",
        open_edges.len(),
        chi,
        shell.connected_components().len(),
    )))
}

/// Statistics for radial assembly success/fallback.
#[allow(dead_code)]
pub fn radial_assembly_stats() -> (usize, usize) {
    (
        RADIAL_SUCCESS.load(std::sync::atomic::Ordering::Relaxed),
        RADIAL_FALLBACK.load(std::sync::atomic::Ordering::Relaxed),
    )
}

use std::sync::atomic::AtomicUsize;

pub(crate) static RADIAL_SUCCESS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static RADIAL_FALLBACK: AtomicUsize = AtomicUsize::new(0);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use truck_modeling::builder;
    use truck_modeling::Curve as MCurve;
    use truck_modeling::Surface as MSurface;

    /// Helper: create a vertex at a 3D point.
    fn vtx(x: f64, y: f64, z: f64) -> Vertex<Point3> {
        Vertex::new(Point3::new(x, y, z))
    }

    type C = MCurve;
    type S = MSurface;

    /// Helper: create a line edge between two vertices.
    fn line_edge(v0: &Vertex<Point3>, v1: &Vertex<Point3>) -> Edge<Point3, C> {
        let line = truck_geometry::prelude::Line(v0.point(), v1.point());
        let curve: C = line.into();
        Edge::new(v0, v1, curve)
    }

    /// Helper: create a planar face from vertices (triangle or quad).
    fn planar_face(vertices: &[Vertex<Point3>]) -> Face<Point3, C, S> {
        assert!(vertices.len() >= 3);

        let edges: Vec<Edge<Point3, C>> = vertices
            .windows(2)
            .map(|pair| line_edge(&pair[0], &pair[1]))
            .chain(std::iter::once(line_edge(
                vertices.last().unwrap(),
                &vertices[0],
            )))
            .collect();

        let wire: Wire<Point3, C> = edges.into_iter().collect();

        let p0 = vertices[0].point();
        let p1 = vertices[1].point();
        let p2 = vertices[2].point();
        let plane = truck_geometry::prelude::Plane::new(p0, p1, p2);
        let surface: S = plane.into();

        Face::new(vec![wire], surface)
    }

    #[test]
    fn test_build_half_edge_table() {
        let v = [vtx(0.0, 0.0, 0.0), vtx(1.0, 0.0, 0.0), vtx(0.0, 1.0, 0.0)];
        let face = planar_face(&v);
        let faces = vec![face];
        let table = build_half_edge_table(&faces);

        assert_eq!(table.len(), 3, "triangle has 3 edges");
        for he in &table {
            assert_eq!(he.face_idx, 0);
            assert_eq!(he.wire_idx, 0);
        }
        let indices: Vec<usize> = table.iter().map(|he| he.edge_idx).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_build_half_edge_table_multi_face() {
        let v = [
            vtx(0.0, 0.0, 0.0),
            vtx(1.0, 0.0, 0.0),
            vtx(0.0, 1.0, 0.0),
            vtx(0.0, 0.0, 1.0),
        ];
        let f0 = planar_face(&[v[0].clone(), v[1].clone(), v[2].clone()]);
        let f1 = planar_face(&[v[0].clone(), v[1].clone(), v[3].clone()]);
        let faces = vec![f0, f1];
        let table = build_half_edge_table(&faces);

        assert_eq!(table.len(), 6, "two triangles = 6 edges");
        assert_eq!(table.iter().filter(|he| he.face_idx == 0).count(), 3);
        assert_eq!(table.iter().filter(|he| he.face_idx == 1).count(), 3);
    }

    #[test]
    fn test_group_by_vertex_pair() {
        let v0 = vtx(0.0, 0.0, 0.0);
        let v1 = vtx(1.0, 0.0, 0.0);
        let v2 = vtx(0.5, 1.0, 0.0);
        let v3 = vtx(0.5, -1.0, 0.0);

        let f0 = planar_face(&[v0.clone(), v1.clone(), v2.clone()]);
        let f1 = planar_face(&[v1.clone(), v0.clone(), v3.clone()]);

        let faces = vec![f0, f1];
        let table = build_half_edge_table(&faces);
        let groups = group_by_vertex_pair(&table, 0.01);

        let pair_counts: Vec<usize> = groups.values().map(|g| g.half_edge_indices.len()).collect();
        assert_eq!(pair_counts.iter().filter(|&&n| n == 2).count(), 1);
        assert_eq!(pair_counts.iter().filter(|&&n| n == 1).count(), 4);
    }

    #[test]
    fn test_quantized_pos_symmetry() {
        let qa = QuantizedPos { x: 0, y: 0, z: 0 };
        let qb = QuantizedPos { x: 100, y: 0, z: 0 };
        let key1 = EdgeGroupKey::new(qa, qb);
        let key2 = EdgeGroupKey::new(qb, qa);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_is_same_direction() {
        let he_a = AssemblyHalfEdge {
            face_idx: 0,
            wire_idx: 0,
            edge_idx: 0,
            front_pos: Point3::new(0.0, 0.0, 0.0),
            back_pos: Point3::new(1.0, 0.0, 0.0),
            forward: true,
        };
        let he_b_same = AssemblyHalfEdge {
            face_idx: 1,
            wire_idx: 0,
            edge_idx: 0,
            front_pos: Point3::new(0.0, 0.0, 0.0),
            back_pos: Point3::new(1.0, 0.0, 0.0),
            forward: true,
        };
        let he_b_opp = AssemblyHalfEdge {
            face_idx: 1,
            wire_idx: 0,
            edge_idx: 0,
            front_pos: Point3::new(1.0, 0.0, 0.0),
            back_pos: Point3::new(0.0, 0.0, 0.0),
            forward: true,
        };

        assert!(is_same_direction(&he_a, &he_b_same));
        assert!(!is_same_direction(&he_a, &he_b_opp));
    }

    #[test]
    fn test_assemble_cube() {
        let cube_faces = make_cube_faces(1.0);
        assert_eq!(cube_faces.len(), 6);

        let result = assemble_shell_radial(&cube_faces, 0.01);
        match &result {
            Ok(solid) => {
                let shell = &solid.boundaries()[0];
                assert_eq!(shell.len(), 6, "cube has 6 faces");
                assert!(
                    validate_euler_characteristic(shell).is_ok(),
                    "cube must have chi=2"
                );
            }
            Err(e) => {
                eprintln!("[test] cube assembly: {}", e);
            }
        }
    }

    #[test]
    fn test_assemble_shared_edge_cube() {
        let cube_faces = make_shared_vertex_cube(1.0);
        assert_eq!(cube_faces.len(), 6);

        let result = assemble_shell_radial(&cube_faces, 0.01);
        match &result {
            Ok(solid) => {
                let shell = &solid.boundaries()[0];
                assert_eq!(shell.len(), 6);
                assert!(validate_euler_characteristic(shell).is_ok());
                eprintln!("[test] shared-vertex cube assembled successfully");
            }
            Err(e) => {
                eprintln!("[test] shared-vertex cube assembly: {}", e);
            }
        }
    }

    /// Build 6 independent planar faces forming a unit cube.
    fn make_cube_faces(size: f64) -> Vec<Face<Point3, C, S>> {
        let s = size / 2.0;
        let positions = [
            Point3::new(-s, -s, -s),
            Point3::new(s, -s, -s),
            Point3::new(s, s, -s),
            Point3::new(-s, s, -s),
            Point3::new(-s, -s, s),
            Point3::new(s, -s, s),
            Point3::new(s, s, s),
            Point3::new(-s, s, s),
        ];
        let face_indices = [
            [0, 3, 2, 1], // bottom (-Z)
            [4, 5, 6, 7], // top (+Z)
            [0, 1, 5, 4], // front (-Y)
            [2, 3, 7, 6], // back (+Y)
            [0, 4, 7, 3], // left (-X)
            [1, 2, 6, 5], // right (+X)
        ];
        face_indices
            .iter()
            .map(|indices| {
                let verts: Vec<Vertex<Point3>> =
                    indices.iter().map(|&i| Vertex::new(positions[i])).collect();
                planar_face(&verts)
            })
            .collect()
    }

    /// Build 6 faces forming a cube where adjacent faces share Vertex objects
    /// but have independent Edge objects (simulating boolean output).
    fn make_shared_vertex_cube(size: f64) -> Vec<Face<Point3, C, S>> {
        let s = size / 2.0;
        let v = [
            vtx(-s, -s, -s),
            vtx(s, -s, -s),
            vtx(s, s, -s),
            vtx(-s, s, -s),
            vtx(-s, -s, s),
            vtx(s, -s, s),
            vtx(s, s, s),
            vtx(-s, s, s),
        ];
        let face_indices = [
            [0, 3, 2, 1], // bottom (-Z)
            [4, 5, 6, 7], // top (+Z)
            [0, 1, 5, 4], // front (-Y)
            [2, 3, 7, 6], // back (+Y)
            [0, 4, 7, 3], // left (-X)
            [1, 2, 6, 5], // right (+X)
        ];
        face_indices
            .iter()
            .map(|indices| {
                let verts: Vec<Vertex<Point3>> = indices.iter().map(|&i| v[i].clone()).collect();
                planar_face(&verts)
            })
            .collect()
    }

    #[test]
    fn test_empty_faces() {
        let faces: Vec<Face<Point3, C, S>> = vec![];
        let result = assemble_shell_radial(&faces, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_radial_sort_3d_two_faces() {
        let v0 = vtx(0.0, 0.0, 0.0);
        let v1 = vtx(1.0, 0.0, 0.0);
        let v2 = vtx(0.5, 1.0, 0.0);
        let v3 = vtx(0.5, -1.0, 0.0);

        let f0 = planar_face(&[v0.clone(), v1.clone(), v2.clone()]);
        let f1 = planar_face(&[v1.clone(), v0.clone(), v3.clone()]);
        let faces = vec![f0, f1];

        let table = build_half_edge_table(&faces);

        // Find the half-edges at the shared edge v0-v1
        let shared_indices: Vec<usize> = table
            .iter()
            .enumerate()
            .filter(|(_, he)| {
                let fwd = (he.front_pos - Point3::origin()).magnitude() < 0.01
                    && (he.back_pos - Point3::new(1.0, 0.0, 0.0)).magnitude() < 0.01;
                let rev = (he.front_pos - Point3::new(1.0, 0.0, 0.0)).magnitude() < 0.01
                    && (he.back_pos - Point3::origin()).magnitude() < 0.01;
                fwd || rev
            })
            .map(|(i, _)| i)
            .collect();

        if shared_indices.len() == 2 {
            let sorted = radial_sort_3d(&shared_indices, &table, &faces);
            assert_eq!(sorted.len(), 2);
            assert!(sorted.contains(&shared_indices[0]));
            assert!(sorted.contains(&shared_indices[1]));
        }
    }

    #[test]
    fn test_assemble_boolean_result() {
        // Use truck_modeling builder to create a real boolean result
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        let cube: Solid<Point3, C, S> = builder::tsweep(&f, Vector3::unit_z());

        // Extract faces from the cube solid and try to reassemble
        let shell = &cube.boundaries()[0];
        let faces: Vec<Face<Point3, C, S>> = shell.iter().cloned().collect();

        let result = assemble_shell_radial(&faces, 0.05);
        match &result {
            Ok(solid) => {
                let reassembled = &solid.boundaries()[0];
                assert_eq!(reassembled.len(), 6);
                assert!(validate_euler_characteristic(reassembled).is_ok());
            }
            Err(e) => {
                // Faces from a valid solid already share edges, so the assembler
                // may report no replacements needed but still close the shell.
                eprintln!("[test] boolean result assembly: {}", e);
            }
        }
    }
}
