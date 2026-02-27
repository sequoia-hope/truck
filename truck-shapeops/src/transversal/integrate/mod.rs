use crate::alternative::Alternative;

use super::*;
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

pub(crate) mod radial_assembly;

/// Per-stage tolerance configuration for boolean operations.
///
/// Different stages of the boolean pipeline have different precision needs:
/// - Mesh collision needs coarser tolerance (speed)
/// - Intersection curves need model-level precision
/// - Vertex welding needs wider tolerance to close gaps
/// - Coplanar detection needs wider tolerance for normal/distance comparison
///
/// Using a single `tol` for all stages causes failures when the tolerance
/// is appropriate for one stage but too large/small for another.
#[derive(Clone, Debug)]
pub struct BooleanTolerance {
    /// Main coincidence/intersection tolerance (model precision).
    pub tau_model: f64,
    /// Mesh collision resolution tolerance (triangulation accuracy).
    pub tau_mesh: f64,
    /// Vertex unification tolerance in `weld_coincident_edges`.
    pub tau_weld: f64,
    /// Coplanar face detection threshold (normal parallelism + plane distance).
    pub tau_coplanar: f64,
    /// IC-on-boundary filter tolerance (was inline `tol * 0.5`).
    pub tau_boundary: f64,
    /// Phase 1 midpoint clustering tolerance in `weld_coincident_edges` (was inline `tol * 5.0`).
    pub tau_edge_cluster: f64,
    /// Minimum parametric face area threshold in `divide_one_face` (was inline `tol * tol`).
    pub tau_area: f64,
}

impl BooleanTolerance {
    /// All stages use the same tolerance. Matches legacy single-tol behavior.
    #[deprecated(note = "Use from_model_tol() for proper per-stage tolerance scaling")]
    #[allow(dead_code)]
    pub fn uniform(tol: f64) -> Self {
        Self {
            tau_model: tol,
            tau_mesh: tol,
            tau_weld: tol,
            tau_coplanar: tol,
            tau_boundary: tol,
            tau_edge_cluster: tol,
            tau_area: tol * tol,
        }
    }

    /// Derive per-stage tolerances from a model tolerance.
    ///
    /// Each stage gets a tolerance scaled to its specific needs:
    /// - `tau_mesh`: same as model (triangulation needs model-level precision)
    /// - `tau_weld`: 0.4x model (conservative: slightly wider than the internal
    ///   default of `tol * 0.2`, but well below the feature-size failure threshold
    ///   of ~0.10 * min_edge. A 2x multiplier was too aggressive and merged
    ///   vertices across small features like narrow bosses.)
    /// - `tau_coplanar`: same as model — the coplanar distance check uses `tol`
    ///   directly (not squared), so a multiplier causes false coplanar detection
    ///   when faces are merely close (e.g., separated by cut_eps offset).
    pub fn from_model_tol(tau_model: f64) -> Self {
        Self {
            tau_model,
            tau_mesh: tau_model,
            tau_weld: 0.4 * tau_model,
            tau_coplanar: tau_model,
            tau_boundary: 0.5 * tau_model,
            tau_edge_cluster: 5.0 * tau_model,
            tau_area: tau_model * tau_model,
        }
    }
}

/// Only solids consisting of faces whose surface is implemented this trait can be used for set operations.
pub trait ShapeOpsSurface:
    ParametricSurface3D
    + ParameterDivision2D
    + SearchParameter<D2, Point = Point3>
    + SearchNearestParameter<D2, Point = Point3>
    + Invertible
    + From<Plane>
    + Send
    + Sync
{
}
impl<S> ShapeOpsSurface for S where
    S: ParametricSurface3D
        + ParameterDivision2D
        + SearchParameter<D2, Point = Point3>
        + SearchNearestParameter<D2, Point = Point3>
        + Invertible
        + From<Plane>
        + Send
        + Sync
{
}

/// Only solids consisting of edges whose curve is implemented this trait can be used for set operations.
pub trait ShapeOpsCurve<S: ShapeOpsSurface>:
    ParametricCurve3D
    + ParameterDivision1D<Point = Point3>
    + Cut
    + Invertible
    + From<IntersectionCurve<BSplineCurve<Point3>, S, S>>
    + From<Line<Point3>>
    + SearchParameter<D1, Point = Point3>
    + SearchNearestParameter<D1, Point = Point3>
    + Send
    + Sync
{
}
impl<C, S: ShapeOpsSurface> ShapeOpsCurve<S> for C where
    C: ParametricCurve3D
        + ParameterDivision1D<Point = Point3>
        + Cut
        + Invertible
        + From<IntersectionCurve<BSplineCurve<Point3>, S, S>>
        + From<Line<Point3>>
        + SearchParameter<D1, Point = Point3>
        + SearchNearestParameter<D1, Point = Point3>
        + Send
        + Sync
{
}

/// Ray-cast from a point against a triangulated shell using robust geometric
/// predicates (Shewchuk's adaptive precision arithmetic).
///
/// Returns the signed crossing count, or `None` if any triangle produces a
/// degenerate (edge-grazing) configuration.
pub(crate) fn try_ray_cast(
    pt: Point3,
    dir: Vector3,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> Option<isize> {
    use super::robust_classify::robust_ray_triangle_cross;

    let ray_origin = [pt.x, pt.y, pt.z];
    let ray_dir = [dir.x, dir.y, dir.z];

    poly_shell.iter().try_fold(0isize, |count, face| {
        let poly = face.surface()?;
        let positions = poly.positions();
        let mut face_count = 0isize;
        // Iterate all faces (tri, quad, n-gon) and fan-triangulate
        for face_verts in poly.face_iter() {
            for i in 2..face_verts.len() {
                let p0 = positions[face_verts[0].pos];
                let p1 = positions[face_verts[i - 1].pos];
                let p2 = positions[face_verts[i].pos];
                let tri = [[p0.x, p0.y, p0.z], [p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]];
                let crossing = robust_ray_triangle_cross(ray_origin, ray_dir, tri)?;
                // Determine sign: if ray direction dot triangle normal is positive,
                // count +1; if negative, count -1.
                if crossing == 1 {
                    let e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
                    let e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
                    let normal = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let dot = normal[0] * dir.x + normal[1] * dir.y + normal[2] * dir.z;
                    if dot > 0.0 {
                        face_count += 1;
                    } else {
                        face_count -= 1;
                    }
                }
            }
        }
        Some(count + face_count)
    })
}

/// Compute the maximum extent (bounding box diagonal) of a triangulated shell.
/// Used for scale-adaptive perturbation in ray-cast classification.
pub(crate) fn compute_shell_extent_poly(
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> f64 {
    let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);
    for face in poly_shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for v in wire.vertex_iter() {
                let pt = v.point();
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                min_z = min_z.min(pt.z);
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
                max_z = max_z.max(pt.z);
            }
        }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    dx.max(dy).max(dz).max(1.0)
}

/// Irrational ray directions that avoid grid alignment in triangulated meshes.
/// Each direction has a dominant axis component plus small irrational offsets.
/// 8 directions provide robust majority voting for corner-coplanar geometry.
pub(crate) fn irrational_ray_dirs() -> [Vector3; 8] {
    let sqrt2 = std::f64::consts::SQRT_2;
    let sqrt3 = 3.0f64.sqrt();
    let sqrt5 = 5.0f64.sqrt();
    let sqrt7 = 7.0f64.sqrt();
    let sqrt11 = 11.0f64.sqrt();
    let sqrt13 = 13.0f64.sqrt();
    let sqrt17 = 17.0f64.sqrt();
    let sqrt19 = 19.0f64.sqrt();
    let sqrt23 = 23.0f64.sqrt();
    let sqrt29 = 29.0f64.sqrt();
    [
        Vector3::new(1.0, sqrt2 / 10.0, sqrt3 / 10.0),
        Vector3::new(sqrt2 / 10.0, 1.0, sqrt5 / 10.0),
        Vector3::new(sqrt3 / 10.0, sqrt5 / 10.0, 1.0),
        Vector3::new(sqrt7 / 10.0, sqrt11 / 10.0, sqrt13 / 10.0),
        // Additional directions for corner-coplanar robustness
        Vector3::new(1.0, sqrt17 / 10.0, -sqrt19 / 10.0),
        Vector3::new(-sqrt23 / 10.0, 1.0, sqrt29 / 10.0),
        Vector3::new(sqrt19 / 10.0, -sqrt17 / 10.0, 1.0),
        Vector3::new(sqrt29 / 10.0, sqrt23 / 10.0, sqrt17 / 10.0),
    ]
}

/// Compute a geometric normal from boundary vertices using Newell's method.
/// Returns None if the face has fewer than 3 vertices or the normal is degenerate.
pub(crate) fn geometric_face_normal(verts: &[Point3]) -> Option<Vector3> {
    if verts.len() < 3 {
        return None;
    }
    // Newell's method: robust for non-planar polygons
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..verts.len() {
        let curr = verts[i];
        let next = verts[(i + 1) % verts.len()];
        nx += (curr.y - next.y) * (curr.z + next.z);
        ny += (curr.z - next.z) * (curr.x + next.x);
        nz += (curr.x - next.x) * (curr.y + next.y);
    }
    let n = Vector3::new(nx, ny, nz);
    let mag = n.magnitude();
    if mag < 1e-15 {
        return None;
    }
    Some(n / mag)
}

/// Ray-cast a face against a triangulated shell to determine inside/outside.
///
/// Uses scale-adaptive perturbation and irrational ray directions with majority
/// voting to avoid false classifications from grid-aligned triangulation.
///
/// When a BVH is provided, uses R-tree acceleration to skip triangles whose
/// bounding boxes don't intersect the ray. Falls back to linear scan otherwise.
fn ray_cast_classify<C, S>(
    face: &Face<Point3, C, S>,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
    bvh: Option<&rstar::RTree<bvh::BvhTriangle>>,
) -> Option<isize> {
    let verts: Vec<_> = face.boundaries()[0]
        .vertex_iter()
        .map(|v| v.point())
        .collect();

    // Scale perturbation with model extent so it stays effective at any scale.
    let extent = compute_shell_extent_poly(poly_shell);
    let scale = extent.max(1.0);
    let perturb = Vector3::new(
        1.4142135623730951e-6 * scale,
        1.7320508075688772e-6 * scale,
        2.2360679774997896e-6 * scale,
    );

    let dirs = irrational_ray_dirs();

    // Choose ray-cast function: BVH-accelerated or linear fallback.
    let cast_ray = |pt: Point3, d: Vector3| -> Option<isize> {
        if let Some(bvh) = bvh {
            bvh::try_ray_cast_bvh(pt, d, bvh)
        } else {
            try_ray_cast(pt, d, poly_shell)
        }
    };

    // Majority vote: cast all 8 irrational rays, take majority (need >=3 agreeing).
    let majority_vote = |pt: Point3| -> Option<isize> {
        let mut inside = 0u32;
        let mut outside = 0u32;
        for &d in &dirs {
            if let Some(c) = cast_ray(pt, d) {
                // Use parity (odd/even) of the absolute crossing count, not the
                // sign. The parity is always correct: odd crossings = inside,
                // even crossings = outside.
                if c.unsigned_abs() % 2 == 1 {
                    inside += 1;
                } else {
                    outside += 1;
                }
            }
        }
        if inside >= 3 {
            Some(1)
        } else if outside >= 3 {
            Some(0)
        } else {
            None
        }
    };

    // Bidirectional vote: try both +perturb and -perturb. If they agree, the
    // result is reliable. If they disagree, the point is on the other shell's
    // boundary surface — classify as outside (0).
    let bidirectional_vote = |base: Point3| -> Option<isize> {
        let pt_pos = base + perturb;
        let pt_neg = base - perturb;
        let vote_pos = majority_vote(pt_pos);
        let vote_neg = majority_vote(pt_neg);
        match (vote_pos, vote_neg) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(_a), Some(_b)) => Some(0), // Disagree → on boundary → outside
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    };

    // Strategy 1: Face centroid — best for split faces where vertices sit on
    // splitting edges. Skip for ring faces (faces with holes).
    let has_holes = face.boundaries().len() > 1;
    let n = verts.len().max(1) as f64;
    let centroid_v = verts.iter().fold(Vector3::new(0.0, 0.0, 0.0), |a, &p| {
        a + (p - Point3::origin())
    }) / n;
    let centroid = Point3::origin() + centroid_v;

    if !has_holes {
        if let Some(c) = bidirectional_vote(centroid) {
            return Some(c);
        }
    }

    // Strategy 2: Boundary vertices — try each vertex (different grid positions).
    for &v in &verts {
        if let Some(c) = bidirectional_vote(v) {
            return Some(c);
        }
    }

    // Strategy 3: Escalated perturbation — break corner-coplanar degeneracy
    // by using 1000x larger offsets to move test points well away from surfaces.
    let large_perturb = Vector3::new(1.0e-3 * scale, 1.2e-3 * scale, 1.4e-3 * scale);
    let escalated_vote = |base: Point3| -> Option<isize> {
        let pt_pos = base + large_perturb;
        let pt_neg = base - large_perturb;
        let vote_pos = majority_vote(pt_pos);
        let vote_neg = majority_vote(pt_neg);
        match (vote_pos, vote_neg) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(_), Some(_)) => Some(0), // Disagree → on boundary → outside
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    };
    if let Some(c) = escalated_vote(centroid) {
        return Some(c);
    }
    for &v in &verts {
        if let Some(c) = escalated_vote(v) {
            return Some(c);
        }
    }

    // Strategy 4: Face-normal ray — cast along geometric normal of the face.
    {
        let normal = geometric_face_normal(&verts);
        if let Some(dir) = normal {
            let origin = centroid + dir * (1e-4 * scale);
            if let Some(c) = cast_ray(origin, dir) {
                return Some(if c.unsigned_abs() % 2 == 1 { 1 } else { 0 });
            }
            // Try opposite direction
            if let Some(c) = cast_ray(origin, -dir) {
                return Some(if c.unsigned_abs() % 2 == 1 { 1 } else { 0 });
            }
        }
    }

    // Last resort: accept any single non-None result from any direction.
    let centroid_perturbed = centroid + perturb;
    for &d in &dirs {
        if let Some(c) = cast_ray(centroid_perturbed, d) {
            return Some(if c.unsigned_abs() % 2 == 1 { 1 } else { 0 });
        }
    }

    None
}

/// Edge-neighbor propagation for faces where ray-cast failed.
///
/// Iteratively classifies unresolved faces by majority-voting their edge
/// adjacency with already-classified faces. This handles corner-coplanar
/// geometry where all ray-cast strategies fail but neighboring faces are
/// correctly classified.
fn classify_by_edge_neighbors<C: Clone, S: Clone>(
    unresolved: Vec<Face<Point3, C, S>>,
    and_shell: &mut Shell<Point3, C, S>,
    or_shell: &mut Shell<Point3, C, S>,
    report: Option<&mut diagnostics::EdgeNeighborReport>,
) -> std::result::Result<(), BooleanStageError> {
    use rustc_hash::FxHashSet;
    let initial_count = unresolved.len();
    eprintln!(
        "[classify] {} faces unresolved after ray-cast, trying edge-neighbor propagation",
        initial_count
    );
    let mut remaining = unresolved;
    let mut rounds_executed = 0u32;
    let mut faces_resolved = 0usize;
    for _round in 0..10 {
        if remaining.is_empty() {
            break;
        }
        rounds_executed += 1;
        let mut and_eids: FxHashSet<EdgeID<C>> = FxHashSet::default();
        for f in and_shell.iter() {
            for wire in f.boundaries() {
                for edge in wire.edge_iter() {
                    and_eids.insert(edge.id());
                }
            }
        }
        let mut or_eids: FxHashSet<EdgeID<C>> = FxHashSet::default();
        for f in or_shell.iter() {
            for wire in f.boundaries() {
                for edge in wire.edge_iter() {
                    or_eids.insert(edge.id());
                }
            }
        }
        let mut next = Vec::new();
        let mut progress = false;
        for face in remaining {
            let mut and_adj = 0usize;
            let mut or_adj = 0usize;
            for wire in face.boundaries() {
                for edge in wire.edge_iter() {
                    if and_eids.contains(&edge.id()) {
                        and_adj += 1;
                    }
                    if or_eids.contains(&edge.id()) {
                        or_adj += 1;
                    }
                }
            }
            if and_adj > or_adj {
                and_shell.push(face);
                progress = true;
                faces_resolved += 1;
            } else if or_adj > and_adj {
                or_shell.push(face);
                progress = true;
                faces_resolved += 1;
            } else if and_adj > 0 {
                // Tie with both And/Or neighbors: prefer And (intersection boundary).
                and_shell.push(face);
                progress = true;
                faces_resolved += 1;
            } else {
                next.push(face);
            }
        }
        remaining = next;
        if !progress {
            break;
        }
    }
    if let Some(rpt) = report {
        rpt.faces_entered = initial_count;
        rpt.rounds_executed = rounds_executed as usize;
        rpt.faces_resolved = faces_resolved;
    }
    if !remaining.is_empty() {
        eprintln!(
            "[classify] {} faces still unresolved after edge-neighbor propagation",
            remaining.len()
        );
        return Err(BooleanStageError::Classification);
    }
    eprintln!(
        "[classify] edge-neighbor propagation resolved all {} faces",
        initial_count
    );
    Ok(())
}

type AltCurveShell<C, S> =
    Shell<Point3, Alternative<C, IntersectionCurve<PolylineCurve<Point3>, S, S>>, S>;

fn altshell_to_shell<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    altshell: &AltCurveShell<C, S>,
    tol: f64,
) -> Option<Shell<Point3, C, S>> {
    altshell.try_mapped(
        |p| Some(*p),
        |c| match c {
            Alternative::FirstType(c) => Some(c.clone()),
            Alternative::SecondType(ic) => {
                let range = ic.range_tuple();
                let bsp = BSplineCurve::cubic_approximation(ic, range, tol, tol * 1000.0, 100)
                    .or_else(|| BSplineCurve::quadratic_approximation(ic, range, tol, 100))?;
                Some(
                    IntersectionCurve::new(ic.surface0().clone(), ic.surface1().clone(), bsp)
                        .into(),
                )
            }
        },
        |s| Some(s.clone()),
    )
}

/// Structured error type for boolean pipeline stages.
#[derive(Debug)]
pub enum BooleanStageError {
    /// loops_store creation failed (intersection curve construction)
    LoopsStoreCreation,
    /// Face division failed (splitting faces along intersection curves)
    FaceDivision,
    /// Ray-cast classification of a face fragment was ambiguous
    Classification,
    /// Final shell assembly failed (Solid::try_new returned Err)
    ShellAssembly(String),
}

impl std::fmt::Display for BooleanStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoopsStoreCreation => write!(f, "loops store creation failed"),
            Self::FaceDivision => write!(f, "face division failed"),
            Self::Classification => write!(f, "face classification ambiguous"),
            Self::ShellAssembly(detail) => write!(f, "shell assembly failed: {}", detail),
        }
    }
}

/// The 4 classified face buckets from a shell pair boolean operation.
struct ClassifiedShellBuckets<P, C, S> {
    /// shell0 faces inside shell1 (And)
    and0: Shell<P, C, S>,
    /// shell0 faces outside shell1 (Or)
    or0: Shell<P, C, S>,
    /// shell1 faces inside shell0 (And)
    and1: Shell<P, C, S>,
    /// shell1 faces outside shell0 (Or)
    or1: Shell<P, C, S>,
}

fn classify_one_pair_of_shells_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tols: &BooleanTolerance,
    mut diag: Option<&mut BooleanDiagnostics>,
) -> std::result::Result<ClassifiedShellBuckets<Point3, C, S>, BooleanStageError> {
    nonpositive_tolerance!(tols.tau_model);
    let _total_start = std::time::Instant::now();
    eprintln!(
        "[classify] input: shell0={} faces, shell1={} faces",
        shell0.len(),
        shell1.len(),
    );
    let poly_shell0 = shell0.triangulation(tols.tau_mesh);
    let poly_shell1 = shell1.triangulation(tols.tau_mesh);
    // Build BVH (R-tree) once per shell for O(log n) ray-cast acceleration.
    let bvh0 = bvh::build_triangle_bvh(&poly_shell0);
    let bvh1 = bvh::build_triangle_bvh(&poly_shell1);
    let altshell0: AltCurveShell<C, S> =
        shell0.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let altshell1: AltCurveShell<C, S> =
        shell1.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let loops_store::LoopsStoreQuadruple {
        geom_loops_store0: loops_store0,
        geom_loops_store1: loops_store1,
        coplanar_faces0,
        coplanar_faces1,
        contained_faces0,
        contained_faces1: _contained_faces1,
        ..
    } = loops_store::create_loops_stores(
        &altshell0,
        &poly_shell0,
        &altshell1,
        &poly_shell1,
        tols.tau_model,
        Some(tols.tau_coplanar),
        tols.tau_boundary,
    )
    .ok_or(BooleanStageError::LoopsStoreCreation)?;
    if !contained_faces0.is_empty() {
        eprintln!("[injection] contained_faces0={:?}", contained_faces0,);
    }
    let _loops_store_elapsed = _total_start.elapsed();
    {
        for (fi, loops) in loops_store1.iter().enumerate() {
            let total_edges: usize = loops.iter().map(|w| w.len()).sum();
            let _total_wires = loops.len();
            if total_edges > 4 {
                let wire_info: Vec<String> = loops
                    .iter()
                    .enumerate()
                    .map(|(wi, w)| {
                        let verts: Vec<String> = w
                            .vertex_iter()
                            .take(8)
                            .map(|v| {
                                let p = v.point();
                                format!("({:.2},{:.2},{:.2})", p.x, p.y, p.z)
                            })
                            .collect();
                        format!(
                            "w{}[{}e,st={:?}]: {}",
                            wi,
                            w.len(),
                            w.status(),
                            verts.join("->")
                        )
                    })
                    .collect();
                eprintln!(
                    "[loops_store_summary] shell1 face {}: {}",
                    fi,
                    wire_info.join(" | "),
                );
            }
        }
    }
    let (mut cls0, coplanar_fids0) = divide_face::divide_faces_with_coplanar(
        &altshell0,
        &loops_store0,
        tols.tau_model,
        &coplanar_faces0,
        tols.tau_area,
    )
    .ok_or(BooleanStageError::FaceDivision)?;
    cls0.integrate_by_component();
    let (mut cls1, coplanar_fids1) = divide_face::divide_faces_with_coplanar(
        &altshell1,
        &loops_store1,
        tols.tau_model,
        &coplanar_faces1,
        tols.tau_area,
    )
    .ok_or(BooleanStageError::FaceDivision)?;
    cls1.integrate_by_component();
    let _divide_elapsed = _total_start.elapsed();
    // Reset overlapping coplanar fragments to Unknown for re-classification.
    cls0.reset_overlapping_coplanar(&coplanar_fids0, shell1, true, tols.tau_coplanar);
    cls1.reset_overlapping_coplanar(&coplanar_fids1, shell0, false, tols.tau_coplanar);
    let [mut and0, mut or0, unknown0] = cls0.and_or_unknown();
    eprintln!(
        "[classify] shell0: and={}, or={}, unknown={}",
        and0.len(),
        or0.len(),
        unknown0.len(),
    );
    // Classify unknown0 faces: coplanar → ray-cast → edge-neighbor propagation.
    // The edge-neighbor fallback handles corner-coplanar geometry where all
    // ray-cast strategies fail but neighboring faces are correctly classified.
    {
        let mut unresolved = Vec::new();
        for face in unknown0 {
            let coplanar_action = coplanar_overlay::classify_coplanar_via_overlay(
                &face,
                shell1,
                true,
                tols.tau_coplanar,
            )
            .or_else(|| {
                coplanar::classify_coplanar_fragment(&face, shell1, true, tols.tau_coplanar)
            });
            if let Some(action) = coplanar_action {
                match action {
                    coplanar::CoplanarAction::Remove => {}
                    coplanar::CoplanarAction::And => and0.push(face),
                    coplanar::CoplanarAction::Or => or0.push(face),
                }
                continue;
            }
            // Primary: winding number classification (no rays, no perturbation).
            // Fallback: ray-cast if winding number is ambiguous.
            let classified = winding::winding_classify_face(&face, &poly_shell1)
                .or_else(|| ray_cast_classify(&face, &poly_shell1, Some(&bvh1)));
            if let Some(count) = classified {
                if count == 1 {
                    and0.push(face);
                } else {
                    or0.push(face);
                }
            } else {
                unresolved.push(face);
            }
        }
        if !unresolved.is_empty() {
            classify_by_edge_neighbors(
                unresolved,
                &mut and0,
                &mut or0,
                diag.as_mut().map(|d| &mut d.edge_neighbor),
            )?;
        }
    }
    eprintln!(
        "[classify] shell0 final: and={}, or={}",
        and0.len(),
        or0.len(),
    );
    let [mut and1, mut or1, unknown1] = cls1.and_or_unknown();
    eprintln!(
        "[classify] shell1: and={}, or={}, unknown={}",
        and1.len(),
        or1.len(),
        unknown1.len(),
    );
    // Classify unknown1 faces: coplanar → ray-cast → edge-neighbor propagation.
    {
        let mut unresolved = Vec::new();
        for face in unknown1 {
            let coplanar_action = coplanar_overlay::classify_coplanar_via_overlay(
                &face,
                shell0,
                false,
                tols.tau_coplanar,
            )
            .or_else(|| {
                coplanar::classify_coplanar_fragment(&face, shell0, false, tols.tau_coplanar)
            });
            if let Some(action) = coplanar_action {
                match action {
                    coplanar::CoplanarAction::Remove => {}
                    coplanar::CoplanarAction::And => and1.push(face),
                    coplanar::CoplanarAction::Or => or1.push(face),
                }
                continue;
            }
            // Primary: winding number classification.
            // Fallback: ray-cast if winding number is ambiguous.
            let classified = winding::winding_classify_face(&face, &poly_shell0)
                .or_else(|| ray_cast_classify(&face, &poly_shell0, Some(&bvh0)));
            if let Some(count) = classified {
                if count == 1 {
                    and1.push(face);
                } else {
                    or1.push(face);
                }
            } else {
                unresolved.push(face);
            }
        }
        if !unresolved.is_empty() {
            classify_by_edge_neighbors(
                unresolved,
                &mut and1,
                &mut or1,
                diag.as_mut().map(|d| &mut d.edge_neighbor),
            )?;
        }
    }
    eprintln!(
        "[classify] shell1 final: and={}, or={}",
        and1.len(),
        or1.len(),
    );
    // Post-classification fixup: reclassify contained coplanar faces.
    //
    // When shell0 face i is contained in shell1 face j (both coplanar):
    // 1. The injection inserts shell0[i]'s boundary wire into shell1[j]'s loops
    // 2. Face division splits shell1[j] into ring (with hole) + inner
    // 3. The inner face and shell0[i] are both coplanar overlap → both should be And
    // 4. The ring face keeps its hole → hole boundary edges pair with lateral
    //    surface edges (same Arc<Edge> from injection clone), giving refs=2
    //
    // Two fixups needed:
    //   A) Reclassify the inner face in or1 → and1
    //   B) Reclassify the contained shell0 face in or0 → and0
    if !contained_faces0.is_empty() {
        let ref_faces: Vec<_> = contained_faces0
            .iter()
            .filter_map(|&idx| {
                if idx < shell0.len() {
                    Some(&shell0[idx])
                } else {
                    None
                }
            })
            .collect();

        // (A) Reclassify inner face: find or1 faces coplanar with the
        // contained shell0 face whose outer-wire centroid is inside it.
        let mut reclass1 = Vec::new();
        or1.retain(|face| {
            let fi = match coplanar::face_sample_info(face) {
                Some(i) => i,
                None => return true,
            };
            let centroid = {
                let boundaries = face.boundaries();
                match boundaries.first() {
                    Some(wire) => {
                        let mut sum = Vector3::zero();
                        let mut count = 0usize;
                        for v in wire.vertex_iter() {
                            let p = v.point();
                            sum += Vector3::new(p.x, p.y, p.z);
                            count += 1;
                        }
                        if count > 0 {
                            Point3::new(
                                sum.x / count as f64,
                                sum.y / count as f64,
                                sum.z / count as f64,
                            )
                        } else {
                            fi.point
                        }
                    }
                    None => fi.point,
                }
            };
            for ref_face in &ref_faces {
                let fj = match coplanar::face_sample_info(ref_face) {
                    Some(j) => j,
                    None => continue,
                };
                if let Some(true) = coplanar::check_coplanar(&fi, &fj, tols.tau_coplanar) {
                    if coplanar::point_in_face(centroid, ref_face, tols.tau_coplanar) {
                        reclass1.push(face.clone());
                        return false;
                    }
                }
            }
            true
        });
        if !reclass1.is_empty() {
            eprintln!(
                "[classify] contained fixup: {} or1 inner faces → and1",
                reclass1.len(),
            );
            and1.extend(reclass1);
        }

        // (B) Reclassify contained shell0 faces: the contained face is
        // coplanar with a shell1 face and fully inside it. For union, it's
        // an overlap region that should be And. The standard classifier may
        // misclassify it as Or because winding number is ~0.5 for coplanar
        // faces. Match by edge ID sharing with the original shell0 face.
        let ref_edge_ids: rustc_hash::FxHashSet<u64> = ref_faces
            .iter()
            .flat_map(|f| {
                f.absolute_boundaries()
                    .iter()
                    .flat_map(|w| w.iter().map(|e| e.id().raw()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut reclass0 = Vec::new();
        or0.retain(|face| {
            let face_edge_ids: Vec<u64> = face
                .absolute_boundaries()
                .iter()
                .flat_map(|w| w.iter().map(|e| e.id().raw()))
                .collect();
            // If this or0 face shares ALL edges with a ref_face, it IS the
            // contained face (same geometry, same edge arcs).
            let all_shared = !face_edge_ids.is_empty()
                && face_edge_ids.iter().all(|eid| ref_edge_ids.contains(eid));
            if all_shared {
                reclass0.push(face.clone());
                false
            } else {
                true
            }
        });
        if !reclass0.is_empty() {
            eprintln!(
                "[classify] contained fixup: {} or0 contained faces → and0",
                reclass0.len(),
            );
            and0.extend(reclass0);
        }
    }

    eprintln!(
        "[classify] totals: and0+and1={}, or0+or1={}",
        and0.len() + and1.len(),
        or0.len() + or1.len(),
    );
    let _classify_elapsed = _total_start.elapsed();
    let result = ClassifiedShellBuckets {
        and0: altshell_to_shell(&and0, tols.tau_model).ok_or(BooleanStageError::ShellAssembly(
            "altshell_to_shell(and0)".into(),
        ))?,
        or0: altshell_to_shell(&or0, tols.tau_model).ok_or(BooleanStageError::ShellAssembly(
            "altshell_to_shell(or0)".into(),
        ))?,
        and1: altshell_to_shell(&and1, tols.tau_model).ok_or(BooleanStageError::ShellAssembly(
            "altshell_to_shell(and1)".into(),
        ))?,
        or1: altshell_to_shell(&or1, tols.tau_model).ok_or(BooleanStageError::ShellAssembly(
            "altshell_to_shell(or1)".into(),
        ))?,
    };
    let _altshell_elapsed = _total_start.elapsed();

    // Populate diagnostics if requested.
    if let Some(diag) = diag {
        use super::diagnostics::*;
        diag.tolerance = ToleranceReport {
            tau_model: tols.tau_model,
            tau_mesh: tols.tau_mesh,
            tau_weld: tols.tau_weld,
            tau_coplanar: tols.tau_coplanar,
            tau_boundary: tols.tau_boundary,
            tau_edge_cluster: tols.tau_edge_cluster,
            tau_area: tols.tau_area,
        };
        diag.classification.faces_coplanar = coplanar_faces0.len() + coplanar_faces1.len();
        diag.classification.shell0_and = result.and0.len();
        diag.classification.shell0_or = result.or0.len();
        diag.classification.shell1_and = result.and1.len();
        diag.classification.shell1_or = result.or1.len();
        diag.timing.loops_store = _loops_store_elapsed;
        diag.timing.divide_faces = _divide_elapsed - _loops_store_elapsed;
        diag.timing.classification = _classify_elapsed - _divide_elapsed;
        diag.timing.altshell = _altshell_elapsed - _classify_elapsed;
        diag.timing.total = _altshell_elapsed;
    }

    Ok(result)
}

fn process_one_pair_of_shells_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<[Shell<Point3, C, S>; 2], BooleanStageError> {
    let ClassifiedShellBuckets {
        mut and0,
        mut or0,
        mut and1,
        mut or1,
    } = classify_one_pair_of_shells_result_with_tol(shell0, shell1, tols, None)?;
    and0.append(&mut and1);
    or0.append(&mut or1);
    Ok([and0, or0])
}

/// Repair wires by graph-based reconstruction with gap filling.
///
/// When face division or edge canonicalization produces wires with:
/// - Edges in wrong order (not following vertex connectivity)
/// - Missing short edges at intersection boundary points
///
/// This function:
/// 1. Collects all edges from all wires into a single pool
/// 2. Builds an undirected vertex adjacency graph
/// 3. Finds degree-1 vertices (gap endpoints) and creates Line edges to fill gaps
/// 4. Traverses the graph to build properly-ordered closed wires
///
/// Returns `Some(repaired_wires)` if all edges can be arranged into closed loops.
fn repair_wires_with_gap_fill<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    wires: &[Wire<Point3, C>],
) -> Option<Vec<Wire<Point3, C>>> {
    use std::collections::BTreeMap;
    type Vid = VertexID<Point3>;

    // Collect all edges from all wires
    let mut all_edges: Vec<Edge<Point3, C>> = Vec::new();
    for wire in wires {
        for edge in wire.iter() {
            all_edges.push(edge.clone());
        }
    }

    if all_edges.is_empty() {
        return None;
    }

    // Build undirected adjacency: vertex -> list of (edge_index, other_vertex)
    // Use oriented vertices (edge.front() and edge.back())
    let mut adj: BTreeMap<Vid, Vec<(usize, Vid)>> = BTreeMap::new();
    let mut vertex_map: BTreeMap<Vid, Vertex<Point3>> = BTreeMap::new();

    for (idx, edge) in all_edges.iter().enumerate() {
        let fv = edge.front().clone();
        let bv = edge.back().clone();
        let fid = fv.id();
        let bid = bv.id();
        adj.entry(fid).or_default().push((idx, bid));
        adj.entry(bid).or_default().push((idx, fid));
        vertex_map.insert(fid, fv);
        vertex_map.insert(bid, bv);
    }

    // Find degree-1 vertices (gap endpoints)
    let degree1: Vec<Vid> = adj
        .iter()
        .filter(|(_, neighbors)| neighbors.len() == 1)
        .map(|(vid, _)| *vid)
        .collect();

    // Fill gaps: pair degree-1 vertices by proximity and create Line edges
    {
        eprintln!(
            "[gap_repair] {} degree-1 vertices in {} edges",
            degree1.len(),
            all_edges.len(),
        );
        for vid in &degree1 {
            if let Some(v) = vertex_map.get(vid) {
                let p = v.point();
                eprintln!(
                    "  degree-1: vid={:?} pos=({:.3},{:.3},{:.3})",
                    vid, p.x, p.y, p.z
                );
            }
        }
    }
    if !degree1.len().is_multiple_of(2) {
        // Odd number of gap endpoints — can't pair them
        return None;
    }

    let mut gap_edges: Vec<Edge<Point3, C>> = Vec::new();
    if !degree1.is_empty() {
        let mut remaining: Vec<Vid> = degree1;
        while !remaining.is_empty() {
            let v0 = remaining.remove(0);
            let p0 = vertex_map.get(&v0)?.point();

            // Find nearest remaining vertex
            let mut best_idx = 0;
            let mut best_dist = f64::MAX;
            for (i, vid) in remaining.iter().enumerate() {
                let pi = vertex_map.get(vid)?.point();
                let dist = (pi - p0).magnitude();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            let v1 = remaining.remove(best_idx);

            // Skip gap filling if the gap is too large (> 5 units — likely wrong face)
            if best_dist > 5.0 {
                eprintln!("[gap_repair] gap too large: {:.4}", best_dist);
                return None;
            }

            let vf = vertex_map.get(&v0)?;
            let vb = vertex_map.get(&v1)?;

            // Create a Line edge from v0 to v1
            let pf = vf.point();
            let pb = vb.point();
            eprintln!(
                "[gap_repair] filling gap: ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) dist={:.4}",
                pf.x, pf.y, pf.z, pb.x, pb.y, pb.z, best_dist,
            );
            let line_curve = Line(pf, pb);
            let curve: C = C::from(line_curve);
            if let Ok(gap_edge) = Edge::try_new(vf, vb, curve) {
                let edge_idx = all_edges.len() + gap_edges.len();
                adj.entry(v0).or_default().push((edge_idx, v1));
                adj.entry(v1).or_default().push((edge_idx, v0));
                gap_edges.push(gap_edge);
            } else {
                eprintln!("[gap_repair] Edge::try_new failed for gap edge");
                return None;
            }
        }
    }

    // Combine original + gap edges
    let total_edges: Vec<Edge<Point3, C>> =
        all_edges.iter().chain(gap_edges.iter()).cloned().collect();

    // Graph traversal to build closed wires
    let mut used = vec![false; total_edges.len()];
    let mut result_wires: Vec<Wire<Point3, C>> = Vec::new();

    while let Some(start_idx) = used.iter().position(|&u| !u) {
        let start_edge = &total_edges[start_idx];
        let start_vid = start_edge.front().id();
        let mut current_vid = start_edge.back().id();
        used[start_idx] = true;

        let mut wire_edges: Vec<Edge<Point3, C>> = vec![start_edge.clone()];
        let mut stuck = false;

        while current_vid != start_vid {
            // Find unused edge adjacent to current_vid
            let neighbors = match adj.get(&current_vid) {
                Some(n) => n,
                None => {
                    stuck = true;
                    break;
                }
            };

            let mut found = false;
            for &(edge_idx, other_vid) in neighbors {
                if used[edge_idx] {
                    continue;
                }
                used[edge_idx] = true;
                let edge = &total_edges[edge_idx];
                // Determine correct orientation for this edge
                if edge.front().id() == current_vid {
                    wire_edges.push(edge.clone());
                } else {
                    wire_edges.push(edge.inverse());
                }
                current_vid = other_vid;
                found = true;
                break;
            }

            if !found {
                stuck = true;
                break;
            }
        }

        if stuck {
            eprintln!("[gap_repair] graph traversal stuck");
            return None;
        }

        let wire: Wire<Point3, C> = wire_edges.into_iter().collect();
        if !wire.is_closed() {
            eprintln!(
                "[gap_repair] reconstructed wire not closed ({} edges)",
                wire.len()
            );
            return None;
        }
        eprintln!("[gap_repair] built wire with {} edges", wire.len());
        result_wires.push(wire);
    }

    if result_wires.is_empty() {
        return None;
    }

    Some(result_wires)
}

/// Try to recover a face with non-simple wires via splitting, merging, or
/// relaxed validation.
///
/// Recovery strategies (tried in order):
/// 1. Recursive wire splitting: handles wires with multiple repeated vertices
///    (common after chained boolean operations where vertex dedup + weld compound).
/// 2. Wire merging: when individual wires are simple but share vertices across
///    boundaries (T-junctions), merge touching wires into a single wire that
///    traces both outer boundary and hole. This produces a non-simple single wire
///    but avoids edge over-counting in the canonical edge pipeline.
/// 3. Gap-filling wire repair: detects degree-1 vertices (gaps) in the edge graph,
///    creates Line edges to fill gaps, and rebuilds properly-ordered wires.
/// 4. Fallback: `Face::new_unchecked` for any remaining case where wires are
///    non-empty and closed.
fn try_split_non_simple_wires<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    wires: &[Wire<Point3, C>],
    surface: &S,
    ori: bool,
) -> Option<Face<Point3, C, S>> {
    let mut new_wires: Vec<Wire<Point3, C>> = Vec::new();
    let mut any_split = false;

    for wire in wires {
        if wire.is_simple() {
            new_wires.push(wire.clone());
            continue;
        }

        let mut split_result: Vec<Wire<Point3, C>> = Vec::new();
        if split_wire_recursive(wire, &mut split_result, 0) {
            new_wires.extend(split_result);
            any_split = true;
        } else {
            // Could not split — keep original wire
            new_wires.push(wire.clone());
        }
    }

    // Try strict validation first (handles wire-splitting case)
    if any_split {
        if let Ok(mut new_face) = Face::try_new(new_wires.clone(), surface.clone()) {
            if !ori {
                new_face.invert();
            }
            return Some(new_face);
        }
    }

    // Gap-filling wire repair: detect missing edges (degree-1 vertices) and
    // reconstruct properly-ordered wires with Line edges filling the gaps.
    // This handles cases where face division or edge canonicalization produces
    // wires with edges in wrong order or missing short intersection boundary edges.
    let all_closed = new_wires.iter().all(|w| !w.is_empty() && w.is_closed());
    if !all_closed {
        {
            let non_closed_count = new_wires.iter().filter(|w| !w.is_closed()).count();
            let total_edges: usize = new_wires.iter().map(|w| w.len()).sum();
            eprintln!(
                "[gap_repair] trying repair: {} wires, {} non-closed, {} total edges",
                new_wires.len(),
                non_closed_count,
                total_edges,
            );
        }
        if let Some(repaired) = repair_wires_with_gap_fill(&new_wires) {
            // Try strict validation first
            if let Ok(mut new_face) = Face::try_new(repaired.clone(), surface.clone()) {
                if !ori {
                    new_face.invert();
                }
                return Some(new_face);
            }
            // Accept via new_unchecked if all repaired wires are closed
            let all_repaired_closed = repaired.iter().all(|w| !w.is_empty() && w.is_closed());
            if !repaired.is_empty() && all_repaired_closed {
                let mut new_face = Face::new_unchecked(repaired, surface.clone());
                if !ori {
                    new_face.invert();
                }
                return Some(new_face);
            }
        }
    }

    // Final fallback: accept non-disjoint or non-simple wires via new_unchecked.
    // Non-disjoint wires (sharing vertices at T-junctions) are geometrically
    // valid — each edge still appears exactly once per face. The shared vertex
    // creates a singular vertex but doesn't affect edge counting for shell
    // condition (Closed/Oriented).
    if !new_wires.is_empty() && all_closed {
        let mut new_face = Face::new_unchecked(new_wires, surface.clone());
        if !ori {
            new_face.invert();
        }
        return Some(new_face);
    }

    None
}

/// Unify geometrically-coincident vertices in a shell via spatial grid.
///
/// After boolean operations, a solid may contain vertices at the same
/// geometric position but with different identities. This function scans
/// all vertices, groups them by position (within `tol`), picks a canonical
/// vertex for each group, and rebuilds edges with canonical vertices.
/// Degenerate edges (both endpoints same after unification) are dropped.
///
/// This is the standalone Phase 0 of `weld_coincident_edges`, useful as
/// a pre-healing step before a subsequent boolean operation.
pub fn heal_shell_vertices<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) {
    use rustc_hash::{FxHashMap, FxHashSet};
    type Vid = VertexID<Point3>;

    let unify_tol = tol;

    // Collect all unique vertices (by ID)
    let mut all_verts: Vec<Vertex<Point3>> = Vec::new();
    let mut seen: FxHashSet<Vid> = FxHashSet::default();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for v in wire.vertex_iter() {
                if seen.insert(v.id()) {
                    all_verts.push(v.clone());
                }
            }
        }
    }

    // Spatial grid: cell key -> list of canonical vertices in that cell
    let cell = unify_tol;
    let mut grid: FxHashMap<(i64, i64, i64), Vec<Vertex<Point3>>> = FxHashMap::default();
    let mut unify_map: FxHashMap<Vid, Vertex<Point3>> = FxHashMap::default();

    for v in &all_verts {
        let pt = v.point();
        let key = (
            (pt.x / cell).round() as i64,
            (pt.y / cell).round() as i64,
            (pt.z / cell).round() as i64,
        );
        let mut found = None;
        'search: for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    let nkey = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(verts) = grid.get(&nkey) {
                        for canon in verts {
                            if (canon.point() - pt).magnitude() < unify_tol {
                                found = Some(canon.clone());
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        match found {
            Some(canon) if canon.id() != v.id() => {
                unify_map.insert(v.id(), canon);
            }
            None => {
                grid.entry(key).or_default().push(v.clone());
            }
            _ => {}
        }
    }

    if unify_map.is_empty() {
        return;
    }

    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .filter_map(|edge| {
                            let abs = edge.absolute_clone();
                            let f = unify_map.get(&abs.front().id());
                            let b = unify_map.get(&abs.back().id());
                            if f.is_none() && b.is_none() {
                                return Some(edge.clone());
                            }
                            let new_front = f.cloned().unwrap_or_else(|| abs.front().clone());
                            let new_back = b.cloned().unwrap_or_else(|| abs.back().clone());
                            if new_front.id() == new_back.id() {
                                return None; // Drop degenerate edge
                            }
                            let new_abs = Edge::try_new(&new_front, &new_back, abs.curve()).ok()?;
                            if edge.orientation() {
                                Some(new_abs)
                            } else {
                                Some(new_abs.inverse())
                            }
                        })
                        .collect();
                    edges.into()
                })
                .collect();
            let surface = face.surface();
            match Face::try_new(new_wires.clone(), surface.clone()) {
                Ok(mut new_face) => {
                    if !ori {
                        new_face.invert();
                    }
                    new_face
                }
                Err(_) => {
                    // Preserve unified vertex IDs via new_unchecked
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
        .collect();

    *shell = new_faces.into_iter().collect();
}

/// a single Edge identity → `ShellCondition::Closed`.
///
/// Three phases:
///  0. Position-based vertex unification via spatial grid.
///  1. Build canonical edge map by (front_vertex_id, back_vertex_id).
///  2. Rebuild faces, replacing non-canonical edges with the canonical one.
fn weld_coincident_edges<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
    weld_tol: Option<f64>,
) {
    use rustc_hash::{FxHashMap, FxHashSet};
    type Vid = VertexID<Point3>;

    // Phase 0: Position-based vertex unification via spatial grid.
    {
        use truck_base::tolerance::TOLERANCE;
        let unify_tol = match weld_tol {
            Some(wt) => wt.max(TOLERANCE.sqrt()),
            None => (tol * 0.2).max(TOLERANCE.sqrt()),
        };
        // Collect all unique vertices (by ID)
        let mut all_verts: Vec<Vertex<Point3>> = Vec::new();
        let mut seen: FxHashSet<Vid> = FxHashSet::default();
        for face in shell.iter() {
            for wire in face.absolute_boundaries().iter() {
                for v in wire.vertex_iter() {
                    if seen.insert(v.id()) {
                        all_verts.push(v.clone());
                    }
                }
            }
        }

        // Spatial grid: cell key -> list of canonical vertices in that cell
        let cell = unify_tol;
        let mut grid: FxHashMap<(i64, i64, i64), Vec<Vertex<Point3>>> = FxHashMap::default();
        let mut unify_map: FxHashMap<Vid, Vertex<Point3>> = FxHashMap::default();

        for v in &all_verts {
            let pt = v.point();
            let key = (
                (pt.x / cell).round() as i64,
                (pt.y / cell).round() as i64,
                (pt.z / cell).round() as i64,
            );
            let mut found = None;
            'search: for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dz in -1i64..=1 {
                        let nkey = (key.0 + dx, key.1 + dy, key.2 + dz);
                        if let Some(verts) = grid.get(&nkey) {
                            for canon in verts {
                                if (canon.point() - pt).magnitude() < unify_tol {
                                    found = Some(canon.clone());
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            match found {
                Some(canon) if canon.id() != v.id() => {
                    unify_map.insert(v.id(), canon);
                }
                None => {
                    grid.entry(key).or_default().push(v.clone());
                }
                _ => {}
            }
        }

        // Rebuild faces with unified vertices
        if !unify_map.is_empty() {
            let new_faces: Vec<Face<Point3, C, S>> = shell
                .iter()
                .map(|face| {
                    let ori = face.orientation();
                    let new_wires: Vec<Wire<Point3, C>> = face
                        .absolute_boundaries()
                        .iter()
                        .map(|wire| {
                            let edges: Vec<Edge<Point3, C>> = wire
                                .iter()
                                .filter_map(|edge| {
                                    let abs = edge.absolute_clone();
                                    let f = unify_map.get(&abs.front().id());
                                    let b = unify_map.get(&abs.back().id());
                                    if f.is_none() && b.is_none() {
                                        return Some(edge.clone());
                                    }
                                    let new_front =
                                        f.cloned().unwrap_or_else(|| abs.front().clone());
                                    let new_back = b.cloned().unwrap_or_else(|| abs.back().clone());
                                    // Drop degenerate edges where vertex unification
                                    // collapsed both endpoints to the same vertex. The
                                    // adjacent edges also had their vertices unified to
                                    // this same canonical, so the wire remains connected.
                                    if new_front.id() == new_back.id() {
                                        return None;
                                    }
                                    let new_abs =
                                        Edge::try_new(&new_front, &new_back, abs.curve()).ok()?;
                                    if edge.orientation() {
                                        Some(new_abs)
                                    } else {
                                        Some(new_abs.inverse())
                                    }
                                })
                                .collect();
                            edges.into()
                        })
                        .collect();
                    let surface = face.surface();
                    // Use try_new to avoid panic on non-simple wires after
                    // vertex unification; fall back to wire splitting, then
                    // new_unchecked with unified vertices. NEVER fall back to
                    // face.clone() — that restores OLD vertex IDs which makes
                    // Phase 1 unable to match edges with their canonical partners.
                    match Face::try_new(new_wires.clone(), surface.clone()) {
                        Ok(mut new_face) => {
                            if !ori {
                                new_face.invert();
                            }
                            new_face
                        }
                        Err(_e) => {
                            // Try wire splitting before falling back
                            if let Some(split_face) =
                                try_split_non_simple_wires(&new_wires, &surface, ori)
                            {
                                split_face
                            } else {
                                // Use new_unchecked to preserve unified vertex IDs.
                                // A non-simple wire is better than reverting to old IDs.
                                let all_closed =
                                    new_wires.iter().all(|w| !w.is_empty() && w.is_closed());
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
                    }
                })
                .collect();
            *shell = new_faces.into_iter().collect();
        }
    }

    // Phase 1: Edge canonicalization via multi-point curve matching.
    //
    // Two edges with the same vertex pair are canonical partners if their
    // curves agree at 3 sample points (t=0.25, 0.5, 0.75 within parameter
    // range) within 3 * tau_model (accounts for independent BSpline
    // approximation error). Edges from the same face are never merged
    // (they represent distinct geometric features like inner vs outer boundary).
    //
    // This replaces the previous single-midpoint clustering approach that
    // used tau_edge_cluster = 5.0 * tau_model, which was too loose for some
    // configurations and could incorrectly merge distinct edges.
    //
    // Uses BTreeMap<(VertexID, VertexID), _> for deterministic iteration order.
    // VertexID is now SequentialID — creation-order monotonic integers — so
    // BTreeMap key order is fully deterministic across runs.
    type VidPairKey = (VertexID<Point3>, VertexID<Point3>);

    // Edge candidate: edge with face ownership and 3-point curve samples.
    struct EdgeCandidate<C2> {
        edge: Edge<Point3, C2>,
        face_idx: usize,
        samples: [[f64; 3]; 3],
    }

    // Build edge candidate list with face ownership
    let mut candidates: Vec<EdgeCandidate<C>> = Vec::new();
    let mut candidate_vid_pair: Vec<VidPairKey> = Vec::new();

    for (face_idx, face) in shell.iter().enumerate() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let abs = edge.absolute_clone();
                let fid = abs.front().id();
                let bid = abs.back().id();

                let vid_pair = if fid <= bid { (fid, bid) } else { (bid, fid) };

                // Sample curve at 3 points within parameter range
                let curve = abs.curve();
                let (t0, t1) = curve.range_tuple();
                let dt = t1 - t0;
                let samples = [
                    {
                        let p = curve.subs(t0 + dt * 0.25);
                        [p.x, p.y, p.z]
                    },
                    {
                        let p = curve.subs(t0 + dt * 0.5);
                        [p.x, p.y, p.z]
                    },
                    {
                        let p = curve.subs(t0 + dt * 0.75);
                        [p.x, p.y, p.z]
                    },
                ];

                candidate_vid_pair.push(vid_pair);
                candidates.push(EdgeCandidate {
                    edge: abs,
                    face_idx,
                    samples,
                });
            }
        }
    }

    // Group candidate indices by vertex pair
    let mut pair_groups: std::collections::BTreeMap<VidPairKey, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, pair) in candidate_vid_pair.iter().enumerate() {
        pair_groups.entry(*pair).or_default().push(i);
    }

    // Within each group, find canonical pairs via multi-point matching.
    // Edges from DIFFERENT faces whose curves agree at 3 sample points
    // within tau_model are canonical partners.
    let mut edge_to_canonical: std::collections::BTreeMap<EdgeID<C>, Edge<Point3, C>> =
        std::collections::BTreeMap::new();

    for indices in pair_groups.values() {
        if indices.len() <= 1 {
            continue;
        }

        let mut matched = vec![false; indices.len()];
        for i in 0..indices.len() {
            if matched[i] {
                continue;
            }
            let ci = &candidates[indices[i]];

            for j in (i + 1)..indices.len() {
                if matched[j] {
                    continue;
                }
                let cj = &candidates[indices[j]];

                // Skip edges from the same face — they represent distinct
                // geometric features (e.g., inner vs outer boundary).
                if ci.face_idx == cj.face_idx {
                    continue;
                }

                // Check 3-point curve agreement within 3 * tau_model.
                // Factor of 3 accounts for two independent BSpline
                // approximations of the same intersection curve, each
                // within tau_model of the true curve (worst case 2*tau_model
                // apart, plus margin). Still much tighter than the old
                // single-midpoint clustering at 5 * tau_model.
                let match_tol = tol * 3.0;
                let agree = (0..3).all(|k| {
                    let dx = ci.samples[k][0] - cj.samples[k][0];
                    let dy = ci.samples[k][1] - cj.samples[k][1];
                    let dz = ci.samples[k][2] - cj.samples[k][2];
                    (dx * dx + dy * dy + dz * dz).sqrt() < match_tol
                });

                if agree && ci.edge.id() != cj.edge.id() {
                    // ci is canonical, cj maps to ci
                    edge_to_canonical.insert(cj.edge.id(), ci.edge.clone());
                    matched[j] = true;
                }
            }
        }
    }

    // Phase 2: Rebuild faces with canonical edges.
    //
    // For each edge in each wire, replace it with the canonical edge from Phase 1.
    // IMPORTANT: Track which canonical edges are already used within each wire.
    // T-junction wires (from new_unchecked in Phase 0) may have two edges that
    // map to the same canonical — replacing both would create duplicate edge
    // references within the wire, violating the closed-shell invariant. When a
    // canonical is already used in the wire, keep the original edge identity.
    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let mut used_canonicals: std::collections::BTreeSet<EdgeID<C>> =
                        std::collections::BTreeSet::new();
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .map(|edge| match edge_to_canonical.get(&edge.id()) {
                            Some(c) if !used_canonicals.contains(&c.id()) => {
                                used_canonicals.insert(c.id());
                                let abs_edge = edge.absolute_clone();
                                // Use geometric position to determine direction since
                                // vertex IDs may differ after mapped() or Phase 0 fallback
                                let front_dist =
                                    (abs_edge.front().point() - c.front().point()).magnitude();
                                let cross_dist =
                                    (abs_edge.front().point() - c.back().point()).magnitude();
                                let same_dir = front_dist <= cross_dist;
                                if same_dir == edge.orientation() {
                                    c.clone()
                                } else {
                                    c.inverse()
                                }
                            }
                            _ => edge.clone(),
                        })
                        .collect();
                    edges.into()
                })
                .collect();
            let surface = face.surface();
            match Face::try_new(new_wires.clone(), surface.clone()) {
                Ok(mut new_face) => {
                    if !ori {
                        new_face.invert();
                    }
                    new_face
                }
                Err(_) => {
                    if let Some(split_face) = try_split_non_simple_wires(&new_wires, &surface, ori)
                    {
                        split_face
                    } else {
                        face.clone()
                    }
                }
            }
        })
        .collect();

    *shell = new_faces.into_iter().collect();

    // Phase 3: Fix over-counted edges.
    // After canonicalization + wire splitting/merging, some edges may appear
    // in 3+ face references (e.g., merged non-simple wires referencing the
    // same canonical edge twice within one face, plus normal sharing with
    // adjacent faces). Replace excess references (3rd, 4th, ...) with fresh
    // edge clones to restore the 2-reference invariant for Closed shells.
    //
    // Uses BTreeMap/BTreeSet for deterministic iteration order.
    //
    // IMPORTANT: Count each edge at most once per face. T-junction faces
    // (from new_unchecked) may reference the same edge twice within one wire,
    // but that's one face contribution. Without per-face dedup, a T-junction
    // edge gets count 3 (2 within-face + 1 adjacent) and Phase 3 incorrectly
    // clones the adjacent face's reference, breaking the shared edge.
    {
        // Count edge references across faces (dedup within each face)
        let mut edge_ref_count: std::collections::BTreeMap<EdgeID<C>, usize> =
            std::collections::BTreeMap::new();
        for face in shell.iter() {
            let mut face_edges: std::collections::BTreeSet<EdgeID<C>> =
                std::collections::BTreeSet::new();
            for wire in face.absolute_boundaries().iter() {
                for edge in wire.iter() {
                    face_edges.insert(edge.id());
                }
            }
            for eid in face_edges {
                *edge_ref_count.entry(eid).or_insert(0) += 1;
            }
        }

        let overcounted: std::collections::BTreeSet<EdgeID<C>> = edge_ref_count
            .iter()
            .filter(|(_, count)| **count > 2)
            .map(|(id, _)| *id)
            .collect();

        if !overcounted.is_empty() {
            // Rebuild faces, cloning excess edge references
            let mut edge_seen_count: std::collections::BTreeMap<EdgeID<C>, usize> =
                std::collections::BTreeMap::new();
            let fixed_faces: Vec<Face<Point3, C, S>> = shell
                .iter()
                .map(|face| {
                    let ori = face.orientation();
                    let mut any_cloned = false;
                    let new_wires: Vec<Wire<Point3, C>> = face
                        .absolute_boundaries()
                        .iter()
                        .map(|wire| {
                            let edges: Vec<Edge<Point3, C>> = wire
                                .iter()
                                .map(|edge| {
                                    if !overcounted.contains(&edge.id()) {
                                        return edge.clone();
                                    }
                                    let count = edge_seen_count.entry(edge.id()).or_insert(0);
                                    *count += 1;
                                    if *count <= 2 {
                                        // First two references keep the canonical edge
                                        edge.clone()
                                    } else {
                                        // 3rd+ reference: clone with fresh identity
                                        any_cloned = true;
                                        let abs = edge.absolute_clone();
                                        let fresh = Edge::new(abs.front(), abs.back(), abs.curve());
                                        if edge.orientation() {
                                            fresh
                                        } else {
                                            fresh.inverse()
                                        }
                                    }
                                })
                                .collect();
                            edges.into()
                        })
                        .collect();
                    if !any_cloned {
                        return face.clone();
                    }
                    let surface = face.surface();
                    let all_closed = new_wires.iter().all(|w| !w.is_empty() && w.is_closed());
                    if all_closed {
                        let mut new_face = Face::new_unchecked(new_wires, surface);
                        if !ori {
                            new_face.invert();
                        }
                        new_face
                    } else {
                        face.clone()
                    }
                })
                .collect();
            *shell = fixed_faces.into_iter().collect();
        }
    }
}

/// Force-merge open edges by geometric endpoint position.
///
/// Unlike `weld_coincident_edges` which requires 3-point curve agreement,
/// this function matches open edges purely by endpoint positions. This handles
/// cases where two IC approximation curves represent the same intersection
/// but have different BSpline parameterizations that don't match at sample points.
///
/// Only touches edges with ref_count == 1 (open edges). Returns the number
/// of edges merged.
fn force_merge_open_edges<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) -> usize {
    use std::collections::BTreeMap;

    // Step 1: Find ALL edges and count refs. Collect non-manifold edges (refs != 2).
    let mut edge_ref_count: BTreeMap<EdgeID<C>, usize> = BTreeMap::new();
    let mut edge_by_id: BTreeMap<EdgeID<C>, Edge<Point3, C>> = BTreeMap::new();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let abs = edge.absolute_clone();
                *edge_ref_count.entry(abs.id()).or_insert(0) += 1;
                edge_by_id.entry(abs.id()).or_insert(abs);
            }
        }
    }

    let non_manifold_edges: Vec<Edge<Point3, C>> = edge_ref_count
        .iter()
        .filter(|(_, &count)| count != 2)
        .filter_map(|(id, _)| edge_by_id.get(id).cloned())
        .collect();

    if non_manifold_edges.len() < 2 {
        return 0;
    }

    // Step 2: Group non-manifold edges by endpoint positions (N-way grouping).
    // All edges at the same geometric position map to a single canonical edge.
    let mut groups: Vec<Vec<usize>> = Vec::new(); // each group is indices into non_manifold_edges
    let mut assigned = vec![false; non_manifold_edges.len()];

    for i in 0..non_manifold_edges.len() {
        if assigned[i] {
            continue;
        }
        let ei = &non_manifold_edges[i];
        let fi = ei.front().point();
        let bi = ei.back().point();

        let mut group = vec![i];
        assigned[i] = true;

        for j in (i + 1)..non_manifold_edges.len() {
            if assigned[j] {
                continue;
            }
            let ej = &non_manifold_edges[j];
            let fj = ej.front().point();
            let bj = ej.back().point();

            let same_dir = (fi - fj).magnitude() < tol && (bi - bj).magnitude() < tol;
            let opp_dir = (fi - bj).magnitude() < tol && (bi - fj).magnitude() < tol;

            if same_dir || opp_dir {
                group.push(j);
                assigned[j] = true;
            }
        }

        if group.len() >= 2 {
            groups.push(group);
        }
    }

    if groups.is_empty() {
        return 0;
    }

    // Step 3: Build merge map — all edges in a group map to the canonical (first) edge.
    // Prefer the edge with refs=2 as canonical (it's already properly shared).
    // Otherwise use the first edge.
    let mut merge_map: BTreeMap<EdgeID<C>, Edge<Point3, C>> = BTreeMap::new();
    let mut merge_count = 0usize;

    for group in &groups {
        // Find canonical: prefer edge with refs=2, else first
        let canonical_idx = group
            .iter()
            .find(|&&idx| {
                edge_ref_count
                    .get(&non_manifold_edges[idx].id())
                    .copied()
                    .unwrap_or(0)
                    == 2
            })
            .copied()
            .unwrap_or(group[0]);
        let canonical = &non_manifold_edges[canonical_idx];

        for &idx in group {
            if idx == canonical_idx {
                continue;
            }
            merge_map.insert(non_manifold_edges[idx].id(), canonical.clone());
            merge_count += 1;
        }
    }

    if merge_map.is_empty() {
        return 0;
    }

    // Step 4: Rebuild faces with merged edges
    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let mut any_merged = false;
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .map(|edge| {
                            let abs = edge.absolute_clone();
                            match merge_map.get(&abs.id()) {
                                Some(canonical) => {
                                    any_merged = true;
                                    // Use geometric position to determine direction
                                    let front_dist = (abs.front().point()
                                        - canonical.front().point())
                                    .magnitude();
                                    let cross_dist = (abs.front().point()
                                        - canonical.back().point())
                                    .magnitude();
                                    let same_dir = front_dist <= cross_dist;
                                    if same_dir == edge.orientation() {
                                        canonical.clone()
                                    } else {
                                        canonical.inverse()
                                    }
                                }
                                None => edge.clone(),
                            }
                        })
                        .collect();
                    edges.into()
                })
                .collect();

            if !any_merged {
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
        .collect();

    *shell = new_faces.into_iter().collect();
    merge_count
}

/// Information about an edge that appears in != 2 faces (open or over-shared).
#[derive(Debug)]
#[allow(dead_code)]
pub struct OpenEdgeInfo {
    /// Front vertex position.
    pub front: Point3,
    /// Back vertex position.
    pub back: Point3,
    /// Curve midpoint position.
    pub midpoint: Point3,
    /// Number of face references for this edge (should be 2 for closed shell).
    pub face_count: usize,
}

/// Diagnose open (unclosed) edges in a shell.
///
/// Returns a list of edges that don't have exactly 2 face references.
/// For a valid `Closed` shell, this should return an empty list.
/// Useful for debugging boolean pipeline failures where `Solid::try_new`
/// rejects the shell due to open boundaries.
pub fn diagnose_open_edges<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
) -> Vec<OpenEdgeInfo> {
    let mut edge_ref_count: std::collections::BTreeMap<EdgeID<C>, usize> =
        std::collections::BTreeMap::new();
    let mut edge_data: std::collections::BTreeMap<EdgeID<C>, (Point3, Point3, Point3)> =
        std::collections::BTreeMap::new();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let eid = edge.id();
                *edge_ref_count.entry(eid).or_insert(0) += 1;
                edge_data.entry(eid).or_insert_with(|| {
                    let abs = edge.absolute_clone();
                    let curve = abs.curve();
                    let (t0, t1) = curve.range_tuple();
                    let midpoint = curve.subs((t0 + t1) * 0.5);
                    (abs.front().point(), abs.back().point(), midpoint)
                });
            }
        }
    }

    let mut result = Vec::new();
    for (eid, &count) in &edge_ref_count {
        if count != 2 {
            if let Some(&(front, back, midpoint)) = edge_data.get(eid) {
                result.push(OpenEdgeInfo {
                    front,
                    back,
                    midpoint,
                    face_count: count,
                });
            }
        }
    }
    result
}

/// Euler characteristic validation for a shell: V - E + F = 2 (genus-0).
///
/// Returns `Ok(())` if the Euler formula holds, or `Err` with the
/// computed values `(V, E, F, chi)` if it doesn't.
/// This is a topological invariant for closed genus-0 surfaces (simple solids).
/// For solids with through-holes (genus > 0), chi = 2 - 2g.
pub fn validate_euler_characteristic<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
) -> std::result::Result<(), (usize, usize, usize, i64)> {
    use std::collections::BTreeSet;
    type Vid = VertexID<Point3>;

    let mut vertices: BTreeSet<Vid> = BTreeSet::new();
    let mut edges: BTreeSet<EdgeID<C>> = BTreeSet::new();
    let f = shell.len();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                edges.insert(edge.id());
                vertices.insert(edge.front().id());
                vertices.insert(edge.back().id());
            }
        }
    }

    let v = vertices.len();
    let e = edges.len();
    let chi = v as i64 - e as i64 + f as i64;

    if chi == 2 {
        Ok(())
    } else {
        Err((v, e, f, chi))
    }
}

/// Check that all wires in a shell are simple (no repeated vertices).
///
/// Returns a list of `(face_index, wire_index)` for non-simple wires.
pub fn find_non_simple_wires<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for (fi, face) in shell.iter().enumerate() {
        for (wi, wire) in face.absolute_boundaries().iter().enumerate() {
            if !wire.is_simple() {
                result.push((fi, wi));
            }
        }
    }
    result
}

/// Fill open edge loops with planar faces.
///
/// When the boolean pipeline drops a face fragment (e.g., a small region at the
/// intersection of a cut tool and a boss), the shell ends up with a closed loop
/// of open edges. This function:
/// 1. Finds all open edges (face_count == 1)
/// 2. Chains them into closed loops via shared vertices
/// 3. For each loop, computes a best-fit plane (Newell's method)
/// 4. Creates a Face with the loop wire on the computed plane
/// 5. Adds the face to the shell
///
/// Returns true if any faces were added.
#[allow(dead_code)]
fn fill_open_edge_loops<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) -> bool {
    use std::collections::BTreeMap;
    type Vid = VertexID<Point3>;

    // Collect open edges (face_count == 1)
    let mut edge_ref_count: BTreeMap<EdgeID<C>, usize> = BTreeMap::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                *edge_ref_count.entry(edge.id()).or_insert(0) += 1;
            }
        }
    }

    let mut open_edges: Vec<Edge<Point3, C>> = Vec::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                if edge_ref_count.get(&edge.id()).copied().unwrap_or(0) == 1 {
                    open_edges.push(edge.absolute_clone());
                }
            }
        }
    }

    if open_edges.is_empty() {
        return false;
    }

    // Deduplicate by edge ID
    {
        let mut seen: std::collections::BTreeSet<EdgeID<C>> = std::collections::BTreeSet::new();
        open_edges.retain(|e| seen.insert(e.id()));
    }

    // Build vertex adjacency graph for open edges only
    let mut adj: BTreeMap<Vid, Vec<(usize, Vid)>> = BTreeMap::new();
    for (idx, edge) in open_edges.iter().enumerate() {
        let fid = edge.front().id();
        let bid = edge.back().id();
        adj.entry(fid).or_default().push((idx, bid));
        adj.entry(bid).or_default().push((idx, fid));
    }

    // Check if all vertices have degree 2 (required for valid loops)
    let all_degree_2 = adj.values().all(|neighbors| neighbors.len() == 2);
    if !all_degree_2 {
        {
            let non2: Vec<_> = adj.iter().filter(|(_, n)| n.len() != 2).collect();
            eprintln!(
                "[fill_loops] {} vertices with degree != 2, can't form simple loops",
                non2.len(),
            );
        }
        return false;
    }

    // Collect open edge IDs for lookup
    let open_eids: std::collections::BTreeSet<EdgeID<C>> =
        open_edges.iter().map(|e| e.id()).collect();

    // Determine the correct traversal direction for open edges.
    // Each open edge appears in exactly one existing face. The filled face
    // must traverse each edge in the OPPOSITE direction to maintain consistent
    // shell orientation (each edge traversed once forward and once backward).
    let mut edge_dir_in_shell: BTreeMap<EdgeID<C>, bool> = BTreeMap::new();
    for face in shell.iter() {
        // Use absolute_boundaries() to get edge orientations that include
        // the face's own orientation flag. This gives the EFFECTIVE traversal
        // direction for consistent shell orientation.
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                if open_eids.contains(&edge.id()) {
                    // Store the effective orientation this edge has in the existing face
                    edge_dir_in_shell.insert(edge.id(), edge.orientation());
                }
            }
        }
    }

    // Traverse to find closed loops, using OPPOSITE direction from shell
    let mut used = vec![false; open_edges.len()];
    let mut loops: Vec<Vec<Edge<Point3, C>>> = Vec::new();

    while let Some(start_idx) = used.iter().position(|&u| !u) {
        let start_edge = &open_edges[start_idx];
        // Use OPPOSITE direction from what the existing face uses
        let shell_ori = edge_dir_in_shell
            .get(&start_edge.id())
            .copied()
            .unwrap_or(true);
        // In the filled face, traverse in opposite direction
        let first_edge = if shell_ori {
            start_edge.inverse()
        } else {
            start_edge.clone()
        };
        let start_vid = first_edge.front().id();
        let mut current_vid = first_edge.back().id();
        used[start_idx] = true;

        let mut loop_edges: Vec<Edge<Point3, C>> = vec![first_edge];
        let mut stuck = false;

        while current_vid != start_vid {
            let neighbors = match adj.get(&current_vid) {
                Some(n) => n,
                None => {
                    stuck = true;
                    break;
                }
            };

            let mut found = false;
            for &(edge_idx, _other_vid) in neighbors {
                if used[edge_idx] {
                    continue;
                }
                used[edge_idx] = true;
                let edge = &open_edges[edge_idx];
                let shell_ori2 = edge_dir_in_shell.get(&edge.id()).copied().unwrap_or(true);
                // Opposite direction from shell
                let oriented_edge = if shell_ori2 {
                    edge.inverse()
                } else {
                    edge.clone()
                };
                current_vid = oriented_edge.back().id();
                loop_edges.push(oriented_edge);
                found = true;
                break;
            }

            if !found {
                stuck = true;
                break;
            }
        }

        if stuck || loop_edges.is_empty() {
            continue;
        }

        // Verify the wire is closed
        let wire_check: Wire<Point3, C> = loop_edges.iter().cloned().collect();
        if !wire_check.is_closed() {
            // If not closed, skip (direction mismatch)
            continue;
        }

        loops.push(loop_edges);
    }

    if loops.is_empty() {
        return false;
    }

    eprintln!(
        "[fill_loops] found {} closed loop(s) from {} open edges",
        loops.len(),
        open_edges.len(),
    );

    let mut added_any = false;

    for loop_edges in &loops {
        let points: Vec<Point3> = loop_edges.iter().map(|e| e.front().point()).collect();
        let n = points.len();
        if n < 3 {
            continue;
        }

        // Check planarity: use axis-aligned range check.
        let (mut min_pt, mut max_pt) = (points[0], points[0]);
        for p in &points {
            min_pt.x = min_pt.x.min(p.x);
            min_pt.y = min_pt.y.min(p.y);
            min_pt.z = min_pt.z.min(p.z);
            max_pt.x = max_pt.x.max(p.x);
            max_pt.y = max_pt.y.max(p.y);
            max_pt.z = max_pt.z.max(p.z);
        }
        let extent = max_pt - min_pt;
        let max_extent = extent.x.max(extent.y).max(extent.z);
        let min_extent = extent.x.min(extent.y).min(extent.z);

        // The loop is planar enough if the smallest extent is < 20% of the largest
        // or if the smallest extent is within model tolerance.
        if min_extent > max_extent * 0.2 && min_extent > tol * 20.0 {
            eprintln!(
                "[fill_loops] loop not planar enough (min_extent {:.4} > 20% of max_extent {:.4})",
                min_extent, max_extent,
            );
            continue;
        }

        // Compute the centroid
        let center = {
            let mut c = Vector3::new(0.0, 0.0, 0.0);
            for p in &points {
                c.x += p.x;
                c.y += p.y;
                c.z += p.z;
            }
            Point3::new(c.x / n as f64, c.y / n as f64, c.z / n as f64)
        };

        // Use the smallest-extent axis as the plane normal direction.
        // This is more robust than Newell's method when vertices are nearly
        // (but not exactly) coplanar due to perturbation.
        let normal = if extent.x <= extent.y && extent.x <= extent.z {
            Vector3::unit_x()
        } else if extent.y <= extent.x && extent.y <= extent.z {
            Vector3::unit_y()
        } else {
            Vector3::unit_z()
        };

        // Determine the sign of the normal using Newell's method on the projected polygon
        let mut newell_sign = 0.0_f64;
        for i in 0..n {
            let curr = &points[i];
            let next = &points[(i + 1) % n];
            newell_sign += (curr.y - next.y) * (curr.z + next.z) * normal.x
                + (curr.z - next.z) * (curr.x + next.x) * normal.y
                + (curr.x - next.x) * (curr.y + next.y) * normal.z;
        }
        let normal = if newell_sign < 0.0 { normal } else { -normal };

        // Create a Plane surface. The normal follows the winding order of the loop,
        // which was oriented opposite to the shell (outward from solid).
        let u_axis = if normal.x.abs() < 0.9 {
            normal.cross(Vector3::unit_x()).normalize()
        } else {
            normal.cross(Vector3::unit_y()).normalize()
        };
        let v_axis = normal.cross(u_axis).normalize();

        let plane = Plane::new(center, center + u_axis, center + v_axis);
        let surface: S = S::from(plane);

        // Build wire from loop edges
        let wire: Wire<Point3, C> = loop_edges.iter().cloned().collect();

        // Create face — try strict first, then unchecked
        let face = match Face::try_new(vec![wire.clone()], surface.clone()) {
            Ok(f) => f,
            Err(_) => Face::new_unchecked(vec![wire], surface),
        };

        eprintln!(
            "[fill_loops] added planar face with {} edges, normal=({:.3},{:.3},{:.3})",
            loop_edges.len(),
            normal.x,
            normal.y,
            normal.z,
        );
        shell.push(face);
        added_any = true;
    }

    added_any
}

// --- Phase 4: Single-Pass Shell Assembly (v2) ---
//
// Simplified assembly that relies on Phases 1+3 producing correct-by-construction
// face fragments with shared vertex identities. Only IC edge canonicalization
// and minimal weld are needed. Falls back to progressively wider welds on failure.

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

static ASSEMBLY_V2_SUCCESS: AtomicUsize = AtomicUsize::new(0);
static ASSEMBLY_V2_FALLBACK: AtomicUsize = AtomicUsize::new(0);

/// Query v2 assembly success/fallback counters.
pub fn v2_assembly_stats() -> (usize, usize) {
    (
        ASSEMBLY_V2_SUCCESS.load(AtomicOrdering::Relaxed),
        ASSEMBLY_V2_FALLBACK.load(AtomicOrdering::Relaxed),
    )
}

/// Query radial assembly success/fallback counters.
#[allow(dead_code)]
pub fn radial_assembly_stats() -> (usize, usize) {
    radial_assembly::radial_assembly_stats()
}

/// Canonicalize IC edges: unify edges that represent the same intersection
/// curve but were independently created from BSpline approximations.
///
/// Only performs Phase 1 of weld_coincident_edges (multi-point curve matching)
/// without the Phase 0 spatial grid vertex unification, since pave blocks
/// and FBG already share vertices natively.
fn canonicalize_ic_edges<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) {
    // Use the full weld with conservative tolerance — Phase 0 vertex unification
    // is idempotent when vertices are already shared, so it's safe.
    weld_coincident_edges(shell, tol, None);
}

/// Shell assembly with progressive weld tolerance escalation (v2).
///
/// Tries 3 levels of weld tolerance before giving up:
///  - Level 0: default (0.2× tau_model) — conservative, avoids merging across features
///  - Level 1: tau_weld (0.4× tau_model) — wider, catches IC approximation gaps
///  - Level 2: tau_edge_cluster (5.0× tau_model) — aggressive, last resort
///
/// At each level, if the shell closes, attempts `Solid::try_new` or accepts
/// a `ShellCondition::Closed` shell via `new_unchecked`.
fn assemble_boolean_shell_v2<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    use truck_topology::shell::ShellCondition;

    /// Try to close the shell: Solid::try_new → orientation repair → shell_condition Closed.
    fn try_close_shell<C2: ShapeOpsCurve<S2>, S2: ShapeOpsSurface>(
        shell: &Shell<Point3, C2, S2>,
        label: &str,
    ) -> Option<Solid<Point3, C2, S2>> {
        let boundaries = shell.connected_components();
        if let Ok(solid) = Solid::try_new(boundaries) {
            eprintln!("[v2_assembly] closed at {}", label);
            return Some(solid);
        }
        let boundaries = shell.connected_components();
        let all_closed = boundaries
            .iter()
            .all(|s| s.shell_condition() == ShellCondition::Closed);
        if all_closed {
            eprintln!("[v2_assembly] closed (unchecked) at {}", label);
            return Some(Solid::new_unchecked(boundaries));
        }

        // Fallback: if the shell has exactly 1 connected component, all edges
        // have refs=2, but condition is Regular (orientation inconsistency from
        // coplanar face division), accept via new_unchecked. Guards:
        // - Single component: multi-component Regular usually means wrong topology
        // - All edges refs=2: no open/over-shared edges
        // - Euler chi=2: wrong face count means wrong topology, not just orientation
        if boundaries.len() == 1 {
            let comp = &boundaries[0];
            if comp.shell_condition() == ShellCondition::Regular {
                let mut edge_refs: std::collections::HashMap<u64, usize> =
                    std::collections::HashMap::new();
                let mut vertex_ids: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                let mut face_count = 0usize;
                for face in comp.iter() {
                    face_count += 1;
                    for wire in face.absolute_boundaries().iter() {
                        for edge in wire.iter() {
                            *edge_refs.entry(edge.id().raw()).or_insert(0) += 1;
                            vertex_ids.insert(edge.front().id().raw());
                            vertex_ids.insert(edge.back().id().raw());
                        }
                    }
                }
                let edge_count = edge_refs.len();
                let vertex_count = vertex_ids.len();
                let chi = vertex_count as i64 - edge_count as i64 + face_count as i64;
                let all_refs_2 = edge_refs.values().all(|&r| r == 2);
                if all_refs_2 && chi == 2 {
                    eprintln!(
                        "[v2_assembly] accepting Regular shell (1 comp, 0 open edges, chi=2) at {}",
                        label,
                    );
                    return Some(Solid::new_unchecked(boundaries));
                }
            }
        }
        None
    }

    // Progressive weld levels: (label, weld_tol override)
    let levels: [(&str, Option<f64>); 3] = [
        ("default(0.2x)", None),                 // Level 0: 0.2× tau_model
        ("tau_weld(0.4x)", Some(tols.tau_weld)), // Level 1: 0.4× tau_model
        ("tau_edge_cluster(5.0x)", Some(tols.tau_edge_cluster)), // Level 2: 5.0× tau_model
    ];

    let mut last_open_count = 0usize;
    // Track best shell state: snapshot when open count is lowest, so we can
    // try force_merge on the best state even if a later weld makes things worse.
    let mut best_open_count = usize::MAX;
    let mut best_shell: Option<Shell<Point3, C, S>> = None;
    let mut best_label = String::new();

    for (level, (label, weld_override)) in levels.iter().enumerate() {
        // Level 0 uses canonicalize_ic_edges (includes Phase 0 + Phase 1).
        // Higher levels apply incremental weld with wider tolerance.
        if level == 0 {
            canonicalize_ic_edges(shell, tols.tau_model);
        } else {
            weld_coincident_edges(shell, tols.tau_model, *weld_override);
        }

        let open = diagnose_open_edges(shell);
        eprintln!(
            "[v2_assembly] level {} ({}): {} faces, {} open edges",
            level,
            label,
            shell.len(),
            open.len(),
        );

        if open.is_empty() {
            if let Some(solid) = try_close_shell(shell, &format!("level {} ({})", level, label)) {
                return Ok(solid);
            }
        }

        // Log open edge positions at level 0 with face details
        if level == 0 && !open.is_empty() && open.len() <= 12 {
            // For each open edge, find which faces reference it
            let mut eid_to_face: std::collections::BTreeMap<EdgeID<C>, Vec<usize>> =
                std::collections::BTreeMap::new();
            for (fi, face) in shell.iter().enumerate() {
                for wire in face.absolute_boundaries().iter() {
                    for edge in wire.iter() {
                        eid_to_face.entry(edge.id()).or_default().push(fi);
                    }
                }
            }
            for (eidx, oe) in open.iter().enumerate() {
                // Find matching edge IDs by position
                for (fi, face) in shell.iter().enumerate() {
                    for wire in face.absolute_boundaries().iter() {
                        for edge in wire.iter() {
                            let ef = edge.front().point();
                            let eb = edge.back().point();
                            let same = (ef - oe.front).magnitude() < 0.01
                                && (eb - oe.back).magnitude() < 0.01;
                            let opp = (ef - oe.back).magnitude() < 0.01
                                && (eb - oe.front).magnitude() < 0.01;
                            if same || opp {
                                let refs =
                                    eid_to_face.get(&edge.id()).map(|v| v.len()).unwrap_or(0);
                                eprintln!(
                                    "[axis_diag] open[{}] face {} has edge at axis, eid={:?}, refs={}, dir={}",
                                    eidx, fi, edge.id(), refs, if same { "same" } else { "opp" },
                                );
                            }
                        }
                    }
                }
            }
        }
        // Log open edge positions at each level for debugging
        if !open.is_empty() && open.len() <= 8 {
            for (idx, oe) in open.iter().enumerate() {
                eprintln!(
                    "[v2_assembly] L{} open_edge[{}]: ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4}) refs={}",
                    level, idx,
                    oe.front.x, oe.front.y, oe.front.z,
                    oe.back.x, oe.back.y, oe.back.z,
                    oe.face_count,
                );
            }
        }

        // Track best state for later force_merge attempt
        if !open.is_empty() && open.len() < best_open_count {
            best_open_count = open.len();
            best_shell = Some(shell.clone());
            best_label = format!("level {}", level);
        }

        last_open_count = open.len();
    }

    // Force-merge: try on current shell first, then on best-state shell if different.
    // This handles the case where an aggressive weld (level 2) corrupts the shell
    // that was in better shape at level 1.
    let shells_to_try: Vec<(Shell<Point3, C, S>, String)> = {
        let mut candidates = vec![(shell.clone(), "current".to_string())];
        if let Some(best) = best_shell.take() {
            if best_open_count < last_open_count {
                // Best shell had fewer open edges — try it first
                eprintln!(
                    "[v2_assembly] force_merge: also trying {} ({} open, better than current {})",
                    best_label, best_open_count, last_open_count,
                );
                candidates.insert(0, (best, format!("best({})", best_label)));
            }
        }
        candidates
    };

    for (mut candidate, candidate_label) in shells_to_try {
        let open_before = diagnose_open_edges(&candidate).len();
        if open_before == 0 {
            continue;
        }

        let merged = force_merge_open_edges(&mut candidate, tols.tau_edge_cluster);
        if merged > 0 {
            let open = diagnose_open_edges(&candidate);
            eprintln!(
                "[v2_assembly] force_merge({}): merged {} pairs, {} open remain",
                candidate_label,
                merged,
                open.len(),
            );

            if open.is_empty() {
                if let Some(solid) =
                    try_close_shell(&candidate, &format!("force_merge({})", candidate_label))
                {
                    return Ok(solid);
                }
            }

            last_open_count = last_open_count.min(open.len());
        }
    }

    Err(BooleanStageError::ShellAssembly(format!(
        "v2: {} open edges after all levels (3 weld + force_merge)",
        last_open_count
    )))
}

/// Finalize a boolean shell: weld edges and assemble into a Solid.
///
/// Tries radial assembly first (topology-first edge pairing), falls back
/// to v2 assembly (progressive weld tolerance escalation) on failure.
fn finalize_boolean_shell<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    // Try radial assembly first
    let faces: Vec<Face<Point3, C, S>> = shell.iter().cloned().collect();
    match radial_assembly::assemble_shell_radial(&faces, tols.tau_model) {
        Ok(solid) => {
            radial_assembly::RADIAL_SUCCESS.fetch_add(1, AtomicOrdering::Relaxed);
            eprintln!("[finalize] radial assembly succeeded");
            return Ok(solid);
        }
        Err(e) => {
            radial_assembly::RADIAL_FALLBACK.fetch_add(1, AtomicOrdering::Relaxed);
            eprintln!(
                "[finalize] radial assembly failed ({}), falling back to v2",
                e
            );
        }
    }

    // Fallback: v2 progressive weld assembly
    assemble_boolean_shell_v2(shell, tols)
}

/// Finalize with diagnostics recovery tracking.
///
/// Tries radial assembly first, falls back to v2 progressive weld.
/// Populates recovery diagnostics.
fn finalize_boolean_shell_with_recovery_v2<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tols: &BooleanTolerance,
    recovery: &mut diagnostics::RecoveryReport,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    // Try radial assembly first
    let faces: Vec<Face<Point3, C, S>> = shell.iter().cloned().collect();
    match radial_assembly::assemble_shell_radial(&faces, tols.tau_model) {
        Ok(solid) => {
            radial_assembly::RADIAL_SUCCESS.fetch_add(1, AtomicOrdering::Relaxed);
            eprintln!("[finalize_diag] radial assembly succeeded");
            let result_shell = &solid.boundaries()[0];
            populate_euler(&mut Some(recovery), result_shell);
            recovery.recovery_level = 0;
            return Ok(solid);
        }
        Err(e) => {
            radial_assembly::RADIAL_FALLBACK.fetch_add(1, AtomicOrdering::Relaxed);
            eprintln!(
                "[finalize_diag] radial assembly failed ({}), falling back to v2",
                e
            );
        }
    }

    // Fallback: v2 progressive weld assembly
    let result = assemble_boolean_shell_v2(shell, tols);
    populate_euler(&mut Some(recovery), shell);

    match &result {
        Ok(_) => {
            recovery.recovery_level = 0;
        }
        Err(_) => {
            recovery.recovery_level = 3;
        }
    }

    result
}

/// Populate Euler characteristic in recovery report.
fn populate_euler<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    recovery: &mut Option<&mut diagnostics::RecoveryReport>,
    shell: &Shell<Point3, C, S>,
) {
    if let Some(ref mut r) = *recovery {
        match validate_euler_characteristic(shell) {
            Ok(()) => {
                r.euler_valid = true;
                r.euler_chi = 2;
            }
            Err((_v, _e, _f, chi)) => {
                r.euler_valid = false;
                r.euler_chi = chi;
            }
        }
    }
}

/// AND operation between two solids, returning a structured error on failure.
pub fn and_result<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    and_result_with_tol(solid0, solid1, &BooleanTolerance::from_model_tol(tol))
}

/// AND operation with per-stage tolerance control.
///
/// SequentialID provides deterministic ordering natively, so no
/// explicit context wrapper is needed.
pub fn and_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [mut and_shell, _] = process_one_pair_of_shells_result_with_tol(shell0, shell1, tols)?;
    for shell in iter0 {
        let [res, _] = process_one_pair_of_shells_result_with_tol(&and_shell, shell, tols)?;
        and_shell = res;
    }
    for shell in iter1 {
        let [res, _] = process_one_pair_of_shells_result_with_tol(&and_shell, shell, tols)?;
        and_shell = res;
    }
    finalize_boolean_shell(&mut and_shell, tols)
}

/// AND operation with per-stage tolerance control, returning diagnostics.
///
/// Same as `and_result_with_tol`, but also collects and returns a
/// `BooleanDiagnostics` report with classification, intersection,
/// division, and recovery statistics.
pub fn and_result_with_tol_diag<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<(Solid<Point3, C, S>, BooleanDiagnostics), BooleanStageError> {
    let mut diag = BooleanDiagnostics::default();
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let mut classified =
        classify_one_pair_of_shells_result_with_tol(shell0, shell1, tols, Some(&mut diag))?;
    let mut and_shell = classified.and0;
    and_shell.append(&mut classified.and1);
    for shell in iter0 {
        let mut classified =
            classify_one_pair_of_shells_result_with_tol(&and_shell, shell, tols, Some(&mut diag))?;
        and_shell = classified.and0;
        and_shell.append(&mut classified.and1);
    }
    for shell in iter1 {
        let mut classified =
            classify_one_pair_of_shells_result_with_tol(&and_shell, shell, tols, Some(&mut diag))?;
        and_shell = classified.and0;
        and_shell.append(&mut classified.and1);
    }
    let solid = finalize_boolean_shell_with_recovery_v2(&mut and_shell, tols, &mut diag.recovery)?;
    Ok((solid, diag))
}

/// AND operation between two solids.
pub fn and<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> Option<Solid<Point3, C, S>> {
    and_result(solid0, solid1, tol).ok()
}

/// AND operation with per-stage tolerance control.
pub fn and_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> Option<Solid<Point3, C, S>> {
    and_result_with_tol(solid0, solid1, tols).ok()
}

/// OR operation between two solids, returning a structured error on failure.
pub fn or_result<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    or_result_with_tol(solid0, solid1, &BooleanTolerance::from_model_tol(tol))
}

/// OR operation with per-stage tolerance control.
///
/// SequentialID provides deterministic ordering natively.
pub fn or_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [_, mut or_shell] = process_one_pair_of_shells_result_with_tol(shell0, shell1, tols)?;
    for shell in iter0 {
        let [_, res] = process_one_pair_of_shells_result_with_tol(&or_shell, shell, tols)?;
        or_shell = res;
    }
    for shell in iter1 {
        let [_, res] = process_one_pair_of_shells_result_with_tol(&or_shell, shell, tols)?;
        or_shell = res;
    }
    finalize_boolean_shell(&mut or_shell, tols)
}

/// OR operation with per-stage tolerance control, returning diagnostics.
///
/// Same as `or_result_with_tol`, but also collects and returns a
/// `BooleanDiagnostics` report.
pub fn or_result_with_tol_diag<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<(Solid<Point3, C, S>, BooleanDiagnostics), BooleanStageError> {
    let mut diag = BooleanDiagnostics::default();
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let mut classified =
        classify_one_pair_of_shells_result_with_tol(shell0, shell1, tols, Some(&mut diag))?;
    let mut or_shell = classified.or0;
    or_shell.append(&mut classified.or1);
    for shell in iter0 {
        let mut classified =
            classify_one_pair_of_shells_result_with_tol(&or_shell, shell, tols, Some(&mut diag))?;
        or_shell = classified.or0;
        or_shell.append(&mut classified.or1);
    }
    for shell in iter1 {
        let mut classified =
            classify_one_pair_of_shells_result_with_tol(&or_shell, shell, tols, Some(&mut diag))?;
        or_shell = classified.or0;
        or_shell.append(&mut classified.or1);
    }
    let solid = finalize_boolean_shell_with_recovery_v2(&mut or_shell, tols, &mut diag.recovery)?;
    Ok((solid, diag))
}

/// OR operation between two solids.
pub fn or<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> Option<Solid<Point3, C, S>> {
    or_result(solid0, solid1, tol).ok()
}

/// OR operation with per-stage tolerance control.
pub fn or_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> Option<Solid<Point3, C, S>> {
    or_result_with_tol(solid0, solid1, tols).ok()
}

/// Difference operation: A \ B, returning a structured error on failure.
/// Selects faces of A outside B (Or), plus faces of B inside A (And) with inverted orientation.
pub fn difference_result<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    difference_result_with_tol(solid0, solid1, &BooleanTolerance::from_model_tol(tol))
}

/// Difference operation with per-stage tolerance control.
///
/// SequentialID provides deterministic ordering natively.
///
/// For multi-shell solid0 (A), each shell is processed independently against
/// solid1 (B): `(A0 ∪ A1) \ B = (A0 \ B) ∪ (A1 \ B)`. This prevents
/// disjoint shells from being lost when they don't intersect each other.
pub fn difference_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    let shells0 = solid0.boundaries();
    let shells1 = solid1.boundaries();
    let mut all_diff_faces: Vec<Face<Point3, C, S>> = Vec::new();

    for shell0 in shells0.iter() {
        let mut shell1_iter = shells1.iter();
        let first_shell1 = shell1_iter.next().unwrap();
        let ClassifiedShellBuckets { or0, and1, .. } =
            classify_one_pair_of_shells_result_with_tol(shell0, first_shell1, tols, None)?;
        let mut diff_faces: Vec<Face<Point3, C, S>> = or0.into_iter().collect();
        for face in and1.into_iter() {
            diff_faces.push(face.inverse());
        }
        for additional_b in shell1_iter {
            let diff_shell: Shell<Point3, C, S> = diff_faces.into_iter().collect();
            let classified =
                classify_one_pair_of_shells_result_with_tol(&diff_shell, additional_b, tols, None)?;
            diff_faces = classified.or0.into_iter().collect();
            for face in classified.and1.into_iter() {
                diff_faces.push(face.inverse());
            }
        }
        all_diff_faces.extend(diff_faces);
    }

    let mut diff_shell: Shell<Point3, C, S> = all_diff_faces.into_iter().collect();
    finalize_boolean_shell(&mut diff_shell, tols)
}

/// Difference operation with per-stage tolerance control, returning diagnostics.
///
/// Same as `difference_result_with_tol`, but also collects and returns a
/// `BooleanDiagnostics` report.
pub fn difference_result_with_tol_diag<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<(Solid<Point3, C, S>, BooleanDiagnostics), BooleanStageError> {
    let mut diag = BooleanDiagnostics::default();
    let shells0 = solid0.boundaries();
    let shells1 = solid1.boundaries();
    let mut all_diff_faces: Vec<Face<Point3, C, S>> = Vec::new();

    for shell0 in shells0.iter() {
        let mut shell1_iter = shells1.iter();
        let first_shell1 = shell1_iter.next().unwrap();
        let ClassifiedShellBuckets { or0, and1, .. } = classify_one_pair_of_shells_result_with_tol(
            shell0,
            first_shell1,
            tols,
            Some(&mut diag),
        )?;
        let mut diff_faces: Vec<Face<Point3, C, S>> = or0.into_iter().collect();
        for face in and1.into_iter() {
            diff_faces.push(face.inverse());
        }
        for additional_b in shell1_iter {
            let diff_shell: Shell<Point3, C, S> = diff_faces.into_iter().collect();
            let classified = classify_one_pair_of_shells_result_with_tol(
                &diff_shell,
                additional_b,
                tols,
                Some(&mut diag),
            )?;
            diff_faces = classified.or0.into_iter().collect();
            for face in classified.and1.into_iter() {
                diff_faces.push(face.inverse());
            }
        }
        all_diff_faces.extend(diff_faces);
    }

    let mut diff_shell: Shell<Point3, C, S> = all_diff_faces.into_iter().collect();
    let solid = finalize_boolean_shell_with_recovery_v2(&mut diff_shell, tols, &mut diag.recovery)?;
    Ok((solid, diag))
}

/// Difference operation: A \ B.
/// Selects faces of A outside B (Or), plus faces of B inside A (And) with inverted orientation.
pub fn difference<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> Option<Solid<Point3, C, S>> {
    difference_result(solid0, solid1, tol).ok()
}

/// Difference operation with per-stage tolerance control.
pub fn difference_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> Option<Solid<Point3, C, S>> {
    difference_result_with_tol(solid0, solid1, tols).ok()
}

#[cfg(test)]
mod tests;
