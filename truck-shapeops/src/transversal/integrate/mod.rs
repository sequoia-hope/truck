use crate::alternative::Alternative;

use super::*;
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

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
}

impl BooleanTolerance {
    /// All stages use the same tolerance. Matches legacy single-tol behavior.
    pub fn uniform(tol: f64) -> Self {
        Self {
            tau_model: tol,
            tau_mesh: tol,
            tau_weld: tol,
            tau_coplanar: tol,
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
pub(crate) fn irrational_ray_dirs() -> [Vector3; 4] {
    let sqrt2 = std::f64::consts::SQRT_2;
    let sqrt3 = 3.0f64.sqrt();
    let sqrt5 = 5.0f64.sqrt();
    let sqrt7 = 7.0f64.sqrt();
    let sqrt11 = 11.0f64.sqrt();
    let sqrt13 = 13.0f64.sqrt();
    [
        Vector3::new(1.0, sqrt2 / 10.0, sqrt3 / 10.0),
        Vector3::new(sqrt2 / 10.0, 1.0, sqrt5 / 10.0),
        Vector3::new(sqrt3 / 10.0, sqrt5 / 10.0, 1.0),
        Vector3::new(sqrt7 / 10.0, sqrt11 / 10.0, sqrt13 / 10.0),
    ]
}

/// Ray-cast a face against a triangulated shell to determine inside/outside.
///
/// Uses scale-adaptive perturbation and irrational ray directions with majority
/// voting to avoid false classifications from grid-aligned triangulation.
fn ray_cast_classify<C, S>(
    face: &Face<Point3, C, S>,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
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

    // Majority vote: cast all 4 irrational rays, take majority (need >=2 agreeing).
    let majority_vote = |pt: Point3| -> Option<isize> {
        let mut inside = 0u32;
        let mut outside = 0u32;
        for &d in &dirs {
            if let Some(c) = try_ray_cast(pt, d, poly_shell) {
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
        if inside >= 2 {
            Some(1)
        } else if outside >= 2 {
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

    // Last resort: accept any single non-None result from any direction.
    let centroid_perturbed = centroid + perturb;
    for &d in &dirs {
        if let Some(c) = try_ray_cast(centroid_perturbed, d, poly_shell) {
            return Some(if c.unsigned_abs() % 2 == 1 { 1 } else { 0 });
        }
    }

    None
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
) -> std::result::Result<ClassifiedShellBuckets<Point3, C, S>, BooleanStageError> {
    nonpositive_tolerance!(tols.tau_model);
    let poly_shell0 = shell0.triangulation(tols.tau_mesh);
    let poly_shell1 = shell1.triangulation(tols.tau_mesh);
    let altshell0: AltCurveShell<C, S> =
        shell0.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let altshell1: AltCurveShell<C, S> =
        shell1.mapped(|x| *x, |c| Alternative::FirstType(c.clone()), Clone::clone);
    let loops_store::LoopsStoreQuadruple {
        geom_loops_store0: loops_store0,
        geom_loops_store1: loops_store1,
        coplanar_faces0,
        coplanar_faces1,
        ..
    } = loops_store::create_loops_stores(
        &altshell0,
        &poly_shell0,
        &altshell1,
        &poly_shell1,
        tols.tau_model,
        Some(tols.tau_coplanar),
    )
    .ok_or(BooleanStageError::LoopsStoreCreation)?;
    let (mut cls0, coplanar_fids0) = divide_face::divide_faces_with_coplanar(
        &altshell0,
        &loops_store0,
        tols.tau_model,
        &coplanar_faces0,
    )
    .ok_or(BooleanStageError::FaceDivision)?;
    cls0.integrate_by_component();
    let (mut cls1, coplanar_fids1) = divide_face::divide_faces_with_coplanar(
        &altshell1,
        &loops_store1,
        tols.tau_model,
        &coplanar_faces1,
    )
    .ok_or(BooleanStageError::FaceDivision)?;
    cls1.integrate_by_component();
    // Reset overlapping coplanar fragments to Unknown for re-classification.
    cls0.reset_overlapping_coplanar(&coplanar_fids0, shell1, true, tols.tau_coplanar);
    cls1.reset_overlapping_coplanar(&coplanar_fids1, shell0, false, tols.tau_coplanar);
    let [mut and0, mut or0, unknown0] = cls0.and_or_unknown();
    unknown0
        .into_iter()
        .try_for_each(|face| {
            if let Some(action) =
                coplanar::classify_coplanar_fragment(&face, shell1, true, tols.tau_coplanar)
            {
                match action {
                    coplanar::CoplanarAction::Remove => {}
                    coplanar::CoplanarAction::And => and0.push(face),
                    coplanar::CoplanarAction::Or => or0.push(face),
                }
                return Some(());
            }
            let count = ray_cast_classify(&face, &poly_shell1)?;
            if count == 1 {
                and0.push(face);
            } else {
                or0.push(face);
            }
            Some(())
        })
        .ok_or(BooleanStageError::Classification)?;
    let [mut and1, mut or1, unknown1] = cls1.and_or_unknown();
    unknown1
        .into_iter()
        .try_for_each(|face| {
            if let Some(action) =
                coplanar::classify_coplanar_fragment(&face, shell0, false, tols.tau_coplanar)
            {
                match action {
                    coplanar::CoplanarAction::Remove => {}
                    coplanar::CoplanarAction::And => and1.push(face),
                    coplanar::CoplanarAction::Or => or1.push(face),
                }
                return Some(());
            }
            let count = ray_cast_classify(&face, &poly_shell0)?;
            if count == 1 {
                and1.push(face);
            } else {
                or1.push(face);
            }
            Some(())
        })
        .ok_or(BooleanStageError::Classification)?;
    Ok(ClassifiedShellBuckets {
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
    })
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
    } = classify_one_pair_of_shells_result_with_tol(shell0, shell1, tols)?;
    and0.append(&mut and1);
    or0.append(&mut or1);
    Ok([and0, or0])
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
/// 3. Fallback: `Face::new_unchecked` for any remaining case where wires are
///    non-empty and closed.
fn try_split_non_simple_wires<C: Clone, S: Clone>(
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

    // Final fallback: accept non-disjoint or non-simple wires via new_unchecked.
    // Non-disjoint wires (sharing vertices at T-junctions) are geometrically
    // valid — each edge still appears exactly once per face. The shared vertex
    // creates a singular vertex but doesn't affect edge counting for shell
    // condition (Closed/Oriented).
    let all_closed = new_wires.iter().all(|w| !w.is_empty() && w.is_closed());
    if !new_wires.is_empty() && all_closed {
        let mut new_face = Face::new_unchecked(new_wires, surface.clone());
        if !ori {
            new_face.invert();
        }
        return Some(new_face);
    }

    None
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
        let unify_tol = weld_tol.unwrap_or_else(|| (tol * 0.2).max(TOLERANCE.sqrt()));

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
                                    // Skip degenerate edges where vertex unification
                                    // collapsed both endpoints to the same vertex.
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
                    // to the original face.
                    match Face::try_new(new_wires.clone(), surface.clone()) {
                        Ok(mut new_face) => {
                            if !ori {
                                new_face.invert();
                            }
                            new_face
                        }
                        Err(_e) => {
                            // Try wire splitting before falling back to original
                            if let Some(split_face) =
                                try_split_non_simple_wires(&new_wires, &surface, ori)
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
        }
    }

    // Phase 1: Build canonical edge map using curve midpoint clustering.
    //
    // Two edges between the same vertex pair may represent different geometric
    // features (e.g., outer boundary + hole of a T-junction face). By clustering
    // edges with matching (vertex_pair, curve_midpoint), each distinct geometric
    // feature gets its own canonical edge, preventing within-face conflicts and
    // ensuring each edge pair is correctly shared between exactly 2 faces.
    type VPairKey = (Vid, Vid);

    // Collect edges grouped by vertex pair, with curve midpoints
    type EdgeMid<C2> = (Edge<Point3, C2>, Point3);
    let mut pair_groups: FxHashMap<VPairKey, Vec<EdgeMid<C>>> = FxHashMap::default();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let abs = edge.absolute_clone();
                let fid = abs.front().id();
                let bid = abs.back().id();

                // Compute curve midpoint for geometric matching
                let curve = abs.curve();
                let (t0, t1) = curve.range_tuple();
                let midpoint = curve.subs((t0 + t1) * 0.5);

                // Normalize key: check both orderings, or create new
                let key = if pair_groups.contains_key(&(fid, bid)) {
                    (fid, bid)
                } else if pair_groups.contains_key(&(bid, fid)) {
                    (bid, fid)
                } else {
                    (fid, bid)
                };

                pair_groups.entry(key).or_default().push((abs, midpoint));
            }
        }
    }

    // For each vertex pair, cluster by midpoint proximity and assign canonicals
    let mid_tol = tol * 5.0;
    let mut edge_to_canonical: FxHashMap<EdgeID<C>, Edge<Point3, C>> = FxHashMap::default();

    for edges in pair_groups.values() {
        if edges.len() <= 1 {
            continue;
        }

        // Cluster by midpoint proximity
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; edges.len()];

        for i in 0..edges.len() {
            if assigned[i] {
                continue;
            }
            assigned[i] = true;
            let mut cluster = vec![i];
            let ref_mid = edges[i].1;

            for j in (i + 1)..edges.len() {
                if assigned[j] {
                    continue;
                }
                if (edges[j].1 - ref_mid).magnitude() < mid_tol {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }

        // Assign canonical within each cluster
        for cluster in clusters {
            if cluster.len() <= 1 {
                continue;
            }
            let canon = &edges[cluster[0]].0;
            for &idx in &cluster {
                if edges[idx].0.id() != canon.id() {
                    edge_to_canonical.insert(edges[idx].0.id(), canon.clone());
                }
            }
        }
    }

    // Phase 2: Rebuild faces with canonical edges
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
                        .map(|edge| match edge_to_canonical.get(&edge.id()) {
                            Some(c) => {
                                let abs_edge = edge.absolute_clone();
                                let same_dir = abs_edge.front().id() == c.front().id();
                                if same_dir == edge.orientation() {
                                    c.clone()
                                } else {
                                    c.inverse()
                                }
                            }
                            None => edge.clone(),
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
    {
        // Count total references per edge ID across all faces
        let mut edge_ref_count: FxHashMap<EdgeID<C>, usize> = FxHashMap::default();
        for face in shell.iter() {
            for wire in face.absolute_boundaries().iter() {
                for edge in wire.iter() {
                    *edge_ref_count.entry(edge.id()).or_insert(0) += 1;
                }
            }
        }

        let overcounted: FxHashSet<EdgeID<C>> = edge_ref_count
            .iter()
            .filter(|(_, count)| **count > 2)
            .map(|(id, _)| *id)
            .collect();

        if !overcounted.is_empty() {
            // Rebuild faces, cloning excess edge references
            let mut edge_seen_count: FxHashMap<EdgeID<C>, usize> = FxHashMap::default();
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

/// Finalize a boolean shell: weld edges and assemble into a Solid.
///
/// Runs `weld_coincident_edges`, then tries `Solid::try_new`. If the shell
/// is `Oriented` (open boundary) instead of `Closed`, retries with progressively
/// wider weld tolerances to close small gaps.
///
/// Boolean operations may produce T-junction vertices (from vertex unification
/// in `add_polygon_vertex`) where two wires share a vertex. These vertices are
/// topologically "singular" (non-manifold local topology), but geometrically
/// valid for the CAD pipeline. When the shell is `Closed` but has singular
/// vertices, we accept the solid via `new_unchecked`.
fn finalize_boolean_shell<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    use truck_topology::shell::ShellCondition;

    weld_coincident_edges(shell, tols.tau_model, None);

    // Try standard assembly
    let boundaries = shell.connected_components();
    if let Ok(solid) = Solid::try_new(boundaries) {
        return Ok(solid);
    }

    // If the shell isn't closed, try wider weld tolerances to close gaps.
    let wider_tols = [tols.tau_model * 2.0, tols.tau_model * 5.0];
    for &wider in &wider_tols {
        weld_coincident_edges(shell, tols.tau_model, Some(wider));
        let boundaries = shell.connected_components();
        if let Ok(solid) = Solid::try_new(boundaries) {
            return Ok(solid);
        }
    }

    // Accept Closed shells with singular vertices from T-junctions.
    // Singular vertices arise from vertex unification at intersection curve
    // endpoints where two face boundaries meet at a single point.
    // Solid::try_new rejects them but the geometry is correct for
    // downstream tessellation and further booleans.
    let boundaries = shell.connected_components();
    let acceptable = boundaries.iter().all(|s| !s.is_empty() && s.is_connected());
    let all_closed = boundaries
        .iter()
        .all(|s| s.shell_condition() == ShellCondition::Closed);
    if acceptable && all_closed {
        return Ok(Solid::new_unchecked(boundaries));
    }

    let boundaries = shell.connected_components();
    Solid::try_new(boundaries)
        .map_err(|e| BooleanStageError::ShellAssembly(format!("Solid::try_new failed: {:?}", e)))
}

/// AND operation between two solids, returning a structured error on failure.
pub fn and_result<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    and_result_with_tol(solid0, solid1, &BooleanTolerance::uniform(tol))
}

/// AND operation with per-stage tolerance control.
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
    or_result_with_tol(solid0, solid1, &BooleanTolerance::uniform(tol))
}

/// OR operation with per-stage tolerance control.
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
    difference_result_with_tol(solid0, solid1, &BooleanTolerance::uniform(tol))
}

/// Difference operation with per-stage tolerance control.
pub fn difference_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let ClassifiedShellBuckets { or0, and1, .. } =
        classify_one_pair_of_shells_result_with_tol(shell0, shell1, tols)?;
    // Difference = or0 (A faces outside B) + inverted and1 (B faces inside A, flipped)
    let mut diff_faces: Vec<Face<Point3, C, S>> = or0.into_iter().collect();
    for face in and1.into_iter() {
        diff_faces.push(face.inverse());
    }
    let mut diff_shell: Shell<Point3, C, S> = diff_faces.into_iter().collect();
    // Handle additional boundary shells (multi-shell solids)
    for shell in iter0 {
        let classified = classify_one_pair_of_shells_result_with_tol(&diff_shell, shell, tols)?;
        let mut faces: Vec<Face<Point3, C, S>> = classified.or0.into_iter().collect();
        for face in classified.and1.into_iter() {
            faces.push(face.inverse());
        }
        diff_shell = faces.into_iter().collect();
    }
    for shell in iter1 {
        let classified = classify_one_pair_of_shells_result_with_tol(&diff_shell, shell, tols)?;
        let mut faces: Vec<Face<Point3, C, S>> = classified.or0.into_iter().collect();
        for face in classified.and1.into_iter() {
            faces.push(face.inverse());
        }
        diff_shell = faces.into_iter().collect();
    }
    finalize_boolean_shell(&mut diff_shell, tols)
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
