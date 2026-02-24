use crate::alternative::Alternative;

use super::*;
use std::cell::RefCell;
use truck_base::id::{DetContext, DetId};
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

// ---------------------------------------------------------------------------
// Deterministic ID context (thread-local, scoped to each boolean operation)
// ---------------------------------------------------------------------------

thread_local! {
    static DET_CONTEXT: RefCell<Option<DetContext>> = const { RefCell::new(None) };
}

/// Run a closure with a fresh deterministic ID context.
///
/// All calls to `next_det_id()` within `f` produce sequential IDs starting
/// from 0. Contexts do not nest — calling `with_det_context` while one is
/// already active replaces it.
fn with_det_context<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    DET_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(DetContext::new());
    });
    let result = f();
    DET_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = None;
    });
    result
}

/// Assign deterministic IDs to all unique vertices in a shell, ordered by
/// spatial position (lexicographic x, y, z). Returns a map from pointer-based
/// `VertexID` to `DetId`.
///
/// Falls back to sequential pointer-ID ordering if not in a `with_det_context`
/// scope (so existing non-det-context callers still work).
fn assign_vertex_det_ids<C, S>(
    shell: &Shell<Point3, C, S>,
) -> rustc_hash::FxHashMap<VertexID<Point3>, DetId> {
    type Vid = VertexID<Point3>;
    let mut unique_verts: Vec<(Vid, Point3)> = Vec::new();
    let mut seen: rustc_hash::FxHashSet<Vid> = rustc_hash::FxHashSet::default();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for v in wire.vertex_iter() {
                if seen.insert(v.id()) {
                    unique_verts.push((v.id(), v.point()));
                }
            }
        }
    }
    // Sort by spatial position for deterministic ordering.
    unique_verts.sort_by(|a, b| {
        let pa = (a.1.x, a.1.y, a.1.z);
        let pb = (b.1.x, b.1.y, b.1.z);
        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Assign sequential DetIds (either from thread-local context or standalone).
    let has_ctx = DET_CONTEXT.with(|ctx| ctx.borrow().is_some());
    unique_verts
        .iter()
        .enumerate()
        .map(|(i, (vid, _))| {
            let det = if has_ctx {
                DET_CONTEXT.with(|ctx| ctx.borrow().as_ref().unwrap().next_id())
            } else {
                DetId::from_raw(i as u64)
            };
            (*vid, det)
        })
        .collect()
}

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

    // Majority vote: cast all 4 irrational rays, take majority (need >=2 agreeing).
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
        if let Some(c) = cast_ray(centroid_perturbed, d) {
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
    diag: Option<&mut BooleanDiagnostics>,
) -> std::result::Result<ClassifiedShellBuckets<Point3, C, S>, BooleanStageError> {
    nonpositive_tolerance!(tols.tau_model);
    let _total_start = std::time::Instant::now();
    #[cfg(debug_assertions)]
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
    let _loops_store_elapsed = _total_start.elapsed();
    #[cfg(debug_assertions)]
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
    #[cfg(debug_assertions)]
    eprintln!(
        "[classify] shell0: and={}, or={}, unknown={}",
        and0.len(),
        or0.len(),
        unknown0.len(),
    );
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
            let count = ray_cast_classify(&face, &poly_shell1, Some(&bvh1))?;
            if count == 1 {
                and0.push(face);
            } else {
                or0.push(face);
            }
            Some(())
        })
        .ok_or(BooleanStageError::Classification)?;
    #[cfg(debug_assertions)]
    eprintln!(
        "[classify] shell0 final: and={}, or={}",
        and0.len(),
        or0.len(),
    );
    let [mut and1, mut or1, unknown1] = cls1.and_or_unknown();
    #[cfg(debug_assertions)]
    eprintln!(
        "[classify] shell1: and={}, or={}, unknown={}",
        and1.len(),
        or1.len(),
        unknown1.len(),
    );
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
            let count = ray_cast_classify(&face, &poly_shell0, Some(&bvh0))?;
            if count == 1 {
                and1.push(face);
            } else {
                or1.push(face);
            }
            Some(())
        })
        .ok_or(BooleanStageError::Classification)?;
    #[cfg(debug_assertions)]
    eprintln!(
        "[classify] shell1 final: and={}, or={}",
        and1.len(),
        or1.len(),
    );
    #[cfg(debug_assertions)]
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
    #[cfg(debug_assertions)]
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
                #[cfg(debug_assertions)]
                eprintln!("[gap_repair] gap too large: {:.4}", best_dist);
                return None;
            }

            let vf = vertex_map.get(&v0)?;
            let vb = vertex_map.get(&v1)?;

            // Create a Line edge from v0 to v1
            let pf = vf.point();
            let pb = vb.point();
            #[cfg(debug_assertions)]
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
                #[cfg(debug_assertions)]
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
            #[cfg(debug_assertions)]
            eprintln!("[gap_repair] graph traversal stuck");
            return None;
        }

        let wire: Wire<Point3, C> = wire_edges.into_iter().collect();
        if !wire.is_closed() {
            #[cfg(debug_assertions)]
            eprintln!(
                "[gap_repair] reconstructed wire not closed ({} edges)",
                wire.len()
            );
            return None;
        }
        #[cfg(debug_assertions)]
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
        #[cfg(debug_assertions)]
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
    // Uses BTreeMap<(DetId, DetId), _> for deterministic iteration order.
    // DetIds are assigned to vertices by spatial position ordering (see
    // assign_vertex_det_ids), so the BTreeMap key order is the same across
    // runs regardless of pointer addresses.

    // Assign deterministic IDs to post-Phase-0 vertices for ordering.
    let vid_to_det = assign_vertex_det_ids(shell);
    type DetPairKey = (DetId, DetId);

    // Edge candidate: edge with face ownership and 3-point curve samples.
    struct EdgeCandidate<C2> {
        edge: Edge<Point3, C2>,
        face_idx: usize,
        samples: [[f64; 3]; 3],
    }

    // Build edge candidate list with face ownership
    let mut candidates: Vec<EdgeCandidate<C>> = Vec::new();
    let mut candidate_det_pair: Vec<DetPairKey> = Vec::new();

    for (face_idx, face) in shell.iter().enumerate() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let abs = edge.absolute_clone();
                let fid = abs.front().id();
                let bid = abs.back().id();

                // Map pointer-derived IDs to deterministic IDs.
                let det_fid = vid_to_det
                    .get(&fid)
                    .copied()
                    .unwrap_or(DetId::from_raw(u64::MAX));
                let det_bid = vid_to_det
                    .get(&bid)
                    .copied()
                    .unwrap_or(DetId::from_raw(u64::MAX - 1));

                let det_pair = if det_fid <= det_bid {
                    (det_fid, det_bid)
                } else {
                    (det_bid, det_fid)
                };

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

                candidate_det_pair.push(det_pair);
                candidates.push(EdgeCandidate {
                    edge: abs,
                    face_idx,
                    samples,
                });
            }
        }
    }

    // Group candidate indices by vertex pair
    let mut pair_groups: std::collections::BTreeMap<DetPairKey, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, pair) in candidate_det_pair.iter().enumerate() {
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
                                let same_dir = abs_edge.front().id() == c.front().id();
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

/// Targeted re-weld for specific open edges in a shell.
///
/// Unlike `weld_coincident_edges` which processes all edges, this function
/// focuses only on edges that have != 2 face references (open edges).
/// For each open edge, it searches all other edges for a geometric twin
/// (same vertex positions + midpoint within tolerance) and replaces the
/// duplicate with the canonical one.
fn targeted_open_edge_reweld<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) {
    // Collect all edges with their geometric data
    struct EdgeInfo<C2> {
        id: EdgeID<C2>,
        front: Point3,
        back: Point3,
        midpoint: Point3,
        face_count: usize,
    }

    let mut edge_ref_count: std::collections::BTreeMap<EdgeID<C>, usize> =
        std::collections::BTreeMap::new();
    let mut edge_info: std::collections::BTreeMap<EdgeID<C>, (Point3, Point3, Point3)> =
        std::collections::BTreeMap::new();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let eid = edge.id();
                *edge_ref_count.entry(eid).or_insert(0) += 1;
                edge_info.entry(eid).or_insert_with(|| {
                    let abs = edge.absolute_clone();
                    let curve = abs.curve();
                    let (t0, t1) = curve.range_tuple();
                    let midpoint = curve.subs((t0 + t1) * 0.5);
                    (abs.front().point(), abs.back().point(), midpoint)
                });
            }
        }
    }

    // Find open edges (face_count == 1) and build edge info
    let mut all_edges: Vec<EdgeInfo<C>> = Vec::new();
    for (&eid, &count) in &edge_ref_count {
        if let Some(&(front, back, midpoint)) = edge_info.get(&eid) {
            all_edges.push(EdgeInfo {
                id: eid,
                front,
                back,
                midpoint,
                face_count: count,
            });
        }
    }

    // Find pairs of open edges (count=1) that are geometric twins
    let mut reweld_map: std::collections::BTreeMap<EdgeID<C>, EdgeID<C>> =
        std::collections::BTreeMap::new();

    let open_edges: Vec<usize> = all_edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.face_count == 1)
        .map(|(i, _)| i)
        .collect();

    for i in 0..open_edges.len() {
        let ei = &all_edges[open_edges[i]];
        if reweld_map.contains_key(&ei.id) {
            continue;
        }
        for j in (i + 1)..open_edges.len() {
            let ej = &all_edges[open_edges[j]];
            if reweld_map.contains_key(&ej.id) {
                continue;
            }
            // Check if these edges are geometric twins (same vertices, same midpoint)
            let fwd_match =
                (ei.front - ej.front).magnitude() < tol && (ei.back - ej.back).magnitude() < tol;
            let rev_match =
                (ei.front - ej.back).magnitude() < tol && (ei.back - ej.front).magnitude() < tol;
            let mid_match = (ei.midpoint - ej.midpoint).magnitude() < tol;

            if (fwd_match || rev_match) && mid_match {
                // Map the higher-ID edge to the lower-ID edge (canonical)
                if ei.id < ej.id {
                    reweld_map.insert(ej.id, ei.id);
                } else {
                    reweld_map.insert(ei.id, ej.id);
                }
            }
        }
    }

    if reweld_map.is_empty() {
        return;
    }

    // Build a map from edge ID to actual Edge object (canonical edges)
    let mut canonical_edges: std::collections::BTreeMap<EdgeID<C>, Edge<Point3, C>> =
        std::collections::BTreeMap::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let eid = edge.id();
                if reweld_map.values().any(|&target| target == eid) {
                    canonical_edges
                        .entry(eid)
                        .or_insert_with(|| edge.absolute_clone());
                }
            }
        }
    }

    // Rebuild faces, replacing non-canonical edges with their canonical twin
    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let mut any_replaced = false;
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .map(|edge| {
                            if let Some(&target_id) = reweld_map.get(&edge.id()) {
                                if let Some(canonical) = canonical_edges.get(&target_id) {
                                    any_replaced = true;
                                    let abs = edge.absolute_clone();
                                    let same_dir = abs.front().id() == canonical.front().id();
                                    if same_dir == edge.orientation() {
                                        canonical.clone()
                                    } else {
                                        canonical.inverse()
                                    }
                                } else {
                                    edge.clone()
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
}

/// Position-based edge reweld: groups edges purely by geometric position
/// (vertex coordinates + curve midpoint) and unifies duplicates.
///
/// This bypasses vertex ID-based matching entirely, handling cases where
/// Phase 0 vertex unification failed or was skipped to prevent degenerate
/// edges. Each geometric edge group gets a single canonical edge, and all
/// faces are rebuilt to reference the canonical.
fn position_based_edge_reweld<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) {
    // Quantize a position to a grid cell for grouping
    let quantize = |p: Point3| -> (i64, i64, i64) {
        let inv = 1.0 / tol;
        (
            (p.x * inv).round() as i64,
            (p.y * inv).round() as i64,
            (p.z * inv).round() as i64,
        )
    };

    // Geometric key for an edge: quantized (front, back, midpoint), normalized
    // so that the smaller vertex comes first (for direction-independent matching).
    type GeoKey = ((i64, i64, i64), (i64, i64, i64), (i64, i64, i64));

    fn make_key(front: (i64, i64, i64), back: (i64, i64, i64), mid: (i64, i64, i64)) -> GeoKey {
        if front <= back {
            (front, back, mid)
        } else {
            (back, front, mid)
        }
    }

    // Collect all edges with their geometric keys
    type EdgeEntry<C> = (EdgeID<C>, Edge<Point3, C>, bool);
    let mut edge_groups: std::collections::BTreeMap<GeoKey, Vec<EdgeEntry<C>>> =
        std::collections::BTreeMap::new();

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                let abs = edge.absolute_clone();
                let front_pt = abs.front().point();
                let back_pt = abs.back().point();
                let curve = abs.curve();
                let (t0, t1) = curve.range_tuple();
                let mid_pt = curve.subs((t0 + t1) * 0.5);

                let qf = quantize(front_pt);
                let qb = quantize(back_pt);
                let qm = quantize(mid_pt);
                let key = make_key(qf, qb, qm);
                let fwd = qf <= qb; // track direction relative to canonical key

                edge_groups
                    .entry(key)
                    .or_default()
                    .push((edge.id(), abs, fwd));
            }
        }
    }

    // Build replacement map: for each group with 2+ distinct edges, pick canonical
    let mut edge_to_canonical: std::collections::BTreeMap<EdgeID<C>, Edge<Point3, C>> =
        std::collections::BTreeMap::new();

    for group in edge_groups.values() {
        if group.len() <= 1 {
            continue;
        }

        // Deduplicate by edge ID within the group
        let mut seen_ids: std::collections::BTreeSet<EdgeID<C>> = std::collections::BTreeSet::new();
        let mut unique: Vec<&(EdgeID<C>, Edge<Point3, C>, bool)> = Vec::new();
        for entry in group {
            if seen_ids.insert(entry.0) {
                unique.push(entry);
            }
        }

        if unique.len() <= 1 {
            continue;
        }

        // Pick the first unique edge as canonical
        let canonical = &unique[0].1;
        for entry in &unique[1..] {
            edge_to_canonical.insert(entry.0, canonical.clone());
        }
    }

    if edge_to_canonical.is_empty() {
        return;
    }

    // Rebuild faces with canonical edges
    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let mut any_replaced = false;
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let edges: Vec<Edge<Point3, C>> = wire
                        .iter()
                        .map(|edge| {
                            if let Some(canonical) = edge_to_canonical.get(&edge.id()) {
                                any_replaced = true;
                                let abs = edge.absolute_clone();
                                // Use geometric position to determine direction since
                                // vertex IDs may differ (that's the whole point of
                                // position-based reweld)
                                let front_dist =
                                    (abs.front().point() - canonical.front().point()).magnitude();
                                let cross_dist =
                                    (abs.front().point() - canonical.back().point()).magnitude();
                                let same_dir = front_dist <= cross_dist;
                                if same_dir == edge.orientation() {
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
}

/// Split open edges at interior vertices from the shell.
///
/// After face division, a divided face may have split an edge into fragments,
/// but the adjacent face (undivided) still references the original unsplit edge.
/// This creates open edges: the original appears once, each fragment appears once.
///
/// This function finds open edges that have shell vertices on their interior
/// (between endpoints), splits the edges at those vertices, and updates the
/// faces. After splitting, the fragment vertex-pairs match those from the
/// divided face, enabling `weld_coincident_edges` to canonicalize them.
fn split_open_edges_at_interior_vertices<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell: &mut Shell<Point3, C, S>,
    tol: f64,
) -> bool {
    use std::collections::BTreeMap;
    type Vid = VertexID<Point3>;

    // Collect all open edges (face_count == 1)
    let mut edge_ref_count: BTreeMap<EdgeID<C>, usize> = BTreeMap::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                *edge_ref_count.entry(edge.id()).or_insert(0) += 1;
            }
        }
    }

    let open_eids: std::collections::BTreeSet<EdgeID<C>> = edge_ref_count
        .iter()
        .filter(|(_, &count)| count == 1)
        .map(|(id, _)| *id)
        .collect();

    if open_eids.is_empty() {
        return false;
    }

    // Collect all vertices in the shell with their positions
    let mut all_vertices: Vec<(Vid, Point3)> = Vec::new();
    let mut seen_vids: std::collections::BTreeSet<Vid> = std::collections::BTreeSet::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for v in wire.vertex_iter() {
                if seen_vids.insert(v.id()) {
                    all_vertices.push((v.id(), v.point()));
                }
            }
        }
    }

    // For each open edge, find interior vertices that lie on it
    // An interior vertex Vm is on edge E if:
    // 1. Vm is not at E's front or back vertex
    // 2. Vm lies within tol of E's curve
    // 3. Vm's parameter on E's curve is between E's parameter range
    struct SplitInfo<C2> {
        edge_id: EdgeID<C2>,
        vertex: Vertex<Point3>,
        param: f64,
    }

    let mut splits: Vec<SplitInfo<C>> = Vec::new();

    // Build vertex lookup for the shell
    let mut vertex_by_id: BTreeMap<Vid, Vertex<Point3>> = BTreeMap::new();
    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for v in wire.vertex_iter() {
                vertex_by_id.entry(v.id()).or_insert_with(|| v.clone());
            }
        }
    }

    for face in shell.iter() {
        for wire in face.absolute_boundaries().iter() {
            for edge in wire.iter() {
                if !open_eids.contains(&edge.id()) {
                    continue;
                }

                let abs = edge.absolute_clone();
                let fid = abs.front().id();
                let bid = abs.back().id();
                let curve = abs.curve();
                let (t0, t1) = curve.range_tuple();

                // Check each vertex in the shell
                for &(vid, pt) in &all_vertices {
                    if vid == fid || vid == bid {
                        continue;
                    }

                    // Check if this vertex lies on the edge's curve
                    if let Some(t) = curve.search_nearest_parameter(pt, None, 10) {
                        if t <= t0 + tol * 0.01 || t >= t1 - tol * 0.01 {
                            continue; // At endpoints, not interior
                        }
                        let curve_pt = curve.subs(t);
                        let dist = (curve_pt - pt).magnitude();
                        if dist < tol {
                            splits.push(SplitInfo {
                                edge_id: edge.id(),
                                vertex: vertex_by_id.get(&vid).unwrap().clone(),
                                param: t,
                            });
                        }
                    }
                }
            }
        }
    }

    if splits.is_empty() {
        return false;
    }

    #[cfg(debug_assertions)]
    eprintln!(
        "[split_propagation] found {} edges to split at interior vertices",
        splits.len(),
    );

    // Group splits by edge ID
    let mut splits_by_edge: BTreeMap<EdgeID<C>, Vec<(Vertex<Point3>, f64)>> = BTreeMap::new();
    for s in splits {
        splits_by_edge
            .entry(s.edge_id)
            .or_default()
            .push((s.vertex, s.param));
    }

    // Sort split points by parameter for each edge
    for points in splits_by_edge.values_mut() {
        points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        points.dedup_by(|a, b| (a.1 - b.1).abs() < tol * 0.01);
    }

    // Rebuild faces, splitting open edges at interior vertices
    let new_faces: Vec<Face<Point3, C, S>> = shell
        .iter()
        .map(|face| {
            let ori = face.orientation();
            let mut any_split = false;
            let new_wires: Vec<Wire<Point3, C>> = face
                .absolute_boundaries()
                .iter()
                .map(|wire| {
                    let mut edges: Vec<Edge<Point3, C>> = Vec::new();
                    for edge in wire.iter() {
                        if let Some(split_points) = splits_by_edge.get(&edge.id()) {
                            let abs = edge.absolute_clone();
                            let curve = abs.curve();
                            let (t0, t1) = curve.range_tuple();

                            let _ = (t0, t1); // used only for range check above

                            // Build fragment list: front → split1 → split2 → ... → back
                            let mut vertices: Vec<Vertex<Point3>> = vec![abs.front().clone()];
                            for (v, _t) in split_points {
                                vertices.push(v.clone());
                            }
                            vertices.push(abs.back().clone());

                            // Cut curve at each split parameter to get fragments
                            let split_params: Vec<f64> =
                                split_points.iter().map(|(_, t)| *t).collect();
                            let mut remaining = curve.clone();
                            let mut fragment_curves: Vec<C> = Vec::new();
                            for &t in &split_params {
                                let right_half = remaining.cut(t);
                                fragment_curves.push(remaining.clone());
                                remaining = right_half;
                            }
                            fragment_curves.push(remaining);

                            // Create fragment edges
                            let mut split_ok = true;
                            for i in 0..vertices.len() - 1 {
                                if let Ok(frag) = Edge::try_new(
                                    &vertices[i],
                                    &vertices[i + 1],
                                    fragment_curves[i].clone(),
                                ) {
                                    if edge.orientation() {
                                        edges.push(frag);
                                    } else {
                                        edges.push(frag.inverse());
                                    }
                                } else {
                                    split_ok = false;
                                    break;
                                }
                            }
                            if !split_ok {
                                edges.clear();
                                edges.push(edge.clone());
                            } else {
                                any_split = true;
                            }
                        } else {
                            edges.push(edge.clone());
                        }
                    }
                    edges.into()
                })
                .collect();

            if !any_split {
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
                    // Accept via new_unchecked
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
    true
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
        #[cfg(debug_assertions)]
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

    #[cfg(debug_assertions)]
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
            #[cfg(debug_assertions)]
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

        #[cfg(debug_assertions)]
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

    #[cfg(debug_assertions)]
    {
        let open = diagnose_open_edges(shell);
        eprintln!(
            "[finalize] after weld #1: {} faces, {} open edges",
            shell.len(),
            open.len(),
        );
    }

    // Try standard assembly
    let boundaries = shell.connected_components();
    if let Ok(solid) = Solid::try_new(boundaries) {
        return Ok(solid);
    }

    // If the shell isn't closed, try wider weld tolerances to close gaps.
    let wider_tols = [tols.tau_model * 2.0, tols.tau_model * 5.0];
    for (_i, &wider) in wider_tols.iter().enumerate() {
        weld_coincident_edges(shell, tols.tau_model, Some(wider));

        #[cfg(debug_assertions)]
        {
            let open = diagnose_open_edges(shell);
            eprintln!(
                "[finalize] after weld #{}: {} faces, {} open edges",
                _i + 2,
                shell.len(),
                open.len(),
            );
        }

        let boundaries = shell.connected_components();
        if let Ok(solid) = Solid::try_new(boundaries) {
            return Ok(solid);
        }
    }

    // Targeted open-edge re-weld: find specific unclosed edges and search
    // for geometrically-coincident twins to unify. This is more surgical
    // than blanket weld_coincident_edges retries — it targets only the
    // specific edges that prevent shell closure.
    {
        let open_edges = diagnose_open_edges(shell);
        #[cfg(debug_assertions)]
        if !open_edges.is_empty() {
            eprintln!(
                "[finalize] {} open edges before targeted reweld:",
                open_edges.len()
            );
            for oe in &open_edges {
                eprintln!(
                    "  f=({:.3},{:.3},{:.3}) b=({:.3},{:.3},{:.3}) mid=({:.3},{:.3},{:.3}) count={}",
                    oe.front.x, oe.front.y, oe.front.z,
                    oe.back.x, oe.back.y, oe.back.z,
                    oe.midpoint.x, oe.midpoint.y, oe.midpoint.z,
                    oe.face_count,
                );
            }
        }
        if !open_edges.is_empty() {
            targeted_open_edge_reweld(shell, tols.tau_model * 10.0);
            let boundaries = shell.connected_components();
            if let Ok(solid) = Solid::try_new(boundaries) {
                return Ok(solid);
            }
        }
    }

    // Position-based edge reweld: bypasses vertex ID matching entirely,
    // grouping edges by geometric position. Handles cases where Phase 0
    // couldn't unify vertices without creating degenerate edges, leaving
    // geometrically-coincident edges with different vertex identities.
    {
        let reweld_tols = [
            tols.tau_model * 2.0,
            tols.tau_model * 5.0,
            tols.tau_model * 10.0,
        ];
        for &rtol in &reweld_tols {
            position_based_edge_reweld(shell, rtol);
            let boundaries = shell.connected_components();
            if let Ok(solid) = Solid::try_new(boundaries) {
                return Ok(solid);
            }
        }
    }

    // Split edge propagation: when face division splits a boundary edge at an
    // intersection point, the adjacent face (not divided) still references the
    // original unsplit edge. Split these original edges at interior vertices
    // from the shell, then re-weld to canonicalize the fragments.
    if split_open_edges_at_interior_vertices(shell, tols.tau_model) {
        weld_coincident_edges(shell, tols.tau_model, None);
        let boundaries = shell.connected_components();
        if let Ok(solid) = Solid::try_new(boundaries) {
            return Ok(solid);
        }
        // Try wider tolerance
        weld_coincident_edges(shell, tols.tau_model, Some(tols.tau_model * 5.0));
        let boundaries = shell.connected_components();
        if let Ok(solid) = Solid::try_new(boundaries) {
            return Ok(solid);
        }

        #[cfg(debug_assertions)]
        {
            let open = diagnose_open_edges(shell);
            eprintln!(
                "[finalize] after split propagation + weld: {} open edges",
                open.len(),
            );
        }
    }

    // Post-weld Euler validation (debug diagnostic)
    #[cfg(debug_assertions)]
    {
        match validate_euler_characteristic(shell) {
            Ok(()) => {}
            Err((v, e, f, chi)) => {
                eprintln!(
                    "[finalize] Euler check: V={} E={} F={} chi={} (expected 2)",
                    v, e, f, chi,
                );
            }
        }
        let non_simple = find_non_simple_wires(shell);
        if !non_simple.is_empty() {
            eprintln!(
                "[finalize] {} non-simple wires: {:?}",
                non_simple.len(),
                non_simple,
            );
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
///
/// Wraps the entire operation in a deterministic ID context so that
/// internal `BTreeMap` orderings are run-independent.
pub fn and_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    with_det_context(|| {
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
    })
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
///
/// Wraps the entire operation in a deterministic ID context.
pub fn or_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    with_det_context(|| {
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
    })
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
///
/// Wraps the entire operation in a deterministic ID context.
///
/// For multi-shell solid0 (A), each shell is processed independently against
/// solid1 (B): `(A0 ∪ A1) \ B = (A0 \ B) ∪ (A1 \ B)`. This prevents
/// disjoint shells from being lost when they don't intersect each other.
pub fn difference_result_with_tol<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tols: &BooleanTolerance,
) -> std::result::Result<Solid<Point3, C, S>, BooleanStageError> {
    with_det_context(|| {
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
                let classified = classify_one_pair_of_shells_result_with_tol(
                    &diff_shell,
                    additional_b,
                    tols,
                    None,
                )?;
                diff_faces = classified.or0.into_iter().collect();
                for face in classified.and1.into_iter() {
                    diff_faces.push(face.inverse());
                }
            }
            all_diff_faces.extend(diff_faces);
        }

        let mut diff_shell: Shell<Point3, C, S> = all_diff_faces.into_iter().collect();
        finalize_boolean_shell(&mut diff_shell, tols)
    })
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
