use crate::alternative::Alternative;

use super::*;
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

/// Only solids consisting of faces whose surface is implemented this trait can be used for set operations.
pub trait ShapeOpsSurface:
    ParametricSurface3D
    + ParameterDivision2D
    + SearchParameter<D2, Point = Point3>
    + SearchNearestParameter<D2, Point = Point3>
    + Invertible
    + Send
    + Sync {
}
impl<S> ShapeOpsSurface for S where S: ParametricSurface3D
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
    + Sync {
}
impl<C, S: ShapeOpsSurface> ShapeOpsCurve<S> for C where C: ParametricCurve3D
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

/// Ray-cast from a point against a triangulated shell. Returns the signed
/// crossing count, or None if the ray grazes an edge.
pub(crate) fn try_ray_cast(
    pt: Point3,
    dir: Vector3,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> Option<isize> {
    poly_shell.iter().try_fold(0isize, |count, face| {
        let poly = face.surface()?;
        Some(count + poly.signed_crossing_faces(pt, dir))
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
                let bsp = BSplineCurve::quadratic_approximation(ic, ic.range_tuple(), tol, 100)?;
                Some(
                    IntersectionCurve::new(ic.surface0().clone(), ic.surface1().clone(), bsp)
                        .into(),
                )
            }
        },
        |s| Some(s.clone()),
    )
}

fn process_one_pair_of_shells<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    shell0: &Shell<Point3, C, S>,
    shell1: &Shell<Point3, C, S>,
    tol: f64,
) -> Option<[Shell<Point3, C, S>; 2]> {
    nonpositive_tolerance!(tol);
    let poly_shell0 = shell0.triangulation(tol);
    let poly_shell1 = shell1.triangulation(tol);
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
    } = loops_store::create_loops_stores(&altshell0, &poly_shell0, &altshell1, &poly_shell1, tol)?;
    let (mut cls0, coplanar_fids0) =
        divide_face::divide_faces_with_coplanar(&altshell0, &loops_store0, tol, &coplanar_faces0)?;
    cls0.integrate_by_component();
    let (mut cls1, coplanar_fids1) =
        divide_face::divide_faces_with_coplanar(&altshell1, &loops_store1, tol, &coplanar_faces1)?;
    cls1.integrate_by_component();
    // Reset overlapping coplanar fragments to Unknown for re-classification.
    // Pass the original shells (not altshells) since coplanar classification
    // only needs surface operations.
    cls0.reset_overlapping_coplanar(&coplanar_fids0, shell1, true, tol);
    cls1.reset_overlapping_coplanar(&coplanar_fids1, shell0, false, tol);
    let [mut and0, mut or0, unknown0] = cls0.and_or_unknown();
    unknown0.into_iter().try_for_each(|face| {
        // Try coplanar classification first (against original shell).
        if let Some(action) = coplanar::classify_coplanar_fragment(&face, shell1, true, tol) {
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
    })?;
    let [mut and1, mut or1, unknown1] = cls1.and_or_unknown();
    unknown1.into_iter().try_for_each(|face| {
        // Try coplanar classification first (against original shell).
        if let Some(action) = coplanar::classify_coplanar_fragment(&face, shell0, false, tol) {
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
    })?;
    and0.append(&mut and1);
    or0.append(&mut or1);
    Some([
        altshell_to_shell(&and0, tol)?,
        altshell_to_shell(&or0, tol)?,
    ])
}

/// AND operation between two solids.
pub fn and<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> Option<Solid<Point3, C, S>> {
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [mut and_shell, _] = process_one_pair_of_shells(shell0, shell1, tol)?;
    for shell in iter0 {
        let [res, _] = process_one_pair_of_shells(&and_shell, shell, tol)?;
        and_shell = res;
    }
    for shell in iter1 {
        let [res, _] = process_one_pair_of_shells(&and_shell, shell, tol)?;
        and_shell = res;
    }
    let boundaries = and_shell.connected_components();
    Solid::try_new(boundaries).ok()
}

/// OR operation between two solids.
pub fn or<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    solid0: &Solid<Point3, C, S>,
    solid1: &Solid<Point3, C, S>,
    tol: f64,
) -> Option<Solid<Point3, C, S>> {
    let mut iter0 = solid0.boundaries().iter();
    let mut iter1 = solid1.boundaries().iter();
    let shell0 = iter0.next().unwrap();
    let shell1 = iter1.next().unwrap();
    let [_, mut or_shell] = process_one_pair_of_shells(shell0, shell1, tol)?;
    for shell in iter0 {
        let [_, res] = process_one_pair_of_shells(&or_shell, shell, tol)?;
        or_shell = res;
    }
    for shell in iter1 {
        let [_, res] = process_one_pair_of_shells(&or_shell, shell, tol)?;
        or_shell = res;
    }
    let boundaries = or_shell.connected_components();
    Solid::try_new(boundaries).ok()
}

#[cfg(test)]
mod tests;
