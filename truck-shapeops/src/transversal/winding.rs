//! Generalized winding number classification (Jacobson et al. 2013).
//!
//! Computes the generalized winding number of a query point with respect to a
//! triangulated shell. Unlike ray-cast classification, winding numbers are
//! continuous and smooth — no rays, no perturbation, no majority voting.
//!
//! w(P) = (1/4π) Σ solid_angle(P, triangle_i)
//!
//! w > 0.5 → inside (And), w < 0.5 → outside (Or).

use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

/// Compute the solid angle subtended by triangle (a, b, c) at point p.
///
/// Uses the formula from Van Oosterom & Strackee (1983):
///   tan(Ω/2) = det([a-p, b-p, c-p]) / (|a-p|·|b-p|·|c-p|
///              + (a-p)·(b-p)·|c-p| + (b-p)·(c-p)·|a-p| + (c-p)·(a-p)·|b-p|)
///
/// This is numerically stable and gives signed solid angles (positive when
/// the triangle normal points toward p, negative when away).
#[inline]
fn solid_angle(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let pa = a - p;
    let pb = b - p;
    let pc = c - p;

    let la = (pa.x * pa.x + pa.y * pa.y + pa.z * pa.z).sqrt();
    let lb = (pb.x * pb.x + pb.y * pb.y + pb.z * pb.z).sqrt();
    let lc = (pc.x * pc.x + pc.y * pc.y + pc.z * pc.z).sqrt();

    // Degenerate: query point is AT a vertex
    if la < 1e-15 || lb < 1e-15 || lc < 1e-15 {
        return 0.0;
    }

    // Scalar triple product: det([pa, pb, pc])
    let numerator = pa.x * (pb.y * pc.z - pb.z * pc.y)
        + pa.y * (pb.z * pc.x - pb.x * pc.z)
        + pa.z * (pb.x * pc.y - pb.y * pc.x);

    // Dot products
    let ab = pa.x * pb.x + pa.y * pb.y + pa.z * pb.z;
    let bc = pb.x * pc.x + pb.y * pc.y + pb.z * pc.z;
    let ca = pc.x * pa.x + pc.y * pa.y + pc.z * pa.z;

    let denominator = la * lb * lc + ab * lc + bc * la + ca * lb;

    // atan2 gives the full angle range [-π, π]
    2.0 * numerator.atan2(denominator)
}

/// Compute the generalized winding number of point `p` with respect to the
/// triangulated `poly_shell`.
///
/// Returns a value near 1.0 for points inside a closed surface and near 0.0
/// for points outside. Boundary points give values near 0.5.
pub fn winding_number(
    p: Point3,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> f64 {
    let mut total = 0.0;

    for face in poly_shell.iter() {
        let mesh_opt = face.surface();
        let mesh = match mesh_opt.as_ref() {
            Some(m) => m,
            None => continue,
        };

        let positions = mesh.positions();
        let tri_faces = mesh.tri_faces();

        for tri in tri_faces {
            let a = positions[tri[0].pos];
            let b = positions[tri[1].pos];
            let c = positions[tri[2].pos];

            let sa = solid_angle(p, a, b, c);

            // Face orientation: if the face is inverted, flip the contribution.
            if face.orientation() {
                total += sa;
            } else {
                total -= sa;
            }
        }
    }

    total / (4.0 * std::f64::consts::PI)
}

/// Classify a point as inside (1) or outside (0) of a triangulated shell
/// using the generalized winding number.
///
/// Returns `Some(1)` for inside, `Some(0)` for outside.
/// Returns `None` only if the winding number is within the ambiguity band
/// [0.3, 0.7], indicating a degenerate/boundary query point.
pub fn winding_number_classify(
    p: Point3,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> Option<isize> {
    let w = winding_number(p, poly_shell);

    if w > 0.5 {
        Some(1) // inside
    } else if w < 0.5 {
        // Use a narrow ambiguity band: only return None for values very close to 0.5.
        // In practice, winding numbers are either near 0.0 or near 1.0 for clean geometry.
        if w > 0.3 {
            None // ambiguous — too close to boundary
        } else {
            Some(0) // outside
        }
    } else {
        None // exactly 0.5 — on the boundary
    }
}

/// Classify a face as inside or outside the opposing shell using winding numbers.
///
/// Samples the face centroid (perturbed by the face normal to avoid sampling
/// exactly on a surface). Same interface as `ray_cast_classify`:
/// returns `Some(1)` for inside (And) or `Some(0)` for outside (Or).
pub fn winding_classify_face<C, S>(
    face: &Face<Point3, C, S>,
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> Option<isize>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D,
    S: ParameterDivision2D,
{
    let verts: Vec<_> = face.boundaries()[0]
        .vertex_iter()
        .map(|v| v.point())
        .collect();

    if verts.is_empty() {
        return None;
    }

    let n = verts.len() as f64;
    let centroid_v = verts.iter().fold(Vector3::new(0.0, 0.0, 0.0), |a, &p| {
        a + (p - Point3::origin())
    }) / n;
    let centroid = Point3::origin() + centroid_v;

    // Compute a geometric normal from the vertex polygon for perturbation.
    let normal = if verts.len() >= 3 {
        let v0 = verts[1] - verts[0];
        let v1 = verts[2] - verts[0];
        let n = v0.cross(v1);
        let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
        if len > 1e-12 {
            Some(n / len)
        } else {
            None
        }
    } else {
        None
    };

    // Perturb centroid slightly along face normal to avoid querying exactly
    // on a shared face boundary.
    let small_offset = 1e-6;
    let test_point = if let Some(n) = normal {
        centroid + n * small_offset
    } else {
        // Fallback: use a small irrational offset
        centroid
            + Vector3::new(
                1.4142135623730951e-6,
                1.7320508075688772e-6,
                2.2360679774997896e-6,
            )
    };

    // Primary: winding number at perturbed centroid
    if let Some(c) = winding_number_classify(test_point, poly_shell) {
        return Some(c);
    }

    // Fallback: try opposite perturbation
    if let Some(n) = normal {
        let test_point_neg = centroid - n * small_offset;
        if let Some(c) = winding_number_classify(test_point_neg, poly_shell) {
            return Some(c);
        }
    }

    // Last resort: try each boundary vertex
    for &v in &verts {
        if let Some(c) = winding_number_classify(v, poly_shell) {
            return Some(c);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use truck_modeling::{builder, Solid as ModelingSolid};

    /// Create a unit cube [0,1]^3 and tessellate it.
    fn unit_cube_poly_shell() -> Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>> {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        let solid: ModelingSolid = builder::tsweep(&f, Vector3::unit_z());
        let shell = &solid.boundaries()[0];
        shell.triangulation(0.01)
    }

    #[test]
    fn winding_inside_cube() {
        let poly = unit_cube_poly_shell();
        let w = winding_number(Point3::new(0.5, 0.5, 0.5), &poly);
        assert!(
            (w - 1.0).abs() < 0.1,
            "Center of cube should have winding number ~1.0, got {}",
            w
        );
    }

    #[test]
    fn winding_outside_cube() {
        let poly = unit_cube_poly_shell();
        let w = winding_number(Point3::new(2.0, 2.0, 2.0), &poly);
        assert!(
            w.abs() < 0.1,
            "Point outside cube should have winding number ~0.0, got {}",
            w
        );
    }

    #[test]
    fn classify_inside_cube() {
        let poly = unit_cube_poly_shell();
        let result = winding_number_classify(Point3::new(0.5, 0.5, 0.5), &poly);
        assert_eq!(
            result,
            Some(1),
            "Center of cube should be classified inside"
        );
    }

    #[test]
    fn classify_outside_cube() {
        let poly = unit_cube_poly_shell();
        let result = winding_number_classify(Point3::new(2.0, 2.0, 2.0), &poly);
        assert_eq!(
            result,
            Some(0),
            "Point far from cube should be classified outside"
        );
    }

    #[test]
    fn classify_near_face_cube() {
        let poly = unit_cube_poly_shell();
        // Point just inside top face
        let w_in = winding_number(Point3::new(0.5, 0.5, 0.999), &poly);
        assert!(
            w_in > 0.5,
            "Point just inside should have w > 0.5, got {}",
            w_in
        );
        // Point just outside top face
        let w_out = winding_number(Point3::new(0.5, 0.5, 1.001), &poly);
        assert!(
            w_out < 0.5,
            "Point just outside should have w < 0.5, got {}",
            w_out
        );
    }

    #[test]
    fn classify_near_edge_cube() {
        let poly = unit_cube_poly_shell();
        // Point just inside near an edge
        let result = winding_number_classify(Point3::new(0.999, 0.5, 0.5), &poly);
        assert_eq!(result, Some(1), "Point inside near edge should be inside");
        // Point just outside near an edge
        let result = winding_number_classify(Point3::new(1.001, 0.5, 0.5), &poly);
        assert_eq!(result, Some(0), "Point outside near edge should be outside");
    }

    #[test]
    fn classify_near_corner_cube() {
        let poly = unit_cube_poly_shell();
        // Point just inside near a corner
        let result = winding_number_classify(Point3::new(0.01, 0.01, 0.01), &poly);
        assert_eq!(result, Some(1), "Point inside near corner should be inside");
        // Point just outside near a corner
        let result = winding_number_classify(Point3::new(-0.01, -0.01, -0.01), &poly);
        assert_eq!(
            result,
            Some(0),
            "Point outside near corner should be outside"
        );
    }

    #[test]
    fn winding_number_additivity() {
        // Winding number is 1 inside and 0 outside for any closed surface.
        let poly = unit_cube_poly_shell();
        let test_points = [
            (Point3::new(0.5, 0.5, 0.5), true),
            (Point3::new(0.1, 0.1, 0.1), true),
            (Point3::new(0.9, 0.9, 0.9), true),
            (Point3::new(-1.0, -1.0, -1.0), false),
            (Point3::new(5.0, 0.5, 0.5), false),
            (Point3::new(0.5, -2.0, 0.5), false),
        ];
        for (pt, expected_inside) in &test_points {
            let w = winding_number(*pt, &poly);
            if *expected_inside {
                assert!(w > 0.5, "Point {:?} should be inside (w={:.4})", pt, w);
            } else {
                assert!(w < 0.5, "Point {:?} should be outside (w={:.4})", pt, w);
            }
        }
    }
}
