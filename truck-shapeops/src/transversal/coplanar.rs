use truck_base::cgmath64::*;
use truck_topology::*;

use super::integrate::{ShapeOpsCurve, ShapeOpsSurface};

/// What to do with a coplanar overlap fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoplanarAction {
    /// Remove the face (anti-sense overlap — interior face between solids).
    Remove,
    /// Face belongs in the AND result.
    And,
    /// Face belongs in the OR result.
    Or,
}

/// Classify a face fragment that may be coplanar with faces in the other shell.
///
/// Returns `Some(action)` if the face is a coplanar overlap, `None` otherwise.
pub fn classify_coplanar_fragment<C: ShapeOpsCurve<S>, S: ShapeOpsSurface>(
    face: &Face<Point3, C, S>,
    other_shell: &Shell<Point3, C, S>,
    is_shell0: bool,
    tol: f64,
) -> Option<CoplanarAction> {
    let info = face_sample_info(face)?;

    for other_face in other_shell.iter() {
        let other_info = match face_sample_info(other_face) {
            Some(i) => i,
            None => continue,
        };

        let same_sense = match check_coplanar(&info, &other_info, tol) {
            Some(s) => s,
            None => continue,
        };

        // Check overlap: is the sample point inside the other face's boundary?
        if point_in_face(info.point, other_face, tol) {
            return Some(if !same_sense {
                CoplanarAction::Remove
            } else if is_shell0 {
                CoplanarAction::And
            } else {
                CoplanarAction::Or
            });
        }
    }

    None
}

/// Sample information from a face: a boundary point and the effective normal.
struct FaceSampleInfo {
    point: Point3,
    normal: Vector3,
}

/// Extract a sample point from the face boundary and compute the effective normal.
///
/// Iterates over boundary vertices until `search_parameter` succeeds. This is
/// more robust than using only the first vertex, because some vertices (e.g. on
/// non-XY faces) may land in degenerate positions where the surface
/// parameterisation cannot converge.
fn face_sample_info<C, S>(face: &Face<Point3, C, S>) -> Option<FaceSampleInfo>
where
    C: ShapeOpsCurve<S>,
    S: ShapeOpsSurface,
{
    let boundaries = face.boundaries();
    let wire = boundaries.first()?;
    let surface = face.surface();
    for vertex in wire.vertex_iter() {
        let pt = vertex.point();
        if let Some((u, v)) = surface.search_parameter(pt, None, 100) {
            let n = surface.normal(u, v);
            // Skip degenerate normals (near-zero magnitude)
            if n.x * n.x + n.y * n.y + n.z * n.z < 1e-20 {
                continue;
            }
            let mut n = n;
            if !face.orientation() {
                n = -n;
            }
            return Some(FaceSampleInfo { point: pt, normal: n });
        }
    }
    None
}

/// Check whether two faces are coplanar. Returns `Some(same_sense)` if they are,
/// `None` if not.
fn check_coplanar(info0: &FaceSampleInfo, info1: &FaceSampleInfo, tol: f64) -> Option<bool> {
    let dot = info0.normal.dot(info1.normal);
    // Normals must be (anti-)parallel
    if dot.abs() <= 1.0 - tol {
        return None;
    }
    // Points must lie on the same plane (use tol directly, not sqrt(tol),
    // so that faces separated by eps > tol are not falsely detected as coplanar)
    let d = info0.point - info1.point;
    if d.dot(info0.normal).abs() >= tol {
        return None;
    }
    Some(dot > 0.0)
}

/// Test whether a 3D point lies inside a face's boundary using a 2D projection.
fn point_in_face<C, S>(pt: Point3, face: &Face<Point3, C, S>, _tol: f64) -> bool
where
    C: ShapeOpsCurve<S>,
    S: ShapeOpsSurface,
{
    // Compute face normal to build a local 2D coordinate system
    let surface = face.surface();
    let (u, v) = match surface.search_parameter(pt, None, 100) {
        Some(uv) => uv,
        None => return false,
    };
    let n = surface.normal(u, v);

    // Build orthonormal 2D basis on the face plane
    let (u_axis, v_axis) = make_tangent_basis(n);

    // Project the test point
    let test_2d = project_to_2d(pt, pt, u_axis, v_axis);

    // Project the face boundary vertices into 2D
    let boundaries = face.boundaries();
    let outer_wire = match boundaries.first() {
        Some(w) => w,
        None => return false,
    };

    let polygon: Vec<[f64; 2]> = outer_wire
        .vertex_iter()
        .map(|v| project_to_2d(v.point(), pt, u_axis, v_axis))
        .collect();

    if polygon.len() < 3 {
        return false;
    }

    point_in_polygon(test_2d, &polygon)
}

/// Build an orthonormal tangent basis from a normal vector.
fn make_tangent_basis(n: Vector3) -> (Vector3, Vector3) {
    // Pick the axis least aligned with n to avoid degeneracy
    let seed = if n.x.abs() < 0.9 {
        Vector3::unit_x()
    } else {
        Vector3::unit_y()
    };
    let u_axis = n.cross(seed).normalize();
    let v_axis = n.cross(u_axis).normalize();
    (u_axis, v_axis)
}

/// Project a 3D point into 2D coordinates relative to an origin and tangent basis.
fn project_to_2d(
    pt: Point3,
    origin: Point3,
    u_axis: Vector3,
    v_axis: Vector3,
) -> [f64; 2] {
    let d = pt - origin;
    [d.dot(u_axis), d.dot(v_axis)]
}

/// Ray-casting point-in-polygon test in 2D.
fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);
        if ((yi > point[1]) != (yj > point[1]))
            && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use truck_base::tolerance::TOLERANCE;

    #[test]
    fn test_point_in_polygon_inside() {
        let square = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon([0.5, 0.5], &square));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let square = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(!point_in_polygon([2.0, 0.5], &square));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let tri = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]];
        assert!(point_in_polygon([2.0, 1.0], &tri));
        assert!(!point_in_polygon([0.0, 3.0], &tri));
    }

    #[test]
    fn test_make_tangent_basis_orthonormal() {
        let n = Vector3::unit_z();
        let (u, v) = make_tangent_basis(n);
        assert!((u.dot(v)).abs() < TOLERANCE);
        assert!((u.dot(n)).abs() < TOLERANCE);
        assert!((v.dot(n)).abs() < TOLERANCE);
        assert!((u.magnitude() - 1.0).abs() < TOLERANCE);
        assert!((v.magnitude() - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_make_tangent_basis_x_aligned() {
        let n = Vector3::unit_x();
        let (u, v) = make_tangent_basis(n);
        assert!((u.dot(v)).abs() < TOLERANCE);
        assert!((u.dot(n)).abs() < TOLERANCE);
        assert!((v.dot(n)).abs() < TOLERANCE);
    }

    #[test]
    fn test_check_coplanar_same_plane() {
        let info0 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::unit_z(),
        };
        let info1 = FaceSampleInfo {
            point: Point3::new(1.0, 1.0, 0.0),
            normal: Vector3::unit_z(),
        };
        let result = check_coplanar(&info0, &info1, TOLERANCE);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn test_check_coplanar_anti_parallel() {
        let info0 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::unit_z(),
        };
        let info1 = FaceSampleInfo {
            point: Point3::new(1.0, 1.0, 0.0),
            normal: -Vector3::unit_z(),
        };
        let result = check_coplanar(&info0, &info1, TOLERANCE);
        assert_eq!(result, Some(false));
    }

    #[test]
    fn test_check_coplanar_different_planes() {
        let info0 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::unit_z(),
        };
        let info1 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 1.0),
            normal: Vector3::unit_z(),
        };
        let result = check_coplanar(&info0, &info1, TOLERANCE);
        assert_eq!(result, None);
    }

    #[test]
    fn test_check_coplanar_not_parallel() {
        let info0 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::unit_z(),
        };
        let info1 = FaceSampleInfo {
            point: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::unit_x(),
        };
        let result = check_coplanar(&info0, &info1, TOLERANCE);
        assert_eq!(result, None);
    }
}
