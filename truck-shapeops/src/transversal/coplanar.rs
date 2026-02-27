use truck_base::cgmath64::*;
use truck_topology::*;

use super::integrate::ShapeOpsSurface;
use super::robust_classify::signed_plane_distance;

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
/// Returns `Some(action)` if the face is coplanar with any face in the other
/// shell: overlapping faces get Remove/And/Or based on normal sense, while
/// non-overlapping coplanar faces get `Or` (they lie on the other solid's
/// boundary plane but outside its footprint, so they are outside the solid).
///
/// Returns `None` if the face is not coplanar with any face in the other shell,
/// meaning ray-cast classification should be used instead.
///
/// Note: C1 and C2 may be different curve types (e.g., Alternative vs raw curve)
/// because face fragments use wrapped curve types while the other shell uses
/// original curves.
pub fn classify_coplanar_fragment<C1, C2, S: ShapeOpsSurface>(
    face: &Face<Point3, C1, S>,
    other_shell: &Shell<Point3, C2, S>,
    is_shell0: bool,
    tol: f64,
) -> Option<CoplanarAction> {
    let info = face_sample_info(face)?;
    // Use a representative interior point of the face for the overlap test.
    // The first boundary vertex is often on the splitting line (exactly on
    // the other face's boundary), making point_in_polygon ambiguous.
    // For faces with holes, face_interior_point avoids the hole region.
    let centroid = face_interior_point(face);

    let mut found_same_sense_coplanar = false;

    for other_face in other_shell.iter() {
        let other_info = match face_sample_info(other_face) {
            Some(i) => i,
            None => continue,
        };

        let same_sense = match check_coplanar(&info, &other_info, tol) {
            Some(s) => s,
            None => continue,
        };

        // Check overlap: is the centroid inside the other face's boundary?
        // Fall back to the original sample point if centroid is unavailable.
        let test_point = centroid.unwrap_or(info.point);
        if point_in_face(test_point, other_face, tol) {
            return Some(if !same_sense {
                CoplanarAction::Remove
            } else if is_shell0 {
                CoplanarAction::And
            } else {
                CoplanarAction::Or
            });
        }

        // Only track same-sense coplanar faces for the non-overlapping shortcut.
        // For anti-sense (anti-parallel normals), the non-overlapping region
        // could be inside the other solid (e.g., ring face around a through-hole
        // is inside the inverted cylinder). We must fall through to ray-cast
        // for anti-sense cases.
        if same_sense {
            found_same_sense_coplanar = true;
        }
    }

    // If we found same-sense coplanar faces but the centroid is outside all of
    // them, the face lies on the other solid's boundary plane but outside its
    // footprint. For same-sense faces (parallel normals), this means the face
    // is outside the other solid — classify as Or. This avoids falling through
    // to ray_cast_classify, which gives wrong results when the test point is
    // exactly on the other shell's boundary (perturbation pushes it barely
    // inside/outside non-deterministically).
    if found_same_sense_coplanar {
        Some(CoplanarAction::Or)
    } else {
        None
    }
}

/// Compute a representative interior point of a face.
///
/// For faces with holes (inner boundaries), the centroid of the outer boundary
/// may fall inside a hole. In that case, fall back to the midpoint of the
/// first outer boundary edge, which is always on the face boundary.
fn face_interior_point<C, S>(face: &Face<Point3, C, S>) -> Option<Point3> {
    let boundaries = face.boundaries();
    let outer_wire = boundaries.first()?;
    let verts: Vec<Point3> = outer_wire.vertex_iter().map(|v| v.point()).collect();
    if verts.is_empty() {
        return None;
    }
    let n = verts.len() as f64;
    let sum = verts.iter().fold(Vector3::new(0.0, 0.0, 0.0), |acc, &p| {
        acc + (p - Point3::origin())
    });
    let centroid = Point3::origin() + sum / n;

    // If there are inner boundaries (holes), check that the centroid
    // is not inside any of them. If it is, use the midpoint of the
    // first outer edge instead.
    if boundaries.len() > 1 {
        // Compute a face normal from the outer boundary for 2D projection
        let normal = {
            let mut nx = 0.0f64;
            let mut ny = 0.0f64;
            let mut nz = 0.0f64;
            for i in 0..verts.len() {
                let j = (i + 1) % verts.len();
                let vi = verts[i];
                let vj = verts[j];
                nx += (vi.y - vj.y) * (vi.z + vj.z);
                ny += (vi.z - vj.z) * (vi.x + vj.x);
                nz += (vi.x - vj.x) * (vi.y + vj.y);
            }
            let mag = (nx * nx + ny * ny + nz * nz).sqrt();
            if mag < 1e-15 {
                Vector3::unit_z()
            } else {
                Vector3::new(nx / mag, ny / mag, nz / mag)
            }
        };
        let (u_axis, v_axis) = make_tangent_basis(normal);

        for inner_wire in boundaries.iter().skip(1) {
            let inner_verts: Vec<Point3> = inner_wire.vertex_iter().map(|v| v.point()).collect();
            if inner_verts.len() >= 3 {
                let poly: Vec<[f64; 2]> = inner_verts
                    .iter()
                    .map(|&p| project_to_2d(p, centroid, u_axis, v_axis))
                    .collect();
                let test = project_to_2d(centroid, centroid, u_axis, v_axis);
                if point_in_polygon(test, &poly) {
                    // Centroid is in the hole — use outer edge midpoint instead
                    if verts.len() >= 2 {
                        let p = verts[0];
                        let q = verts[1];
                        return Some(Point3::new(
                            (p.x + q.x) / 2.0,
                            (p.y + q.y) / 2.0,
                            (p.z + q.z) / 2.0,
                        ));
                    }
                    break;
                }
            }
        }
    }
    Some(centroid)
}

/// Sample information from a face: a boundary point and the effective normal.
pub(crate) struct FaceSampleInfo {
    pub(crate) point: Point3,
    pub(crate) normal: Vector3,
}

/// Extract a sample point from the face boundary and compute the effective normal.
///
/// Iterates over boundary vertices until `search_parameter` succeeds. This is
/// more robust than using only the first vertex, because some vertices (e.g. on
/// non-XY faces) may land in degenerate positions where the surface
/// parameterisation cannot converge.
pub(crate) fn face_sample_info<C, S: ShapeOpsSurface>(
    face: &Face<Point3, C, S>,
) -> Option<FaceSampleInfo> {
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
            return Some(FaceSampleInfo {
                point: pt,
                normal: n,
            });
        }
    }
    None
}

/// Check whether two faces are coplanar. Returns `Some(same_sense)` if they are,
/// `None` if not.
///
/// Uses a two-tier approach:
/// 1. **Exact test**: Constructs a reference plane from info0's point and normal,
///    tests info1's point via `robust_orient3d`-based `signed_plane_distance`.
///    If the distance is exactly 0.0, the faces are exactly coplanar (no tolerance
///    needed for the distance check).
/// 2. **Tolerance test**: If not exactly coplanar, checks if the normalized
///    perpendicular distance is within `tol`.
///
/// Angular threshold: `(1 - |dot|) > tol²` approximates `angle > tol` radians.
/// For `tol = 0.025`: threshold = 0.000625 rad ≈ 0.036°.
/// For `tol = 0.05`: threshold = 0.0025 rad ≈ 0.14°.
pub(crate) fn check_coplanar(
    info0: &FaceSampleInfo,
    info1: &FaceSampleInfo,
    tol: f64,
) -> Option<bool> {
    let dot = info0.normal.dot(info1.normal);
    // Normals must be (anti-)parallel: angle between them < tol radians.
    // Using the small-angle approximation: 1 - cos(θ) ≈ θ²/2, so
    // (1 - |dot|) > tol * tol means angle > ~sqrt(2) * tol.
    if (1.0 - dot.abs()) > tol * tol {
        return None;
    }

    // Construct a reference plane from info0's point and normal.
    // We need 3 non-collinear points; use the tangent basis to generate them.
    let (u_axis, v_axis) = make_tangent_basis(info0.normal);
    let p0 = info0.point;
    let p1 = p0 + u_axis;
    let p2 = p0 + v_axis;
    let a = [p0.x, p0.y, p0.z];
    let b = [p1.x, p1.y, p1.z];
    let c = [p2.x, p2.y, p2.z];
    let d = [info1.point.x, info1.point.y, info1.point.z];

    // Use signed_plane_distance for exact + normalized distance check
    match signed_plane_distance(a, b, c, d) {
        Some(0.0) => {
            // Exactly coplanar — no tolerance needed
        }
        Some(dist) if dist.abs() < tol => {
            // Within tolerance — near-coplanar
        }
        _ => {
            // Too far from plane or degenerate reference triangle
            return None;
        }
    }

    Some(dot > 0.0)
}

/// Test whether a 3D point lies inside a face's boundary using a 2D projection.
///
/// Uses `tol` for a boundary proximity guard: if the test point is within
/// `tol * 0.01` distance of any polygon edge in 2D, returns `false` (conservative)
/// to avoid misclassifying boundary-adjacent points as definitively inside.
pub(crate) fn point_in_face<C, S: ShapeOpsSurface>(
    pt: Point3,
    face: &Face<Point3, C, S>,
    tol: f64,
) -> bool {
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

    // Boundary proximity guard: if the test point is within a small distance of
    // any polygon edge, the ray-casting test can give ambiguous results.
    // Use tol * 0.01 as the proximity threshold — tight enough to not reject
    // clearly-interior points, but wide enough to catch boundary-adjacent ones.
    let boundary_tol = tol * 0.01;
    if boundary_tol > 0.0 && point_near_polygon_boundary(test_2d, &polygon, boundary_tol) {
        return false;
    }

    point_in_polygon(test_2d, &polygon)
}

/// Check if a 2D point is within `tol` distance of any edge of the polygon.
fn point_near_polygon_boundary(pt: [f64; 2], polygon: &[[f64; 2]], tol: f64) -> bool {
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, ay) = (polygon[i][0], polygon[i][1]);
        let (bx, by) = (polygon[j][0], polygon[j][1]);
        let (ex, ey) = (bx - ax, by - ay);
        let len_sq = ex * ex + ey * ey;
        if len_sq < 1e-30 {
            continue;
        }
        let t = ((pt[0] - ax) * ex + (pt[1] - ay) * ey) / len_sq;
        let t = t.clamp(0.0, 1.0);
        let cx = ax + t * ex;
        let cy = ay + t * ey;
        let dist = ((pt[0] - cx).powi(2) + (pt[1] - cy).powi(2)).sqrt();
        if dist < tol {
            return true;
        }
    }
    false
}

/// Build an orthonormal tangent basis from a normal vector.
pub(crate) fn make_tangent_basis(n: Vector3) -> (Vector3, Vector3) {
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
pub(crate) fn project_to_2d(
    pt: Point3,
    origin: Point3,
    u_axis: Vector3,
    v_axis: Vector3,
) -> [f64; 2] {
    let d = pt - origin;
    [d.dot(u_axis), d.dot(v_axis)]
}

/// Ray-casting point-in-polygon test in 2D using robust predicates.
///
/// Uses `robust_orient2d` for the crossing side test, which eliminates
/// floating-point misclassification for points near polygon edges.
pub(crate) fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    use super::robust_classify::robust_orient2d;
    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;
    for i in 0..n {
        let (yi, yj) = (polygon[i][1], polygon[j][1]);
        // Only test edges that straddle the horizontal ray from `point`
        if (yi > point[1]) != (yj > point[1]) {
            // Use robust orient2d to determine which side of the edge the
            // point lies on. This replaces the naive floating-point
            // expression `point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi`
            // with an exact orientation test.
            let orient = robust_orient2d(polygon[j], polygon[i], point);
            // The sign convention: if the edge goes upward (yi < yj),
            // point is to the left (inside) when orient > 0.
            // If the edge goes downward (yi > yj), point is inside when orient < 0.
            if yi < yj {
                if orient > 0.0 {
                    inside = !inside;
                }
            } else if orient < 0.0 {
                inside = !inside;
            }
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
