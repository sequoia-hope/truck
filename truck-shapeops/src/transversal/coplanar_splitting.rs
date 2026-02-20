//! Coplanar face detection for boolean operations.
//!
//! When `intersection_curves` returns `None` for coplanar face pairs (because
//! `double_projection` diverges when normals are parallel), this module provides
//! `check_coplanar_faces` to detect such pairs.

use truck_base::cgmath64::*;
use truck_topology::*;

// ─── Face / coplanar helpers ───────────────────────────────────────────

/// Get the outer-boundary vertices of a face as 3D points.
fn face_boundary_vertices<C, S>(face: &Face<Point3, C, S>) -> Vec<Point3> {
    face.absolute_boundaries()
        .first()
        .map(|w| w.vertex_iter().map(|v| v.point()).collect())
        .unwrap_or_default()
}

/// Compute the face normal from boundary vertices using Newell's method.
fn face_normal_from_vertices(verts: &[Point3]) -> Option<Vector3> {
    if verts.len() < 3 {
        return None;
    }
    let mut n = Vector3::new(0.0, 0.0, 0.0);
    for i in 0..verts.len() {
        let j = (i + 1) % verts.len();
        let vi = verts[i];
        let vj = verts[j];
        n.x += (vi.y - vj.y) * (vi.z + vj.z);
        n.y += (vi.z - vj.z) * (vi.x + vj.x);
        n.z += (vi.x - vj.x) * (vi.y + vj.y);
    }
    let mag = n.magnitude();
    if mag < 1e-15 {
        None
    } else {
        Some(n / mag)
    }
}

/// Check if two faces are coplanar AND have overlapping area.
///
/// Returns `Some(same_sense)` if the faces are on the same plane and their
/// 2D bounding boxes overlap (with tolerance). Returns `None` if the faces
/// are not coplanar or if they are on the same plane but have no area overlap
/// (e.g., they only share an edge or point).
pub(crate) fn check_coplanar_faces<C, S>(
    face0: &Face<Point3, C, S>,
    face1: &Face<Point3, C, S>,
    tol: f64,
) -> Option<bool> {
    let verts0 = face_boundary_vertices(face0);
    let verts1 = face_boundary_vertices(face1);

    let mut n0 = face_normal_from_vertices(&verts0)?;
    let mut n1 = face_normal_from_vertices(&verts1)?;

    if !face0.orientation() {
        n0 = -n0;
    }
    if !face1.orientation() {
        n1 = -n1;
    }

    let dot = n0.dot(n1);
    if dot.abs() <= 1.0 - tol {
        return None;
    }

    let d = verts0[0] - verts1[0];
    if d.dot(n0).abs() >= tol {
        return None;
    }

    let same_sense = dot > 0.0;

    // Check that the 2D bounding boxes overlap (not just touching at a line/point).
    // Project both faces onto the shared plane and compare bounding boxes.
    // Without this check, faces on the same plane but with disjoint footprints
    // (e.g., two y=0 faces at different z-ranges) would be falsely flagged as
    // coplanar, causing their classification to be reset to Unknown and then
    // re-classified by ray-casting, which can give wrong results for
    // boundary-adjacent points.
    if verts0.len() >= 3 && verts1.len() >= 3 {
        use super::coplanar::{make_tangent_basis, project_to_2d};
        let (u_axis, v_axis) = make_tangent_basis(n0);
        let origin = verts0[0];

        // Compute 2D bounding boxes
        let mut min0 = [f64::INFINITY; 2];
        let mut max0 = [f64::NEG_INFINITY; 2];
        for &v in &verts0 {
            let p = project_to_2d(v, origin, u_axis, v_axis);
            min0[0] = min0[0].min(p[0]);
            min0[1] = min0[1].min(p[1]);
            max0[0] = max0[0].max(p[0]);
            max0[1] = max0[1].max(p[1]);
        }

        let mut min1 = [f64::INFINITY; 2];
        let mut max1 = [f64::NEG_INFINITY; 2];
        for &v in &verts1 {
            let p = project_to_2d(v, origin, u_axis, v_axis);
            min1[0] = min1[0].min(p[0]);
            min1[1] = min1[1].min(p[1]);
            max1[0] = max1[0].max(p[0]);
            max1[1] = max1[1].max(p[1]);
        }

        // Require overlap area > tol (not just touching at a line/point).
        // Use tol as the minimum overlap in each axis.
        let overlap_x = (max0[0].min(max1[0]) - min0[0].max(min1[0])).max(0.0);
        let overlap_y = (max0[1].min(max1[1]) - min0[1].max(min1[1])).max(0.0);
        if overlap_x < tol || overlap_y < tol {
            return None;
        }

        // After AABB passes, verify actual polygon overlap via vertex containment
        // or edge intersection. Handles L-shaped faces with overlapping AABBs
        // but zero actual overlap area.
        let proj0: Vec<[f64; 2]> = verts0
            .iter()
            .map(|&v| project_to_2d(v, origin, u_axis, v_axis))
            .collect();
        let proj1: Vec<[f64; 2]> = verts1
            .iter()
            .map(|&v| project_to_2d(v, origin, u_axis, v_axis))
            .collect();

        let any_v1_in_0 = proj1.iter().any(|pt| point_in_polygon_2d(*pt, &proj0));
        let any_v0_in_1 = proj0.iter().any(|pt| point_in_polygon_2d(*pt, &proj1));
        let edges_cross = edges_intersect_2d(&proj0, &proj1, tol);

        if !any_v1_in_0 && !any_v0_in_1 && !edges_cross {
            return None;
        }
    }

    Some(same_sense)
}

/// Ray-casting point-in-polygon test in 2D (non-test version for overlap detection).
fn point_in_polygon_2d(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
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

/// Check if any edge of polygon A intersects any edge of polygon B in 2D.
/// Uses segment-segment intersection (O(n*m), fine for small face polygons).
fn edges_intersect_2d(poly_a: &[[f64; 2]], poly_b: &[[f64; 2]], tol: f64) -> bool {
    for i in 0..poly_a.len() {
        let j = (i + 1) % poly_a.len();
        let (ax, ay) = (poly_a[i][0], poly_a[i][1]);
        let (bx, by) = (poly_a[j][0], poly_a[j][1]);
        for k in 0..poly_b.len() {
            let l = (k + 1) % poly_b.len();
            let (cx, cy) = (poly_b[k][0], poly_b[k][1]);
            let (dx, dy) = (poly_b[l][0], poly_b[l][1]);
            // Segment AB vs segment CD
            let denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
            if denom.abs() < 1e-15 {
                continue; // Parallel
            }
            let t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom;
            let u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom;
            // Interior intersection (not at endpoints — endpoints mean shared edge, not overlap)
            if t > tol && t < 1.0 - tol && u > tol && u < 1.0 - tol {
                return true;
            }
        }
    }
    false
}

/// Get the outer-boundary vertices and outward normal of a face.
///
/// Returns `(vertices, normal)` where the normal accounts for face orientation.
pub(crate) fn face_boundary_info<C, S>(
    face: &Face<Point3, C, S>,
) -> Option<(Vec<Point3>, Vector3)> {
    let verts = face_boundary_vertices(face);
    let mut n = face_normal_from_vertices(&verts)?;
    if !face.orientation() {
        n = -n;
    }
    Some((verts, n))
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newell_normal() {
        let verts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let n = face_normal_from_vertices(&verts).unwrap();
        assert!(
            (n.z.abs() - 1.0) < 1e-10,
            "Normal should be ±Z, got {:?}",
            n
        );
    }

    // ── Polygon overlap edge-case tests ──

    #[test]
    fn test_point_in_polygon_2d() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon_2d([0.5, 0.5], &sq));
        assert!(!point_in_polygon_2d([2.0, 0.5], &sq));
    }

    #[test]
    fn test_edges_intersect_2d_crossing() {
        let a = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!(edges_intersect_2d(&a, &b, 0.001));
    }

    #[test]
    fn test_edges_intersect_2d_no_overlap() {
        let a = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let b = [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]];
        assert!(!edges_intersect_2d(&a, &b, 0.001));
    }

    /// U-shape and rectangle on the same plane with overlapping AABBs but
    /// zero actual polygon overlap area. The polygon overlap test should
    /// reject this pair even though the AABB test would accept it.
    #[test]
    fn test_u_shape_vs_rect_overlapping_aabb_no_polygon_overlap() {
        let u_shape = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 1.0],
            [0.5, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [0.0, 3.0],
        ];
        let rect = [[0.6, 1.1], [1.5, 1.1], [1.5, 1.9], [0.6, 1.9]];

        let any_r_in_u = rect.iter().any(|pt| point_in_polygon_2d(*pt, &u_shape));
        let any_u_in_r = u_shape.iter().any(|pt| point_in_polygon_2d(*pt, &rect));
        let edges_cross = edges_intersect_2d(&u_shape, &rect, 0.001);

        assert!(!any_r_in_u, "No rect vertex should be inside U-shape");
        assert!(!any_u_in_r, "No U-shape vertex should be inside rect");
        assert!(!edges_cross, "U-shape and rect edges should not cross");
    }

    /// Two faces sharing only an edge (overlap area = 0).
    #[test]
    fn test_edge_only_sharing() {
        let face0 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let face1 = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];

        let edges_cross = edges_intersect_2d(&face0, &face1, 0.01);
        assert!(!edges_cross, "Shared-edge faces should not have crossing edges");
    }

    /// Two faces sharing only a point.
    #[test]
    fn test_point_only_sharing() {
        let face0 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let face1 = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]];

        let edges_cross = edges_intersect_2d(&face0, &face1, 0.01);
        assert!(
            !edges_cross,
            "Point-only-sharing faces should not have crossing edges"
        );
    }
}
