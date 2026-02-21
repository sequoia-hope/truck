//! Coplanar face detection for boolean operations.
//!
//! When `intersection_curves` returns `None` for coplanar face pairs (because
//! `double_projection` diverges when normals are parallel), this module provides
//! `check_coplanar_faces` to detect such pairs.

use super::robust_classify::robust_orient2d;
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
    // Normals must be (anti-)parallel: angle between them < tol radians.
    // Using the small-angle approximation: 1 - cos(θ) ≈ θ²/2, so
    // (1 - |dot|) > tol * tol means angle > ~sqrt(2) * tol.
    if (1.0 - dot.abs()) > tol * tol {
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

        // Check if polygons share boundary vertices (handles identical/coincident faces)
        let share_boundary = polygons_share_boundary(&proj0, &proj1, tol * 10.0)
            && polygons_share_boundary(&proj1, &proj0, tol * 10.0);
        if share_boundary {
            return Some(same_sense);
        }

        let any_v1_in_0 = proj1.iter().any(|pt| point_in_polygon_2d(*pt, &proj0));
        let any_v0_in_1 = proj0.iter().any(|pt| point_in_polygon_2d(*pt, &proj1));
        let edges_cross = edges_intersect_2d(&proj0, &proj1, tol);

        if !any_v1_in_0 && !any_v0_in_1 && !edges_cross {
            return None;
        }
    }

    Some(same_sense)
}

/// Winding-number point-in-polygon test in 2D using robust orientation predicates.
fn point_in_polygon_2d(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut winding: i32 = 0;
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = polygon[i][1];
        let yj = polygon[j][1];
        if yi <= point[1] {
            if yj > point[1] {
                // Upward crossing — point is left of edge → winding += 1
                if robust_orient2d(polygon[i], polygon[j], point) > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= point[1] {
            // Downward crossing — point is right of edge → winding -= 1
            if robust_orient2d(polygon[i], polygon[j], point) < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

/// Check if any edge of polygon A properly intersects any edge of polygon B in 2D.
/// Uses four robust orientation tests per segment pair (O(n*m), fine for small face polygons).
/// Only detects proper intersections (endpoints strictly on opposite sides), not
/// shared-vertex or collinear overlaps, matching the original endpoint-exclusion semantics.
fn edges_intersect_2d(poly_a: &[[f64; 2]], poly_b: &[[f64; 2]], _tol: f64) -> bool {
    for i in 0..poly_a.len() {
        let j = (i + 1) % poly_a.len();
        let a = poly_a[i];
        let b = poly_a[j];
        for k in 0..poly_b.len() {
            let l = (k + 1) % poly_b.len();
            let c = poly_b[k];
            let d = poly_b[l];

            let d1 = robust_orient2d(c, d, a);
            let d2 = robust_orient2d(c, d, b);
            let d3 = robust_orient2d(a, b, c);
            let d4 = robust_orient2d(a, b, d);

            // Proper intersection: endpoints strictly on opposite sides of each other's line
            if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
                return true;
            }
        }
    }
    false
}

/// Minimum distance from point to a line segment in 2D.
fn point_to_segment_dist_2d(pt: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [pt[0] - a[0], pt[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    if len2 < 1e-30 {
        return (ap[0] * ap[0] + ap[1] * ap[1]).sqrt();
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / len2).clamp(0.0, 1.0);
    let proj = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((pt[0] - proj[0]).powi(2) + (pt[1] - proj[1]).powi(2)).sqrt()
}

/// Check if any vertex of polygon A lies on or very near the boundary of polygon B.
fn polygons_share_boundary(poly_a: &[[f64; 2]], poly_b: &[[f64; 2]], tol: f64) -> bool {
    for &pt in poly_a {
        for k in 0..poly_b.len() {
            let l = (k + 1) % poly_b.len();
            if point_to_segment_dist_2d(pt, poly_b[k], poly_b[l]) < tol {
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
        assert!(
            !edges_cross,
            "Shared-edge faces should not have crossing edges"
        );
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

    // ── Robust predicate tests ──

    #[test]
    fn test_robust_point_in_polygon_inside() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(
            point_in_polygon_2d([0.5, 0.5], &sq),
            "Center of unit square should be inside"
        );
    }

    #[test]
    fn test_robust_point_in_polygon_outside() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(
            !point_in_polygon_2d([2.0, 0.5], &sq),
            "Point far right of unit square should be outside"
        );
    }

    #[test]
    fn test_robust_point_on_edge_boundary() {
        // Point exactly on the bottom edge of the unit square.
        // With the winding number algorithm, the bottom edge (y=0) is
        // "owned" — the upward edge [1,0]→[1,1] registers a crossing
        // because yi <= point[1] (0 <= 0) and yj > point[1] (1 > 0),
        // but the downward edge [0,1]→[0,0] does not cancel it.
        // This is the standard winding number boundary convention.
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let on_bottom = point_in_polygon_2d([0.5, 0.0], &sq);
        assert!(
            on_bottom,
            "Bottom edge is inside (winding number convention)"
        );

        // Top edge: y=1.0. The upward edge [1,0]→[1,1] has yj=1 which is
        // NOT > 1.0, so no crossing is counted → outside.
        let on_top = point_in_polygon_2d([0.5, 1.0], &sq);
        assert!(!on_top, "Top edge is outside (winding number convention)");
    }

    #[test]
    fn test_robust_edges_intersect() {
        // Overlapping squares: edges cross
        let a = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!(
            edges_intersect_2d(&a, &b, 0.001),
            "Overlapping squares should have crossing edges"
        );

        // Separated squares: no edge crossing
        let c = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let d = [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]];
        assert!(
            !edges_intersect_2d(&c, &d, 0.001),
            "Separated squares should not have crossing edges"
        );
    }
}
