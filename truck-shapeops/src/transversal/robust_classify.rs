//! Wrappers around Shewchuk's robust adaptive predicates for
//! geometric classification in boolean operations.

/// Robust 3D orientation test (Shewchuk convention).
///
/// Returns positive if `d` is below the oriented plane of `(a, b, c)` (CW
/// when viewed from d), negative if above (CCW from d), zero if coplanar.
/// Uses Shewchuk's adaptive precision arithmetic for exact sign.
pub(crate) fn robust_orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    robust::orient3d(
        robust::Coord3D {
            x: a[0],
            y: a[1],
            z: a[2],
        },
        robust::Coord3D {
            x: b[0],
            y: b[1],
            z: b[2],
        },
        robust::Coord3D {
            x: c[0],
            y: c[1],
            z: c[2],
        },
        robust::Coord3D {
            x: d[0],
            y: d[1],
            z: d[2],
        },
    )
}

/// Robust 2D orientation test.
///
/// Returns positive if `c` is left of line `(a, b)`,
/// negative if right, zero if collinear.
/// Uses Shewchuk's adaptive precision arithmetic for exact sign.
pub(crate) fn robust_orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    robust::orient2d(
        robust::Coord { x: a[0], y: a[1] },
        robust::Coord { x: b[0], y: b[1] },
        robust::Coord { x: c[0], y: c[1] },
    )
}

/// Simulation of Simplicity (SoS) tie-break for `orient2d`.
///
/// When `robust_orient2d(a, b, c)` returns exactly 0.0 (point `c` is
/// collinear with edge `a→b`), this gives a deterministic non-zero answer
/// based on lexicographic ordering of the vertices. This ensures that
/// point-in-triangle tests never return "degenerate" for rays exactly
/// through a mesh edge — each edge is counted as belonging to exactly one
/// of the two adjacent triangles.
///
/// The rule: sort (a, b, c) lexicographically. The sign is +1 if the
/// sort is an even permutation of the original order, -1 if odd.
fn sos_orient2d_tiebreak(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i32 {
    // Lexicographic comparison for 2D points
    let cmp = |p: &[f64; 2], q: &[f64; 2]| -> std::cmp::Ordering {
        p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1]))
    };
    let pts = [a, b, c];
    // Count the number of swaps needed to sort (bubble sort parity)
    let mut swaps = 0u32;
    let mut idx = [0usize, 1, 2];
    // Simple 3-element sort by parity counting
    if cmp(&pts[idx[0]], &pts[idx[1]]) == std::cmp::Ordering::Greater {
        idx.swap(0, 1);
        swaps += 1;
    }
    if cmp(&pts[idx[1]], &pts[idx[2]]) == std::cmp::Ordering::Greater {
        idx.swap(1, 2);
        swaps += 1;
    }
    if cmp(&pts[idx[0]], &pts[idx[1]]) == std::cmp::Ordering::Greater {
        idx.swap(0, 1);
        swaps += 1;
    }
    let _ = idx; // suppress unused warning
    if swaps.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Classify a point relative to a triangle using robust predicates.
///
/// Tests whether a ray from `ray_origin` along `ray_dir` crosses the triangle
/// defined by `tri[0..3]`. Returns:
/// - `Some(1)` if the ray crosses the triangle (point is "inside" relative to this face)
/// - `Some(0)` if the ray does not cross
/// - `None` if the ray origin lies on the triangle plane (degenerate)
///
/// Uses a proper half-space ray-triangle test:
/// 1. Compute `orient3d(tri, origin)` for the ray origin side of the plane
/// 2. Compute the ray direction's dot with the triangle normal to determine
///    if the ray crosses the plane in the forward direction
/// 3. Compute the actual intersection point and test containment in 2D
///
/// When the intersection point lands exactly on a triangle edge (orient2d == 0),
/// Simulation of Simplicity (SoS) provides a deterministic tie-break, ensuring
/// each shared mesh edge is counted by exactly one of its two adjacent triangles.
pub(crate) fn robust_ray_triangle_cross(
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    tri: [[f64; 3]; 3],
) -> Option<i32> {
    // Step 1: Determine which side of the triangle plane the ray origin lies on.
    let orient_origin = robust_orient3d(tri[0], tri[1], tri[2], ray_origin);

    // If origin is on the plane, degenerate — can't determine crossing.
    if orient_origin == 0.0 {
        return None;
    }

    // Step 2: Compute triangle normal n = (v1-v0) × (v2-v0) and test if the
    // ray direction crosses the plane in the forward direction.
    let e1 = [
        tri[1][0] - tri[0][0],
        tri[1][1] - tri[0][1],
        tri[1][2] - tri[0][2],
    ];
    let e2 = [
        tri[2][0] - tri[0][0],
        tri[2][1] - tri[0][1],
        tri[2][2] - tri[0][2],
    ];
    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    // n · dir: if zero, ray is parallel to the plane — no crossing.
    let n_dot_dir = normal[0] * ray_dir[0] + normal[1] * ray_dir[1] + normal[2] * ray_dir[2];
    if n_dot_dir == 0.0 {
        return Some(0);
    }

    // t = n · (v0 - origin) / (n · dir)
    let diff = [
        tri[0][0] - ray_origin[0],
        tri[0][1] - ray_origin[1],
        tri[0][2] - ray_origin[2],
    ];
    let n_dot_diff = normal[0] * diff[0] + normal[1] * diff[1] + normal[2] * diff[2];
    let t = n_dot_diff / n_dot_dir;

    // If t <= 0, intersection is behind the ray origin — no forward crossing.
    if t <= 0.0 {
        return Some(0);
    }

    // Step 3: Compute the intersection point.
    let cross_pt = [
        ray_origin[0] + ray_dir[0] * t,
        ray_origin[1] + ray_dir[1] * t,
        ray_origin[2] + ray_dir[2] * t,
    ];

    // Step 4: Project to 2D along the dominant normal axis and test containment.
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let (u_idx, v_idx) = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        (1, 2) // project onto YZ
    } else if abs_n[1] >= abs_n[2] {
        (0, 2) // project onto XZ
    } else {
        (0, 1) // project onto XY
    };

    let p = [cross_pt[u_idx], cross_pt[v_idx]];
    let a = [tri[0][u_idx], tri[0][v_idx]];
    let b = [tri[1][u_idx], tri[1][v_idx]];
    let c = [tri[2][u_idx], tri[2][v_idx]];

    let o_ab = robust_orient2d(a, b, p);
    let o_bc = robust_orient2d(b, c, p);
    let o_ca = robust_orient2d(c, a, p);

    // Point is strictly inside if all orientations have the same sign.
    if (o_ab > 0.0 && o_bc > 0.0 && o_ca > 0.0) || (o_ab < 0.0 && o_bc < 0.0 && o_ca < 0.0) {
        return Some(1);
    }

    // Point is strictly outside if signs disagree (and none are zero).
    if o_ab != 0.0 && o_bc != 0.0 && o_ca != 0.0 {
        return Some(0);
    }

    // At least one orient2d is exactly zero — point is on a triangle edge or vertex.
    // Count how many are zero: 1 = edge case (SoS handles), 2+ = vertex case (degenerate).
    let zero_count = [o_ab, o_bc, o_ca].iter().filter(|&&v| v == 0.0).count();

    // Vertex case: ray passes through a triangle vertex where 2+ edges meet.
    // SoS can't resolve this consistently across the fan of triangles sharing
    // the vertex — return None to let the caller handle it with perturbation.
    if zero_count >= 2 {
        return None;
    }

    // Single edge case: apply SoS tie-breaking for a deterministic result.
    // This ensures each shared mesh edge is counted by exactly one of its
    // two adjacent triangles.
    let s_ab = if o_ab == 0.0 {
        sos_orient2d_tiebreak(a, b, p)
    } else {
        o_ab.signum() as i32
    };
    let s_bc = if o_bc == 0.0 {
        sos_orient2d_tiebreak(b, c, p)
    } else {
        o_bc.signum() as i32
    };
    let s_ca = if o_ca == 0.0 {
        sos_orient2d_tiebreak(c, a, p)
    } else {
        o_ca.signum() as i32
    };

    if (s_ab > 0 && s_bc > 0 && s_ca > 0) || (s_ab < 0 && s_bc < 0 && s_ca < 0) {
        Some(1)
    } else {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient3d_exact_coplanar() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.5, 0.5, 0.0];
        assert_eq!(robust_orient3d(a, b, c, d), 0.0);
    }

    #[test]
    fn test_orient3d_clearly_above() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        let result = robust_orient3d(a, b, c, d);
        assert!(result < 0.0, "Expected negative (above), got {result}");
    }

    #[test]
    fn test_orient3d_clearly_below() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, -1.0];
        let result = robust_orient3d(a, b, c, d);
        assert!(result > 0.0, "Expected positive (below), got {result}");
    }

    #[test]
    fn test_orient3d_ill_conditioned() {
        let a = [1e8, 0.0, 0.0];
        let b = [0.0, 1e8, 0.0];
        let c = [0.0, 0.0, 1e8];
        let d = [1e8 / 3.0, 1e8 / 3.0, 1e8 / 3.0 + 1e-7];

        let result = robust_orient3d(a, b, c, d);
        assert_ne!(
            result, 0.0,
            "Robust predicate should detect non-coplanarity"
        );

        let d_below = [1e8 / 3.0, 1e8 / 3.0, 1e8 / 3.0 - 1e-7];
        let result_below = robust_orient3d(a, b, c, d_below);
        assert_ne!(result_below, 0.0);
        assert!(
            result.signum() != result_below.signum(),
            "Signs should differ for points on opposite sides: {result} vs {result_below}"
        );
    }

    #[test]
    fn test_orient2d_collinear() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.5, 0.0];
        assert_eq!(robust_orient2d(a, b, c), 0.0);
    }

    #[test]
    fn test_orient2d_left_right() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];

        let left = [0.5, 1.0];
        let result_left = robust_orient2d(a, b, left);
        assert!(
            result_left > 0.0,
            "Expected positive (left), got {result_left}"
        );

        let right = [0.5, -1.0];
        let result_right = robust_orient2d(a, b, right);
        assert!(
            result_right < 0.0,
            "Expected negative (right), got {result_right}"
        );
    }

    #[test]
    fn test_ray_triangle_cross_hit() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let origin = [0.5, 0.5, -1.0];
        let dir = [0.0, 0.0, 1.0];
        assert_eq!(robust_ray_triangle_cross(origin, dir, tri), Some(1));
    }

    #[test]
    fn test_ray_triangle_cross_miss() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let origin = [10.0, 10.0, -1.0];
        let dir = [0.0, 0.0, 1.0];
        assert_eq!(robust_ray_triangle_cross(origin, dir, tri), Some(0));
    }

    #[test]
    fn test_ray_triangle_cross_degenerate_on_plane() {
        // Origin is ON the triangle plane — should return None.
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let origin = [0.5, 0.5, 0.0];
        let dir = [0.0, 0.0, 1.0];
        assert_eq!(robust_ray_triangle_cross(origin, dir, tri), None);
    }

    #[test]
    fn test_ray_exactly_through_edge() {
        // Ray passes exactly through edge v0-v1 (the x-axis edge).
        // SoS should give a deterministic result (not None).
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let origin = [0.5, 0.0, -1.0];
        let dir = [0.0, 0.0, 1.0];
        let result = robust_ray_triangle_cross(origin, dir, tri);
        assert!(
            result.is_some(),
            "SoS should handle edge case without returning None"
        );
    }

    #[test]
    fn test_ray_through_vertex() {
        // Ray passes exactly through vertex v0. This is a vertex case (2+ orient2d
        // values are zero), so it should return None — SoS only handles single-edge
        // cases. The caller retries with perturbation for vertex degeneracies.
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let origin = [0.0, 0.0, -1.0];
        let dir = [0.0, 0.0, 1.0];
        let result = robust_ray_triangle_cross(origin, dir, tri);
        assert_eq!(
            result, None,
            "Vertex case should return None (handled by caller perturbation)"
        );
    }

    #[test]
    fn test_no_far_point_needed() {
        // Triangle at distance > 1e6 from origin — old 1e6 far-point code would fail.
        let tri = [[2e6, 0.0, 0.0], [2e6, 1.0, 0.0], [2e6, 0.0, 1.0]];
        let origin = [0.0, 0.25, 0.25];
        let dir = [1.0, 0.0, 0.0];
        let result = robust_ray_triangle_cross(origin, dir, tri);
        assert_eq!(result, Some(1), "Should detect crossing at distance > 1e6");
    }

    #[test]
    fn test_ray_behind_origin_no_cross() {
        // Triangle is behind the ray origin — should not count as crossing.
        let tri = [[0.0, 0.0, -5.0], [1.0, 0.0, -5.0], [0.0, 1.0, -5.0]];
        let origin = [0.25, 0.25, 0.0];
        let dir = [0.0, 0.0, 1.0]; // shooting +Z, triangle at Z=-5
        let result = robust_ray_triangle_cross(origin, dir, tri);
        assert_eq!(result, Some(0), "Triangle behind ray should not cross");
    }

    #[test]
    fn test_ray_parallel_to_triangle() {
        // Ray direction is parallel to the triangle plane.
        let tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let origin = [0.0, 0.0, 1.0];
        let dir = [1.0, 0.0, 0.0]; // parallel to XY plane
        let result = robust_ray_triangle_cross(origin, dir, tri);
        assert_eq!(result, Some(0), "Parallel ray should not cross");
    }

    #[test]
    fn test_sos_consistency_adjacent_triangles() {
        // Two triangles sharing an edge. A ray through the shared edge
        // should be counted by exactly one of them (SoS ensures this).
        let shared_edge_a = [0.0, 0.0, 0.0];
        let shared_edge_b = [1.0, 0.0, 0.0];
        let tri1 = [shared_edge_a, shared_edge_b, [0.5, 1.0, 0.0]];
        let tri2 = [shared_edge_a, shared_edge_b, [0.5, -1.0, 0.0]];

        // Ray goes through the shared edge at (0.5, 0, 0)
        let origin = [0.5, 0.0, -1.0];
        let dir = [0.0, 0.0, 1.0];

        let r1 = robust_ray_triangle_cross(origin, dir, tri1);
        let r2 = robust_ray_triangle_cross(origin, dir, tri2);

        assert!(r1.is_some(), "SoS should resolve tri1");
        assert!(r2.is_some(), "SoS should resolve tri2");

        // Exactly one should report a crossing (the pair forms a closed edge).
        let total = r1.unwrap() + r2.unwrap();
        assert_eq!(
            total, 1,
            "Exactly one of two adjacent triangles should count the edge crossing, got {} + {} = {}",
            r1.unwrap(),
            r2.unwrap(),
            total,
        );
    }

    #[test]
    fn test_sos_tiebreak_deterministic() {
        // SoS tie-break should be deterministic (same inputs → same output).
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.5, 0.0]; // collinear
        let r1 = sos_orient2d_tiebreak(a, b, c);
        let r2 = sos_orient2d_tiebreak(a, b, c);
        assert_eq!(r1, r2, "SoS should be deterministic");
        assert!(r1 == 1 || r1 == -1, "SoS should return +1 or -1, got {r1}");
    }
}
