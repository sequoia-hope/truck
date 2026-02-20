//! Wrappers around Shewchuk's robust adaptive predicates for
//! geometric classification in boolean operations.

/// Robust 3D orientation test (Shewchuk convention).
///
/// Returns positive if `d` is below the oriented plane of `(a, b, c)` (CW
/// when viewed from d), negative if above (CCW from d), zero if coplanar.
/// Uses Shewchuk's adaptive precision arithmetic for exact sign.
#[allow(dead_code)]
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

/// Classify a point relative to a triangle using robust predicates.
///
/// Tests whether a ray from `ray_origin` along `ray_dir` crosses the triangle
/// defined by `tri[0..3]`. Returns:
/// - `Some(1)` if the ray crosses the triangle (point is "inside" relative to this face)
/// - `Some(0)` if the ray does not cross
/// - `None` if the configuration is degenerate (ray grazes an edge)
///
/// Uses robust_orient3d to classify the point relative to the triangle plane,
/// and robust_orient2d for the projected 2D containment test.
#[allow(dead_code)]
pub(crate) fn robust_ray_triangle_cross(
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    tri: [[f64; 3]; 3],
) -> Option<i32> {
    // Step 1: Determine which side of the triangle plane the ray origin lies on
    let orient = robust_orient3d(tri[0], tri[1], tri[2], ray_origin);

    // Compute a point far along the ray
    let far_pt = [
        ray_origin[0] + ray_dir[0] * 1e6,
        ray_origin[1] + ray_dir[1] * 1e6,
        ray_origin[2] + ray_dir[2] * 1e6,
    ];
    let orient_far = robust_orient3d(tri[0], tri[1], tri[2], far_pt);

    // If both on same side (or both on plane), no crossing
    if orient * orient_far > 0.0 {
        return Some(0);
    }
    // If origin is on the plane, degenerate
    if orient == 0.0 {
        return None;
    }
    // If far point is on plane, degenerate (ray is tangent)
    if orient_far == 0.0 {
        return None;
    }

    // Step 2: Ray crosses the plane. Now check if the crossing point is inside
    // the triangle using a 2D projection. Choose the projection axis that gives
    // the largest triangle area (based on the triangle normal).
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

    // Find the dominant axis
    let abs_n = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let (u_idx, v_idx) = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
        (1, 2) // project onto YZ
    } else if abs_n[1] >= abs_n[2] {
        (0, 2) // project onto XZ
    } else {
        (0, 1) // project onto XY
    };

    // Compute the ray-plane intersection parameter t
    let t = orient / (orient - orient_far);
    let cross_pt = [
        ray_origin[0] + ray_dir[0] * 1e6 * t,
        ray_origin[1] + ray_dir[1] * 1e6 * t,
        ray_origin[2] + ray_dir[2] * 1e6 * t,
    ];

    // Project to 2D and use robust orient2d for containment test
    let p = [cross_pt[u_idx], cross_pt[v_idx]];
    let a = [tri[0][u_idx], tri[0][v_idx]];
    let b = [tri[1][u_idx], tri[1][v_idx]];
    let c = [tri[2][u_idx], tri[2][v_idx]];

    let o_ab = robust_orient2d(a, b, p);
    let o_bc = robust_orient2d(b, c, p);
    let o_ca = robust_orient2d(c, a, p);

    // Point is inside if all orientations have the same sign
    if (o_ab > 0.0 && o_bc > 0.0 && o_ca > 0.0) || (o_ab < 0.0 && o_bc < 0.0 && o_ca < 0.0) {
        Some(1)
    } else if o_ab == 0.0 || o_bc == 0.0 || o_ca == 0.0 {
        // Point is on a triangle edge — degenerate
        None
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
    fn test_ray_triangle_cross_degenerate_on_edge() {
        let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let origin = [0.5, 0.5, 0.0];
        let dir = [0.0, 0.0, 1.0];
        assert_eq!(robust_ray_triangle_cross(origin, dir, tri), None);
    }
}
