//! BVH (R-tree) acceleration for ray-cast classification.
//!
//! Builds an R-tree over triangulated shell faces so that `try_ray_cast` can
//! reject the vast majority of triangles without testing them individually.

use rstar::{RTree, RTreeObject, AABB};
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::*;

/// A triangle stored in the BVH with its pre-computed normal.
pub(crate) struct BvhTriangle {
    pub vertices: [[f64; 3]; 3],
    /// Pre-computed face normal (not normalized).
    pub normal: [f64; 3],
}

impl RTreeObject for BvhTriangle {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        let [v0, v1, v2] = self.vertices;
        let min = [
            v0[0].min(v1[0]).min(v2[0]),
            v0[1].min(v1[1]).min(v2[1]),
            v0[2].min(v1[2]).min(v2[2]),
        ];
        let max = [
            v0[0].max(v1[0]).max(v2[0]),
            v0[1].max(v1[1]).max(v2[1]),
            v0[2].max(v1[2]).max(v2[2]),
        ];
        AABB::from_corners(min, max)
    }
}

/// Build a BVH (R-tree) from a triangulated polygon shell.
///
/// Iterates all faces of the shell, fan-triangulates each polygon, and
/// inserts the resulting triangles into an R-tree for fast spatial queries.
pub(crate) fn build_triangle_bvh(
    poly_shell: &Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>>,
) -> RTree<BvhTriangle> {
    let mut triangles = Vec::new();

    for face in poly_shell.iter() {
        let Some(poly) = face.surface() else {
            continue;
        };
        let positions = poly.positions();
        for face_verts in poly.face_iter() {
            for i in 2..face_verts.len() {
                let p0 = positions[face_verts[0].pos];
                let p1 = positions[face_verts[i - 1].pos];
                let p2 = positions[face_verts[i].pos];

                let e1 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
                let e2 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
                let normal = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];

                triangles.push(BvhTriangle {
                    vertices: [[p0.x, p0.y, p0.z], [p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]],
                    normal,
                });
            }
        }
    }

    RTree::bulk_load(triangles)
}

/// Test whether a ray's AABB intersects a triangle's AABB.
///
/// The ray AABB extends from `origin` infinitely in `dir` (positive direction only).
/// We approximate "infinitely" with the max extent of the BVH + some margin.
fn ray_intersects_aabb(origin: [f64; 3], dir: [f64; 3], aabb: &AABB<[f64; 3]>) -> bool {
    let lower = aabb.lower();
    let upper = aabb.upper();

    // Slab intersection test (Kay-Kajiya).
    // For each axis, compute the t-range where the ray is inside the slab.
    let mut t_min = 0.0f64; // ray starts at t=0 (origin)
    let mut t_max = f64::MAX;

    for i in 0..3 {
        if dir[i].abs() < 1e-30 {
            // Ray is parallel to this slab — check if origin is inside.
            if origin[i] < lower[i] || origin[i] > upper[i] {
                return false;
            }
        } else {
            let inv_d = 1.0 / dir[i];
            let mut t1 = (lower[i] - origin[i]) * inv_d;
            let mut t2 = (upper[i] - origin[i]) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return false;
            }
        }
    }

    true
}

/// Ray-cast from a point against a BVH-indexed triangulated shell using robust
/// geometric predicates.
///
/// Returns the signed crossing count, or `None` if any triangle produces a
/// degenerate configuration (origin on triangle plane).
pub(crate) fn try_ray_cast_bvh(
    pt: Point3,
    dir: Vector3,
    bvh: &RTree<BvhTriangle>,
) -> Option<isize> {
    use super::robust_classify::robust_ray_triangle_cross;

    let ray_origin = [pt.x, pt.y, pt.z];
    let ray_dir = [dir.x, dir.y, dir.z];

    let mut count = 0isize;

    for tri in bvh.iter() {
        // Quick AABB rejection
        if !ray_intersects_aabb(ray_origin, ray_dir, &tri.envelope()) {
            continue;
        }

        let crossing = robust_ray_triangle_cross(ray_origin, ray_dir, tri.vertices)?;

        if crossing == 1 {
            let dot = tri.normal[0] * dir.x + tri.normal[1] * dir.y + tri.normal[2] * dir.z;
            if dot > 0.0 {
                count += 1;
            } else {
                count -= 1;
            }
        }
    }

    Some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple unit cube poly shell for testing.
    fn unit_cube_poly_shell() -> Shell<Point3, PolylineCurve<Point3>, Option<PolygonMesh>> {
        use truck_modeling::builder;
        type Solid = truck_topology::Solid<Point3, truck_modeling::Curve, truck_modeling::Surface>;
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        let cube: Solid = builder::tsweep(&f, Vector3::unit_z());
        cube.boundaries()[0].triangulation(0.05)
    }

    #[test]
    fn test_bvh_build_nonempty() {
        let poly_shell = unit_cube_poly_shell();
        let bvh = build_triangle_bvh(&poly_shell);
        assert!(bvh.size() > 0, "BVH should contain triangles");
        // A unit cube has 6 faces, each with 2 triangles = 12 minimum
        assert!(
            bvh.size() >= 12,
            "Unit cube should have at least 12 triangles, got {}",
            bvh.size()
        );
    }

    #[test]
    fn test_bvh_matches_linear_scan() {
        use crate::transversal::integrate::{irrational_ray_dirs, try_ray_cast};

        let poly_shell = unit_cube_poly_shell();
        let bvh = build_triangle_bvh(&poly_shell);
        let dirs = irrational_ray_dirs();

        // Test multiple points with both linear and BVH approaches
        let test_points = [
            Point3::new(0.5, 0.5, 0.5),  // inside
            Point3::new(5.0, 5.0, 5.0),  // outside
            Point3::new(0.1, 0.1, 0.1),  // inside near corner
            Point3::new(-1.0, 0.5, 0.5), // outside
        ];

        for &pt in &test_points {
            for &d in &dirs {
                let linear = try_ray_cast(pt, d, &poly_shell);
                let bvh_result = try_ray_cast_bvh(pt, d, &bvh);
                assert_eq!(
                    linear, bvh_result,
                    "BVH and linear scan disagree for pt=({:.1},{:.1},{:.1}) dir=({:.3},{:.3},{:.3}): linear={:?} bvh={:?}",
                    pt.x, pt.y, pt.z, d.x, d.y, d.z, linear, bvh_result
                );
            }
        }
    }

    #[test]
    fn test_ray_aabb_hit() {
        let aabb = AABB::from_corners([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(ray_intersects_aabb(
            [0.5, 0.5, -1.0],
            [0.0, 0.0, 1.0],
            &aabb
        ));
    }

    #[test]
    fn test_ray_aabb_miss() {
        let aabb = AABB::from_corners([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(!ray_intersects_aabb(
            [5.0, 5.0, -1.0],
            [0.0, 0.0, 1.0],
            &aabb
        ));
    }

    #[test]
    fn test_ray_aabb_behind() {
        let aabb = AABB::from_corners([0.0, 0.0, -5.0], [1.0, 1.0, -3.0]);
        // Ray at z=0 shooting +Z should miss box at z=-5..-3
        assert!(!ray_intersects_aabb(
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            &aabb
        ));
    }
}
