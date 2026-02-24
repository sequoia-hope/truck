//! Coplanar face overlay using iOverlay for 2D polygon boolean operations.
//!
//! When two faces lie on the same plane, mesh-based surface-surface intersection
//! produces nothing (coplanar triangles don't "intersect"). This module handles
//! coplanar face pairs by:
//!
//! 1. Projecting both face boundaries into a shared 2D coordinate system
//! 2. Running iOverlay's polygon boolean to compute overlap/difference/complement
//! 3. Returning classified 2D fragments with Inside/Outside/OnBoundary labels
//!
//! The overlay results feed into the face division pipeline, replacing the
//! containment-only logic that previously handled coplanar faces.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use truck_base::cgmath64::*;
use truck_topology::*;

#[allow(unused_imports)]
use super::coplanar::point_in_polygon;
use super::coplanar::{make_tangent_basis, project_to_2d};
use super::coplanar_splitting::face_boundary_info;
use super::integrate::ShapeOpsSurface;
#[allow(unused_imports)]
use super::loops_store::ShapesOpStatus;

/// (outer_contour, hole_contours) from projecting a face to 2D
type FaceContours2D = (Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>);

/// A 2D polygon fragment from overlay, classified with its relation to the other solid.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OverlayFragment {
    /// Outer boundary contour in 2D (counterclockwise)
    pub outer: Vec<[f64; 2]>,
    /// Inner boundary contours (holes) in 2D (clockwise)
    pub holes: Vec<Vec<[f64; 2]>>,
    /// Classification for boolean face selection
    pub status: ShapesOpStatus,
}

/// Result of a coplanar overlay between two face boundaries.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CoplanarOverlayResult {
    /// Fragments belonging to face0 (for injection into face0's loops)
    pub fragments0: Vec<OverlayFragment>,
    /// Fragments belonging to face1 (for injection into face1's loops)
    pub fragments1: Vec<OverlayFragment>,
    /// The 2D coordinate system used for projection
    pub coord_system: CoplanarCoordSystem,
}

/// The shared 2D coordinate system for a coplanar face pair.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CoplanarCoordSystem {
    pub origin: Point3,
    pub u_axis: Vector3,
    pub v_axis: Vector3,
    pub normal: Vector3,
}

impl CoplanarCoordSystem {
    /// Lift a 2D point back to 3D.
    #[allow(dead_code)]
    pub fn to_3d(&self, pt: [f64; 2]) -> Point3 {
        self.origin + self.u_axis * pt[0] + self.v_axis * pt[1]
    }
}

/// Extract 2D polygon contours from a face (outer boundary + holes).
///
/// Returns (outer_contour, hole_contours) in the given coordinate system.
fn face_to_2d_contours<C, S>(
    face: &Face<Point3, C, S>,
    coord: &CoplanarCoordSystem,
) -> Option<FaceContours2D> {
    let boundaries = face.absolute_boundaries();
    let outer_wire = boundaries.first()?;

    let outer: Vec<[f64; 2]> = outer_wire
        .vertex_iter()
        .map(|v| project_to_2d(v.point(), coord.origin, coord.u_axis, coord.v_axis))
        .collect();

    if outer.len() < 3 {
        return None;
    }

    let mut holes = Vec::new();
    for wire in boundaries.iter().skip(1) {
        let hole: Vec<[f64; 2]> = wire
            .vertex_iter()
            .map(|v| project_to_2d(v.point(), coord.origin, coord.u_axis, coord.v_axis))
            .collect();
        if hole.len() >= 3 {
            holes.push(hole);
        }
    }

    Some((outer, holes))
}

/// Build the iOverlay input shape from outer boundary + holes.
///
/// iOverlay expects: `Vec<Vec<[f64; 2]>>` where first contour is outer, rest are holes.
fn build_overlay_shape(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let mut shape = Vec::with_capacity(1 + holes.len());
    shape.push(outer.to_vec());
    for hole in holes {
        shape.push(hole.clone());
    }
    shape
}

/// Compute the centroid of a 2D polygon contour.
#[allow(dead_code)]
fn contour_centroid(contour: &[[f64; 2]]) -> [f64; 2] {
    if contour.is_empty() {
        return [0.0, 0.0];
    }
    let n = contour.len() as f64;
    let (sx, sy) = contour
        .iter()
        .fold((0.0, 0.0), |(sx, sy), p| (sx + p[0], sy + p[1]));
    [sx / n, sy / n]
}

/// Compute the signed area of a 2D polygon contour (positive = CCW, negative = CW).
fn signed_area_2d(contour: &[[f64; 2]]) -> f64 {
    let n = contour.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += contour[i][0] * contour[j][1];
        area -= contour[j][0] * contour[i][1];
    }
    area / 2.0
}

/// Perform 2D polygon overlay between two coplanar faces.
///
/// Returns overlay fragments classified for boolean operations:
/// - Intersection region (overlap): both faces contain this area
/// - Subject-only region: only face0 contains this area
/// - Clip-only region: only face1 contains this area
///
/// The `same_sense` parameter indicates whether the face normals point in the
/// same direction (true) or opposite directions (false/anti-sense).
#[allow(dead_code)]
pub fn compute_coplanar_overlay<C, S: ShapeOpsSurface>(
    face0: &Face<Point3, C, S>,
    face1: &Face<Point3, C, S>,
    same_sense: bool,
    tol: f64,
) -> Option<CoplanarOverlayResult> {
    // Get face boundary info for coordinate system
    let (verts0, normal0) = face_boundary_info(face0)?;
    if verts0.len() < 3 {
        return None;
    }

    // Build shared 2D coordinate system from face0's plane
    let coord = CoplanarCoordSystem {
        origin: verts0[0],
        u_axis: {
            let (u, _) = make_tangent_basis(normal0);
            u
        },
        v_axis: {
            let (_, v) = make_tangent_basis(normal0);
            v
        },
        normal: normal0,
    };

    // Project both faces to 2D
    let (outer0, holes0) = face_to_2d_contours(face0, &coord)?;
    let (outer1, holes1) = face_to_2d_contours(face1, &coord)?;

    // Filter degenerate faces
    let area0 = signed_area_2d(&outer0).abs();
    let area1 = signed_area_2d(&outer1).abs();
    let min_area = tol * tol;
    if area0 < min_area || area1 < min_area {
        return None;
    }

    let shape0 = build_overlay_shape(&outer0, &holes0);
    let shape1 = build_overlay_shape(&outer1, &holes1);

    // Compute three overlay regions:
    // 1. Intersection (A ∩ B) — the overlap region
    // 2. Difference (A \ B) — face0 minus overlap
    // 3. Inverse Difference (B \ A) — face1 minus overlap

    let intersection_shapes: Vec<Vec<Vec<[f64; 2]>>> =
        shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);
    let diff0_shapes: Vec<Vec<Vec<[f64; 2]>>> =
        shape0.overlay(&shape1, OverlayRule::Difference, FillRule::EvenOdd);
    // B \ A = reverse difference
    let diff1_shapes: Vec<Vec<Vec<[f64; 2]>>> =
        shape1.overlay(&shape0, OverlayRule::Difference, FillRule::EvenOdd);

    // Check if there's any actual overlap
    let has_overlap = intersection_shapes.iter().any(|shape| {
        shape
            .first()
            .map(|c| signed_area_2d(c).abs() > min_area)
            .unwrap_or(false)
    });

    if !has_overlap {
        // No overlap — faces are coplanar but disjoint
        return None;
    }

    let mut fragments0 = Vec::new();
    let mut fragments1 = Vec::new();

    // Intersection fragments: these are shared by both faces.
    // For same-sense overlap: And for shell0, Or for shell1 (or vice versa,
    // depending on the boolean op — but we classify as And/Or based on
    // containment which the caller uses to select faces).
    //
    // For anti-sense overlap: the overlap region cancels out (Remove).
    for shape in &intersection_shapes {
        if shape.is_empty() {
            continue;
        }
        let outer = &shape[0];
        if signed_area_2d(outer).abs() < min_area {
            continue;
        }
        let holes: Vec<Vec<[f64; 2]>> = shape.iter().skip(1).cloned().collect();

        if same_sense {
            // Same-sense overlap: face0's overlap → And (inside shell1)
            fragments0.push(OverlayFragment {
                outer: outer.clone(),
                holes: holes.clone(),
                status: ShapesOpStatus::And,
            });
            // face1's overlap → And (inside shell0)
            fragments1.push(OverlayFragment {
                outer: outer.clone(),
                holes,
                status: ShapesOpStatus::And,
            });
        } else {
            // Anti-sense overlap: faces cancel each other (Remove).
            // We mark these as And in both shells so they get included in
            // the "And" bucket. The caller's boolean operation selection
            // (union = or0+or1, difference = or0+inv(and1)) handles them correctly.
            // For union: And faces are discarded (they're "inside" the other solid).
            // For difference: And faces from shell1 get inverted.
            //
            // Actually, for anti-sense overlap in union, we want BOTH faces removed
            // (they're internal). Mark as And so they go to the And bucket.
            fragments0.push(OverlayFragment {
                outer: outer.clone(),
                holes: holes.clone(),
                status: ShapesOpStatus::And,
            });
            fragments1.push(OverlayFragment {
                outer: outer.clone(),
                holes,
                status: ShapesOpStatus::And,
            });
        }
    }

    // Difference fragments (A \ B): parts of face0 NOT overlapping face1.
    // These are outside shell1 → Or.
    for shape in &diff0_shapes {
        if shape.is_empty() {
            continue;
        }
        let outer = &shape[0];
        if signed_area_2d(outer).abs() < min_area {
            continue;
        }
        let holes: Vec<Vec<[f64; 2]>> = shape.iter().skip(1).cloned().collect();

        fragments0.push(OverlayFragment {
            outer: outer.clone(),
            holes,
            status: ShapesOpStatus::Or,
        });
    }

    // Inverse difference fragments (B \ A): parts of face1 NOT overlapping face0.
    // These are outside shell0 → Or.
    for shape in &diff1_shapes {
        if shape.is_empty() {
            continue;
        }
        let outer = &shape[0];
        if signed_area_2d(outer).abs() < min_area {
            continue;
        }
        let holes: Vec<Vec<[f64; 2]>> = shape.iter().skip(1).cloned().collect();

        fragments1.push(OverlayFragment {
            outer: outer.clone(),
            holes,
            status: ShapesOpStatus::Or,
        });
    }

    Some(CoplanarOverlayResult {
        fragments0,
        fragments1,
        coord_system: coord,
    })
}

/// Convert overlay fragments into truck topology edges and inject them as
/// boundary loops into the face's loops_store.
///
/// Each fragment contour becomes a closed wire in the loops_store with the
/// appropriate ShapesOpStatus. The caller then uses divide_face to split
/// the original face along these boundaries.
#[allow(dead_code)]
pub fn inject_overlay_fragments<C>(
    fragments: &[OverlayFragment],
    coord: &CoplanarCoordSystem,
    face_idx: usize,
    loops_store: &mut super::loops_store::LoopsStore<Point3, C>,
    _tol: f64,
) where
    C: Clone + From<truck_geometry::prelude::BSplineCurve<Point3>>,
{
    use super::loops_store::BoundaryWire;
    #[allow(unused_imports)]
    use truck_geometry::prelude::BSplineCurve;

    for fragment in fragments {
        if fragment.outer.len() < 3 {
            continue;
        }

        // Build a closed wire from the fragment's outer boundary
        let wire = contour_to_wire(&fragment.outer, coord);

        // Inject as independent loop (adds both the wire and its inverse)
        if face_idx < loops_store.len() {
            loops_store[face_idx].add_independent_loop(BoundaryWire::new(wire, fragment.status));
        }

        // Also inject holes as independent loops (inverted status)
        for hole in &fragment.holes {
            if hole.len() < 3 {
                continue;
            }
            let hole_wire = contour_to_wire(hole, coord);
            if face_idx < loops_store.len() {
                loops_store[face_idx]
                    .add_independent_loop(BoundaryWire::new(hole_wire, fragment.status));
            }
        }
    }
}

/// Convert a 2D contour into a 3D closed wire using line edges.
#[allow(dead_code)]
fn contour_to_wire<C: From<truck_geometry::prelude::BSplineCurve<Point3>>>(
    contour: &[[f64; 2]],
    coord: &CoplanarCoordSystem,
) -> Wire<Point3, C> {
    use truck_geometry::prelude::BSplineCurve;

    let n = contour.len();
    let vertices: Vec<Vertex<Point3>> = contour
        .iter()
        .map(|&pt| Vertex::new(coord.to_3d(pt)))
        .collect();

    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let p0 = vertices[i].point();
        let p1 = vertices[j].point();
        // Create a linear BSplineCurve (degree 1) between the two points
        let curve = BSplineCurve::new(
            truck_geometry::prelude::KnotVec::bezier_knot(1),
            vec![p0, p1],
        );
        let edge = Edge::new(&vertices[i], &vertices[j], C::from(curve));
        edges.push(edge);
    }

    edges.into_iter().collect()
}

/// Classify a face fragment that may be coplanar with faces in the other shell
/// using 2D polygon overlay (iOverlay).
///
/// Unlike `classify_coplanar_fragment` which tests a single interior point,
/// this projects the entire face boundary and all coplanar faces from the other
/// shell into 2D, merges the other-shell faces via union, then computes the
/// intersection area. This is robust to boundary-proximity issues that cause
/// single-point classification to fail.
///
/// Returns `Some(action)` if the face is coplanar with faces in the other shell
/// and the overlay produces a definitive classification. Returns `None` if:
/// - The face is not coplanar with any other-shell face
/// - The overlay cannot produce a result (degenerate geometry)
pub(crate) fn classify_coplanar_via_overlay<C1, C2, S: ShapeOpsSurface>(
    face: &Face<Point3, C1, S>,
    other_shell: &Shell<Point3, C2, S>,
    is_shell0: bool,
    tol: f64,
) -> Option<CoplanarAction> {
    use super::coplanar::{check_coplanar, face_sample_info};

    let info = face_sample_info(face)?;

    // Collect all coplanar faces from the other shell, split by sense
    let mut same_sense_faces: Vec<usize> = Vec::new();
    let mut anti_sense_faces: Vec<usize> = Vec::new();

    for (idx, other_face) in other_shell.iter().enumerate() {
        let other_info = match face_sample_info(other_face) {
            Some(i) => i,
            None => continue,
        };
        match check_coplanar(&info, &other_info, tol) {
            Some(true) => same_sense_faces.push(idx),
            Some(false) => anti_sense_faces.push(idx),
            None => continue,
        }
    }

    if same_sense_faces.is_empty() && anti_sense_faces.is_empty() {
        return None; // Not coplanar with anything — fall to ray-cast
    }

    // Build 2D coordinate system from the face's plane
    let (u_axis, v_axis) = make_tangent_basis(info.normal);
    let coord = CoplanarCoordSystem {
        origin: info.point,
        u_axis,
        v_axis,
        normal: info.normal,
    };

    // Project the unknown face to 2D
    let (face_outer, face_holes) = face_to_2d_contours(face, &coord)?;
    let face_area = signed_area_2d(&face_outer).abs();
    let min_area = tol * tol;
    if face_area < min_area {
        return None;
    }
    let face_shape = build_overlay_shape(&face_outer, &face_holes);

    // Check anti-sense faces first (Remove takes priority)
    if !anti_sense_faces.is_empty() {
        let merged = merge_other_faces(other_shell, &anti_sense_faces, &coord, min_area);
        if let Some(merged_shape) = merged {
            let intersection: Vec<Vec<Vec<[f64; 2]>>> =
                face_shape.overlay(&merged_shape, OverlayRule::Intersect, FillRule::EvenOdd);
            let int_area: f64 = intersection
                .iter()
                .flat_map(|s| s.first())
                .map(|c| signed_area_2d(c).abs())
                .sum();
            if int_area > min_area {
                return Some(CoplanarAction::Remove);
            }
        }
    }

    // Check same-sense faces
    if !same_sense_faces.is_empty() {
        let merged = merge_other_faces(other_shell, &same_sense_faces, &coord, min_area);
        if let Some(merged_shape) = merged {
            let intersection: Vec<Vec<Vec<[f64; 2]>>> =
                face_shape.overlay(&merged_shape, OverlayRule::Intersect, FillRule::EvenOdd);
            let int_area: f64 = intersection
                .iter()
                .flat_map(|s| s.first())
                .map(|c| signed_area_2d(c).abs())
                .sum();
            if int_area > min_area {
                // Face overlaps with same-sense coplanar faces in other shell → inside
                return Some(if is_shell0 {
                    CoplanarAction::And
                } else {
                    CoplanarAction::Or
                });
            }
        }
        // Same-sense coplanar but no overlap → outside the other solid
        return Some(CoplanarAction::Or);
    }

    None
}

/// Merge multiple coplanar faces from a shell into a single 2D union polygon.
///
/// Projects each face to 2D, then unions them progressively using iOverlay.
fn merge_other_faces<C, S: ShapeOpsSurface>(
    shell: &Shell<Point3, C, S>,
    face_indices: &[usize],
    coord: &CoplanarCoordSystem,
    min_area: f64,
) -> Option<Vec<Vec<[f64; 2]>>> {
    let shell_faces: Vec<_> = shell.iter().collect();
    let mut merged: Option<Vec<Vec<[f64; 2]>>> = None;

    for &idx in face_indices {
        if idx >= shell_faces.len() {
            continue;
        }
        let face = &shell_faces[idx];
        let (outer, holes) = face_to_2d_contours(face, coord)?;
        if signed_area_2d(&outer).abs() < min_area {
            continue;
        }
        let shape = build_overlay_shape(&outer, &holes);

        merged = Some(match merged {
            None => shape,
            Some(existing) => {
                let union: Vec<Vec<Vec<[f64; 2]>>> =
                    existing.overlay(&shape, OverlayRule::Union, FillRule::EvenOdd);
                // Flatten: union returns Vec<shape> where each shape is Vec<contour>
                // We merge them into a single shape list
                union.into_iter().flatten().collect()
            }
        });
    }

    merged
}

use super::coplanar::CoplanarAction;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_area_ccw_square() {
        // CCW square should have positive area
        let sq = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let area = signed_area_2d(&sq);
        assert!(
            area > 0.0,
            "CCW square area should be positive, got {}",
            area
        );
        assert!(
            (area - 1.0).abs() < 1e-10,
            "Area should be 1.0, got {}",
            area
        );
    }

    #[test]
    fn test_signed_area_cw_square() {
        // CW square should have negative area
        let sq = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let area = signed_area_2d(&sq);
        assert!(
            area < 0.0,
            "CW square area should be negative, got {}",
            area
        );
    }

    #[test]
    fn test_contour_centroid() {
        let sq = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let c = contour_centroid(&sq);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlay_two_overlapping_squares() {
        // Two overlapping squares: [0,2]x[0,2] and [1,3]x[0,2]
        // Intersection should be [1,2]x[0,2] (area 2)
        // Diff0 (A\B) should be [0,1]x[0,2] (area 2)
        // Diff1 (B\A) should be [2,3]x[0,2] (area 2)
        let sq0 = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let sq1 = vec![[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]];

        let shape0 = vec![sq0];
        let shape1 = vec![sq1];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);
        let diff0: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Difference, FillRule::EvenOdd);
        let diff1: Vec<Vec<Vec<[f64; 2]>>> =
            shape1.overlay(&shape0, OverlayRule::Difference, FillRule::EvenOdd);

        // Intersection should have 1 shape with area ~2
        assert_eq!(intersection.len(), 1, "Should have 1 intersection shape");
        let int_area = signed_area_2d(&intersection[0][0]).abs();
        assert!(
            (int_area - 2.0).abs() < 0.1,
            "Intersection area should be ~2, got {}",
            int_area
        );

        // Diff0 should have 1 shape with area ~2
        assert_eq!(diff0.len(), 1, "Should have 1 diff0 shape");
        let d0_area = signed_area_2d(&diff0[0][0]).abs();
        assert!(
            (d0_area - 2.0).abs() < 0.1,
            "Diff0 area should be ~2, got {}",
            d0_area
        );

        // Diff1 should have 1 shape with area ~2
        assert_eq!(diff1.len(), 1, "Should have 1 diff1 shape");
        let d1_area = signed_area_2d(&diff1[0][0]).abs();
        assert!(
            (d1_area - 2.0).abs() < 0.1,
            "Diff1 area should be ~2, got {}",
            d1_area
        );
    }

    #[test]
    fn test_overlay_contained_square() {
        // Small square fully inside large square
        // Large: [0,10]x[0,10], Small: [2,5]x[2,5]
        let large = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let small = vec![[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]];

        let shape0 = vec![large];
        let shape1 = vec![small];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);

        // Intersection = small square (area 9)
        assert_eq!(intersection.len(), 1);
        let int_area = signed_area_2d(&intersection[0][0]).abs();
        assert!(
            (int_area - 9.0).abs() < 0.1,
            "Intersection area should be ~9, got {}",
            int_area
        );

        // A \ B should be large minus small (area 91) — produces shape with hole
        let diff0: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Difference, FillRule::EvenOdd);
        assert!(!diff0.is_empty(), "Diff should produce shapes");
        // Net area = outer areas - hole areas
        let total_diff_area: f64 = diff0
            .iter()
            .map(|shape| {
                shape
                    .iter()
                    .map(|c| signed_area_2d(c).abs())
                    .enumerate()
                    .map(|(i, a)| if i == 0 { a } else { -a })
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (total_diff_area - 91.0).abs() < 0.5,
            "Diff area should be ~91, got {}",
            total_diff_area
        );
    }

    #[test]
    fn test_overlay_abutting_squares() {
        // Two abutting squares sharing edge at x=1
        // A: [0,1]x[0,1], B: [1,2]x[0,1]
        let sq0 = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let sq1 = vec![[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];

        let shape0 = vec![sq0.clone()];
        let shape1 = vec![sq1.clone()];

        // Union of abutting squares should be one rectangle
        let union: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Union, FillRule::EvenOdd);
        assert_eq!(
            union.len(),
            1,
            "Union of abutting squares should be 1 shape"
        );
        let union_area = signed_area_2d(&union[0][0]).abs();
        assert!(
            (union_area - 2.0).abs() < 0.1,
            "Union area should be ~2, got {}",
            union_area
        );

        // Intersection should be empty or near-zero (shared edge only)
        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);
        let int_area: f64 = intersection
            .iter()
            .flat_map(|shape| shape.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();
        assert!(
            int_area < 0.01,
            "Intersection of abutting squares should be ~0, got {}",
            int_area
        );
    }

    #[test]
    fn test_overlay_identical_squares() {
        // Two identical squares — intersection = the square itself
        let sq = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        let shape0 = vec![sq.clone()];
        let shape1 = vec![sq.clone()];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);
        assert_eq!(intersection.len(), 1);
        let int_area = signed_area_2d(&intersection[0][0]).abs();
        assert!(
            (int_area - 1.0).abs() < 0.1,
            "Intersection of identical squares should be ~1, got {}",
            int_area
        );

        // Difference should be empty
        let diff: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Difference, FillRule::EvenOdd);
        let diff_area: f64 = diff
            .iter()
            .flat_map(|shape| shape.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();
        assert!(
            diff_area < 0.01,
            "Difference of identical squares should be ~0, got {}",
            diff_area
        );
    }

    #[test]
    fn test_overlay_with_hole() {
        // Square with a hole: outer [0,10]x[0,10], hole [3,7]x[3,7]
        // Clip: [2,5]x[2,5]
        // The intersection should be the clip minus the hole part
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]; // CW for hole
        let clip = vec![[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]];

        let shape0 = vec![outer, hole];
        let shape1 = vec![clip];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            shape0.overlay(&shape1, OverlayRule::Intersect, FillRule::EvenOdd);

        // The intersection should exclude the hole region
        let total_area: f64 = intersection
            .iter()
            .flat_map(|shape| shape.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();

        // Clip area = 9, hole overlap with clip = [3,5]x[3,5] = 4
        // So intersection area should be 9 - 4 = 5
        assert!(
            (total_area - 5.0).abs() < 0.5,
            "Intersection with hole should be ~5, got {}",
            total_area
        );
    }

    /// Test overlay classification logic: small square fully inside large square → And
    #[test]
    fn test_classify_overlay_contained_fragment() {
        // Face: small 3x3 square at (1,1)-(4,4)
        // Other shell face: large 10x10 square
        // The face is fully inside → intersection area = face area → And
        let face_outer = vec![[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]];
        let other_outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

        let face_shape = vec![face_outer.clone()];
        let other_shape = vec![other_outer];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            face_shape.overlay(&other_shape, OverlayRule::Intersect, FillRule::EvenOdd);
        let int_area: f64 = intersection
            .iter()
            .flat_map(|s| s.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();

        let face_area = signed_area_2d(&face_outer).abs();
        let min_area = 0.05 * 0.05; // tol² for tol=0.05

        // Intersection area should equal face area (face is fully contained)
        assert!(
            int_area > min_area,
            "Contained fragment should have intersection area > min_area"
        );
        assert!(
            (int_area - face_area).abs() < 0.1,
            "Intersection should equal face area (~9), got {}",
            int_area
        );
    }

    /// Test overlay classification logic: disjoint same-plane squares → Or
    #[test]
    fn test_classify_overlay_disjoint_fragment() {
        // Face: square at (0,0)-(1,1)
        // Other: square at (5,5)-(6,6)
        // No overlap → intersection area = 0 → Or (same-sense, no overlap)
        let face_outer = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let other_outer = vec![[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]];

        let face_shape = vec![face_outer];
        let other_shape = vec![other_outer];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            face_shape.overlay(&other_shape, OverlayRule::Intersect, FillRule::EvenOdd);
        let int_area: f64 = intersection
            .iter()
            .flat_map(|s| s.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();

        let min_area = 0.05 * 0.05;
        assert!(
            int_area < min_area,
            "Disjoint fragments should have near-zero intersection, got {}",
            int_area
        );
    }

    /// Test overlay classification: face vs merged multi-face union → And
    #[test]
    fn test_classify_overlay_multi_face_merge() {
        // Face: 4x4 square at (1,0)-(5,4)
        // Other shell has 3 faces: [0,3]x[0,2], [2,5]x[0,2], [0,5]x[2,4]
        // Their union covers [0,5]x[0,4] (fully contains the face)
        let face_outer = vec![[1.0, 0.0], [5.0, 0.0], [5.0, 4.0], [1.0, 4.0]];

        let other0 = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]];
        let other1 = vec![[2.0, 0.0], [5.0, 0.0], [5.0, 2.0], [2.0, 2.0]];
        let other2 = vec![[0.0, 2.0], [5.0, 2.0], [5.0, 4.0], [0.0, 4.0]];

        // Merge the 3 other faces via progressive union
        let shape_a = vec![other0];
        let shape_b = vec![other1];
        let union_ab: Vec<Vec<Vec<[f64; 2]>>> =
            shape_a.overlay(&shape_b, OverlayRule::Union, FillRule::EvenOdd);
        let merged_ab: Vec<Vec<[f64; 2]>> = union_ab.into_iter().flatten().collect();

        let shape_c = vec![other2];
        let union_abc: Vec<Vec<Vec<[f64; 2]>>> =
            merged_ab.overlay(&shape_c, OverlayRule::Union, FillRule::EvenOdd);
        let merged: Vec<Vec<[f64; 2]>> = union_abc.into_iter().flatten().collect();

        // Verify merged area is 20 (5x4)
        let merged_area: f64 = merged.iter().map(|c| signed_area_2d(c).abs()).sum();
        assert!(
            merged_area > 15.0,
            "Merged area should be ~20, got {}",
            merged_area
        );

        // Now compute intersection of face vs merged
        let face_shape = vec![face_outer.clone()];
        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            face_shape.overlay(&merged, OverlayRule::Intersect, FillRule::EvenOdd);
        let int_area: f64 = intersection
            .iter()
            .flat_map(|s| s.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();

        let face_area = signed_area_2d(&face_outer).abs();
        let min_area = 0.05 * 0.05;
        assert!(
            int_area > min_area,
            "Multi-face merged overlap should be > min_area"
        );
        assert!(
            (int_area - face_area).abs() < 1.0,
            "Intersection should be close to face area (~16), got {}",
            int_area
        );
    }

    /// Test overlay classification: partial overlap produces correct intersection area
    #[test]
    fn test_classify_overlay_partial_overlap() {
        // Face: [0,4]x[0,4] (area 16)
        // Other: [2,6]x[0,4] (area 16)
        // Overlap: [2,4]x[0,4] (area 8)
        let face_outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let other_outer = vec![[2.0, 0.0], [6.0, 0.0], [6.0, 4.0], [2.0, 4.0]];

        let face_shape = vec![face_outer];
        let other_shape = vec![other_outer];

        let intersection: Vec<Vec<Vec<[f64; 2]>>> =
            face_shape.overlay(&other_shape, OverlayRule::Intersect, FillRule::EvenOdd);
        let int_area: f64 = intersection
            .iter()
            .flat_map(|s| s.first())
            .map(|c| signed_area_2d(c).abs())
            .sum();

        let min_area = 0.05 * 0.05;
        assert!(
            int_area > min_area,
            "Partial overlap should have intersection > min_area"
        );
        assert!(
            (int_area - 8.0).abs() < 0.5,
            "Partial overlap area should be ~8, got {}",
            int_area
        );
    }
}
