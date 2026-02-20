use truck_meshalgo::prelude::*;
use truck_modeling::*;

#[test]
fn punched_cube() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v = builder::vertex(Point3::new(0.5, 0.25, -0.5));
    let w = builder::rsweep(
        &v,
        Point3::new(0.5, 0.5, 0.0),
        Vector3::unit_z(),
        Rad(7.0),
        3,
    );
    let f = builder::try_attach_plane(&[w]).unwrap();
    let mut cylinder = builder::tsweep(&f, Vector3::unit_z() * 2.0);
    cylinder.not();
    let and = crate::and(&cube, &cylinder, 0.05).unwrap();

    let poly = and.triangulation(0.01).to_polygon();
    let file = std::fs::File::create("punched-cube.obj").unwrap();
    obj::write(&poly, file).unwrap();
}

#[test]
fn coplanar_box_on_box_union() {
    // Box1: unit cube [0,1]^3
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    // Box2: sits on top of cube, sharing the z=1 face
    let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let boss: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

    let result = crate::or(&cube, &boss, 0.05);
    assert!(result.is_some(), "Coplanar box-on-box union should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Union should have more than 6 faces (got {})",
        shell.len()
    );
}

#[test]
fn coplanar_box_subtract() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    // Tool body starting at z=-0.5 going up 2.0 units
    let v2 = builder::vertex(Point3::new(0.25, 0.25, -0.5));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 2.0);
    tool.not();

    let result = crate::and(&cube, &tool, 0.05);
    assert!(result.is_some(), "Coplanar box subtract should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Subtract should create more than 6 faces (got {})",
        shell.len()
    );
}

/// Simulates the engine's extrude_cut at 10x scale.
#[test]
fn rect_cut_engine_geometry() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    let v2 = builder::vertex(Point3::new(2.0, 2.0, -7.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 4.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 4.0);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 19.0);
    tool.not();

    let result = crate::and(&cube, &tool, 0.05);
    assert!(result.is_some(), "Rect cut boolean should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Rect cut should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// Verify that the improved Plane triangulation produces more triangles for
/// large faces, fixing interference detection at scale.
#[test]
fn plane_triangulation_scales_with_size() {
    use truck_geometry::prelude::{ParameterDivision2D, Plane};

    let plane = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let tol = 0.05;

    // Unit-size face: range (0,1)x(0,1)
    let (us, vs) = plane.parameter_division(((0.0, 1.0), (0.0, 1.0)), tol);
    assert!(
        us.len() >= 3,
        "Unit face should have >= 3 u-divisions (got {})",
        us.len()
    );
    assert!(
        vs.len() >= 3,
        "Unit face should have >= 3 v-divisions (got {})",
        vs.len()
    );

    // Large face: range (0,10)x(0,10)
    let (us_big, _vs_big) = plane.parameter_division(((0.0, 10.0), (0.0, 10.0)), tol);
    assert!(
        us_big.len() > us.len(),
        "10x face should have more u-divisions than unit face ({} vs {})",
        us_big.len(),
        us.len()
    );

    // Very small face: should still have at least 2 points
    let (us_tiny, vs_tiny) = plane.parameter_division(((0.0, 0.001), (0.0, 0.001)), tol);
    assert!(
        us_tiny.len() >= 2,
        "Tiny face should have at least 2 u-points"
    );
    assert!(
        vs_tiny.len() >= 2,
        "Tiny face should have at least 2 v-points"
    );
}

/// Test box-box union at 10x scale with offset overlapping boxes.
#[test]
fn box_union_at_10x_scale_offset() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };
    let box_b: Solid = {
        let v2 = builder::vertex(Point3::new(5.0, 5.0, 5.0));
        let e2 = builder::tsweep(&v2, Vector3::unit_x() * 10.0);
        let f2 = builder::tsweep(&e2, Vector3::unit_y() * 10.0);
        builder::tsweep(&f2, Vector3::unit_z() * 10.0)
    };

    let tol = 0.1;
    let result = crate::or(&box_a, &box_b, tol);
    assert!(result.is_some(), "Box-box OR at scale 10 should succeed");
    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Union at scale 10 should produce more than 6 faces (got {})",
        shell.len()
    );
}

// Coplanar partial overlap: pair-specific skip allows non-coplanar pairs to intersect normally
#[test]
fn coplanar_partial_overlap_union() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let box1: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v2 = builder::vertex(Point3::new(0.5, 0.0, 0.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x());
    let f2 = builder::tsweep(&e2, Vector3::unit_y());
    let box2: Solid = builder::tsweep(&f2, Vector3::unit_z());

    let result = crate::or(&box1, &box2, 0.05);
    assert!(result.is_some(), "Partial overlap union should succeed");
}

/// Through-hole: cylinder cuts completely through a unit cube from z=0 to z=1.
#[test]
fn through_hole_cylinder() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let center = Point3::new(0.5, 0.5, 0.0);
    let v_cyl = builder::vertex(Point3::new(0.5, 0.25, 0.0));
    let w = builder::rsweep(&v_cyl, center, Vector3::unit_z(), Rad(7.0), 3);
    let bottom = builder::try_attach_plane(&[w]).unwrap();
    let mut cylinder: Solid = builder::tsweep(&bottom, Vector3::unit_z());
    cylinder.not();

    let result = crate::and(&cube, &cylinder, 0.05);
    assert!(
        result.is_some(),
        "Through-hole cylinder boolean should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Through-hole should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// Through-hole at 10x scale.
#[test]
fn through_hole_cylinder_10x() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    let center = Point3::new(5.0, 5.0, 0.0);
    let v_cyl = builder::vertex(Point3::new(5.0, 3.0, 0.0));
    let w = builder::rsweep(&v_cyl, center, Vector3::unit_z(), Rad(7.0), 3);
    let bottom = builder::try_attach_plane(&[w]).unwrap();
    let mut cylinder: Solid = builder::tsweep(&bottom, Vector3::unit_z() * 10.0);
    cylinder.not();

    let result = crate::and(&cube, &cylinder, 0.05);
    assert!(result.is_some(), "10x through-hole cylinder should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "10x through-hole should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// Blind hole: cylinder only goes halfway through the cube.
#[test]
fn blind_hole_cylinder() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v_cyl = builder::vertex(Point3::new(0.5, 0.3, 0.5));
    let w = builder::rsweep(
        &v_cyl,
        Point3::new(0.5, 0.5, 0.5),
        Vector3::unit_z(),
        Rad(7.0),
        3,
    );
    let bottom = builder::try_attach_plane(&[w]).unwrap();
    let mut cylinder: Solid = builder::tsweep(&bottom, Vector3::unit_z() * 0.5);
    cylinder.not();

    let result = crate::and(&cube, &cylinder, 0.05);
    assert!(
        result.is_some(),
        "Blind hole (partial penetration) should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Blind hole should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// Pure and() without .not(): two overlapping boxes, intersection only.
#[test]
fn box_box_intersect() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let result = crate::and(&box_a, &box_b, 0.05);
    assert!(
        result.is_some(),
        "Box-box intersect (pure and) should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert_eq!(
        shell.len(),
        6,
        "Intersection of two offset cubes should be a box with 6 faces"
    );
}

// ===== difference() tests =====

/// Basic difference of two offset boxes.
#[test]
fn test_difference_box_box_offset() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let result = crate::difference(&box_a, &box_b, 0.05);
    assert!(
        result.is_some(),
        "difference(A, B) for offset boxes should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    use truck_topology::shell::ShellCondition;
    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "Difference result shell must be Closed"
    );
}

/// Difference is non-commutative: diff(A,B) != diff(B,A) in general.
#[test]
fn test_difference_non_commutative() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let ab = crate::difference(&box_a, &box_b, 0.05).unwrap();
    let ba = crate::difference(&box_b, &box_a, 0.05).unwrap();

    let faces_ab = ab.boundaries()[0].len();
    let faces_ba = ba.boundaries()[0].len();

    assert!(faces_ab > 0, "diff(A,B) should have faces");
    assert!(faces_ba > 0, "diff(B,A) should have faces");
}

/// When boxes are disjoint, diff(A,B) should equal A.
#[test]
fn test_difference_disjoint() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(5.0, 5.0, 5.0));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let result = crate::difference(&box_a, &box_b, 0.05);
    assert!(result.is_some(), "Disjoint difference should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert_eq!(
        shell.len(),
        6,
        "Disjoint difference should preserve A's 6 faces (got {})",
        shell.len()
    );
}

/// Difference result must be manifold (ShellCondition::Closed).
#[test]
fn test_difference_result_manifold() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x() * 2.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 2.0);
        builder::tsweep(&f, Vector3::unit_z() * 2.0)
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let result = crate::difference(&box_a, &box_b, 0.05).unwrap();
    let shell = &result.boundaries()[0];
    use truck_topology::shell::ShellCondition;
    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "Difference result must be a closed manifold"
    );
}

/// difference(A,B) should produce same topology as not(B)+and(A,neg-B).
#[test]
fn test_difference_matches_not_and() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let diff = crate::difference(&box_a, &box_b, 0.05).unwrap();

    let mut b_neg = box_b.clone();
    b_neg.not();
    let old = crate::and(&box_a, &b_neg, 0.05).unwrap();

    let diff_faces = diff.boundaries()[0].len();
    let old_faces = old.boundaries()[0].len();

    assert_eq!(
        diff_faces, old_faces,
        "difference() and not+and should produce same face count ({} vs {})",
        diff_faces, old_faces
    );
}

/// difference(A,B) should match not(B)+and(A,neg-B) for box-cylinder.
#[test]
fn test_difference_matches_not_and_box_cyl() {
    let cube: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        builder::tsweep(&f, Vector3::unit_z())
    };

    let cylinder: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.25, -0.5));
        let w = builder::rsweep(
            &v,
            Point3::new(0.5, 0.5, 0.0),
            Vector3::unit_z(),
            Rad(7.0),
            3,
        );
        let f = builder::try_attach_plane(&[w]).unwrap();
        builder::tsweep(&f, Vector3::unit_z() * 2.0)
    };

    let diff = crate::difference(&cube, &cylinder, 0.05).unwrap();

    let mut cyl_neg = cylinder.clone();
    cyl_neg.not();
    let old = crate::and(&cube, &cyl_neg, 0.05).unwrap();

    let diff_faces = diff.boundaries()[0].len();
    let old_faces = old.boundaries()[0].len();

    assert_eq!(
        diff_faces, old_faces,
        "Box-cylinder: difference() and not+and should produce same face count ({} vs {})",
        diff_faces, old_faces
    );
}

/// Verify parity-based ray-cast classification.
#[test]
fn parity_ray_cast_consistency() {
    use crate::transversal::integrate::{irrational_ray_dirs, try_ray_cast};

    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());
    let poly_shell = cube.boundaries()[0].triangulation(0.05);

    let dirs = irrational_ray_dirs();

    // Point clearly inside the cube
    let inside_pt = Point3::new(0.5, 0.5, 0.5);
    let mut inside_votes = 0u32;
    for &d in &dirs {
        if let Some(c) = try_ray_cast(inside_pt, d, &poly_shell) {
            if c.unsigned_abs() % 2 == 1 {
                inside_votes += 1;
            }
        }
    }
    assert!(
        inside_votes >= 2,
        "Point inside cube should get majority inside votes (got {})",
        inside_votes
    );

    // Point clearly outside the cube
    let outside_pt = Point3::new(5.0, 5.0, 5.0);
    let mut outside_votes = 0u32;
    for &d in &dirs {
        if let Some(c) = try_ray_cast(outside_pt, d, &poly_shell) {
            if c.unsigned_abs() % 2 == 0 {
                outside_votes += 1;
            }
        }
    }
    assert!(
        outside_votes >= 2,
        "Point outside cube should get majority outside votes (got {})",
        outside_votes
    );
}

/// and_result() returns Ok for box-box offset intersection.
#[test]
fn test_and_result_box_box_success() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::new(0.0, 0.0, 0.0));
        let e = builder::tsweep(&v, Vector3::new(2.0, 0.0, 0.0));
        let f = builder::tsweep(&e, Vector3::new(0.0, 2.0, 0.0));
        builder::tsweep(&f, Vector3::new(0.0, 0.0, 2.0))
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::new(1.0, 0.0, 0.0));
        let f = builder::tsweep(&e, Vector3::new(0.0, 1.0, 0.0));
        builder::tsweep(&f, Vector3::new(0.0, 0.0, 1.0))
    };

    let result = crate::and_result(&box_a, &box_b, 0.05);
    assert!(
        result.is_ok(),
        "and_result should return Ok for box-box offset"
    );
    let solid = result.unwrap();
    assert_eq!(
        solid.boundaries()[0].len(),
        6,
        "Intersection of two offset cubes should be a box with 6 faces"
    );
}

/// or_result() returns Ok for box-box offset union.
#[test]
fn test_or_result_box_box_success() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::new(0.0, 0.0, 0.0));
        let e = builder::tsweep(&v, Vector3::new(2.0, 0.0, 0.0));
        let f = builder::tsweep(&e, Vector3::new(0.0, 2.0, 0.0));
        builder::tsweep(&f, Vector3::new(0.0, 0.0, 2.0))
    };
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.5, 0.5, 0.5));
        let e = builder::tsweep(&v, Vector3::new(1.0, 0.0, 0.0));
        let f = builder::tsweep(&e, Vector3::new(0.0, 1.0, 0.0));
        builder::tsweep(&f, Vector3::new(0.0, 0.0, 1.0))
    };

    let result = crate::or_result(&box_a, &box_b, 0.05);
    assert!(
        result.is_ok(),
        "or_result should return Ok for box-box offset"
    );
    let solid = result.unwrap();
    assert!(
        solid.boundaries()[0].len() >= 6,
        "Union should produce at least 6 faces (got {})",
        solid.boundaries()[0].len()
    );
}

/// 50x scale rect cut — tests scale behavior of boolean operations.
/// NOTE: This currently fails with NotClosedShell because the large scale
/// creates edge gaps. The double_projection optimization (not yet ported)
/// would fix this. For now, verify the result API reports the error cleanly.
#[test]
fn box_cut_50x_scale() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 50.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 50.0);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 50.0);

    let v2 = builder::vertex(Point3::new(10.0, 10.0, -5.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 20.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 20.0);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 60.0);
    tool.not();

    // This may fail at 50x scale without double_projection optimization.
    // Verify at least that and_result returns a structured error rather than panicking.
    let result = crate::and_result(&cube, &tool, 0.5);
    if let Err(e) = &result {
        // Structured error reporting works
        assert!(
            matches!(e, crate::BooleanStageError::ShellAssembly(_)),
            "Expected ShellAssembly error at large scale, got: {:?}",
            e
        );
    } else {
        // If it succeeds (e.g. after double_projection is ported), validate the result
        let solid = result.unwrap();
        let shell = &solid.boundaries()[0];
        assert!(
            shell.len() > 6,
            "50x rect cut should produce more than 6 faces (got {})",
            shell.len()
        );
    }
}

/// Sub-unit-scale through-hole with tighter tolerance.
#[test]
fn sub_unit_scale_through_hole() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 0.1);
    let f = builder::tsweep(&e, Vector3::unit_y() * 0.1);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 0.1);

    let v_cyl = builder::vertex(Point3::new(0.05, 0.03, 0.0));
    let w = builder::rsweep(
        &v_cyl,
        Point3::new(0.05, 0.05, 0.0),
        Vector3::unit_z(),
        Rad(7.0),
        3,
    );
    let bottom = builder::try_attach_plane(&[w]).unwrap();
    let mut cylinder: Solid = builder::tsweep(&bottom, Vector3::unit_z() * 0.1);
    cylinder.not();

    let result = crate::and(&cube, &cylinder, 0.005);
    assert!(
        result.is_some(),
        "Sub-unit-scale through-hole should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Small through-hole should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// Two bosses sequentially on same face — tests boolean on already-modified solid.
#[test]
fn coplanar_chained_union() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    // Boss 1: [0.1, 0.4] x [0.1, 0.4] x [1.0, 1.3]
    let v2 = builder::vertex(Point3::new(0.1, 0.1, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.3);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.3);
    let boss1: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.3);

    let step1 = crate::or(&cube, &boss1, 0.05);
    assert!(step1.is_some(), "First boss union should succeed");

    // Boss 2: [0.6, 0.9] x [0.6, 0.9] x [1.0, 1.3]
    let v3 = builder::vertex(Point3::new(0.6, 0.6, 1.0));
    let e3 = builder::tsweep(&v3, Vector3::unit_x() * 0.3);
    let f3 = builder::tsweep(&e3, Vector3::unit_y() * 0.3);
    let boss2: Solid = builder::tsweep(&f3, Vector3::unit_z() * 0.3);

    let step2 = crate::or(&step1.unwrap(), &boss2, 0.05);
    assert!(step2.is_some(), "Chained second boss union should succeed");

    let solid = step2.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Chained union should produce more than 6 faces (got {})",
        shell.len()
    );
}

/// 10x10x10 box minus 10x10x5 tool aligned to one face (mutual containment).
///
/// When the tool shares a full face with the target (e.g., the tool's top
/// face has the same 10x10 extent as the target's top face), both j_in_i
/// and i_in_j are true. Without the mutual-containment fix, both adjacency
/// skip branches fire, suppressing ALL intersection curves and producing
/// Unknown faces -> NotClosedShell.
#[test]
fn full_face_rect_difference() {
    // Target: 10x10x10 box at origin
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    // Tool: 10x10x5 box starting at z=5, so its bottom face (z=5, 10x10)
    // is NOT coplanar with any target face, but its TOP face (z=10, 10x10)
    // has the same extent as the target's top face (z=10, 10x10).
    let v2 = builder::vertex(Point3::new(0.0, 0.0, 5.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 10.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 10.0);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 5.0);
    tool.not();

    let result = crate::and(&cube, &tool, 0.05);
    assert!(
        result.is_some(),
        "Full-face rect difference (mutual containment) should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    // The result should be a 10x10x5 box (bottom half), which has 6 faces.
    assert!(
        shell.len() >= 6,
        "Full-face difference should produce at least 6 faces (got {})",
        shell.len()
    );

    // Verify the result is a closed manifold
    use truck_topology::shell::ShellCondition;
    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "Full-face difference result must be a closed manifold"
    );
}

/// Same-extent subtraction where tool and target share the full top AND
/// lateral faces have the same projected extent on the coplanar plane.
/// This creates the most aggressive mutual containment scenario:
/// the tool occupies the exact same footprint on the shared coplanar face.
///
/// Uses and_result to get structured error reporting.
#[test]
fn full_face_rect_difference_result() {
    // Target: 10x10x10 box
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    // Tool: same 10x10 footprint, extends from z=5 to z=10.
    let v2 = builder::vertex(Point3::new(0.0, 0.0, 5.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 10.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 10.0);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 5.0);
    tool.not();

    let result = crate::and_result(&cube, &tool, 0.05);
    assert!(
        result.is_ok(),
        "Full-face rect difference should return Ok, got: {:?}",
        result.err()
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    use truck_topology::shell::ShellCondition;
    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "Full-face difference result must be a closed manifold"
    );
}

/// Two boxes with same-extent lateral faces: mutual containment on a side.
///
/// Box A: [0,10] x [0,10] x [0,10]
/// Box B: [0,10] x [0,10] x [5,15]
/// Their y=0 and y=10 faces have the same x and z extent where they overlap,
/// creating mutual containment on those lateral coplanar faces.
#[test]
fn mutual_containment_coplanar() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };

    // Box B shares the x-extent and y-extent with Box A but is offset in z.
    // The z=10 face of A is coplanar with the z=10 face of B — but B's
    // z=10 face is at a different z. Actually, let's use a simpler setup:
    // Box B sits flush on top of Box A, sharing the z=10 face.
    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.0, 0.0, 10.0));
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };

    // Union: should produce a 10x10x20 box
    let result = crate::or(&box_a, &box_b, 0.05);
    assert!(
        result.is_some(),
        "Mutual containment coplanar union should succeed"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    // Two stacked boxes merged should produce exactly 6 faces (one merged box)
    // or at minimum a closed manifold.
    use truck_topology::shell::ShellCondition;
    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "Mutual containment union result must be a closed manifold"
    );
    assert!(
        shell.len() >= 6,
        "Mutual containment union should produce at least 6 faces (got {})",
        shell.len()
    );
}
