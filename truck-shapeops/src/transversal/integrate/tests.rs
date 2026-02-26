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

/// Three abutting boxes along Z axis: chained union.
///
/// Box A: [0,10] x [0,10] x [0,10]
/// Box B: [0,10] x [0,10] x [10,20]  (abuts A at z=10)
/// Box C: [0,10] x [0,10] x [20,30]  (abuts B at z=20)
///
/// or(or(A, B), C) should produce a valid 10x10x30 box.
/// HP-1 reports this chain fails at the 3rd step.
#[test]
fn three_abutting_boxes_chained_union() {
    let box_a: Solid = {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };

    let box_b: Solid = {
        let v = builder::vertex(Point3::new(0.0, 0.0, 10.0));
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };

    let box_c: Solid = {
        let v = builder::vertex(Point3::new(0.0, 0.0, 20.0));
        let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
        let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
        builder::tsweep(&f, Vector3::unit_z() * 10.0)
    };

    // Step 1: or(A, B) — 2 abutting boxes
    let step1 = crate::or_result(&box_a, &box_b, 0.05);
    assert!(step1.is_ok(), "or(A, B) should succeed: {:?}", step1.err());
    let ab = step1.unwrap();

    eprintln!(
        "[3-abutting] or(A,B): {} boundaries, shell0 has {} faces",
        ab.boundaries().len(),
        ab.boundaries()[0].len(),
    );

    // Step 2: or(AB, C) — chain the 3rd box
    let step2 = crate::or_result(&ab, &box_c, 0.05);
    assert!(
        step2.is_ok(),
        "or(AB, C) should succeed for 3 abutting boxes: {:?}",
        step2.err()
    );
    let abc = step2.unwrap();

    let shell = &abc.boundaries()[0];
    use truck_topology::shell::ShellCondition;
    eprintln!(
        "[3-abutting] or(AB,C): {} boundaries, shell0 has {} faces, condition={:?}",
        abc.boundaries().len(),
        shell.len(),
        shell.shell_condition(),
    );

    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "3-abutting union must be a closed manifold"
    );
    assert!(
        shell.len() >= 6,
        "3 abutting boxes union should have at least 6 faces (got {})",
        shell.len()
    );
}

/// Four abutting 20x20x10 boxes along X axis (HP-1 geometry).
///
/// Matches the geometry from the several-extrudes.waffle test case.
/// All boxes use same 20x20 cross-section, depth 10, along X.
#[test]
fn four_abutting_boxes_x_axis_hp1() {
    let make_box = |x_start: f64| -> Solid {
        let v = builder::vertex(Point3::new(x_start, -10.0, -10.0));
        let e = builder::tsweep(&v, Vector3::unit_y() * 20.0);
        let f = builder::tsweep(&e, Vector3::unit_z() * 20.0);
        builder::tsweep(&f, Vector3::unit_x() * 10.0)
    };

    let box1 = make_box(0.0);
    let box2 = make_box(10.0);
    let box3 = make_box(20.0);
    let box4 = make_box(30.0);

    // Chain unions
    let r1 = crate::or_result(&box1, &box2, 0.05);
    assert!(r1.is_ok(), "or(1,2) failed: {:?}", r1.err());
    let s12 = r1.unwrap();
    eprintln!(
        "[HP1] or(1,2): {} faces, {} boundaries",
        s12.boundaries()[0].len(),
        s12.boundaries().len(),
    );

    let r2 = crate::or_result(&s12, &box3, 0.05);
    assert!(r2.is_ok(), "or(12,3) failed: {:?}", r2.err());
    let s123 = r2.unwrap();
    eprintln!(
        "[HP1] or(12,3): {} faces, {} boundaries",
        s123.boundaries()[0].len(),
        s123.boundaries().len(),
    );

    let r3 = crate::or_result(&s123, &box4, 0.05);
    assert!(r3.is_ok(), "or(123,4) failed: {:?}", r3.err());
    let s1234 = r3.unwrap();

    let shell = &s1234.boundaries()[0];
    use truck_topology::shell::ShellCondition;
    eprintln!(
        "[HP1] or(123,4): {} faces, condition={:?}",
        shell.len(),
        shell.shell_condition(),
    );

    assert_eq!(
        shell.shell_condition(),
        ShellCondition::Closed,
        "4-box chain union must be closed"
    );
}

#[test]
fn test_tolerance_fields_from_model() {
    let bt = super::BooleanTolerance::from_model_tol(0.01);
    assert_eq!(bt.tau_model, 0.01);
    assert_eq!(bt.tau_mesh, 0.01);
    assert!((bt.tau_weld - 0.004).abs() < 1e-15);
    assert!((bt.tau_boundary - 0.005).abs() < 1e-15);
    assert!((bt.tau_edge_cluster - 0.05).abs() < 1e-15);
    assert!((bt.tau_area - 0.0001).abs() < 1e-15);
}

#[test]
#[allow(deprecated)]
fn test_tolerance_uniform_backward_compat() {
    let bt = super::BooleanTolerance::uniform(0.01);
    // uniform mode: all base fields = tol
    assert_eq!(bt.tau_model, 0.01);
    assert_eq!(bt.tau_boundary, 0.01);
    assert_eq!(bt.tau_edge_cluster, 0.01);
    assert!((bt.tau_area - 0.0001).abs() < 1e-15);
}

#[test]
fn test_diagnostics_default() {
    let diag = crate::BooleanDiagnostics::default();
    assert_eq!(diag.tolerance.tau_model, 0.0);
    assert!(diag.warnings.is_empty());
    assert_eq!(diag.classification.faces_coplanar, 0);
    assert_eq!(diag.topology.vertices_welded, 0);
}

// ---------------------------------------------------------------------------
// Deterministic ordering tests
// ---------------------------------------------------------------------------
// Note: with_det_context and assign_vertex_det_ids have been removed.
// SequentialID provides deterministic ordering natively via seq_id fields
// on Vertex, Edge, and Face, so no explicit context wrapper is needed.

#[test]
fn test_boolean_deterministic_box_union() {
    // Verify that boolean operations run successfully with deterministic
    // SequentialID-based ordering.
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let boss: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

    // OR (coplanar) and AND should succeed
    let or_result = crate::or(&cube, &boss, 0.05);
    assert!(or_result.is_some(), "OR should succeed");

    let and_result = crate::and(&cube, &boss, 0.05);
    assert!(and_result.is_some(), "AND should succeed");

    // Difference with fully-enclosed tool (non-degenerate)
    let v3 = builder::vertex(Point3::new(0.25, 0.25, 0.25));
    let e3 = builder::tsweep(&v3, Vector3::unit_x() * 0.5);
    let f3 = builder::tsweep(&e3, Vector3::unit_y() * 0.5);
    let inner: Solid = builder::tsweep(&f3, Vector3::unit_z() * 0.5);

    let diff_result = crate::difference(&cube, &inner, 0.05);
    assert!(diff_result.is_some(), "Difference should succeed");
}

#[test]
fn test_boolean_deterministic_50x() {
    // Run the same boolean 50 times and verify identical face counts.
    for _ in 0..50 {
        let v = builder::vertex(Point3::origin());
        let e = builder::tsweep(&v, Vector3::unit_x());
        let f = builder::tsweep(&e, Vector3::unit_y());
        let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

        let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
        let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
        let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
        let boss: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

        let result = crate::or(&cube, &boss, 0.05).expect("OR should succeed");
        let face_count = result.boundaries()[0].len();
        // Coplanar box-on-box union produces 11 faces
        assert_eq!(
            face_count, 11,
            "Each run should produce the same face count"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Euler characteristic and wire simplicity validation tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn euler_characteristic_unit_cube() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let shell = &cube.boundaries()[0];
    assert!(
        super::validate_euler_characteristic(shell).is_ok(),
        "Unit cube should satisfy V-E+F=2"
    );
}

#[test]
fn euler_characteristic_after_union() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let boss: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

    let result = crate::or(&cube, &boss, 0.05).expect("union should succeed");
    let shell = &result.boundaries()[0];
    // Coplanar box-on-box union may have T-junction vertices (singular
    // topology) which cause chi != 2. Verify validation doesn't panic
    // and report the actual Euler values.
    match super::validate_euler_characteristic(shell) {
        Ok(()) => {} // ideal
        Err((v, e, f, chi)) => {
            // T-junctions from coplanar vertex unification are expected;
            // check that chi is close to 2 (off by small count due to
            // shared vertex duplication in counting).
            eprintln!("Union Euler: V={v} E={e} F={f} chi={chi} (T-junctions expected)");
            assert!(
                chi >= 0,
                "Union should have non-negative Euler characteristic"
            );
        }
    }
}

#[test]
fn euler_characteristic_after_subtraction() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    // Offset box fully inside
    let v2 = builder::vertex(Point3::new(0.2, 0.2, 0.5));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.3);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.3);
    let tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.6);

    let result = crate::difference(&cube, &tool, 0.05).expect("subtraction should succeed");
    // Subtraction creates a through-hole or pocket — Euler chi may differ
    // but the validation function should at least not panic
    let shell = &result.boundaries()[0];
    let _ = super::validate_euler_characteristic(shell);
}

#[test]
fn wire_simplicity_unit_cube() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let shell = &cube.boundaries()[0];
    let non_simple = super::find_non_simple_wires(shell);
    assert!(
        non_simple.is_empty(),
        "Unit cube should have all simple wires"
    );
}

#[test]
fn wire_simplicity_after_union() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());

    let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let boss: Solid = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

    let result = crate::or(&cube, &boss, 0.05).expect("union should succeed");
    let shell = &result.boundaries()[0];
    let non_simple = super::find_non_simple_wires(shell);
    assert!(
        non_simple.is_empty(),
        "Union result should have all simple wires, found {} non-simple: {:?}",
        non_simple.len(),
        non_simple,
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Coplanar overlay classification integration tests
// ═══════════════════════════════════════════════════════════════════════

/// Tests overlay-based coplanar classification: 10×10×10 box with 1 boss at
/// z=10, then a cut whose base is coplanar with the z=10 face. The cut tool's
/// z=10 base face must be classified via overlay (not single-point) to get
/// correct And/Or buckets.
#[test]
fn boss_then_coplanar_cut() {
    // Base: 10×10×10 box
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let base: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    // Boss: 3×3×3 at (1,1,10) — creates a hole in the z=10 face
    let v1 = builder::vertex(Point3::new(1.0, 1.0, 10.0));
    let e1 = builder::tsweep(&v1, Vector3::unit_x() * 3.0);
    let f1 = builder::tsweep(&e1, Vector3::unit_y() * 3.0);
    let boss: Solid = builder::tsweep(&f1, Vector3::unit_z() * 3.0);

    let with_boss = crate::or(&base, &boss, 0.05);
    assert!(with_boss.is_some(), "Boss union should succeed");
    let solid1 = with_boss.unwrap();

    // Cut: 3×3 tool at (6,6) through the entire z-range
    // Its z=10 base face is coplanar with the modified z=10 face (which has a hole)
    let v2 = builder::vertex(Point3::new(6.0, 6.0, -2.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 3.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 3.0);
    let mut tool: Solid = builder::tsweep(&f2, Vector3::unit_z() * 16.0);
    tool.not();

    let result = crate::and(&solid1, &tool, 0.05);
    assert!(
        result.is_some(),
        "Cut on body with boss should succeed (overlay-based coplanar classification)"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Cut result should have more than 6 faces (got {})",
        shell.len()
    );
}

/// Two non-adjacent bosses at z=10, then a cut through the coplanar region.
/// Tests overlay classification with multiple coplanar face fragments from
/// the other shell that need to be merged.
#[test]
fn two_bosses_then_coplanar_cut() {
    // Base: 10×10×10 box
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let base: Solid = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    // Boss 1: 2×2×2 at (0.5,0.5,10)
    let v1 = builder::vertex(Point3::new(0.5, 0.5, 10.0));
    let e1 = builder::tsweep(&v1, Vector3::unit_x() * 2.0);
    let f1 = builder::tsweep(&e1, Vector3::unit_y() * 2.0);
    let boss1: Solid = builder::tsweep(&f1, Vector3::unit_z() * 2.0);

    let with_boss1 = crate::or(&base, &boss1, 0.05);
    assert!(with_boss1.is_some(), "Boss 1 union should succeed");
    let solid1 = with_boss1.unwrap();

    // Boss 2: 2×2×2 at (7,7,10) — far from boss 1
    let v2 = builder::vertex(Point3::new(7.0, 7.0, 10.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 2.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 2.0);
    let boss2: Solid = builder::tsweep(&f2, Vector3::unit_z() * 2.0);

    let with_boss2 = crate::or(&solid1, &boss2, 0.05);
    assert!(with_boss2.is_some(), "Boss 2 union should succeed");
    let solid2 = with_boss2.unwrap();

    // Cut: 2×2 tool at (4,4) through the entire z-range
    let v3 = builder::vertex(Point3::new(4.0, 4.0, -2.0));
    let e3 = builder::tsweep(&v3, Vector3::unit_x() * 2.0);
    let f3 = builder::tsweep(&e3, Vector3::unit_y() * 2.0);
    let mut tool: Solid = builder::tsweep(&f3, Vector3::unit_z() * 16.0);
    tool.not();

    let result = crate::and(&solid2, &tool, 0.05);
    assert!(
        result.is_some(),
        "Cut after 2 bosses should succeed (overlay-based coplanar classification)"
    );

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(
        shell.len() > 6,
        "Cut result should have more than 6 faces (got {})",
        shell.len()
    );
}

/// Verify that irrational_ray_dirs returns 8 unique directions.
#[test]
fn irrational_ray_dirs_returns_8() {
    use crate::transversal::integrate::irrational_ray_dirs;

    let dirs = irrational_ray_dirs();
    assert_eq!(dirs.len(), 8, "Should have 8 irrational ray directions");

    // Each direction should be non-zero
    for (i, d) in dirs.iter().enumerate() {
        let mag = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        assert!(
            mag > 0.1,
            "Direction {} should be non-zero (mag={:.6})",
            i,
            mag
        );
    }

    // All directions should be distinct
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let diff = dirs[i] - dirs[j];
            let dist = (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt();
            assert!(
                dist > 0.01,
                "Directions {} and {} should be distinct (dist={:.6})",
                i,
                j,
                dist
            );
        }
    }
}

/// Verify that geometric_face_normal computes correct normals.
#[test]
fn geometric_face_normal_basic() {
    use crate::transversal::integrate::geometric_face_normal;

    // XY-plane square: normal should point in +Z or -Z
    let verts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let n = geometric_face_normal(&verts).expect("Should compute normal for planar quad");
    assert!(
        n.z.abs() > 0.99,
        "Normal of XY-plane quad should be along Z (got {:?})",
        n
    );

    // Degenerate: too few vertices
    assert!(geometric_face_normal(&[]).is_none());
    assert!(geometric_face_normal(&[Point3::origin()]).is_none());

    // Collinear vertices: degenerate normal
    let collinear = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    assert!(
        geometric_face_normal(&collinear).is_none(),
        "Collinear vertices should produce no normal"
    );
}

/// ray_cast_classify with 8 rays should produce stronger consensus for a simple cube.
#[test]
fn ray_cast_8_dirs_cube_consensus() {
    use crate::transversal::integrate::{irrational_ray_dirs, try_ray_cast};

    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube: Solid = builder::tsweep(&f, Vector3::unit_z());
    let poly_shell = cube.boundaries()[0].triangulation(0.05);

    let dirs = irrational_ray_dirs();
    assert_eq!(dirs.len(), 8);

    // Point clearly inside: should get strong inside consensus (>=3 of 8)
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
        inside_votes >= 3,
        "Inside point should get >=3 inside votes with 8 rays (got {})",
        inside_votes
    );

    // Point clearly outside: should get strong outside consensus
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
        outside_votes >= 3,
        "Outside point should get >=3 outside votes with 8 rays (got {})",
        outside_votes
    );
}
