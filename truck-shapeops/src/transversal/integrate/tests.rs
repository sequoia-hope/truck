use truck_meshalgo::prelude::*;
use truck_modeling::*;

#[test]
fn punched_cube() {
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube = builder::tsweep(&f, Vector3::unit_z());

    let v = builder::vertex(Point3::new(0.5, 0.25, -0.5));
    let w = builder::rsweep(&v, Point3::new(0.5, 0.5, 0.0), Vector3::unit_z(), Rad(7.0));
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
    let cube = builder::tsweep(&f, Vector3::unit_z());

    // Box2: sits on top of cube, sharing the z=1 face
    // Smaller footprint: [0.25, 0.75]^2 x [1.0, 1.5]
    let v2 = builder::vertex(Point3::new(0.25, 0.25, 1.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let boss = builder::tsweep(&f2, Vector3::unit_z() * 0.5);

    let result = crate::or(&cube, &boss, 0.05);
    assert!(result.is_some(), "Coplanar box-on-box union should succeed");

    let solid = result.unwrap();
    // Union of a 1.0 cube and a 0.5x0.5x0.5 boss should have more than 6 faces
    let shell = &solid.boundaries()[0];
    assert!(shell.len() > 6, "Union should have more than 6 faces (got {})", shell.len());
}

#[test]
fn coplanar_box_subtract() {
    // Box1: unit cube [0,1]^3
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let cube = builder::tsweep(&f, Vector3::unit_z());

    // Box2: tool body starting at z=-0.5 going up 2.0 units, sharing the z=0 or z=1 face
    // After not(), subtracting removes material
    let v2 = builder::vertex(Point3::new(0.25, 0.25, -0.5));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 0.5);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 0.5);
    let mut tool = builder::tsweep(&f2, Vector3::unit_z() * 2.0);
    tool.not();

    let result = crate::and(&cube, &tool, 0.05);
    assert!(result.is_some(), "Coplanar box subtract should succeed");

    let solid = result.unwrap();
    let shell = &solid.boundaries()[0];
    assert!(shell.len() > 6, "Subtract should create more than 6 faces (got {})", shell.len());
}

/// Simulates the engine's extrude_cut at 10x scale.
/// The key insight is that the engine's tangent frame places the cube at
/// [-10,0]×[0,10]×[0,10], and the tool extends partially outside the cube.
/// At this scale with tol=0.05, truck's box-box boolean is known to be fragile.
#[test]
#[ignore = "truck 0.4: box-box subtract at 10x scale with tol=0.05 returns unchanged cube"]
fn rect_cut_engine_geometry() {
    // Cube: [0,10]^2 x [0,10]
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 10.0);
    let f = builder::tsweep(&e, Vector3::unit_y() * 10.0);
    let cube = builder::tsweep(&f, Vector3::unit_z() * 10.0);

    // Tool: [2,6]^2 x [-7, 12]
    let v2 = builder::vertex(Point3::new(2.0, 2.0, 12.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x() * 4.0);
    let f2 = builder::tsweep(&e2, Vector3::unit_y() * 4.0);
    let mut tool = builder::tsweep(&f2, -Vector3::unit_z() * 19.0);
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

#[test]
#[ignore = "partially-overlapping coplanar faces with multiple shared planes need face splitting"]
fn coplanar_partial_overlap_union() {
    // Box1: [0,1]^3
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x());
    let f = builder::tsweep(&e, Vector3::unit_y());
    let box1 = builder::tsweep(&f, Vector3::unit_z());

    // Box2: [0.5, 1.5] x [0, 1] x [0, 1] — shares the x=1 face partially
    let v2 = builder::vertex(Point3::new(0.5, 0.0, 0.0));
    let e2 = builder::tsweep(&v2, Vector3::unit_x());
    let f2 = builder::tsweep(&e2, Vector3::unit_y());
    let box2 = builder::tsweep(&f2, Vector3::unit_z());

    let result = crate::or(&box1, &box2, 0.05);
    assert!(result.is_some(), "Partial overlap union should succeed");
}
