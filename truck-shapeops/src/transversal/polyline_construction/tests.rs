use super::*;
use truck_base::tolerance::TOLERANCE;

/// Closed polyline segments in two different orderings must produce
/// identical vertex sequences after canonicalization.
#[test]
fn closed_polyline_direction_canonical() {
    // A closed square: A→B→C→D→A
    let a = Point3::new(0.0, 0.0, 0.0);
    let b = Point3::new(1.0, 0.0, 0.0);
    let c = Point3::new(1.0, 1.0, 0.0);
    let d = Point3::new(0.0, 1.0, 0.0);

    // Original order
    let lines_v1 = vec![(a, b), (b, c), (c, d), (d, a)];
    // Shuffled order (different starting edge and direction)
    let lines_v2 = vec![(c, d), (a, b), (d, a), (b, c)];
    // Reversed direction
    let lines_v3 = vec![(a, d), (d, c), (c, b), (b, a)];

    let poly1 = construct_polylines(&lines_v1, TOLERANCE);
    let poly2 = construct_polylines(&lines_v2, TOLERANCE);
    let poly3 = construct_polylines(&lines_v3, TOLERANCE);

    assert_eq!(poly1.len(), 1);
    assert_eq!(poly2.len(), 1);
    assert_eq!(poly3.len(), 1);
    assert_eq!(poly1[0].len(), poly2[0].len());
    assert_eq!(poly1[0].len(), poly3[0].len());

    // All three must produce identical vertex sequences
    for (va, vb) in poly1[0].iter().zip(poly2[0].iter()) {
        assert!(va.near(vb), "v1 vs v2 mismatch: {:?} vs {:?}", va, vb);
    }
    for (va, vc) in poly1[0].iter().zip(poly3[0].iter()) {
        assert!(va.near(vc), "v1 vs v3 mismatch: {:?} vs {:?}", va, vc);
    }
}

/// Closed polyline canonicalization with a 3D triangle.
#[test]
fn closed_polyline_canonical_3d_triangle() {
    let a = Point3::new(1.0, 2.0, 3.0);
    let b = Point3::new(4.0, 0.0, 1.0);
    let c = Point3::new(2.0, 5.0, 0.0);

    // Forward
    let lines_fwd = vec![(a, b), (b, c), (c, a)];
    // Reversed
    let lines_rev = vec![(a, c), (c, b), (b, a)];
    // Starting from different edge
    let lines_rot = vec![(b, c), (c, a), (a, b)];

    let p_fwd = construct_polylines(&lines_fwd, TOLERANCE);
    let p_rev = construct_polylines(&lines_rev, TOLERANCE);
    let p_rot = construct_polylines(&lines_rot, TOLERANCE);

    assert_eq!(p_fwd.len(), 1);
    assert_eq!(p_rev.len(), 1);
    assert_eq!(p_rot.len(), 1);

    for (a, b) in p_fwd[0].iter().zip(p_rev[0].iter()) {
        assert!(a.near(b), "fwd vs rev mismatch: {:?} vs {:?}", a, b);
    }
    for (a, b) in p_fwd[0].iter().zip(p_rot[0].iter()) {
        assert!(a.near(b), "fwd vs rot mismatch: {:?} vs {:?}", a, b);
    }
}

/// Deterministic graph traversal: get_one always picks lex-smallest.
#[test]
fn deterministic_get_one() {
    let lines = vec![
        (Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 5.0, 5.0)),
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
    ];
    let poly = construct_polylines(&lines, TOLERANCE);
    assert_eq!(poly.len(), 2);
    // The polyline starting with (0,0,0) should be emitted first
    // because get_one picks lex-smallest PointIndex.
    assert!(poly[0][0].near(&Point3::new(0.0, 0.0, 0.0)));
}

#[test]
fn construct_polylines_positive0() {
    let lines = vec![
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 0.0, 1.0)),
        (Point3::new(0.0, 1.0, 1.0), Point3::new(1.0, 1.0, 1.0)),
        (Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)),
        (Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(1.0, 1.0, 1.0), Point3::new(0.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 1.0), Point3::new(0.0, 1.0, 1.0)),
    ];
    let polyline = construct_polylines(&lines, TOLERANCE);
    assert_eq!(polyline.len(), 1);
    assert_eq!(polyline[0].len(), 9);

    let mut sign = None;
    for line in polyline[0].windows(2) {
        let a = line[0][0] + line[0][1] * 2.0 + line[0][2] * 4.0;
        let b = line[1][0] + line[1][1] * 2.0 + line[1][2] * 4.0;
        let x = b - a;
        assert!(f64::abs(x) == 1.0 || f64::abs(x) == 7.0);
        let s = f64::signum(x * (x - 2.0) * (x + 2.0));
        if let Some(sign) = sign {
            assert!(s == sign);
        } else {
            sign = Some(s);
        }
    }
}

#[test]
fn construct_polylines_positive1() {
    let lines = vec![
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)),
        (Point3::new(1.0, 1.0, 1.0), Point3::new(0.0, 1.0, 1.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 1.0, 0.0)),
        (Point3::new(0.0, 1.0, 0.0), Point3::new(0.0, 0.0, 0.0)),
        (Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)),
        (Point3::new(0.0, 1.0, 1.0), Point3::new(0.0, 0.0, 1.0)),
    ];
    let polyline = construct_polylines(&lines, TOLERANCE);
    assert_eq!(polyline.len(), 2);
    assert_eq!(polyline[0].len(), 5);
    assert_eq!(polyline[1].len(), 5);
}

#[test]
fn construct_polylines_positive2() {
    let lines = vec![
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 0.0, 1.0)),
        (Point3::new(0.0, 1.0, 1.0), Point3::new(1.0, 1.0, 1.0)),
        (Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(1.0, 1.0, 1.0), Point3::new(0.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 1.0), Point3::new(0.0, 1.0, 1.0)),
    ];
    let polyline = construct_polylines(&lines, TOLERANCE);
    assert_eq!(polyline.len(), 1);
    assert_eq!(polyline[0].len(), 9);

    let mut sign = None;
    for line in polyline[0].windows(2) {
        let a = line[0][0] + line[0][1] * 2.0 + line[0][2] * 4.0;
        let b = line[1][0] + line[1][1] * 2.0 + line[1][2] * 4.0;
        let x = b - a;
        assert!(f64::abs(x) == 1.0 || f64::abs(x) == 7.0);
        let s = f64::signum(x * (x - 2.0) * (x + 2.0));
        if let Some(sign) = sign {
            assert!(s == sign);
        } else {
            sign = Some(s);
        }
    }
}

#[test]
fn construct_polylines_positive3() {
    let lines = vec![
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
        (Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 0.0, 1.0)),
        (Point3::new(0.0, 1.0, 1.0), Point3::new(1.0, 1.0, 1.0)),
        (Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)),
        (Point3::new(1.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)),
        (Point3::new(1.0, 0.0, 1.0), Point3::new(0.0, 1.0, 1.0)),
    ];
    let polyline = construct_polylines(&lines, TOLERANCE);
    assert_eq!(polyline.len(), 1);
    assert_eq!(polyline[0].len(), 8);

    let mut sign = None;
    for line in polyline[0].windows(2) {
        let a = line[0][0] + line[0][1] * 2.0 + line[0][2] * 4.0;
        let b = line[1][0] + line[1][1] * 2.0 + line[1][2] * 4.0;
        let x = b - a;
        assert!(f64::abs(x) == 1.0);
        let s = f64::signum(x * (x - 2.0) * (x + 2.0));
        if let Some(sign) = sign {
            assert!(s == sign);
        } else {
            sign = Some(s);
        }
    }
}
