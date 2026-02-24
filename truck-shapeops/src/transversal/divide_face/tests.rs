use super::*;
use shell::ShellCondition;
use truck_geometry::prelude::*;
use truck_topology::Vertex;
const TOL: f64 = 0.05;

fn line(v0: &Vertex<Point3>, v1: &Vertex<Point3>) -> Edge<Point3, BSplineCurve<Point3>> {
    let curve = BSplineCurve::new(KnotVec::bezier_knot(1), vec![v0.point(), v1.point()]);
    Edge::new(v0, v1, curve)
}

fn parabola(
    v0: &Vertex<Point3>,
    v1: &Vertex<Point3>,
    pt: Point3,
) -> Edge<Point3, BSplineCurve<Point3>> {
    let curve = BSplineCurve::new(KnotVec::bezier_knot(2), vec![v0.point(), pt, v1.point()]);
    Edge::new(v0, v1, curve)
}

#[test]
fn divide_plane_test() {
    let v = Vertex::news([
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 4.0, 0.0),
        Point3::new(-1.0, 1.0, 0.0),
        Point3::new(-1.0, 3.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(1.0, 3.0, 0.0),
    ]);
    let edge = vec![
        parabola(&v[0], &v[1], Point3::new(-4.0, 2.0, 0.0)),
        parabola(&v[0], &v[1], Point3::new(4.0, 2.0, 0.0)),
        line(&v[0], &v[1]),
        parabola(&v[2], &v[3], Point3::new(-3.0, 2.0, 0.0)),
        parabola(&v[2], &v[3], Point3::new(-1.0, 2.0, 0.0)),
        parabola(&v[4], &v[5], Point3::new(1.0, 2.0, 0.0)),
        parabola(&v[4], &v[5], Point3::new(3.0, 2.0, 0.0)),
    ];
    let wire: Vec<Wire<_, _>> = vec![
        vec![edge[1].clone(), edge[0].inverse()].into(),
        vec![edge[2].clone(), edge[0].inverse()].into(),
        vec![edge[1].clone(), edge[2].inverse()].into(),
        vec![edge[3].clone(), edge[4].inverse()].into(),
        vec![edge[5].clone(), edge[6].inverse()].into(),
    ];
    let face = Face::new(
        vec![wire[0].clone(), wire[3].clone(), wire[4].clone()],
        Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ),
    );
    let loops: Loops<_, _> = vec![
        BoundaryWire::new(wire[1].clone(), ShapesOpStatus::Or),
        BoundaryWire::new(wire[2].clone(), ShapesOpStatus::And),
        BoundaryWire::new(wire[3].clone(), ShapesOpStatus::Unknown),
        BoundaryWire::new(wire[4].clone(), ShapesOpStatus::Unknown),
    ]
    .into_iter()
    .collect();
    let res = divide_one_face(&face, &loops, 0.01, 0.01 * 0.01).unwrap();
    assert_eq!(res.len(), 2);
    let (mut or, mut and) = (true, true);
    for (face, status) in res {
        let bdd = face.absolute_boundaries();
        match status {
            ShapesOpStatus::Or => {
                assert_eq!(bdd.len(), 2);
                assert!(bdd[0] == wire[1] || bdd[0] == wire[3]);
                assert!(bdd[1] == wire[1] || bdd[1] == wire[3]);
                assert_ne!(bdd[0], bdd[1]);
                assert!(or);
                or = false;
            }
            ShapesOpStatus::And => {
                assert_eq!(bdd.len(), 2);
                assert!(bdd[0] == wire[2] || bdd[0] == wire[4]);
                assert!(bdd[1] == wire[2] || bdd[1] == wire[4]);
                assert_ne!(bdd[0], bdd[1]);
                assert!(and);
                and = false;
            }
            _ => panic!("There must be no unknown!"),
        }
    }
}

type AlternativeIntersection = crate::alternative::Alternative<
    NurbsCurve<Vector4>,
    IntersectionCurve<PolylineCurve<Point3>, AlternativeSurface, AlternativeSurface>,
>;
type AlternativeSurface = crate::alternative::Alternative<BSplineSurface<Point3>, Plane>;

crate::impl_from!(
    NurbsCurve<Vector4>,
    IntersectionCurve<PolylineCurve<Point3>, AlternativeSurface, AlternativeSurface>
);
crate::impl_from!(BSplineSurface<Point3>, Plane);

fn parabola_surfaces() -> (AlternativeSurface, AlternativeSurface) {
    // define surfaces
    #[rustfmt::skip]
	let ctrl0 = vec![
		vec![Point3::new(-1.0, -1.0, 3.0), Point3::new(-1.0, 0.0, -1.0), Point3::new(-1.0, 1.0, 3.0)],
		vec![Point3::new(0.0, -1.0, -1.0), Point3::new(0.0, 0.0, -5.0), Point3::new(0.0, 1.0, -1.0)],
		vec![Point3::new(1.0, -1.0, 3.0), Point3::new(1.0, 0.0, -1.0), Point3::new(1.0, 1.0, 3.0)],
	];
    #[rustfmt::skip]
	let ctrl1 = vec![
		vec![Point3::new(-1.0, -1.0, -3.0), Point3::new(-1.0, 0.0, 1.0), Point3::new(-1.0, 1.0, -3.0)],
		vec![Point3::new(0.0, -1.0, 1.0), Point3::new(0.0, 0.0, 5.0), Point3::new(0.0, 1.0, 1.0)],
		vec![Point3::new(1.0, -1.0, -3.0), Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, -3.0)],
	];
    (
        BSplineSurface::new((KnotVec::bezier_knot(2), KnotVec::bezier_knot(2)), ctrl0).into(),
        BSplineSurface::new((KnotVec::bezier_knot(2), KnotVec::bezier_knot(2)), ctrl1).into(),
    )
}

#[test]
fn independent_intersection() {
    // prepare geoetries
    let arc00: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(1.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 0.0),
            Vector4::new(-1.0, 0.0, 1.0, 1.0),
        ],
    ))
    .into();
    let arc01: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(-1.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 0.0),
            Vector4::new(1.0, 0.0, 1.0, 1.0),
        ],
    ))
    .into();
    let arc10: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(1.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 0.0),
            Vector4::new(-1.0, 0.0, -1.0, 1.0),
        ],
    ))
    .into();
    let arc11: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(-1.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 0.0),
            Vector4::new(1.0, 0.0, -1.0, 1.0),
        ],
    ))
    .into();
    let (surface0, surface1) = parabola_surfaces();
    let plane0: AlternativeSurface = Plane::new(
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    )
    .into();
    let plane1: AlternativeSurface = Plane::new(
        Point3::new(0.0, 0.0, -1.0),
        Point3::new(1.0, 0.0, -1.0),
        Point3::new(0.0, 1.0, -1.0),
    )
    .into();

    // prepare topologies
    let v00 = Vertex::new(Point3::new(1.0, 0.0, 1.0));
    let v01 = Vertex::new(Point3::new(-1.0, 0.0, 1.0));
    let v10 = Vertex::new(Point3::new(1.0, 0.0, -1.0));
    let v11 = Vertex::new(Point3::new(-1.0, 0.0, -1.0));
    let wire0: Wire<_, _> = vec![Edge::new(&v00, &v01, arc00), Edge::new(&v01, &v00, arc01)].into();
    let wire1: Wire<_, _> = vec![Edge::new(&v10, &v11, arc10), Edge::new(&v11, &v10, arc11)].into();
    let shell0: Shell<_, _, _> = vec![
        Face::new(vec![wire0.clone()], plane0),
        Face::new(vec![wire0], surface0).inverse(),
    ]
    .into();
    assert_eq!(shell0.shell_condition(), ShellCondition::Closed);
    let shell1: Shell<_, _, _> = vec![
        Face::new(vec![wire1.clone()], plane1).inverse(),
        Face::new(vec![wire1], surface1),
    ]
    .into();
    assert_eq!(shell1.shell_condition(), ShellCondition::Closed);
    let poly_shell0 = shell0.triangulation(TOL);
    let poly_shell1 = shell1.triangulation(TOL);

    let LoopsStoreQuadruple {
        geom_loops_store0: loops_store0,
        geom_loops_store1: loops_store1,
        ..
    } = create_loops_stores(
        &shell0,
        &poly_shell0,
        &shell1,
        &poly_shell1,
        TOL,
        None,
        TOL * 0.5,
    )
    .unwrap();
    let [and0, or0, unknown0] = divide_faces_with_coplanar(
        &shell0,
        &loops_store0,
        TOL,
        &rustc_hash::FxHashSet::default(),
        TOL * TOL,
    )
    .unwrap()
    .0
    .and_or_unknown();
    let [and1, or1, unknown1] = divide_faces_with_coplanar(
        &shell1,
        &loops_store1,
        TOL,
        &rustc_hash::FxHashSet::default(),
        TOL * TOL,
    )
    .unwrap()
    .0
    .and_or_unknown();
    assert_eq!(and0.len(), 1);
    assert_eq!(or0.len(), 1);
    assert_eq!(unknown0.len(), 1);
    assert_eq!(and1.len(), 1);
    assert_eq!(or1.len(), 1);
    assert_eq!(unknown1.len(), 1);

    match unknown0[0].surface() {
        AlternativeSurface::FirstType(_) => panic!("This face must plane!"),
        AlternativeSurface::SecondType(_) => {}
    }
    match unknown1[0].surface() {
        AlternativeSurface::FirstType(_) => panic!("This face must plane!"),
        AlternativeSurface::SecondType(_) => {}
    }

    let and_shell: Shell<_, _, _> = vec![and0[0].clone(), and1[0].clone()].into();
    assert_eq!(and_shell.shell_condition(), ShellCondition::Closed);
}

// ============================================================================
// Sprint A: Biangle wire detection tests
// ============================================================================

/// Verify is_biangle_wire correctly identifies degenerate biangle wires
/// (same edge forward + backward) and rejects normal wires.
#[test]
fn biangle_wire_detection() {
    let v0 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(4.0, 0.0, 0.0));
    let edge = line(&v0, &v1);

    // Biangle: same edge forward and backward (shared Arc, same EdgeID)
    let biangle: Wire<_, _> = vec![edge.inverse(), edge.clone()].into();
    assert!(
        is_biangle_wire(&biangle),
        "wire with same edge in both orientations must be detected as biangle"
    );

    // Normal 2-edge wire with DIFFERENT edges (different Arcs)
    let v2 = Vertex::new(Point3::new(2.0, 2.0, 0.0));
    let normal: Wire<_, _> = vec![line(&v0, &v2), line(&v2, &v0)].into();
    assert!(
        !is_biangle_wire(&normal),
        "wire with two different edges is not biangle"
    );

    // 3-edge triangle: not biangle (wrong length)
    let triangle: Wire<_, _> =
        vec![line(&v0, &v1), line(&v1, &v2), line(&v2, &v0)].into();
    assert!(!is_biangle_wire(&triangle), "3-edge wire is not biangle");

    // Single edge: not biangle (wrong length)
    let single: Wire<_, _> = vec![line(&v0, &v1)].into();
    assert!(!is_biangle_wire(&single), "single-edge wire is not biangle");
}

/// Verify that divide_one_face filters biangle wires and still produces
/// valid face fragments. This is a regression test for Sprint A: without
/// biangle filtering, the biangle + IC wires could cause Face::try_new
/// to fail with NotDisjointWires/NotSimpleWire on complex shells.
#[test]
fn divide_face_with_biangle_wire_produces_fragments() {
    // Square face on z=0 plane with IC vertices on the boundary
    let v = Vertex::news([
        Point3::new(0.0, 0.0, 0.0), // v0
        Point3::new(4.0, 0.0, 0.0), // v1
        Point3::new(4.0, 4.0, 0.0), // v2
        Point3::new(0.0, 4.0, 0.0), // v3
        Point3::new(2.0, 0.0, 0.0), // v4: IC vertex on bottom edge
        Point3::new(2.0, 4.0, 0.0), // v5: IC vertex on top edge
    ]);

    // After IC vertex insertion, the boundary is split at v4 and v5.
    // add_edge creates two closed IC wires + possibly a biangle.
    // Simulate the post-add_edge state:
    // Left half: v4→v5→v3→v0→v4 (And)
    let left_wire: Wire<_, _> = vec![
        line(&v[4], &v[5]),
        line(&v[5], &v[3]),
        line(&v[3], &v[0]),
        line(&v[0], &v[4]),
    ]
    .into();
    // Right half: v5→v4→v1→v2→v5 (Or)
    let right_wire: Wire<_, _> = vec![
        line(&v[5], &v[4]),
        line(&v[4], &v[1]),
        line(&v[1], &v[2]),
        line(&v[2], &v[5]),
    ]
    .into();
    // Biangle: same IC edge forward + backward (simulates add_edge None,None case)
    let ic_edge = line(&v[4], &v[5]);
    let biangle_wire: Wire<_, _> = vec![ic_edge.inverse(), ic_edge].into();
    assert!(is_biangle_wire(&biangle_wire), "test setup: must be biangle");

    let face = Face::new(
        vec![left_wire.clone()],
        Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ),
    );

    // Loops with left half (And), right half (Or), and biangle (Unknown)
    let loops: Loops<_, _> = vec![
        BoundaryWire::new(left_wire, ShapesOpStatus::And),
        BoundaryWire::new(right_wire, ShapesOpStatus::Or),
        BoundaryWire::new(biangle_wire, ShapesOpStatus::Unknown),
    ]
    .into_iter()
    .collect();

    let res = divide_one_face(&face, &loops, TOL, TOL * TOL);
    assert!(res.is_some(), "divide_one_face should not return None");
    let fragments = res.unwrap();
    assert!(
        !fragments.is_empty(),
        "biangle wire must be filtered — face should produce fragments, got 0"
    );
}

// ============================================================================
// Sprint B: Non-simple wire splitting tests
// ============================================================================

/// Verify split_wire_recursive handles a figure-8 wire (vertex visited twice)
/// by splitting into two simple closed sub-wires.
#[test]
fn split_wire_recursive_figure_eight() {
    // Wire: v0→v1→v2→v0→v3→v4→v0 (v0 appears at positions 0 and 3)
    let v = Vertex::news([
        Point3::new(0.0, 0.0, 0.0),  // v0: repeated vertex
        Point3::new(2.0, 1.0, 0.0),  // v1
        Point3::new(1.0, 2.0, 0.0),  // v2
        Point3::new(-2.0, 1.0, 0.0), // v3
        Point3::new(-1.0, -2.0, 0.0), // v4
    ]);
    let wire: Wire<_, _> = vec![
        line(&v[0], &v[1]),
        line(&v[1], &v[2]),
        line(&v[2], &v[0]),
        line(&v[0], &v[3]),
        line(&v[3], &v[4]),
        line(&v[4], &v[0]),
    ]
    .into();
    assert!(wire.is_closed(), "test setup: wire must be closed");
    assert!(!wire.is_simple(), "test setup: wire must be non-simple (v0 repeated)");

    let mut output = Vec::new();
    let success = super::super::split_wire_recursive(&wire, &mut output, 0);
    assert!(success, "split_wire_recursive should succeed on figure-8");
    assert_eq!(output.len(), 2, "figure-8 should split into exactly 2 sub-wires");
    for (i, w) in output.iter().enumerate() {
        assert!(w.is_simple(), "sub-wire {} must be simple", i);
        assert!(w.is_closed(), "sub-wire {} must be closed", i);
        assert!(w.len() >= 3, "sub-wire {} must have ≥3 edges (not degenerate)", i);
    }
}

/// Verify split_wire_recursive handles a triple-visit vertex
/// (vertex appears 3 times in a 9-edge wire, similar to k8 diagnostic).
#[test]
fn split_wire_recursive_triple_visit() {
    // Wire: v0→v1→v2→v0→v3→v4→v0→v5→v6→v0
    // v0 appears as front vertex at positions 0, 3, and 6
    let v = Vertex::news([
        Point3::new(0.0, 0.0, 0.0),  // v0: visited 3 times
        Point3::new(3.0, 0.0, 0.0),  // v1
        Point3::new(2.0, 3.0, 0.0),  // v2
        Point3::new(-3.0, 0.0, 0.0), // v3
        Point3::new(-2.0, 3.0, 0.0), // v4
        Point3::new(-1.0, -3.0, 0.0), // v5
        Point3::new(1.0, -3.0, 0.0), // v6
    ]);
    let wire: Wire<_, _> = vec![
        line(&v[0], &v[1]),
        line(&v[1], &v[2]),
        line(&v[2], &v[0]),
        line(&v[0], &v[3]),
        line(&v[3], &v[4]),
        line(&v[4], &v[0]),
        line(&v[0], &v[5]),
        line(&v[5], &v[6]),
        line(&v[6], &v[0]),
    ]
    .into();
    assert!(wire.is_closed());
    assert!(!wire.is_simple());

    let mut output = Vec::new();
    let success = super::super::split_wire_recursive(&wire, &mut output, 0);
    assert!(success, "should split triple-visit wire");
    assert!(
        output.len() >= 2,
        "should produce ≥2 sub-wires from triple-visit, got {}",
        output.len()
    );
    for (i, w) in output.iter().enumerate() {
        assert!(w.is_simple(), "sub-wire {} must be simple", i);
        assert!(w.is_closed(), "sub-wire {} must be closed", i);
        assert!(
            !is_biangle_wire(w),
            "sub-wire {} must not be biangle",
            i
        );
    }
}

// ============================================================================
// Sprint D: Zero-fragment face preservation test
// ============================================================================

/// When divide_one_face produces 0 fragments (area cancellation),
/// divide_faces_with_coplanar must preserve the original face as Unknown,
/// not silently drop it. This is the root cause of the "14 unknown faces"
/// cascade in the k8 diagnostic.
#[test]
fn zero_fragment_face_preserved_in_classification() {
    // Build a 10×10 square face on z=0 plane
    let v = Vertex::news([
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 10.0, 0.0),
        Point3::new(0.0, 10.0, 0.0),
    ]);
    let outer_wire: Wire<_, _> = vec![
        line(&v[0], &v[1]),
        line(&v[1], &v[2]),
        line(&v[2], &v[3]),
        line(&v[3], &v[0]),
    ]
    .into();
    let surface = Plane::new(
        Point3::origin(),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    );
    let face = Face::new(vec![outer_wire.clone()], surface);
    let shell: Shell<_, _, _> = vec![face].into();

    // Create a nearly-matching clockwise inner wire that cancels the outer area.
    // Inner: (ε,ε) → (ε,10-ε) → (10-ε,10-ε) → (10-ε,ε) → (ε,ε) [clockwise]
    // Area ≈ -(10-2ε)² ≈ -99.996 when ε=0.001
    // |outer_area + inner_area| = |100 - 99.996| = 0.004 < TOL (0.05) → cleared
    let eps = 0.001;
    let vi = Vertex::news([
        Point3::new(eps, eps, 0.0),
        Point3::new(10.0 - eps, eps, 0.0),
        Point3::new(10.0 - eps, 10.0 - eps, 0.0),
        Point3::new(eps, 10.0 - eps, 0.0),
    ]);
    let inner_wire: Wire<_, _> = vec![
        line(&vi[0], &vi[3]), // (ε,ε) → (ε,10-ε): up
        line(&vi[3], &vi[2]), // (ε,10-ε) → (10-ε,10-ε): right
        line(&vi[2], &vi[1]), // (10-ε,10-ε) → (10-ε,ε): down
        line(&vi[1], &vi[0]), // (10-ε,ε) → (ε,ε): left
    ]
    .into();

    // Build LoopsStore from the shell, then inject the inner wire with And status
    // to force divide_one_face to be called (not the all-Unknown shortcut).
    let mut loops_store: LoopsStore<_, _> = shell.face_iter().collect();
    loops_store[0].push(BoundaryWire::new(inner_wire, ShapesOpStatus::And));

    let result = divide_faces_with_coplanar(
        &shell,
        &loops_store,
        TOL,
        &rustc_hash::FxHashSet::default(),
        TOL * TOL,
    );

    assert!(result.is_some(), "divide_faces_with_coplanar must not abort");
    let (cls, _) = result.unwrap();
    let [and, or, unknown] = cls.and_or_unknown();
    let total = and.len() + or.len() + unknown.len();
    assert_eq!(
        total, 1,
        "zero-fragment face must be preserved (not lost), got {} faces",
        total
    );
    assert_eq!(
        unknown.len(),
        1,
        "preserved face should be classified as Unknown for downstream ray-cast"
    );
}
