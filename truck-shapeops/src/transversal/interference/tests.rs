//! Tests for IC-edge crossing computation and pave block assembly.

use super::*;
use truck_base::cgmath64::*;
use truck_meshalgo::prelude::PolylineCurve;
use truck_topology::*;

/// Helper: create a unit square wire in the XY plane with corners at
/// (0,0,0), (1,0,0), (1,1,0), (0,1,0).
///
/// Returns (wire, vertices) where vertices are [v00, v10, v11, v01].
fn make_unit_square_wire() -> (Wire<Point3, ()>, Vec<Vertex<Point3>>) {
    let v00 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v10 = Vertex::new(Point3::new(1.0, 0.0, 0.0));
    let v11 = Vertex::new(Point3::new(1.0, 1.0, 0.0));
    let v01 = Vertex::new(Point3::new(0.0, 1.0, 0.0));

    let e0 = Edge::new(&v00, &v10, ()); // bottom: y=0
    let e1 = Edge::new(&v10, &v11, ()); // right: x=1
    let e2 = Edge::new(&v11, &v01, ()); // top: y=1
    let e3 = Edge::new(&v01, &v00, ()); // left: x=0

    let wire = wire![e0, e1, e2, e3];
    let verts = vec![v00, v10, v11, v01];
    (wire, verts)
}

/// Helper: create a polyline IC from a list of points.
fn make_ic(points: Vec<Point3>) -> PolylineCurve<Point3> {
    PolylineCurve::from(points)
}

// -----------------------------------------------------------------------
// Test 1: Corner vertex touching (MV3 case)
// -----------------------------------------------------------------------

#[test]
fn corner_touch_at_origin() {
    let (wire, verts) = make_unit_square_wire();
    let tol = 0.01;

    // IC polyline that starts exactly at the origin vertex (0,0,0)
    // and goes to (0.5, 0.5, 0)
    let ic = make_ic(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.5, 0.5, 0.0)]);

    let crossings = compute_ic_edge_crossings(&ic, 0, &wire, tol);

    // Should detect a corner touch at the origin
    let corner_touches: Vec<_> = crossings.iter().filter(|c| c.is_corner_touch).collect();
    assert!(
        !corner_touches.is_empty(),
        "Should detect corner touch at origin, got {} crossings total",
        crossings.len()
    );

    // The corner touch vertex should be at the origin
    let ct = &corner_touches[0];
    let pt = ct.vertex.point();
    assert!(
        (pt - Point3::new(0.0, 0.0, 0.0)).magnitude() < tol,
        "Corner touch vertex should be at origin, got {:?}",
        pt
    );
    assert!(ct.is_corner_touch);
}

#[test]
fn corner_touch_near_vertex() {
    let (wire, _verts) = make_unit_square_wire();
    let tol = 0.01;

    // IC endpoint near (but not exactly at) (1,0,0) — within tolerance
    let ic = make_ic(vec![
        Point3::new(1.001, 0.001, 0.0), // within tol of v10
        Point3::new(0.5, 0.5, 0.0),
    ]);

    let crossings = compute_ic_edge_crossings(&ic, 0, &wire, tol);

    let corner_touches: Vec<_> = crossings.iter().filter(|c| c.is_corner_touch).collect();
    // Near-vertex IC endpoint should still be detected (within tol=0.01,
    // distance ~ 0.0014)
    assert!(
        !corner_touches.is_empty(),
        "Should detect corner touch near (1,0,0), got {} crossings",
        crossings.len()
    );
}

// -----------------------------------------------------------------------
// Test 2: Pave block ordering
// -----------------------------------------------------------------------

#[test]
fn pave_blocks_sorted_by_param() {
    // Create an edge and two crossings at different params
    let v0 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(10.0, 0.0, 0.0));
    let edge = Edge::new(&v0, &v1, ());

    let c1 = IcVertex {
        vertex: Vertex::new(Point3::new(3.0, 0.0, 0.0)),
        edge_param: 0.3,
        ic_index: 0,
        ic_param: 0.5,
        is_corner_touch: false,
    };
    let c2 = IcVertex {
        vertex: Vertex::new(Point3::new(7.0, 0.0, 0.0)),
        edge_param: 0.7,
        ic_index: 1,
        ic_param: 0.8,
        is_corner_touch: false,
    };

    let blocks = FaceInterference::build_edge_pave_blocks(&edge, &[c1, c2]);

    assert_eq!(blocks.len(), 3, "2 crossings should produce 3 pave blocks");

    // Check ordering: param ranges should be contiguous and non-overlapping
    assert!(
        (blocks[0].param_range.0 - 0.0).abs() < 1e-10,
        "First block starts at 0.0"
    );
    assert!(
        (blocks[0].param_range.1 - 0.3).abs() < 1e-10,
        "First block ends at 0.3"
    );
    assert!(
        (blocks[1].param_range.0 - 0.3).abs() < 1e-10,
        "Second block starts at 0.3"
    );
    assert!(
        (blocks[1].param_range.1 - 0.7).abs() < 1e-10,
        "Second block ends at 0.7"
    );
    assert!(
        (blocks[2].param_range.0 - 0.7).abs() < 1e-10,
        "Third block starts at 0.7"
    );
    assert!(
        (blocks[2].param_range.1 - 1.0).abs() < 1e-10,
        "Third block ends at 1.0"
    );

    // IC provenance
    assert!(blocks[0].start_ic.is_none());
    assert_eq!(blocks[0].end_ic, Some(0));
    assert_eq!(blocks[1].start_ic, Some(0));
    assert_eq!(blocks[1].end_ic, Some(1));
    assert_eq!(blocks[2].start_ic, Some(1));
    assert!(blocks[2].end_ic.is_none());
}

// -----------------------------------------------------------------------
// Test 3: Edge with 0 crossings → single full-span pave block
// -----------------------------------------------------------------------

#[test]
fn zero_crossings_full_span() {
    let v0 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(5.0, 0.0, 0.0));
    let edge = Edge::new(&v0, &v1, ());

    let blocks = FaceInterference::<()>::build_edge_pave_blocks(&edge, &[]);

    assert_eq!(blocks.len(), 1, "0 crossings → 1 full-span pave block");
    assert!(
        (blocks[0].param_range.0 - 0.0).abs() < 1e-10,
        "Full span starts at 0.0"
    );
    assert!(
        (blocks[0].param_range.1 - 1.0).abs() < 1e-10,
        "Full span ends at 1.0"
    );
    assert!(blocks[0].sub_edge.is_none(), "Full span has no sub-edge");
    assert!(blocks[0].start_ic.is_none());
    assert!(blocks[0].end_ic.is_none());
}

// -----------------------------------------------------------------------
// Test 4: Edge with 2 crossings → 3 ordered pave blocks
// -----------------------------------------------------------------------

#[test]
fn two_crossings_three_blocks() {
    let v0 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(10.0, 0.0, 0.0));
    let edge = Edge::new(&v0, &v1, ());

    let c1 = IcVertex {
        vertex: Vertex::new(Point3::new(2.0, 0.0, 0.0)),
        edge_param: 0.2,
        ic_index: 0,
        ic_param: 0.1,
        is_corner_touch: false,
    };
    let c2 = IcVertex {
        vertex: Vertex::new(Point3::new(8.0, 0.0, 0.0)),
        edge_param: 0.8,
        ic_index: 1,
        ic_param: 0.9,
        is_corner_touch: false,
    };

    let blocks = FaceInterference::build_edge_pave_blocks(&edge, &[c1, c2]);

    assert_eq!(blocks.len(), 3);

    // No gaps: each block's end param == next block's start param
    for window in blocks.windows(2) {
        assert!(
            (window[0].param_range.1 - window[1].param_range.0).abs() < 1e-10,
            "No gap between blocks: {} vs {}",
            window[0].param_range.1,
            window[1].param_range.0,
        );
    }

    // Start vertex of first block = edge front
    assert_eq!(blocks[0].start_vertex, v0);
    // End vertex of last block = edge back
    assert_eq!(blocks[2].end_vertex, v1);
}

// -----------------------------------------------------------------------
// Test 5: Simple box geometry — IC crosses bottom edge
// -----------------------------------------------------------------------

#[test]
fn ic_crosses_bottom_edge_of_box() {
    let (wire, _verts) = make_unit_square_wire();
    let tol = 0.01;

    // IC polyline crossing the bottom edge (y=0) at x=0.5
    // Goes from (0.5, -0.2, 0) to (0.5, 0.2, 0)
    let ic = make_ic(vec![
        Point3::new(0.5, -0.2, 0.0),
        Point3::new(0.5, 0.2, 0.0),
    ]);

    let crossings = compute_ic_edge_crossings(&ic, 0, &wire, tol);

    // Should find at least one crossing on the bottom edge
    let interior_crossings: Vec<_> = crossings.iter().filter(|c| !c.is_corner_touch).collect();
    assert!(
        !interior_crossings.is_empty(),
        "IC crossing bottom edge should produce interior crossing, got {} total crossings",
        crossings.len()
    );

    // The crossing should be near (0.5, 0, 0)
    let c = &interior_crossings[0];
    let pt = c.vertex.point();
    assert!(
        (pt.x - 0.5).abs() < tol,
        "Crossing x should be ~0.5, got {}",
        pt.x
    );
    assert!(pt.y.abs() < tol, "Crossing y should be ~0, got {}", pt.y);
    assert_eq!(c.ic_index, 0);
    assert!(
        c.edge_param > 0.1 && c.edge_param < 0.9,
        "Crossing should be interior on the edge, param={}",
        c.edge_param
    );
}

#[test]
fn ic_crosses_two_edges_of_box() {
    let (wire, _verts) = make_unit_square_wire();
    let tol = 0.01;

    // IC that crosses both the bottom edge (y=0) and top edge (y=1)
    // Vertical line at x=0.5
    let ic = make_ic(vec![
        Point3::new(0.5, -0.5, 0.0),
        Point3::new(0.5, 0.5, 0.0),
        Point3::new(0.5, 1.5, 0.0),
    ]);

    let crossings = compute_ic_edge_crossings(&ic, 0, &wire, tol);
    let interior_crossings: Vec<_> = crossings.iter().filter(|c| !c.is_corner_touch).collect();

    assert!(
        interior_crossings.len() >= 2,
        "IC crossing both top and bottom edges should produce >= 2 crossings, got {}",
        interior_crossings.len()
    );
}

// -----------------------------------------------------------------------
// Test 6: Deduplication
// -----------------------------------------------------------------------

#[test]
fn deduplicate_nearby_crossings() {
    let tol = 0.01;

    let mut crossings = vec![
        IcVertex {
            vertex: Vertex::new(Point3::new(3.0, 0.0, 0.0)),
            edge_param: 0.300,
            ic_index: 0,
            ic_param: 0.5,
            is_corner_touch: false,
        },
        IcVertex {
            vertex: Vertex::new(Point3::new(3.005, 0.0, 0.0)),
            edge_param: 0.3005, // within tol of the first
            ic_index: 0,
            ic_param: 0.501,
            is_corner_touch: false,
        },
        IcVertex {
            vertex: Vertex::new(Point3::new(7.0, 0.0, 0.0)),
            edge_param: 0.700,
            ic_index: 1,
            ic_param: 0.8,
            is_corner_touch: false,
        },
    ];

    deduplicate_crossings(&mut crossings, tol);

    assert_eq!(
        crossings.len(),
        2,
        "Two crossings within tol on same IC should merge to one"
    );
    assert_eq!(crossings[0].ic_index, 0);
    assert_eq!(crossings[1].ic_index, 1);
}

#[test]
fn deduplicate_keeps_different_ics() {
    let tol = 0.01;

    let mut crossings = vec![
        IcVertex {
            vertex: Vertex::new(Point3::new(3.0, 0.0, 0.0)),
            edge_param: 0.300,
            ic_index: 0,
            ic_param: 0.5,
            is_corner_touch: false,
        },
        IcVertex {
            vertex: Vertex::new(Point3::new(3.005, 0.0, 0.0)),
            edge_param: 0.3005,
            ic_index: 1, // different IC
            ic_param: 0.501,
            is_corner_touch: false,
        },
    ];

    deduplicate_crossings(&mut crossings, tol);

    assert_eq!(
        crossings.len(),
        2,
        "Crossings from different ICs should not be deduplicated"
    );
}

// -----------------------------------------------------------------------
// Test 7: FaceInterference new/add_crossing
// -----------------------------------------------------------------------

#[test]
fn face_interference_basic_operations() {
    let mut fi = FaceInterference::<()>::new();
    assert_eq!(fi.crossing_count(), 0);

    fi.add_crossing(IcVertex {
        vertex: Vertex::new(Point3::new(1.0, 0.0, 0.0)),
        edge_param: 0.5,
        ic_index: 0,
        ic_param: 0.5,
        is_corner_touch: false,
    });
    assert_eq!(fi.crossing_count(), 1);

    fi.add_crossing(IcVertex {
        vertex: Vertex::new(Point3::new(2.0, 0.0, 0.0)),
        edge_param: 0.2,
        ic_index: 1,
        ic_param: 0.3,
        is_corner_touch: true,
    });
    assert_eq!(fi.crossing_count(), 2);

    fi.sort_crossings();
    assert!(
        fi.ic_vertices[0].edge_param <= fi.ic_vertices[1].edge_param,
        "After sort, crossings should be ordered by edge_param"
    );
}

// -----------------------------------------------------------------------
// Test 8: InterferenceTable creation and counting
// -----------------------------------------------------------------------

#[test]
fn interference_table_creation() {
    let table = InterferenceTable::<()>::new(5, 3);
    assert_eq!(table.shell0.len(), 5);
    assert_eq!(table.shell1.len(), 3);
    assert_eq!(table.total_crossings(), 0);
    assert_eq!(table.total_ic_segments(), 0);

    // Display
    let s = format!("{}", table);
    assert!(s.contains("5 faces"));
    assert!(s.contains("3 faces"));
}

// -----------------------------------------------------------------------
// Test 9: Segment-segment closest approach
// -----------------------------------------------------------------------

#[test]
fn segment_closest_parallel() {
    // Two parallel segments, offset by 1 in Y
    let result = segment_segment_closest(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    );
    assert!(
        (result.distance - 1.0).abs() < 1e-10,
        "Parallel segments 1 apart: dist={}",
        result.distance
    );
}

#[test]
fn segment_closest_crossing() {
    // Two segments that cross at (0.5, 0.5, 0)
    let result = segment_segment_closest(
        Point3::new(0.0, 0.5, 0.0),
        Point3::new(1.0, 0.5, 0.0),
        Point3::new(0.5, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
    );
    assert!(
        result.distance < 1e-10,
        "Crossing segments: dist={}",
        result.distance
    );
    assert!(
        (result.s - 0.5).abs() < 1e-10,
        "s should be 0.5, got {}",
        result.s
    );
    assert!(
        (result.t - 0.5).abs() < 1e-10,
        "t should be 0.5, got {}",
        result.t
    );
}

#[test]
fn segment_closest_skew_3d() {
    // Two skew segments in 3D, closest approach at midpoints
    let result = segment_segment_closest(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 1.0),
        Point3::new(0.5, 0.0, -1.0),
    );
    // Closest point on seg A is (0.5, 0, 0), on seg B is (0.5, 0, 0)
    assert!(
        result.distance < 1e-10,
        "Skew segments meeting: dist={}",
        result.distance
    );
}

// -----------------------------------------------------------------------
// Test 10: Polyline arc fractions
// -----------------------------------------------------------------------

#[test]
fn arc_fractions_basic() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(3.0, 0.0, 0.0),
    ];
    let fracs = polyline_arc_fractions(&points);
    assert_eq!(fracs.len(), 4);
    assert!((fracs[0] - 0.0).abs() < 1e-10);
    assert!((fracs[1] - 1.0 / 3.0).abs() < 1e-10);
    assert!((fracs[2] - 2.0 / 3.0).abs() < 1e-10);
    assert!((fracs[3] - 1.0).abs() < 1e-10);
}

// -----------------------------------------------------------------------
// Test 11: Assemble pave blocks for a wire
// -----------------------------------------------------------------------

#[test]
fn assemble_pave_blocks_no_crossings() {
    let (wire, _verts) = make_unit_square_wire();
    let tol = 0.01;

    let pave_blocks = assemble_pave_blocks(&wire, &[], tol);

    // Each of the 4 edges should have exactly 1 full-span pave block
    assert_eq!(pave_blocks.len(), 4);
    for (_eid, blocks) in &pave_blocks {
        assert_eq!(blocks.len(), 1);
        assert!((blocks[0].param_range.0 - 0.0).abs() < 1e-10);
        assert!((blocks[0].param_range.1 - 1.0).abs() < 1e-10);
    }
}

// -----------------------------------------------------------------------
// Test 12: populate_face_interference
// -----------------------------------------------------------------------

#[test]
fn populate_face_interference_basic() {
    let (wire, _verts) = make_unit_square_wire();
    let tol = 0.01;

    let mut fi = FaceInterference::<()>::new();

    // IC crossing the bottom edge at x=0.5
    let ic = make_ic(vec![
        Point3::new(0.5, -0.2, 0.0),
        Point3::new(0.5, 0.2, 0.0),
    ]);

    populate_face_interference(&mut fi, &wire, &[(0, ic)], tol);

    assert!(
        fi.crossing_count() >= 1,
        "Should have at least 1 crossing, got {}",
        fi.crossing_count()
    );
    assert!(
        !fi.edge_pave_blocks.is_empty(),
        "Should have pave blocks for edges"
    );
}

// -----------------------------------------------------------------------
// Test 13: find_corner_touch_snap — corner vertex detection
// -----------------------------------------------------------------------

#[test]
fn corner_touch_detected_at_vertex() {
    let boundary_verts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 10.0, 0.0),
        Point3::new(0.0, 10.0, 0.0),
    ];
    let tol = 0.01;

    // IC endpoint exactly at origin
    let result = find_corner_touch_snap(Point3::new(0.0, 0.0, 0.0), &boundary_verts, tol);
    assert!(result.is_some(), "Exact corner should snap");
    let snapped = result.unwrap();
    assert!(
        (snapped - Point3::new(0.0, 0.0, 0.0)).magnitude() < 1e-15,
        "Should snap to exact origin"
    );

    // IC endpoint near origin (within tol)
    let result = find_corner_touch_snap(Point3::new(0.001, 0.001, 0.001), &boundary_verts, tol);
    assert!(result.is_some(), "Near-corner should snap");
    let snapped = result.unwrap();
    assert!(
        (snapped - Point3::new(0.0, 0.0, 0.0)).magnitude() < 1e-15,
        "Should snap to origin, got {:?}",
        snapped
    );
}

#[test]
fn interior_crossing_not_corner_touch() {
    let boundary_verts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 10.0, 0.0),
        Point3::new(0.0, 10.0, 0.0),
    ];
    let tol = 0.01;

    // IC endpoint at edge midpoint — far from any vertex
    let result = find_corner_touch_snap(Point3::new(5.0, 0.0, 0.0), &boundary_verts, tol);
    assert!(
        result.is_none(),
        "Edge midpoint should NOT snap to a vertex"
    );

    // IC endpoint in face interior
    let result = find_corner_touch_snap(Point3::new(3.0, 4.0, 0.0), &boundary_verts, tol);
    assert!(
        result.is_none(),
        "Face interior point should NOT snap to a vertex"
    );
}

// -----------------------------------------------------------------------
// Test 14: interference_to_boundary_wires fallback
// -----------------------------------------------------------------------

#[test]
fn interference_to_wires_no_crossings() {
    let (wire, _verts) = make_unit_square_wire();
    let fi = FaceInterference::<()>::new();

    let wires = interference_to_boundary_wires(&fi, &wire);
    assert_eq!(wires.len(), 1, "No crossings → 1 original wire");
}
