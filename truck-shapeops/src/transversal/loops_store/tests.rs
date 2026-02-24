use super::*;
use std::fmt::Debug;
const TOL: f64 = 0.01;

crate::impl_from!(
    NurbsCurve<Vector4>,
    IntersectionCurve<PolylineCurve, BSplineSurface<Point3>, BSplineSurface<Point3>>
);
type AlternativeIntersection = crate::alternative::Alternative<
    NurbsCurve<Vector4>,
    IntersectionCurve<PolylineCurve, BSplineSurface<Point3>, BSplineSurface<Point3>>,
>;

struct DebugDisplay<'a, T, Format> {
    entity: &'a T,
    format: Format,
}

impl<P: Debug, C: Debug> Debug for DebugDisplay<'_, Loops<P, C>, WireDisplayFormat> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Loops")
            .field(
                &self
                    .entity
                    .0
                    .iter()
                    .map(|wire| wire.display(self.format))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<P: Debug, C: Debug> Debug for DebugDisplay<'_, LoopsStore<P, C>, WireDisplayFormat> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.entity.0.iter().map(|loops| DebugDisplay {
                entity: loops,
                format: self.format,
            }))
            .finish()
    }
}

impl<P: Debug, C: Debug> LoopsStore<P, C> {
    fn display(&self, format: WireDisplayFormat) -> DebugDisplay<'_, Self, WireDisplayFormat> {
        DebugDisplay {
            entity: self,
            format,
        }
    }
}

fn parabola_surfaces() -> (BSplineSurface<Point3>, BSplineSurface<Point3>) {
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
        BSplineSurface::new((KnotVec::bezier_knot(2), KnotVec::bezier_knot(2)), ctrl0),
        BSplineSurface::new((KnotVec::bezier_knot(2), KnotVec::bezier_knot(2)), ctrl1),
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
    let arc01 = NurbsCurve::new(BSplineCurve::new(
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
    let arc11 = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(-1.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 0.0),
            Vector4::new(1.0, 0.0, -1.0, 1.0),
        ],
    ))
    .into();
    let (surface0, surface1) = parabola_surfaces();

    // prepare topologies
    let v00 = Vertex::new(Point3::new(1.0, 0.0, 1.0));
    let v01 = Vertex::new(Point3::new(-1.0, 0.0, 1.0));
    let v10 = Vertex::new(Point3::new(1.0, 0.0, -1.0));
    let v11 = Vertex::new(Point3::new(-1.0, 0.0, -1.0));
    let wire0: Wire<_, _> = vec![Edge::new(&v00, &v01, arc00), Edge::new(&v01, &v00, arc01)].into();
    let wire1: Wire<_, _> = vec![Edge::new(&v10, &v11, arc10), Edge::new(&v11, &v10, arc11)].into();
    let geom_shell0: Shell<_, _, _> = vec![Face::new(vec![wire0], surface0).inverse()].into();
    let geom_shell1: Shell<_, _, _> = vec![Face::new(vec![wire1], surface1)].into();
    let poly_shell0 = geom_shell0.triangulation(TOL);
    let poly_shell1 = geom_shell1.triangulation(TOL);

    // exec create loops stores!
    let LoopsStoreQuadruple {
        geom_loops_store0,
        geom_loops_store1,
        ..
    } = create_loops_stores(
        &geom_shell0,
        &poly_shell0,
        &geom_shell1,
        &poly_shell1,
        TOL,
        None,
        TOL * 0.5,
    )
    .unwrap();

    // check the topology
    let vertex_format = VertexDisplayFormat::AsPoint;
    let edge_id_format = EdgeDisplayFormat::VerticesTupleAndID { vertex_format };
    let wire_id_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_id_format,
    };
    let edge_geom_format = EdgeDisplayFormat::VerticesTupleAndCurve { vertex_format };
    let wire_geom_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_geom_format,
    };
    assert_eq!(geom_loops_store0.len(), 1);
    assert_eq!(geom_loops_store0[0].len(), 3);
    assert_eq!(geom_loops_store0[0][0].len(), 2);
    assert_eq!(geom_loops_store0[0][1].len(), 2);
    assert_eq!(geom_loops_store0[0][2].len(), 2);
    assert!(
        geom_loops_store0[0][0].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][1].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][2].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[0][2].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert_eq!(geom_loops_store1.len(), 1);
    assert_eq!(geom_loops_store1[0].len(), 3);
    assert_eq!(geom_loops_store1[0][0].len(), 2);
    assert_eq!(geom_loops_store1[0][1].len(), 2);
    assert_eq!(geom_loops_store1[0][2].len(), 2);
    assert!(
        geom_loops_store1[0][0].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][1].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][2].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[0][2].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );

    // check the boundary status
    let mut flags = [true; 3];
    for i in 0..3 {
        let bw = &geom_loops_store0[0][i];
        match bw.status() {
            ShapesOpStatus::Unknown => {
                let curve = bw[0].oriented_curve();
                assert_near!(curve.subs(0.0)[2], 1.0);
                assert!(flags[0]);
                flags[0] = false;
            }
            ShapesOpStatus::And => {
                let curve = bw[0].oriented_curve();
                let pt = curve.subs(0.5) - Point3::origin();
                let der = curve.der(0.5);
                assert!(pt.cross(der)[2] > 0.0);
                assert!(flags[1]);
                flags[1] = false;
            }
            ShapesOpStatus::Or => {
                let curve = bw[0].oriented_curve();
                let pt = curve.subs(0.5) - Point3::origin();
                let der = curve.der(0.5);
                assert!(pt.cross(der)[2] < 0.0);
                assert!(flags[2]);
                flags[2] = false;
            }
        }
    }
    let mut flags = [true; 3];
    for i in 0..3 {
        let bw = &geom_loops_store1[0][i];
        match bw.status() {
            ShapesOpStatus::Unknown => {
                let curve = bw[0].oriented_curve();
                assert_near!(curve.subs(0.0)[2], -1.0);
                assert!(flags[0]);
                flags[0] = false;
            }
            ShapesOpStatus::Or => {
                let curve = bw[0].oriented_curve();
                let pt = curve.subs(0.5) - Point3::origin();
                let der = curve.der(0.5);
                assert!(pt.cross(der)[2] < 0.0);
                assert!(flags[1]);
                flags[1] = false;
            }
            ShapesOpStatus::And => {
                let curve = bw[0].oriented_curve();
                let pt = curve.subs(0.5) - Point3::origin();
                let der = curve.der(0.5);
                assert!(pt.cross(der)[2] > 0.0);
                assert!(flags[2]);
                flags[2] = false;
            }
        }
    }
}

#[test]
fn rotated_intersection() {
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
    let arc02: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(1.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, -3.0, 1.0),
            Vector4::new(-1.0, 0.0, 1.0, 1.0),
        ],
    ))
    .into();
    let arc10: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(0.0, -1.0, -1.0, 1.0),
            Vector4::new(1.0, 0.0, 0.0, 0.0),
            Vector4::new(0.0, 1.0, -1.0, 1.0),
        ],
    ))
    .into();
    let arc11: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(0.0, 1.0, -1.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 0.0),
            Vector4::new(0.0, -1.0, -1.0, 1.0),
        ],
    ))
    .into();
    let arc12: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(0.0, -1.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, 3.0, 1.0),
            Vector4::new(0.0, 1.0, -1.0, 1.0),
        ],
    ))
    .into();
    let (surface0, surface1) = parabola_surfaces();

    let v00 = Vertex::new(Point3::new(1.0, 0.0, 1.0));
    let v01 = Vertex::new(Point3::new(-1.0, 0.0, 1.0));
    let edge00 = Edge::new(&v00, &v01, arc00);
    let edge01 = Edge::new(&v01, &v00, arc01);
    let edge02 = Edge::new(&v00, &v01, arc02);
    let wire00: Wire<_, _> = vec![edge00, edge02.inverse()].into();
    let wire01: Wire<_, _> = vec![edge01, edge02].into();
    let face00 = Face::new(vec![wire00], surface0.clone());
    let face01 = Face::new(vec![wire01], surface0);
    let geom_shell0: Shell<_, _, _> = vec![face00.inverse(), face01.inverse()].into();

    let v10 = Vertex::new(Point3::new(0.0, -1.0, -1.0));
    let v11 = Vertex::new(Point3::new(0.0, 1.0, -1.0));
    let edge10 = Edge::new(&v10, &v11, arc10);
    let edge11 = Edge::new(&v11, &v10, arc11);
    let edge12 = Edge::new(&v10, &v11, arc12);
    let wire10: Wire<_, _> = vec![edge10, edge12.inverse()].into();
    let wire11: Wire<_, _> = vec![edge12, edge11].into();
    let face10 = Face::new(vec![wire10], surface1.clone());
    let face11 = Face::new(vec![wire11], surface1);
    let geom_shell1: Shell<_, _, _> = vec![face10, face11].into();

    let poly_shell0 = geom_shell0.triangulation(TOL);
    let poly_shell1 = geom_shell1.triangulation(TOL);
    let file = std::fs::File::create("polyshell0.obj").unwrap();
    obj::write(&poly_shell0.to_polygon(), file).unwrap();
    let file = std::fs::File::create("polyshell1.obj").unwrap();
    obj::write(&poly_shell1.to_polygon(), file).unwrap();
    let LoopsStoreQuadruple {
        geom_loops_store0,
        geom_loops_store1,
        ..
    } = create_loops_stores(
        &geom_shell0,
        &poly_shell0,
        &geom_shell1,
        &poly_shell1,
        TOL,
        None,
        TOL * 0.5,
    )
    .unwrap();

    let vertex_format = VertexDisplayFormat::AsPoint;
    let edge_id_format = EdgeDisplayFormat::VerticesTupleAndID { vertex_format };
    let wire_id_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_id_format,
    };
    let edge_geom_format = EdgeDisplayFormat::VerticesTupleAndCurve { vertex_format };
    let wire_geom_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_geom_format,
    };
    assert_eq!(geom_loops_store0.len(), 2);
    assert_eq!(geom_loops_store0[0].len(), 2);
    assert!(
        geom_loops_store0[0][0].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][1].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store0[0][0].len(), geom_loops_store0[0][1].len());
    let compatible0 = match geom_loops_store0[0][0].status() {
        ShapesOpStatus::And => a == 3,
        ShapesOpStatus::Or => a == 5,
        _ => false,
    };
    let compatible1 = match geom_loops_store0[0][1].status() {
        ShapesOpStatus::And => b == 3,
        ShapesOpStatus::Or => b == 5,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 15,
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert_eq!(geom_loops_store0[1].len(), 2);
    assert!(
        geom_loops_store0[1][0].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[1][1].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[1][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[1][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store0[1][0].len(), geom_loops_store0[1][1].len());
    let compatible0 = match geom_loops_store0[1][0].status() {
        ShapesOpStatus::And => a == 3,
        ShapesOpStatus::Or => a == 5,
        _ => false,
    };
    let compatible1 = match geom_loops_store0[1][1].status() {
        ShapesOpStatus::And => b == 3,
        ShapesOpStatus::Or => b == 5,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 15,
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert_eq!(geom_loops_store1.len(), 2);
    assert_eq!(geom_loops_store1[0].len(), 2);
    assert!(
        geom_loops_store1[0][0].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][1].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store1[0][0].len(), geom_loops_store1[0][1].len());
    let compatible0 = match geom_loops_store1[0][0].status() {
        ShapesOpStatus::And => a == 3,
        ShapesOpStatus::Or => a == 5,
        _ => false,
    };
    let compatible1 = match geom_loops_store1[0][1].status() {
        ShapesOpStatus::And => b == 3,
        ShapesOpStatus::Or => b == 5,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 15,
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert_eq!(geom_loops_store1[1].len(), 2);
    assert!(
        geom_loops_store1[1][0].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[1][1].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[1][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[1][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store1[1][0].len(), geom_loops_store1[1][1].len());
    let compatible0 = match geom_loops_store1[1][0].status() {
        ShapesOpStatus::And => a == 3,
        ShapesOpStatus::Or => a == 5,
        _ => false,
    };
    let compatible1 = match geom_loops_store1[1][1].status() {
        ShapesOpStatus::And => b == 3,
        ShapesOpStatus::Or => b == 5,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 15,
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
}

#[test]
fn crossing_edges() {
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
    let arc02: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(1.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, -3.0, 1.0),
            Vector4::new(-1.0, 0.0, 1.0, 1.0),
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
    let arc12: AlternativeIntersection = NurbsCurve::new(BSplineCurve::new(
        KnotVec::bezier_knot(2),
        vec![
            Vector4::new(1.0, 0.0, -1.0, 1.0),
            Vector4::new(0.0, 0.0, 3.0, 1.0),
            Vector4::new(-1.0, 0.0, -1.0, 1.0),
        ],
    ))
    .into();
    let (surface0, surface1) = parabola_surfaces();

    let v00 = Vertex::new(Point3::new(1.0, 0.0, 1.0));
    let v01 = Vertex::new(Point3::new(-1.0, 0.0, 1.0));
    let edge00 = Edge::new(&v00, &v01, arc00);
    let edge01 = Edge::new(&v01, &v00, arc01);
    let edge02 = Edge::new(&v00, &v01, arc02);
    let wire00: Wire<_, _> = vec![edge00, edge02.inverse()].into();
    let wire01: Wire<_, _> = vec![edge01, edge02].into();
    let face00 = Face::new(vec![wire00], surface0.clone());
    let face01 = Face::new(vec![wire01], surface0);
    let geom_shell0: Shell<_, _, _> = vec![face00.inverse(), face01.inverse()].into();

    let v10 = Vertex::new(Point3::new(1.0, 0.0, -1.0));
    let v11 = Vertex::new(Point3::new(-1.0, 0.0, -1.0));
    let edge10 = Edge::new(&v10, &v11, arc10);
    let edge11 = Edge::new(&v11, &v10, arc11);
    let edge12 = Edge::new(&v10, &v11, arc12);
    let wire10: Wire<_, _> = vec![edge10, edge12.inverse()].into();
    let wire11: Wire<_, _> = vec![edge11, edge12].into();
    let face10 = Face::new(vec![wire10], surface1.clone());
    let face11 = Face::new(vec![wire11], surface1);
    let geom_shell1: Shell<_, _, _> = vec![face10, face11].into();

    let poly_shell0 = geom_shell0.triangulation(TOL);
    let poly_shell1 = geom_shell1.triangulation(TOL);
    let LoopsStoreQuadruple {
        geom_loops_store0,
        geom_loops_store1,
        ..
    } = create_loops_stores(
        &geom_shell0,
        &poly_shell0,
        &geom_shell1,
        &poly_shell1,
        TOL,
        None,
        TOL * 0.5,
    )
    .unwrap();

    let vertex_format = VertexDisplayFormat::AsPoint;
    let edge_id_format = EdgeDisplayFormat::VerticesTupleAndID { vertex_format };
    let wire_id_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_id_format,
    };
    let edge_geom_format = EdgeDisplayFormat::VerticesTupleAndCurve { vertex_format };
    let wire_geom_format = WireDisplayFormat::EdgesListTuple {
        edge_format: edge_geom_format,
    };
    assert_eq!(geom_loops_store0.len(), 2);
    assert_eq!(geom_loops_store0[0].len(), 2);
    assert!(
        geom_loops_store0[0][0].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][1].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store0[0][0].len(), geom_loops_store0[0][1].len());
    let compatible0 = match geom_loops_store0[0][0].status() {
        ShapesOpStatus::And => a == 2,
        ShapesOpStatus::Or => a == 4,
        _ => false,
    };
    let compatible1 = match geom_loops_store0[0][1].status() {
        ShapesOpStatus::And => b == 2,
        ShapesOpStatus::Or => b == 4,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 8,
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert_eq!(geom_loops_store0[1].len(), 2);
    assert!(
        geom_loops_store0[1][0].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[1][1].is_closed(),
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert!(
        geom_loops_store0[1][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    assert!(
        geom_loops_store0[1][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store0.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store0[1][0].len(), geom_loops_store0[1][1].len());
    let compatible0 = match geom_loops_store0[1][0].status() {
        ShapesOpStatus::And => a == 2,
        ShapesOpStatus::Or => a == 4,
        _ => false,
    };
    let compatible1 = match geom_loops_store0[1][1].status() {
        ShapesOpStatus::And => b == 2,
        ShapesOpStatus::Or => b == 4,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 8,
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
    assert_eq!(geom_loops_store1.len(), 2);
    assert_eq!(geom_loops_store1[0].len(), 2);
    assert!(
        geom_loops_store1[0][0].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][1].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[0][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[0][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store1[0][0].len(), geom_loops_store1[0][1].len());
    let compatible0 = match geom_loops_store1[0][0].status() {
        ShapesOpStatus::And => a == 2,
        ShapesOpStatus::Or => a == 4,
        _ => false,
    };
    let compatible1 = match geom_loops_store1[0][1].status() {
        ShapesOpStatus::And => b == 2,
        ShapesOpStatus::Or => b == 4,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 8,
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert_eq!(geom_loops_store1[1].len(), 2);
    assert!(
        geom_loops_store1[1][0].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[1][1].is_closed(),
        "{:?}",
        geom_loops_store1.display(wire_id_format)
    );
    assert!(
        geom_loops_store1[1][0].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    assert!(
        geom_loops_store1[1][1].is_geometric_consistent(),
        "{:?}",
        geom_loops_store1.display(wire_geom_format)
    );
    let (a, b) = (geom_loops_store1[1][0].len(), geom_loops_store1[1][1].len());
    let compatible0 = match geom_loops_store1[1][0].status() {
        ShapesOpStatus::And => a == 2,
        ShapesOpStatus::Or => a == 4,
        _ => false,
    };
    let compatible1 = match geom_loops_store1[1][1].status() {
        ShapesOpStatus::And => b == 2,
        ShapesOpStatus::Or => b == 4,
        _ => false,
    };
    assert!(
        compatible0 && compatible1 && a * b == 8,
        "{:?}",
        geom_loops_store0.display(wire_id_format)
    );
}

// ============================================================================
// Sprint A: Biangle wire creation in add_edge
// ============================================================================

fn line_bspline(
    v0: &Vertex<Point3>,
    v1: &Vertex<Point3>,
) -> Edge<Point3, BSplineCurve<Point3>> {
    let curve = BSplineCurve::new(KnotVec::bezier_knot(1), vec![v0.point(), v1.point()]);
    Edge::new(v0, v1, curve)
}

/// When add_edge finds no existing wire endpoint match (the (None, None) case),
/// it creates a 2-edge wire [edge.inverse(), edge] — a degenerate biangle.
/// This test documents the biangle creation behavior.
#[test]
fn add_edge_no_match_creates_biangle() {
    let v0 = Vertex::new(Point3::new(1.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(3.0, 0.0, 0.0));
    let edge = line_bspline(&v0, &v1);

    // Empty Loops — no existing wires for the edge endpoints to match
    let mut loops: Loops<Point3, BSplineCurve<Point3>> = Loops(Vec::new());
    loops.add_edge(edge, ShapesOpStatus::And);

    assert_eq!(loops.len(), 1, "add_edge should create one wire");
    assert_eq!(loops[0].len(), 2, "wire should have exactly 2 edges");
    assert!(
        is_biangle_wire(&loops[0]),
        "add_edge (None, None) creates a biangle wire"
    );
}

/// When add_edge matches one endpoint, the result should NOT be a biangle.
#[test]
fn add_edge_one_match_not_biangle() {
    let v0 = Vertex::new(Point3::new(0.0, 0.0, 0.0));
    let v1 = Vertex::new(Point3::new(2.0, 0.0, 0.0));
    let v2 = Vertex::new(Point3::new(4.0, 0.0, 0.0));

    // Start with a biangle wire from first edge (None, None case)
    let mut loops: Loops<Point3, BSplineCurve<Point3>> = Loops(Vec::new());
    loops.add_edge(line_bspline(&v0, &v1), ShapesOpStatus::And);
    assert_eq!(loops.len(), 1);

    // Add second edge that shares vertex v1 — should match and merge
    loops.add_edge(line_bspline(&v1, &v2), ShapesOpStatus::Or);

    // After merging, no wire should be a biangle
    for (i, bw) in loops.iter().enumerate() {
        assert!(
            !is_biangle_wire(bw),
            "wire {} should not be biangle after matching merge",
            i
        );
    }
}

// ============================================================================
// Sprint C: AABB overlap utility tests
// ============================================================================

/// Two AABBs far apart should not overlap.
#[test]
fn aabbs_overlap_disjoint() {
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
    assert!(
        !aabbs_overlap(&a, &b, 0.01),
        "disjoint AABBs must not overlap"
    );
}

/// Two AABBs with a small gap should overlap only with sufficient margin.
#[test]
fn aabbs_overlap_gap_within_margin() {
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([1.05, 0.0, 0.0], [2.0, 1.0, 1.0]);
    assert!(
        !aabbs_overlap(&a, &b, 0.0),
        "gap=0.05 should not overlap with margin=0"
    );
    assert!(
        !aabbs_overlap(&a, &b, 0.02),
        "gap=0.05 should not overlap with margin=0.02 (total=0.04)"
    );
    assert!(
        aabbs_overlap(&a, &b, 0.03),
        "gap=0.05 should overlap with margin=0.03 (total=0.06)"
    );
}

/// Two overlapping AABBs should always report overlap.
#[test]
fn aabbs_overlap_intersecting() {
    let a = ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = ([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
    assert!(
        aabbs_overlap(&a, &b, 0.0),
        "overlapping AABBs must always overlap"
    );
}

/// AABBs that are exactly touching should not overlap with zero margin
/// but should overlap with any positive margin.
#[test]
fn aabbs_overlap_touching() {
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    // Exactly touching: a.max[0]=1.0, b.min[0]=1.0
    // aabbs_overlap checks a.min[d] - margin > b.max[d] + margin
    // 0.0 - 0.0 > 2.0 + 0.0 → false (ok)
    // 1.0 - 0.0 > 1.0 + 0.0 → false (not strictly greater)
    // So touching AABBs DO overlap with margin=0.
    assert!(
        aabbs_overlap(&a, &b, 0.0),
        "touching AABBs overlap (not strictly separated)"
    );
}

/// AABB overlap is symmetric.
#[test]
fn aabbs_overlap_symmetric() {
    let a = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = ([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
    let margin = 0.6;
    assert_eq!(
        aabbs_overlap(&a, &b, margin),
        aabbs_overlap(&b, &a, margin),
        "AABB overlap must be symmetric"
    );
}
