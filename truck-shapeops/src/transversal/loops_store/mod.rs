#![allow(clippy::many_single_char_names)]

use super::*;
use rustc_hash::FxHashMap as HashMap;
use truck_base::cgmath64::*;
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::*;
use truck_topology::{Vertex, *};

type PolylineCurve = truck_meshalgo::prelude::PolylineCurve<Point3>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapesOpStatus {
    Unknown,
    And,
    Or,
}

impl ShapesOpStatus {
    fn not(self) -> Self {
        match self {
            Self::Unknown => Self::Unknown,
            Self::And => Self::Or,
            Self::Or => Self::And,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundaryWire<P, C> {
    wire: Wire<P, C>,
    status: ShapesOpStatus,
}

impl<P, C> BoundaryWire<P, C> {
    #[inline(always)]
    pub fn new(wire: Wire<P, C>, status: ShapesOpStatus) -> Self {
        Self { wire, status }
    }
    #[inline(always)]
    pub fn status(&self) -> ShapesOpStatus {
        self.status
    }
    #[inline(always)]
    pub fn invert(&mut self) {
        self.wire.invert();
        self.status = self.status.not();
    }
    #[inline(always)]
    pub fn inverse(&self) -> Self {
        Self {
            wire: self.wire.inverse(),
            status: self.status.not(),
        }
    }
}

impl ShapesOpStatus {
    fn from_is_curve<C, S0, S1>(curve: &IntersectionCurve<C, S0, S1>) -> Option<ShapesOpStatus>
    where
        C: ParametricCurve3D + BoundedCurve,
        S0: ParametricSurface3D + SearchNearestParameter<D2, Point = Point3>,
        S1: ParametricSurface3D + SearchNearestParameter<D2, Point = Point3>,
    {
        let (t0, t1) = curve.range_tuple();
        let t = (t0 + t1) / 2.0;
        let (_, pt0, pt1) = curve.search_triple(t, 100)?;
        let der = curve.leader().der(t);
        let normal0 = curve.surface0().normal(pt0[0], pt0[1]);
        let normal1 = curve.surface1().normal(pt1[0], pt1[1]);
        match normal0.cross(der).dot(normal1) > 0.0 {
            true => Some(ShapesOpStatus::Or),
            false => Some(ShapesOpStatus::And),
        }
    }
}

impl<P, C> std::ops::Deref for BoundaryWire<P, C> {
    type Target = Wire<P, C>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.wire
    }
}

impl<P, C> std::ops::DerefMut for BoundaryWire<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.wire
    }
}

#[derive(Clone, Debug)]
pub struct Loops<P, C>(Vec<BoundaryWire<P, C>>);
#[derive(Clone, Debug)]
pub struct LoopsStore<P, C>(Vec<Loops<P, C>>);

impl<P, C> std::ops::Deref for Loops<P, C> {
    type Target = Vec<BoundaryWire<P, C>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P, C> std::ops::DerefMut for Loops<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<P, C> std::ops::Deref for LoopsStore<P, C> {
    type Target = Vec<Loops<P, C>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P, C> std::ops::DerefMut for LoopsStore<P, C> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<P, C> FromIterator<BoundaryWire<P, C>> for Loops<P, C> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = BoundaryWire<P, C>>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<'a, P, C, S> From<&'a Face<P, C, S>> for Loops<P, C> {
    #[inline(always)]
    fn from(face: &'a Face<P, C, S>) -> Loops<P, C> {
        face.absolute_boundaries()
            .iter()
            .map(|wire| BoundaryWire::new(wire.clone(), ShapesOpStatus::Unknown))
            .collect()
    }
}

impl<'a, P: 'a, C: 'a, S: 'a> FromIterator<&'a Face<P, C, S>> for LoopsStore<P, C> {
    fn from_iter<I: IntoIterator<Item = &'a Face<P, C, S>>>(iter: I) -> Self {
        Self(iter.into_iter().map(Loops::from).collect())
    }
}

impl<'a, P, C> IntoIterator for &'a LoopsStore<P, C> {
    type Item = <&'a Vec<Loops<P, C>> as IntoIterator>::Item;
    type IntoIter = <&'a Vec<Loops<P, C>> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
enum ParameterKind {
    Front,
    Back,
    Inner(f64),
}

impl ParameterKind {
    fn try_new(t: f64, (t0, t1): (f64, f64)) -> Option<ParameterKind> {
        if t0.near(&t) {
            Some(ParameterKind::Front)
        } else if t1.near(&t) {
            Some(ParameterKind::Back)
        } else if t0 < t && t < t1 {
            Some(ParameterKind::Inner(t))
        } else {
            None
        }
    }
}

impl<P: Copy, C: Clone> Loops<P, C> {
    fn search_parameter(&self, pt: P) -> Option<(usize, usize, ParameterKind)>
    where
        C: BoundedCurve<Point = P> + SearchParameter<D1, Point = P>,
    {
        self.iter()
            .enumerate()
            .flat_map(move |(i, wire)| wire.iter().enumerate().map(move |(j, edge)| (i, j, edge)))
            .find_map(|(i, j, edge)| {
                let curve = edge.curve();
                curve.search_parameter(pt, None, 1).and_then(|t| {
                    let kind = ParameterKind::try_new(t, curve.range_tuple())?;
                    Some((i, j, kind))
                })
            })
    }

    fn change_vertex(
        &mut self,
        old_vertex: &Vertex<P>,
        new_vertex: &Vertex<P>,
        emap: &mut HashMap<EdgeID<C>, Edge<P, C>>,
    ) {
        self.iter_mut()
            .flat_map(|wire| wire.iter_mut())
            .for_each(|edge| {
                let mut new_edge = if edge.absolute_front() == old_vertex {
                    emap.entry(edge.id()).or_insert_with(|| {
                        Edge::new(new_vertex, edge.absolute_back(), edge.curve())
                    })
                } else if edge.absolute_back() == old_vertex {
                    emap.entry(edge.id()).or_insert_with(|| {
                        Edge::new(edge.absolute_front(), new_vertex, edge.curve())
                    })
                } else {
                    return;
                }
                .clone();
                if !edge.orientation() {
                    new_edge.invert();
                }
                // Remove the edge from the HashMap when it is no longer there because ID reassignment will occur.
                if edge.count() == 1 {
                    emap.remove(&edge.id());
                }
                *edge = new_edge;
            })
    }

    fn swap_edge_into_wire(&mut self, edge_id: EdgeID<C>, new_wire: &Wire<P, C>) {
        self.iter_mut().for_each(|wire| {
            let mut iter = wire.iter().enumerate();
            if let Some((idx, edge)) = iter.find(|(_, edge)| edge.id() == edge_id) {
                if edge.orientation() {
                    wire.swap_edge_into_wire(idx, new_wire.clone());
                } else {
                    wire.swap_edge_into_wire(idx, new_wire.inverse());
                }
            }
        });
    }

    #[inline(always)]
    pub(crate) fn add_independent_loop(&mut self, r#loop: BoundaryWire<P, C>) {
        self.push(r#loop.inverse());
        self.push(r#loop);
    }

    fn add_edge(
        &mut self,
        edge0: Edge<P, C>,
        status: ShapesOpStatus,
    ) -> [Option<(usize, usize)>; 2] {
        let a = self.iter().enumerate().find_map(|(i, wire)| {
            wire.iter().enumerate().find_map(|(j, edge)| {
                if edge.front() == edge0.back() {
                    Some((i, j))
                } else {
                    None
                }
            })
        });
        let b = self.iter().enumerate().find_map(|(i, wire)| {
            wire.iter().enumerate().find_map(|(j, edge)| {
                if edge.front() == edge0.front() {
                    Some((i, j))
                } else {
                    None
                }
            })
        });
        if let Some((wire_index0, edge_index0)) = a {
            self[wire_index0].rotate_left(edge_index0);
            self[wire_index0].push_front(edge0.clone());
            self[wire_index0].push_back(edge0.inverse());
        }
        match (a, b) {
            (Some((wire_index0, edge_index0)), Some((wire_index1, edge_index1))) => {
                if wire_index0 == wire_index1 {
                    let len = self[wire_index0].len() - 2;
                    let edge_index1 = (len + edge_index1 - edge_index0) % len + 1;
                    let new_wire = self[wire_index0].split_off(edge_index1);
                    self[wire_index0].status = status;
                    self.push(BoundaryWire::new(new_wire, status.not()));
                } else {
                    let mut new_wire0 = self[wire_index1].clone();
                    let mut new_wire1 = new_wire0.split_off(edge_index1);
                    new_wire0.append(&mut self[wire_index0]);
                    new_wire0.append(&mut new_wire1);
                    self[wire_index0] = new_wire0;
                    self.swap_remove(wire_index1);
                }
            }
            (None, Some((wire_index1, edge_index1))) => {
                self[wire_index1].rotate_left(edge_index1);
                self[wire_index1].push_front(edge0.inverse());
                self[wire_index1].push_back(edge0);
            }
            (None, None) => self.push(BoundaryWire::new(
                vec![edge0.inverse(), edge0].into(),
                ShapesOpStatus::Unknown,
            )),
            _ => {}
        }
        [a, b]
    }
}

impl<P: Copy + Tolerance, C: Clone> LoopsStore<P, C> {
    #[inline(always)]
    fn change_vertex(
        &mut self,
        old_vertex: &Vertex<P>,
        new_vertex: &Vertex<P>,
        emap: &mut HashMap<EdgeID<C>, Edge<P, C>>,
    ) {
        self.iter_mut()
            .for_each(|loops| loops.change_vertex(old_vertex, new_vertex, emap));
    }

    #[inline(always)]
    fn swap_edge_into_wire(&mut self, edge_id: EdgeID<C>, new_wire: &Wire<P, C>) {
        self.iter_mut()
            .for_each(|loops| loops.swap_edge_into_wire(edge_id, new_wire))
    }

    fn add_polygon_vertex(
        &mut self,
        loops_index: usize,
        v: &Vertex<P>,
        emap: &mut HashMap<EdgeID<C>, Edge<P, C>>,
    ) -> Option<(usize, usize, ParameterKind)>
    where
        C: Cut<Point = P> + SearchParameter<D1, Point = P>,
    {
        let pt = v.point();
        let (wire_index, edge_index, kind) = self[loops_index].search_parameter(pt)?;
        match kind {
            ParameterKind::Front => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_front()
                    .clone();
                self.change_vertex(&old_vertex, v, emap);
            }
            ParameterKind::Back => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_back()
                    .clone();
                self.change_vertex(&old_vertex, v, emap);
            }
            ParameterKind::Inner(t) => {
                let edge = self[loops_index][wire_index][edge_index].absolute_clone();
                let edge_id = edge.id();
                let (edge0, edge1) = edge.cut_with_parameter(v, t)?;
                let new_wire: Wire<_, _> = vec![edge0, edge1].into();
                self.swap_edge_into_wire(edge_id, &new_wire);
            }
        }
        Some((wire_index, edge_index, kind))
    }
}

impl<C> LoopsStore<Point3, C> {
    fn add_geom_vertex<S>(
        &mut self,
        (loops_index, wire_index, edge_index): (usize, usize, usize),
        v: &Vertex<Point3>,
        kind: ParameterKind,
        another_surface: &S,
        emap: &mut HashMap<EdgeID<C>, Edge<Point3, C>>,
    ) -> Option<()>
    where
        C: Cut<Point = Point3, Vector = Vector3> + SearchNearestParameter<D1, Point = Point3>,
        S: ParametricSurface3D + SearchNearestParameter<D2, Point = Point3>,
    {
        match kind {
            ParameterKind::Front => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_front()
                    .clone();
                v.set_point(old_vertex.point());
                self.change_vertex(&old_vertex, v, emap);
            }
            ParameterKind::Back => {
                let old_vertex = self[loops_index][wire_index][edge_index]
                    .absolute_back()
                    .clone();
                v.set_point(old_vertex.point());
                self.change_vertex(&old_vertex, v, emap);
            }
            ParameterKind::Inner(_) => {
                let curve = self[loops_index][wire_index][edge_index].curve();
                let (pt, t, _) =
                    curve_surface_projection(&curve, None, another_surface, None, v.point(), 100)?;
                v.set_point(pt);
                let edge = self[loops_index][wire_index][edge_index].absolute_clone();
                let edge_id = edge.id();
                let (edge0, edge1) = edge.cut_with_parameter(v, t)?;
                let new_wire: Wire<_, _> = vec![edge0, edge1].into();
                self.swap_edge_into_wire(edge_id, &new_wire);
            }
        }
        Some(())
    }
}

fn curve_surface_projection<C, S>(
    curve: &C,
    curve_hint: Option<f64>,
    surface: &S,
    surface_hint: Option<(f64, f64)>,
    point: Point3,
    trials: usize,
) -> Option<(Point3, f64, Point2)>
where
    C: ParametricCurve3D + SearchNearestParameter<D1, Point = Point3>,
    S: ParametricSurface3D + SearchNearestParameter<D2, Point = Point3>,
{
    if trials == 0 {
        return None;
    }
    let t = curve.search_nearest_parameter(point, curve_hint, 10)?;
    let pt0 = curve.subs(t);
    let (u, v) = surface.search_nearest_parameter(point, surface_hint, 10)?;
    let pt1 = surface.subs(u, v);
    if point.near(&pt0) && point.near(&pt1) && pt0.near(&pt1) {
        Some((point, t, Point2::new(u, v)))
    } else {
        let l = curve.der(t);
        let n = surface.normal(u, v);
        let t0 = (pt1 - pt0).dot(n) / l.dot(n);
        curve_surface_projection(
            curve,
            Some(t),
            surface,
            Some((u, v)),
            pt0 + t0 * l,
            trials - 1,
        )
    }
}

/// Check if a point is within tolerance of any boundary edge of a face.
/// Used to filter degenerate intersection curves that lie along shared boundaries.
///
/// Uses a tight boundary tolerance (tol * 0.5) to avoid filtering real ICs
/// that are near boundaries due to perturbation offsets. Coplanar shared-
/// boundary ICs have midpoints at ~0 distance from the boundary, while
/// perturbation-offset ICs have midpoints at ~tol distance.
fn is_midpoint_on_face_boundary<C, S>(mid: Point3, face: &Face<Point3, C, S>, tol: f64) -> bool {
    let boundary_tol = tol * 0.5;
    for wire in face.absolute_boundaries().iter() {
        for edge in wire.iter() {
            let p = edge.front().point();
            let q = edge.back().point();
            let seg = q - p;
            let seg_len_sq = seg.dot(seg);
            if seg_len_sq < 1e-30 {
                if (mid - p).magnitude() < boundary_tol {
                    return true;
                }
                continue;
            }
            let t = (mid - p).dot(seg) / seg_len_sq;
            let t = t.clamp(0.0, 1.0);
            let closest = p + seg * t;
            if (mid - closest).magnitude() < boundary_tol {
                return true;
            }
        }
    }
    false
}

fn create_independent_loop<P, C, D>(mut poly_curve0: C) -> Wire<P, D>
where
    C: Cut<Point = P>,
    D: From<C>,
{
    let (t0, t1) = poly_curve0.range_tuple();
    let t = (t0 + t1) / 2.0;
    let poly_curve1 = poly_curve0.cut(t);
    let v0 = Vertex::new(poly_curve0.front());
    let v1 = Vertex::new(poly_curve1.front());
    let edge0 = Edge::new(&v0, &v1, poly_curve0.into());
    let edge1 = Edge::new(&v1, &v0, poly_curve1.into());
    wire![edge0, edge1]
}

/// Compute the centroid of a set of 3D points.
fn compute_centroid(verts: &[Point3]) -> Point3 {
    let n = verts.len().max(1) as f64;
    let sum = verts.iter().fold(Vector3::new(0.0, 0.0, 0.0), |acc, &p| {
        acc + (p - Point3::origin())
    });
    Point3::origin() + sum / n
}

/// Clone a face's outer boundary wire and inject it as an independent loop
/// into the target loops_store, creating a hole (ring + disk split).
fn inject_boundary_wire<C: Clone>(
    source_face_wire: &Wire<Point3, C>,
    target_loops: &mut Loops<Point3, C>,
    status: ShapesOpStatus,
) {
    let cloned_wire = source_face_wire.clone();
    target_loops.add_independent_loop(BoundaryWire::new(cloned_wire, status));
}

/// For each coplanar face pair, check if one face's boundary is fully contained
/// within the other's. If so, inject the contained boundary as an independent
/// loop into the containing face's loops_store.
///
/// This handles through-holes: when a cylinder cuts completely through a box,
/// the cylinder's bottom circle is coplanar with the box's bottom face. Normal
/// intersection curve detection fails (parallel normals), but this function
/// detects the containment and creates the hole.
#[allow(clippy::too_many_arguments)]
fn inject_coplanar_boundary_loops<C, S>(
    geom_shell0: &Shell<Point3, C, S>,
    poly_shell0: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    geom_shell1: &Shell<Point3, C, S>,
    poly_shell1: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    geom_loops_store0: &mut LoopsStore<Point3, C>,
    poly_loops_store0: &mut LoopsStore<Point3, PolylineCurve>,
    geom_loops_store1: &mut LoopsStore<Point3, C>,
    poly_loops_store1: &mut LoopsStore<Point3, PolylineCurve>,
    coplanar_faces0: &rustc_hash::FxHashSet<usize>,
    coplanar_faces1: &rustc_hash::FxHashSet<usize>,
    tol: f64,
) where
    C: Clone,
    S: Clone,
{
    use super::coplanar::{make_tangent_basis, point_in_polygon, project_to_2d};
    use super::coplanar_splitting::{check_coplanar_faces, face_boundary_info};

    let store0_len = geom_loops_store0.len();
    let store1_len = geom_loops_store1.len();

    for &i in coplanar_faces0 {
        if i >= store0_len {
            continue;
        }
        for &j in coplanar_faces1 {
            if j >= store1_len {
                continue;
            }
            // Verify this specific pair is coplanar
            if check_coplanar_faces(&geom_shell0[i], &geom_shell1[j], tol).is_none() {
                continue;
            }

            // Get boundary info for both faces
            let (verts_i, normal_i) = match face_boundary_info(&geom_shell0[i]) {
                Some(info) => info,
                None => continue,
            };
            let (verts_j, _normal_j) = match face_boundary_info(&geom_shell1[j]) {
                Some(info) => info,
                None => continue,
            };

            if verts_i.len() < 3 || verts_j.len() < 3 {
                continue;
            }

            // Build 2D projection using face i's tangent basis
            let (u_axis, v_axis) = make_tangent_basis(normal_i);
            let origin = verts_i[0];

            let poly_i_2d: Vec<[f64; 2]> = verts_i
                .iter()
                .map(|&p| project_to_2d(p, origin, u_axis, v_axis))
                .collect();
            let poly_j_2d: Vec<[f64; 2]> = verts_j
                .iter()
                .map(|&p| project_to_2d(p, origin, u_axis, v_axis))
                .collect();

            // Test if face_j is fully contained within face_i
            let j_in_i = poly_j_2d.iter().all(|pt| point_in_polygon(*pt, &poly_i_2d));
            // Test if face_i is fully contained within face_j
            let i_in_j = poly_i_2d.iter().all(|pt| point_in_polygon(*pt, &poly_j_2d));

            if j_in_i {
                let centroid_j = compute_centroid(&verts_j);
                let nudge = centroid_j + normal_i * tol;
                let status = determine_injection_status(nudge, poly_shell1, tol);

                for wire in geom_shell1[j].absolute_boundaries().iter() {
                    inject_boundary_wire(wire, &mut geom_loops_store0[i], status);
                }
                for wire in poly_shell1[j].absolute_boundaries().iter() {
                    inject_boundary_wire(wire, &mut poly_loops_store0[i], status);
                }
            }

            if i_in_j {
                let centroid_i = compute_centroid(&verts_i);
                let (_, normal_j) = match face_boundary_info(&geom_shell1[j]) {
                    Some(info) => info,
                    None => continue,
                };
                let nudge = centroid_i + normal_j * tol;
                let status = determine_injection_status(nudge, poly_shell0, tol);

                for wire in geom_shell0[i].absolute_boundaries().iter() {
                    inject_boundary_wire(wire, &mut geom_loops_store1[j], status);
                }
                for wire in poly_shell0[i].absolute_boundaries().iter() {
                    inject_boundary_wire(wire, &mut poly_loops_store1[j], status);
                }
            }
        }
    }
}

/// Determine the ShapesOpStatus for an injected boundary loop via ray-cast.
fn determine_injection_status(
    nudge_point: Point3,
    other_poly_shell: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    _tol: f64,
) -> ShapesOpStatus {
    use super::integrate::{compute_shell_extent_poly, irrational_ray_dirs, try_ray_cast};

    let extent = compute_shell_extent_poly(other_poly_shell);
    let scale = extent.max(1.0);
    let perturb = Vector3::new(
        1.4142135623730951e-6 * scale,
        1.7320508075688772e-6 * scale,
        2.2360679774997896e-6 * scale,
    );
    let nudge_point = nudge_point + perturb;

    let dirs = irrational_ray_dirs();

    let mut inside = 0u32;
    let mut outside = 0u32;
    for &dir in &dirs {
        if let Some(c) = try_ray_cast(nudge_point, dir, other_poly_shell) {
            if c.unsigned_abs() % 2 == 1 {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }

    if inside > outside {
        ShapesOpStatus::And
    } else {
        ShapesOpStatus::Or
    }
}

pub struct LoopsStoreQuadruple<C> {
    pub geom_loops_store0: LoopsStore<Point3, C>,
    pub _poly_loops_store0: LoopsStore<Point3, PolylineCurve>,
    pub geom_loops_store1: LoopsStore<Point3, C>,
    pub _poly_loops_store1: LoopsStore<Point3, PolylineCurve>,
    /// Face indices in shell0 that are coplanar with some face in shell1.
    pub coplanar_faces0: rustc_hash::FxHashSet<usize>,
    /// Face indices in shell1 that are coplanar with some face in shell0.
    pub coplanar_faces1: rustc_hash::FxHashSet<usize>,
}

pub fn create_loops_stores<C, S>(
    geom_shell0: &Shell<Point3, C, S>,
    poly_shell0: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    geom_shell1: &Shell<Point3, C, S>,
    poly_shell1: &Shell<Point3, PolylineCurve, Option<PolygonMesh>>,
    tol: f64,
    coplanar_tol: Option<f64>,
) -> Option<LoopsStoreQuadruple<C>>
where
    C: SearchNearestParameter<D1, Point = Point3>
        + SearchParameter<D1, Point = Point3>
        + Cut<Point = Point3, Vector = Vector3>
        + From<IntersectionCurve<PolylineCurve, S, S>>,
    S: ParametricSurface3D + SearchNearestParameter<D2, Point = Point3>,
{
    let coplanar_tol = coplanar_tol.unwrap_or(tol);
    let mut geom_loops_store0: LoopsStore<_, _> = geom_shell0.face_iter().collect();
    let mut poly_loops_store0: LoopsStore<_, _> = poly_shell0.face_iter().collect();
    let mut geom_loops_store1: LoopsStore<_, _> = geom_shell1.face_iter().collect();
    let mut poly_loops_store1: LoopsStore<_, _> = poly_shell1.face_iter().collect();
    let store0_len = geom_loops_store0.len();
    let store1_len = geom_loops_store1.len();
    // Pre-identify coplanar face pairs for asymmetric classification later.
    let mut coplanar_faces0 = rustc_hash::FxHashSet::default();
    let mut coplanar_faces1 = rustc_hash::FxHashSet::default();
    // Track which specific pairs are coplanar for containment analysis.
    let mut coplanar_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..store0_len {
        for j in 0..store1_len {
            if coplanar_splitting::check_coplanar_faces(
                &geom_shell0[i],
                &geom_shell1[j],
                coplanar_tol,
            )
            .is_some()
            {
                coplanar_faces0.insert(i);
                coplanar_faces1.insert(j);
                coplanar_pairs.push((i, j));
            }
        }
    }
    // Pre-compute containment-based adjacency skip set.
    //
    // When a coplanar face is fully contained in its partner (e.g., a cylinder
    // cap inside a cube face for through-holes), the contained face's boundary
    // will be injected via inject_coplanar_boundary_loops. Intersection curves
    // from the container face with faces ADJACENT to the contained face would
    // duplicate the injected boundary, producing non-manifold topology.
    let coplanar_adj_skip = {
        use super::coplanar::{make_tangent_basis, point_in_polygon, project_to_2d};
        use super::coplanar_splitting::face_boundary_info;
        type Vid = VertexID<Point3>;

        // Collect boundary vertex IDs per face for adjacency detection
        let face_vids0: Vec<rustc_hash::FxHashSet<Vid>> = (0..store0_len)
            .map(|i| {
                geom_shell0[i]
                    .absolute_boundaries()
                    .iter()
                    .flat_map(|w| w.vertex_iter())
                    .map(|v| v.id())
                    .collect()
            })
            .collect();
        let face_vids1: Vec<rustc_hash::FxHashSet<Vid>> = (0..store1_len)
            .map(|j| {
                geom_shell1[j]
                    .absolute_boundaries()
                    .iter()
                    .flat_map(|w| w.vertex_iter())
                    .map(|v| v.id())
                    .collect()
            })
            .collect();

        let mut skip: rustc_hash::FxHashSet<(usize, usize)> = rustc_hash::FxHashSet::default();

        for &(i, j) in &coplanar_pairs {
            let (verts_i, normal_i) = match face_boundary_info(&geom_shell0[i]) {
                Some(info) => info,
                None => continue,
            };
            let (verts_j, _) = match face_boundary_info(&geom_shell1[j]) {
                Some(info) => info,
                None => continue,
            };
            if verts_i.len() < 3 || verts_j.len() < 3 {
                continue;
            }
            let (u_axis, v_axis) = make_tangent_basis(normal_i);
            let origin = verts_i[0];
            let poly_i_2d: Vec<[f64; 2]> = verts_i
                .iter()
                .map(|&p| project_to_2d(p, origin, u_axis, v_axis))
                .collect();
            let poly_j_2d: Vec<[f64; 2]> = verts_j
                .iter()
                .map(|&p| project_to_2d(p, origin, u_axis, v_axis))
                .collect();

            let j_in_i = poly_j_2d.iter().all(|pt| point_in_polygon(*pt, &poly_i_2d));
            let i_in_j = poly_i_2d.iter().all(|pt| point_in_polygon(*pt, &poly_j_2d));

            // When BOTH j_in_i and i_in_j are true, the faces have the same
            // extent (mutual containment). Neither is strictly contained in
            // the other, so we must NOT skip any adjacency — the coplanar
            // face pair itself is already handled downstream, and intersection
            // curves from adjacent faces are legitimate. Skipping both
            // directions would suppress ALL intersection curves, producing
            // Unknown faces and NotClosedShell errors.
            if j_in_i && !i_in_j {
                // face j (shell1) is strictly contained in face i (shell0).
                // Skip (i, k) for all faces k in shell1 adjacent to j.
                for k in 0..store1_len {
                    if k == j {
                        continue;
                    }
                    if face_vids1[k].iter().any(|vid| face_vids1[j].contains(vid)) {
                        skip.insert((i, k));
                    }
                }
            }
            if i_in_j && !j_in_i {
                // face i (shell0) is strictly contained in face j (shell1).
                // Skip (k, j) for all faces k in shell0 adjacent to i.
                for k in 0..store0_len {
                    if k == i {
                        continue;
                    }
                    if face_vids0[k].iter().any(|vid| face_vids0[i].contains(vid)) {
                        skip.insert((k, j));
                    }
                }
            }
        }

        skip
    };
    (0..store0_len)
        .flat_map(move |i| (0..store1_len).map(move |j| (i, j)))
        .try_for_each(|(face_index0, face_index1)| {
            // Skip when THIS SPECIFIC pair is coplanar.
            if coplanar_faces0.contains(&face_index0)
                && coplanar_faces1.contains(&face_index1)
                && coplanar_splitting::check_coplanar_faces(
                    &geom_shell0[face_index0],
                    &geom_shell1[face_index1],
                    tol,
                )
                .is_some()
            {
                return Some(());
            }
            // Skip coplanar-adjacent pairs when containment holds.
            // These produce intersection curves that duplicate the coplanar
            // boundary injection, causing non-manifold topology.
            if coplanar_adj_skip.contains(&(face_index0, face_index1)) {
                return Some(());
            }
            let ori0 = geom_shell0[face_index0].orientation();
            let ori1 = geom_shell1[face_index1].orientation();
            let surface0 = geom_shell0[face_index0].surface();
            let surface1 = geom_shell1[face_index1].surface();
            let polygon0 = poly_shell0[face_index0].surface()?;
            let polygon1 = poly_shell1[face_index1].surface()?;
            let ics = intersection_curve::intersection_curves(
                surface0.clone(),
                &polygon0,
                surface1.clone(),
                &polygon1,
            )?;
            ics.into_iter()
                .try_for_each(|(polyline, intersection_curve)| {
                    let mut intersection_curve = intersection_curve.into();
                    let status = ShapesOpStatus::from_is_curve(&intersection_curve)?;
                    let (status0, status1) = match (ori0, ori1) {
                        (true, true) => (status, status.not()),
                        (true, false) => (status.not(), status.not()),
                        (false, true) => (status, status),
                        (false, false) => (status.not(), status),
                    };
                    if polyline.front().near(&polyline.back()) {
                        // Pre-validate: skip degenerate closed ICs where all points
                        // collapse to essentially the same location.
                        let is_degenerate = {
                            let pts: &Vec<Point3> = &polyline.0;
                            if pts.len() < 2 {
                                true
                            } else {
                                let center = pts[0];
                                pts.iter().all(|p| (*p - center).magnitude() < tol)
                            }
                        };
                        if is_degenerate {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[boolean] Skipping degenerate closed IC: {} points within tol",
                                polyline.0.len()
                            );
                        } else {
                            let geom_wire = create_independent_loop(intersection_curve);
                            let poly_wire = create_independent_loop(polyline);
                            poly_loops_store0[face_index0].add_independent_loop(BoundaryWire::new(
                                poly_wire.clone(),
                                status0,
                            ));
                            poly_loops_store1[face_index1]
                                .add_independent_loop(BoundaryWire::new(poly_wire, status1));
                            geom_loops_store0[face_index0].add_independent_loop(BoundaryWire::new(
                                geom_wire.clone(),
                                status0,
                            ));
                            geom_loops_store1[face_index1]
                                .add_independent_loop(BoundaryWire::new(geom_wire, status1));
                        }
                    } else {
                        // Pre-filter degenerate boundary-touching curves BEFORE any
                        // vertex insertion. Coplanar-adjacent intersection curves may
                        // lie along shared boundary edges, corrupting loop stores.
                        let mid = polyline.front().midpoint(polyline.back());
                        let on_boundary0 =
                            is_midpoint_on_face_boundary(mid, &geom_shell0[face_index0], tol);
                        let on_boundary1 =
                            is_midpoint_on_face_boundary(mid, &geom_shell1[face_index1], tol);
                        if on_boundary0 && on_boundary1 {
                            return Some(());
                        }
                        // Wrap vertex insertion in a defensive closure: if vertex
                        // projection fails (e.g., at coplanar face boundaries), skip
                        // this curve rather than aborting the entire pipeline.
                        let _ = (|| -> Option<()> {
                            let pv0 = Vertex::new(polyline.front());
                            let pv1 = Vertex::new(polyline.back());
                            let gv0 = Vertex::new(polyline.front());
                            let gv1 = Vertex::new(polyline.back());
                            let mut pemap0 = HashMap::default();
                            let mut pemap1 = HashMap::default();
                            let mut gemap0 = HashMap::default();
                            let mut gemap1 = HashMap::default();
                            if let Some((wire_index, edge_index, kind)) =
                                poly_loops_store0.add_polygon_vertex(face_index0, &pv0, &mut pemap0)
                            {
                                geom_loops_store0.add_geom_vertex(
                                    (face_index0, wire_index, edge_index),
                                    &gv0,
                                    kind,
                                    &surface1,
                                    &mut gemap0,
                                )?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.first_mut().unwrap() = gv0.point();
                            }
                            if let Some((wire_index, edge_index, kind)) =
                                poly_loops_store0.add_polygon_vertex(face_index0, &pv1, &mut pemap1)
                            {
                                geom_loops_store0.add_geom_vertex(
                                    (face_index0, wire_index, edge_index),
                                    &gv1,
                                    kind,
                                    &surface1,
                                    &mut gemap1,
                                )?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.last_mut().unwrap() = gv1.point();
                            }
                            if let Some((wire_index, edge_index, kind)) =
                                poly_loops_store1.add_polygon_vertex(face_index1, &pv0, &mut pemap0)
                            {
                                geom_loops_store1.add_geom_vertex(
                                    (face_index1, wire_index, edge_index),
                                    &gv0,
                                    kind,
                                    &surface0,
                                    &mut gemap0,
                                )?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.first_mut().unwrap() = gv0.point();
                            }
                            if let Some((wire_index, edge_index, kind)) =
                                poly_loops_store1.add_polygon_vertex(face_index1, &pv1, &mut pemap1)
                            {
                                geom_loops_store1.add_geom_vertex(
                                    (face_index1, wire_index, edge_index),
                                    &gv1,
                                    kind,
                                    &surface0,
                                    &mut gemap1,
                                )?;
                                let polyline = intersection_curve.leader_mut();
                                *polyline.last_mut().unwrap() = gv1.point();
                            }
                            let pedge = Edge::new(&pv0, &pv1, polyline);
                            let gedge = Edge::new(&gv0, &gv1, intersection_curve.into());
                            poly_loops_store0[face_index0].add_edge(pedge.clone(), status0);
                            geom_loops_store0[face_index0].add_edge(gedge.clone(), status0);
                            poly_loops_store1[face_index1].add_edge(pedge, status1);
                            geom_loops_store1[face_index1].add_edge(gedge, status1);
                            Some(())
                        })();
                    }
                    Some(())
                })
        })?;

    // Inject coplanar boundary loops for through-hole detection.
    // When a cylindrical cut exits through the opposite face, the cylinder's
    // cap is coplanar with that face. Normal intersection curves fail (parallel
    // normals), so we detect full containment and inject the contained face's
    // boundary as a hole in the containing face.
    if !coplanar_faces0.is_empty() || !coplanar_faces1.is_empty() {
        inject_coplanar_boundary_loops(
            geom_shell0,
            poly_shell0,
            geom_shell1,
            poly_shell1,
            &mut geom_loops_store0,
            &mut poly_loops_store0,
            &mut geom_loops_store1,
            &mut poly_loops_store1,
            &coplanar_faces0,
            &coplanar_faces1,
            tol,
        );
    }

    Some(LoopsStoreQuadruple {
        geom_loops_store0,
        _poly_loops_store0: poly_loops_store0,
        geom_loops_store1,
        _poly_loops_store1: poly_loops_store1,
        coplanar_faces0,
        coplanar_faces1,
    })
}

#[cfg(test)]
mod tests;
