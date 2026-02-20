#![allow(clippy::many_single_char_names)]

use super::faces_classification::FacesClassification;
use super::loops_store::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::Deref;
use truck_meshalgo::prelude::*;
use truck_topology::*;

fn create_parameter_boundary<P, C, S>(
    face: &Face<P, C, S>,
    wire: &Wire<P, C>,
    polys: &mut HashMap<EdgeID<C>, PolylineCurve<P>>,
    tol: f64,
) -> Option<PolylineCurve<Point2>>
where
    P: Copy,
    C: BoundedCurve<Point = P> + ParameterDivision1D<Point = P>,
    S: Clone + SearchParameter<D2, Point = P>,
{
    let surface = face.surface();
    let pt = wire.front_vertex().unwrap().point();
    let p: Point2 = surface.search_parameter(pt, None, 100)?.into();
    let vec = wire.edge_iter().try_fold(vec![p], |mut vec, edge| {
        let poly = polys.entry(edge.id()).or_insert_with(|| {
            let curve = edge.curve();
            let div = curve.parameter_division(curve.range_tuple(), tol).1;
            PolylineCurve(div)
        });
        let mut p = *vec.last().unwrap();
        let closure = |q: &P| -> Option<Point2> {
            p = surface.search_parameter(*q, Some(p.into()), 100)?.into();
            Some(p)
        };
        let add: Option<Vec<Point2>> = match edge.orientation() {
            true => poly.iter().skip(1).map(closure).collect(),
            false => poly.iter().rev().skip(1).map(closure).collect(),
        };
        vec.append(&mut add?);
        Some(vec)
    })?;
    Some(PolylineCurve(vec))
}

#[derive(Clone, Debug)]
struct WireChunk<'a, C> {
    poly: PolylineCurve<Point2>,
    wire: &'a BoundaryWire<Point3, C>,
}

type FaceWithShapesOpStatus<C, S> = (Face<Point3, C, S>, ShapesOpStatus);
fn divide_one_face<C, S>(
    face: &Face<Point3, C, S>,
    loops: &Loops<Point3, C>,
    tol: f64,
) -> Option<Vec<FaceWithShapesOpStatus<C, S>>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let (mut pre_faces, mut negative_wires) = (Vec::new(), Vec::new());
    let mut map = HashMap::default();
    loops.iter().try_for_each(|wire| {
        let poly = create_parameter_boundary(face, wire, &mut map, tol)?;
        let area = poly.area();
        // Skip degenerate loops with negligible area (coplanar artifacts)
        if area.abs() < tol {
            return Some(());
        }
        match area > 0.0 {
            true => pre_faces.push(vec![WireChunk { poly, wire }]),
            false => negative_wires.push(WireChunk { poly, wire }),
        }
        Some(())
    })?;
    negative_wires.into_iter().try_for_each(|chunk| {
        let pt = chunk.poly.front();
        let idx = pre_faces.iter().position(|face| face[0].poly.include(pt));
        if let Some(i) = idx {
            let outer_area = pre_faces[i][0].poly.area();
            let chunk_area = chunk.poly.area();
            // When inner loop exactly matches outer boundary (areas cancel),
            // the face is consumed by the intersection — remove it.
            if (outer_area + chunk_area).abs() < tol {
                pre_faces[i].clear();
            } else {
                pre_faces[i].push(chunk);
            }
        }
        Some(())
    })?;
    let vec: Vec<_> = pre_faces
        .into_iter()
        .filter(|pre_face| !pre_face.is_empty())
        .filter_map(|pre_face| {
            let surface = face.surface();
            let op = pre_face
                .iter()
                .find(|chunk| chunk.wire.status() != ShapesOpStatus::Unknown);
            let status = match op {
                Some(chunk) => chunk.wire.status(),
                None => ShapesOpStatus::Unknown,
            };
            let wires: Vec<Wire<Point3, C>> = pre_face
                .into_iter()
                .map(|chunk| chunk.wire.deref().clone())
                .collect();
            match Face::try_new(wires, surface) {
                Ok(mut new_face) => {
                    if !face.orientation() {
                        new_face.invert();
                    }
                    Some((new_face, status))
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[boolean] Face::try_new failed in divide_one_face: {:?}",
                        _e
                    );
                    None
                }
            }
        })
        .collect();
    Some(vec)
}

#[allow(dead_code)]
pub fn divide_faces<C, S>(
    shell: &Shell<Point3, C, S>,
    loops_store: &LoopsStore<Point3, C>,
    tol: f64,
) -> Option<FacesClassification<Point3, C, S>>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let (cls, _) =
        divide_faces_with_coplanar(shell, loops_store, tol, &rustc_hash::FxHashSet::default())?;
    Some(cls)
}

/// Like `divide_faces` but tracks fragments from coplanar faces.
/// Returns (classification, coplanar_fragment_face_ids) so the caller can re-force
/// coplanar fragments to Unknown after `integrate_by_component`.
#[allow(clippy::type_complexity)]
pub fn divide_faces_with_coplanar<C, S>(
    shell: &Shell<Point3, C, S>,
    loops_store: &LoopsStore<Point3, C>,
    tol: f64,
    coplanar_faces: &rustc_hash::FxHashSet<usize>,
) -> Option<(FacesClassification<Point3, C, S>, Vec<FaceID<S>>)>
where
    C: BoundedCurve<Point = Point3> + ParameterDivision1D<Point = Point3>,
    S: Clone + SearchParameter<D2, Point = Point3>,
{
    let mut res = FacesClassification::<Point3, C, S>::default();
    let mut coplanar_fragment_ids = Vec::new();
    shell
        .iter()
        .zip(loops_store)
        .enumerate()
        .try_for_each(|(idx, (face, loops))| {
            let is_coplanar = coplanar_faces.contains(&idx);
            if loops
                .iter()
                .all(|wire| wire.status() == ShapesOpStatus::Unknown)
            {
                // Rebuild from loops_store wires (not face.clone()) to preserve
                // vertex substitutions from add_polygon_vertex. This is needed
                // for weld_coincident_edges to find shared Vertex objects.
                let wires: Vec<Wire<Point3, C>> =
                    loops.iter().map(|bw| bw.deref().clone()).collect();
                let rebuilt = match Face::try_new(wires, face.surface()) {
                    Ok(mut f) => {
                        if !face.orientation() {
                            f.invert();
                        }
                        f
                    }
                    Err(_) => face.clone(),
                };
                if is_coplanar {
                    coplanar_fragment_ids.push(rebuilt.id());
                }
                res.push(rebuilt, ShapesOpStatus::Unknown);
            } else {
                // Pre-validate: check for degenerate loops that would cause
                // panics in parameter_division. If any wire has near-zero
                // spatial extent, skip division and use the undivided face.
                let has_degenerate_loop = loops.iter().any(|wire| {
                    let pts: Vec<Point3> = wire.vertex_iter().map(|v| v.point()).collect();
                    if pts.len() < 2 {
                        return true;
                    }
                    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
                    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
                    let (mut min_z, mut max_z) = (f64::MAX, f64::MIN);
                    for p in &pts {
                        min_x = min_x.min(p.x);
                        max_x = max_x.max(p.x);
                        min_y = min_y.min(p.y);
                        max_y = max_y.max(p.y);
                        min_z = min_z.min(p.z);
                        max_z = max_z.max(p.z);
                    }
                    let extent = (max_x - min_x).max(max_y - min_y).max(max_z - min_z);
                    extent < tol * 0.1
                });

                if has_degenerate_loop {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[boolean] Skipping degenerate loop in divide_face \
                         — using undivided face"
                    );
                    if is_coplanar {
                        coplanar_fragment_ids.push(face.id());
                    }
                    res.push(face.clone(), ShapesOpStatus::Unknown);
                } else {
                    match divide_one_face(face, loops, tol) {
                        Some(vec) => {
                            vec.into_iter().for_each(|(face, status)| {
                                if is_coplanar {
                                    coplanar_fragment_ids.push(face.id());
                                }
                                res.push(face, status);
                            });
                        }
                        None => return None,
                    }
                }
            }
            Some(())
        })?;
    Some((res, coplanar_fragment_ids))
}

#[cfg(test)]
mod tests;
