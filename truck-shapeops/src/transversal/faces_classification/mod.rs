use super::coplanar;
use super::integrate::ShapeOpsSurface;
use super::loops_store::ShapesOpStatus;
// Order-insensitive: FxHashMap<FaceID, Status> is used as a lookup table only.
// Iteration over faces uses self.shell (a Vec, deterministic order), not the map.
use rustc_hash::FxHashMap as HashMap;
use truck_base::cgmath64::*;
use truck_topology::*;

#[derive(Clone, Debug)]
pub struct FacesClassification<P, C, S> {
    shell: Shell<P, C, S>,
    status: HashMap<FaceID<S>, ShapesOpStatus>,
}

impl<P, C, S> Default for FacesClassification<P, C, S> {
    fn default() -> Self {
        Self {
            shell: Default::default(),
            status: HashMap::default(),
        }
    }
}

impl<P, C, S> FacesClassification<P, C, S> {
    pub fn push(&mut self, face: Face<P, C, S>, status: ShapesOpStatus) {
        self.status.insert(face.id(), status);
        self.shell.push(face);
    }

    pub fn and_or_unknown(&self) -> [Shell<P, C, S>; 3] {
        let [mut and, mut or, mut unknown] = <[Shell<P, C, S>; 3]>::default();
        for face in &self.shell {
            match self.status.get(&face.id()).unwrap() {
                ShapesOpStatus::And => and.push(face.clone()),
                ShapesOpStatus::Or => or.push(face.clone()),
                ShapesOpStatus::Unknown => unknown.push(face.clone()),
            }
        }
        [and, or, unknown]
    }

    pub fn integrate_by_component(&mut self) {
        use rustc_hash::FxHashSet;
        let [and, or, unknown] = self.and_or_unknown();
        let and_edge_ids: FxHashSet<EdgeID<C>> = and
            .extract_boundaries()
            .iter()
            .flatten()
            .map(|edge| edge.id())
            .collect();
        let or_edge_ids: FxHashSet<EdgeID<C>> = or
            .extract_boundaries()
            .iter()
            .flatten()
            .map(|edge| edge.id())
            .collect();
        let components = unknown.connected_components();
        for comp in components {
            let boundary = comp.extract_boundaries();
            let mut and_count = 0usize;
            let mut or_count = 0usize;
            for edge in boundary.iter().flatten() {
                if and_edge_ids.contains(&edge.id()) {
                    and_count += 1;
                }
                if or_edge_ids.contains(&edge.id()) {
                    or_count += 1;
                }
            }
            let status = if and_count > 0 && or_count == 0 {
                Some(ShapesOpStatus::And)
            } else if or_count > 0 && and_count == 0 {
                Some(ShapesOpStatus::Or)
            } else if and_count > or_count {
                Some(ShapesOpStatus::And)
            } else if or_count > and_count {
                Some(ShapesOpStatus::Or)
            } else {
                // Tied or no boundary matches — leave as Unknown for ray_cast_classify.
                // The improved 8-ray ray_cast_classify with escalated perturbation and
                // face-normal fallback handles these cases more accurately than a
                // static tiebreak preference.
                None
            };
            if let Some(s) = status {
                comp.iter().for_each(|face| {
                    *self.status.get_mut(&face.id()).unwrap() = s;
                })
            }
        }
    }
}

impl<C, S> FacesClassification<Point3, C, S> {
    /// Reset coplanar fragment statuses to Unknown, but only for fragments
    /// that actually overlap with a coplanar face in the other shell.
    /// Non-overlapping fragments (e.g., ring faces around holes) keep their
    /// intersection curve status.
    ///
    /// `C2` may differ from `C` (e.g., Alternative<..> vs raw curve type) because
    /// the classified shell uses wrapped curves while the other shell has original curves.
    pub fn reset_overlapping_coplanar<C2>(
        &mut self,
        coplanar_fids: &[FaceID<S>],
        other_shell: &Shell<Point3, C2, S>,
        is_shell0: bool,
        tol: f64,
    ) where
        S: ShapeOpsSurface,
    {
        for fid in coplanar_fids {
            let face = match self.shell.iter().find(|f| &f.id() == fid) {
                Some(f) => f.clone(),
                None => continue,
            };
            if coplanar::classify_coplanar_fragment(&face, other_shell, is_shell0, tol).is_some() {
                if let Some(status) = self.status.get_mut(fid) {
                    *status = ShapesOpStatus::Unknown;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
