// Phase 1 infrastructure — types used in interference.rs tests and future pipeline.
#![allow(dead_code)]

//! Pave Block types for the deterministic boolean pipeline.
//!
//! A *pave block* is a segment of a boundary edge between two IC crossings
//! (or between an original vertex and an IC crossing). By computing explicit
//! IC-edge intersection records ("interference"), we avoid tolerance-based
//! endpoint projection and the figure-8 wire bug (MV3).
//!
//! These types are the Phase 1 data model. Phase 2 wires the interference
//! table into face division; Phase 3 replaces divide_one_face entirely.

use std::collections::BTreeMap;
use truck_base::cgmath64::*;
use truck_topology::*;

use super::loops_store::ShapesOpStatus;

/// Crossing point where an IC meets a face boundary edge.
///
/// Records the precise parameter on both the boundary edge and the IC
/// polyline, plus a flag for corner-touch degeneracies (MV3 fix).
#[derive(Debug, Clone)]
pub struct IcVertex {
    /// Shared truck vertex at the crossing point (SequentialID-based).
    pub vertex: Vertex<Point3>,
    /// Parameter on the boundary edge \[0,1\] range.
    pub edge_param: f64,
    /// Which IC this crossing belongs to (index into the face's IC list).
    pub ic_index: usize,
    /// Parameter along the IC polyline (cumulative arc-length fraction).
    pub ic_param: f64,
    /// True if the IC touches a pre-existing boundary vertex (no edge split
    /// needed). Both edges meeting at that corner keep their original
    /// endpoints; the IC segment starts/ends at the existing corner vertex.
    pub is_corner_touch: bool,
}

/// Segment of a boundary edge between IC crossings.
///
/// Given N crossings on a boundary edge, there are N+1 pave blocks. Each
/// records the sub-range of the original edge and which IC (if any) bounds
/// its start and end.
#[derive(Debug, Clone)]
pub struct PaveBlock<C> {
    /// Reference to the original full boundary edge.
    pub original_edge: Edge<Point3, C>,
    /// Start vertex of this pave block segment.
    pub start_vertex: Vertex<Point3>,
    /// End vertex of this pave block segment.
    pub end_vertex: Vertex<Point3>,
    /// Parameter range on the original edge \[0,1\].
    pub param_range: (f64, f64),
    /// Sub-edge if the pave block is a proper sub-segment. `None` if this
    /// pave block spans the full original edge (no crossings).
    pub sub_edge: Option<Edge<Point3, C>>,
    /// IC index at the start of this segment. `None` means the segment
    /// starts at the original edge's front vertex.
    pub start_ic: Option<usize>,
    /// IC index at the end of this segment. `None` means the segment
    /// ends at the original edge's back vertex.
    pub end_ic: Option<usize>,
}

/// Segment of an IC between boundary crossings on a face.
///
/// Each IC crossing pair on the same face produces an `IcSegment` that
/// represents the portion of the IC lying inside that face.
#[derive(Debug, Clone)]
pub struct IcSegment<C> {
    /// Which IC this segment comes from.
    pub ic_index: usize,
    /// The edge representing this IC segment.
    pub edge: Edge<Point3, C>,
    /// Classification status of this IC segment (And/Or/Unknown).
    pub status: ShapesOpStatus,
}

/// All pave blocks and IC segments for one face.
///
/// This is the per-face view of the interference table: every boundary
/// edge is decomposed into pave blocks, and every IC passing through the
/// face has its interior segments recorded.
#[derive(Debug, Clone)]
pub struct FaceInterference<C> {
    /// Pave blocks for each boundary edge, keyed by EdgeID for
    /// deterministic iteration (BTreeMap).
    pub edge_pave_blocks: BTreeMap<EdgeID<C>, Vec<PaveBlock<C>>>,
    /// IC segments passing through this face.
    pub ic_segments: Vec<IcSegment<C>>,
    /// All IC crossing vertices on this face's boundary.
    pub ic_vertices: Vec<IcVertex>,
}

impl<C> FaceInterference<C> {
    /// Create an empty `FaceInterference`.
    pub fn new() -> Self {
        FaceInterference {
            edge_pave_blocks: BTreeMap::new(),
            ic_segments: Vec::new(),
            ic_vertices: Vec::new(),
        }
    }

    /// Record a new IC crossing on this face's boundary.
    pub fn add_crossing(&mut self, crossing: IcVertex) {
        self.ic_vertices.push(crossing);
    }

    /// Number of IC crossings recorded so far.
    pub fn crossing_count(&self) -> usize {
        self.ic_vertices.len()
    }

    /// Sort crossings by `(edge_param, ic_index)` for deterministic
    /// processing when building pave blocks.
    pub fn sort_crossings(&mut self) {
        self.ic_vertices.sort_by(|a, b| {
            a.edge_param
                .partial_cmp(&b.edge_param)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.ic_index.cmp(&b.ic_index))
        });
    }

    /// Build pave blocks for a single boundary edge from the collected
    /// crossings. Crossings must already be filtered to this edge and
    /// sorted by `edge_param`. Creates N+1 pave blocks for N crossings.
    ///
    /// `edge_crossings` — the crossings on this specific edge, sorted by
    ///   `edge_param`.
    /// `edge` — the original boundary edge.
    pub fn build_edge_pave_blocks(
        edge: &Edge<Point3, C>,
        edge_crossings: &[IcVertex],
    ) -> Vec<PaveBlock<C>>
    where
        C: Clone,
    {
        let mut blocks = Vec::with_capacity(edge_crossings.len() + 1);
        let front = edge.front().clone();
        let back = edge.back().clone();

        if edge_crossings.is_empty() {
            // No crossings → single full-span pave block.
            blocks.push(PaveBlock {
                original_edge: edge.clone(),
                start_vertex: front,
                end_vertex: back,
                param_range: (0.0, 1.0),
                sub_edge: None,
                start_ic: None,
                end_ic: None,
            });
            return blocks;
        }

        // First block: from edge front to first crossing
        let first = &edge_crossings[0];
        blocks.push(PaveBlock {
            original_edge: edge.clone(),
            start_vertex: front,
            end_vertex: first.vertex.clone(),
            param_range: (0.0, first.edge_param),
            sub_edge: None, // Sub-edge creation deferred to Phase 2
            start_ic: None,
            end_ic: Some(first.ic_index),
        });

        // Interior blocks: between consecutive crossings
        for window in edge_crossings.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            blocks.push(PaveBlock {
                original_edge: edge.clone(),
                start_vertex: prev.vertex.clone(),
                end_vertex: next.vertex.clone(),
                param_range: (prev.edge_param, next.edge_param),
                sub_edge: None,
                start_ic: Some(prev.ic_index),
                end_ic: Some(next.ic_index),
            });
        }

        // Last block: from last crossing to edge back
        let last = &edge_crossings[edge_crossings.len() - 1];
        blocks.push(PaveBlock {
            original_edge: edge.clone(),
            start_vertex: last.vertex.clone(),
            end_vertex: back,
            param_range: (last.edge_param, 1.0),
            sub_edge: None,
            start_ic: Some(last.ic_index),
            end_ic: None,
        });

        blocks
    }
}

impl<C> Default for FaceInterference<C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Full interference table for a shell pair.
///
/// Indexed by face index within each shell. Each entry contains the
/// pave blocks and IC segments for that face.
#[derive(Debug, Clone)]
pub struct InterferenceTable<C> {
    /// Per-face interference for shell 0.
    pub shell0: Vec<FaceInterference<C>>,
    /// Per-face interference for shell 1.
    pub shell1: Vec<FaceInterference<C>>,
}

impl<C> InterferenceTable<C> {
    /// Create an empty interference table sized for the given shell face
    /// counts.
    pub fn new(shell0_face_count: usize, shell1_face_count: usize) -> Self {
        InterferenceTable {
            shell0: (0..shell0_face_count)
                .map(|_| FaceInterference::new())
                .collect(),
            shell1: (0..shell1_face_count)
                .map(|_| FaceInterference::new())
                .collect(),
        }
    }

    /// Total number of IC crossings across both shells.
    pub fn total_crossings(&self) -> usize {
        let s0: usize = self.shell0.iter().map(|fi| fi.crossing_count()).sum();
        let s1: usize = self.shell1.iter().map(|fi| fi.crossing_count()).sum();
        s0 + s1
    }

    /// Total number of IC segments across both shells.
    pub fn total_ic_segments(&self) -> usize {
        let s0: usize = self.shell0.iter().map(|fi| fi.ic_segments.len()).sum();
        let s1: usize = self.shell1.iter().map(|fi| fi.ic_segments.len()).sum();
        s0 + s1
    }
}

impl<C> std::fmt::Display for InterferenceTable<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InterferenceTable {{ shell0: {} faces, shell1: {} faces, crossings: {}, ic_segments: {} }}",
            self.shell0.len(),
            self.shell1.len(),
            self.total_crossings(),
            self.total_ic_segments(),
        )
    }
}
