use std::time::Duration;

/// Diagnostic report emitted after every boolean operation.
#[derive(Debug, Clone, Default)]
pub struct BooleanDiagnostics {
    /// Tolerance values used for this operation.
    pub tolerance: ToleranceReport,
    /// Face classification counts.
    pub classification: ClassificationReport,
    /// Topology repair counts.
    pub topology: TopologyReport,
    /// Timing per stage.
    pub timing: TimingReport,
    /// Intersection curve computation report.
    pub intersection: IntersectionReport,
    /// Face division report.
    pub division: DivisionReport,
    /// Shell closure recovery report.
    pub recovery: RecoveryReport,
    /// Edge-neighbor classification propagation report.
    pub edge_neighbor: EdgeNeighborReport,
    /// Warnings about near-tolerance decisions.
    pub warnings: Vec<String>,
}

/// Tolerance values used for a boolean operation.
#[derive(Debug, Clone, Default)]
pub struct ToleranceReport {
    /// Main coincidence/intersection tolerance.
    pub tau_model: f64,
    /// Mesh collision resolution tolerance.
    pub tau_mesh: f64,
    /// Vertex unification tolerance.
    pub tau_weld: f64,
    /// Coplanar face detection threshold.
    pub tau_coplanar: f64,
    /// IC-on-boundary filter tolerance.
    pub tau_boundary: f64,
    /// Phase 1 midpoint clustering tolerance.
    pub tau_edge_cluster: f64,
    /// Minimum parametric face area threshold.
    pub tau_area: f64,
}

/// Face classification counts from a boolean operation.
#[derive(Debug, Clone, Default)]
pub struct ClassificationReport {
    /// Faces that were split by intersection curves.
    pub faces_ic_split: usize,
    /// Faces classified as coplanar.
    pub faces_coplanar: usize,
    /// Faces classified via ray-cast.
    pub faces_ray_cast: usize,
    /// Faces integrated by connected component.
    pub faces_component_integrated: usize,
    /// Shell 0 faces classified as And (inside other solid).
    pub shell0_and: usize,
    /// Shell 0 faces classified as Or (outside other solid).
    pub shell0_or: usize,
    /// Shell 1 faces classified as And (inside other solid).
    pub shell1_and: usize,
    /// Shell 1 faces classified as Or (outside other solid).
    pub shell1_or: usize,
}

/// Topology repair counts from a boolean operation.
#[derive(Debug, Clone, Default)]
pub struct TopologyReport {
    /// Vertices unified during weld.
    pub vertices_welded: usize,
    /// Edges canonicalized in Phase 1.
    pub edges_canonicalized: usize,
    /// Wires split at repeated vertices.
    pub wires_split: usize,
    /// Wires repaired (non-simple to simple).
    pub wires_repaired: usize,
}

/// Intersection curve computation report.
#[derive(Debug, Clone, Default)]
pub struct IntersectionReport {
    /// Number of face pairs tested for intersection.
    pub face_pairs_tested: usize,
    /// Number of intersection curves produced.
    pub intersection_curves_produced: usize,
    /// Degenerate closed ICs filtered out.
    pub degenerate_closed_ics_filtered: usize,
    /// Short ICs filtered out.
    pub short_ics_filtered: usize,
    /// Coincident-edge ICs filtered out.
    pub coincident_edge_ics_filtered: usize,
}

/// Face division report.
#[derive(Debug, Clone, Default)]
pub struct DivisionReport {
    /// Faces successfully divided along intersection curves.
    pub faces_divided: usize,
    /// Total face fragments produced.
    pub total_fragments: usize,
    /// Faces where division failed.
    pub faces_division_failed: usize,
    /// Embedded wires removed during division.
    pub embedded_wires_removed: usize,
    /// Merge-splice recovery operations performed.
    pub merge_splice_recoveries: usize,
}

/// Shell closure recovery report.
#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    /// Recovery level reached (0 = no recovery needed, 1-6 = recovery stages).
    pub recovery_level: u8,
    /// Open edges remaining after weld.
    pub open_edges_after_weld: usize,
    /// Whether the Euler characteristic is valid (V-E+F=2).
    pub euler_valid: bool,
    /// Euler characteristic value.
    pub euler_chi: i64,
}

/// Edge-neighbor classification propagation report.
#[derive(Debug, Clone, Default)]
pub struct EdgeNeighborReport {
    /// Faces entered into edge-neighbor classification.
    pub faces_entered: usize,
    /// Rounds of iterative propagation executed.
    pub rounds_executed: usize,
    /// Faces resolved by edge-neighbor propagation.
    pub faces_resolved: usize,
}

/// Timing per stage of a boolean operation.
#[derive(Debug, Clone, Default)]
pub struct TimingReport {
    /// Time for loops_store creation (intersection curves).
    pub loops_store: Duration,
    /// Time for face division along intersection curves.
    pub divide_faces: Duration,
    /// Time for face classification (ray-cast + coplanar).
    pub classification: Duration,
    /// Time for edge welding and canonicalization.
    pub weld: Duration,
    /// Time for altshell-to-shell conversion.
    pub altshell: Duration,
    /// Total time for the entire boolean operation.
    pub total: Duration,
}
