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
