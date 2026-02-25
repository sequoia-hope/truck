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
    /// Pre-heal vertex unification report.
    pub pre_heal: Option<PreHealReport>,
    /// Perturbation cascade report.
    pub cascade: Option<CascadeReport>,
    /// Classification method summary.
    pub classification_summary: Option<ClassificationSummary>,
    /// Warnings about near-tolerance decisions.
    pub warnings: Vec<String>,
}

impl BooleanDiagnostics {
    /// One-line human-readable summary of key metrics.
    pub fn summary_line(&self) -> String {
        let c = &self.classification;
        let total_classified = c.shell0_and + c.shell0_or + c.shell1_and + c.shell1_or;
        let r = &self.recovery;
        let en = &self.edge_neighbor;
        let mut parts = vec![
            format!("faces={}", total_classified),
            format!("and={}+{}", c.shell0_and, c.shell1_and),
            format!("or={}+{}", c.shell0_or, c.shell1_or),
        ];
        if r.recovery_level > 0 {
            parts.push(format!("recovery=L{}", r.recovery_level));
        }
        if en.faces_resolved > 0 {
            parts.push(format!("edge_nbr={}", en.faces_resolved));
        }
        if let Some(ref cs) = self.cascade {
            parts.push(format!("cascade={}", cs.attempts));
        }
        format!("[bool-diag] {}", parts.join(" "))
    }
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

/// Pre-heal vertex unification report.
#[derive(Debug, Clone, Default)]
pub struct PreHealReport {
    /// Number of unique vertices before healing.
    pub vertex_count: usize,
    /// Number of vertices unified by healing.
    pub healed_count: usize,
    /// Whether non-manifold issues were detected.
    pub non_manifold_detected: bool,
}

/// Perturbation cascade report from `try_boolean_with_perturbation`.
#[derive(Debug, Clone, Default)]
pub struct CascadeReport {
    /// Total number of attempts made.
    pub attempts: usize,
    /// Names of strategies tried.
    pub strategies_tried: Vec<String>,
    /// Name of the strategy that succeeded (if any).
    pub final_strategy: Option<String>,
    /// Whether the cascade exhausted all attempts.
    pub exhausted: bool,
}

/// Classification method summary: how many faces were classified by each method.
#[derive(Debug, Clone, Default)]
pub struct ClassificationSummary {
    /// Faces classified via coplanar overlay.
    pub faces_by_overlay: usize,
    /// Faces classified via coplanar fragment test.
    pub faces_by_coplanar: usize,
    /// Faces classified via ray-cast.
    pub faces_by_raycast: usize,
    /// Faces classified via edge-neighbor propagation.
    pub faces_by_edge_neighbor: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_default_values() {
        let diag = BooleanDiagnostics::default();
        assert_eq!(diag.classification.shell0_and, 0);
        assert_eq!(diag.classification.shell0_or, 0);
        assert_eq!(diag.classification.shell1_and, 0);
        assert_eq!(diag.classification.shell1_or, 0);
        assert_eq!(diag.recovery.recovery_level, 0);
        assert_eq!(diag.edge_neighbor.faces_resolved, 0);
        assert!(diag.pre_heal.is_none());
        assert!(diag.cascade.is_none());
        assert!(diag.classification_summary.is_none());
        assert!(diag.warnings.is_empty());
    }

    #[test]
    fn test_summary_line_format() {
        let mut diag = BooleanDiagnostics::default();
        diag.classification.shell0_and = 3;
        diag.classification.shell0_or = 5;
        diag.classification.shell1_and = 2;
        diag.classification.shell1_or = 4;
        let line = diag.summary_line();
        assert!(line.starts_with("[bool-diag] "));
        assert!(line.contains("faces=14"));
        assert!(line.contains("and=3+2"));
        assert!(line.contains("or=5+4"));
        // No recovery or edge-neighbor info when not used
        assert!(!line.contains("recovery="));
        assert!(!line.contains("edge_nbr="));
    }

    #[test]
    fn test_summary_line_with_recovery() {
        let mut diag = BooleanDiagnostics::default();
        diag.classification.shell0_and = 1;
        diag.classification.shell1_or = 1;
        diag.recovery.recovery_level = 3;
        diag.edge_neighbor.faces_resolved = 2;
        diag.cascade = Some(CascadeReport {
            attempts: 5,
            strategies_tried: vec!["direct".into(), "scale-expand".into()],
            final_strategy: Some("scale-expand".into()),
            exhausted: false,
        });
        let line = diag.summary_line();
        assert!(line.contains("recovery=L3"));
        assert!(line.contains("edge_nbr=2"));
        assert!(line.contains("cascade=5"));
    }

    #[test]
    fn test_pre_heal_report_default() {
        let report = PreHealReport::default();
        assert_eq!(report.vertex_count, 0);
        assert_eq!(report.healed_count, 0);
        assert!(!report.non_manifold_detected);
    }

    #[test]
    fn test_cascade_report_default() {
        let report = CascadeReport::default();
        assert_eq!(report.attempts, 0);
        assert!(report.strategies_tried.is_empty());
        assert!(report.final_strategy.is_none());
        assert!(!report.exhausted);
    }
}
