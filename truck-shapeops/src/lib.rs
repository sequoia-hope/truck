//! Crate for operation shapes. Provides boolean operations to Solid, and shape healing for importing shapes from other CAD systems.

#![cfg_attr(not(debug_assertions), deny(warnings))]
#![deny(clippy::all, rust_2018_idioms)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

mod healing;
pub use healing::{RobustSplitClosedEdgesAndFaces, SplitClosedEdgesAndFaces};
mod transversal;
pub use transversal::{
    and, and_result, and_result_with_tol, and_result_with_tol_diag, and_with_tol,
    diagnose_open_edges, difference, difference_result, difference_result_with_tol,
    difference_result_with_tol_diag, difference_with_tol, find_non_simple_wires,
    heal_shell_vertices, or, or_result, or_result_with_tol, or_result_with_tol_diag, or_with_tol,
    validate_euler_characteristic, BooleanDiagnostics, BooleanStageError, BooleanTolerance,
    OpenEdgeInfo, ShapeOpsCurve, ShapeOpsSurface,
};
/// Diagnostic sub-types for boolean operations.
pub mod diagnostics {
    pub use crate::transversal::diagnostics::*;
}
mod alternative;
mod fillet;
