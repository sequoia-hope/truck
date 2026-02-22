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
    and, and_result, and_result_with_tol, and_with_tol, difference, difference_result,
    difference_result_with_tol, difference_with_tol, heal_shell_vertices, or, or_result,
    or_result_with_tol, or_with_tol, BooleanStageError, BooleanTolerance, ShapeOpsCurve,
    ShapeOpsSurface,
};
mod alternative;
mod fillet;
