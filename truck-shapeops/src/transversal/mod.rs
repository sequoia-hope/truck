pub(crate) mod coplanar;
mod coplanar_splitting;
mod divide_face;
mod faces_classification;
mod integrate;
mod intersection_curve;
mod loops_store;
mod polyline_construction;
pub(crate) mod robust_classify;
pub use integrate::{
    and, and_result, and_result_with_tol, and_with_tol, difference, difference_result,
    difference_result_with_tol, difference_with_tol, or, or_result, or_result_with_tol,
    or_with_tol, BooleanStageError, BooleanTolerance, ShapeOpsCurve, ShapeOpsSurface,
};
