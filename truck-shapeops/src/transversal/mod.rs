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
    and, and_result, difference, difference_result, or, or_result, BooleanStageError,
    ShapeOpsCurve, ShapeOpsSurface,
};
