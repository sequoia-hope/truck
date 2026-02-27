//! Analytical intersection curves for common surface pairs.
//!
//! When both surfaces in an intersection pair are analytically recognizable
//! (e.g., plane + cylinder), this module computes exact intersection curves
//! (circle or ellipse) and samples them as high-quality polylines, avoiding
//! the numerical drift of mesh-based intersection extraction.

use std::f64::consts::PI;
use std::ops::Bound;
use truck_base::cgmath64::*;
use truck_geometry::prelude::*;
use truck_meshalgo::prelude::PolylineCurve;

/// Parameters describing a detected plane surface.
#[derive(Debug, Clone)]
struct PlaneParams {
    origin: Point3,
    normal: Vector3,
}

/// Parameters describing a detected cylinder surface.
#[derive(Debug, Clone)]
struct CylinderParams {
    center: Point3,
    axis: Vector3,
    radius: f64,
}

/// Parameters describing a detected cone surface.
///
/// A cone is defined by its apex, axis direction, and half-angle.
/// Points on the cone satisfy: angle(P - apex, axis) = half_angle.
#[derive(Debug, Clone)]
struct ConeParams {
    apex: Point3,
    axis: Vector3,
    half_angle: f64,
}

/// Parameters describing a detected sphere surface.
///
/// A sphere is defined by its center and radius. Points on the sphere
/// satisfy: |P - center| = radius.
#[derive(Debug, Clone)]
struct SphereParams {
    center: Point3,
    radius: f64,
}

/// Parameters describing a detected torus surface.
///
/// A torus is defined by a center of revolution, a revolution axis,
/// a major radius (distance from axis to the generatrix circle center),
/// and a minor radius (radius of the generatrix circle). Points on the
/// torus satisfy: (sqrt((P - center)_perp² ) - R)² + (P - center)_along² = r²
/// where _perp and _along are components perpendicular and along the axis.
#[derive(Debug, Clone)]
struct TorusParams {
    center: Point3,    // center of revolution
    axis: Vector3,     // revolution axis (unit)
    major_radius: f64, // distance from axis to generatrix circle center (R)
    minor_radius: f64, // radius of generatrix circle (r)
}

/// Parameters describing a plane-cylinder intersection ellipse/circle.
///
/// The curve is parameterized as:
///   X(θ) = center + cos(θ) * axis_u + sin(θ) * axis_v
/// For perpendicular cuts, |axis_u| = |axis_v| = radius (circle).
/// For oblique cuts, the magnitudes differ (ellipse).
#[derive(Debug, Clone)]
pub(crate) struct EllipseParams {
    center: Point3,
    axis_u: Vector3,
    axis_v: Vector3,
}

/// Extract the midpoint of a parameter range.
fn param_mid(range: &(Bound<f64>, Bound<f64>), default_half: f64) -> f64 {
    let lo = match range.0 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => -default_half,
    };
    let hi = match range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => default_half,
    };
    (lo + hi) * 0.5
}

/// Extract the start of a parameter range.
fn param_start(range: &(Bound<f64>, Bound<f64>), default: f64) -> f64 {
    match range.0 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => default,
    }
}

/// Fit a circle through three 3D points (circumscribed circle center).
///
/// Returns the center of the circumscribed circle, or None if the points
/// are collinear.
fn circumcenter_3d(p1: Point3, p2: Point3, p3: Point3) -> Option<Point3> {
    let a = p2 - p1;
    let b = p3 - p1;
    let cross = a.cross(b);
    let cross_sq = cross.dot(cross);
    if cross_sq < 1e-20 {
        return None;
    }
    let a_sq = a.dot(a);
    let b_sq = b.dot(b);
    let t1 = cross.cross(a) * b_sq;
    let t2 = b.cross(cross) * a_sq;
    let center_offset = (t1 + t2) / (2.0 * cross_sq);
    Some(p1 + center_offset)
}

/// Try to detect a plane from a generic parametric surface.
///
/// A plane has zero second derivatives and no periodicity.
fn detect_plane<S: ParametricSurface3D>(surface: &S) -> Option<PlaneParams> {
    if surface.u_period().is_some() || surface.v_period().is_some() {
        return None;
    }

    let (u_range, v_range) = surface.parameter_range();
    let u = param_mid(&u_range, 1.0);
    let v = param_mid(&v_range, 1.0);

    let uu = surface.uuder(u, v);
    let uv = surface.uvder(u, v);
    let vv = surface.vvder(u, v);

    let tol = 1e-10;
    if uu.magnitude2() > tol || uv.magnitude2() > tol || vv.magnitude2() > tol {
        return None;
    }

    let origin = surface.subs(u, v);
    let normal = surface.normal(u, v);
    if normal.magnitude2() < tol {
        return None;
    }

    Some(PlaneParams {
        origin,
        normal: normal.normalize(),
    })
}

/// Try to detect a cylinder from a generic parametric surface.
///
/// A cylinder has exactly one periodic direction (period ≈ 2π) and zero
/// curvature in the non-periodic (axial) direction.
fn detect_cylinder<S: ParametricSurface3D>(surface: &S) -> Option<CylinderParams> {
    let u_per = surface.u_period();
    let v_per = surface.v_period();

    // Exactly one periodic direction with period ≈ 2π
    let (is_v_periodic, period) = match (u_per, v_per) {
        (Some(p), None) => (false, p),
        (None, Some(p)) => (true, p),
        _ => return None,
    };

    if (period - 2.0 * PI).abs() > 0.1 {
        return None;
    }

    let (u_range, v_range) = surface.parameter_range();
    let u = param_mid(&u_range, 1.0);
    let v = param_mid(&v_range, 1.0);

    if is_v_periodic {
        // v is angular, u is axial (standard RevolutedCurve<Line> layout)
        let uu = surface.uuder(u, v);
        if uu.magnitude2() > 1e-8 {
            return None;
        }

        let axis = surface.uder(u, v);
        if axis.magnitude2() < 1e-20 {
            return None;
        }
        let axis = axis.normalize();

        // Sample 3 points at different v values to fit circle
        let v_start = param_start(&v_range, 0.0);
        let p1 = surface.subs(u, v_start);
        let p2 = surface.subs(u, v_start + period / 3.0);
        let p3 = surface.subs(u, v_start + 2.0 * period / 3.0);

        let center = circumcenter_3d(p1, p2, p3)?;
        let radius = (p1 - center).magnitude();
        if radius < 1e-10 {
            return None;
        }

        // Verify with a 4th point
        let p4 = surface.subs(u, v_start + period / 4.0);
        if ((p4 - center).magnitude() - radius).abs() > radius * 0.01 {
            return None;
        }

        Some(CylinderParams {
            center,
            axis,
            radius,
        })
    } else {
        // u is angular, v is axial
        let vv = surface.vvder(u, v);
        if vv.magnitude2() > 1e-8 {
            return None;
        }

        let axis = surface.vder(u, v);
        if axis.magnitude2() < 1e-20 {
            return None;
        }
        let axis = axis.normalize();

        let u_start = param_start(&u_range, 0.0);
        let p1 = surface.subs(u_start, v);
        let p2 = surface.subs(u_start + period / 3.0, v);
        let p3 = surface.subs(u_start + 2.0 * period / 3.0, v);

        let center = circumcenter_3d(p1, p2, p3)?;
        let radius = (p1 - center).magnitude();
        if radius < 1e-10 {
            return None;
        }

        let p4 = surface.subs(u_start + period / 4.0, v);
        if ((p4 - center).magnitude() - radius).abs() > radius * 0.01 {
            return None;
        }

        Some(CylinderParams {
            center,
            axis,
            radius,
        })
    }
}

/// Try to detect a cone from a generic parametric surface.
///
/// A cone is a surface of revolution (one periodic direction, period ≈ 2π)
/// with a straight generatrix (zero second derivative in the axial direction)
/// whose radius varies linearly along the axis. The radius at different axial
/// positions must differ (otherwise it's a cylinder).
fn detect_cone<S: ParametricSurface3D>(surface: &S) -> Option<ConeParams> {
    let u_per = surface.u_period();
    let v_per = surface.v_period();

    // Exactly one periodic direction with period ≈ 2π
    let (is_v_periodic, period) = match (u_per, v_per) {
        (Some(p), None) => (false, p),
        (None, Some(p)) => (true, p),
        _ => return None,
    };

    if (period - 2.0 * PI).abs() > 0.1 {
        return None;
    }

    let (u_range, v_range) = surface.parameter_range();

    // Get axial parameter range and angular start
    let (axial_start, axial_end, angular_start, u_is_angular) = if is_v_periodic {
        let u_s = param_start(&u_range, 0.0);
        let u_e = match u_range.1 {
            Bound::Included(v) | Bound::Excluded(v) => v,
            Bound::Unbounded => 1.0,
        };
        let v_s = param_start(&v_range, 0.0);
        (u_s, u_e, v_s, false)
    } else {
        let v_s = param_start(&v_range, 0.0);
        let v_e = match v_range.1 {
            Bound::Included(v) | Bound::Excluded(v) => v,
            Bound::Unbounded => 1.0,
        };
        let u_s = param_start(&u_range, 0.0);
        (v_s, v_e, u_s, true)
    };

    let axial_mid = (axial_start + axial_end) * 0.5;

    // Check axial direction is straight (line generatrix)
    if is_v_periodic {
        let uu = surface.uuder(axial_mid, angular_start);
        if uu.magnitude2() > 1e-8 {
            return None;
        }
    } else {
        let vv = surface.vvder(angular_start, axial_mid);
        if vv.magnitude2() > 1e-8 {
            return None;
        }
    }

    // Sample circles at 3 axial positions (start, mid, end).
    // Some may fail if the axial endpoint is at the apex (radius=0).
    let sample_positions = [axial_start, axial_mid, axial_end];
    let mut circles: Vec<(f64, Point3, f64)> = Vec::new(); // (axial_param, center, radius)

    for &axial in &sample_positions {
        if let Some((center, radius)) =
            circle_radius_at(surface, axial, angular_start, period, u_is_angular)
        {
            circles.push((axial, center, radius));
        }
    }

    // Need at least 2 valid circles to determine cone parameters
    if circles.len() < 2 {
        return None;
    }

    // All radii equal → cylinder, not cone
    let r_min = circles.iter().map(|c| c.2).fold(f64::MAX, f64::min);
    let r_max = circles.iter().map(|c| c.2).fold(0.0_f64, f64::max);
    if (r_max - r_min) < r_max * 0.01 {
        return None;
    }

    // Compute axis from direction between circle centers
    let (t0, c0, r0) = circles[0];
    let (t1, c1, r1) = circles[1];
    let axis_vec = c1 - c0;
    if axis_vec.magnitude2() < 1e-20 {
        return None;
    }
    let axis = axis_vec.normalize();

    // Compute apex by linear extrapolation of radius to zero
    // r(t) = r0 + (r1 - r0) / (t1 - t0) * (t - t0)
    // r(t_apex) = 0 → t_apex = t0 - r0 * (t1 - t0) / (r1 - r0)
    let dt = t1 - t0;
    let dr = r1 - r0;
    if dt.abs() < 1e-15 || dr.abs() < 1e-15 {
        return None;
    }
    let t_apex = t0 - r0 * dt / dr;
    // Apex is on the axis line: c0 + (t_apex - t0) / dt * (c1 - c0)
    let apex = c0 + ((t_apex - t0) / dt) * axis_vec;

    // Half-angle from first circle with non-zero radius
    let dist_to_apex = (c0 - apex).magnitude();
    if dist_to_apex < 1e-12 {
        return None;
    }
    let half_angle = (r0 / dist_to_apex).atan();

    // Verify with a third circle if available
    if circles.len() >= 3 {
        let (_t2, c2, r2) = circles[2];
        let dist2 = (c2 - apex).magnitude();
        let expected_r2 = dist2 * half_angle.tan();
        if (r2 - expected_r2).abs() > r2.max(expected_r2) * 0.05 {
            return None;
        }
    }

    Some(ConeParams {
        apex,
        axis,
        half_angle,
    })
}

/// Sample 3 points around the periodic direction at a given axial position,
/// fit a circle, and return (center, radius).
///
/// When `u_is_angular` is true, the angular parameter is u and the axial
/// position is given as v; otherwise angular is v and axial is u.
fn circle_radius_at<S: ParametricSurface3D>(
    surface: &S,
    axial_pos: f64,
    angular_start: f64,
    period: f64,
    u_is_angular: bool,
) -> Option<(Point3, f64)> {
    let (p1, p2, p3, p4) = if u_is_angular {
        (
            surface.subs(angular_start, axial_pos),
            surface.subs(angular_start + period / 3.0, axial_pos),
            surface.subs(angular_start + 2.0 * period / 3.0, axial_pos),
            surface.subs(angular_start + period / 4.0, axial_pos),
        )
    } else {
        (
            surface.subs(axial_pos, angular_start),
            surface.subs(axial_pos, angular_start + period / 3.0),
            surface.subs(axial_pos, angular_start + 2.0 * period / 3.0),
            surface.subs(axial_pos, angular_start + period / 4.0),
        )
    };

    let center = circumcenter_3d(p1, p2, p3)?;
    let radius = (p1 - center).magnitude();
    if radius < 1e-10 {
        return None;
    }

    // Verify with 4th point
    if ((p4 - center).magnitude() - radius).abs() > radius * 0.02 {
        return None;
    }

    Some((center, radius))
}

/// Try to detect a sphere from a generic parametric surface.
///
/// A sphere is a surface of revolution (one periodic direction, period ≈ 2π)
/// with a curved generatrix (non-zero axial second derivative, unlike
/// cylinder/cone which have straight generatrices). All surface points
/// are equidistant from a single center point.
fn detect_sphere<S: ParametricSurface3D>(surface: &S) -> Option<SphereParams> {
    let u_per = surface.u_period();
    let v_per = surface.v_period();

    // Need exactly one periodic direction with period ≈ 2π.
    // A sphere may also present as both-periodic in some representations;
    // in that case pick the one closer to 2π.
    let (is_v_periodic, period) = match (u_per, v_per) {
        (Some(p), None) => (false, p),
        (None, Some(p)) => (true, p),
        (Some(p1), Some(p2)) => {
            if (p1 - 2.0 * PI).abs() < 0.1 {
                (false, p1)
            } else if (p2 - 2.0 * PI).abs() < 0.1 {
                (true, p2)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    if (period - 2.0 * PI).abs() > 0.1 {
        return None;
    }

    let (u_range, v_range) = surface.parameter_range();
    let u_is_angular = !is_v_periodic;

    let angular_start = if u_is_angular {
        param_start(&u_range, 0.0)
    } else {
        param_start(&v_range, 0.0)
    };

    let axial_range = if u_is_angular { &v_range } else { &u_range };
    let ax_start = param_start(axial_range, 0.0);
    let ax_end = match axial_range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => PI,
    };

    // Must NOT have a straight generatrix (that would be cylinder or cone).
    // Check that axial second derivative is non-zero (curved generatrix).
    let ax_mid = (ax_start + ax_end) * 0.5;
    let ang_mid = angular_start + period * 0.25;
    let axial_curv = if u_is_angular {
        surface.vvder(ang_mid, ax_mid)
    } else {
        surface.uuder(ax_mid, ang_mid)
    };
    if axial_curv.magnitude2() < 1e-8 {
        return None; // straight generatrix → cylinder or cone, not sphere
    }

    // Sample circles at several axial positions
    let n_axial = 5;
    let mut circles: Vec<(f64, Point3, f64)> = Vec::new();
    for i in 0..n_axial {
        let t = ax_start + (ax_end - ax_start) * (i as f64 + 0.5) / n_axial as f64;
        if let Some((center, radius)) =
            circle_radius_at(surface, t, angular_start, period, u_is_angular)
        {
            if radius > 1e-10 {
                circles.push((t, center, radius));
            }
        }
    }

    if circles.len() < 3 {
        return None;
    }

    // For a sphere, all circle centers lie on the revolution axis, and there
    // exists a unique point (sphere center) equidistant from all points on
    // all circles. Use two circles to find the sphere center:
    //   Let s = distance from c0 to sphere center along axis.
    //   s² + r0² = R² and (d01 - s)² + r1² = R²
    //   → s = (d01² + r0² - r1²) / (2 * d01)
    let (_t0, c0, r0) = circles[0];
    let (_t1, c1, r1) = circles[1];
    let axis_vec = c1 - c0;
    if axis_vec.magnitude2() < 1e-20 {
        return None;
    }
    let axis = axis_vec.normalize();
    let d01 = axis_vec.magnitude();

    let s = (d01 * d01 + r0 * r0 - r1 * r1) / (2.0 * d01);
    let sphere_center = c0 + s * axis;
    let r_sq = s * s + r0 * r0;
    if r_sq < 1e-20 {
        return None;
    }
    let sphere_radius = r_sq.sqrt();

    // Verify with remaining circles
    for &(_t, ci, ri) in &circles[2..] {
        let di = (ci - sphere_center).dot(axis).abs();
        let expected_r_sq = (sphere_radius * sphere_radius - di * di).max(0.0);
        let expected_r = expected_r_sq.sqrt();
        if (ri - expected_r).abs() > sphere_radius * 0.02 {
            return None;
        }
    }

    // Also verify with actual surface points at different angular positions
    for i in 0..4 {
        let theta = period * i as f64 / 4.0 + angular_start;
        let axial = ax_start + (ax_end - ax_start) * 0.3;
        let pt = if u_is_angular {
            surface.subs(theta, axial)
        } else {
            surface.subs(axial, theta)
        };
        let dist = (pt - sphere_center).magnitude();
        if (dist - sphere_radius).abs() > sphere_radius * 0.02 {
            return None;
        }
    }

    Some(SphereParams {
        center: sphere_center,
        radius: sphere_radius,
    })
}

/// Try to detect a torus from a generic parametric surface.
///
/// Handles two cases:
/// 1. Both u and v periodic with period ≈ 2π (truck `Torus` analytic type)
/// 2. One direction periodic (revolution), the other has a closed generatrix
///    (e.g., `RevolutedCurve<NurbsCurve>` where the NURBS approximates a circle
///    but doesn't self-report as periodic)
///
/// Detection algorithm:
/// 1. Check for at least one periodic direction with period ≈ 2π
/// 2. For the other direction, check if it's closed (front ≈ back)
/// 3. Sample circles at several positions — radii must be bounded away from 0
/// 4. Fit major_radius (R) and minor_radius (r)
/// 5. Verify sampled points lie on the fitted torus
fn detect_torus<S: ParametricSurface3D>(surface: &S) -> Option<TorusParams> {
    let u_per = surface.u_period();
    let v_per = surface.v_period();

    let (u_range, v_range) = surface.parameter_range();
    let u_start = param_start(&u_range, 0.0);
    let u_end = match u_range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => 2.0 * PI,
    };
    let v_start = param_start(&v_range, 0.0);
    let v_end = match v_range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => 2.0 * PI,
    };

    let u_span = u_end - u_start;
    let v_span = v_end - v_start;

    // A torus needs at least one periodic direction (revolution angle).
    // The other direction is the generatrix (may be periodic for analytical torus,
    // or non-periodic NURBS for RevolutedCurve<NurbsCurve>).
    //
    // For RevolutedCurve<NurbsCurve> (circle profile revolved around axis):
    //   u = generatrix (NURBS circle): range [0,1], NOT periodic, NOT closed
    //   v = revolution angle: range [0,2π), period = 2π
    //
    // For analytical Torus type:
    //   u = one angle: range [0,2π), period = 2π
    //   v = other angle: range [0,2π), period = 2π

    // Check for periodic direction with period ≈ 2π
    let u_periodic =
        matches!(u_per, Some(p) if (p - 2.0 * PI).abs() < 0.1) || ((u_span - 2.0 * PI).abs() < 0.1);
    let v_periodic =
        matches!(v_per, Some(p) if (p - 2.0 * PI).abs() < 0.1) || ((v_span - 2.0 * PI).abs() < 0.1);

    // Need at least one periodic direction to be a surface of revolution
    if !u_periodic && !v_periodic {
        return None;
    }

    // For each periodic direction, sample revolution circles and try to fit torus.
    // The periodic direction sweeps revolution circles; the other is the generatrix.
    //
    // Strategy: sample circles at N positions along the generatrix direction.
    // For each position, fit a circle using 3 points in the revolution direction.
    // Then check: are circle centers collinear? Do radii vary between R-r and R+r?

    // For each periodic direction, sample revolution circles and extract torus params.
    //
    // For a torus with the periodic direction as revolution angle:
    // - Fix generatrix parameter, sweep revolution → circle
    // - All circles share the SAME center (the torus center on the axis)
    // - Radii vary between R-r and R+r
    // - Circle normals are all parallel (= axis direction)
    //
    // We extract axis from circle normals, torus center from circle centers,
    // and R, r from the radius variation.

    let try_revolution_direction = |rev_start: f64,
                                    rev_span: f64,
                                    gen_start: f64,
                                    gen_span: f64,
                                    rev_is_u: bool|
     -> Option<TorusParams> {
        let sample = |rev: f64, gx: f64| -> Point3 {
            if rev_is_u {
                surface.subs(rev, gx)
            } else {
                surface.subs(gx, rev)
            }
        };

        // Step 1: Get the revolution axis from a circle at the generatrix midpoint.
        let g_mid = gen_start + gen_span * 0.5;
        let p1 = sample(rev_start, g_mid);
        let p2 = sample(rev_start + rev_span / 3.0, g_mid);
        let p3 = sample(rev_start + 2.0 * rev_span / 3.0, g_mid);
        let mid_center = circumcenter_3d(p1, p2, p3)?;
        let mid_radius = (p1 - mid_center).magnitude();
        if mid_radius < 1e-10 {
            return None;
        }
        let p4 = sample(rev_start + rev_span / 4.0, g_mid);
        if ((p4 - mid_center).magnitude() - mid_radius).abs() > mid_radius * 0.05 {
            return None; // not a circle
        }
        let axis_raw = (p2 - p1).cross(p3 - p1);
        if axis_raw.magnitude2() < 1e-20 {
            return None;
        }
        let axis = axis_raw.normalize();

        // Step 2: Verify that circles at a few other positions share the same axis.
        for &frac in &[0.2, 0.35, 0.65, 0.8] {
            let g = gen_start + gen_span * frac;
            let q1 = sample(rev_start, g);
            let q2 = sample(rev_start + rev_span / 3.0, g);
            let q3 = sample(rev_start + 2.0 * rev_span / 3.0, g);
            let n = (q2 - q1).cross(q3 - q1);
            if n.magnitude2() > 1e-20 {
                let dot = n.normalize().dot(axis).abs();
                if dot < 0.95 {
                    return None; // different axis → not revolution circles
                }
            }
        }

        // Step 3: Find R and r using geometric exploration.
        //
        // Problem: NURBS generatrix parameterization may be highly non-uniform,
        // making parameter-uniform sampling cluster near one position.
        //
        // Solution: Use GEOMETRIC binary search to find the extremes.
        // At the midpoint we have mid_center (on axis) and mid_radius.
        // The perpendicular distance from mid_center to the axis gives us the
        // axis-offset at this generatrix position.
        //
        // For a torus: revolution radius varies from R-r to R+r.
        // We search for the generatrix positions with min and max radius.

        // First, sample radii at a fine grid of generatrix positions
        let n_samples = 200;
        let mut best_min_radius = f64::MAX;
        let mut best_max_radius = 0.0_f64;
        let mut best_min_g = gen_start;
        let mut best_max_g = gen_start;
        let mut any_center: Option<Point3> = None;

        for j in 0..n_samples {
            let g = gen_start + gen_span * j as f64 / n_samples as f64;
            let q1 = sample(rev_start, g);
            let q2 = sample(rev_start + rev_span / 3.0, g);
            let q3 = sample(rev_start + 2.0 * rev_span / 3.0, g);
            if let Some(center) = circumcenter_3d(q1, q2, q3) {
                let r = (q1 - center).magnitude();
                if r > 1e-10 {
                    if r < best_min_radius {
                        best_min_radius = r;
                        best_min_g = g;
                    }
                    if r > best_max_radius {
                        best_max_radius = r;
                        best_max_g = g;
                    }
                    if any_center.is_none() {
                        any_center = Some(center);
                    }
                }
            }
        }

        if best_max_radius < 1e-10 || any_center.is_none() {
            return None;
        }

        // Refine min and max with binary search around the found positions
        let refine_extremum = |init_g: f64, seeking_min: bool| -> f64 {
            let delta = gen_span / n_samples as f64;
            let mut best_g = init_g;
            let mut best_r = if seeking_min { f64::MAX } else { 0.0_f64 };

            // Search in neighborhood of init_g
            for step in &[delta, delta / 4.0, delta / 16.0, delta / 64.0] {
                let g_lo = (best_g - step * 4.0).max(gen_start);
                let g_hi = (best_g + step * 4.0).min(gen_start + gen_span);
                for k in 0..20 {
                    let g = g_lo + (g_hi - g_lo) * k as f64 / 19.0;
                    let q1 = sample(rev_start, g);
                    let q2 = sample(rev_start + rev_span / 3.0, g);
                    let q3 = sample(rev_start + 2.0 * rev_span / 3.0, g);
                    if let Some(center) = circumcenter_3d(q1, q2, q3) {
                        let r = (q1 - center).magnitude();
                        let is_better = if seeking_min {
                            r < best_r && r > 1e-10
                        } else {
                            r > best_r
                        };
                        if is_better {
                            best_r = r;
                            best_g = g;
                        }
                    }
                }
            }
            best_r
        };

        let r_min_refined = refine_extremum(best_min_g, true);
        let r_max_refined = refine_extremum(best_max_g, false);

        // Use the ANY center to get torus center (all centers lie on axis)
        // The torus center is best estimated from the mid_center (step 1)
        let center_centroid = mid_center;

        let major_radius = (r_max_refined + r_min_refined) / 2.0;
        let minor_radius = (r_max_refined - r_min_refined) / 2.0;

        if minor_radius < major_radius * 0.005 {
            return None; // cylinder (no variation)
        }
        if minor_radius < 1e-10 {
            return None;
        }
        if major_radius < minor_radius * 0.1 {
            return None; // spindle
        }

        let torus = TorusParams {
            center: center_centroid,
            axis,
            major_radius,
            minor_radius,
        };

        if verify_torus_fit_ranges(surface, &torus, u_start, u_span, v_start, v_span) {
            return Some(torus);
        }

        None
    };

    // Try v as revolution direction (most common for RevolutedCurve: v=revolution angle)
    if v_periodic {
        if let Some(torus) = try_revolution_direction(v_start, v_span, u_start, u_span, false) {
            return Some(torus);
        }
    }

    // Try u as revolution direction
    if u_periodic {
        if let Some(torus) = try_revolution_direction(u_start, u_span, v_start, v_span, true) {
            return Some(torus);
        }
    }

    None
}

/// Verify that sampled surface points lie on the fitted torus.
fn verify_torus_fit_ranges<S: ParametricSurface3D>(
    surface: &S,
    torus: &TorusParams,
    u_start: f64,
    u_span: f64,
    v_start: f64,
    v_span: f64,
) -> bool {
    let n_samples = 8;
    for i in 0..n_samples {
        let u = u_start + u_span * i as f64 / n_samples as f64;
        for j in 0..n_samples {
            let v = v_start + v_span * j as f64 / n_samples as f64;
            let pt = surface.subs(u, v);
            let d = torus_distance(&pt, torus);
            if d > torus.minor_radius * 0.1 {
                return false;
            }
        }
    }
    true
}

/// Detect a surface of revolution's axis from a generic parametric surface.
///
/// A surface of revolution has one periodic direction (period ≈ 2π) where
/// sweeping traces circles. This function extracts just the revolution axis
/// (a point + direction), which is sufficient for per-point refinement
/// when paired with a plane.
///
/// Unlike `detect_torus`, this works on partial arcs (e.g., individual faces
/// of a multi-face torus solid) because it only needs one revolution circle
/// to determine the axis — it does NOT need to determine R or r.
fn detect_revolution_axis<S: ParametricSurface3D>(surface: &S) -> Option<RevolutionAxisParams> {
    let u_per = surface.u_period();
    let v_per = surface.v_period();

    let (u_range, v_range) = surface.parameter_range();
    let u_start = param_start(&u_range, 0.0);
    let u_end = match u_range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => 2.0 * PI,
    };
    let v_start = param_start(&v_range, 0.0);
    let v_end = match v_range.1 {
        Bound::Included(v) | Bound::Excluded(v) => v,
        Bound::Unbounded => 2.0 * PI,
    };

    let u_span = u_end - u_start;
    let v_span = v_end - v_start;

    // Need exactly one periodic direction with period ≈ 2π
    let u_periodic =
        matches!(u_per, Some(p) if (p - 2.0 * PI).abs() < 0.1) || ((u_span - 2.0 * PI).abs() < 0.1);
    let v_periodic =
        matches!(v_per, Some(p) if (p - 2.0 * PI).abs() < 0.1) || ((v_span - 2.0 * PI).abs() < 0.1);

    if !u_periodic && !v_periodic {
        return None;
    }

    // Must NOT be a plane (no periodicity expected, but guard)
    // Must have curved generatrix to distinguish from cylinder/cone
    // (Those are already handled by their own detectors, but we verify
    // that this IS a revolution surface by checking circle fit.)

    let try_axis = |rev_start: f64,
                    rev_span: f64,
                    gen_start: f64,
                    gen_span: f64,
                    rev_is_u: bool|
     -> Option<RevolutionAxisParams> {
        let sample = |rev: f64, gx: f64| -> Point3 {
            if rev_is_u {
                surface.subs(rev, gx)
            } else {
                surface.subs(gx, rev)
            }
        };

        // Sample a revolution circle at the generatrix midpoint
        let g_mid = gen_start + gen_span * 0.5;
        let p1 = sample(rev_start, g_mid);
        let p2 = sample(rev_start + rev_span / 3.0, g_mid);
        let p3 = sample(rev_start + 2.0 * rev_span / 3.0, g_mid);

        let center = circumcenter_3d(p1, p2, p3)?;
        let radius = (p1 - center).magnitude();
        if radius < 1e-10 {
            return None;
        }

        // Verify it's actually a circle (4th point check)
        let p4 = sample(rev_start + rev_span / 4.0, g_mid);
        if ((p4 - center).magnitude() - radius).abs() > radius * 0.05 {
            return None;
        }

        // Extract axis direction from the circle normal
        let axis_raw = (p2 - p1).cross(p3 - p1);
        if axis_raw.magnitude2() < 1e-20 {
            return None;
        }
        let axis = axis_raw.normalize();

        // Verify axis at another generatrix position (if the generatrix span allows)
        if gen_span > 1e-6 {
            let g_other = gen_start + gen_span * 0.25;
            let q1 = sample(rev_start, g_other);
            let q2 = sample(rev_start + rev_span / 3.0, g_other);
            let q3 = sample(rev_start + 2.0 * rev_span / 3.0, g_other);
            let n2 = (q2 - q1).cross(q3 - q1);
            if n2.magnitude2() > 1e-20 {
                let dot = n2.normalize().dot(axis).abs();
                if dot < 0.95 {
                    return None; // different axis → not a surface of revolution
                }
            }
        }

        Some(RevolutionAxisParams {
            axis_point: center,
            axis_dir: axis,
        })
    };

    // Try v as revolution direction (most common for RevolutedCurve)
    if v_periodic {
        if let Some(params) = try_axis(v_start, v_span, u_start, u_span, false) {
            return Some(params);
        }
    }

    // Try u as revolution direction
    if u_periodic {
        if let Some(params) = try_axis(u_start, u_span, v_start, v_span, true) {
            return Some(params);
        }
    }

    None
}

/// Refine a single point onto the revolution-surface ∩ plane intersection.
///
/// For a point P near a revolution-surface/plane intersection, the refined
/// point preserves P's azimuthal angle and perpendicular distance from the
/// revolution axis, and finds the axial position where this lands on the
/// cutting plane.
///
/// Result: P' = axis_point + t*axis_dir + radius*radial_dir
/// where t is chosen so that n · (P' - plane_origin) = 0.
fn refine_point_on_revsurf_plane(pt: &Point3, rsp: &RevSurfPlaneParams) -> Point3 {
    let a = rsp.axis_dir;
    let n = rsp.plane_normal;

    // Build orthonormal frame perpendicular to axis
    let e_r = if a.x.abs() < 0.9 {
        a.cross(Vector3::unit_x()).normalize()
    } else {
        a.cross(Vector3::unit_y()).normalize()
    };
    let e_t = a.cross(e_r).normalize();

    // Decompose pt into cylindrical coordinates relative to axis
    let d = *pt - rsp.axis_point;
    let r_coord = d.dot(e_r);
    let t_coord = d.dot(e_t);
    let radius = (r_coord * r_coord + t_coord * t_coord).sqrt();
    if radius < 1e-12 {
        return *pt; // on axis — can't determine azimuthal angle
    }

    let theta = t_coord.atan2(r_coord);
    let radial = theta.cos() * e_r + theta.sin() * e_t;

    // Solve for axial position t:
    //   n · (axis_point + t*a + radius*radial - plane_origin) = 0
    //   t * (n·a) = n · (plane_origin - axis_point) - radius * (n·radial)
    let n_dot_a = n.dot(a);
    if n_dot_a.abs() < 1e-12 {
        // Plane parallel to axis — fall back to simple plane projection
        let dist = n.dot(*pt - rsp.plane_origin);
        return *pt - dist * n;
    }

    let numerator = n.dot(rsp.plane_origin - rsp.axis_point) - radius * n.dot(radial);
    let t = numerator / n_dot_a;

    rsp.axis_point + t * a + radius * radial
}

/// Compute the distance from a point to the torus surface.
///
/// For a torus with center C, axis A, major radius R, minor radius r:
/// distance = | sqrt(perp² ) - R | - r  where perp = |P-C - ((P-C)·A)·A|
/// This simplifies to: sqrt( (sqrt(perp²) - R)² + along² ) - r
fn torus_distance(pt: &Point3, torus: &TorusParams) -> f64 {
    let v = *pt - torus.center;
    let along = v.dot(torus.axis);
    let perp_vec = v - along * torus.axis;
    let perp = perp_vec.magnitude();
    let dist_to_tube_center = ((perp - torus.major_radius).powi(2) + along * along).sqrt();
    (dist_to_tube_center - torus.minor_radius).abs()
}

/// Compute the intersection of a plane and torus.
///
/// For the common CAD case where the plane contains the torus axis or is
/// perpendicular to it, the intersection is one or two circles.
///
/// General torus-plane intersection is a quartic curve. We handle:
/// 1. Plane perpendicular to torus axis → 1 or 2 circles
/// 2. Plane containing the torus axis → 2 circles (inner + outer)
/// 3. General oblique plane → sample points on the exact intersection
///
/// Returns a vector of EllipseParams (circles are a special case of ellipses).
fn compute_torus_plane_intersection(
    plane: &PlaneParams,
    torus: &TorusParams,
) -> Option<Vec<EllipseParams>> {
    let n = plane.normal;
    let a = torus.axis;
    let dot_na = n.dot(a).abs();

    // Case 1: Plane perpendicular to torus axis (dot_na ≈ 1)
    // The intersection is 0, 1, or 2 circles in the plane.
    if dot_na > 1.0 - 1e-6 {
        return compute_torus_perpendicular_plane(plane, torus);
    }

    // Case 2: Plane contains the torus axis (dot_na ≈ 0)
    // Intersection is 2 circles (inner ring at R-r and outer ring at R+r)
    if dot_na < 1e-6 {
        return compute_torus_axial_plane(plane, torus);
    }

    // Case 3: General oblique plane — use sampling approach
    // Sample points on the torus-plane intersection by sweeping the revolution angle
    // and finding where the generatrix circle intersects the plane
    compute_torus_oblique_plane(plane, torus)
}

/// Torus-plane intersection when plane is perpendicular to the torus axis.
///
/// The plane cuts the torus into 0, 1, or 2 circles:
/// - If |d| > r: no intersection
/// - If |d| = r: 1 circle at radius R (tangent)
/// - If |d| < r: 2 circles at radii R ± sqrt(r² - d²)
fn compute_torus_perpendicular_plane(
    plane: &PlaneParams,
    torus: &TorusParams,
) -> Option<Vec<EllipseParams>> {
    // Signed distance from torus center to plane along the axis
    let d = plane.normal.dot(torus.center - plane.origin);

    let r = torus.minor_radius;
    let big_r = torus.major_radius;

    // No intersection if plane is beyond the torus extent
    if d.abs() >= r - 1e-10 {
        return None;
    }

    // The cross-section radii: R ± sqrt(r² - d²)
    let h = (r * r - d * d).sqrt();
    let r_outer = big_r + h;
    let r_inner = big_r - h;

    // Build orthonormal basis in the plane
    let n = plane.normal;
    let e1 = if n.x.abs() < 0.9 {
        n.cross(Vector3::unit_x()).normalize()
    } else {
        n.cross(Vector3::unit_y()).normalize()
    };
    let e2 = n.cross(e1).normalize();

    // Circle center: projection of torus center onto the plane
    let circle_center = torus.center - d * plane.normal;

    let mut ellipses = Vec::new();

    // Outer circle (always exists if d < r)
    if r_outer > 1e-12 {
        ellipses.push(EllipseParams {
            center: circle_center,
            axis_u: r_outer * e1,
            axis_v: r_outer * e2,
        });
    }

    // Inner circle (exists if R > h, i.e., the inner radius is positive)
    if r_inner > 1e-12 {
        ellipses.push(EllipseParams {
            center: circle_center,
            axis_u: r_inner * e1,
            axis_v: r_inner * e2,
        });
    }

    if ellipses.is_empty() {
        None
    } else {
        Some(ellipses)
    }
}

/// Torus-plane intersection when plane contains the torus axis.
///
/// The plane slices through the torus tube, producing 2 circles:
/// one at the "outer" edge (radius R+r projected) and one at the "inner" edge (radius R-r projected).
/// Actually, when a plane contains the torus axis, it cuts through each tube cross-section
/// producing 2 circles of radius equal to the minor radius.
fn compute_torus_axial_plane(
    plane: &PlaneParams,
    torus: &TorusParams,
) -> Option<Vec<EllipseParams>> {
    let n = plane.normal;
    let a = torus.axis;
    let r = torus.minor_radius;
    let big_r = torus.major_radius;

    // The plane contains the axis. Find the direction perpendicular to
    // both the normal and the axis — this is the "radial" direction in the plane.
    let radial = n.cross(a);
    let radial_mag = radial.magnitude();
    if radial_mag < 1e-12 {
        return None; // degenerate
    }
    let radial = radial / radial_mag;

    // Two tube cross-section centers: at distance R from center along radial
    let center1 = torus.center + big_r * radial;
    let center2 = torus.center - big_r * radial;

    // Each cross-section is a circle of radius r in the plane defined by (a, radial)
    // But we need the circle to lie in the cutting plane.
    // The cutting plane contains both 'a' and 'radial', so the circles lie in this plane.
    let e1 = a;
    let e2 = radial;

    let mut ellipses = Vec::new();

    // Check plane distance to each center (should be ~0 since plane contains the axis)
    let d1 = n.dot(center1 - plane.origin).abs();
    let d2 = n.dot(center2 - plane.origin).abs();

    if d1 < r * 0.1 {
        ellipses.push(EllipseParams {
            center: center1,
            axis_u: r * e1,
            axis_v: r * e2,
        });
    }

    if d2 < r * 0.1 {
        ellipses.push(EllipseParams {
            center: center2,
            axis_u: r * e1,
            axis_v: r * e2,
        });
    }

    if ellipses.is_empty() {
        None
    } else {
        Some(ellipses)
    }
}

/// Torus-plane intersection for a general oblique plane.
///
/// The intersection is a quartic curve. We approximate it by sweeping the revolution
/// angle and finding where each generatrix circle intersects the plane. This produces
/// a high-quality polyline that the mesh-based IC pipeline can use for refinement.
///
/// For the CAD boolean use case, we produce circle approximations by fitting ellipses
/// to the sampled intersection points.
fn compute_torus_oblique_plane(
    plane: &PlaneParams,
    torus: &TorusParams,
) -> Option<Vec<EllipseParams>> {
    let n = plane.normal;
    let a = torus.axis;
    let big_r = torus.major_radius;
    let r = torus.minor_radius;

    // Build orthonormal frame for the torus: a, e_r, e_t
    let e_r = if a.x.abs() < 0.9 {
        a.cross(Vector3::unit_x()).normalize()
    } else {
        a.cross(Vector3::unit_y()).normalize()
    };
    let e_t = a.cross(e_r).normalize();

    // For each revolution angle theta, the generatrix circle center is at:
    //   C(theta) = torus.center + R * (cos(theta) * e_r + sin(theta) * e_t)
    // The generatrix circle has radius r and lies in the plane spanned by
    // the axis 'a' and the radial direction at theta.
    //
    // A generatrix circle intersects the cutting plane in 0 or 2 points.
    // We sweep theta and collect all intersection points.

    let n_sweep = 256;
    let mut intersection_pts: Vec<Point3> = Vec::new();

    for i in 0..n_sweep {
        let theta = 2.0 * PI * i as f64 / n_sweep as f64;
        let ct = theta.cos();
        let st = theta.sin();

        let radial = ct * e_r + st * e_t;
        let tube_center = torus.center + big_r * radial;

        // Generatrix circle: P(phi) = tube_center + r * (cos(phi) * a + sin(phi) * radial)
        // Intersect with plane: n . (P(phi) - plane.origin) = 0
        // n . (tube_center - plane.origin) + r * (cos(phi) * n.a + sin(phi) * n.radial) = 0
        let d = n.dot(tube_center - plane.origin);
        let coeff_cos = r * n.dot(a);
        let coeff_sin = r * n.dot(radial);

        // d + coeff_cos * cos(phi) + coeff_sin * sin(phi) = 0
        // A * cos(phi - delta) = -d  where A = sqrt(cc² + cs²), delta = atan2(cs, cc)
        let amplitude = (coeff_cos * coeff_cos + coeff_sin * coeff_sin).sqrt();
        if amplitude < 1e-15 {
            continue; // generatrix circle parallel to plane
        }

        let ratio = -d / amplitude;
        if ratio.abs() > 1.0 - 1e-12 {
            if ratio.abs() > 1.0 + 1e-10 {
                continue; // no intersection
            }
            // Tangent — single point
            let delta = coeff_sin.atan2(coeff_cos);
            let phi = delta; // cos(phi - delta) = ±1
            let pt = tube_center + r * (phi.cos() * a + phi.sin() * radial);
            intersection_pts.push(pt);
        } else {
            // Two intersection points
            let delta = coeff_sin.atan2(coeff_cos);
            let alpha = ratio.clamp(-1.0, 1.0).acos();
            for &phi_base in &[delta + alpha, delta - alpha] {
                let pt = tube_center + r * (phi_base.cos() * a + phi_base.sin() * radial);
                intersection_pts.push(pt);
            }
        }
    }

    if intersection_pts.len() < 6 {
        return None;
    }

    // Separate intersection points into distinct curves.
    // A general torus-plane intersection has 1 or 2 closed curves.
    // Sort points into curves by connectivity (nearest-neighbor chain).
    let curves = separate_into_curves(&intersection_pts, torus.minor_radius * 0.5);

    let mut ellipses = Vec::new();
    for curve_pts in &curves {
        if curve_pts.len() < 4 {
            continue;
        }
        // Fit an ellipse to each curve
        if let Some(ell) = fit_ellipse_to_points(curve_pts, &n) {
            ellipses.push(ell);
        }
    }

    if ellipses.is_empty() {
        None
    } else {
        Some(ellipses)
    }
}

/// Separate a set of intersection points into distinct closed curves
/// by nearest-neighbor chaining with a distance threshold.
fn separate_into_curves(points: &[Point3], max_gap: f64) -> Vec<Vec<Point3>> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut used = vec![false; points.len()];
    let mut curves = Vec::new();

    loop {
        // Find the first unused point
        let start = used.iter().position(|&u| !u);
        let start = match start {
            Some(s) => s,
            None => break,
        };

        let mut curve = vec![points[start]];
        used[start] = true;

        // Greedily chain nearest unused points
        loop {
            let last = curve.last().unwrap();
            let mut best_idx = None;
            let mut best_dist = max_gap * max_gap;

            for (i, pt) in points.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let d2 = (*pt - *last).magnitude2();
                if d2 < best_dist {
                    best_dist = d2;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    curve.push(points[idx]);
                    used[idx] = true;
                }
                None => break,
            }
        }

        if curve.len() >= 4 {
            curves.push(curve);
        } else {
            // Mark points as used but don't create a curve
        }
    }

    curves
}

/// Fit an ellipse (or circle) to a set of 3D points that lie approximately in a plane.
///
/// Uses the plane normal to project points to 2D, computes the center and axes
/// from the extremes along principal directions in the plane.
fn fit_ellipse_to_points(points: &[Point3], plane_normal: &Vector3) -> Option<EllipseParams> {
    if points.len() < 4 {
        return None;
    }

    // Compute centroid
    let n = points.len() as f64;
    let centroid = Point3::new(
        points.iter().map(|p| p.x).sum::<f64>() / n,
        points.iter().map(|p| p.y).sum::<f64>() / n,
        points.iter().map(|p| p.z).sum::<f64>() / n,
    );

    // Build 2D frame in the plane
    let pn = plane_normal.normalize();
    let e1 = if pn.x.abs() < 0.9 {
        pn.cross(Vector3::unit_x()).normalize()
    } else {
        pn.cross(Vector3::unit_y()).normalize()
    };
    let e2 = pn.cross(e1).normalize();

    // Project to 2D and find principal axes via covariance
    let mut cov_xx = 0.0;
    let mut cov_xy = 0.0;
    let mut cov_yy = 0.0;

    for pt in points {
        let d = *pt - centroid;
        let x = d.dot(e1);
        let y = d.dot(e2);
        cov_xx += x * x;
        cov_xy += x * y;
        cov_yy += y * y;
    }

    cov_xx /= n;
    cov_xy /= n;
    cov_yy /= n;

    // Eigenvalues of 2D covariance matrix
    let trace = cov_xx + cov_yy;
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    let discriminant = (trace * trace - 4.0 * det).max(0.0);
    let lambda1 = (trace + discriminant.sqrt()) / 2.0;
    let lambda2 = (trace - discriminant.sqrt()) / 2.0;

    if lambda1 < 1e-20 || lambda2 < 1e-20 {
        return None;
    }

    // Eigenvectors
    let (d1, d2) = if cov_xy.abs() > 1e-15 {
        let ev1_x = lambda1 - cov_yy;
        let ev1_y = cov_xy;
        let ev1_mag = (ev1_x * ev1_x + ev1_y * ev1_y).sqrt();
        let d1_2d = (ev1_x / ev1_mag, ev1_y / ev1_mag);
        let d2_2d = (-d1_2d.1, d1_2d.0);
        (
            (d1_2d.0 * e1 + d1_2d.1 * e2).normalize(),
            (d2_2d.0 * e1 + d2_2d.1 * e2).normalize(),
        )
    } else {
        (e1, e2)
    };

    // Semi-axes from covariance eigenvalues.
    // For uniformly sampled points on an ellipse, the covariance eigenvalue
    // equals semi_axis² / 2.
    let semi_a = (2.0 * lambda1).sqrt();
    let semi_b = (2.0 * lambda2).sqrt();

    if semi_a < 1e-12 || semi_b < 1e-12 {
        return None;
    }

    Some(EllipseParams {
        center: centroid,
        axis_u: semi_a * d1,
        axis_v: semi_b * d2,
    })
}

/// Compute the intersection of a plane and sphere.
///
/// The intersection is always a circle (or empty/tangent point).
/// Returns None if the plane doesn't intersect the sphere interior
/// (tangent or no intersection).
fn compute_sphere_plane_intersection(
    plane: &PlaneParams,
    sphere: &SphereParams,
) -> Option<EllipseParams> {
    let n = plane.normal;
    // Signed distance from sphere center to plane
    let d = n.dot(sphere.center - plane.origin);

    // No intersection if distance >= radius (tangent or miss)
    if d.abs() >= sphere.radius - 1e-10 {
        return None;
    }

    // Circle center: projection of sphere center onto plane
    let circle_center = sphere.center - d * n;
    // Circle radius: sqrt(R² - d²)
    let circle_radius = (sphere.radius * sphere.radius - d * d).sqrt();

    if circle_radius < 1e-12 {
        return None;
    }

    // Build orthonormal basis in the plane
    let e1 = if n.x.abs() < 0.9 {
        n.cross(Vector3::unit_x()).normalize()
    } else {
        n.cross(Vector3::unit_y()).normalize()
    };
    let e2 = n.cross(e1).normalize();

    Some(EllipseParams {
        center: circle_center,
        axis_u: circle_radius * e1,
        axis_v: circle_radius * e2,
    })
}

/// Compute the intersection of a plane and cylinder.
///
/// Returns the ellipse parameters, or None if the plane is parallel to
/// the cylinder axis (intersection would be lines, not a curve).
fn compute_plane_cylinder_intersection(
    plane: &PlaneParams,
    cyl: &CylinderParams,
) -> Option<EllipseParams> {
    let n = plane.normal;
    let a = cyl.axis;
    let dot_na = n.dot(a);

    // Plane parallel to cylinder axis → line intersections (not supported)
    if dot_na.abs() < 1e-6 {
        return None;
    }

    // Ellipse center: intersection of cylinder axis with plane
    let t = n.dot(plane.origin - cyl.center) / dot_na;
    let center = cyl.center + t * a;

    // Build orthonormal frame E1, E2 perpendicular to cylinder axis
    let e1 = if a.x.abs() < 0.9 {
        a.cross(Vector3::unit_x()).normalize()
    } else {
        a.cross(Vector3::unit_y()).normalize()
    };
    let e2 = a.cross(e1).normalize();

    // Project cylinder cross-section basis vectors onto the plane.
    // The intersection curve: X(θ) = center + cos(θ)*U + sin(θ)*V
    // where U = R*(E1 - (N·E1)/(N·A)*A), V = R*(E2 - (N·E2)/(N·A)*A).
    let coeff_e1 = n.dot(e1) / dot_na;
    let coeff_e2 = n.dot(e2) / dot_na;

    let axis_u = cyl.radius * (e1 - coeff_e1 * a);
    let axis_v = cyl.radius * (e2 - coeff_e2 * a);

    Some(EllipseParams {
        center,
        axis_u,
        axis_v,
    })
}

/// Compute the intersection of a plane and cone.
///
/// Returns the ellipse/circle parameters, or None for degenerate cases:
/// - Plane through apex → two lines (degenerate)
/// - Plane parallel to generator → parabola (not an ellipse)
/// - Plane intersects both nappes → hyperbola (not an ellipse)
fn compute_plane_cone_intersection(
    plane: &PlaneParams,
    cone: &ConeParams,
) -> Option<EllipseParams> {
    let n = plane.normal;
    let a = cone.axis;
    let dot_na = n.dot(a);

    // Distance from apex to the plane (signed along normal)
    let apex_dist = n.dot(plane.origin - cone.apex);

    // Plane passes through apex → degenerate (two lines)
    if apex_dist.abs() < 1e-10 {
        return None;
    }

    // Classify the conic section by comparing the angle between
    // the plane normal and the cone axis to the half-angle.
    //
    // Let α = half-angle, β = angle between plane normal and cone axis.
    // - β = 0 (perpendicular to axis) → circle
    // - 0 < β < π/2 - α → ellipse
    // - β = π/2 - α → parabola (plane parallel to generator)
    // - β > π/2 - α → hyperbola
    let cos_beta = dot_na.abs();
    let sin_alpha = cone.half_angle.sin();

    // Plane parallel to generator → parabola (not supported)
    if (cos_beta - sin_alpha).abs() < 1e-6 {
        return None;
    }

    // Hyperbola: cos(β) < sin(α) means the plane cuts both nappes
    if cos_beta < sin_alpha {
        return None;
    }

    // Build orthonormal frame E1, E2 perpendicular to cone axis
    let e1 = if a.x.abs() < 0.9 {
        a.cross(Vector3::unit_x()).normalize()
    } else {
        a.cross(Vector3::unit_y()).normalize()
    };
    let e2 = a.cross(e1).normalize();

    let tan_alpha = cone.half_angle.tan();
    let ne1 = n.dot(e1);
    let ne2 = n.dot(e2);

    // Exact cone-plane intersection point at angle θ:
    //   P(θ) = apex + t(θ) * (a + tan(α)*(cos(θ)*e1 + sin(θ)*e2))
    //   where t(θ) = apex_dist / (dot_na + tan(α)*(cos(θ)*ne1 + sin(θ)*ne2))
    let sample_point = |theta: f64| -> Option<Point3> {
        let denom = dot_na + tan_alpha * (theta.cos() * ne1 + theta.sin() * ne2);
        if denom.abs() < 1e-15 {
            return None;
        }
        let t = apex_dist / denom;
        if t < 0.0 {
            return None; // opposite nappe
        }
        let dir = a + tan_alpha * (theta.cos() * e1 + theta.sin() * e2);
        Some(cone.apex + t * dir)
    };

    // The ellipse axes align with two orthogonal directions in the cutting
    // plane: d1 = projection of cone axis onto the plane (or arbitrary if
    // perpendicular), and d2 = n × d1.
    let a_proj = a - dot_na * n;
    let (d1, d2) = if a_proj.magnitude2() > 1e-12 {
        let d1 = a_proj.normalize();
        let d2 = n.cross(d1).normalize();
        (d1, d2)
    } else {
        // Perpendicular cut → circle, use any basis in the plane
        let d1 = if n.x.abs() < 0.9 {
            n.cross(Vector3::unit_x()).normalize()
        } else {
            n.cross(Vector3::unit_y()).normalize()
        };
        let d2 = n.cross(d1).normalize();
        (d1, d2)
    };

    // Sample many exact points and find the extremes along d1 and d2.
    let n_samples = 256;
    let mut d1_min = f64::MAX;
    let mut d1_max = f64::MIN;
    let mut d2_min = f64::MAX;
    let mut d2_max = f64::MIN;
    let mut ref_point = None;

    for i in 0..n_samples {
        let theta = 2.0 * PI * i as f64 / n_samples as f64;
        let pt = sample_point(theta)?;
        if ref_point.is_none() {
            ref_point = Some(pt);
        }
        let rp = ref_point.unwrap();
        let v = pt - rp;
        let coord1 = v.dot(d1);
        let coord2 = v.dot(d2);
        d1_min = d1_min.min(coord1);
        d1_max = d1_max.max(coord1);
        d2_min = d2_min.min(coord2);
        d2_max = d2_max.max(coord2);
    }

    let rp = ref_point?;

    // Ellipse center and semi-axes from the extremes
    let center_d1 = (d1_min + d1_max) * 0.5;
    let center_d2 = (d2_min + d2_max) * 0.5;
    let semi_a = (d1_max - d1_min) * 0.5;
    let semi_b = (d2_max - d2_min) * 0.5;

    if semi_a < 1e-12 || semi_b < 1e-12 {
        return None;
    }

    let center = rp + center_d1 * d1 + center_d2 * d2;
    let axis_u = semi_a * d1;
    let axis_v = semi_b * d2;

    Some(EllipseParams {
        center,
        axis_u,
        axis_v,
    })
}

/// Sample an ellipse as a closed polyline with `n_segments` segments.
#[cfg(test)]
fn sample_ellipse(ellipse: &EllipseParams, n_segments: usize) -> PolylineCurve<Point3> {
    let mut points: Vec<Point3> = (0..n_segments)
        .map(|i| {
            let theta = 2.0 * PI * i as f64 / n_segments as f64;
            ellipse.center + theta.cos() * ellipse.axis_u + theta.sin() * ellipse.axis_v
        })
        .collect();
    // Close the loop exactly (avoid floating-point drift at θ=2π)
    points.push(points[0]);
    PolylineCurve(points)
}

/// Parameters for a detected revolution surface axis.
///
/// A surface of revolution has one periodic direction (period ≈ 2π) sweeping
/// circles around an axis. This struct captures just the axis, which is
/// sufficient for per-point refinement when paired with a plane.
#[derive(Debug, Clone)]
struct RevolutionAxisParams {
    axis_point: Point3, // a point on the revolution axis
    axis_dir: Vector3,  // unit direction along the axis
}

/// Parameters for revolution-surface + plane per-point refinement.
///
/// Instead of computing global intersection curves (which requires full torus
/// parameters that are hard to extract from partial-arc faces), this stores
/// just the revolution axis and plane. Refinement works per-point by:
/// 1. Finding the revolution circle through the point
/// 2. Intersecting that circle with the plane
/// 3. Picking the closest intersection point
#[derive(Debug, Clone)]
struct RevSurfPlaneParams {
    axis_point: Point3,
    axis_dir: Vector3,
    plane_origin: Point3,
    plane_normal: Vector3,
}

/// Opaque handle to one or more analytical intersection curves (ellipses/circles).
///
/// Used to refine mesh-based polylines by projecting their points onto
/// the closest exact intersection curve. Most surface pairs produce a single
/// ellipse; cylinder-cylinder produces two.
///
/// For revolution-surface + plane pairs where full surface parameters can't
/// be determined (e.g., partial torus arcs), `revsurf_plane` provides
/// per-point refinement using circle-plane intersection.
#[derive(Debug, Clone)]
pub struct AnalyticalIC {
    ellipses: Vec<EllipseParams>,
    revsurf_plane: Option<RevSurfPlaneParams>,
}

/// Try to detect an analytical plane-cylinder intersection.
///
/// Returns an `AnalyticalIC` that can be used to refine mesh-based polylines,
/// or None if the surfaces are not a recognizable plane-cylinder pair.
pub fn try_analytical_plane_cylinder_ic<S0, S1>(
    surface0: &S0,
    surface1: &S1,
    _tol: f64,
) -> Option<AnalyticalIC>
where
    S0: ParametricSurface3D,
    S1: ParametricSurface3D,
{
    let (plane, cyl) =
        if let (Some(p), Some(c)) = (detect_plane(surface0), detect_cylinder(surface1)) {
            (p, c)
        } else if let (Some(p), Some(c)) = (detect_plane(surface1), detect_cylinder(surface0)) {
            (p, c)
        } else {
            return None;
        };

    let ellipse = compute_plane_cylinder_intersection(&plane, &cyl)?;
    Some(AnalyticalIC {
        ellipses: vec![ellipse],
        revsurf_plane: None,
    })
}

/// Try to detect an analytical plane-cone intersection.
///
/// Returns an `AnalyticalIC` that can be used to refine mesh-based polylines,
/// or None if the surfaces are not a recognizable plane-cone pair or if the
/// intersection is degenerate (through apex, parabola, or hyperbola).
pub fn try_analytical_plane_cone_ic<S0, S1>(
    surface0: &S0,
    surface1: &S1,
    _tol: f64,
) -> Option<AnalyticalIC>
where
    S0: ParametricSurface3D,
    S1: ParametricSurface3D,
{
    let (plane, cone) = if let (Some(p), Some(c)) = (detect_plane(surface0), detect_cone(surface1))
    {
        (p, c)
    } else if let (Some(p), Some(c)) = (detect_plane(surface1), detect_cone(surface0)) {
        (p, c)
    } else {
        return None;
    };

    let ellipse = compute_plane_cone_intersection(&plane, &cone)?;
    Some(AnalyticalIC {
        ellipses: vec![ellipse],
        revsurf_plane: None,
    })
}

/// Try to detect an analytical plane-sphere intersection.
///
/// Returns an `AnalyticalIC` that can be used to refine mesh-based polylines,
/// or None if the surfaces are not a recognizable plane-sphere pair or if the
/// intersection is degenerate (tangent or no intersection).
pub fn try_analytical_plane_sphere_ic<S0, S1>(
    surface0: &S0,
    surface1: &S1,
    _tol: f64,
) -> Option<AnalyticalIC>
where
    S0: ParametricSurface3D,
    S1: ParametricSurface3D,
{
    let (plane, sphere) =
        if let (Some(p), Some(s)) = (detect_plane(surface0), detect_sphere(surface1)) {
            (p, s)
        } else if let (Some(p), Some(s)) = (detect_plane(surface1), detect_sphere(surface0)) {
            (p, s)
        } else {
            return None;
        };

    let ellipse = compute_sphere_plane_intersection(&plane, &sphere)?;
    Some(AnalyticalIC {
        ellipses: vec![ellipse],
        revsurf_plane: None,
    })
}

/// Compute the intersection of two equal-radius cylinders with intersecting axes.
///
/// Returns two ellipses, or None for degenerate/unsupported cases:
/// - Unequal radii (degree-4 algebraic curves)
/// - Parallel or coaxial axes
/// - Skew (non-intersecting) axes
/// - Near-parallel axes (angle < 60 deg)
fn compute_cylinder_cylinder_intersection(
    cyl0: &CylinderParams,
    cyl1: &CylinderParams,
) -> Option<Vec<EllipseParams>> {
    let r0 = cyl0.radius;
    let r1 = cyl1.radius;
    let r_max = r0.max(r1);

    // Guard: unequal radii (>1% relative)
    if (r0 - r1).abs() / r_max > 0.01 {
        eprintln!(
            "[analytical] cylinder-cylinder: unequal radii r0={:.4} r1={:.4}, skipping",
            r0, r1
        );
        return None;
    }

    let r = (r0 + r1) * 0.5; // average radius

    let a0 = cyl0.axis;
    let a1 = cyl1.axis;

    let cos_angle = a0.dot(a1).abs();

    // Guard: parallel axes (cos > 1 - 1e-6)
    if cos_angle > 1.0 - 1e-6 {
        eprintln!("[analytical] cylinder-cylinder: parallel axes, skipping");
        return None;
    }

    // Guard: near-parallel (angle < 60 deg means |cos| > 0.5)
    if cos_angle > 0.5 {
        eprintln!(
            "[analytical] cylinder-cylinder: angle {:.1}° < 60°, skipping",
            cos_angle.acos().to_degrees()
        );
        return None;
    }

    // Compute closest distance between axes (skew test).
    // Two lines: P0 + t*a0 and P1 + s*a1.
    // Closest distance = |((P1 - P0) . (a0 x a1))| / |a0 x a1|
    let cross = a0.cross(a1);
    let cross_mag = cross.magnitude();
    if cross_mag < 1e-12 {
        return None; // degenerate cross product
    }
    let diff = cyl1.center - cyl0.center;
    let dist = diff.dot(cross).abs() / cross_mag;

    // Guard: non-intersecting axes (dist > 5% of R)
    if dist > 0.05 * r {
        eprintln!(
            "[analytical] cylinder-cylinder: skew axes dist={:.6} > 5% of R={:.4}, skipping",
            dist, r
        );
        return None;
    }

    // Compute axis intersection point.
    // Solve: P0 + t*a0 = P1 + s*a1 (approximately, in the closest-point sense)
    // Using: t = ((P1-P0) . a0 - ((P1-P0) . a1)(a0 . a1)) / (1 - (a0.a1)²)
    let d01 = a0.dot(a1);
    let denom = 1.0 - d01 * d01;
    if denom.abs() < 1e-12 {
        return None;
    }
    let d_diff_a0 = diff.dot(a0);
    let d_diff_a1 = diff.dot(a1);
    let t = (d_diff_a0 - d_diff_a1 * d01) / denom;
    let origin = cyl0.center + t * a0;

    // Ensure axes form an acute angle for consistent frame construction.
    let a1_oriented = if a0.dot(a1) < 0.0 { -a1 } else { a1 };
    let cos_alpha = a0.dot(a1_oriented).clamp(-1.0, 1.0);
    let alpha = cos_alpha.acos(); // angle between axes (0, π/2]

    // Build orthonormal frame:
    //   e1 = a0
    //   e2 = (a1_oriented - (a1_oriented . a0) * a0).normalize()
    //   e3 = e1 x e2
    let e1 = a0;
    let e2_unnorm = a1_oriented - cos_alpha * a0;
    let e2_mag = e2_unnorm.magnitude();
    if e2_mag < 1e-12 {
        return None;
    }
    let e2 = e2_unnorm / e2_mag;
    let e3 = e1.cross(e2);

    let half_alpha = alpha * 0.5;
    let cot_half = half_alpha.cos() / half_alpha.sin();
    let tan_half = half_alpha.tan();

    // Curve 1: X(t) = origin + R*cos(t) * (cot(α/2)*e1 + e2) + R*sin(t) * e3
    let axis_u1 = r * (cot_half * e1 + e2);
    let axis_v1 = r * e3;

    // Curve 2: X(t) = origin + R*cos(t) * (-tan(α/2)*e1 + e2) + R*sin(t) * e3
    let axis_u2 = r * (-tan_half * e1 + e2);
    let axis_v2 = r * e3;

    Some(vec![
        EllipseParams {
            center: origin,
            axis_u: axis_u1,
            axis_v: axis_v1,
        },
        EllipseParams {
            center: origin,
            axis_u: axis_u2,
            axis_v: axis_v2,
        },
    ])
}

/// Try to detect an analytical cylinder-cylinder intersection.
///
/// Returns an `AnalyticalIC` with two ellipses that can be used to refine
/// mesh-based polylines, or None if the surfaces are not a recognizable
/// equal-radius cylinder pair with intersecting axes.
pub fn try_analytical_cylinder_cylinder_ic<S0, S1>(
    surface0: &S0,
    surface1: &S1,
    _tol: f64,
) -> Option<AnalyticalIC>
where
    S0: ParametricSurface3D,
    S1: ParametricSurface3D,
{
    let cyl0 = detect_cylinder(surface0)?;
    let cyl1 = detect_cylinder(surface1)?;
    let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1)?;
    Some(AnalyticalIC {
        ellipses,
        revsurf_plane: None,
    })
}

/// Try to detect an analytical torus/revolution-surface + plane intersection.
///
/// Two paths:
/// 1. Full torus detection → ellipse-based refinement (works for analytical `Torus` type)
/// 2. Revolution axis detection → per-point circle-plane refinement (works for
///    partial-arc `RevolutedCurve<NurbsCurve>` faces from the revolve pipeline)
///
/// Returns an `AnalyticalIC` that can be used to refine mesh-based polylines,
/// or None if neither surface pair is a recognizable torus/revsurf + plane.
pub fn try_analytical_torus_plane_ic<S0, S1>(
    surface0: &S0,
    surface1: &S1,
    _tol: f64,
) -> Option<AnalyticalIC>
where
    S0: ParametricSurface3D,
    S1: ParametricSurface3D,
{
    // Path 1: Full torus detection → ellipse-based refinement
    let p0 = detect_plane(surface0);
    let p1 = detect_plane(surface1);
    let t0 = detect_torus(surface0);
    let t1 = detect_torus(surface1);

    if let Some((plane, torus)) = match (&p0, &t0, &p1, &t1) {
        (Some(p), _, _, Some(t)) => Some((p.clone(), t.clone())),
        (_, Some(t), Some(p), _) => Some((p.clone(), t.clone())),
        _ => None,
    } {
        if let Some(ellipses) = compute_torus_plane_intersection(&plane, &torus) {
            return Some(AnalyticalIC {
                ellipses,
                revsurf_plane: None,
            });
        }
    }

    // Path 2: Revolution axis detection → per-point refinement
    // This handles partial-arc faces where detect_torus fails (can't determine R, r)
    // but we can still detect the revolution axis and do per-point circle-plane refinement.
    let r0 = detect_revolution_axis(surface0);
    let r1 = detect_revolution_axis(surface1);

    let (plane, rev) = match (&p0, &r1, &p1, &r0) {
        (Some(p), Some(r), _, _) => (p, r),
        (_, _, Some(p), Some(r)) => (p, r),
        _ => return None,
    };

    Some(AnalyticalIC {
        ellipses: Vec::new(),
        revsurf_plane: Some(RevSurfPlaneParams {
            axis_point: rev.axis_point,
            axis_dir: rev.axis_dir,
            plane_origin: plane.origin,
            plane_normal: plane.normal,
        }),
    })
}

/// Select the ellipse closest to a polyline from a set of candidates.
///
/// For each candidate ellipse, computes the sum of squared projection
/// distances across all polyline points. Returns the ellipse with the
/// smallest total distance. O(n*k) where k is typically 1-2.
fn pick_closest_ellipse<'a>(
    polyline: &PolylineCurve<Point3>,
    ellipses: &'a [EllipseParams],
) -> &'a EllipseParams {
    debug_assert!(!ellipses.is_empty());
    if ellipses.len() == 1 {
        return &ellipses[0];
    }
    let mut best_idx = 0;
    let mut best_cost = f64::MAX;
    for (i, ell) in ellipses.iter().enumerate() {
        let cost: f64 = polyline
            .0
            .iter()
            .map(|pt| {
                let proj = project_to_ellipse(pt, ell);
                (*pt - proj).magnitude2()
            })
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best_idx = i;
        }
    }
    &ellipses[best_idx]
}

/// Refine a mesh-based polyline by projecting each point onto the closest
/// analytical intersection curve (ellipse/circle).
///
/// This preserves the mesh-based topology (start/end points, open/closed,
/// number of segments) while improving point accuracy. The refined points lie
/// exactly on the analytical intersection, eliminating BSpline drift.
pub fn refine_polyline(
    mesh_polyline: &PolylineCurve<Point3>,
    analytical: &AnalyticalIC,
) -> PolylineCurve<Point3> {
    // Prefer per-point revolution-surface refinement when available
    if let Some(rsp) = &analytical.revsurf_plane {
        let points: Vec<Point3> = mesh_polyline
            .0
            .iter()
            .map(|pt| refine_point_on_revsurf_plane(pt, rsp))
            .collect();
        return PolylineCurve(points);
    }

    // Fall back to ellipse projection
    if analytical.ellipses.is_empty() {
        return mesh_polyline.clone();
    }
    let ellipse = pick_closest_ellipse(mesh_polyline, &analytical.ellipses);
    let points: Vec<Point3> = mesh_polyline
        .0
        .iter()
        .map(|pt| project_to_ellipse(pt, ellipse))
        .collect();
    PolylineCurve(points)
}

/// Project a 3D point onto the nearest point on an ellipse.
///
/// The ellipse is parameterized as:
///   X(θ) = center + cos(θ) * axis_u + sin(θ) * axis_v
///
/// We find the angle θ that minimizes |pt - X(θ)|.
fn project_to_ellipse(pt: &Point3, ellipse: &EllipseParams) -> Point3 {
    let d = *pt - ellipse.center;

    // Compute the 2D coordinates in the ellipse frame.
    // We need to solve: minimize |d - cos(θ)*U - sin(θ)*V|²
    // Taking derivative and setting to zero:
    //   sin(θ)*(d·U) - cos(θ)*(d·V) + sin(θ)*cos(θ)*(|V|²-|U|²)
    //     + (cos²θ - sin²θ)*(U·V) = 0
    //
    // For a circle (|U|=|V|, U·V=0), this simplifies to:
    //   θ = atan2(d·V, d·U)
    //
    // For a general ellipse, we use Newton's method starting from the
    // circle approximation.

    let du = d.dot(ellipse.axis_u);
    let dv = d.dot(ellipse.axis_v);

    // Initial guess from circle approximation
    let mut theta = dv.atan2(du);

    // Newton refinement for the general ellipse case
    let uu = ellipse.axis_u.dot(ellipse.axis_u);
    let vv = ellipse.axis_v.dot(ellipse.axis_v);
    let uv = ellipse.axis_u.dot(ellipse.axis_v);

    // Only refine if axes are not orthogonal/equal (non-circular ellipse)
    if uv.abs() > 1e-12 || (uu - vv).abs() > 1e-12 * (uu + vv) {
        for _ in 0..5 {
            let ct = theta.cos();
            let st = theta.sin();

            // f(θ) = ∂/∂θ |pt - X(θ)|² / 2
            //       = sin(θ)*(d·U) - cos(θ)*(d·V)
            //         + sin(θ)*cos(θ)*(|V|²-|U|²) + (cos²θ-sin²θ)*(U·V)
            let f = st * du - ct * dv + st * ct * (vv - uu) + (ct * ct - st * st) * uv;

            // f'(θ)
            let fp = ct * du + st * dv + (ct * ct - st * st) * (vv - uu) - 4.0 * st * ct * uv;

            if fp.abs() < 1e-15 {
                break;
            }
            theta -= f / fp;
        }
    }

    ellipse.center + theta.cos() * ellipse.axis_u + theta.sin() * ellipse.axis_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circumcenter_equilateral() {
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(-0.5, 3.0_f64.sqrt() / 2.0, 0.0);
        let p3 = Point3::new(-0.5, -(3.0_f64.sqrt()) / 2.0, 0.0);
        let center = circumcenter_3d(p1, p2, p3).unwrap();
        assert!(
            (center - Point3::origin()).magnitude() < 1e-10,
            "center = {:?}",
            center
        );
    }

    #[test]
    fn test_circumcenter_collinear_returns_none() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        let p3 = Point3::new(2.0, 0.0, 0.0);
        assert!(circumcenter_3d(p1, p2, p3).is_none());
    }

    #[test]
    fn test_plane_cylinder_perpendicular() {
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal: Vector3::new(0.0, 0.0, 1.0),
        };
        let cyl = CylinderParams {
            center: Point3::new(0.0, 0.0, 0.0),
            axis: Vector3::new(0.0, 0.0, 1.0),
            radius: 1.0,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();

        // Center at (0,0,5)
        assert!((ellipse.center - Point3::new(0.0, 0.0, 5.0)).magnitude() < 1e-10);
        // Circle: both axes have magnitude = radius = 1
        assert!((ellipse.axis_u.magnitude() - 1.0).abs() < 1e-10);
        assert!((ellipse.axis_v.magnitude() - 1.0).abs() < 1e-10);
        // Axes perpendicular
        assert!(ellipse.axis_u.dot(ellipse.axis_v).abs() < 1e-10);

        // All sampled points at distance 1 from Z-axis and at z=5
        let poly = sample_ellipse(&ellipse, 64);
        for pt in &poly.0 {
            let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
            assert!((r - 1.0).abs() < 1e-10, "r = {r}");
            assert!((pt.z - 5.0).abs() < 1e-10, "z = {}", pt.z);
        }
    }

    #[test]
    fn test_plane_cylinder_oblique_45deg() {
        let normal = Vector3::new(0.0, 1.0, 1.0).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();
        assert!((ellipse.center - Point3::origin()).magnitude() < 1e-10);

        let poly = sample_ellipse(&ellipse, 64);
        for pt in &poly.0 {
            // On the plane
            let d = normal.dot(*pt - Point3::origin()).abs();
            assert!(d < 1e-10, "dist_to_plane = {d}");
            // On the cylinder
            let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
            assert!((r - 1.0).abs() < 1e-10, "r = {r}");
        }

        // Check ellipse dimensions: minor=1, major=√2 for 45°
        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        let (minor, major) = if semi_a < semi_b {
            (semi_a, semi_b)
        } else {
            (semi_b, semi_a)
        };
        assert!((minor - 1.0).abs() < 1e-10, "minor = {minor}");
        assert!((major - 2.0_f64.sqrt()).abs() < 1e-10, "major = {major}");
    }

    #[test]
    fn test_plane_parallel_to_cylinder_returns_none() {
        let plane = PlaneParams {
            origin: Point3::new(5.0, 0.0, 0.0),
            normal: Vector3::unit_x(),
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };
        assert!(compute_plane_cylinder_intersection(&plane, &cyl).is_none());
    }

    #[test]
    fn test_detect_plane_from_truck_plane() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 0.0, 5.0),
            Point3::new(0.0, 1.0, 5.0),
        );
        let detected = detect_plane(&plane);
        assert!(detected.is_some(), "Failed to detect Plane");
        let params = detected.unwrap();
        assert!(
            (params.normal - Vector3::unit_z()).magnitude() < 1e-10
                || (params.normal + Vector3::unit_z()).magnitude() < 1e-10,
            "normal = {:?}",
            params.normal
        );
    }

    #[test]
    fn test_detect_cylinder_from_revolved_line() {
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        let detected = detect_cylinder(&cylinder);
        assert!(detected.is_some(), "Failed to detect cylinder");
        let params = detected.unwrap();
        assert!(
            (params.axis - Vector3::unit_z()).magnitude() < 0.01
                || (params.axis + Vector3::unit_z()).magnitude() < 0.01,
            "axis = {:?}",
            params.axis
        );
        assert!(
            (params.radius - 2.0).abs() < 0.01,
            "radius = {}",
            params.radius
        );
    }

    #[test]
    fn test_plane_not_detected_as_cylinder() {
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(detect_cylinder(&plane).is_none());
    }

    #[test]
    fn test_cylinder_not_detected_as_plane() {
        let line = Line(Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 5.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(detect_plane(&cylinder).is_none());
    }

    #[test]
    fn test_analytical_fallback_non_plane_cylinder() {
        // BSpline saddle surfaces — neither plane nor cylinder
        let ctrl = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 1.0)],
            vec![Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 0.0)],
        ];
        let surface0 = BSplineSurface::new(
            (KnotVec::bezier_knot(1), KnotVec::bezier_knot(1)),
            ctrl.clone(),
        );
        let surface1 =
            BSplineSurface::new((KnotVec::bezier_knot(1), KnotVec::bezier_knot(1)), ctrl);
        assert!(try_analytical_plane_cylinder_ic(&surface0, &surface1, 0.05).is_none());
    }

    #[test]
    fn test_analytical_plane_cylinder_detection() {
        // Full detection with actual truck surface types
        let plane = Plane::new(
            Point3::new(-5.0, -5.0, 3.0),
            Point3::new(5.0, -5.0, 3.0),
            Point3::new(-5.0, 5.0, 3.0),
        );
        let line = Line(Point3::new(1.5, 0.0, 0.0), Point3::new(1.5, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());

        let result = try_analytical_plane_cylinder_ic(&plane, &cylinder, 0.05);
        assert!(result.is_some(), "Failed to detect plane-cylinder pair");
    }

    #[test]
    fn test_refine_polyline_circle() {
        // Create an analytical IC for a Z-axis cylinder, radius 1.5, plane at z=3
        let plane = Plane::new(
            Point3::new(-5.0, -5.0, 3.0),
            Point3::new(5.0, -5.0, 3.0),
            Point3::new(-5.0, 5.0, 3.0),
        );
        let line = Line(Point3::new(1.5, 0.0, 0.0), Point3::new(1.5, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());

        let analytical = try_analytical_plane_cylinder_ic(&plane, &cylinder, 0.05).unwrap();

        // Noisy mesh polyline (points roughly on the circle but drifted)
        let mesh_poly = PolylineCurve(vec![
            Point3::new(1.48, 0.05, 3.001),
            Point3::new(0.03, 1.52, 2.998),
            Point3::new(-1.51, -0.02, 3.002),
        ]);

        let refined = refine_polyline(&mesh_poly, &analytical);
        assert_eq!(refined.0.len(), 3);

        for pt in &refined.0 {
            // On the cylinder (distance 1.5 from Z-axis)
            let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
            assert!((r - 1.5).abs() < 1e-10, "r = {r}, expected 1.5");
            // On the plane (z=3)
            assert!((pt.z - 3.0).abs() < 1e-10, "z = {}, expected 3.0", pt.z);
        }
    }

    #[test]
    fn test_refine_preserves_point_count() {
        let analytical = AnalyticalIC {
            ellipses: vec![EllipseParams {
                center: Point3::new(0.0, 0.0, 5.0),
                axis_u: Vector3::new(2.0, 0.0, 0.0),
                axis_v: Vector3::new(0.0, 2.0, 0.0),
            }],
            revsurf_plane: None,
        };

        let mesh_poly = PolylineCurve(vec![
            Point3::new(2.1, 0.0, 5.0),
            Point3::new(0.0, 1.9, 5.0),
            Point3::new(-2.05, 0.0, 5.0),
            Point3::new(0.0, -1.95, 5.0),
            Point3::new(2.1, 0.0, 5.0),
        ]);

        let refined = refine_polyline(&mesh_poly, &analytical);
        assert_eq!(refined.0.len(), 5);
    }

    #[test]
    fn test_project_to_ellipse_circle() {
        let ellipse = EllipseParams {
            center: Point3::origin(),
            axis_u: Vector3::new(3.0, 0.0, 0.0),
            axis_v: Vector3::new(0.0, 3.0, 0.0),
        };

        let proj = project_to_ellipse(&Point3::new(4.0, 0.0, 0.0), &ellipse);
        assert!((proj - Point3::new(3.0, 0.0, 0.0)).magnitude() < 1e-10);

        let proj = project_to_ellipse(&Point3::new(0.0, 5.0, 0.0), &ellipse);
        assert!((proj - Point3::new(0.0, 3.0, 0.0)).magnitude() < 1e-10);
    }

    #[test]
    fn test_sample_ellipse_closure() {
        let ellipse = EllipseParams {
            center: Point3::new(1.0, 2.0, 3.0),
            axis_u: Vector3::new(1.0, 0.0, 0.0),
            axis_v: Vector3::new(0.0, 1.0, 0.0),
        };
        let poly = sample_ellipse(&ellipse, 32);
        assert_eq!(poly.0.len(), 33);
        // First and last points should be identical
        assert_eq!(poly.0[0], poly.0[32]);
    }

    // --- Plane-Cylinder edge case tests (Sprint 40) ---

    /// Helper: verify all sampled points lie on both the plane and the cylinder.
    fn assert_points_on_plane_and_cylinder(
        ellipse: &EllipseParams,
        plane_origin: Point3,
        plane_normal: Vector3,
        cyl_axis: Vector3,
        cyl_center: Point3,
        radius: f64,
    ) {
        let poly = sample_ellipse(ellipse, 128);
        for pt in &poly.0 {
            // On the plane
            let d = plane_normal.dot(*pt - plane_origin).abs();
            assert!(d < 1e-8, "point not on plane: dist = {d}");
            // On the cylinder (distance from axis = radius)
            let v = *pt - cyl_center;
            let along = v.dot(cyl_axis) * cyl_axis;
            let perp = v - along;
            let r = perp.magnitude();
            assert!(
                (r - radius).abs() < 1e-8,
                "point not on cylinder: r = {r}, expected {radius}"
            );
        }
    }

    #[test]
    fn test_plane_cylinder_oblique_30deg() {
        // Plane normal at 30° from Z-axis (60° from XY-plane)
        let angle = 30.0_f64.to_radians();
        let normal = Vector3::new(0.0, angle.sin(), angle.cos()).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();

        // Expected semi-axes: minor = R = 2, major = R / cos(30°)
        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        let (minor, major) = if semi_a < semi_b {
            (semi_a, semi_b)
        } else {
            (semi_b, semi_a)
        };
        let expected_major = 2.0 / angle.cos();
        assert!((minor - 2.0).abs() < 1e-8, "minor = {minor}, expected 2.0");
        assert!(
            (major - expected_major).abs() < 1e-8,
            "major = {major}, expected {expected_major}"
        );

        assert_points_on_plane_and_cylinder(
            &ellipse,
            Point3::origin(),
            normal,
            Vector3::unit_z(),
            Point3::origin(),
            2.0,
        );
    }

    #[test]
    fn test_plane_cylinder_oblique_60deg() {
        // Plane normal at 60° from Z-axis
        let angle = 60.0_f64.to_radians();
        let normal = Vector3::new(angle.sin(), 0.0, angle.cos()).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.5,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();

        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        let (minor, major) = if semi_a < semi_b {
            (semi_a, semi_b)
        } else {
            (semi_b, semi_a)
        };
        let expected_major = 1.5 / angle.cos();
        assert!((minor - 1.5).abs() < 1e-8, "minor = {minor}, expected 1.5");
        assert!(
            (major - expected_major).abs() < 1e-8,
            "major = {major}, expected {expected_major}"
        );

        assert_points_on_plane_and_cylinder(
            &ellipse,
            Point3::origin(),
            normal,
            Vector3::unit_z(),
            Point3::origin(),
            1.5,
        );
    }

    #[test]
    fn test_plane_cylinder_oblique_85deg() {
        // Nearly grazing: plane normal at 85° from Z-axis (5° from parallel)
        let angle = 85.0_f64.to_radians();
        let normal = Vector3::new(0.0, angle.sin(), angle.cos()).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();

        // At 85°, major axis = R / cos(85°) ≈ 11.47
        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        let (minor, major) = if semi_a < semi_b {
            (semi_a, semi_b)
        } else {
            (semi_b, semi_a)
        };
        let expected_major = 1.0 / angle.cos();
        assert!((minor - 1.0).abs() < 1e-8, "minor = {minor}, expected 1.0");
        assert!(
            (major - expected_major).abs() < 1e-6,
            "major = {major}, expected {expected_major}"
        );

        assert_points_on_plane_and_cylinder(
            &ellipse,
            Point3::origin(),
            normal,
            Vector3::unit_z(),
            Point3::origin(),
            1.0,
        );
    }

    #[test]
    fn test_plane_cylinder_near_parallel_returns_none() {
        // Plane normal nearly perpendicular to cylinder axis (nearly parallel plane)
        // dot(normal, axis) < 1e-6, so should return None
        let normal = Vector3::new(1.0, 0.0, 1e-7).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };
        let cyl = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };
        assert!(
            compute_plane_cylinder_intersection(&plane, &cyl).is_none(),
            "Near-parallel plane should return None"
        );
    }

    #[test]
    fn test_plane_cylinder_arbitrary_orientation() {
        // Cylinder with axis along [1,1,1] normalized
        let axis = Vector3::new(1.0, 1.0, 1.0).normalize();
        let cyl = CylinderParams {
            center: Point3::new(1.0, 2.0, 3.0),
            axis,
            radius: 0.5,
        };
        // Plane perpendicular to cylinder axis at center
        let plane = PlaneParams {
            origin: Point3::new(1.0, 2.0, 3.0),
            normal: axis,
        };

        let ellipse = compute_plane_cylinder_intersection(&plane, &cyl).unwrap();

        // Perpendicular cut → circle with radius 0.5
        assert!(
            (ellipse.axis_u.magnitude() - 0.5).abs() < 1e-10,
            "axis_u mag = {}",
            ellipse.axis_u.magnitude()
        );
        assert!(
            (ellipse.axis_v.magnitude() - 0.5).abs() < 1e-10,
            "axis_v mag = {}",
            ellipse.axis_v.magnitude()
        );
        assert!(
            ellipse.axis_u.dot(ellipse.axis_v).abs() < 1e-10,
            "axes not perpendicular"
        );

        assert_points_on_plane_and_cylinder(
            &ellipse,
            Point3::new(1.0, 2.0, 3.0),
            axis,
            axis,
            Point3::new(1.0, 2.0, 3.0),
            0.5,
        );
    }

    #[test]
    fn test_detect_cylinder_arbitrary_axis() {
        // Cylinder with axis along [1,1,1] normalized
        let axis = Vector3::new(1.0, 1.0, 1.0).normalize();
        // Build a line at distance 3.0 from the axis
        // Find a point perpendicular to axis at distance 3
        let perp = axis.cross(Vector3::unit_x()).normalize();
        let p0 = Point3::origin() + 3.0 * perp;
        let p1 = p0 + 10.0 * axis;
        let line = Line(p0, p1);
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), axis);

        let detected = detect_cylinder(&cylinder);
        assert!(
            detected.is_some(),
            "Failed to detect arbitrary-axis cylinder"
        );
        let params = detected.unwrap();
        assert!(
            (params.axis - axis).magnitude() < 0.01 || (params.axis + axis).magnitude() < 0.01,
            "axis = {:?}, expected {:?}",
            params.axis,
            axis
        );
        assert!(
            (params.radius - 3.0).abs() < 0.05,
            "radius = {}, expected 3.0",
            params.radius
        );
    }

    // --- Plane-Cone tests (Sprint 40) ---

    /// Helper: verify all sampled points lie on both the plane and the cone.
    fn assert_points_on_plane_and_cone(
        ellipse: &EllipseParams,
        plane_origin: Point3,
        plane_normal: Vector3,
        cone_apex: Point3,
        cone_axis: Vector3,
        half_angle: f64,
    ) {
        let poly = sample_ellipse(ellipse, 128);
        for pt in &poly.0 {
            // On the plane
            let d = plane_normal.dot(*pt - plane_origin).abs();
            assert!(d < 1e-6, "point not on plane: dist = {d}");
            // On the cone: angle from apex to point along axis = half_angle
            let v = *pt - cone_apex;
            let v_mag = v.magnitude();
            if v_mag < 1e-12 {
                continue; // at apex, skip
            }
            let cos_angle = v.dot(cone_axis) / v_mag;
            let actual_angle = cos_angle.abs().acos();
            assert!(
                (actual_angle - half_angle).abs() < 1e-5,
                "point not on cone: angle = {}, expected {}",
                actual_angle.to_degrees(),
                half_angle.to_degrees()
            );
        }
    }

    #[test]
    fn test_plane_cone_perpendicular_circle() {
        // Cone with apex at origin, axis along Z, half-angle 30°
        let half_angle = 30.0_f64.to_radians();
        let cone = ConeParams {
            apex: Point3::origin(),
            axis: Vector3::unit_z(),
            half_angle,
        };
        // Plane perpendicular to axis at z=5
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal: Vector3::unit_z(),
        };

        let ellipse = compute_plane_cone_intersection(&plane, &cone).unwrap();

        // At z=5, cone radius = 5 * tan(30°) ≈ 2.887
        let expected_radius = 5.0 * half_angle.tan();
        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        // Perpendicular → circle: both axes equal
        assert!(
            (semi_a - expected_radius).abs() < 1e-6,
            "axis_u = {semi_a}, expected {expected_radius}"
        );
        assert!(
            (semi_b - expected_radius).abs() < 1e-6,
            "axis_v = {semi_b}, expected {expected_radius}"
        );
        // Axes perpendicular
        assert!(
            ellipse.axis_u.dot(ellipse.axis_v).abs() < 1e-8,
            "axes not perpendicular"
        );

        assert_points_on_plane_and_cone(
            &ellipse,
            Point3::new(0.0, 0.0, 5.0),
            Vector3::unit_z(),
            Point3::origin(),
            Vector3::unit_z(),
            half_angle,
        );
    }

    #[test]
    fn test_plane_cone_oblique_ellipse() {
        // Cone with apex at origin, axis along Z, half-angle 20°
        let half_angle = 20.0_f64.to_radians();
        let cone = ConeParams {
            apex: Point3::origin(),
            axis: Vector3::unit_z(),
            half_angle,
        };
        // Plane tilted 15° from perpendicular (normal at 15° from Z-axis)
        let tilt = 15.0_f64.to_radians();
        let normal = Vector3::new(0.0, tilt.sin(), tilt.cos()).normalize();
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal,
        };

        let ellipse = compute_plane_cone_intersection(&plane, &cone).unwrap();

        // Should be an ellipse (not a circle since tilt ≠ 0)
        let semi_a = ellipse.axis_u.magnitude();
        let semi_b = ellipse.axis_v.magnitude();
        let (minor, major) = if semi_a < semi_b {
            (semi_a, semi_b)
        } else {
            (semi_b, semi_a)
        };
        assert!(
            major > minor * 1.01,
            "Expected ellipse (major > minor), got semi_a={semi_a}, semi_b={semi_b}"
        );

        assert_points_on_plane_and_cone(
            &ellipse,
            Point3::new(0.0, 0.0, 5.0),
            normal,
            Point3::origin(),
            Vector3::unit_z(),
            half_angle,
        );
    }

    #[test]
    fn test_plane_cone_through_apex_returns_none() {
        // Plane passes through the apex → degenerate (two lines)
        let half_angle = 30.0_f64.to_radians();
        let cone = ConeParams {
            apex: Point3::origin(),
            axis: Vector3::unit_z(),
            half_angle,
        };
        let plane = PlaneParams {
            origin: Point3::origin(), // passes through apex
            normal: Vector3::unit_z(),
        };

        assert!(
            compute_plane_cone_intersection(&plane, &cone).is_none(),
            "Plane through apex should return None"
        );
    }

    #[test]
    fn test_plane_cone_parallel_to_generator_returns_none() {
        // Plane normal perpendicular to generator → parabola
        // For half-angle α, generator direction is (sin(α), 0, cos(α)).
        // Plane parallel to generator means n · generator = 0, so
        // normal in the plane containing the generator and perpendicular to it:
        // n = (cos(α), 0, -sin(α))  →  cos(β) = |n·a| = sin(α) → parabola
        let half_angle = 30.0_f64.to_radians();
        let cone = ConeParams {
            apex: Point3::origin(),
            axis: Vector3::unit_z(),
            half_angle,
        };
        let normal = Vector3::new(half_angle.cos(), 0.0, -half_angle.sin()).normalize();
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal,
        };

        assert!(
            compute_plane_cone_intersection(&plane, &cone).is_none(),
            "Plane parallel to generator should return None (parabola)"
        );
    }

    #[test]
    fn test_plane_cone_hyperbola_returns_none() {
        // Plane nearly parallel to axis → cos(β) < sin(α) → hyperbola
        let half_angle = 45.0_f64.to_radians();
        let cone = ConeParams {
            apex: Point3::origin(),
            axis: Vector3::unit_z(),
            half_angle,
        };
        // Normal nearly perpendicular to axis: cos(β) ≈ 0 < sin(45°) ≈ 0.707
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal: Vector3::unit_x(),
        };

        assert!(
            compute_plane_cone_intersection(&plane, &cone).is_none(),
            "Hyperbola case should return None"
        );
    }

    #[test]
    fn test_detect_cone_from_revolved_line() {
        // A cone is a RevolutedCurve<Line> where the line is angled
        // Line from (0,0,0) to (1,0,5) revolving around Z-axis → cone
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 5.0));
        let cone_surface = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());

        let detected = detect_cone(&cone_surface);
        assert!(
            detected.is_some(),
            "Failed to detect cone from RevolutedCurve"
        );
        let params = detected.unwrap();
        assert!(
            (params.axis - Vector3::unit_z()).magnitude() < 0.01
                || (params.axis + Vector3::unit_z()).magnitude() < 0.01,
            "axis = {:?}",
            params.axis
        );
        // half_angle = atan(2/5) ≈ 21.8°
        let expected_half_angle = (2.0_f64 / 5.0).atan();
        assert!(
            (params.half_angle - expected_half_angle).abs() < 0.05,
            "half_angle = {} deg, expected {} deg",
            params.half_angle.to_degrees(),
            expected_half_angle.to_degrees()
        );
    }

    #[test]
    fn test_cylinder_not_detected_as_cone() {
        // A cylinder should NOT be detected as a cone (constant radius)
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(
            detect_cone(&cylinder).is_none(),
            "Cylinder should not be detected as cone"
        );
    }

    #[test]
    fn test_plane_cone_full_detection_pipeline() {
        // Full end-to-end test: RevolutedCurve cone + truck Plane → analytical IC
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 5.0));
        let cone_surface = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        let plane = Plane::new(
            Point3::new(-5.0, -5.0, 3.0),
            Point3::new(5.0, -5.0, 3.0),
            Point3::new(-5.0, 5.0, 3.0),
        );

        let result = try_analytical_plane_cone_ic(&plane, &cone_surface, 0.05);
        assert!(
            result.is_some(),
            "Failed to detect plane-cone pair from truck surfaces"
        );

        // Verify the resulting IC
        let ic = result.unwrap();
        let poly = sample_ellipse(&ic.ellipses[0], 64);
        for pt in &poly.0 {
            // All points should be at z ≈ 3
            assert!((pt.z - 3.0).abs() < 0.1, "z = {}, expected ~3.0", pt.z);
            // At z=3, cone radius = 3 * (2/5) = 1.2
            let r = (pt.x * pt.x + pt.y * pt.y).sqrt();
            assert!(
                (r - 1.2).abs() < 0.1,
                "r = {r}, expected ~1.2 (cone at z=3)"
            );
        }
    }

    // --- Sphere-Plane tests (Sprint 41) ---

    /// Helper: verify all sampled points lie on both the plane and the sphere.
    fn assert_points_on_plane_and_sphere(
        ellipse: &EllipseParams,
        plane_origin: Point3,
        plane_normal: Vector3,
        sphere_center: Point3,
        sphere_radius: f64,
    ) {
        let poly = sample_ellipse(ellipse, 128);
        for pt in &poly.0 {
            // On the plane
            let d = plane_normal.dot(*pt - plane_origin).abs();
            assert!(d < 1e-8, "point not on plane: dist = {d}");
            // On the sphere
            let r = (*pt - sphere_center).magnitude();
            assert!(
                (r - sphere_radius).abs() < 1e-8,
                "point not on sphere: dist from center = {r}, expected {sphere_radius}"
            );
        }
    }

    #[test]
    fn test_sphere_plane_equator() {
        // Sphere centered at origin, radius 5.
        // Plane through center (equator) → circle of radius 5.
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 5.0,
        };
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal: Vector3::unit_z(),
        };

        let ellipse = compute_sphere_plane_intersection(&plane, &sphere).unwrap();

        // Circle in XY plane with radius 5
        assert!((ellipse.axis_u.magnitude() - 5.0).abs() < 1e-10);
        assert!((ellipse.axis_v.magnitude() - 5.0).abs() < 1e-10);
        assert!(ellipse.axis_u.dot(ellipse.axis_v).abs() < 1e-10);

        assert_points_on_plane_and_sphere(
            &ellipse,
            Point3::origin(),
            Vector3::unit_z(),
            Point3::origin(),
            5.0,
        );
    }

    #[test]
    fn test_sphere_plane_offset_circle() {
        // Sphere radius 5 at origin, plane at z=3.
        // Circle radius = sqrt(25 - 9) = 4.
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 5.0,
        };
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 3.0),
            normal: Vector3::unit_z(),
        };

        let ellipse = compute_sphere_plane_intersection(&plane, &sphere).unwrap();

        assert!((ellipse.axis_u.magnitude() - 4.0).abs() < 1e-10);
        assert!((ellipse.axis_v.magnitude() - 4.0).abs() < 1e-10);
        assert!((ellipse.center.z - 3.0).abs() < 1e-10);

        assert_points_on_plane_and_sphere(
            &ellipse,
            Point3::new(0.0, 0.0, 3.0),
            Vector3::unit_z(),
            Point3::origin(),
            5.0,
        );
    }

    #[test]
    fn test_sphere_plane_tangent_returns_none() {
        // Plane tangent to sphere — just touching
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 5.0,
        };
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 5.0),
            normal: Vector3::unit_z(),
        };

        assert!(compute_sphere_plane_intersection(&plane, &sphere).is_none());
    }

    #[test]
    fn test_sphere_plane_no_intersection_returns_none() {
        // Plane doesn't intersect sphere at all
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 5.0,
        };
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 10.0),
            normal: Vector3::unit_z(),
        };

        assert!(compute_sphere_plane_intersection(&plane, &sphere).is_none());
    }

    #[test]
    fn test_sphere_plane_oblique_through_center() {
        // Sphere radius 3 at (1, 2, 3), oblique plane through sphere center.
        // Plane through center → circle radius = sphere radius = 3.
        let sphere = SphereParams {
            center: Point3::new(1.0, 2.0, 3.0),
            radius: 3.0,
        };
        let normal = Vector3::new(1.0, 1.0, 1.0).normalize();
        let plane = PlaneParams {
            origin: Point3::new(1.0, 2.0, 3.0),
            normal,
        };

        let ellipse = compute_sphere_plane_intersection(&plane, &sphere).unwrap();

        assert!((ellipse.axis_u.magnitude() - 3.0).abs() < 1e-10);
        assert!((ellipse.axis_v.magnitude() - 3.0).abs() < 1e-10);

        assert_points_on_plane_and_sphere(
            &ellipse,
            Point3::new(1.0, 2.0, 3.0),
            normal,
            Point3::new(1.0, 2.0, 3.0),
            3.0,
        );
    }

    #[test]
    fn test_sphere_plane_small_circle() {
        // Sphere radius 10 at origin, plane at z=9.
        // Circle radius = sqrt(100 - 81) = sqrt(19) ≈ 4.359.
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 10.0,
        };
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 9.0),
            normal: Vector3::unit_z(),
        };

        let ellipse = compute_sphere_plane_intersection(&plane, &sphere).unwrap();
        let expected_r = 19.0_f64.sqrt();

        assert!(
            (ellipse.axis_u.magnitude() - expected_r).abs() < 1e-10,
            "axis_u = {}, expected {}",
            ellipse.axis_u.magnitude(),
            expected_r
        );
        assert!(
            (ellipse.axis_v.magnitude() - expected_r).abs() < 1e-10,
            "axis_v = {}, expected {}",
            ellipse.axis_v.magnitude(),
            expected_r
        );

        assert_points_on_plane_and_sphere(
            &ellipse,
            Point3::new(0.0, 0.0, 9.0),
            Vector3::unit_z(),
            Point3::origin(),
            10.0,
        );
    }

    #[test]
    fn test_sphere_plane_negative_offset() {
        // Sphere radius 5 at origin, plane at z=-4.
        // Circle radius = sqrt(25 - 16) = 3.
        let sphere = SphereParams {
            center: Point3::origin(),
            radius: 5.0,
        };
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, -4.0),
            normal: Vector3::unit_z(),
        };

        let ellipse = compute_sphere_plane_intersection(&plane, &sphere).unwrap();

        assert!((ellipse.axis_u.magnitude() - 3.0).abs() < 1e-10);
        assert!((ellipse.axis_v.magnitude() - 3.0).abs() < 1e-10);
        assert!((ellipse.center.z - (-4.0)).abs() < 1e-10);

        assert_points_on_plane_and_sphere(
            &ellipse,
            Point3::new(0.0, 0.0, -4.0),
            Vector3::unit_z(),
            Point3::origin(),
            5.0,
        );
    }

    #[test]
    fn test_cylinder_not_detected_as_sphere() {
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(
            detect_sphere(&cylinder).is_none(),
            "Cylinder should not be detected as sphere"
        );
    }

    #[test]
    fn test_cone_not_detected_as_sphere() {
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 5.0));
        let cone = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(
            detect_sphere(&cone).is_none(),
            "Cone should not be detected as sphere"
        );
    }

    #[test]
    fn test_plane_not_detected_as_sphere() {
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(
            detect_sphere(&plane).is_none(),
            "Plane should not be detected as sphere"
        );
    }

    #[test]
    fn test_sphere_pipeline_rejects_cylinder() {
        // try_analytical_plane_sphere_ic should return None for plane+cylinder
        let plane = Plane::new(
            Point3::new(-5.0, -5.0, 3.0),
            Point3::new(5.0, -5.0, 3.0),
            Point3::new(-5.0, 5.0, 3.0),
        );
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());

        assert!(
            try_analytical_plane_sphere_ic(&plane, &cylinder, 0.05).is_none(),
            "Cylinder should not be detected as sphere in pipeline"
        );
    }

    // --- Cylinder-Cylinder tests (Sprint 42) ---

    /// Helper: verify all sampled points lie on both cylinders.
    fn assert_points_on_both_cylinders(
        ellipse: &EllipseParams,
        cyl0: &CylinderParams,
        cyl1: &CylinderParams,
        tol: f64,
    ) {
        let poly = sample_ellipse(ellipse, 128);
        for (i, pt) in poly.0.iter().enumerate() {
            // Distance from cylinder 0 axis
            let v0 = *pt - cyl0.center;
            let along0 = v0.dot(cyl0.axis) * cyl0.axis;
            let perp0 = v0 - along0;
            let r0 = perp0.magnitude();
            assert!(
                (r0 - cyl0.radius).abs() < tol,
                "point[{}] not on cyl0: r={:.8}, expected {:.8}, diff={:.2e}",
                i,
                r0,
                cyl0.radius,
                (r0 - cyl0.radius).abs()
            );

            // Distance from cylinder 1 axis
            let v1 = *pt - cyl1.center;
            let along1 = v1.dot(cyl1.axis) * cyl1.axis;
            let perp1 = v1 - along1;
            let r1 = perp1.magnitude();
            assert!(
                (r1 - cyl1.radius).abs() < tol,
                "point[{}] not on cyl1: r={:.8}, expected {:.8}, diff={:.2e}",
                i,
                r1,
                cyl1.radius,
                (r1 - cyl1.radius).abs()
            );
        }
    }

    /// CC1: Two perpendicular equal-radius cylinders along X and Z axes.
    #[test]
    fn test_cc1_perpendicular_equal_radius() {
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_x(),
            radius: 2.0,
        };

        let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1).unwrap();
        assert_eq!(ellipses.len(), 2, "Expected 2 ellipses");

        // Both centers at origin
        for (i, ell) in ellipses.iter().enumerate() {
            assert!(
                (ell.center - Point3::origin()).magnitude() < 1e-10,
                "ellipse[{}] center not at origin: {:?}",
                i,
                ell.center
            );
        }

        // For perpendicular axes: both semi-axis pairs are (R*sqrt(2), R)
        let r = 2.0;
        let expected_major = r * 2.0_f64.sqrt();
        for (i, ell) in ellipses.iter().enumerate() {
            let su = ell.axis_u.magnitude();
            let sv = ell.axis_v.magnitude();
            let (minor, major) = if su < sv { (su, sv) } else { (sv, su) };
            assert!(
                (minor - r).abs() < 1e-10,
                "ellipse[{}] minor={:.8}, expected {:.8}",
                i,
                minor,
                r
            );
            assert!(
                (major - expected_major).abs() < 1e-10,
                "ellipse[{}] major={:.8}, expected {:.8}",
                i,
                major,
                expected_major
            );
        }

        // All points lie on both cylinders
        for ell in &ellipses {
            assert_points_on_both_cylinders(ell, &cyl0, &cyl1, 1e-6);
        }
    }

    /// CC2: Unequal radii → should return None.
    #[test]
    fn test_cc2_unequal_radii_returns_none() {
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_x(),
            radius: 3.0,
        };

        assert!(
            compute_cylinder_cylinder_intersection(&cyl0, &cyl1).is_none(),
            "Unequal radii should return None"
        );
    }

    /// CC3: Parallel axes → should return None.
    #[test]
    fn test_cc3_parallel_axes_returns_none() {
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::new(5.0, 0.0, 0.0),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };

        assert!(
            compute_cylinder_cylinder_intersection(&cyl0, &cyl1).is_none(),
            "Parallel axes should return None"
        );
    }

    /// CC4: Coaxial cylinders → should return None.
    #[test]
    fn test_cc4_coaxial_returns_none() {
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::new(0.0, 0.0, 5.0),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };

        assert!(
            compute_cylinder_cylinder_intersection(&cyl0, &cyl1).is_none(),
            "Coaxial cylinders should return None"
        );
    }

    /// CC5: Skew (non-intersecting) axes → should return None.
    #[test]
    fn test_cc5_skew_axes_returns_none() {
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };
        // Axis along X but offset in Y by 10 (well beyond 5% of R=1)
        let cyl1 = CylinderParams {
            center: Point3::new(0.0, 10.0, 0.0),
            axis: Vector3::unit_x(),
            radius: 1.0,
        };

        assert!(
            compute_cylinder_cylinder_intersection(&cyl0, &cyl1).is_none(),
            "Skew axes should return None"
        );
    }

    /// CC6: 85-degree angle between axes (well above 60° threshold).
    #[test]
    fn test_cc6_85_degree_angle() {
        let alpha = 85.0_f64.to_radians();
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.5,
        };
        let cyl1 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::new(alpha.sin(), 0.0, alpha.cos()).normalize(),
            radius: 1.5,
        };

        let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1).unwrap();
        assert_eq!(ellipses.len(), 2);

        let r = 1.5;
        let half_alpha = alpha * 0.5;
        let expected_semi1 = r / half_alpha.sin();
        let expected_semi2 = r / half_alpha.cos();

        // Check semi-axes for curve 1
        let su = ellipses[0].axis_u.magnitude();
        let sv = ellipses[0].axis_v.magnitude();
        let (minor, major) = if su < sv { (su, sv) } else { (sv, su) };
        assert!(
            (minor - r).abs() < 1e-6,
            "curve1 minor={:.8}, expected {:.8}",
            minor,
            r
        );
        assert!(
            (major - expected_semi1).abs() < 1e-6,
            "curve1 major={:.8}, expected {:.8}",
            major,
            expected_semi1
        );

        // Check semi-axes for curve 2
        let su = ellipses[1].axis_u.magnitude();
        let sv = ellipses[1].axis_v.magnitude();
        let (minor, major) = if su < sv { (su, sv) } else { (sv, su) };
        assert!(
            (minor - r).abs() < 1e-6,
            "curve2 minor={:.8}, expected {:.8}",
            minor,
            r
        );
        assert!(
            (major - expected_semi2).abs() < 1e-6,
            "curve2 major={:.8}, expected {:.8}",
            major,
            expected_semi2
        );

        for ell in &ellipses {
            assert_points_on_both_cylinders(ell, &cyl0, &cyl1, 1e-6);
        }
    }

    /// CC7: Arbitrary orientation — cylinders with non-axis-aligned directions.
    #[test]
    fn test_cc7_arbitrary_orientation() {
        let a0 = Vector3::new(1.0, 1.0, 0.0).normalize();
        let a1 = Vector3::new(0.0, 0.0, 1.0);
        let cyl0 = CylinderParams {
            center: Point3::new(1.0, 2.0, 3.0),
            axis: a0,
            radius: 1.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::new(1.0, 2.0, 3.0),
            axis: a1,
            radius: 1.0,
        };

        let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1).unwrap();
        assert_eq!(ellipses.len(), 2);

        // Both centers at (1, 2, 3)
        for (i, ell) in ellipses.iter().enumerate() {
            assert!(
                (ell.center - Point3::new(1.0, 2.0, 3.0)).magnitude() < 1e-8,
                "ellipse[{}] center at {:?}, expected (1,2,3)",
                i,
                ell.center
            );
        }

        for ell in &ellipses {
            assert_points_on_both_cylinders(ell, &cyl0, &cyl1, 1e-6);
        }
    }

    /// CC8: Off-center intersection — axes don't pass through origin but do intersect.
    #[test]
    fn test_cc8_off_center_intersection() {
        let cyl0 = CylinderParams {
            center: Point3::new(5.0, 5.0, 0.0),
            axis: Vector3::unit_z(),
            radius: 3.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::new(5.0, 0.0, 5.0),
            axis: Vector3::unit_y(),
            radius: 3.0,
        };

        // Axes: (5,5,t) and (5,s,5) → intersect at (5,5,5)
        let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1).unwrap();
        assert_eq!(ellipses.len(), 2);

        // Center should be at (5, 5, 5)
        for (i, ell) in ellipses.iter().enumerate() {
            assert!(
                (ell.center - Point3::new(5.0, 5.0, 5.0)).magnitude() < 1e-8,
                "ellipse[{}] center at {:?}, expected (5,5,5)",
                i,
                ell.center
            );
        }

        for ell in &ellipses {
            assert_points_on_both_cylinders(ell, &cyl0, &cyl1, 1e-6);
        }
    }

    /// CC9: Detection guards — near-parallel angle (55°) returns None.
    #[test]
    fn test_cc9_near_parallel_returns_none() {
        let alpha = 55.0_f64.to_radians(); // < 60° threshold
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 1.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::new(alpha.sin(), 0.0, alpha.cos()).normalize(),
            radius: 1.0,
        };

        assert!(
            compute_cylinder_cylinder_intersection(&cyl0, &cyl1).is_none(),
            "Angle {:.0}° < 60° should return None",
            alpha.to_degrees()
        );
    }

    /// CC10: Polyline refinement with pick_closest_ellipse.
    #[test]
    fn test_cc10_polyline_refinement() {
        // Two perpendicular equal-radius cylinders, R=2
        let cyl0 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let cyl1 = CylinderParams {
            center: Point3::origin(),
            axis: Vector3::unit_x(),
            radius: 2.0,
        };

        let ellipses = compute_cylinder_cylinder_intersection(&cyl0, &cyl1).unwrap();
        let analytical = AnalyticalIC {
            ellipses,
            revsurf_plane: None,
        };

        // Create a noisy polyline near the first ellipse
        let ell0 = &analytical.ellipses[0];
        let mesh_poly = PolylineCurve(
            (0..10)
                .map(|i| {
                    let theta = 2.0 * PI * i as f64 / 10.0;
                    // Exact point + noise
                    ell0.center
                        + theta.cos() * ell0.axis_u
                        + theta.sin() * ell0.axis_v
                        + Vector3::new(0.01, -0.02, 0.015)
                })
                .collect(),
        );

        let refined = refine_polyline(&mesh_poly, &analytical);
        assert_eq!(refined.0.len(), mesh_poly.0.len(), "Point count must match");

        // Refined points should be much closer to the ellipse than noisy ones
        for (i, pt) in refined.0.iter().enumerate() {
            let dist0 = dist_to_closest_ellipse(pt, &analytical.ellipses);
            assert!(
                dist0 < 1e-8,
                "refined point[{}] dist={:.2e} to nearest ellipse",
                i,
                dist0
            );
        }
    }

    /// CC11: pick_closest_ellipse selects the correct ellipse.
    #[test]
    fn test_cc11_pick_closest_ellipse() {
        let ell0 = EllipseParams {
            center: Point3::origin(),
            axis_u: Vector3::new(3.0, 0.0, 0.0),
            axis_v: Vector3::new(0.0, 2.0, 0.0),
        };
        let ell1 = EllipseParams {
            center: Point3::origin(),
            axis_u: Vector3::new(0.0, 3.0, 0.0),
            axis_v: Vector3::new(0.0, 0.0, 2.0),
        };

        // Polyline near ell0 (in XY plane)
        let poly_near_0 = PolylineCurve(vec![
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-3.0, 0.0, 0.0),
        ]);
        let candidates0 = [ell0.clone(), ell1.clone()];
        let chosen = pick_closest_ellipse(&poly_near_0, &candidates0);
        // Should pick ell0 (same plane)
        assert!(
            (chosen.center - ell0.center).magnitude() < 1e-10
                && (chosen.axis_u - ell0.axis_u).magnitude() < 1e-10,
            "Should pick ell0 for XY-plane polyline"
        );

        // Polyline near ell1 (in YZ plane)
        let poly_near_1 = PolylineCurve(vec![
            Point3::new(0.0, 3.0, 0.0),
            Point3::new(0.0, 0.0, 2.0),
            Point3::new(0.0, -3.0, 0.0),
        ]);
        let candidates1 = [ell0, ell1.clone()];
        let chosen = pick_closest_ellipse(&poly_near_1, &candidates1);
        assert!(
            (chosen.center - ell1.center).magnitude() < 1e-10
                && (chosen.axis_u - ell1.axis_u).magnitude() < 1e-10,
            "Should pick ell1 for YZ-plane polyline"
        );
    }

    /// CC12: Full pipeline — detect cylinder-cylinder from RevolutedCurve surfaces.
    #[test]
    fn test_cc12_full_pipeline() {
        // Cylinder 0: along Z-axis, radius 2
        let line0 = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cyl_surface0 =
            RevolutedCurve::by_revolution(line0, Point3::origin(), Vector3::unit_z());

        // Cylinder 1: along X-axis, radius 2
        let line1 = Line(Point3::new(0.0, 2.0, 0.0), Point3::new(10.0, 2.0, 0.0));
        let cyl_surface1 =
            RevolutedCurve::by_revolution(line1, Point3::origin(), Vector3::unit_x());

        let result = try_analytical_cylinder_cylinder_ic(&cyl_surface0, &cyl_surface1, 0.05);
        assert!(
            result.is_some(),
            "Failed to detect cylinder-cylinder pair from truck surfaces"
        );

        let ic = result.unwrap();
        assert_eq!(ic.ellipses.len(), 2, "Expected 2 ellipses");

        // Verify all points lie on both cylinders
        let cyl0_params = detect_cylinder(&cyl_surface0).unwrap();
        let cyl1_params = detect_cylinder(&cyl_surface1).unwrap();
        for ell in &ic.ellipses {
            assert_points_on_both_cylinders(ell, &cyl0_params, &cyl1_params, 1e-4);
        }
    }

    /// Helper: compute minimum distance from a point to any ellipse in the set.
    fn dist_to_closest_ellipse(pt: &Point3, ellipses: &[EllipseParams]) -> f64 {
        ellipses
            .iter()
            .map(|ell| (*pt - project_to_ellipse(pt, ell)).magnitude())
            .fold(f64::MAX, f64::min)
    }

    // --- Torus-Plane tests (Sprint 47 / Phase B) ---

    /// Helper: verify all sampled points lie on both the plane and the torus.
    fn assert_points_on_plane_and_torus(
        ellipse: &EllipseParams,
        plane_origin: Point3,
        plane_normal: Vector3,
        torus: &TorusParams,
        tol: f64,
    ) {
        let poly = sample_ellipse(ellipse, 128);
        for (i, pt) in poly.0.iter().enumerate() {
            // On the plane
            let d = plane_normal.dot(*pt - plane_origin).abs();
            assert!(d < tol, "point[{}] not on plane: dist = {:.2e}", i, d);
            // On the torus
            let td = torus_distance(pt, torus);
            assert!(td < tol, "point[{}] not on torus: dist = {:.2e}", i, td);
        }
    }

    /// TP1: Detect torus from truck's Torus surface type.
    #[test]
    fn test_tp1_detect_torus_from_truck_torus() {
        // Torus centered at origin, R=5, r=1, axis along Z
        let torus_surface = Torus::new(Point3::origin(), 5.0, 1.0);

        let detected = detect_torus(&torus_surface);
        assert!(detected.is_some(), "Failed to detect torus");
        let params = detected.unwrap();
        assert!(
            (params.axis - Vector3::unit_z()).magnitude() < 0.1
                || (params.axis + Vector3::unit_z()).magnitude() < 0.1,
            "axis = {:?}, expected Z-axis",
            params.axis
        );
        assert!(
            (params.major_radius - 5.0).abs() < 0.3,
            "major_radius = {}, expected 5.0",
            params.major_radius
        );
        assert!(
            (params.minor_radius - 1.0).abs() < 0.15,
            "minor_radius = {}, expected 1.0",
            params.minor_radius
        );
    }

    /// TP2: Cylinder should not be detected as torus.
    #[test]
    fn test_tp2_cylinder_not_detected_as_torus() {
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(
            detect_torus(&cylinder).is_none(),
            "Cylinder should not be detected as torus"
        );
    }

    /// TP3: Plane should not be detected as torus.
    #[test]
    fn test_tp3_plane_not_detected_as_torus() {
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(
            detect_torus(&plane).is_none(),
            "Plane should not be detected as torus"
        );
    }

    /// TP4: Cone should not be detected as torus.
    #[test]
    fn test_tp4_cone_not_detected_as_torus() {
        let line = Line(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 5.0));
        let cone = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        assert!(
            detect_torus(&cone).is_none(),
            "Cone should not be detected as torus"
        );
    }

    /// TP5: Torus-plane perpendicular intersection (2 circles).
    #[test]
    fn test_tp5_torus_perpendicular_plane_two_circles() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 1.0,
        };
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal: Vector3::unit_z(),
        };

        let ellipses = compute_torus_plane_intersection(&plane, &torus).unwrap();

        // Plane through center, perpendicular to axis → 2 circles
        // Outer: R + r = 6, Inner: R - r = 4
        assert_eq!(ellipses.len(), 2, "Expected 2 circles");

        let r0 = ellipses[0].axis_u.magnitude();
        let r1 = ellipses[1].axis_u.magnitude();
        let (outer, inner) = if r0 > r1 { (r0, r1) } else { (r1, r0) };

        assert!(
            (outer - 6.0).abs() < 1e-10,
            "outer radius = {outer}, expected 6.0"
        );
        assert!(
            (inner - 4.0).abs() < 1e-10,
            "inner radius = {inner}, expected 4.0"
        );

        for ell in &ellipses {
            assert_points_on_plane_and_torus(
                ell,
                Point3::origin(),
                Vector3::unit_z(),
                &torus,
                1e-6,
            );
        }
    }

    /// TP6: Torus-plane perpendicular offset (plane above center).
    #[test]
    fn test_tp6_torus_perpendicular_plane_offset() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 2.0,
        };
        // Plane at z=1 (within minor radius)
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 1.0),
            normal: Vector3::unit_z(),
        };

        let ellipses = compute_torus_plane_intersection(&plane, &torus).unwrap();
        assert_eq!(ellipses.len(), 2, "Expected 2 circles for offset plane");

        // h = sqrt(r² - d²) = sqrt(4 - 1) = sqrt(3)
        // outer = R + h ≈ 5 + 1.732 = 6.732
        // inner = R - h ≈ 5 - 1.732 = 3.268
        let h = 3.0_f64.sqrt();
        let expected_outer = 5.0 + h;
        let expected_inner = 5.0 - h;

        let r0 = ellipses[0].axis_u.magnitude();
        let r1 = ellipses[1].axis_u.magnitude();
        let (outer, inner) = if r0 > r1 { (r0, r1) } else { (r1, r0) };

        assert!(
            (outer - expected_outer).abs() < 1e-8,
            "outer = {outer}, expected {expected_outer}"
        );
        assert!(
            (inner - expected_inner).abs() < 1e-8,
            "inner = {inner}, expected {expected_inner}"
        );

        for ell in &ellipses {
            assert_points_on_plane_and_torus(
                ell,
                Point3::new(0.0, 0.0, 1.0),
                Vector3::unit_z(),
                &torus,
                1e-6,
            );
        }
    }

    /// TP7: Torus-plane perpendicular plane beyond extent → None.
    #[test]
    fn test_tp7_torus_perpendicular_no_intersection() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 1.0,
        };
        // Plane at z=2 — beyond minor radius (1.0)
        let plane = PlaneParams {
            origin: Point3::new(0.0, 0.0, 2.0),
            normal: Vector3::unit_z(),
        };

        assert!(
            compute_torus_plane_intersection(&plane, &torus).is_none(),
            "Plane beyond torus should return None"
        );
    }

    /// TP8: Torus-plane axial intersection (plane contains axis).
    #[test]
    fn test_tp8_torus_axial_plane() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 1.0,
        };
        // XZ plane (contains the Z axis)
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal: Vector3::unit_y(),
        };

        let ellipses = compute_torus_plane_intersection(&plane, &torus).unwrap();

        // Axial plane → 2 circles of radius r=1
        assert_eq!(ellipses.len(), 2, "Expected 2 circles for axial plane");

        for ell in &ellipses {
            let r = ell.axis_u.magnitude();
            assert!(
                (r - 1.0).abs() < 1e-8,
                "circle radius = {r}, expected 1.0 (minor radius)"
            );
            assert_points_on_plane_and_torus(
                ell,
                Point3::origin(),
                Vector3::unit_y(),
                &torus,
                1e-6,
            );
        }

        // Centers should be at distance R=5 from origin along X
        let c0_dist = (ellipses[0].center - Point3::origin()).magnitude();
        let c1_dist = (ellipses[1].center - Point3::origin()).magnitude();
        assert!(
            (c0_dist - 5.0).abs() < 1e-8,
            "center0 dist = {c0_dist}, expected 5.0"
        );
        assert!(
            (c1_dist - 5.0).abs() < 1e-8,
            "center1 dist = {c1_dist}, expected 5.0"
        );
    }

    /// TP9: Torus-plane oblique intersection.
    #[test]
    fn test_tp9_torus_oblique_plane() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 1.5,
        };
        // Plane tilted 45° from Z axis
        let normal = Vector3::new(0.0, 1.0, 1.0).normalize();
        let plane = PlaneParams {
            origin: Point3::origin(),
            normal,
        };

        let result = compute_torus_plane_intersection(&plane, &torus);
        assert!(
            result.is_some(),
            "Oblique torus-plane should produce curves"
        );

        let ellipses = result.unwrap();
        assert!(!ellipses.is_empty(), "Should produce at least one curve");

        // All sampled points should lie on both the plane and the torus
        for (i, ell) in ellipses.iter().enumerate() {
            let poly = sample_ellipse(ell, 64);
            for pt in &poly.0 {
                // On the plane
                let d = normal.dot(*pt - Point3::origin()).abs();
                // Oblique fit is approximate, allow larger tolerance
                assert!(d < 0.5, "ellipse[{i}] point not on plane: dist = {d:.4}");
                // On the torus
                let td = torus_distance(pt, &torus);
                assert!(td < 0.5, "ellipse[{i}] point not on torus: dist = {td:.4}");
            }
        }
    }

    /// TP10: Torus torus_distance function sanity check.
    #[test]
    fn test_tp10_torus_distance() {
        let torus = TorusParams {
            center: Point3::origin(),
            axis: Vector3::unit_z(),
            major_radius: 5.0,
            minor_radius: 1.0,
        };

        // Point on the torus: at angle theta=0, phi=0 → (R+r, 0, 0) = (6, 0, 0)
        let on_torus = Point3::new(6.0, 0.0, 0.0);
        assert!(
            torus_distance(&on_torus, &torus) < 1e-10,
            "Point on torus should have distance ~0"
        );

        // Point on torus: (R-r, 0, 0) = (4, 0, 0)
        let on_torus_inner = Point3::new(4.0, 0.0, 0.0);
        assert!(
            torus_distance(&on_torus_inner, &torus) < 1e-10,
            "Point on inner torus should have distance ~0"
        );

        // Point on torus: (0, R+r, 0) = (0, 6, 0)
        let on_torus_y = Point3::new(0.0, 6.0, 0.0);
        assert!(
            torus_distance(&on_torus_y, &torus) < 1e-10,
            "Point on torus Y-direction should have distance ~0"
        );

        // Point on torus: (R, 0, r) = (5, 0, 1)
        let on_torus_top = Point3::new(5.0, 0.0, 1.0);
        assert!(
            torus_distance(&on_torus_top, &torus) < 1e-10,
            "Point on top of torus tube should have distance ~0"
        );

        // Point outside torus: (10, 0, 0) → distance from tube = |10-5| - 1 = 4
        let outside = Point3::new(10.0, 0.0, 0.0);
        assert!(
            (torus_distance(&outside, &torus) - 4.0).abs() < 1e-10,
            "Point outside should have distance 4"
        );
    }

    /// TP11: Full pipeline — detect torus from truck Torus + plane → analytical IC.
    #[test]
    fn test_tp11_full_pipeline_torus_plane() {
        // Torus centered at origin, R=3, r=1, axis along Z
        let torus_surface = Torus::new(Point3::origin(), 3.0, 1.0);

        // Plane perpendicular to Z at z=0
        let plane = Plane::new(
            Point3::new(-10.0, -10.0, 0.0),
            Point3::new(10.0, -10.0, 0.0),
            Point3::new(-10.0, 10.0, 0.0),
        );

        let result = try_analytical_torus_plane_ic(&plane, &torus_surface, 0.05);
        assert!(
            result.is_some(),
            "Failed to detect torus-plane pair from truck surfaces"
        );

        let ic = result.unwrap();
        // Perpendicular plane through center → 2 circles: R+r=4, R-r=2
        assert!(
            ic.ellipses.len() >= 1,
            "Expected at least 1 circle, got {}",
            ic.ellipses.len()
        );
    }

    /// TP12: Torus pipeline accepts cylinder as revolution surface (valid fallback).
    ///
    /// A cylinder IS a valid revolution surface, so the revolution-axis fallback
    /// correctly detects it. In the real dispatch chain, the earlier
    /// `try_analytical_plane_cylinder_ic` takes priority and produces better
    /// ellipse-based refinement. But the fallback is still correct.
    #[test]
    fn test_tp12_torus_pipeline_cylinder_as_revsurf() {
        let plane = Plane::new(
            Point3::new(-5.0, -5.0, 3.0),
            Point3::new(5.0, -5.0, 3.0),
            Point3::new(-5.0, 5.0, 3.0),
        );
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cylinder = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());

        // The revolution-axis fallback detects this (cylinder is a revolution surface)
        let result = try_analytical_torus_plane_ic(&plane, &cylinder, 0.05);
        assert!(
            result.is_some(),
            "Cylinder is a valid revolution surface for revsurf_plane refinement"
        );
        // Full torus detection should NOT fire (cylinder is not a torus)
        let ic = result.unwrap();
        assert!(
            ic.ellipses.is_empty(),
            "No torus ellipses — this was detected via revolution axis fallback"
        );
        assert!(
            ic.revsurf_plane.is_some(),
            "Should have revsurf_plane params"
        );
    }

    /// TP13: Revolution axis detection on a RevolutedCurve (torus).
    #[test]
    fn test_tp13_detect_revolution_axis_from_torus() {
        let torus_surface = Torus::new(Point3::origin(), 5.0, 1.0);
        let axis = detect_revolution_axis(&torus_surface);
        assert!(axis.is_some(), "Should detect revolution axis on torus");
        let params = axis.unwrap();
        // Axis direction should be Z
        assert!(
            (params.axis_dir - Vector3::unit_z()).magnitude() < 0.1
                || (params.axis_dir + Vector3::unit_z()).magnitude() < 0.1,
            "axis_dir = {:?}, expected Z",
            params.axis_dir
        );
    }

    /// TP14: Revolution axis detection on a RevolutedCurve line (cylinder).
    #[test]
    fn test_tp14_detect_revolution_axis_from_cylinder() {
        let line = Line(Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 10.0));
        let cyl = RevolutedCurve::by_revolution(line, Point3::origin(), Vector3::unit_z());
        let axis = detect_revolution_axis(&cyl);
        assert!(axis.is_some(), "Should detect revolution axis on cylinder");
        let params = axis.unwrap();
        assert!(
            (params.axis_dir - Vector3::unit_z()).magnitude() < 0.1
                || (params.axis_dir + Vector3::unit_z()).magnitude() < 0.1,
            "axis_dir = {:?}, expected Z",
            params.axis_dir
        );
    }

    /// TP15: Plane should not be detected as revolution surface.
    #[test]
    fn test_tp15_plane_not_detected_as_revsurf() {
        let plane = Plane::new(
            Point3::origin(),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(
            detect_revolution_axis(&plane).is_none(),
            "Plane should not be detected as revolution surface"
        );
    }

    /// TP16: Per-point revsurf refinement projects onto circle-plane intersection.
    #[test]
    fn test_tp16_revsurf_plane_refinement() {
        let rsp = RevSurfPlaneParams {
            axis_point: Point3::origin(),
            axis_dir: Vector3::unit_z(),
            plane_origin: Point3::new(0.0, 0.0, 3.0),
            plane_normal: Vector3::unit_z(),
        };

        // A point near the torus at z≈3, radius≈5 from Z-axis
        let noisy_pt = Point3::new(5.1, 0.2, 2.8);
        let refined = refine_point_on_revsurf_plane(&noisy_pt, &rsp);

        // Refined point should be exactly on the plane (z = 3)
        assert!(
            (refined.z - 3.0).abs() < 1e-10,
            "refined.z = {}, expected 3.0",
            refined.z
        );

        // Refined point should preserve the revolution radius
        // (perpendicular distance from axis = same as original point)
        let orig_r = (noisy_pt.x * noisy_pt.x + noisy_pt.y * noisy_pt.y).sqrt();
        let refined_r = (refined.x * refined.x + refined.y * refined.y).sqrt();
        assert!(
            (refined_r - orig_r).abs() < 0.01,
            "refined_r = {}, orig_r = {}",
            refined_r,
            orig_r
        );
    }
}
