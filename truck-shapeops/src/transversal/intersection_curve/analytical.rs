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

/// Parameters describing a plane-cylinder intersection ellipse/circle.
///
/// The curve is parameterized as:
///   X(θ) = center + cos(θ) * axis_u + sin(θ) * axis_v
/// For perpendicular cuts, |axis_u| = |axis_v| = radius (circle).
/// For oblique cuts, the magnitudes differ (ellipse).
#[derive(Debug, Clone)]
struct EllipseParams {
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

/// Opaque handle to an analytical intersection curve (ellipse or circle).
///
/// Used to refine mesh-based polylines by projecting their points onto
/// the exact intersection curve.
#[derive(Debug, Clone)]
pub struct AnalyticalIC {
    ellipse: EllipseParams,
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
    Some(AnalyticalIC { ellipse })
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
    Some(AnalyticalIC { ellipse })
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
    Some(AnalyticalIC { ellipse })
}

/// Refine a mesh-based polyline by projecting each point onto the analytical
/// intersection curve (ellipse/circle).
///
/// This preserves the mesh-based topology (start/end points, open/closed,
/// number of segments) while improving point accuracy. The refined points lie
/// exactly on the analytical intersection, eliminating BSpline drift.
pub fn refine_polyline(
    mesh_polyline: &PolylineCurve<Point3>,
    analytical: &AnalyticalIC,
) -> PolylineCurve<Point3> {
    let ellipse = &analytical.ellipse;
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
            ellipse: EllipseParams {
                center: Point3::new(0.0, 0.0, 5.0),
                axis_u: Vector3::new(2.0, 0.0, 0.0),
                axis_v: Vector3::new(0.0, 2.0, 0.0),
            },
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
        let poly = sample_ellipse(&ic.ellipse, 64);
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
}
