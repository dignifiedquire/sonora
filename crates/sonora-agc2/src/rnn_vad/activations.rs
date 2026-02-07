//! Neural network activation functions.
//!
//! Ported from `webrtc/third_party/rnnoise/src/rnn_activations.h`.

/// Lookup table for the tanh approximation.
const TANSIG_TABLE: [f32; 201] = [
    0.000000, 0.039979, 0.079830, 0.119427, 0.158649, 0.197375, 0.235496, 0.272905, 0.309507,
    0.345214, 0.379949, 0.413644, 0.446244, 0.477700, 0.507977, 0.537050, 0.564900, 0.591519,
    0.616909, 0.641077, 0.664037, 0.685809, 0.706419, 0.725897, 0.744277, 0.761594, 0.777888,
    0.793199, 0.807569, 0.821040, 0.833655, 0.845456, 0.856485, 0.866784, 0.876393, 0.885352,
    0.893698, 0.901468, 0.908698, 0.915420, 0.921669, 0.927473, 0.932862, 0.937863, 0.942503,
    0.946806, 0.950795, 0.954492, 0.957917, 0.961090, 0.964028, 0.966747, 0.969265, 0.971594,
    0.973749, 0.975743, 0.977587, 0.979293, 0.980869, 0.982327, 0.983675, 0.984921, 0.986072,
    0.987136, 0.988119, 0.989027, 0.989867, 0.990642, 0.991359, 0.992020, 0.992631, 0.993196,
    0.993718, 0.994199, 0.994644, 0.995055, 0.995434, 0.995784, 0.996108, 0.996407, 0.996682,
    0.996937, 0.997172, 0.997389, 0.997590, 0.997775, 0.997946, 0.998104, 0.998249, 0.998384,
    0.998508, 0.998623, 0.998728, 0.998826, 0.998916, 0.999000, 0.999076, 0.999147, 0.999213,
    0.999273, 0.999329, 0.999381, 0.999428, 0.999472, 0.999513, 0.999550, 0.999585, 0.999617,
    0.999646, 0.999673, 0.999699, 0.999722, 0.999743, 0.999763, 0.999781, 0.999798, 0.999813,
    0.999828, 0.999841, 0.999853, 0.999865, 0.999875, 0.999885, 0.999893, 0.999902, 0.999909,
    0.999916, 0.999923, 0.999929, 0.999934, 0.999939, 0.999944, 0.999948, 0.999952, 0.999956,
    0.999959, 0.999962, 0.999965, 0.999968, 0.999970, 0.999973, 0.999975, 0.999977, 0.999978,
    0.999980, 0.999982, 0.999983, 0.999984, 0.999986, 0.999987, 0.999988, 0.999989, 0.999990,
    0.999990, 0.999991, 0.999992, 0.999992, 0.999993, 0.999994, 0.999994, 0.999994, 0.999995,
    0.999995, 0.999996, 0.999996, 0.999996, 0.999997, 0.999997, 0.999997, 0.999997, 0.999997,
    0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999999, 0.999999, 0.999999,
    0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
    0.999999, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000,
];

/// Approximated tanh using a lookup table with linear interpolation.
///
/// Must match the C++ `rnnoise::TansigApproximated` exactly for bit-exact
/// neural network inference.
#[inline]
pub fn tansig_approximated(x: f32) -> f32 {
    // Tests are reversed to catch NaNs — must match C++ exactly.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "intentional NaN handling matching C++"
    )]
    if !(x < 8.0) {
        return 1.0;
    }
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "intentional NaN handling matching C++"
    )]
    if !(x > -8.0) {
        return -1.0;
    }
    let (x_abs, sign) = if x < 0.0 {
        (-x, -1.0_f32)
    } else {
        (x, 1.0_f32)
    };
    // Look-up.
    let i = (0.5 + 25.0 * x_abs) as usize;
    let y = TANSIG_TABLE[i];
    // Map i back to x's scale (undo 25 factor).
    let x_residual = x_abs - 0.04 * i as f32;
    let y = y + x_residual * (1.0 - y * y) * (1.0 - y * x_residual);
    sign * y
}

/// Approximated sigmoid: `0.5 + 0.5 * tansig(0.5 * x)`.
#[inline]
pub fn sigmoid_approximated(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approximated(0.5 * x)
}

/// Rectified linear unit: `max(0, x)`.
#[inline]
pub fn rectified_linear_unit(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tansig_symmetry() {
        // tansig(-x) == -tansig(x)
        for &x in &[0.0_f32, 0.5, 1.0, 2.0, 4.0, 7.9] {
            let pos = tansig_approximated(x);
            let neg = tansig_approximated(-x);
            assert!(
                (pos + neg).abs() < 1e-6,
                "Symmetry failed for x={x}: tansig({x})={pos}, tansig(-{x})={neg}"
            );
        }
    }

    #[test]
    fn tansig_boundary_values() {
        assert_eq!(tansig_approximated(0.0), 0.0);
        assert_eq!(tansig_approximated(8.0), 1.0);
        assert_eq!(tansig_approximated(100.0), 1.0);
        assert_eq!(tansig_approximated(-8.0), -1.0);
        assert_eq!(tansig_approximated(-100.0), -1.0);
    }

    #[test]
    fn tansig_nan_handling() {
        // NaN should return 1.0 (same as C++ which tests !(x < 8.f) first)
        assert_eq!(tansig_approximated(f32::NAN), 1.0);
    }

    #[test]
    fn tansig_approximately_monotonic() {
        // The lookup table approximation can have tiny non-monotonicity at
        // f32 precision in the saturation region. Allow small regressions.
        let mut prev = tansig_approximated(-7.99);
        let mut x = -7.9;
        while x <= 7.99 {
            let current = tansig_approximated(x);
            assert!(
                current >= prev - 1e-6,
                "Not monotonic at x={x}: prev={prev}, current={current}"
            );
            prev = current;
            x += 0.01;
        }
    }

    #[test]
    fn sigmoid_boundary_values() {
        // sigmoid(0) = 0.5
        let s0 = sigmoid_approximated(0.0);
        assert!((s0 - 0.5).abs() < 1e-6, "sigmoid(0) = {s0}, expected 0.5");
        // sigmoid(large) ≈ 1.0
        assert!((sigmoid_approximated(100.0) - 1.0).abs() < 1e-4);
        // sigmoid(-large) ≈ 0.0
        assert!(sigmoid_approximated(-100.0).abs() < 1e-4);
    }

    #[test]
    fn relu_values() {
        assert_eq!(rectified_linear_unit(0.0), 0.0);
        assert_eq!(rectified_linear_unit(1.5), 1.5);
        assert_eq!(rectified_linear_unit(-1.5), 0.0);
        assert_eq!(rectified_linear_unit(-0.001), 0.0);
    }
}
