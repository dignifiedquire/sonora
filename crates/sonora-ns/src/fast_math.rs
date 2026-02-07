//! Fast math approximations for the noise suppressor.
//!
//! These are intentionally approximate — they trade precision for speed.
//! The specific approximation methods must match the C++ implementation
//! exactly for bit-reproducible noise suppression output.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/fast_math.cc`

/// Fast log base 2 approximation using IEEE 754 float bit tricks.
///
/// Extracts the exponent from the float representation and uses it
/// as a rough log2 estimate. Accuracy is ~1% for typical inputs.
///
/// # Panics
///
/// Debug-asserts that `x > 0.0`.
fn fast_log2f(x: f32) -> f32 {
    debug_assert!(x > 0.0);
    // Reinterpret float bits as u32, convert to float, scale by 1/2^23,
    // and subtract bias to get approximate log2.
    let bits = x.to_bits();
    let out = bits as f32;
    out * 1.192_092_9e-7 - 126.942_695 // 1/2^23, bias
}

/// Fast square root approximation.
///
/// Currently delegates to `f32::sqrt()` (the C++ code has a TODO for
/// a fast implementation too).
pub fn sqrt_fast_approximation(f: f32) -> f32 {
    f.sqrt()
}

/// Fast natural log approximation: `ln(x) ≈ log2(x) * ln(2)`.
pub fn log_approximation(x: f32) -> f32 {
    use core::f32::consts::LN_2;
    fast_log2f(x) * LN_2
}

/// Batch natural log approximation.
pub fn log_approximation_batch(x: &[f32], y: &mut [f32]) {
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = log_approximation(*xi);
    }
}

/// Fast 2^x approximation.
///
/// Currently delegates to `f32::exp2()` (the C++ code has a TODO for
/// a fast implementation too).
pub fn pow2_approximation(p: f32) -> f32 {
    p.exp2()
}

/// Fast x^p approximation: `x^p = 2^(p * log2(x))`.
pub fn pow_approximation(x: f32, p: f32) -> f32 {
    pow2_approximation(p * fast_log2f(x))
}

/// Fast e^x approximation: `e^x = 10^(x * log10(e))`.
pub fn exp_approximation(x: f32) -> f32 {
    use core::f32::consts::LOG10_E;
    pow_approximation(10.0, x * LOG10_E)
}

/// Batch e^x approximation.
pub fn exp_approximation_batch(x: &[f32], y: &mut [f32]) {
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = exp_approximation(*xi);
    }
}

/// Batch e^(-x) approximation (sign-flipped exponent).
pub fn exp_approximation_sign_flip(x: &[f32], y: &mut [f32]) {
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi = exp_approximation(-*xi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log2f_powers_of_two() {
        // log2(1) = 0, log2(2) = 1, log2(4) = 2, etc.
        assert!((fast_log2f(1.0) - 0.0).abs() < 0.1);
        assert!((fast_log2f(2.0) - 1.0).abs() < 0.1);
        assert!((fast_log2f(4.0) - 2.0).abs() < 0.1);
        assert!((fast_log2f(8.0) - 3.0).abs() < 0.1);
        assert!((fast_log2f(0.5) - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn log_approximation_positive_values() {
        use core::f32::consts::E;
        // ln(1) = 0
        assert!(log_approximation(1.0).abs() < 0.1);
        // ln(e) ≈ 1
        assert!((log_approximation(E) - 1.0).abs() < 0.1);
    }

    #[test]
    fn exp_approximation_known_values() {
        use core::f32::consts::E;
        // e^0 = 1
        assert!((exp_approximation(0.0) - 1.0).abs() < 0.1);
        // e^1 ≈ 2.718
        assert!((exp_approximation(1.0) - E).abs() < 0.5);
    }

    #[test]
    fn pow_approximation_squares() {
        // 2^2 = 4
        assert!((pow_approximation(2.0, 2.0) - 4.0).abs() < 0.5);
        // 3^2 = 9
        assert!((pow_approximation(3.0, 2.0) - 9.0).abs() < 1.0);
    }

    #[test]
    fn sqrt_matches_std() {
        use core::f32::consts::SQRT_2;
        assert_eq!(sqrt_fast_approximation(4.0), 2.0);
        assert_eq!(sqrt_fast_approximation(0.0), 0.0);
        assert!((sqrt_fast_approximation(2.0) - SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn batch_operations() {
        let x = [1.0_f32, 2.0, 4.0, 8.0];
        let mut y = [0.0_f32; 4];

        log_approximation_batch(&x, &mut y);
        for (i, &yi) in y.iter().enumerate() {
            let expected = x[i].ln();
            assert!(
                (yi - expected).abs() < 0.2,
                "log mismatch at {i}: got {yi}, expected {expected}"
            );
        }

        let x2 = [0.0_f32, 0.5, 1.0, 2.0];
        exp_approximation_batch(&x2, &mut y);
        for (i, &yi) in y.iter().enumerate() {
            let expected = x2[i].exp();
            assert!(
                (yi - expected).abs() / expected.max(1.0) < 0.15,
                "exp mismatch at {i}: got {yi}, expected {expected}"
            );
        }
    }

    #[test]
    fn exp_sign_flip() {
        let x = [0.0_f32, 1.0, 2.0];
        let mut y = [0.0_f32; 3];
        exp_approximation_sign_flip(&x, &mut y);
        for (i, &yi) in y.iter().enumerate() {
            let expected = (-x[i]).exp();
            assert!(
                (yi - expected).abs() / expected.max(1.0) < 0.15,
                "exp_sign_flip mismatch at {i}: got {yi}, expected {expected}"
            );
        }
    }

    /// Verify our fast_log2f matches the C++ implementation exactly.
    /// The C++ uses the same bit-reinterpret trick with identical constants.
    #[test]
    fn fast_log2f_matches_cpp_bit_trick() {
        // The C++ function: float out = *(uint32_t*)&in; out *= 1.1920929e-7f; out -= 126.942695f;
        // Our Rust: let out = x.to_bits() as f32; out * 1.192_092_9e-7 - 126.942_695
        // These should produce identical results for the same input.
        let test_values = [0.001_f32, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0];
        for &v in &test_values {
            let bits = v.to_bits();
            let cpp_result = bits as f32 * 1.192_092_9e-7 - 126.942_695;
            let rust_result = fast_log2f(v);
            assert_eq!(
                rust_result, cpp_result,
                "fast_log2f({v}) mismatch: rust={rust_result}, cpp={cpp_result}"
            );
        }
    }
}
