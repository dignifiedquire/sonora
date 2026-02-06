//! SIMD abstraction layer for WebRTC Audio Processing.
//!
//! Provides a portable interface over SSE2, AVX2+FMA, and NEON intrinsics
//! with runtime CPU feature detection and scalar fallback.
//!
//! # Design
//!
//! Rather than exposing low-level intrinsics, this crate exposes
//! **high-level operations** that audio processing algorithms need.
//! Each backend implements these operations using platform-specific
//! intrinsics, ensuring bit-exact compatibility with the C++ code.
//!
//! Operations are added incrementally as porting phases require them.

mod fallback;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;

/// High-level SIMD operations used across audio processing modules.
///
/// Each method has a scalar fallback and optional SIMD acceleration.
/// Implementations must produce identical results (within floating-point
/// representation) for the same inputs regardless of backend.
pub trait SimdOps: Send + Sync {
    /// Compute the dot product of two aligned float slices.
    ///
    /// `a` and `b` must have the same length. Returns sum of a[i]*b[i].
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute two dot products in parallel (for sinc resampler convolution).
    ///
    /// Returns (dot(input, k1), dot(input, k2)). All slices must have the
    /// same length.
    fn dual_dot_product(&self, input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32);

    /// Element-wise multiply-accumulate: acc[i] += a[i] * b[i]
    ///
    /// `acc`, `a`, and `b` must have the same length.
    fn multiply_accumulate(&self, acc: &mut [f32], a: &[f32], b: &[f32]);

    /// Compute the sum of all elements in a slice.
    fn sum(&self, x: &[f32]) -> f32;
}

/// Returns the best available SIMD backend for the current CPU.
///
/// Uses runtime feature detection on x86/x86_64. On aarch64, NEON is
/// always available. Falls back to scalar on unknown architectures.
pub fn get_simd_ops() -> &'static dyn SimdOps {
    #[cfg(feature = "force-scalar")]
    {
        return &fallback::ScalarOps;
    }

    #[cfg(not(feature = "force-scalar"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !cfg!(feature = "force-sse2") {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return &avx2::Avx2Ops;
                }
            }
            if is_x86_feature_detected!("sse2") {
                return &sse2::Sse2Ops;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return &neon::NeonOps;
        }

        #[allow(unreachable_code)]
        &fallback::ScalarOps
    }
}

/// Returns the name of the SIMD backend that would be selected.
pub fn detected_backend() -> &'static str {
    #[cfg(feature = "force-scalar")]
    {
        return "scalar";
    }

    #[cfg(not(feature = "force-scalar"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return "avx2+fma";
            }
            if is_x86_feature_detected!("sse2") {
                return "sse2";
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return "neon";
        }

        #[allow(unreachable_code)]
        "scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detected_backend() {
        let name = detected_backend();
        assert!(!name.is_empty());
        println!("Detected SIMD backend: {name}");
    }

    #[test]
    fn test_get_simd_ops_returns_valid() {
        let ops = get_simd_ops();
        // Basic sanity: dot product of empty slices is 0
        assert_eq!(ops.dot_product(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_simple() {
        let ops = get_simd_ops();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let result = ops.dot_product(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_dual_dot_product_simple() {
        let ops = get_simd_ops();
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let k1 = [1.0f32, 0.0, 1.0, 0.0];
        let k2 = [0.0f32, 1.0, 0.0, 1.0];
        let (d1, d2) = ops.dual_dot_product(&input, &k1, &k2);
        // d1 = 1*1 + 2*0 + 3*1 + 4*0 = 4
        // d2 = 1*0 + 2*1 + 3*0 + 4*1 = 6
        assert!((d1 - 4.0).abs() < 1e-6);
        assert!((d2 - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_accumulate_simple() {
        let ops = get_simd_ops();
        let mut acc = [10.0f32, 20.0, 30.0, 40.0];
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        ops.multiply_accumulate(&mut acc, &a, &b);
        // acc[0] = 10 + 1*5 = 15, acc[1] = 20 + 2*6 = 32, etc.
        assert!((acc[0] - 15.0).abs() < 1e-6);
        assert!((acc[1] - 32.0).abs() < 1e-6);
        assert!((acc[2] - 51.0).abs() < 1e-6);
        assert!((acc[3] - 72.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_simple() {
        let ops = get_simd_ops();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((ops.sum(&x) - 15.0).abs() < 1e-6);
    }

    /// Compare SIMD backend against scalar fallback with larger inputs.
    #[test]
    fn test_dot_product_matches_scalar() {
        let scalar = &fallback::ScalarOps as &dyn SimdOps;
        let simd = get_simd_ops();

        // Test with various sizes including non-multiples of 4 and 8
        for size in [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.005).collect();

            let scalar_result = scalar.dot_product(&a, &b);
            let simd_result = simd.dot_product(&a, &b);

            assert!(
                (scalar_result - simd_result).abs() < 1e-3,
                "Mismatch for size {size}: scalar={scalar_result}, simd={simd_result}"
            );
        }
    }

    #[test]
    fn test_dual_dot_product_matches_scalar() {
        let scalar = &fallback::ScalarOps as &dyn SimdOps;
        let simd = get_simd_ops();

        for size in [0, 1, 4, 7, 16, 31, 64, 128, 256] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let k1: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.003).collect();
            let k2: Vec<f32> = (0..size).map(|i| 0.5 + (i as f32) * 0.002).collect();

            let (s1, s2) = scalar.dual_dot_product(&input, &k1, &k2);
            let (d1, d2) = simd.dual_dot_product(&input, &k1, &k2);

            assert!(
                (s1 - d1).abs() < 1e-3,
                "k1 mismatch for size {size}: scalar={s1}, simd={d1}"
            );
            assert!(
                (s2 - d2).abs() < 1e-3,
                "k2 mismatch for size {size}: scalar={s2}, simd={d2}"
            );
        }
    }

    #[test]
    fn test_multiply_accumulate_matches_scalar() {
        let scalar = &fallback::ScalarOps as &dyn SimdOps;
        let simd = get_simd_ops();

        for size in [0, 1, 4, 7, 16, 31, 64, 128] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.005).collect();

            let mut acc_scalar: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let mut acc_simd = acc_scalar.clone();

            scalar.multiply_accumulate(&mut acc_scalar, &a, &b);
            simd.multiply_accumulate(&mut acc_simd, &a, &b);

            for i in 0..size {
                assert!(
                    (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                    "Mismatch at index {i} for size {size}: scalar={}, simd={}",
                    acc_scalar[i],
                    acc_simd[i]
                );
            }
        }
    }
}
