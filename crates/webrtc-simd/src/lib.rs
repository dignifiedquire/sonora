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

/// Available SIMD backends, selected at runtime based on CPU features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Scalar fallback — works on all platforms.
    Scalar,
    /// x86/x86_64 SSE2 (128-bit, 4 floats at a time).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    /// x86/x86_64 AVX2 + FMA (256-bit, 8 floats at a time).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// ARM aarch64 NEON (128-bit, 4 floats at a time).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl SimdBackend {
    /// Returns the name of this backend.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => "sse2",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => "avx2+fma",
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon",
        }
    }

    /// Compute the dot product of two float slices.
    ///
    /// `a` and `b` must have the same length. Returns sum of a[i]*b[i].
    pub fn dot_product(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Self::Scalar => fallback::dot_product(a, b),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::dot_product(a, b) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::dot_product(a, b) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::dot_product(a, b) },
        }
    }

    /// Compute two dot products in parallel (for sinc resampler convolution).
    ///
    /// Returns (dot(input, k1), dot(input, k2)). All slices must have the
    /// same length.
    pub fn dual_dot_product(self, input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
        debug_assert_eq!(input.len(), k1.len());
        debug_assert_eq!(input.len(), k2.len());
        match self {
            Self::Scalar => fallback::dual_dot_product(input, k1, k2),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::dual_dot_product(input, k1, k2) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::dual_dot_product(input, k1, k2) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::dual_dot_product(input, k1, k2) },
        }
    }

    /// Element-wise multiply-accumulate: acc[i] += a[i] * b[i]
    ///
    /// `acc`, `a`, and `b` must have the same length.
    pub fn multiply_accumulate(self, acc: &mut [f32], a: &[f32], b: &[f32]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        match self {
            Self::Scalar => fallback::multiply_accumulate(acc, a, b),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::multiply_accumulate(acc, a, b) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::multiply_accumulate(acc, a, b) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::multiply_accumulate(acc, a, b) },
        }
    }

    /// Compute the sum of all elements in a slice.
    pub fn sum(self, x: &[f32]) -> f32 {
        match self {
            Self::Scalar => fallback::sum(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::sum(x) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::sum(x) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::sum(x) },
        }
    }
}

/// Detect the best available SIMD backend for the current CPU.
///
/// Uses runtime feature detection on x86/x86_64. On aarch64, NEON is
/// always available. Falls back to scalar on unknown architectures.
pub fn detect_backend() -> SimdBackend {
    #[cfg(feature = "force-scalar")]
    {
        return SimdBackend::Scalar;
    }

    #[cfg(not(feature = "force-scalar"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !cfg!(feature = "force-sse2") {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return SimdBackend::Avx2;
                }
            }
            if is_x86_feature_detected!("sse2") {
                return SimdBackend::Sse2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return SimdBackend::Neon;
        }

        #[allow(unreachable_code, reason = "fallback for architectures without SIMD")]
        SimdBackend::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        println!("Detected SIMD backend: {}", backend.name());
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn test_backend_is_copy() {
        let a = detect_backend();
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_dot_product_simple() {
        let ops = detect_backend();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let result = ops.dot_product(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_dual_dot_product_simple() {
        let ops = detect_backend();
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let k1 = [1.0f32, 0.0, 1.0, 0.0];
        let k2 = [0.0f32, 1.0, 0.0, 1.0];
        let (d1, d2) = ops.dual_dot_product(&input, &k1, &k2);
        assert!((d1 - 4.0).abs() < 1e-6);
        assert!((d2 - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_accumulate_simple() {
        let ops = detect_backend();
        let mut acc = [10.0f32, 20.0, 30.0, 40.0];
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        ops.multiply_accumulate(&mut acc, &a, &b);
        assert!((acc[0] - 15.0).abs() < 1e-6);
        assert!((acc[1] - 32.0).abs() < 1e-6);
        assert!((acc[2] - 51.0).abs() < 1e-6);
        assert!((acc[3] - 72.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_simple() {
        let ops = detect_backend();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((ops.sum(&x) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_slices() {
        let ops = detect_backend();
        assert_eq!(ops.dot_product(&[], &[]), 0.0);
        assert_eq!(ops.sum(&[]), 0.0);
        let (d1, d2) = ops.dual_dot_product(&[], &[], &[]);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
    }

    /// Compare SIMD backend against scalar fallback with larger inputs.
    #[test]
    fn test_dot_product_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

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
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

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
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

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
