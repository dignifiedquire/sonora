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
            // SAFETY: detect_backend() only returns Sse2 after confirming sse2 support.
            Self::Sse2 => unsafe { sse2::dot_product(a, b) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma support.
            Self::Avx2 => unsafe { avx2::dot_product(a, b) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is always available on aarch64.
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
            // SAFETY: detect_backend() only returns Sse2 after confirming sse2 support.
            Self::Sse2 => unsafe { sse2::dual_dot_product(input, k1, k2) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma support.
            Self::Avx2 => unsafe { avx2::dual_dot_product(input, k1, k2) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is always available on aarch64.
            Self::Neon => unsafe { neon::dual_dot_product(input, k1, k2) },
        }
    }

    /// Sinc resampler convolution: dual dot product with interpolation.
    ///
    /// Computes `(1-f)*dot(input,k1) + f*dot(input,k2)` where `f` is the
    /// `kernel_interpolation_factor`. Unlike [`dual_dot_product`] followed by
    /// scalar interpolation, this performs interpolation on SIMD vectors
    /// *before* horizontal reduction, matching C++ `SincResampler::Convolve_*`
    /// rounding behavior exactly.
    ///
    /// The scalar fallback interpolates in `f64` matching C++ `Convolve_C`.
    pub fn convolve_sinc(
        self,
        input: &[f32],
        k1: &[f32],
        k2: &[f32],
        kernel_interpolation_factor: f64,
    ) -> f32 {
        debug_assert_eq!(input.len(), k1.len());
        debug_assert_eq!(input.len(), k2.len());
        match self {
            Self::Scalar => fallback::convolve_sinc(input, k1, k2, kernel_interpolation_factor),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Sse2 after confirming sse2 support.
            Self::Sse2 => unsafe {
                sse2::convolve_sinc(input, k1, k2, kernel_interpolation_factor)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma support.
            Self::Avx2 => unsafe {
                avx2::convolve_sinc(input, k1, k2, kernel_interpolation_factor)
            },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is always available on aarch64.
            Self::Neon => unsafe {
                neon::convolve_sinc(input, k1, k2, kernel_interpolation_factor)
            },
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
            // SAFETY: detect_backend() only returns Sse2 after confirming sse2 support.
            Self::Sse2 => unsafe { sse2::multiply_accumulate(acc, a, b) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma support.
            Self::Avx2 => unsafe { avx2::multiply_accumulate(acc, a, b) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is always available on aarch64.
            Self::Neon => unsafe { neon::multiply_accumulate(acc, a, b) },
        }
    }

    /// Compute the sum of all elements in a slice.
    pub fn sum(self, x: &[f32]) -> f32 {
        match self {
            Self::Scalar => fallback::sum(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Sse2 after confirming sse2 support.
            Self::Sse2 => unsafe { sse2::sum(x) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma support.
            Self::Avx2 => unsafe { avx2::sum(x) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is always available on aarch64.
            Self::Neon => unsafe { neon::sum(x) },
        }
    }

    /// Elementwise square root: x[i] = sqrt(x[i])
    pub fn elementwise_sqrt(self, x: &mut [f32]) {
        match self {
            Self::Scalar => fallback::elementwise_sqrt(x),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::elementwise_sqrt(x) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::elementwise_sqrt(x) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::elementwise_sqrt(x) },
        }
    }

    /// Elementwise vector multiplication: z[i] = x[i] * y[i]
    ///
    /// `x`, `y`, and `z` must have the same length.
    pub fn elementwise_multiply(self, x: &[f32], y: &[f32], z: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), z.len());
        match self {
            Self::Scalar => fallback::elementwise_multiply(x, y, z),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::elementwise_multiply(x, y, z) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::elementwise_multiply(x, y, z) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::elementwise_multiply(x, y, z) },
        }
    }

    /// Elementwise accumulate: z[i] += x[i]
    ///
    /// `x` and `z` must have the same length.
    pub fn elementwise_accumulate(self, x: &[f32], z: &mut [f32]) {
        debug_assert_eq!(x.len(), z.len());
        match self {
            Self::Scalar => fallback::elementwise_accumulate(x, z),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::elementwise_accumulate(x, z) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::elementwise_accumulate(x, z) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::elementwise_accumulate(x, z) },
        }
    }

    /// Compute the power spectrum: out[i] = re[i]^2 + im[i]^2
    ///
    /// `re`, `im`, and `out` must have the same length.
    pub fn power_spectrum(self, re: &[f32], im: &[f32], out: &mut [f32]) {
        debug_assert_eq!(re.len(), im.len());
        debug_assert_eq!(re.len(), out.len());
        match self {
            Self::Scalar => fallback::power_spectrum(re, im, out),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::power_spectrum(re, im, out) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::power_spectrum(re, im, out) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::power_spectrum(re, im, out) },
        }
    }

    /// Elementwise minimum: out[i] = min(a[i], b[i])
    ///
    /// `a`, `b`, and `out` must have the same length.
    pub fn elementwise_min(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        match self {
            Self::Scalar => fallback::elementwise_min(a, b, out),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::elementwise_min(a, b, out) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::elementwise_min(a, b, out) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::elementwise_min(a, b, out) },
        }
    }

    /// Elementwise maximum: out[i] = max(a[i], b[i])
    ///
    /// `a`, `b`, and `out` must have the same length.
    pub fn elementwise_max(self, a: &[f32], b: &[f32], out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        match self {
            Self::Scalar => fallback::elementwise_max(a, b, out),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe { sse2::elementwise_max(a, b, out) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { avx2::elementwise_max(a, b, out) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::elementwise_max(a, b, out) },
        }
    }

    /// Complex multiply-accumulate (AEC3 conjugate convention):
    ///   acc_re[i] += x_re[i]*h_re[i] + x_im[i]*h_im[i]
    ///   acc_im[i] += x_re[i]*h_im[i] - x_im[i]*h_re[i]
    ///
    /// All slices must have the same length.
    pub fn complex_multiply_accumulate(
        self,
        x_re: &[f32],
        x_im: &[f32],
        h_re: &[f32],
        h_im: &[f32],
        acc_re: &mut [f32],
        acc_im: &mut [f32],
    ) {
        debug_assert_eq!(x_re.len(), x_im.len());
        debug_assert_eq!(x_re.len(), h_re.len());
        debug_assert_eq!(x_re.len(), h_im.len());
        debug_assert_eq!(x_re.len(), acc_re.len());
        debug_assert_eq!(x_re.len(), acc_im.len());
        match self {
            Self::Scalar => {
                fallback::complex_multiply_accumulate(x_re, x_im, h_re, h_im, acc_re, acc_im);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe {
                sse2::complex_multiply_accumulate(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe {
                avx2::complex_multiply_accumulate(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe {
                neon::complex_multiply_accumulate(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
        }
    }
    /// Standard complex multiply-accumulate:
    ///   acc_re[i] += x_re[i]*h_re[i] - x_im[i]*h_im[i]
    ///   acc_im[i] += x_re[i]*h_im[i] + x_im[i]*h_re[i]
    ///
    /// All slices must have the same length.
    pub fn complex_multiply_accumulate_standard(
        self,
        x_re: &[f32],
        x_im: &[f32],
        h_re: &[f32],
        h_im: &[f32],
        acc_re: &mut [f32],
        acc_im: &mut [f32],
    ) {
        debug_assert_eq!(x_re.len(), x_im.len());
        debug_assert_eq!(x_re.len(), h_re.len());
        debug_assert_eq!(x_re.len(), h_im.len());
        debug_assert_eq!(x_re.len(), acc_re.len());
        debug_assert_eq!(x_re.len(), acc_im.len());
        match self {
            Self::Scalar => {
                fallback::complex_multiply_accumulate_standard(
                    x_re, x_im, h_re, h_im, acc_re, acc_im,
                );
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Sse2 => unsafe {
                sse2::complex_multiply_accumulate_standard(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2 => unsafe {
                avx2::complex_multiply_accumulate_standard(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe {
                neon::complex_multiply_accumulate_standard(x_re, x_im, h_re, h_im, acc_re, acc_im);
            },
        }
    }
}

// Runtime CPU feature detection via cpufeatures (atomic-cached).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(has_avx2_fma, "avx2", "fma");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(has_sse2, "sse2");

/// Detect the best available SIMD backend for the current CPU.
///
/// Uses runtime feature detection on x86/x86_64 (cached atomically after
/// first call via `cpufeatures`). On aarch64, NEON is always available.
/// Falls back to scalar on unknown architectures.
pub fn detect_backend() -> SimdBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if has_avx2_fma::get() {
            return SimdBackend::Avx2;
        }
        if has_sse2::get() {
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

    #[test]
    fn test_elementwise_sqrt_simple() {
        let ops = detect_backend();
        let mut x = [4.0f32, 9.0, 16.0, 25.0, 36.0];
        ops.elementwise_sqrt(&mut x);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 3.0).abs() < 1e-6);
        assert!((x[2] - 4.0).abs() < 1e-6);
        assert!((x[3] - 5.0).abs() < 1e-6);
        assert!((x[4] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_sqrt_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 15, 16, 31, 64, 65, 128] {
            let mut x_scalar: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5 + 0.1).collect();
            let mut x_simd = x_scalar.clone();

            scalar.elementwise_sqrt(&mut x_scalar);
            simd.elementwise_sqrt(&mut x_simd);

            for i in 0..size {
                assert!(
                    (x_scalar[i] - x_simd[i]).abs() < 1e-6,
                    "sqrt mismatch at index {i} for size {size}: scalar={}, simd={}",
                    x_scalar[i],
                    x_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_elementwise_multiply_simple() {
        let ops = detect_backend();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let mut z = [0.0f32; 5];
        ops.elementwise_multiply(&x, &y, &mut z);
        assert!((z[0] - 5.0).abs() < 1e-6);
        assert!((z[1] - 8.0).abs() < 1e-6);
        assert!((z[2] - 9.0).abs() < 1e-6);
        assert!((z[3] - 8.0).abs() < 1e-6);
        assert!((z[4] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_multiply_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let y: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.005).collect();
            let mut z_scalar = vec![0.0f32; size];
            let mut z_simd = vec![0.0f32; size];

            scalar.elementwise_multiply(&x, &y, &mut z_scalar);
            simd.elementwise_multiply(&x, &y, &mut z_simd);

            for i in 0..size {
                assert!(
                    (z_scalar[i] - z_simd[i]).abs() < 1e-6,
                    "multiply mismatch at index {i} for size {size}: scalar={}, simd={}",
                    z_scalar[i],
                    z_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_elementwise_accumulate_simple() {
        let ops = detect_backend();
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut z = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        ops.elementwise_accumulate(&x, &mut z);
        assert!((z[0] - 11.0).abs() < 1e-6);
        assert!((z[1] - 22.0).abs() < 1e-6);
        assert!((z[2] - 33.0).abs() < 1e-6);
        assert!((z[3] - 44.0).abs() < 1e-6);
        assert!((z[4] - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_accumulate_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let x: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let mut z_scalar: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let mut z_simd = z_scalar.clone();

            scalar.elementwise_accumulate(&x, &mut z_scalar);
            simd.elementwise_accumulate(&x, &mut z_simd);

            for i in 0..size {
                assert!(
                    (z_scalar[i] - z_simd[i]).abs() < 1e-6,
                    "accumulate mismatch at index {i} for size {size}: scalar={}, simd={}",
                    z_scalar[i],
                    z_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_power_spectrum_simple() {
        let ops = detect_backend();
        let re = [3.0f32, 0.0, 1.0, 2.0, 5.0];
        let im = [4.0f32, 1.0, 0.0, 3.0, 12.0];
        let mut out = [0.0f32; 5];
        ops.power_spectrum(&re, &im, &mut out);
        assert!((out[0] - 25.0).abs() < 1e-6); // 9 + 16
        assert!((out[1] - 1.0).abs() < 1e-6); // 0 + 1
        assert!((out[2] - 1.0).abs() < 1e-6); // 1 + 0
        assert!((out[3] - 13.0).abs() < 1e-6); // 4 + 9
        assert!((out[4] - 169.0).abs() < 1e-6); // 25 + 144
    }

    #[test]
    fn test_power_spectrum_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let re: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let im: Vec<f32> = (0..size).map(|i| 2.0 - (i as f32) * 0.07).collect();
            let mut out_scalar = vec![0.0f32; size];
            let mut out_simd = vec![0.0f32; size];

            scalar.power_spectrum(&re, &im, &mut out_scalar);
            simd.power_spectrum(&re, &im, &mut out_simd);

            for i in 0..size {
                assert!(
                    (out_scalar[i] - out_simd[i]).abs() < 1e-4,
                    "power_spectrum mismatch at index {i} for size {size}: scalar={}, simd={}",
                    out_scalar[i],
                    out_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_elementwise_min_simple() {
        let ops = detect_backend();
        let a = [1.0f32, 5.0, 3.0, 8.0, 2.0];
        let b = [4.0f32, 2.0, 7.0, 1.0, 9.0];
        let mut out = [0.0f32; 5];
        ops.elementwise_min(&a, &b, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_elementwise_min_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let b: Vec<f32> = (0..size).map(|i| 2.0 - (i as f32) * 0.07).collect();
            let mut out_scalar = vec![0.0f32; size];
            let mut out_simd = vec![0.0f32; size];

            scalar.elementwise_min(&a, &b, &mut out_scalar);
            simd.elementwise_min(&a, &b, &mut out_simd);

            for i in 0..size {
                assert!(
                    (out_scalar[i] - out_simd[i]).abs() < 1e-6,
                    "min mismatch at index {i} for size {size}: scalar={}, simd={}",
                    out_scalar[i],
                    out_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_elementwise_max_simple() {
        let ops = detect_backend();
        let a = [1.0f32, 5.0, 3.0, 8.0, 2.0];
        let b = [4.0f32, 2.0, 7.0, 1.0, 9.0];
        let mut out = [0.0f32; 5];
        ops.elementwise_max(&a, &b, &mut out);
        assert_eq!(out, [4.0, 5.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_elementwise_max_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let b: Vec<f32> = (0..size).map(|i| 2.0 - (i as f32) * 0.07).collect();
            let mut out_scalar = vec![0.0f32; size];
            let mut out_simd = vec![0.0f32; size];

            scalar.elementwise_max(&a, &b, &mut out_scalar);
            simd.elementwise_max(&a, &b, &mut out_simd);

            for i in 0..size {
                assert!(
                    (out_scalar[i] - out_simd[i]).abs() < 1e-6,
                    "max mismatch at index {i} for size {size}: scalar={}, simd={}",
                    out_scalar[i],
                    out_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_complex_multiply_accumulate_simple() {
        let ops = detect_backend();
        // (1+2j) * (3+4j) in AEC3 conjugate convention:
        //   re = 1*3 + 2*4 = 11
        //   im = 1*4 - 2*3 = -2
        let x_re = [1.0f32];
        let x_im = [2.0f32];
        let h_re = [3.0f32];
        let h_im = [4.0f32];
        let mut acc_re = [0.0f32];
        let mut acc_im = [0.0f32];
        ops.complex_multiply_accumulate(&x_re, &x_im, &h_re, &h_im, &mut acc_re, &mut acc_im);
        assert!((acc_re[0] - 11.0).abs() < 1e-6);
        assert!((acc_im[0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_complex_multiply_accumulate_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let x_re: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let x_im: Vec<f32> = (0..size).map(|i| 2.0 - (i as f32) * 0.07).collect();
            let h_re: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.5).collect();
            let h_im: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.03).collect();

            let mut acc_re_scalar = vec![0.5f32; size];
            let mut acc_im_scalar = vec![-0.3f32; size];
            let mut acc_re_simd = acc_re_scalar.clone();
            let mut acc_im_simd = acc_im_scalar.clone();

            scalar.complex_multiply_accumulate(
                &x_re,
                &x_im,
                &h_re,
                &h_im,
                &mut acc_re_scalar,
                &mut acc_im_scalar,
            );
            simd.complex_multiply_accumulate(
                &x_re,
                &x_im,
                &h_re,
                &h_im,
                &mut acc_re_simd,
                &mut acc_im_simd,
            );

            for i in 0..size {
                assert!(
                    (acc_re_scalar[i] - acc_re_simd[i]).abs() < 1e-4,
                    "cma re mismatch at {i} for size {size}: scalar={}, simd={}",
                    acc_re_scalar[i],
                    acc_re_simd[i]
                );
                assert!(
                    (acc_im_scalar[i] - acc_im_simd[i]).abs() < 1e-4,
                    "cma im mismatch at {i} for size {size}: scalar={}, simd={}",
                    acc_im_scalar[i],
                    acc_im_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_complex_multiply_accumulate_standard_simple() {
        let ops = detect_backend();
        // (1+2j) * (3+4j) in standard convention:
        //   re = 1*3 - 2*4 = -5
        //   im = 1*4 + 2*3 = 10
        let x_re = [1.0f32];
        let x_im = [2.0f32];
        let h_re = [3.0f32];
        let h_im = [4.0f32];
        let mut acc_re = [0.0f32];
        let mut acc_im = [0.0f32];
        ops.complex_multiply_accumulate_standard(
            &x_re,
            &x_im,
            &h_re,
            &h_im,
            &mut acc_re,
            &mut acc_im,
        );
        assert!((acc_re[0] - (-5.0)).abs() < 1e-6);
        assert!((acc_im[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_multiply_accumulate_standard_matches_scalar() {
        let scalar = SimdBackend::Scalar;
        let simd = detect_backend();

        for size in [0, 1, 4, 7, 8, 16, 31, 64, 65, 128] {
            let x_re: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let x_im: Vec<f32> = (0..size).map(|i| 2.0 - (i as f32) * 0.07).collect();
            let h_re: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.5).collect();
            let h_im: Vec<f32> = (0..size).map(|i| 1.0 - (i as f32) * 0.03).collect();

            let mut acc_re_scalar = vec![0.5f32; size];
            let mut acc_im_scalar = vec![-0.3f32; size];
            let mut acc_re_simd = acc_re_scalar.clone();
            let mut acc_im_simd = acc_im_scalar.clone();

            scalar.complex_multiply_accumulate_standard(
                &x_re,
                &x_im,
                &h_re,
                &h_im,
                &mut acc_re_scalar,
                &mut acc_im_scalar,
            );
            simd.complex_multiply_accumulate_standard(
                &x_re,
                &x_im,
                &h_re,
                &h_im,
                &mut acc_re_simd,
                &mut acc_im_simd,
            );

            for i in 0..size {
                assert!(
                    (acc_re_scalar[i] - acc_re_simd[i]).abs() < 1e-4,
                    "std cma re mismatch at {i} for size {size}: scalar={}, simd={}",
                    acc_re_scalar[i],
                    acc_re_simd[i]
                );
                assert!(
                    (acc_im_scalar[i] - acc_im_simd[i]).abs() < 1e-4,
                    "std cma im mismatch at {i} for size {size}: scalar={}, simd={}",
                    acc_im_scalar[i],
                    acc_im_simd[i]
                );
            }
        }
    }
}
