//! SIMD-accelerated vector math operations for AEC3.
//!
//! Ported from `modules/audio_processing/aec3/vector_math.h`.
//! Delegates to `webrtc_simd` for platform-specific acceleration.

use webrtc_simd::SimdBackend;

/// Provides SIMD-optimized elementwise vector operations.
pub(crate) struct VectorMath {
    backend: SimdBackend,
}

impl VectorMath {
    pub(crate) fn new(backend: SimdBackend) -> Self {
        Self { backend }
    }

    /// Elementwise square root: `x[k] = sqrt(x[k])`.
    pub(crate) fn sqrt(&self, x: &mut [f32]) {
        self.backend.elementwise_sqrt(x);
    }

    /// Elementwise multiply: `z[k] = x[k] * y[k]`.
    pub(crate) fn multiply(&self, x: &[f32], y: &[f32], z: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), z.len());
        self.backend.elementwise_multiply(x, y, z);
    }

    /// Elementwise accumulate: `z[k] += x[k]`.
    pub(crate) fn accumulate(&self, x: &[f32], z: &mut [f32]) {
        debug_assert_eq!(x.len(), z.len());
        self.backend.elementwise_accumulate(x, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::FFT_LENGTH_BY_2_PLUS_1;

    #[test]
    fn sqrt_matches_scalar() {
        let vm = VectorMath::new(webrtc_simd::detect_backend());
        let mut x = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for (k, v) in x.iter_mut().enumerate() {
            *v = (2.0 / 3.0) * k as f32;
        }
        let mut z = x;
        vm.sqrt(&mut z);
        for k in 0..z.len() {
            assert!(
                (z[k] - x[k].sqrt()).abs() < 0.0001,
                "mismatch at {k}: got {}, expected {}",
                z[k],
                x[k].sqrt()
            );
        }
    }

    #[test]
    fn multiply_matches_scalar() {
        let vm = VectorMath::new(webrtc_simd::detect_backend());
        let mut x = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut y = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..x.len() {
            x[k] = k as f32;
            y[k] = (2.0 / 3.0) * k as f32;
        }
        let mut z = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        vm.multiply(&x, &y, &mut z);
        for k in 0..z.len() {
            assert_eq!(z[k], x[k] * y[k], "mismatch at {k}");
        }
    }

    #[test]
    fn accumulate_matches_scalar() {
        let vm = VectorMath::new(webrtc_simd::detect_backend());
        let mut x = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut z = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..x.len() {
            x[k] = k as f32;
            z[k] = 2.0 * k as f32;
        }
        vm.accumulate(&x, &mut z);
        for k in 0..z.len() {
            assert_eq!(z[k], x[k] + 2.0 * x[k] as f32, "mismatch at {k}");
        }
    }
}
