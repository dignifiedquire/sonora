//! SIMD-accelerated vector math operations for AEC3.
//!
//! Ported from `modules/audio_processing/aec3/vector_math.h`.
//! Delegates to `sonora_simd` for platform-specific acceleration.

use sonora_simd::SimdBackend;

/// Provides SIMD-optimized elementwise vector operations.
#[derive(Debug)]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::FFT_LENGTH_BY_2_PLUS_1;

    #[test]
    fn sqrt_matches_scalar() {
        let vm = VectorMath::new(sonora_simd::detect_backend());
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
}
