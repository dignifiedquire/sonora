//! Optimized vector math operations for the RNN VAD.
//!
//! Delegates to [`webrtc_simd`] for SIMD-accelerated dot product.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/vector_math.h`.

use webrtc_simd::SimdBackend;

/// Provides optimized mathematical operations on vectors.
#[derive(Debug, Clone, Copy)]
pub struct VectorMath {
    backend: SimdBackend,
}

impl VectorMath {
    /// Creates a new `VectorMath` using the given SIMD backend.
    pub fn new(backend: SimdBackend) -> Self {
        Self { backend }
    }

    /// Computes the dot product between two equally sized slices.
    pub fn dot_product(&self, x: &[f32], y: &[f32]) -> f32 {
        self.backend.dot_product(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webrtc_simd::detect_backend;

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const X: [f32; 19] = [
        0.31593041,
        0.9350786,
        -0.25252445,
        -0.86956251,
        -0.9673632,
        0.54571901,
        -0.72504495,
        -0.79509912,
        -0.25525012,
        -0.73340473,
        0.15747377,
        -0.04370565,
        0.76135145,
        -0.57239645,
        0.68616848,
        0.3740298,
        0.34710799,
        -0.92207423,
        0.10738454,
    ];
    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const ENERGY_OF_X: f32 = 7.315563958160327;
    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const ENERGY_OF_X_SUBSPAN: f32 = 6.333327669592963;
    const SIZE_OF_X_SUBSPAN: usize = 16;

    #[test]
    fn dot_product_scalar() {
        let vector_math = VectorMath::new(SimdBackend::Scalar);
        let energy = vector_math.dot_product(&X, &X);
        assert!(
            (energy - ENERGY_OF_X).abs() < 1e-5,
            "scalar full: {energy} != {ENERGY_OF_X}"
        );
        let energy_sub = vector_math.dot_product(&X[..SIZE_OF_X_SUBSPAN], &X[..SIZE_OF_X_SUBSPAN]);
        assert!(
            (energy_sub - ENERGY_OF_X_SUBSPAN).abs() < 1e-5,
            "scalar subspan: {energy_sub} != {ENERGY_OF_X_SUBSPAN}"
        );
    }

    #[test]
    fn dot_product_detected_backend() {
        let backend = detect_backend();
        let vector_math = VectorMath::new(backend);
        let energy = vector_math.dot_product(&X, &X);
        assert!(
            (energy - ENERGY_OF_X).abs() < 1e-5,
            "{}: full: {energy} != {ENERGY_OF_X}",
            backend.name()
        );
        let energy_sub = vector_math.dot_product(&X[..SIZE_OF_X_SUBSPAN], &X[..SIZE_OF_X_SUBSPAN]);
        assert!(
            (energy_sub - ENERGY_OF_X_SUBSPAN).abs() < 1e-5,
            "{}: subspan: {energy_sub} != {ENERGY_OF_X_SUBSPAN}",
            backend.name()
        );
    }
}
