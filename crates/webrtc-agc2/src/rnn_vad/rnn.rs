//! RNN model for voice activity detection.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/rnn.cc`.

use super::common::FeatureVector;
use super::fc_layer::{ActivationFunction, FullyConnectedLayer};
use super::gru_layer::GatedRecurrentLayer;
use super::vector_math::VectorMath;
use super::weights::{
    HIDDEN_GRU_BIAS, HIDDEN_GRU_RECURRENT_WEIGHTS, HIDDEN_GRU_WEIGHTS, HIDDEN_LAYER_OUTPUT_SIZE,
    INPUT_DENSE_BIAS, INPUT_DENSE_WEIGHTS, INPUT_LAYER_INPUT_SIZE, INPUT_LAYER_OUTPUT_SIZE,
    OUTPUT_DENSE_BIAS, OUTPUT_DENSE_WEIGHTS, OUTPUT_LAYER_OUTPUT_SIZE,
};
use webrtc_simd::SimdBackend;

/// Recurrent network with hard-coded architecture and weights for VAD.
///
/// Architecture: FC(42→24, tanh) → GRU(24→24) → FC(24→1, sigmoid)
#[derive(Debug)]
pub struct RnnVad {
    input: FullyConnectedLayer,
    hidden: GatedRecurrentLayer,
    output: FullyConnectedLayer,
}

impl RnnVad {
    /// Creates a new RNN VAD model.
    pub fn new(backend: SimdBackend) -> Self {
        let vector_math = VectorMath::new(backend);
        let input = FullyConnectedLayer::new(
            INPUT_LAYER_INPUT_SIZE,
            INPUT_LAYER_OUTPUT_SIZE,
            &INPUT_DENSE_BIAS,
            &INPUT_DENSE_WEIGHTS,
            ActivationFunction::TansigApproximated,
            vector_math,
        );
        let hidden = GatedRecurrentLayer::new(
            INPUT_LAYER_OUTPUT_SIZE,
            HIDDEN_LAYER_OUTPUT_SIZE,
            &HIDDEN_GRU_BIAS,
            &HIDDEN_GRU_WEIGHTS,
            &HIDDEN_GRU_RECURRENT_WEIGHTS,
            vector_math,
        );
        // The output layer is just 24x1. The unoptimized code is faster.
        let scalar_math = VectorMath::new(SimdBackend::Scalar);
        let output = FullyConnectedLayer::new(
            HIDDEN_LAYER_OUTPUT_SIZE,
            OUTPUT_LAYER_OUTPUT_SIZE,
            &OUTPUT_DENSE_BIAS,
            &OUTPUT_DENSE_WEIGHTS,
            ActivationFunction::SigmoidApproximated,
            scalar_math,
        );
        debug_assert_eq!(input.size(), hidden.input_size());
        debug_assert_eq!(hidden.size(), output.input_size());
        Self {
            input,
            hidden,
            output,
        }
    }

    /// Resets the hidden state.
    pub fn reset(&mut self) {
        self.hidden.reset();
    }

    /// Observes `feature_vector` and `is_silence`, updates the RNN and returns
    /// the current voice probability.
    pub fn compute_vad_probability(
        &mut self,
        feature_vector: &FeatureVector,
        is_silence: bool,
    ) -> f32 {
        if is_silence {
            self.reset();
            return 0.0;
        }
        self.input.compute_output(feature_vector.as_slice());
        self.hidden.compute_output(self.input.output());
        self.output.compute_output(self.hidden.output());
        debug_assert_eq!(self.output.size(), 1);
        self.output.output()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const FEATURES: FeatureVector = FeatureVector {
        average: [-1.00131, -0.627069, -7.81097, 7.86285, -2.87145, 3.32365],
        higher_bands_cepstrum: [
            -0.653161, 0.529839, -0.425307, 0.25583, 0.235094, 0.230527, -0.144687, 0.182785,
            0.57102, 0.125039, 0.479482, -0.0255439, -0.0073141, -0.147346, -0.217106, -0.0846906,
        ],
        first_derivative: [-8.34943, 3.09065, 1.42628, -0.85235, -0.220207, -0.811163],
        second_derivative: [2.09032, -2.01425, -0.690268, -0.925327, -0.541354, 0.58455],
        bands_cross_correlation: [
            -0.606726, -0.0372358, 0.565991, 0.435854, 0.420812, 0.162198,
        ],
        pitch_period: -2.13,
        spectral_variability: 10.0089,
    };

    fn warm_up_rnn_vad(rnn_vad: &mut RnnVad) {
        for _ in 0..10 {
            rnn_vad.compute_vad_probability(&FEATURES, false);
        }
    }

    #[test]
    fn check_zero_probability_with_silence() {
        let backend = webrtc_simd::detect_backend();
        let mut rnn_vad = RnnVad::new(backend);
        warm_up_rnn_vad(&mut rnn_vad);
        assert_eq!(rnn_vad.compute_vad_probability(&FEATURES, true), 0.0);
    }

    #[test]
    fn check_rnn_vad_reset() {
        let backend = webrtc_simd::detect_backend();
        let mut rnn_vad = RnnVad::new(backend);
        warm_up_rnn_vad(&mut rnn_vad);
        let pre = rnn_vad.compute_vad_probability(&FEATURES, false);
        rnn_vad.reset();
        warm_up_rnn_vad(&mut rnn_vad);
        let post = rnn_vad.compute_vad_probability(&FEATURES, false);
        assert_eq!(pre, post);
    }

    #[test]
    fn check_rnn_vad_silence() {
        let backend = webrtc_simd::detect_backend();
        let mut rnn_vad = RnnVad::new(backend);
        warm_up_rnn_vad(&mut rnn_vad);
        let pre = rnn_vad.compute_vad_probability(&FEATURES, false);
        rnn_vad.compute_vad_probability(&FEATURES, true);
        warm_up_rnn_vad(&mut rnn_vad);
        let post = rnn_vad.compute_vad_probability(&FEATURES, false);
        assert_eq!(pre, post);
    }
}
