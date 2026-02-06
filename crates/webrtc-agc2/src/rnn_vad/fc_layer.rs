//! Fully-connected neural network layer.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/rnn_fc.cc`.

use super::activations::{sigmoid_approximated, tansig_approximated};
use super::vector_math::VectorMath;
use super::weights::WEIGHTS_SCALE;

/// Activation function for a neural network cell.
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    TansigApproximated,
    SigmoidApproximated,
}

/// Maximum number of units for an FC layer.
pub const FC_LAYER_MAX_UNITS: usize = 24;

/// Fully-connected layer with a custom activation function.
#[derive(Debug)]
pub struct FullyConnectedLayer {
    input_size: usize,
    output_size: usize,
    bias: Vec<f32>,
    weights: Vec<f32>,
    vector_math: VectorMath,
    activation: ActivationFunction,
    output: [f32; FC_LAYER_MAX_UNITS],
}

impl FullyConnectedLayer {
    /// Creates a new fully-connected layer.
    ///
    /// `bias` and `weights` are i8 quantized values that get scaled by
    /// `WEIGHTS_SCALE` and transposed during construction.
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: &[i8],
        weights: &[i8],
        activation: ActivationFunction,
        vector_math: VectorMath,
    ) -> Self {
        debug_assert!(output_size <= FC_LAYER_MAX_UNITS);
        debug_assert_eq!(bias.len(), output_size);
        debug_assert_eq!(weights.len(), input_size * output_size);

        let scaled_bias = scale_params(bias);
        let preprocessed_weights = preprocess_weights(weights, input_size, output_size);

        Self {
            input_size,
            output_size,
            bias: scaled_bias,
            weights: preprocessed_weights,
            vector_math,
            activation,
            output: [0.0; FC_LAYER_MAX_UNITS],
        }
    }

    /// Returns the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the output as a slice.
    pub fn output(&self) -> &[f32] {
        &self.output[..self.output_size]
    }

    /// Returns the output size.
    pub fn size(&self) -> usize {
        self.output_size
    }

    /// Computes the fully-connected layer output.
    pub fn compute_output(&mut self, input: &[f32]) {
        debug_assert_eq!(input.len(), self.input_size);

        let activation_fn: fn(f32) -> f32 = match self.activation {
            ActivationFunction::TansigApproximated => tansig_approximated,
            ActivationFunction::SigmoidApproximated => sigmoid_approximated,
        };

        for o in 0..self.output_size {
            let w_start = o * self.input_size;
            let w_end = w_start + self.input_size;
            self.output[o] = activation_fn(
                self.bias[o]
                    + self
                        .vector_math
                        .dot_product(input, &self.weights[w_start..w_end]),
            );
        }
    }
}

/// Scales i8 parameters to f32.
fn scale_params(params: &[i8]) -> Vec<f32> {
    params.iter().map(|&x| WEIGHTS_SCALE * x as f32).collect()
}

/// Transposes and scales weight matrix from i8 to f32.
///
/// C++ stores weights in column-major order (input_size rows x output_size cols).
/// We transpose to row-major (output_size rows x input_size cols) so that each
/// output neuron's weights are contiguous for efficient dot product.
fn preprocess_weights(weights: &[i8], input_size: usize, output_size: usize) -> Vec<f32> {
    if output_size == 1 {
        return scale_params(weights);
    }
    let mut w = vec![0.0_f32; weights.len()];
    for o in 0..output_size {
        for i in 0..input_size {
            w[o * input_size + i] = WEIGHTS_SCALE * weights[i * output_size + o] as f32;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn_vad::weights;
    use webrtc_simd::detect_backend;

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const FC_INPUT: [f32; 42] = [
        -1.00131, -0.627069, -7.81097, 7.86285, -2.87145, 3.32365, -0.653161, 0.529839, -0.425307,
        0.25583, 0.235094, 0.230527, -0.144687, 0.182785, 0.57102, 0.125039, 0.479482, -0.0255439,
        -0.0073141, -0.147346, -0.217106, -0.0846906, -8.34943, 3.09065, 1.42628, -0.85235,
        -0.220207, -0.811163, 2.09032, -2.01425, -0.690268, -0.925327, -0.541354, 0.58455,
        -0.606726, -0.0372358, 0.565991, 0.435854, 0.420812, 0.162198, -2.13, 10.0089,
    ];

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const FC_EXPECTED_OUTPUT: [f32; 24] = [
        -0.623293, -0.988299, 0.999378, 0.967168, 0.103087, -0.978545, -0.856347, 0.346675, 1.0,
        -0.717442, -0.544176, 0.960363, 0.983443, 0.999991, -0.824335, 0.984742, 0.990208,
        0.938179, 0.875092, 0.999846, 0.997707, -0.999382, 0.973153, -0.966605,
    ];

    #[test]
    fn fully_connected_layer_output() {
        let vector_math = VectorMath::new(detect_backend());
        let mut fc = FullyConnectedLayer::new(
            42,
            24,
            &weights::INPUT_DENSE_BIAS,
            &weights::INPUT_DENSE_WEIGHTS,
            ActivationFunction::TansigApproximated,
            vector_math,
        );
        fc.compute_output(&FC_INPUT);
        let output = fc.output();
        for (i, (&expected, &actual)) in FC_EXPECTED_OUTPUT.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-5,
                "output[{i}]: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn fully_connected_layer_scalar() {
        let vector_math = VectorMath::new(webrtc_simd::SimdBackend::Scalar);
        let mut fc = FullyConnectedLayer::new(
            42,
            24,
            &weights::INPUT_DENSE_BIAS,
            &weights::INPUT_DENSE_WEIGHTS,
            ActivationFunction::TansigApproximated,
            vector_math,
        );
        fc.compute_output(&FC_INPUT);
        let output = fc.output();
        for (i, (&expected, &actual)) in FC_EXPECTED_OUTPUT.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-5,
                "scalar output[{i}]: expected {expected}, got {actual}"
            );
        }
    }
}
