//! Gated Recurrent Unit (GRU) neural network layer.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/rnn_gru.cc`.

use super::activations::sigmoid_approximated;
use super::vector_math::VectorMath;
use super::weights::WEIGHTS_SCALE;

/// Maximum number of units for a GRU layer.
pub const GRU_LAYER_MAX_UNITS: usize = 24;

/// Number of gates in a GRU (update, reset, output).
const NUM_GRU_GATES: usize = 3;

/// Recurrent layer with gated recurrent units (GRUs).
///
/// Uses sigmoid for update/reset gates and ReLU for the output gate.
#[derive(Debug)]
pub struct GatedRecurrentLayer {
    input_size: usize,
    output_size: usize,
    bias: Vec<f32>,
    weights: Vec<f32>,
    recurrent_weights: Vec<f32>,
    vector_math: VectorMath,
    state: [f32; GRU_LAYER_MAX_UNITS],
}

impl GatedRecurrentLayer {
    /// Creates a new GRU layer.
    ///
    /// `bias`, `weights`, and `recurrent_weights` are i8 quantized values
    /// that get scaled, transposed, and rearranged during construction.
    pub fn new(
        input_size: usize,
        output_size: usize,
        bias: &[i8],
        weights: &[i8],
        recurrent_weights: &[i8],
        vector_math: VectorMath,
    ) -> Self {
        debug_assert!(output_size <= GRU_LAYER_MAX_UNITS);
        debug_assert_eq!(bias.len(), NUM_GRU_GATES * output_size);
        debug_assert_eq!(weights.len(), NUM_GRU_GATES * input_size * output_size);
        debug_assert_eq!(
            recurrent_weights.len(),
            NUM_GRU_GATES * output_size * output_size
        );

        let preprocessed_bias = preprocess_gru_tensor(bias, output_size);
        let preprocessed_weights = preprocess_gru_tensor(weights, output_size);
        let preprocessed_recurrent = preprocess_gru_tensor(recurrent_weights, output_size);

        let mut layer = Self {
            input_size,
            output_size,
            bias: preprocessed_bias,
            weights: preprocessed_weights,
            recurrent_weights: preprocessed_recurrent,
            vector_math,
            state: [0.0; GRU_LAYER_MAX_UNITS],
        };
        layer.reset();
        layer
    }

    /// Returns the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the output (state) as a slice.
    pub fn output(&self) -> &[f32] {
        &self.state[..self.output_size]
    }

    /// Returns the output size.
    pub fn size(&self) -> usize {
        self.output_size
    }

    /// Resets the GRU state to zero.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Computes the recurrent layer output and updates the state.
    pub fn compute_output(&mut self, input: &[f32]) {
        debug_assert_eq!(input.len(), self.input_size);

        let stride_weights = self.input_size * self.output_size;
        let stride_recurrent = self.output_size * self.output_size;

        // Update gate: u = sigmoid(W_u * input + R_u * state + b_u)
        let mut update = [0.0_f32; GRU_LAYER_MAX_UNITS];
        self.compute_update_reset_gate(
            input,
            0, // gate offset
            stride_weights,
            stride_recurrent,
            &mut update,
        );

        // Reset gate: r = sigmoid(W_r * input + R_r * state + b_r)
        let mut reset = [0.0_f32; GRU_LAYER_MAX_UNITS];
        self.compute_update_reset_gate(
            input,
            1, // gate offset
            stride_weights,
            stride_recurrent,
            &mut reset,
        );

        // State gate: s' = u * s + (1-u) * relu(W_o * input + R_o * (s*r) + b_o)
        self.compute_state_gate(input, &update, &reset, stride_weights, stride_recurrent);
    }

    /// Computes update or reset gate: `g = sigmoid(W*input + R*state + b)`.
    fn compute_update_reset_gate(
        &self,
        input: &[f32],
        gate_index: usize,
        stride_weights: usize,
        stride_recurrent: usize,
        gate: &mut [f32; GRU_LAYER_MAX_UNITS],
    ) {
        let bias_offset = gate_index * self.output_size;
        let w_offset = gate_index * stride_weights;
        let r_offset = gate_index * stride_recurrent;
        let state = &self.state[..self.output_size];

        for (o, gate_val) in gate.iter_mut().enumerate().take(self.output_size) {
            let mut x = self.bias[bias_offset + o];
            x += self.vector_math.dot_product(
                input,
                &self.weights[w_offset + o * self.input_size..w_offset + (o + 1) * self.input_size],
            );
            x += self.vector_math.dot_product(
                state,
                &self.recurrent_weights
                    [r_offset + o * self.output_size..r_offset + (o + 1) * self.output_size],
            );
            *gate_val = sigmoid_approximated(x);
        }
    }

    /// Computes state gate: `s' = u * s + (1-u) * relu(W*input + R*(s*r) + b)`.
    fn compute_state_gate(
        &mut self,
        input: &[f32],
        update: &[f32; GRU_LAYER_MAX_UNITS],
        reset: &[f32; GRU_LAYER_MAX_UNITS],
        stride_weights: usize,
        stride_recurrent: usize,
    ) {
        let bias_offset = 2 * self.output_size;
        let w_offset = 2 * stride_weights;
        let r_offset = 2 * stride_recurrent;

        // Compute reset_x_state = state * reset
        let mut reset_x_state = [0.0_f32; GRU_LAYER_MAX_UNITS];
        for o in 0..self.output_size {
            reset_x_state[o] = self.state[o] * reset[o];
        }

        for (o, &u) in update.iter().enumerate().take(self.output_size) {
            let mut x = self.bias[bias_offset + o];
            x += self.vector_math.dot_product(
                input,
                &self.weights[w_offset + o * self.input_size..w_offset + (o + 1) * self.input_size],
            );
            x += self.vector_math.dot_product(
                &reset_x_state[..self.output_size],
                &self.recurrent_weights
                    [r_offset + o * self.output_size..r_offset + (o + 1) * self.output_size],
            );
            // ReLU activation + state update
            self.state[o] = u * self.state[o] + (1.0 - u) * x.max(0.0);
        }
    }
}

/// Preprocesses a GRU tensor: transposes, casts i8→f32, and scales.
///
/// The source tensor has layout `[n, NUM_GRU_GATES, output_size]` where `n`
/// is inferred from the tensor size. The output is rearranged to
/// `[NUM_GRU_GATES, output_size, n]` with each element scaled by `WEIGHTS_SCALE`.
fn preprocess_gru_tensor(tensor_src: &[i8], output_size: usize) -> Vec<f32> {
    let n = tensor_src.len() / (output_size * NUM_GRU_GATES);
    debug_assert_eq!(tensor_src.len(), n * output_size * NUM_GRU_GATES);

    let stride_src = NUM_GRU_GATES * output_size;
    let stride_dst = n * output_size;

    let mut tensor_dst = vec![0.0_f32; tensor_src.len()];
    for g in 0..NUM_GRU_GATES {
        for o in 0..output_size {
            for i in 0..n {
                tensor_dst[g * stride_dst + o * n + i] =
                    WEIGHTS_SCALE * tensor_src[i * stride_src + g * output_size + o] as f32;
            }
        }
    }
    tensor_dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use webrtc_simd::detect_backend;

    const GRU_INPUT_SIZE: usize = 5;
    const GRU_OUTPUT_SIZE: usize = 4;

    const GRU_BIAS: [i8; 12] = [96, -99, -81, -114, 49, 119, -118, 68, -76, 91, 121, 125];

    const GRU_WEIGHTS: [i8; 60] = [
        // Input 0.
        124, 9, 1, 116, // Update.
        -66, -21, -118, -110, // Reset.
        104, 75, -23, -51, // Output.
        // Input 1.
        -72, -111, 47, 93, // Update.
        77, -98, 41, -8, // Reset.
        40, -23, -43, -107, // Output.
        // Input 2.
        9, -73, 30, -32, // Update.
        -2, 64, -26, 91, // Reset.
        -48, -24, -28, -104, // Output.
        // Input 3.
        74, -46, 116, 15, // Update.
        32, 52, -126, -38, // Reset.
        -121, 12, -16, 110, // Output.
        // Input 4.
        -95, 66, -103, -35, // Update.
        -38, 3, -126, -61, // Reset.
        28, 98, -117, -43, // Output.
    ];

    const GRU_RECURRENT_WEIGHTS: [i8; 48] = [
        // Output 0.
        -3, 87, 50, 51, // Update.
        -22, 27, -39, 62, // Reset.
        31, -83, -52, -48, // Output.
        // Output 1.
        -6, 83, -19, 104, // Update.
        105, 48, 23, 68, // Reset.
        23, 40, 7, -120, // Output.
        // Output 2.
        64, -62, 117, 85, // Update.
        51, -43, 54, -105, // Reset.
        120, 56, -128, -107, // Output.
        // Output 3.
        39, 50, -17, -47, // Update.
        -117, 14, 108, 12, // Reset.
        -7, -72, 103, -87, // Output.
    ];

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const GRU_INPUT_SEQUENCE: [f32; 20] = [
        0.89395463, 0.93224651, 0.55788344, 0.32341808, 0.93355054, 0.13475326, 0.97370994,
        0.14253306, 0.93710381, 0.76093364, 0.65780413, 0.41657975, 0.49403164, 0.46843281,
        0.75138855, 0.24517593, 0.47657707, 0.57064998, 0.435184, 0.19319285,
    ];

    #[allow(clippy::excessive_precision, reason = "values from C++ test data")]
    const GRU_EXPECTED_OUTPUT_SEQUENCE: [f32; 16] = [
        0.0239123, 0.5773077, 0.0, 0.0, 0.01282811, 0.64330572, 0.0, 0.04863098, 0.00781069,
        0.75267816, 0.0, 0.02579715, 0.00471378, 0.59162533, 0.11087593, 0.01334511,
    ];

    fn test_gated_recurrent_layer(mut gru: GatedRecurrentLayer) {
        let input_sequence_length = GRU_INPUT_SEQUENCE.len() / gru.input_size();
        let output_sequence_length = GRU_EXPECTED_OUTPUT_SEQUENCE.len() / gru.size();
        assert_eq!(input_sequence_length, output_sequence_length);

        gru.reset();
        for i in 0..input_sequence_length {
            let input_start = i * gru.input_size();
            let input_end = input_start + gru.input_size();
            gru.compute_output(&GRU_INPUT_SEQUENCE[input_start..input_end]);

            let output_start = i * gru.size();
            let expected = &GRU_EXPECTED_OUTPUT_SEQUENCE[output_start..output_start + gru.size()];
            let actual = gru.output();
            for (j, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (exp - act).abs() < 3e-6,
                    "step {i}, output[{j}]: expected {exp}, got {act}"
                );
            }
        }
    }

    #[test]
    fn gated_recurrent_layer_output() {
        let vector_math = VectorMath::new(detect_backend());
        let gru = GatedRecurrentLayer::new(
            GRU_INPUT_SIZE,
            GRU_OUTPUT_SIZE,
            &GRU_BIAS,
            &GRU_WEIGHTS,
            &GRU_RECURRENT_WEIGHTS,
            vector_math,
        );
        test_gated_recurrent_layer(gru);
    }

    #[test]
    fn gated_recurrent_layer_scalar() {
        let vector_math = VectorMath::new(webrtc_simd::SimdBackend::Scalar);
        let gru = GatedRecurrentLayer::new(
            GRU_INPUT_SIZE,
            GRU_OUTPUT_SIZE,
            &GRU_BIAS,
            &GRU_WEIGHTS,
            &GRU_RECURRENT_WEIGHTS,
            vector_math,
        );
        test_gated_recurrent_layer(gru);
    }
}
