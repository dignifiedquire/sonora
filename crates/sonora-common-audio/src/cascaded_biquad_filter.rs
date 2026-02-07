//! Cascaded biquad (IIR) filter — direct form 1.
//!
//! Ported from `modules/audio_processing/utility/cascaded_biquad_filter.h/cc`.

/// Coefficients for a single second-order section.
#[derive(Debug, Clone, Copy)]
pub struct BiQuadCoefficients {
    pub b: [f32; 3],
    pub a: [f32; 2],
}

/// State for a single biquad section.
#[derive(Debug, Clone)]
struct BiQuad {
    coefficients: BiQuadCoefficients,
    x: [f32; 2],
    y: [f32; 2],
}

impl BiQuad {
    fn new(coefficients: BiQuadCoefficients) -> Self {
        Self {
            coefficients,
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn reset(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// Cascaded biquad filter applying multiple second-order sections in series.
pub struct CascadedBiQuadFilter {
    biquads: Vec<BiQuad>,
}

impl CascadedBiQuadFilter {
    pub fn new(coefficients: &[BiQuadCoefficients]) -> Self {
        Self {
            biquads: coefficients.iter().map(|c| BiQuad::new(*c)).collect(),
        }
    }

    /// Filters `x` into `y` (separate input/output).
    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        if self.biquads.is_empty() {
            y.copy_from_slice(x);
            return;
        }
        Self::apply_biquad(x, y, &mut self.biquads[0]);
        for k in 1..self.biquads.len() {
            // Split borrow: process y in-place through remaining stages.
            let (_, rest) = self.biquads.split_at_mut(k);
            let bq = &mut rest[0];
            // In-place: read from y, write to y.
            let c_b_0 = bq.coefficients.b[0];
            let c_b_1 = bq.coefficients.b[1];
            let c_b_2 = bq.coefficients.b[2];
            let c_a_0 = bq.coefficients.a[0];
            let c_a_1 = bq.coefficients.a[1];
            let mut m_x_0 = bq.x[0];
            let mut m_x_1 = bq.x[1];
            let mut m_y_0 = bq.y[0];
            let mut m_y_1 = bq.y[1];
            for v in y.iter_mut() {
                let tmp = *v;
                *v = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
                m_x_1 = m_x_0;
                m_x_0 = tmp;
                m_y_1 = m_y_0;
                m_y_0 = *v;
            }
            bq.x = [m_x_0, m_x_1];
            bq.y = [m_y_0, m_y_1];
        }
    }

    /// Filters `y` in-place through all stages.
    pub fn process_in_place(&mut self, y: &mut [f32]) {
        for bq in &mut self.biquads {
            let c_b_0 = bq.coefficients.b[0];
            let c_b_1 = bq.coefficients.b[1];
            let c_b_2 = bq.coefficients.b[2];
            let c_a_0 = bq.coefficients.a[0];
            let c_a_1 = bq.coefficients.a[1];
            let mut m_x_0 = bq.x[0];
            let mut m_x_1 = bq.x[1];
            let mut m_y_0 = bq.y[0];
            let mut m_y_1 = bq.y[1];
            for v in y.iter_mut() {
                let tmp = *v;
                *v = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
                m_x_1 = m_x_0;
                m_x_0 = tmp;
                m_y_1 = m_y_0;
                m_y_0 = *v;
            }
            bq.x = [m_x_0, m_x_1];
            bq.y = [m_y_0, m_y_1];
        }
    }

    pub fn reset(&mut self) {
        for bq in &mut self.biquads {
            bq.reset();
        }
    }

    fn apply_biquad(x: &[f32], y: &mut [f32], bq: &mut BiQuad) {
        debug_assert_eq!(x.len(), y.len());
        let c_b_0 = bq.coefficients.b[0];
        let c_b_1 = bq.coefficients.b[1];
        let c_b_2 = bq.coefficients.b[2];
        let c_a_0 = bq.coefficients.a[0];
        let c_a_1 = bq.coefficients.a[1];
        let mut m_x_0 = bq.x[0];
        let mut m_x_1 = bq.x[1];
        let mut m_y_0 = bq.y[0];
        let mut m_y_1 = bq.y[1];
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            let tmp = *xi;
            *yi = c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1;
            m_x_1 = m_x_0;
            m_x_0 = tmp;
            m_y_1 = m_y_0;
            m_y_0 = *yi;
        }
        bq.x = [m_x_0, m_x_1];
        bq.y = [m_y_0, m_y_1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lowpass_coefficients() -> BiQuadCoefficients {
        // Simple lowpass: b = [0.25, 0.5, 0.25], a = [0.1, 0.2]
        BiQuadCoefficients {
            b: [0.25, 0.5, 0.25],
            a: [0.1, 0.2],
        }
    }

    #[test]
    fn empty_filter_is_passthrough() {
        let mut filter = CascadedBiQuadFilter::new(&[]);
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        filter.process(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn single_stage_produces_output() {
        let mut filter = CascadedBiQuadFilter::new(&[lowpass_coefficients()]);
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0f32; 8];
        filter.process(&input, &mut output);
        // First output should be b[0] * 1.0 = 0.25
        assert!((output[0] - 0.25).abs() < 1e-6);
        // Subsequent outputs should be non-zero due to filter memory.
        assert!(output[1] != 0.0);
    }

    #[test]
    fn process_in_place_matches_process() {
        let coeffs = [lowpass_coefficients()];
        let mut filter1 = CascadedBiQuadFilter::new(&coeffs);
        let mut filter2 = CascadedBiQuadFilter::new(&coeffs);

        let input = [1.0, 0.5, -0.3, 0.7, -0.1, 0.4, 0.0, -0.5];
        let mut output1 = [0.0f32; 8];
        filter1.process(&input, &mut output1);

        let mut output2 = input;
        filter2.process_in_place(&mut output2);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let coeffs = [lowpass_coefficients()];
        let mut filter = CascadedBiQuadFilter::new(&coeffs);

        let input = [1.0, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        filter.process(&input, &mut output);

        filter.reset();

        let mut output2 = [0.0f32; 4];
        filter.process(&input, &mut output2);

        // After reset, output should be the same as the first time.
        for (a, b) in output.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn multi_stage_filter() {
        let coeffs = [lowpass_coefficients(), lowpass_coefficients()];
        let mut filter = CascadedBiQuadFilter::new(&coeffs);
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0f32; 8];
        filter.process(&input, &mut output);
        // Two stages should further smooth the impulse response.
        assert!(output[0].abs() < 0.25);
    }
}
