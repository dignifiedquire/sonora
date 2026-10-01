//! Cascaded biquad (IIR) filter — direct form 1.
//!
//! Ported from `modules/audio_processing/utility/cascaded_biquad_filter.h/cc`.

use sonora_simd::NATIVE_FMA;

/// Coefficients for a single second-order (biquad) IIR section.
///
/// Transfer function: `H(z) = (b[0] + b[1]*z^-1 + b[2]*z^-2) / (1 + a[0]*z^-1 + a[1]*z^-2)`
#[derive(Debug, Clone, Copy)]
pub struct BiQuadCoefficients {
    /// Feedforward (numerator) coefficients `[b0, b1, b2]`.
    pub b: [f32; 3],
    /// Feedback (denominator) coefficients `[a1, a2]` (the leading 1 is implicit).
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
#[derive(Debug)]
pub struct CascadedBiQuadFilter {
    biquads: Vec<BiQuad>,
}

impl CascadedBiQuadFilter {
    /// Creates a new cascaded filter from the given second-order sections.
    pub fn new(coefficients: &[BiQuadCoefficients]) -> Self {
        Self {
            biquads: coefficients.iter().map(|c| BiQuad::new(*c)).collect(),
        }
    }

    /// Filters `x` into `y` (separate input/output).
    pub fn process(&mut self, x: &[f32], y: &mut [f32]) {
        self.process_with::<NATIVE_FMA>(x, y);
    }

    /// [`Self::process`], with the recursion fused by [`f32::mul_add`] if
    /// `FMA`, else in the C++ expression and its operation order. The public
    /// methods pass [`NATIVE_FMA`]; the tests run both forms. Always inlined,
    /// so that LLVM optimizes this body as part of `process`.
    #[inline(always)]
    fn process_with<const FMA: bool>(&mut self, x: &[f32], y: &mut [f32]) {
        if self.biquads.is_empty() {
            y.copy_from_slice(x);
            return;
        }
        Self::apply_biquad::<FMA>(x, y, &mut self.biquads[0]);
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
                *v = if FMA {
                    c_b_0.mul_add(
                        tmp,
                        c_b_1.mul_add(
                            m_x_0,
                            c_b_2.mul_add(m_x_1, (-c_a_0).mul_add(m_y_0, -c_a_1 * m_y_1)),
                        ),
                    )
                } else {
                    c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1
                };
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
    // rustc makes small functions that call nothing but intrinsics available
    // for inlining in other crates; this wrapper calls
    // `process_in_place_with`, so it needs `#[inline]` for that.
    #[inline]
    pub fn process_in_place(&mut self, y: &mut [f32]) {
        self.process_in_place_with::<NATIVE_FMA>(y);
    }

    /// [`Self::process_in_place`], in the form `FMA` selects; see
    /// [`Self::process_with`]. Not forced inline, unlike `process_with`: other
    /// crates inline `process_in_place`, and forcing this body into it would
    /// change how they inline the filter.
    fn process_in_place_with<const FMA: bool>(&mut self, y: &mut [f32]) {
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
                *v = if FMA {
                    c_b_0.mul_add(
                        tmp,
                        c_b_1.mul_add(
                            m_x_0,
                            c_b_2.mul_add(m_x_1, (-c_a_0).mul_add(m_y_0, -c_a_1 * m_y_1)),
                        ),
                    )
                } else {
                    c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1
                };
                m_x_1 = m_x_0;
                m_x_0 = tmp;
                m_y_1 = m_y_0;
                m_y_0 = *v;
            }
            bq.x = [m_x_0, m_x_1];
            bq.y = [m_y_0, m_y_1];
        }
    }

    /// Resets all filter states to zero.
    pub fn reset(&mut self) {
        for bq in &mut self.biquads {
            bq.reset();
        }
    }

    fn apply_biquad<const FMA: bool>(x: &[f32], y: &mut [f32], bq: &mut BiQuad) {
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
            *yi = if FMA {
                c_b_0.mul_add(
                    tmp,
                    c_b_1.mul_add(
                        m_x_0,
                        c_b_2.mul_add(m_x_1, (-c_a_0).mul_add(m_y_0, -c_a_1 * m_y_1)),
                    ),
                )
            } else {
                c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1
            };
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

    /// Reference cascade: the fused `mul_add` chain, or the C++ expression
    /// `c_b_0 * tmp + c_b_1 * m_x_0 + c_b_2 * m_x_1 - c_a_0 * m_y_0 - c_a_1 * m_y_1`.
    fn reference_cascade(coeffs: &[BiQuadCoefficients], x: &[f32], fused: bool) -> Vec<f32> {
        let mut y = x.to_vec();
        for c in coeffs {
            let (mut x0, mut x1, mut y0, mut y1) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
            for v in &mut y {
                let tmp = *v;
                *v = if fused {
                    c.b[0].mul_add(
                        tmp,
                        c.b[1].mul_add(x0, c.b[2].mul_add(x1, (-c.a[0]).mul_add(y0, -c.a[1] * y1))),
                    )
                } else {
                    c.b[0] * tmp + c.b[1] * x0 + c.b[2] * x1 - c.a[0] * y0 - c.a[1] * y1
                };
                x1 = x0;
                x0 = tmp;
                y1 = y0;
                y0 = *v;
            }
        }
        y
    }

    /// A high-pass section, then the low-pass section above. Two stages
    /// cover all three loops: `apply_biquad` and the in-place stage loop
    /// inside `process_with`, and the loop in `process_in_place_with`.
    fn two_stages() -> [BiQuadCoefficients; 2] {
        [
            BiQuadCoefficients {
                b: [0.972_613, -1.945_226, 0.972_613],
                a: [-1.944_48, 0.945_976],
            },
            lowpass_coefficients(),
        ]
    }

    /// Input on which the fused and the C++ forms of [`two_stages`] differ.
    fn test_signal() -> Vec<f32> {
        (0..32)
            .map(|i| (i as f32 * 0.618_034).fract() - 0.5)
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Output bits of `process_with::<FMA>` and `process_in_place_with::<FMA>`,
    /// each on a new filter.
    fn run_forms<const FMA: bool>(coeffs: &[BiQuadCoefficients], x: &[f32]) -> [Vec<u32>; 2] {
        let mut y = vec![0.0_f32; x.len()];
        CascadedBiQuadFilter::new(coeffs).process_with::<FMA>(x, &mut y);
        let mut in_place = x.to_vec();
        CascadedBiQuadFilter::new(coeffs).process_in_place_with::<FMA>(&mut in_place);
        [bits(&y), bits(&in_place)]
    }

    /// Both forms of all three loops, on every target: the fused form must
    /// match the `mul_add` chain, and the plain form the C++ expression, bit
    /// for bit.
    #[test]
    fn recursion_matches_fused_and_plain_references() {
        let coeffs = two_stages();
        let input = test_signal();
        let fused = bits(&reference_cascade(&coeffs, &input, true));
        let plain = bits(&reference_cascade(&coeffs, &input, false));
        assert_ne!(fused, plain);

        assert_eq!(run_forms::<true>(&coeffs, &input), [fused.clone(), fused]);
        assert_eq!(run_forms::<false>(&coeffs, &input), [plain.clone(), plain]);
    }

    /// `process` and `process_in_place` run the [`NATIVE_FMA`] form.
    #[test]
    fn public_methods_use_native_fma() {
        let coeffs = two_stages();
        let input = test_signal();
        // The forms differ on this input, so the comparison below can fail.
        assert_ne!(
            run_forms::<true>(&coeffs, &input),
            run_forms::<false>(&coeffs, &input)
        );

        let mut y = vec![0.0_f32; input.len()];
        CascadedBiQuadFilter::new(&coeffs).process(&input, &mut y);
        let mut in_place = input.clone();
        CascadedBiQuadFilter::new(&coeffs).process_in_place(&mut in_place);
        assert_eq!(
            [bits(&y), bits(&in_place)],
            run_forms::<NATIVE_FMA>(&coeffs, &input)
        );
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
