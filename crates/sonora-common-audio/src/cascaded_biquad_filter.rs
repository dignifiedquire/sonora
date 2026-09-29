//! Cascaded biquad (IIR) filter — direct form 1.
//!
//! Ported from `modules/audio_processing/utility/cascaded_biquad_filter.h/cc`.

/// Whether the filter recursion uses [`f32::mul_add`].
///
/// Only AArch64 (`aarch64` and `arm64ec`), and x86 built with the `fma`
/// target feature, take this path: there `mul_add` is one instruction.
/// Without native FMA it is an `fmaf` library call per operation. Every other
/// target, including targets with FMA such as riscv64gc, uses the plain C++
/// expression, in its operation order.
const USE_FMA: bool = cfg!(any(
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_feature = "fma"
));

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
                // Fused only on AArch64 and on x86 with `fma`; see `USE_FMA`.
                *v = if USE_FMA {
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
                *v = if USE_FMA {
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
            *yi = if USE_FMA {
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

    /// The `USE_FMA` policy, restated so that a change to it fails the test
    /// below: fuse on AArch64 (`aarch64` and `arm64ec`) and on x86 with the
    /// `fma` feature. Elsewhere `mul_add` can be an `fmaf` library call and
    /// C++ built without FP contraction does not fuse, so other targets must
    /// match the C++ expression bit for bit. On aarch64 the output must stay
    /// the fused output it has always been.
    const EXPECT_FUSED: bool = cfg!(any(
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_feature = "fma"
    ));

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

    /// Checks all three filter loops against [`EXPECT_FUSED`].
    #[test]
    fn recursion_matches_fma_policy() {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        // A high-pass section, then the low-pass section above. Two stages
        // cover all three loops: `apply_biquad` and the in-place stage loop
        // inside `process`, and the loop in `process_in_place`.
        let coeffs = [
            BiQuadCoefficients {
                b: [0.972_613, -1.945_226, 0.972_613],
                a: [-1.944_48, 0.945_976],
            },
            lowpass_coefficients(),
        ];
        let input: Vec<f32> = (0..32)
            .map(|i| (i as f32 * 0.618_034).fract() - 0.5)
            .collect();

        let fused = reference_cascade(&coeffs, &input, true);
        let plain = reference_cascade(&coeffs, &input, false);
        assert_ne!(bits(&fused), bits(&plain));
        let expected = bits(if EXPECT_FUSED { &fused } else { &plain });

        let mut output = vec![0.0_f32; input.len()];
        CascadedBiQuadFilter::new(&coeffs).process(&input, &mut output);
        assert_eq!(bits(&output), expected);

        let mut in_place = input.clone();
        CascadedBiQuadFilter::new(&coeffs).process_in_place(&mut in_place);
        assert_eq!(bits(&in_place), expected);
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
