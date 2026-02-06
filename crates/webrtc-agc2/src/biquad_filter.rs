//! IIR biquad filter.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/biquad_filter.h`.

// Used by adaptive_digital_gain_controller (Step 8).
#![allow(dead_code, reason = "consumed by later AGC2 modules")]

/// Biquad filter coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BiquadFilterConfig {
    /// Feedforward coefficients `[b0, b1, b2]`.
    pub b: [f32; 3],
    /// Feedback coefficients `[a1, a2]`.
    pub a: [f32; 2],
}

/// Second-order IIR (biquad) filter.
///
/// Transfer function: `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
#[derive(Debug)]
pub(crate) struct BiquadFilter {
    config: BiquadFilterConfig,
    /// State: `[x[n-1], x[n-2], y[n-1], y[n-2]]`.
    state: [f32; 4],
}

impl BiquadFilter {
    /// Creates a new biquad filter with the given configuration.
    pub(crate) fn new(config: BiquadFilterConfig) -> Self {
        Self {
            config,
            state: [0.0; 4],
        }
    }

    /// Resets the filter state to zero.
    pub(crate) fn reset(&mut self) {
        self.state = [0.0; 4];
    }

    /// Reconfigures the filter and resets state.
    pub(crate) fn set_config(&mut self, config: BiquadFilterConfig) {
        self.config = config;
        self.reset();
    }

    /// Processes input samples `x` and writes filtered output to `y`.
    ///
    /// `x` and `y` may be the same slice for in-place processing.
    pub(crate) fn process(&mut self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        for k in 0..x.len() {
            let tmp = x[k];
            y[k] = self.config.b[0] * tmp
                + self.config.b[1] * self.state[0]
                + self.config.b[2] * self.state[1]
                - self.config.a[0] * self.state[2]
                - self.config.a[1] * self.state[3];
            self.state[1] = self.state[0];
            self.state[0] = tmp;
            self.state[3] = self.state[2];
            self.state[2] = y[k];
        }
    }

    /// Processes samples in-place.
    pub(crate) fn process_in_place(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let tmp = *sample;
            *sample = self.config.b[0] * tmp
                + self.config.b[1] * self.state[0]
                + self.config.b[2] * self.state[1]
                - self.config.a[0] * self.state[2]
                - self.config.a[1] * self.state[3];
            self.state[1] = self.state[0];
            self.state[0] = tmp;
            self.state[3] = self.state[2];
            self.state[2] = *sample;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_SIZE: usize = 8;
    const NUM_FRAMES: usize = 4;

    const BIQUAD_INPUT_SEQ: [[f32; FRAME_SIZE]; NUM_FRAMES] = [
        [
            -87.166_29,
            -8.029_022,
            101.619_58,
            -0.294_296,
            -5.825_764,
            -8.890_625,
            10.310_432,
            54.845_333,
        ],
        [
            -64.647_644,
            -6.883_945,
            11.059_189,
            -95.242_54,
            -108.870_834,
            11.024_944,
            63.044_1,
            -52.709_583,
        ],
        [
            -32.350_53,
            -18.108_028,
            -74.022_34,
            -8.986_874,
            -1.525_581,
            103.705_51,
            6.346_226,
            -14.319_557,
        ],
        [
            22.645_832,
            -64.597_15,
            55.462_52,
            -109.393_19,
            10.117_825,
            -40.019_64,
            -98.612_23,
            -8.330_326,
        ],
    ];

    // Computed as `scipy.signal.butter(N=2, Wn=60/24000, btype='highpass')`.
    const BIQUAD_CONFIG: BiquadFilterConfig = BiquadFilterConfig {
        b: [0.994_461_8, -1.988_923_5, 0.994_461_8],
        a: [-1.988_892_9, 0.988_954_25],
    };

    // Comparing to scipy. The expected output is generated as follows:
    // zi = np.float32([0, 0])
    // for i in range(4):
    //   yn, zi = scipy.signal.lfilter(B, A, x[i], zi=zi)
    //   print(yn)
    const BIQUAD_OUTPUT_SEQ: [[f32; FRAME_SIZE]; NUM_FRAMES] = [
        [
            -86.683_55,
            -7.021_753_5,
            102.102_9,
            -0.374_873_33,
            -5.872_058_5,
            -8.855_216,
            10.337_726,
            54.511_57,
        ],
        [
            -64.925_316,
            -6.763_96,
            11.155_345,
            -94.680_73,
            -107.181_78,
            13.246_425,
            64.842_89,
            -50.978_226,
        ],
        [
            -30.157_965,
            -15.648_509,
            -71.066_63,
            -5.588_323,
            1.911_753_5,
            106.557_2,
            8.571_83,
            -12.062_985,
        ],
        [
            24.842_866,
            -62.180_94,
            57.914_88,
            -106.656_86,
            13.387_601,
            -36.603_67,
            -94.448_8,
            -3.599_203_5,
        ],
    ];

    /// Checks that the relative error between expected and computed values
    /// is within tolerance.
    fn expect_near_relative(expected: &[f32], computed: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), computed.len());
        for (i, (&exp, &comp)) in expected.iter().zip(computed.iter()).enumerate() {
            let abs_diff = (exp - comp).abs();
            if abs_diff == 0.0 {
                continue;
            }
            let den = if exp == 0.0 { 1.0 } else { exp.abs() };
            assert!(
                abs_diff / den <= tolerance,
                "index {i}: expected {exp}, computed {comp}, relative error {}",
                abs_diff / den,
            );
        }
    }

    #[test]
    fn filter_not_in_place() {
        let mut filter = BiquadFilter::new(BIQUAD_CONFIG);
        let mut samples = [0.0_f32; FRAME_SIZE];
        for (input, expected) in BIQUAD_INPUT_SEQ.iter().zip(&BIQUAD_OUTPUT_SEQ) {
            filter.process(input, &mut samples);
            expect_near_relative(expected, &samples, 2e-4);
        }
    }

    #[test]
    fn filter_in_place() {
        let mut filter = BiquadFilter::new(BIQUAD_CONFIG);
        let mut samples = [0.0_f32; FRAME_SIZE];
        for (input, expected) in BIQUAD_INPUT_SEQ.iter().zip(&BIQUAD_OUTPUT_SEQ) {
            samples.copy_from_slice(input);
            filter.process_in_place(&mut samples);
            expect_near_relative(expected, &samples, 2e-4);
        }
    }

    #[test]
    fn set_config_different_output() {
        let mut filter = BiquadFilter::new(BiquadFilterConfig {
            b: [0.978_030_5, -1.956_061, 0.978_030_5],
            a: [-1.955_578_2, 0.956_543_7],
        });
        let mut samples1 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples1);
        }

        filter.set_config(BiquadFilterConfig {
            b: [0.097_631_07, 0.195_262_15, 0.097_631_07],
            a: [-0.942_809_04, 0.333_333_33],
        });
        let mut samples2 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples2);
        }

        assert_ne!(samples1, samples2);
    }

    #[test]
    fn set_config_resets_state() {
        let mut filter = BiquadFilter::new(BIQUAD_CONFIG);
        let mut samples1 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples1);
        }

        filter.set_config(BIQUAD_CONFIG);
        let mut samples2 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples2);
        }

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn reset() {
        let mut filter = BiquadFilter::new(BIQUAD_CONFIG);
        let mut samples1 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples1);
        }

        filter.reset();
        let mut samples2 = [0.0_f32; FRAME_SIZE];
        for input in &BIQUAD_INPUT_SEQ {
            filter.process(input, &mut samples2);
        }

        assert_eq!(samples1, samples2);
    }
}
