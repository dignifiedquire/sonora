//! Spectral noise estimation combining quantile tracking with parametric modeling.
//!
//! During the startup phase, a parametric model (white + pink noise) supplements
//! the quantile noise estimate. After startup, the estimate is refined using
//! speech probability to avoid tracking speech energy as noise.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/noise_estimator.cc`

use crate::config::{FFT_SIZE_BY_2_PLUS_1, SHORT_STARTUP_PHASE_BLOCKS};
use crate::fast_math::{exp_approximation, log_approximation, pow_approximation};
use crate::quantile_noise_estimator::QuantileNoiseEstimator;
use crate::suppression_params::SuppressionParams;

use core::f32::consts::LN_10;

// Log(i) lookup table for i in 0..129.
// clang-format off
#[allow(clippy::excessive_precision, reason = "match C++ source table exactly")]
const LOG_TABLE: [f32; 129] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.609438, 1.791759, 1.945910, 2.079442, 2.197225, LN_10, 2.397895,
    2.484907, 2.564949, 2.639057, 2.708050, 2.772589, 2.833213, 2.890372, 2.944439, 2.995732,
    3.044522, 3.091043, 3.135494, 3.178054, 3.218876, 3.258097, 3.295837, 3.332205, 3.367296,
    3.401197, 3.433987, 3.465736, 3.496507, 3.526361, 3.555348, 3.583519, 3.610918, 3.637586,
    3.663562, 3.688879, 3.713572, 3.737669, 3.761200, 3.784190, 3.806663, 3.828641, 3.850147,
    3.871201, 3.891820, 3.912023, 3.931826, 3.951244, 3.970292, 3.988984, 4.007333, 4.025352,
    4.043051, 4.060443, 4.077538, 4.094345, 4.110874, 4.127134, 4.143135, 4.158883, 4.174387,
    4.189655, 4.204693, 4.219508, 4.234107, 4.248495, 4.262680, 4.276666, 4.290460, 4.304065,
    4.317488, 4.330733, 4.343805, 4.356709, 4.369448, 4.382027, 4.394449, 4.406719, 4.418841,
    4.430817, 4.442651, 4.454347, 4.465908, 4.477337, 4.488636, 4.499810, 4.510859, 4.521789,
    4.532599, 4.543295, 4.553877, 4.564348, 4.574711, 4.584968, 4.595119, 4.605170, 4.615121,
    4.624973, 4.634729, 4.644391, 4.653960, 4.663439, 4.672829, 4.682131, 4.691348, 4.700480,
    4.709530, 4.718499, 4.727388, 4.736198, 4.744932, 4.753591, 4.762174, 4.770685, 4.779124,
    4.787492, 4.795791, 4.804021, 4.812184, 4.820282, 4.828314, 4.836282, 4.844187, 4.852030,
];

/// Spectral noise estimator.
///
/// Combines a quantile-based noise tracker with parametric white/pink noise
/// modeling during the startup phase. After startup, uses speech probability
/// to separate noise from speech in the spectrum update.
#[derive(Debug)]
pub struct NoiseEstimator {
    suppression_params: &'static SuppressionParams,
    white_noise_level: f32,
    pink_noise_numerator: f32,
    pink_noise_exp: f32,
    prev_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    conservative_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    parametric_noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    noise_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    quantile_noise_estimator: QuantileNoiseEstimator,
}

impl NoiseEstimator {
    pub fn new(suppression_params: &'static SuppressionParams) -> Self {
        Self {
            suppression_params,
            white_noise_level: 0.0,
            pink_noise_numerator: 0.0,
            pink_noise_exp: 0.0,
            prev_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            conservative_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            parametric_noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            noise_spectrum: [0.0; FFT_SIZE_BY_2_PLUS_1],
            quantile_noise_estimator: QuantileNoiseEstimator::new(),
        }
    }

    /// Prepare the estimator for a new frame analysis.
    ///
    /// Copies the current noise spectrum to `prev_noise_spectrum`.
    pub fn prepare_analysis(&mut self) {
        self.prev_noise_spectrum = self.noise_spectrum;
    }

    /// First step of the estimator update.
    ///
    /// Runs the quantile noise estimator and, during startup, blends the
    /// result with a parametric (white + pink) noise model.
    pub fn pre_update(
        &mut self,
        num_analyzed_frames: i32,
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
    ) {
        self.quantile_noise_estimator
            .estimate(signal_spectrum, &mut self.noise_spectrum);

        if num_analyzed_frames < SHORT_STARTUP_PHASE_BLOCKS {
            // Compute simplified noise model during startup.
            const START_BAND: usize = 5;
            let mut sum_log_i_log_magn = 0.0f32;
            let mut sum_log_i = 0.0f32;
            let mut sum_log_i_square = 0.0f32;
            let mut sum_log_magn = 0.0f32;

            for i in START_BAND..FFT_SIZE_BY_2_PLUS_1 {
                let log_i = LOG_TABLE[i];
                sum_log_i += log_i;
                sum_log_i_square += log_i * log_i;
                let log_signal = log_approximation(signal_spectrum[i]);
                sum_log_magn += log_signal;
                sum_log_i_log_magn += log_i * log_signal;
            }

            // Estimate the parameter for the level of the white noise.
            const ONE_BY_FFT_SIZE_BY_2_PLUS_1: f32 = 1.0 / FFT_SIZE_BY_2_PLUS_1 as f32;
            self.white_noise_level += signal_spectral_sum
                * ONE_BY_FFT_SIZE_BY_2_PLUS_1
                * self.suppression_params.over_subtraction_factor;

            // Estimate pink noise parameters.
            let denom = sum_log_i_square * (FFT_SIZE_BY_2_PLUS_1 - START_BAND) as f32
                - sum_log_i * sum_log_i;
            let num = sum_log_i_square * sum_log_magn - sum_log_i * sum_log_i_log_magn;
            debug_assert!(denom != 0.0);
            let mut pink_noise_adjustment = num / denom;

            // Constrain the estimated spectrum to be positive.
            pink_noise_adjustment = pink_noise_adjustment.max(0.0);
            self.pink_noise_numerator += pink_noise_adjustment;

            let num = sum_log_i * sum_log_magn
                - (FFT_SIZE_BY_2_PLUS_1 - START_BAND) as f32 * sum_log_i_log_magn;
            debug_assert!(denom != 0.0);
            pink_noise_adjustment = num / denom;

            // Constrain the pink noise power to be in the interval [0, 1].
            pink_noise_adjustment = pink_noise_adjustment.clamp(0.0, 1.0);

            self.pink_noise_exp += pink_noise_adjustment;

            let one_by_num_analyzed_frames_plus_1 = 1.0 / (num_analyzed_frames as f32 + 1.0);

            // Calculate the frequency-independent parts of parametric noise estimate.
            let mut parametric_exp = 0.0f32;
            let mut parametric_num = 0.0f32;
            if self.pink_noise_exp > 0.0 {
                // Use pink noise estimate.
                parametric_num = exp_approximation(
                    self.pink_noise_numerator * one_by_num_analyzed_frames_plus_1,
                );
                parametric_num *= num_analyzed_frames as f32 + 1.0;
                parametric_exp = self.pink_noise_exp * one_by_num_analyzed_frames_plus_1;
            }

            const ONE_BY_SHORT_STARTUP_PHASE_BLOCKS: f32 = 1.0 / SHORT_STARTUP_PHASE_BLOCKS as f32;
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                // Estimate the background noise using the white and pink noise
                // parameters.
                if self.pink_noise_exp == 0.0 {
                    // Use white noise estimate.
                    self.parametric_noise_spectrum[i] = self.white_noise_level;
                } else {
                    // Use pink noise estimate.
                    let use_band = if i < START_BAND { START_BAND } else { i };
                    let parametric_denom = pow_approximation(use_band as f32, parametric_exp);
                    debug_assert!(parametric_denom != 0.0);
                    self.parametric_noise_spectrum[i] = parametric_num / parametric_denom;
                }
            }

            // Weight quantile noise with modeled noise.
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                self.noise_spectrum[i] *= num_analyzed_frames as f32;
                let tmp = self.parametric_noise_spectrum[i]
                    * (SHORT_STARTUP_PHASE_BLOCKS - num_analyzed_frames) as f32;
                self.noise_spectrum[i] += tmp * one_by_num_analyzed_frames_plus_1;
                self.noise_spectrum[i] *= ONE_BY_SHORT_STARTUP_PHASE_BLOCKS;
            }
        }
    }

    /// Second step of the estimator update.
    ///
    /// Refines the noise spectrum using speech probability — bins with high
    /// speech probability are updated conservatively, while noise-dominated
    /// bins are tracked more aggressively.
    pub fn post_update(
        &mut self,
        speech_probability: &[f32],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        // Time-avg parameter for noise_spectrum update.
        const NOISE_UPDATE: f32 = 0.9;

        let mut gamma = NOISE_UPDATE;
        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            let prob_speech = speech_probability[i];
            let prob_non_speech = 1.0 - prob_speech;

            // Temporary noise update used for speech frames if update value is
            // less than previous.
            let noise_update_tmp = gamma * self.prev_noise_spectrum[i]
                + (1.0 - gamma)
                    * (prob_non_speech * signal_spectrum[i]
                        + prob_speech * self.prev_noise_spectrum[i]);

            // Time-constant based on speech/noise_spectrum state.
            let gamma_old = gamma;

            // Increase gamma for frame likely to be speech.
            const PROB_RANGE: f32 = 0.2;
            gamma = if prob_speech > PROB_RANGE {
                0.99
            } else {
                NOISE_UPDATE
            };

            // Conservative noise_spectrum update.
            if prob_speech < PROB_RANGE {
                self.conservative_noise_spectrum[i] +=
                    0.05 * (signal_spectrum[i] - self.conservative_noise_spectrum[i]);
            }

            // Noise_spectrum update.
            if gamma == gamma_old {
                self.noise_spectrum[i] = noise_update_tmp;
            } else {
                self.noise_spectrum[i] = gamma * self.prev_noise_spectrum[i]
                    + (1.0 - gamma)
                        * (prob_non_speech * signal_spectrum[i]
                            + prob_speech * self.prev_noise_spectrum[i]);
                // Allow for noise_spectrum update downwards: If noise_spectrum
                // update decreases the noise_spectrum, it is safe, so allow it.
                self.noise_spectrum[i] = self.noise_spectrum[i].min(noise_update_tmp);
            }
        }
    }

    /// Returns the current noise spectral estimate.
    pub fn noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.noise_spectrum
    }

    /// Returns the noise spectrum from the previous frame.
    pub fn prev_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.prev_noise_spectrum
    }

    /// Returns a noise spectral estimate based on white and pink noise parameters.
    pub fn parametric_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.parametric_noise_spectrum
    }

    /// Returns the conservative noise spectral estimate.
    pub fn conservative_noise_spectrum(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.conservative_noise_spectrum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SuppressionLevel;

    fn default_estimator() -> NoiseEstimator {
        NoiseEstimator::new(SuppressionParams::for_level(SuppressionLevel::K12dB))
    }

    #[test]
    fn initial_state_is_zeroed() {
        let est = default_estimator();
        assert_eq!(*est.noise_spectrum(), [0.0; FFT_SIZE_BY_2_PLUS_1]);
        assert_eq!(*est.prev_noise_spectrum(), [0.0; FFT_SIZE_BY_2_PLUS_1]);
        assert_eq!(
            *est.parametric_noise_spectrum(),
            [0.0; FFT_SIZE_BY_2_PLUS_1]
        );
        assert_eq!(
            *est.conservative_noise_spectrum(),
            [0.0; FFT_SIZE_BY_2_PLUS_1]
        );
    }

    #[test]
    fn prepare_analysis_copies_noise_to_prev() {
        let mut est = default_estimator();
        let signal = [5.0f32; FFT_SIZE_BY_2_PLUS_1];
        est.pre_update(0, &signal, signal.iter().sum());

        // After pre_update, noise_spectrum should be nonzero.
        assert!(est.noise_spectrum().iter().any(|&x| x != 0.0));

        est.prepare_analysis();
        assert_eq!(est.prev_noise_spectrum(), est.noise_spectrum());
    }

    #[test]
    fn pre_update_during_startup_blends_parametric() {
        let mut est = default_estimator();
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // During startup (frame 0), parametric model should contribute.
        est.pre_update(0, &signal, sum);
        let noise_0 = *est.noise_spectrum();

        // Run a few more startup frames.
        for frame in 1..10 {
            est.prepare_analysis();
            est.pre_update(frame, &signal, sum);
        }
        let noise_9 = *est.noise_spectrum();

        // Both should be nonzero.
        assert!(noise_0.iter().any(|&x| x > 0.0));
        assert!(noise_9.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn pre_update_after_startup_uses_quantile_only() {
        let mut est = default_estimator();
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // Run through full startup phase.
        for frame in 0..SHORT_STARTUP_PHASE_BLOCKS {
            est.prepare_analysis();
            est.pre_update(frame, &signal, sum);
        }

        // After startup, parametric_noise_spectrum should still be from last startup frame
        // (it's not updated post-startup).
        let parametric = *est.parametric_noise_spectrum();
        est.prepare_analysis();
        est.pre_update(SHORT_STARTUP_PHASE_BLOCKS, &signal, sum);
        assert_eq!(*est.parametric_noise_spectrum(), parametric);
    }

    #[test]
    fn post_update_with_no_speech_tracks_signal() {
        let mut est = default_estimator();
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // Run some frames to establish noise estimate.
        for frame in 0..60 {
            est.prepare_analysis();
            est.pre_update(frame, &signal, sum);
        }

        // Post-update with zero speech probability should track the signal.
        let speech_prob = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        est.prepare_analysis();
        est.pre_update(60, &signal, sum);
        est.post_update(&speech_prob, &signal);

        // Conservative estimate should also be updated.
        assert!(est.conservative_noise_spectrum().iter().any(|&x| x > 0.0));
    }

    #[test]
    fn post_update_with_full_speech_preserves_previous() {
        let mut est = default_estimator();
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // Run some frames.
        for frame in 0..60 {
            est.prepare_analysis();
            est.pre_update(frame, &signal, sum);
            let speech_prob = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
            est.post_update(&speech_prob, &signal);
        }

        // Now feed a loud signal with high speech probability.
        let loud_signal = [1000.0f32; FFT_SIZE_BY_2_PLUS_1];
        let loud_sum: f32 = loud_signal.iter().sum();
        est.prepare_analysis();
        let prev = *est.prev_noise_spectrum();
        est.pre_update(60, &loud_signal, loud_sum);
        let speech_prob = [0.9f32; FFT_SIZE_BY_2_PLUS_1];
        est.post_update(&speech_prob, &loud_signal);

        // With high speech probability, noise should stay close to previous.
        for (i, (&noise, &p)) in est.noise_spectrum().iter().zip(prev.iter()).enumerate() {
            let diff = (noise - p).abs();
            let scale = p.max(1.0);
            assert!(
                diff / scale < 0.5,
                "bin {i}: noise {noise} drifted too far from prev {p} during speech",
            );
        }
    }

    #[test]
    fn log_table_values() {
        // Spot-check a few known values.
        assert_eq!(LOG_TABLE[0], 0.0);
        assert_eq!(LOG_TABLE[10], LN_10);
        // ln(5) ≈ 1.609438
        assert!((LOG_TABLE[5] - 5.0_f32.ln()).abs() < 0.001);
        // ln(128) ≈ 4.852030
        assert!((LOG_TABLE[128] - 128.0_f32.ln()).abs() < 0.001);
    }
}
