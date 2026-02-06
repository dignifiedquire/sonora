//! Speech probability estimation.
//!
//! Combines a prior signal model (from feature histograms) with per-bin
//! log-likelihood ratio (LRT) to produce per-frequency-bin speech
//! probability estimates.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/speech_probability_estimator.cc`

use crate::config::{FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS};
use crate::fast_math::exp_approximation_sign_flip;
use crate::signal_model_estimator::SignalModelEstimator;

/// Per-bin speech probability estimator.
#[derive(Debug)]
pub struct SpeechProbabilityEstimator {
    signal_model_estimator: SignalModelEstimator,
    prior_speech_prob: f32,
    speech_probability: [f32; FFT_SIZE_BY_2_PLUS_1],
}

impl Default for SpeechProbabilityEstimator {
    fn default() -> Self {
        Self {
            signal_model_estimator: SignalModelEstimator::default(),
            prior_speech_prob: 0.5,
            speech_probability: [0.0; FFT_SIZE_BY_2_PLUS_1],
        }
    }
}

impl SpeechProbabilityEstimator {
    /// Compute speech probability for the current frame.
    #[allow(clippy::too_many_arguments, reason = "matches C++ API signature")]
    pub fn update(
        &mut self,
        num_analyzed_frames: i32,
        prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
        signal_energy: f32,
    ) {
        // Update models.
        if num_analyzed_frames < LONG_STARTUP_PHASE_BLOCKS {
            self.signal_model_estimator
                .adjust_normalization(num_analyzed_frames, signal_energy);
        }
        self.signal_model_estimator.update(
            prior_snr,
            post_snr,
            conservative_noise_spectrum,
            signal_spectrum,
            signal_spectral_sum,
            signal_energy,
        );

        let model = self.signal_model_estimator.model();
        let prior_model = self.signal_model_estimator.prior_model();

        // Width parameter in sigmoid map for prior model.
        const WIDTH_PRIOR_0: f32 = 4.0;
        // Width for pause region: lower range, so increase width in tanh map.
        const WIDTH_PRIOR_1: f32 = 2.0 * WIDTH_PRIOR_0;

        // Average LRT feature: use larger width in tanh map for pause regions.
        let width_prior = if model.lrt < prior_model.lrt {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };

        // Compute indicator function: sigmoid map.
        let indicator0 = 0.5 * ((width_prior * (model.lrt - prior_model.lrt)).tanh() + 1.0);

        // Spectral flatness feature: use larger width in tanh map for pause regions.
        let width_prior = if model.spectral_flatness > prior_model.flatness_threshold {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };

        // Compute indicator function: sigmoid map.
        let indicator1 = 0.5
            * ((width_prior * (prior_model.flatness_threshold - model.spectral_flatness)).tanh()
                + 1.0);

        // For template spectrum-difference: use larger width in tanh map for
        // pause regions.
        let width_prior = if model.spectral_diff < prior_model.template_diff_threshold {
            WIDTH_PRIOR_1
        } else {
            WIDTH_PRIOR_0
        };

        // Compute indicator function: sigmoid map.
        let indicator2 = 0.5
            * ((width_prior * (model.spectral_diff - prior_model.template_diff_threshold)).tanh()
                + 1.0);

        // Combine the indicator function with the feature weights.
        let ind_prior = prior_model.lrt_weighting * indicator0
            + prior_model.flatness_weighting * indicator1
            + prior_model.difference_weighting * indicator2;

        // Compute the prior probability.
        self.prior_speech_prob += 0.1 * (ind_prior - self.prior_speech_prob);

        // Make sure probabilities are within range: keep floor to 0.01.
        self.prior_speech_prob = self.prior_speech_prob.clamp(0.01, 1.0);

        // Final speech probability: combine prior model with LR factor.
        let gain_prior = (1.0 - self.prior_speech_prob) / (self.prior_speech_prob + 0.0001);

        let mut inv_lrt = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        exp_approximation_sign_flip(&model.avg_log_lrt, &mut inv_lrt);
        for (sp, &il) in self.speech_probability.iter_mut().zip(inv_lrt.iter()) {
            *sp = 1.0 / (1.0 + gain_prior * il);
        }
    }

    /// Returns the prior speech probability.
    pub fn prior_probability(&self) -> f32 {
        self.prior_speech_prob
    }

    /// Returns the per-bin speech probability.
    pub fn probability(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.speech_probability
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let est = SpeechProbabilityEstimator::default();
        assert_eq!(est.prior_probability(), 0.5);
        assert_eq!(est.probability(), &[0.0; FFT_SIZE_BY_2_PLUS_1]);
    }

    #[test]
    fn update_produces_valid_probabilities() {
        let mut est = SpeechProbabilityEstimator::default();
        let prior_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let cons_noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        est.update(0, &prior_snr, &post_snr, &cons_noise, &signal, sum, sum);

        // All probabilities should be in [0, 1].
        for &p in est.probability() {
            assert!(
                (0.0..=1.0).contains(&p),
                "probability {p} out of range [0, 1]"
            );
        }
        // Prior probability should be in [0.01, 1].
        assert!(est.prior_probability() >= 0.01);
        assert!(est.prior_probability() <= 1.0);
    }

    #[test]
    fn high_snr_gives_high_speech_probability() {
        let mut est = SpeechProbabilityEstimator::default();
        let signal = [100.0f32; FFT_SIZE_BY_2_PLUS_1];
        let noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // High prior_snr and post_snr indicate strong speech.
        let prior_snr = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [10.0f32; FFT_SIZE_BY_2_PLUS_1];

        for frame in 0..100 {
            est.update(frame, &prior_snr, &post_snr, &noise, &signal, sum, sum);
        }

        // After many frames with high SNR, speech probability should be high.
        let avg_prob: f32 = est.probability().iter().sum::<f32>() / FFT_SIZE_BY_2_PLUS_1 as f32;
        assert!(
            avg_prob > 0.5,
            "avg speech prob {avg_prob} should be > 0.5 with high SNR"
        );
    }

    #[test]
    fn low_snr_gives_low_speech_probability() {
        let mut est = SpeechProbabilityEstimator::default();
        let signal = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // Very low SNR = noise-dominated.
        let prior_snr = [0.01f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [0.01f32; FFT_SIZE_BY_2_PLUS_1];

        for frame in 0..100 {
            est.update(frame, &prior_snr, &post_snr, &noise, &signal, sum, sum);
        }

        let avg_prob: f32 = est.probability().iter().sum::<f32>() / FFT_SIZE_BY_2_PLUS_1 as f32;
        assert!(
            avg_prob < 0.5,
            "avg speech prob {avg_prob} should be < 0.5 with low SNR"
        );
    }
}
