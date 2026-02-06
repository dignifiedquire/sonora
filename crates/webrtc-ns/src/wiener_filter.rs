//! Wiener filter for frequency-domain noise reduction.
//!
//! Estimates a per-bin gain filter from signal and noise spectra using
//! a directed-decision SNR estimator. During startup, the filter is
//! blended with a parametric estimate.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/wiener_filter.cc`

use crate::config::{FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS, SHORT_STARTUP_PHASE_BLOCKS};
use crate::fast_math::sqrt_fast_approximation;
use crate::suppression_params::SuppressionParams;

/// Wiener filter for noise suppression.
#[derive(Debug)]
pub struct WienerFilter {
    suppression_params: &'static SuppressionParams,
    spectrum_prev_process: [f32; FFT_SIZE_BY_2_PLUS_1],
    initial_spectral_estimate: [f32; FFT_SIZE_BY_2_PLUS_1],
    filter: [f32; FFT_SIZE_BY_2_PLUS_1],
}

impl WienerFilter {
    pub fn new(suppression_params: &'static SuppressionParams) -> Self {
        Self {
            suppression_params,
            spectrum_prev_process: [0.0; FFT_SIZE_BY_2_PLUS_1],
            initial_spectral_estimate: [0.0; FFT_SIZE_BY_2_PLUS_1],
            filter: [1.0; FFT_SIZE_BY_2_PLUS_1],
        }
    }

    /// Update the filter estimate from current signal and noise spectra.
    pub fn update(
        &mut self,
        num_analyzed_frames: i32,
        noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        prev_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        parametric_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        let over_sub = self.suppression_params.over_subtraction_factor;
        let min_gain = self.suppression_params.minimum_attenuating_gain;

        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            // Previous estimate based on previous frame with gain filter.
            let prev_tsa =
                self.spectrum_prev_process[i] / (prev_noise_spectrum[i] + 0.0001) * self.filter[i];

            // Current estimate.
            let current_tsa = if signal_spectrum[i] > noise_spectrum[i] {
                signal_spectrum[i] / (noise_spectrum[i] + 0.0001) - 1.0
            } else {
                0.0
            };

            // Directed decision estimate is sum of two terms: current estimate
            // and previous estimate.
            let snr_prior = 0.98 * prev_tsa + (1.0 - 0.98) * current_tsa;
            self.filter[i] = snr_prior / (over_sub + snr_prior);
            self.filter[i] = self.filter[i].clamp(min_gain, 1.0);
        }

        if num_analyzed_frames < SHORT_STARTUP_PHASE_BLOCKS {
            const ONE_BY_SHORT_STARTUP_PHASE_BLOCKS: f32 = 1.0 / SHORT_STARTUP_PHASE_BLOCKS as f32;
            for i in 0..FFT_SIZE_BY_2_PLUS_1 {
                self.initial_spectral_estimate[i] += signal_spectrum[i];
                let mut filter_initial =
                    self.initial_spectral_estimate[i] - over_sub * parametric_noise_spectrum[i];
                filter_initial /= self.initial_spectral_estimate[i] + 0.0001;
                filter_initial = filter_initial.clamp(min_gain, 1.0);

                // Weight the two suppression filters.
                filter_initial *= (SHORT_STARTUP_PHASE_BLOCKS - num_analyzed_frames) as f32;
                self.filter[i] *= num_analyzed_frames as f32;
                self.filter[i] += filter_initial;
                self.filter[i] *= ONE_BY_SHORT_STARTUP_PHASE_BLOCKS;
            }
        }

        self.spectrum_prev_process.copy_from_slice(signal_spectrum);
    }

    /// Compute an overall gain scaling factor.
    ///
    /// Adjusts the overall gain based on the energy ratio before/after
    /// filtering and the prior speech probability. Returns 1.0 during
    /// startup or when attenuation adjustment is disabled.
    pub fn compute_overall_scaling_factor(
        &self,
        num_analyzed_frames: i32,
        prior_speech_probability: f32,
        energy_before_filtering: f32,
        energy_after_filtering: f32,
    ) -> f32 {
        if !self.suppression_params.use_attenuation_adjustment
            || num_analyzed_frames <= LONG_STARTUP_PHASE_BLOCKS
        {
            return 1.0;
        }

        let mut gain =
            sqrt_fast_approximation(energy_after_filtering / (energy_before_filtering + 1.0));

        // Scaling for new version. Threshold in final energy gain factor.
        const B_LIM: f32 = 0.5;
        let mut scale_factor1 = 1.0f32;
        if gain > B_LIM {
            scale_factor1 = 1.0 + 1.3 * (gain - B_LIM);
            if gain * scale_factor1 > 1.0 {
                scale_factor1 = 1.0 / gain;
            }
        }

        let mut scale_factor2 = 1.0f32;
        if gain < B_LIM {
            // Do not reduce scale too much for pause regions: attenuation here
            // should be controlled by flooring.
            gain = gain.max(self.suppression_params.minimum_attenuating_gain);
            scale_factor2 = 1.0 - 0.3 * (B_LIM - gain);
        }

        // Combine both scales with speech/noise prob: note prior
        // (prior_speech_probability) is not frequency dependent.
        prior_speech_probability * scale_factor1 + (1.0 - prior_speech_probability) * scale_factor2
    }

    /// Returns the per-bin filter gains.
    pub fn filter(&self) -> &[f32; FFT_SIZE_BY_2_PLUS_1] {
        &self.filter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SuppressionLevel;

    fn default_filter() -> WienerFilter {
        WienerFilter::new(SuppressionParams::for_level(SuppressionLevel::K12dB))
    }

    #[test]
    fn initial_filter_is_unity() {
        let wf = default_filter();
        assert_eq!(*wf.filter(), [1.0; FFT_SIZE_BY_2_PLUS_1]);
    }

    #[test]
    fn update_with_no_noise_keeps_high_gain() {
        let mut wf = default_filter();
        let signal = [100.0f32; FFT_SIZE_BY_2_PLUS_1];
        let noise = [0.001f32; FFT_SIZE_BY_2_PLUS_1];
        let prev_noise = [0.001f32; FFT_SIZE_BY_2_PLUS_1];
        let parametric = [0.001f32; FFT_SIZE_BY_2_PLUS_1];

        wf.update(60, &noise, &prev_noise, &parametric, &signal);

        // With very high SNR, filter should be close to 1.0.
        for &f in wf.filter() {
            assert!(f > 0.9, "filter {f} should be close to 1.0 with high SNR");
        }
    }

    #[test]
    fn update_with_high_noise_attenuates() {
        let mut wf = default_filter();
        let signal = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let noise = [100.0f32; FFT_SIZE_BY_2_PLUS_1];
        let prev_noise = [100.0f32; FFT_SIZE_BY_2_PLUS_1];
        let parametric = [100.0f32; FFT_SIZE_BY_2_PLUS_1];

        wf.update(60, &noise, &prev_noise, &parametric, &signal);

        // With signal << noise, filter should be at minimum gain.
        let min_gain =
            SuppressionParams::for_level(SuppressionLevel::K12dB).minimum_attenuating_gain;
        for &f in wf.filter() {
            assert_eq!(f, min_gain);
        }
    }

    #[test]
    fn filter_values_are_bounded() {
        let mut wf = default_filter();
        let min_gain =
            SuppressionParams::for_level(SuppressionLevel::K12dB).minimum_attenuating_gain;

        // Run several frames with varying signal levels.
        for frame in 0..100 {
            let level = if frame % 2 == 0 { 100.0 } else { 1.0 };
            let signal = [level; FFT_SIZE_BY_2_PLUS_1];
            let noise = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
            let prev_noise = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
            let parametric = [10.0f32; FFT_SIZE_BY_2_PLUS_1];

            wf.update(frame, &noise, &prev_noise, &parametric, &signal);

            for &f in wf.filter() {
                assert!(
                    f >= min_gain && f <= 1.0,
                    "filter {f} out of bounds [{min_gain}, 1.0]"
                );
            }
        }
    }

    #[test]
    fn overall_scaling_disabled_during_startup() {
        let wf = default_filter();
        let scale = wf.compute_overall_scaling_factor(0, 0.5, 100.0, 50.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn overall_scaling_disabled_without_attenuation_adjustment() {
        let wf = WienerFilter::new(SuppressionParams::for_level(SuppressionLevel::K6dB));
        // K6dB has use_attenuation_adjustment = false.
        let scale = wf.compute_overall_scaling_factor(300, 0.5, 100.0, 50.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn overall_scaling_with_high_gain() {
        let wf = default_filter();
        // energy_after ≈ energy_before → gain ≈ 1.0 > B_LIM
        let scale = wf.compute_overall_scaling_factor(300, 0.5, 100.0, 90.0);
        assert!(
            scale > 0.0 && scale <= 2.0,
            "scale {scale} out of reasonable range"
        );
    }

    #[test]
    fn overall_scaling_with_low_gain() {
        let wf = default_filter();
        // energy_after << energy_before → gain < B_LIM
        let scale = wf.compute_overall_scaling_factor(300, 0.5, 100.0, 1.0);
        assert!(
            scale > 0.0 && scale <= 2.0,
            "scale {scale} out of reasonable range"
        );
    }
}
