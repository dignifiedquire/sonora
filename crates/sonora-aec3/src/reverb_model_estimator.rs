//! Reverb model parameter estimation coordinator.
//!
//! Ported from `modules/audio_processing/aec3/reverb_model_estimator.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::EchoCanceller3Config;
use crate::reverb_decay_estimator::ReverbDecayEstimator;
use crate::reverb_frequency_response::ReverbFrequencyResponse;

/// Estimates the model parameters for the reverberant echo.
pub(crate) struct ReverbModelEstimator {
    reverb_decay_estimators: Vec<ReverbDecayEstimator>,
    reverb_frequency_responses: Vec<ReverbFrequencyResponse>,
}

impl ReverbModelEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let reverb_decay_estimators = (0..num_capture_channels)
            .map(|_| ReverbDecayEstimator::new(config))
            .collect();
        let reverb_frequency_responses = (0..num_capture_channels)
            .map(|_| {
                ReverbFrequencyResponse::new(
                    config.ep_strength.use_conservative_tail_frequency_response,
                )
            })
            .collect();
        Self {
            reverb_decay_estimators,
            reverb_frequency_responses,
        }
    }

    /// Updates the estimates based on new data from all capture channels.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn update(
        &mut self,
        impulse_responses: &[Vec<f32>],
        frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        linear_filter_qualities: &[Option<f32>],
        filter_delays_blocks: &[i32],
        usable_linear_estimates: &[bool],
        stationary_block: bool,
    ) {
        let num_capture_channels = self.reverb_decay_estimators.len();
        debug_assert_eq!(num_capture_channels, impulse_responses.len());
        debug_assert_eq!(num_capture_channels, frequency_responses.len());
        debug_assert_eq!(num_capture_channels, usable_linear_estimates.len());

        for ch in 0..num_capture_channels {
            // Estimate the frequency response for the reverb.
            self.reverb_frequency_responses[ch].update(
                &frequency_responses[ch],
                filter_delays_blocks[ch] as usize,
                linear_filter_qualities[ch],
                stationary_block,
            );

            // Estimate the reverb decay.
            self.reverb_decay_estimators[ch].update(
                &impulse_responses[ch],
                linear_filter_qualities[ch],
                filter_delays_blocks[ch],
                usable_linear_estimates[ch],
                stationary_block,
            );
        }
    }

    /// Returns the reverb decay. The parameter `mild` indicates which
    /// exponential decay to return.
    pub(crate) fn reverb_decay(&self, mild: bool) -> f32 {
        // TODO(peah): Correct to properly support multiple channels.
        self.reverb_decay_estimators[0].decay(mild)
    }

    /// Returns the frequency response of the reverberant echo.
    pub(crate) fn get_reverb_frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        // TODO(peah): Correct to properly support multiple channels.
        self.reverb_frequency_responses[0].frequency_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_with_multiple_channels() {
        let config = EchoCanceller3Config::default();
        let estimator = ReverbModelEstimator::new(&config, 2);
        let decay = estimator.reverb_decay(false);
        assert!(decay > 0.0);
        let freq_resp = estimator.get_reverb_frequency_response();
        for &v in freq_resp {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn update_with_stationary_signal() {
        let config = EchoCanceller3Config::default();
        let mut estimator = ReverbModelEstimator::new(&config, 1);
        let len = config.filter.refined.length_blocks;
        let impulse_responses = vec![vec![0.0f32; len * 64]];
        let frequency_responses = vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; len]];
        let qualities = vec![Some(1.0)];
        let delays = vec![2];
        let usable = vec![true];
        let initial_decay = estimator.reverb_decay(false);
        estimator.update(
            &impulse_responses,
            &frequency_responses,
            &qualities,
            &delays,
            &usable,
            true, // stationary
        );
        // Stationary signal should not change anything.
        assert_eq!(estimator.reverb_decay(false), initial_decay);
    }
}
