//! Peak envelope level estimator with exponential smoothing.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/fixed_digital_level_estimator.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::SUB_FRAMES_IN_FRAME;

/// Initial filter state level.
const INITIAL_FILTER_STATE_LEVEL: f32 = 0.0;

/// Instant attack.
const ATTACK_FILTER_CONSTANT: f32 = 0.0;

/// Limiter decay constant.
/// Computed as `10 ** (-1/20 * subframe_duration / decay_ms)` where:
/// - `subframe_duration` is `FRAME_DURATION_MS / SUB_FRAMES_IN_FRAME`;
/// - `decay_ms` is 20.0 (from `agc2_testing_common.h`).
const DECAY_FILTER_CONSTANT: f32 = 0.9971259;

/// Produces a smooth signal level estimate from an input audio stream.
/// The estimate smoothing is done through exponential filtering.
pub(crate) struct FixedDigitalLevelEstimator {
    filter_state_level: f32,
    samples_in_frame: i32,
    samples_in_sub_frame: i32,
}

impl FixedDigitalLevelEstimator {
    /// Creates a new estimator.
    ///
    /// `samples_per_channel` must be divisible by `SUB_FRAMES_IN_FRAME`.
    pub(crate) fn new(samples_per_channel: usize) -> Self {
        let mut est = Self {
            filter_state_level: INITIAL_FILTER_STATE_LEVEL,
            samples_in_frame: 0,
            samples_in_sub_frame: 0,
        };
        est.set_samples_per_channel(samples_per_channel);
        est.check_parameter_combination();
        est
    }

    /// Computes the level envelope for a multi-channel frame.
    ///
    /// The input is assumed to be in FloatS16 format. Returns `SUB_FRAMES_IN_FRAME`
    /// level estimates, one per sub-frame.
    pub(crate) fn compute_level(
        &mut self,
        frame: &[&[f32]],
    ) -> [f32; SUB_FRAMES_IN_FRAME as usize] {
        let num_channels = frame.len();
        debug_assert!(num_channels > 0);
        debug_assert_eq!(frame[0].len(), self.samples_in_frame as usize);

        // Compute max envelope without smoothing.
        let mut envelope = [0.0_f32; SUB_FRAMES_IN_FRAME as usize];
        let sub_frame_len = self.samples_in_sub_frame as usize;
        for channel in frame {
            for (sub_frame, env) in envelope.iter_mut().enumerate() {
                let sub_frame_samples =
                    &channel[sub_frame * sub_frame_len..(sub_frame + 1) * sub_frame_len];
                for &sample in sub_frame_samples {
                    *env = env.max(sample.abs());
                }
            }
        }

        // Make sure envelope increases happen one step earlier so that the
        // corresponding *gain decrease* doesn't miss a sudden signal
        // increase due to interpolation.
        for sub_frame in 0..SUB_FRAMES_IN_FRAME as usize - 1 {
            if envelope[sub_frame] < envelope[sub_frame + 1] {
                envelope[sub_frame] = envelope[sub_frame + 1];
            }
        }

        // Add attack / decay smoothing.
        for env in &mut envelope {
            let envelope_value = *env;
            if envelope_value > self.filter_state_level {
                *env = envelope_value * (1.0 - ATTACK_FILTER_CONSTANT)
                    + self.filter_state_level * ATTACK_FILTER_CONSTANT;
            } else {
                *env = envelope_value * (1.0 - DECAY_FILTER_CONSTANT)
                    + self.filter_state_level * DECAY_FILTER_CONSTANT;
            }
            self.filter_state_level = *env;
        }

        envelope
    }

    /// Changes the sample rate (samples per channel for a 10ms frame).
    pub(crate) fn set_samples_per_channel(&mut self, samples_per_channel: usize) {
        self.samples_in_frame = samples_per_channel as i32;
        self.samples_in_sub_frame = self.samples_in_frame / SUB_FRAMES_IN_FRAME;
        self.check_parameter_combination();
    }

    /// Resets the level estimator internal state.
    pub(crate) fn reset(&mut self) {
        self.filter_state_level = INITIAL_FILTER_STATE_LEVEL;
    }

    /// Returns the last computed audio level.
    pub(crate) fn last_audio_level(&self) -> f32 {
        self.filter_state_level
    }

    fn check_parameter_combination(&self) {
        debug_assert!(self.samples_in_frame > 0);
        debug_assert!(SUB_FRAMES_IN_FRAME <= self.samples_in_frame);
        debug_assert_eq!(self.samples_in_frame % SUB_FRAMES_IN_FRAME, 0);
        debug_assert!(self.samples_in_sub_frame > 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{FRAME_DURATION_MS, SUB_FRAMES_IN_FRAME, dbfs_to_float_s16};

    const INPUT_LEVEL: f32 = 10000.0;

    /// Decay time constant in ms (from agc2_testing_common.h).
    const DECAY_MS: f32 = 20.0;

    /// Returns `sample_rate / 100` (samples per channel for 10ms).
    fn sample_rate_to_default_channel_size(sample_rate: usize) -> usize {
        sample_rate / 100
    }

    /// Run audio at specified settings through the level estimator, and
    /// verify that the output level falls within the bounds.
    fn test_level_estimator(
        samples_per_channel: usize,
        num_channels: usize,
        input_level_linear_scale: f32,
        expected_min: f32,
        expected_max: f32,
    ) {
        let mut level_estimator = FixedDigitalLevelEstimator::new(samples_per_channel);

        // Create constant-level frame.
        let channel_data = vec![input_level_linear_scale; samples_per_channel];
        let channels: Vec<&[f32]> = vec![channel_data.as_slice(); num_channels];

        for i in 0..500 {
            let level = level_estimator.compute_level(&channels);

            // Give the estimator some time to ramp up.
            if i < 50 {
                continue;
            }

            for x in &level {
                assert!(
                    expected_min <= *x,
                    "level {x} < expected_min {expected_min} at frame {i}"
                );
                assert!(
                    *x <= expected_max,
                    "level {x} > expected_max {expected_max} at frame {i}"
                );
            }
        }
    }

    /// Returns time it takes for the level estimator to decrease its level
    /// estimate by `level_reduction_db`.
    fn time_ms_to_decrease_level(
        samples_per_channel: usize,
        num_channels: usize,
        input_level_db: f32,
        level_reduction_db: f32,
    ) -> f32 {
        let input_level = dbfs_to_float_s16(input_level_db);
        debug_assert!(level_reduction_db > 0.0);

        let channel_data = vec![input_level; samples_per_channel];
        let channels: Vec<&[f32]> = vec![channel_data.as_slice(); num_channels];

        let mut level_estimator = FixedDigitalLevelEstimator::new(samples_per_channel);

        // Give the LevelEstimator plenty of time to ramp up and stabilize.
        let mut last_level = 0.0_f32;
        for _ in 0..500 {
            let level_envelope = level_estimator.compute_level(&channels);
            last_level = *level_envelope.last().unwrap();
        }

        // Set input to 0.
        let zero_channel_data = vec![0.0_f32; samples_per_channel];
        let zero_channels: Vec<&[f32]> = vec![zero_channel_data.as_slice(); num_channels];

        let reduced_level_linear = dbfs_to_float_s16(input_level_db - level_reduction_db);
        let mut sub_frames_until_level_reduction = 0;
        while last_level > reduced_level_linear {
            let level_envelope = level_estimator.compute_level(&zero_channels);
            for v in &level_envelope {
                assert!(*v < last_level, "level should decrease monotonically");
                sub_frames_until_level_reduction += 1;
                last_level = *v;
                if last_level <= reduced_level_linear {
                    break;
                }
            }
        }
        sub_frames_until_level_reduction as f32 * FRAME_DURATION_MS as f32
            / SUB_FRAMES_IN_FRAME as f32
    }

    #[test]
    fn estimator_should_not_crash() {
        test_level_estimator(
            sample_rate_to_default_channel_size(8000),
            1,
            0.0,
            f32::MIN,
            f32::MAX,
        );
    }

    #[test]
    fn estimator_should_estimate_constant_level() {
        test_level_estimator(
            sample_rate_to_default_channel_size(10000),
            1,
            INPUT_LEVEL,
            INPUT_LEVEL * 0.99,
            INPUT_LEVEL * 1.01,
        );
    }

    #[test]
    fn estimator_should_estimate_constant_level_for_many_channels() {
        let num_channels = 10;
        test_level_estimator(
            sample_rate_to_default_channel_size(20000),
            num_channels,
            INPUT_LEVEL,
            INPUT_LEVEL * 0.99,
            INPUT_LEVEL * 1.01,
        );
    }

    #[test]
    fn time_to_decrease_for_low_level() {
        let level_reduction_db = 25.0;
        let initial_low_level = -40.0;
        let expected_time = level_reduction_db * DECAY_MS;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(22000),
            1,
            initial_low_level,
            level_reduction_db,
        );

        assert!(
            expected_time * 0.9 <= time_to_decrease,
            "time_to_decrease {time_to_decrease} < expected {}",
            expected_time * 0.9
        );
        assert!(
            time_to_decrease <= expected_time * 1.1,
            "time_to_decrease {time_to_decrease} > expected {}",
            expected_time * 1.1
        );
    }

    #[test]
    fn time_to_decrease_for_full_scale_level() {
        let level_reduction_db = 25.0;
        let expected_time = level_reduction_db * DECAY_MS;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(26000),
            1,
            0.0,
            level_reduction_db,
        );

        assert!(
            expected_time * 0.9 <= time_to_decrease,
            "time_to_decrease {time_to_decrease} < expected {}",
            expected_time * 0.9
        );
        assert!(
            time_to_decrease <= expected_time * 1.1,
            "time_to_decrease {time_to_decrease} > expected {}",
            expected_time * 1.1
        );
    }

    #[test]
    fn time_to_decrease_for_multiple_channels() {
        let level_reduction_db = 25.0;
        let expected_time = level_reduction_db * DECAY_MS;
        let num_channels = 10;

        let time_to_decrease = time_ms_to_decrease_level(
            sample_rate_to_default_channel_size(28000),
            num_channels,
            0.0,
            level_reduction_db,
        );

        assert!(
            expected_time * 0.9 <= time_to_decrease,
            "time_to_decrease {time_to_decrease} < expected {}",
            expected_time * 0.9
        );
        assert!(
            time_to_decrease <= expected_time * 1.1,
            "time_to_decrease {time_to_decrease} > expected {}",
            expected_time * 1.1
        );
    }
}
