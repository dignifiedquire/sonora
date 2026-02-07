//! Output limiter with interpolated gain curve and per-sample scaling.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/limiter.h/.cc`.

use crate::common::{
    MAX_FLOAT_S16_VALUE, MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL, MIN_FLOAT_S16_VALUE,
    SUB_FRAMES_IN_FRAME,
};
use crate::fixed_digital_level_estimator::FixedDigitalLevelEstimator;
use crate::interpolated_gain_curve::InterpolatedGainCurve;

/// Power used for interpolation of the first sub-frame during attack.
/// Reduces the chances of over-shooting (and hence saturation).
const ATTACK_FIRST_SUBFRAME_INTERPOLATION_POWER: f32 = 8.0;

fn interpolate_first_subframe(last_factor: f32, current_factor: f32, subframe: &mut [f32]) {
    let n = subframe.len() as f32;
    let p = ATTACK_FIRST_SUBFRAME_INTERPOLATION_POWER;
    for (i, sample) in subframe.iter_mut().enumerate() {
        let t = i as f32 / n;
        *sample = (1.0 - t).powf(p) * (last_factor - current_factor) + current_factor;
    }
}

fn compute_per_sample_subframe_factors(
    scaling_factors: &[f32],
    per_sample_scaling_factors: &mut [f32],
) {
    let num_subframes = scaling_factors.len() - 1;
    let subframe_size = per_sample_scaling_factors.len() / num_subframes;
    debug_assert_eq!(per_sample_scaling_factors.len() % num_subframes, 0);

    // Handle first sub-frame differently in case of attack.
    let is_attack = scaling_factors[0] > scaling_factors[1];
    if is_attack {
        interpolate_first_subframe(
            scaling_factors[0],
            scaling_factors[1],
            &mut per_sample_scaling_factors[..subframe_size],
        );
    }

    let start = if is_attack { 1 } else { 0 };
    for i in start..num_subframes {
        let subframe_start = i * subframe_size;
        let scaling_start = scaling_factors[i];
        let scaling_end = scaling_factors[i + 1];
        let scaling_diff = (scaling_end - scaling_start) / subframe_size as f32;
        for j in 0..subframe_size {
            per_sample_scaling_factors[subframe_start + j] =
                scaling_start + scaling_diff * j as f32;
        }
    }
}

fn scale_samples(per_sample_scaling_factors: &[f32], signal: &mut [&mut [f32]]) {
    let samples_per_channel = signal.first().map_or(0, |ch| ch.len());
    debug_assert_eq!(samples_per_channel, per_sample_scaling_factors.len());
    for channel in signal.iter_mut() {
        for (sample, &factor) in channel.iter_mut().zip(per_sample_scaling_factors) {
            *sample = (*sample * factor).clamp(MIN_FLOAT_S16_VALUE, MAX_FLOAT_S16_VALUE);
        }
    }
}

/// Output limiter that applies gain curve compression and hard clipping.
pub struct Limiter {
    interp_gain_curve: InterpolatedGainCurve,
    level_estimator: FixedDigitalLevelEstimator,
    scaling_factors: [f32; SUB_FRAMES_IN_FRAME as usize + 1],
    per_sample_scaling_factors: [f32; MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL],
    last_scaling_factor: f32,
}

impl Limiter {
    /// Creates a new limiter.
    ///
    /// `samples_per_channel` must be <= `MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL`
    /// and divisible by `SUB_FRAMES_IN_FRAME`.
    pub fn new(samples_per_channel: usize) -> Self {
        debug_assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);
        Self {
            interp_gain_curve: InterpolatedGainCurve::default(),
            level_estimator: FixedDigitalLevelEstimator::new(samples_per_channel),
            scaling_factors: [0.0; SUB_FRAMES_IN_FRAME as usize + 1],
            per_sample_scaling_factors: [0.0; MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL],
            last_scaling_factor: 1.0,
        }
    }

    /// Applies limiter and hard-clipping to the signal.
    pub fn process(&mut self, signal: &mut [&mut [f32]]) {
        let samples_per_channel = signal.first().map_or(0, |ch| ch.len());
        debug_assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);

        // Build immutable view of signal for level estimation.
        let signal_refs: Vec<&[f32]> = signal.iter().map(|ch| &**ch).collect();
        let level_estimate = self.level_estimator.compute_level(&signal_refs);

        debug_assert_eq!(level_estimate.len() + 1, self.scaling_factors.len());
        self.scaling_factors[0] = self.last_scaling_factor;
        for (i, &level) in level_estimate.iter().enumerate() {
            self.scaling_factors[i + 1] = self.interp_gain_curve.look_up_gain_to_apply(level);
        }

        compute_per_sample_subframe_factors(
            &self.scaling_factors,
            &mut self.per_sample_scaling_factors[..samples_per_channel],
        );
        scale_samples(
            &self.per_sample_scaling_factors[..samples_per_channel],
            signal,
        );

        self.last_scaling_factor = self.scaling_factors[SUB_FRAMES_IN_FRAME as usize];
    }

    /// Changes the sample rate.
    pub fn set_samples_per_channel(&mut self, samples_per_channel: usize) {
        debug_assert!(samples_per_channel <= MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL);
        self.level_estimator
            .set_samples_per_channel(samples_per_channel);
    }

    /// Resets the internal state.
    pub fn reset(&mut self) {
        self.level_estimator.reset();
    }

    /// Returns the last audio level from the level estimator.
    pub fn last_audio_level(&self) -> f32 {
        self.level_estimator.last_audio_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{MAX_ABS_FLOAT_S16_VALUE, dbfs_to_float_s16};

    #[test]
    fn limiter_should_construct_and_run() {
        let samples_per_channel = 480;
        let mut limiter = Limiter::new(samples_per_channel);

        let mut buffer = vec![MAX_ABS_FLOAT_S16_VALUE; samples_per_channel];
        let mut channels: Vec<&mut [f32]> = vec![buffer.as_mut_slice()];
        limiter.process(&mut channels);
    }

    #[test]
    fn output_volume_above_threshold() {
        let samples_per_channel = 480;
        let input_level = (MAX_ABS_FLOAT_S16_VALUE + dbfs_to_float_s16(1.0)) / 2.0;
        let mut limiter = Limiter::new(samples_per_channel);

        // Give the level estimator time to adapt.
        for _ in 0..5 {
            let mut buffer = vec![input_level; samples_per_channel];
            let mut channels: Vec<&mut [f32]> = vec![buffer.as_mut_slice()];
            limiter.process(&mut channels);
        }

        let mut buffer = vec![input_level; samples_per_channel];
        let mut channels: Vec<&mut [f32]> = vec![buffer.as_mut_slice()];
        limiter.process(&mut channels);
        for &sample in channels[0].iter() {
            assert!(
                0.9 * MAX_ABS_FLOAT_S16_VALUE < sample,
                "sample {sample} should be above threshold {}",
                0.9 * MAX_ABS_FLOAT_S16_VALUE
            );
        }
    }
}
