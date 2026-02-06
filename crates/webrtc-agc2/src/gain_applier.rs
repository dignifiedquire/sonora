//! Per-sample gain application with linear ramping.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/gain_applier.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{MAX_FLOAT_S16_VALUE, MIN_FLOAT_S16_VALUE};

/// Returns true when the gain factor is so close to 1 that it would
/// not affect int16 samples.
fn gain_close_to_one(gain_factor: f32) -> bool {
    let threshold = 1.0 / MAX_FLOAT_S16_VALUE;
    (1.0 - threshold..=1.0 + threshold).contains(&gain_factor)
}

/// Clamps all samples in multi-channel signal to FloatS16 range.
fn clip_signal(signal: &mut [&mut [f32]]) {
    for channel in signal.iter_mut() {
        for sample in channel.iter_mut() {
            *sample = sample.clamp(MIN_FLOAT_S16_VALUE, MAX_FLOAT_S16_VALUE);
        }
    }
}

/// Applies gain with linear ramping across the frame.
fn apply_gain_with_ramping(
    last_gain: f32,
    gain_at_end: f32,
    inverse_samples_per_channel: f32,
    signal: &mut [&mut [f32]],
) {
    // Do not modify the signal.
    if last_gain == gain_at_end && gain_close_to_one(gain_at_end) {
        return;
    }

    // Gain is constant and different from 1.
    if last_gain == gain_at_end {
        for channel in signal.iter_mut() {
            for sample in channel.iter_mut() {
                *sample *= gain_at_end;
            }
        }
        return;
    }

    // The gain changes. Ramp linearly to avoid discontinuities.
    let increment = (gain_at_end - last_gain) * inverse_samples_per_channel;
    for channel in signal.iter_mut() {
        let mut gain = last_gain;
        for sample in channel.iter_mut() {
            *sample *= gain;
            gain += increment;
        }
    }
}

/// Applies a gain factor to multi-channel audio with linear ramping between
/// frames and optional hard clipping.
pub(crate) struct GainApplier {
    hard_clip_samples: bool,
    last_gain_factor: f32,
    current_gain_factor: f32,
    samples_per_channel: i32,
    inverse_samples_per_channel: f32,
}

impl GainApplier {
    pub(crate) fn new(hard_clip_samples: bool, initial_gain_factor: f32) -> Self {
        Self {
            hard_clip_samples,
            last_gain_factor: initial_gain_factor,
            current_gain_factor: initial_gain_factor,
            samples_per_channel: -1,
            inverse_samples_per_channel: -1.0,
        }
    }

    /// Applies gain to the signal with linear ramping between frames.
    pub(crate) fn apply_gain(&mut self, signal: &mut [&mut [f32]]) {
        let spc = signal.first().map_or(0, |ch| ch.len()) as i32;
        if spc != self.samples_per_channel {
            self.initialize(spc);
        }

        apply_gain_with_ramping(
            self.last_gain_factor,
            self.current_gain_factor,
            self.inverse_samples_per_channel,
            signal,
        );

        self.last_gain_factor = self.current_gain_factor;

        if self.hard_clip_samples {
            clip_signal(signal);
        }
    }

    /// Sets the target gain factor for the next frame.
    pub(crate) fn set_gain_factor(&mut self, gain_factor: f32) {
        debug_assert!(gain_factor > 0.0);
        self.current_gain_factor = gain_factor;
    }

    pub(crate) fn get_gain_factor(&self) -> f32 {
        self.current_gain_factor
    }

    fn initialize(&mut self, samples_per_channel: i32) {
        debug_assert!(samples_per_channel > 0);
        self.samples_per_channel = samples_per_channel;
        self.inverse_samples_per_channel = 1.0 / samples_per_channel as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a multi-channel audio frame filled with a constant value.
    fn make_frame(num_channels: usize, samples_per_channel: usize, value: f32) -> Vec<Vec<f32>> {
        vec![vec![value; samples_per_channel]; num_channels]
    }

    /// Converts `Vec<Vec<f32>>` to `&mut [&mut [f32]]` for the API.
    fn as_mut_slices(frame: &mut [Vec<f32>]) -> Vec<&mut [f32]> {
        frame.iter_mut().map(|ch| ch.as_mut_slice()).collect()
    }

    #[test]
    fn initial_gain_is_respected() {
        let initial_signal_level = 123.0_f32;
        let gain_factor = 10.0_f32;
        let mut frame = make_frame(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(true, gain_factor);

        let mut slices = as_mut_slices(&mut frame);
        gain_applier.apply_gain(&mut slices);
        assert!(
            (frame[0][0] - initial_signal_level * gain_factor).abs() < 0.1,
            "expected ~{}, got {}",
            initial_signal_level * gain_factor,
            frame[0][0]
        );
    }

    #[test]
    fn clipping_is_done() {
        let initial_signal_level = 30000.0_f32;
        let gain_factor = 10.0_f32;
        let mut frame = make_frame(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(true, gain_factor);

        let mut slices = as_mut_slices(&mut frame);
        gain_applier.apply_gain(&mut slices);
        assert!(
            (frame[0][0] - i16::MAX as f32).abs() < 0.1,
            "expected ~{}, got {}",
            i16::MAX,
            frame[0][0]
        );
    }

    #[test]
    fn clipping_is_not_done() {
        let initial_signal_level = 30000.0_f32;
        let gain_factor = 10.0_f32;
        let mut frame = make_frame(1, 1, initial_signal_level);
        let mut gain_applier = GainApplier::new(false, gain_factor);

        let mut slices = as_mut_slices(&mut frame);
        gain_applier.apply_gain(&mut slices);
        assert!(
            (frame[0][0] - initial_signal_level * gain_factor).abs() < 0.1,
            "expected ~{}, got {}",
            initial_signal_level * gain_factor,
            frame[0][0]
        );
    }

    #[test]
    fn ramping_is_done() {
        let initial_signal_level = 30000.0_f32;
        let initial_gain_factor = 1.0_f32;
        let target_gain_factor = 0.5_f32;
        let num_channels = 3;
        let samples_per_channel = 4;
        let mut frame = make_frame(num_channels, samples_per_channel, initial_signal_level);
        let mut gain_applier = GainApplier::new(false, initial_gain_factor);

        gain_applier.set_gain_factor(target_gain_factor);
        let mut slices = as_mut_slices(&mut frame);
        gain_applier.apply_gain(&mut slices);

        // The maximal gain change should be close to linear interpolation.
        for (channel, ch_data) in frame.iter().enumerate() {
            let mut max_signal_change = 0.0_f32;
            let mut last_signal_level = initial_signal_level;
            for &sample in ch_data {
                let current_change = (last_signal_level - sample).abs();
                max_signal_change = max_signal_change.max(current_change);
                last_signal_level = sample;
            }
            let total_gain_change =
                ((initial_gain_factor - target_gain_factor) * initial_signal_level).abs();
            assert!(
                (max_signal_change - total_gain_change / samples_per_channel as f32).abs() < 0.1,
                "channel {channel}: max_change {max_signal_change}, expected ~{}",
                total_gain_change / samples_per_channel as f32,
            );
        }

        // Next frame should have the desired level.
        let mut next_frame = make_frame(num_channels, samples_per_channel, initial_signal_level);
        let mut slices = as_mut_slices(&mut next_frame);
        gain_applier.apply_gain(&mut slices);

        // The first sample should have the new gain (no ramping needed).
        assert!(
            (next_frame[0][0] - initial_signal_level * target_gain_factor).abs() < 0.1,
            "expected ~{}, got {}",
            initial_signal_level * target_gain_factor,
            next_frame[0][0]
        );
    }
}
