//! Nearend detector — selects whether the suppressor is in nearend or echo
//! state.
//!
//! Ported from `modules/audio_processing/aec3/nearend_detector.h`,
//! `dominant_nearend_detector.h/cc`, and `subband_nearend_detector.h/cc`.
//!
//! The C++ code uses a virtual interface (`NearendDetector`). In Rust we use
//! an enum that dispatches to one of two concrete implementations.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::{DominantNearendDetection, SubbandNearendDetection};
use crate::moving_average::MovingAverage;

/// Nearend detector dispatching to either dominant or subband variant.
pub(crate) enum NearendDetector {
    Dominant(DominantNearendDetector),
    Subband(SubbandNearendDetector),
}

impl NearendDetector {
    /// Returns whether the current state is the nearend state.
    pub(crate) fn is_nearend_state(&self) -> bool {
        match self {
            Self::Dominant(d) => d.is_nearend_state(),
            Self::Subband(s) => s.is_nearend_state(),
        }
    }

    /// Updates the state selection based on latest spectral estimates.
    pub(crate) fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        initial_state: bool,
    ) {
        match self {
            Self::Dominant(d) => d.update(
                nearend_spectrum,
                residual_echo_spectrum,
                comfort_noise_spectrum,
                initial_state,
            ),
            Self::Subband(s) => s.update(nearend_spectrum, comfort_noise_spectrum),
        }
    }
}

/// Dominant nearend detector — uses low-frequency energy comparison with
/// trigger/hold counters.
pub(crate) struct DominantNearendDetector {
    enr_threshold: f32,
    enr_exit_threshold: f32,
    snr_threshold: f32,
    hold_duration: i32,
    trigger_threshold: i32,
    use_during_initial_phase: bool,
    num_capture_channels: usize,
    nearend_state: bool,
    trigger_counters: Vec<i32>,
    hold_counters: Vec<i32>,
}

impl DominantNearendDetector {
    pub(crate) fn new(config: &DominantNearendDetection, num_capture_channels: usize) -> Self {
        Self {
            enr_threshold: config.enr_threshold,
            enr_exit_threshold: config.enr_exit_threshold,
            snr_threshold: config.snr_threshold,
            hold_duration: config.hold_duration,
            trigger_threshold: config.trigger_threshold,
            use_during_initial_phase: config.use_during_initial_phase,
            num_capture_channels,
            nearend_state: false,
            trigger_counters: vec![0; num_capture_channels],
            hold_counters: vec![0; num_capture_channels],
        }
    }

    pub(crate) fn is_nearend_state(&self) -> bool {
        self.nearend_state
    }

    pub(crate) fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        initial_state: bool,
    ) {
        self.nearend_state = false;

        let low_frequency_energy =
            |spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1]| -> f32 { spectrum[1..16].iter().sum() };

        for ch in 0..self.num_capture_channels {
            let ne_sum = low_frequency_energy(&nearend_spectrum[ch]);
            let echo_sum = low_frequency_energy(&residual_echo_spectrum[ch]);
            let noise_sum = low_frequency_energy(&comfort_noise_spectrum[ch]);

            // Detect strong active nearend if the nearend is sufficiently
            // stronger than the echo and the nearend noise.
            if (!initial_state || self.use_during_initial_phase)
                && echo_sum < self.enr_threshold * ne_sum
                && ne_sum > self.snr_threshold * noise_sum
            {
                self.trigger_counters[ch] += 1;
                if self.trigger_counters[ch] >= self.trigger_threshold {
                    // After a period of strong active nearend activity, flag
                    // nearend mode.
                    self.hold_counters[ch] = self.hold_duration;
                    self.trigger_counters[ch] = self.trigger_threshold;
                }
            } else {
                // Forget previously detected strong active nearend activity.
                self.trigger_counters[ch] = (self.trigger_counters[ch] - 1).max(0);
            }

            // Exit nearend-state early at strong echo.
            if echo_sum > self.enr_exit_threshold * ne_sum
                && echo_sum > self.snr_threshold * noise_sum
            {
                self.hold_counters[ch] = 0;
            }

            // Remain in any nearend mode for a certain duration.
            self.hold_counters[ch] = (self.hold_counters[ch] - 1).max(0);
            self.nearend_state = self.nearend_state || self.hold_counters[ch] > 0;
        }
    }
}

/// Subband nearend detector — compares nearend energy in two subbands.
pub(crate) struct SubbandNearendDetector {
    nearend_threshold: f32,
    snr_threshold: f32,
    subband1_low: usize,
    subband1_high: usize,
    subband2_low: usize,
    subband2_high: usize,
    num_capture_channels: usize,
    nearend_smoothers: Vec<MovingAverage>,
    one_over_subband_length1: f32,
    one_over_subband_length2: f32,
    nearend_state: bool,
}

impl SubbandNearendDetector {
    pub(crate) fn new(config: &SubbandNearendDetection, num_capture_channels: usize) -> Self {
        Self {
            nearend_threshold: config.nearend_threshold,
            snr_threshold: config.snr_threshold,
            subband1_low: config.subband1.low,
            subband1_high: config.subband1.high,
            subband2_low: config.subband2.low,
            subband2_high: config.subband2.high,
            num_capture_channels,
            nearend_smoothers: (0..num_capture_channels)
                .map(|_| MovingAverage::new(FFT_LENGTH_BY_2_PLUS_1, config.nearend_average_blocks))
                .collect(),
            one_over_subband_length1: 1.0 / (config.subband1.high - config.subband1.low + 1) as f32,
            one_over_subband_length2: 1.0 / (config.subband2.high - config.subband2.low + 1) as f32,
            nearend_state: false,
        }
    }

    pub(crate) fn is_nearend_state(&self) -> bool {
        self.nearend_state
    }

    pub(crate) fn update(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        self.nearend_state = false;
        for ch in 0..self.num_capture_channels {
            let noise = &comfort_noise_spectrum[ch];
            let mut nearend = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.nearend_smoothers[ch].average(&nearend_spectrum[ch], &mut nearend);

            // Noise power of the first region.
            let noise_power: f32 = noise[self.subband1_low..=self.subband1_high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length1;

            // Nearend power of the first region.
            let nearend_power_subband1: f32 = nearend[self.subband1_low..=self.subband1_high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length1;

            // Nearend power of the second region.
            let nearend_power_subband2: f32 = nearend[self.subband2_low..=self.subband2_high]
                .iter()
                .sum::<f32>()
                * self.one_over_subband_length2;

            // One channel is sufficient to trigger nearend state.
            self.nearend_state = self.nearend_state
                || (nearend_power_subband1 < self.nearend_threshold * nearend_power_subband2
                    && nearend_power_subband1 > self.snr_threshold * noise_power);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_initially_not_nearend() {
        let config = DominantNearendDetection::default();
        let det = DominantNearendDetector::new(&config, 1);
        assert!(!det.is_nearend_state());
    }

    #[test]
    fn dominant_detects_strong_nearend() {
        let config = DominantNearendDetection::default();
        let mut det = DominantNearendDetector::new(&config, 1);

        let mut nearend = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut echo = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut noise = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];

        // Strong nearend, weak echo, weak noise.
        for k in 0..16 {
            nearend[0][k] = 1000.0;
            echo[0][k] = 0.01;
            noise[0][k] = 0.01;
        }

        // Trigger enough times to reach threshold + hold.
        for _ in 0..config.trigger_threshold + config.hold_duration + 1 {
            det.update(&nearend, &echo, &noise, false);
        }
        assert!(det.is_nearend_state());
    }

    #[test]
    fn subband_initially_not_nearend() {
        let config = SubbandNearendDetection::default();
        let det = SubbandNearendDetector::new(&config, 1);
        assert!(!det.is_nearend_state());
    }
}
