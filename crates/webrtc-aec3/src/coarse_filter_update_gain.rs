//! Coarse filter update gain computation.
//!
//! Computes the fixed gain for the coarse (shadow) adaptive filter using a
//! simple NLMS-like step size based on render power.
//!
//! Ported from
//! `modules/audio_processing/aec3/coarse_filter_update_gain.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::CoarseConfiguration;
use crate::fft_data::FftData;
use crate::render_signal_analyzer::RenderSignalAnalyzer;

/// Computes the fixed gain for the coarse adaptive filter.
pub(crate) struct CoarseFilterUpdateGain {
    current_config: CoarseConfiguration,
    target_config: CoarseConfiguration,
    old_target_config: CoarseConfiguration,
    config_change_duration_blocks: i32,
    one_by_config_change_duration_blocks: f32,
    poor_signal_excitation_counter: usize,
    call_counter: usize,
    config_change_counter: i32,
}

impl CoarseFilterUpdateGain {
    pub(crate) fn new(config: &CoarseConfiguration, config_change_duration_blocks: usize) -> Self {
        debug_assert!(config_change_duration_blocks > 0);
        let mut gain = Self {
            current_config: config.clone(),
            target_config: config.clone(),
            old_target_config: config.clone(),
            config_change_duration_blocks: config_change_duration_blocks as i32,
            one_by_config_change_duration_blocks: 1.0 / config_change_duration_blocks as f32,
            poor_signal_excitation_counter: 0,
            call_counter: 0,
            config_change_counter: 0,
        };
        gain.set_config(config, true);
        gain
    }

    /// Takes action in the case of a known echo path change.
    pub(crate) fn handle_echo_path_change(&mut self) {
        self.poor_signal_excitation_counter = 0;
        self.call_counter = 0;
    }

    /// Computes the gain.
    pub(crate) fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        e_coarse: &FftData,
        size_partitions: usize,
        saturated_capture_signal: bool,
        g: &mut FftData,
    ) {
        self.call_counter += 1;
        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_signal_excitation_counter = 0;
        }

        // Do not update the filter if the render is not sufficiently excited.
        self.poor_signal_excitation_counter += 1;
        if self.poor_signal_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            g.clear();
            return;
        }

        // Compute mu.
        let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if render_power[k] > self.current_config.noise_gate {
                mu[k] = self.current_config.rate / render_power[k];
            }
        }

        // Avoid updating the filter close to narrow bands in the render signals.
        render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);

        // G = mu * E.
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            g.re[k] = mu[k] * e_coarse.re[k];
            g.im[k] = mu[k] * e_coarse.im[k];
        }
    }

    /// Sets a new config.
    pub(crate) fn set_config(&mut self, config: &CoarseConfiguration, immediate_effect: bool) {
        if immediate_effect {
            self.old_target_config = config.clone();
            self.current_config = config.clone();
            self.target_config = config.clone();
            self.config_change_counter = 0;
        } else {
            self.old_target_config = self.current_config.clone();
            self.target_config = config.clone();
            self.config_change_counter = self.config_change_duration_blocks;
        }
    }

    fn update_current_config(&mut self) {
        if self.config_change_counter > 0 {
            self.config_change_counter -= 1;
            if self.config_change_counter > 0 {
                let change_factor =
                    self.config_change_counter as f32 * self.one_by_config_change_duration_blocks;

                let average = |from: f32, to: f32, weight: f32| from * weight + to * (1.0 - weight);

                self.current_config.rate = average(
                    self.old_target_config.rate,
                    self.target_config.rate,
                    change_factor,
                );
                self.current_config.noise_gate = average(
                    self.old_target_config.noise_gate,
                    self.target_config.noise_gate,
                    change_factor,
                );
            } else {
                self.current_config = self.target_config.clone();
                self.old_target_config = self.target_config.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CoarseConfiguration {
        CoarseConfiguration {
            length_blocks: 13,
            rate: 0.7,
            noise_gate: 20075.0,
        }
    }

    #[test]
    fn saturation_zeros_gain() {
        let mut gain = CoarseFilterUpdateGain::new(&default_config(), 250);
        let render_power = [100_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let rsa = RenderSignalAnalyzer::default();
        let e = FftData::default();
        let mut g = FftData::default();

        // Saturated → G should be zero.
        gain.compute(&render_power, &rsa, &e, 10, true, &mut g);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert_eq!(g.re[k], 0.0);
            assert_eq!(g.im[k], 0.0);
        }
    }

    #[test]
    fn gain_nonzero_after_warmup() {
        let mut gain = CoarseFilterUpdateGain::new(&default_config(), 250);
        let render_power = [100_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let rsa = RenderSignalAnalyzer::default();

        let mut e = FftData::default();
        e.re.fill(1.0);
        let mut g = FftData::default();

        // Run enough calls to pass the warmup period.
        let size_partitions = 5;
        for _ in 0..=size_partitions * 2 {
            gain.compute(&render_power, &rsa, &e, size_partitions, false, &mut g);
        }

        // After warmup, gain should be non-zero.
        let has_nonzero = g.re.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Gain should be non-zero after warmup");
    }

    #[test]
    fn echo_path_change_resets() {
        let mut gain = CoarseFilterUpdateGain::new(&default_config(), 250);
        gain.call_counter = 100;
        gain.poor_signal_excitation_counter = 100;
        gain.handle_echo_path_change();
        assert_eq!(gain.call_counter, 0);
        assert_eq!(gain.poor_signal_excitation_counter, 0);
    }
}
