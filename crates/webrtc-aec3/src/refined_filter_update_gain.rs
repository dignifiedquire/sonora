//! Refined filter update gain computation.
//!
//! Computes the adaptive gain for the refined (main) adaptive filter using
//! an NLMS-like step size with error-dependent scaling.
//!
//! Ported from
//! `modules/audio_processing/aec3/refined_filter_update_gain.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::RefinedConfiguration;
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::fft_data::FftData;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::subtractor_output::SubtractorOutput;

const H_ERROR_INITIAL: f32 = 10_000.0;
const POOR_EXCITATION_COUNTER_INITIAL: usize = 1000;

/// Computes the adaptive gain for the refined adaptive filter.
pub(crate) struct RefinedFilterUpdateGain {
    config_change_duration_blocks: i32,
    one_by_config_change_duration_blocks: f32,
    current_config: RefinedConfiguration,
    target_config: RefinedConfiguration,
    old_target_config: RefinedConfiguration,
    h_error: [f32; FFT_LENGTH_BY_2_PLUS_1],
    poor_excitation_counter: usize,
    call_counter: usize,
    config_change_counter: i32,
}

impl RefinedFilterUpdateGain {
    pub(crate) fn new(config: &RefinedConfiguration, config_change_duration_blocks: usize) -> Self {
        debug_assert!(config_change_duration_blocks > 0);
        let mut gain = Self {
            config_change_duration_blocks: config_change_duration_blocks as i32,
            one_by_config_change_duration_blocks: 1.0 / config_change_duration_blocks as f32,
            current_config: config.clone(),
            target_config: config.clone(),
            old_target_config: config.clone(),
            h_error: [H_ERROR_INITIAL; FFT_LENGTH_BY_2_PLUS_1],
            poor_excitation_counter: POOR_EXCITATION_COUNTER_INITIAL,
            call_counter: 0,
            config_change_counter: 0,
        };
        gain.set_config(config, true);
        gain
    }

    /// Takes action in the case of a known echo path change.
    pub(crate) fn handle_echo_path_change(&mut self, echo_path_variability: &EchoPathVariability) {
        if echo_path_variability.delay_change != DelayAdjustment::None {
            self.h_error.fill(H_ERROR_INITIAL);
        }
        if !echo_path_variability.gain_change {
            self.poor_excitation_counter = POOR_EXCITATION_COUNTER_INITIAL;
            self.call_counter = 0;
        }
    }

    /// Computes the gain.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn compute(
        &mut self,
        render_power: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        render_signal_analyzer: &RenderSignalAnalyzer,
        subtractor_output: &SubtractorOutput,
        erl: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        size_partitions: usize,
        saturated_capture_signal: bool,
        disallow_leakage_diverged: bool,
        gain_fft: &mut FftData,
    ) {
        let e_refined = &subtractor_output.e_refined_fft;
        let e2_refined = &subtractor_output.e2_refined;
        let e2_coarse = &subtractor_output.e2_coarse;

        self.call_counter += 1;
        self.update_current_config();

        if render_signal_analyzer.poor_signal_excitation() {
            self.poor_excitation_counter = 0;
        }

        // Do not update the filter if the render is not sufficiently excited.
        self.poor_excitation_counter += 1;
        if self.poor_excitation_counter < size_partitions
            || saturated_capture_signal
            || self.call_counter <= size_partitions
        {
            gain_fft.clear();
        } else {
            // mu = H_error / (0.5 * H_error * X2 + n * E2).
            let mut mu = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                if render_power[k] >= self.current_config.noise_gate {
                    mu[k] = self.h_error[k]
                        / (0.5 * self.h_error[k] * render_power[k]
                            + size_partitions as f32 * e2_refined[k]);
                }
            }

            // Avoid updating the filter close to narrow bands.
            render_signal_analyzer.mask_regions_around_narrow_bands(&mut mu);

            // H_error -= 0.5 * mu * X2 * H_error.
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.h_error[k] -= 0.5 * mu[k] * render_power[k] * self.h_error[k];
            }

            // G = mu * E.
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                gain_fft.re[k] = mu[k] * e_refined.re[k];
                gain_fft.im[k] = mu[k] * e_refined.im[k];
            }
        }

        // H_error += factor * erl.
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if e2_refined[k] <= e2_coarse[k] || disallow_leakage_diverged {
                self.h_error[k] += self.current_config.leakage_converged * erl[k];
            } else {
                self.h_error[k] += self.current_config.leakage_diverged * erl[k];
            }
            self.h_error[k] = self.h_error[k].max(self.current_config.error_floor);
            self.h_error[k] = self.h_error[k].min(self.current_config.error_ceil);
        }
    }

    /// Sets a new config.
    pub(crate) fn set_config(&mut self, config: &RefinedConfiguration, immediate_effect: bool) {
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

                self.current_config.leakage_converged = average(
                    self.old_target_config.leakage_converged,
                    self.target_config.leakage_converged,
                    change_factor,
                );
                self.current_config.leakage_diverged = average(
                    self.old_target_config.leakage_diverged,
                    self.target_config.leakage_diverged,
                    change_factor,
                );
                self.current_config.error_floor = average(
                    self.old_target_config.error_floor,
                    self.target_config.error_floor,
                    change_factor,
                );
                self.current_config.error_ceil = average(
                    self.old_target_config.error_ceil,
                    self.target_config.error_ceil,
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

    fn default_config() -> RefinedConfiguration {
        RefinedConfiguration {
            length_blocks: 12,
            leakage_converged: 0.00005,
            leakage_diverged: 0.05,
            error_floor: 0.001,
            error_ceil: 2.0,
            noise_gate: 20075.0,
        }
    }

    #[test]
    fn saturation_zeros_gain() {
        let mut gain = RefinedFilterUpdateGain::new(&default_config(), 250);
        let render_power = [100_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let rsa = RenderSignalAnalyzer::default();
        let output = SubtractorOutput::default();
        let erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut g = FftData::default();

        gain.compute(&render_power, &rsa, &output, &erl, 10, true, false, &mut g);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert_eq!(g.re[k], 0.0);
            assert_eq!(g.im[k], 0.0);
        }
    }

    #[test]
    fn gain_nonzero_after_warmup() {
        let mut gain = RefinedFilterUpdateGain::new(&default_config(), 250);
        let render_power = [100_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let rsa = RenderSignalAnalyzer::default();
        let erl = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut g = FftData::default();

        let mut output = SubtractorOutput::default();
        output.e_refined_fft.re.fill(1.0);
        output.e2_refined.fill(1.0);
        output.e2_coarse.fill(1.0);

        let size_partitions = 5;
        for _ in 0..=size_partitions * 2 {
            gain.compute(
                &render_power,
                &rsa,
                &output,
                &erl,
                size_partitions,
                false,
                false,
                &mut g,
            );
        }

        let has_nonzero = g.re.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Gain should be non-zero after warmup");
    }

    #[test]
    fn echo_path_change_resets_h_error() {
        let mut gain = RefinedFilterUpdateGain::new(&default_config(), 250);

        // Modify h_error.
        gain.h_error.fill(42.0);

        let variability = EchoPathVariability::new(false, DelayAdjustment::NewDetectedDelay, false);
        gain.handle_echo_path_change(&variability);

        // h_error should be reset to initial.
        for &v in &gain.h_error {
            assert!((v - H_ERROR_INITIAL).abs() < 1e-6);
        }
    }

    #[test]
    fn h_error_clamped_to_bounds() {
        let config = default_config();
        let mut gain = RefinedFilterUpdateGain::new(&config, 250);
        let render_power = [100_000.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let rsa = RenderSignalAnalyzer::default();
        let erl = [1e10f32; FFT_LENGTH_BY_2_PLUS_1]; // Large ERL to push error up.
        let mut g = FftData::default();

        let mut output = SubtractorOutput::default();
        output.e_refined_fft.re.fill(1.0);
        output.e2_refined.fill(1.0);
        output.e2_coarse.fill(1.0);

        // Run many iterations to push h_error toward bounds.
        for _ in 0..2000 {
            gain.compute(&render_power, &rsa, &output, &erl, 5, false, false, &mut g);
        }

        // h_error should be clamped at error_ceil.
        for &v in &gain.h_error {
            assert!(v <= config.error_ceil + 1e-6, "h_error={v} > error_ceil");
            assert!(v >= config.error_floor - 1e-6, "h_error={v} < error_floor");
        }
    }
}
