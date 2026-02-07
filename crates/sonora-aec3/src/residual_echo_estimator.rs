//! Residual echo power estimation.
//!
//! Ported from `modules/audio_processing/aec3/residual_echo_estimator.h/cc`.

use crate::aec_state::AecState;
use crate::common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::EchoCanceller3Config;
use crate::render_buffer::RenderBuffer;
use crate::reverb_model::ReverbModel;
use crate::spectrum_buffer::SpectrumBuffer;

const DEFAULT_TRANSPARENT_MODE_GAIN: f32 = 0.01;

/// Estimates the residual echo power based on ERLE and the linear power
/// estimate.
fn linear_estimate(
    s2_linear: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erle: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    debug_assert_eq!(s2_linear.len(), erle.len());
    debug_assert_eq!(s2_linear.len(), r2.len());
    for ch in 0..r2.len() {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            debug_assert!(erle[ch][k] > 0.0);
            r2[ch][k] = s2_linear[ch][k] / erle[ch][k];
        }
    }
}

/// Estimates the residual echo power based on the echo path gain.
fn non_linear_estimate(
    echo_path_gain: f32,
    x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
) {
    for ch in 0..r2.len() {
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            r2[ch][k] = x2[k] * echo_path_gain;
        }
    }
}

/// Applies a soft noise gate to the echo generating power.
fn apply_noise_gate(config: &crate::config::EchoModel, x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        if config.noise_gate_power > x2[k] {
            x2[k] = (x2[k] - config.noise_gate_slope * (config.noise_gate_power - x2[k])).max(0.0);
        }
    }
}

/// Computes the render indexes to analyze around the delay.
fn get_render_indexes_to_analyze(
    spectrum_buffer: &SpectrumBuffer,
    echo_model: &crate::config::EchoModel,
    filter_delay_blocks: i32,
) -> (usize, usize) {
    let window_start = (filter_delay_blocks - echo_model.render_pre_window_size as i32).max(0);
    let window_end = filter_delay_blocks + echo_model.render_post_window_size as i32;
    let idx_start = spectrum_buffer
        .index
        .offset_index(spectrum_buffer.index.read, window_start);
    let idx_stop = spectrum_buffer
        .index
        .offset_index(spectrum_buffer.index.read, window_end + 1);
    (idx_start, idx_stop)
}

/// Estimates the echo generating signal power as gated maximal power over a
/// time window.
fn echo_generating_power(
    num_render_channels: usize,
    spectrum_buffer: &SpectrumBuffer,
    echo_model: &crate::config::EchoModel,
    filter_delay_blocks: i32,
    x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let (idx_start, idx_stop) =
        get_render_indexes_to_analyze(spectrum_buffer, echo_model, filter_delay_blocks);

    x2.fill(0.0);
    if num_render_channels == 1 {
        let mut k = idx_start;
        while k != idx_stop {
            for j in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[j] = x2[j].max(spectrum_buffer.buffer[k][0][j]);
            }
            k = spectrum_buffer.index.inc_index(k);
        }
    } else {
        let mut k = idx_start;
        while k != idx_stop {
            let mut render_power = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for ch in 0..num_render_channels {
                let channel_power = &spectrum_buffer.buffer[k][ch];
                for j in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    render_power[j] += channel_power[j];
                }
            }
            for j in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2[j] = x2[j].max(render_power[j]);
            }
            k = spectrum_buffer.index.inc_index(k);
        }
    }
}

/// Estimates the residual echo power after echo cancellation.
pub(crate) struct ResidualEchoEstimator {
    config: EchoCanceller3Config,
    num_render_channels: usize,
    early_reflections_transparent_mode_gain: f32,
    late_reflections_transparent_mode_gain: f32,
    early_reflections_general_gain: f32,
    late_reflections_general_gain: f32,
    erle_onset_compensation_in_dominant_nearend: bool,
    x2_noise_floor: [f32; FFT_LENGTH_BY_2_PLUS_1],
    x2_noise_floor_counter: [i32; FFT_LENGTH_BY_2_PLUS_1],
    echo_reverb: ReverbModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReverbType {
    Linear,
    NonLinear,
}

impl ResidualEchoEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_render_channels: usize) -> Self {
        // Field trials default to not enabled — use default gains.
        let early_reflections_general_gain = config.ep_strength.default_gain;
        let late_reflections_general_gain = config.ep_strength.default_gain;
        let erle_onset_compensation_in_dominant_nearend = config
            .ep_strength
            .erle_onset_compensation_in_dominant_nearend;

        let mut estimator = Self {
            config: config.clone(),
            num_render_channels,
            early_reflections_transparent_mode_gain: DEFAULT_TRANSPARENT_MODE_GAIN,
            late_reflections_transparent_mode_gain: DEFAULT_TRANSPARENT_MODE_GAIN,
            early_reflections_general_gain,
            late_reflections_general_gain,
            erle_onset_compensation_in_dominant_nearend,
            x2_noise_floor: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            x2_noise_floor_counter: [0; FFT_LENGTH_BY_2_PLUS_1],
            echo_reverb: ReverbModel::new(),
        };
        estimator.reset();
        estimator
    }

    /// Estimates the residual echo power.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn estimate(
        &mut self,
        aec_state: &AecState,
        render_buffer: &RenderBuffer<'_>,
        _capture: &[[f32; FFT_LENGTH_BY_2]],
        _linear_aec_output: &[[f32; FFT_LENGTH_BY_2]],
        s2_linear: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        _e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        dominant_nearend: bool,
        r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
        r2_unbounded: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        debug_assert_eq!(r2.len(), y2.len());
        debug_assert_eq!(r2.len(), s2_linear.len());

        let num_capture_channels = r2.len();

        // Estimate the power of stationary noise in the render signal.
        self.update_render_noise_power(render_buffer);

        // NeuralResidualEchoEstimator is skipped (not ported).

        // Estimate the residual echo power.
        if aec_state.usable_linear_estimate() {
            // When there is saturated echo, assume the same spectral content
            // as is present in the microphone signal.
            if aec_state.saturated_echo() {
                for ch in 0..num_capture_channels {
                    r2[ch].copy_from_slice(&y2[ch]);
                    r2_unbounded[ch].copy_from_slice(&y2[ch]);
                }
            } else {
                let onset_compensated =
                    self.erle_onset_compensation_in_dominant_nearend || !dominant_nearend;
                linear_estimate(s2_linear, aec_state.erle(onset_compensated), r2);
                linear_estimate(s2_linear, aec_state.erle_unbounded(), r2_unbounded);
            }

            self.update_reverb(
                ReverbType::Linear,
                aec_state,
                render_buffer,
                dominant_nearend,
            );
            self.add_reverb(r2);
            self.add_reverb(r2_unbounded);
        } else {
            let echo_path_gain = self.get_echo_path_gain(aec_state, true);

            // When there is saturated echo, assume the same spectral content
            // as is present in the microphone signal.
            if aec_state.saturated_echo() {
                for ch in 0..num_capture_channels {
                    r2[ch].copy_from_slice(&y2[ch]);
                    r2_unbounded[ch].copy_from_slice(&y2[ch]);
                }
            } else {
                // Estimate the echo generating signal power.
                let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                echo_generating_power(
                    self.num_render_channels,
                    render_buffer.get_spectrum_buffer(),
                    &self.config.echo_model,
                    aec_state.min_direct_path_filter_delay(),
                    &mut x2,
                );
                if !aec_state.use_stationarity_properties() {
                    apply_noise_gate(&self.config.echo_model, &mut x2);
                }

                // Subtract the stationary noise power.
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    x2[k] -= self.config.echo_model.stationary_gate_slope * self.x2_noise_floor[k];
                    x2[k] = x2[k].max(0.0);
                }

                non_linear_estimate(echo_path_gain, &x2, r2);
                non_linear_estimate(echo_path_gain, &x2, r2_unbounded);
            }

            if self.config.echo_model.model_reverb_in_nonlinear_mode
                && !aec_state.transparent_mode_active()
            {
                self.update_reverb(
                    ReverbType::NonLinear,
                    aec_state,
                    render_buffer,
                    dominant_nearend,
                );
                self.add_reverb(r2);
                self.add_reverb(r2_unbounded);
            }
        }

        if aec_state.use_stationarity_properties() {
            let mut residual_scaling = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            aec_state.get_residual_echo_scaling(&mut residual_scaling);
            for ch in 0..num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    r2[ch][k] *= residual_scaling[k];
                    r2_unbounded[ch][k] *= residual_scaling[k];
                }
            }
        }
    }

    fn reset(&mut self) {
        self.echo_reverb.reset();
        self.x2_noise_floor_counter
            .fill(self.config.echo_model.noise_floor_hold as i32);
        self.x2_noise_floor
            .fill(self.config.echo_model.min_noise_floor_power);
    }

    fn update_render_noise_power(&mut self, render_buffer: &RenderBuffer<'_>) {
        let x2 = render_buffer.spectrum(0);
        let render_power: Vec<f32>;
        let render_power_ref: &[f32];

        if self.num_render_channels > 1 {
            let mut power_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for ch in 0..self.num_render_channels {
                let channel_power = &x2[ch];
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    power_data[k] += channel_power[k];
                }
            }
            render_power = power_data.to_vec();
            render_power_ref = &render_power;
        } else {
            render_power_ref = &x2[0];
        }

        // Estimate the stationary noise power in a minimum statistics manner.
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if render_power_ref[k] < self.x2_noise_floor[k] {
                // Decrease rapidly.
                self.x2_noise_floor[k] = render_power_ref[k];
                self.x2_noise_floor_counter[k] = 0;
            } else {
                // Increase in a delayed, leaky manner.
                if self.x2_noise_floor_counter[k] >= self.config.echo_model.noise_floor_hold as i32
                {
                    self.x2_noise_floor[k] = (self.x2_noise_floor[k] * 1.1)
                        .max(self.config.echo_model.min_noise_floor_power);
                } else {
                    self.x2_noise_floor_counter[k] += 1;
                }
            }
        }
    }

    fn update_reverb(
        &mut self,
        reverb_type: ReverbType,
        aec_state: &AecState,
        render_buffer: &RenderBuffer<'_>,
        dominant_nearend: bool,
    ) {
        // Choose reverb partition based on echo power model type.
        let first_reverb_partition = match reverb_type {
            ReverbType::Linear => aec_state.filter_length_blocks() as i32 + 1,
            ReverbType::NonLinear => aec_state.min_direct_path_filter_delay() + 1,
        };

        // Compute render power for the reverb.
        let x2 = render_buffer.spectrum(first_reverb_partition);
        let render_power: [f32; FFT_LENGTH_BY_2_PLUS_1];
        let render_power_ref: &[f32];

        if self.num_render_channels > 1 {
            let mut power_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            for ch in 0..self.num_render_channels {
                let channel_power = &x2[ch];
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    power_data[k] += channel_power[k];
                }
            }
            render_power = power_data;
            render_power_ref = &render_power;
        } else {
            render_power_ref = &x2[0];
        }

        // Update the reverb estimate.
        let reverb_decay = aec_state.reverb_decay(dominant_nearend);
        match reverb_type {
            ReverbType::Linear => {
                self.echo_reverb.update_reverb(
                    render_power_ref,
                    aec_state.get_reverb_frequency_response(),
                    reverb_decay,
                );
            }
            ReverbType::NonLinear => {
                let echo_path_gain = self.get_echo_path_gain(aec_state, false);
                self.echo_reverb.update_reverb_no_freq_shaping(
                    render_power_ref,
                    echo_path_gain,
                    reverb_decay,
                );
            }
        }
    }

    fn add_reverb(&self, r2: &mut [[f32; FFT_LENGTH_BY_2_PLUS_1]]) {
        let reverb_power = self.echo_reverb.reverb();
        for ch in 0..r2.len() {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                r2[ch][k] += reverb_power[k];
            }
        }
    }

    fn get_echo_path_gain(&self, aec_state: &AecState, gain_for_early_reflections: bool) -> f32 {
        let gain_amplitude = if aec_state.transparent_mode_active() {
            if gain_for_early_reflections {
                self.early_reflections_transparent_mode_gain
            } else {
                self.late_reflections_transparent_mode_gain
            }
        } else if gain_for_early_reflections {
            self.early_reflections_general_gain
        } else {
            self.late_reflections_general_gain
        };
        gain_amplitude * gain_amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::block_buffer::BlockBuffer;
    use crate::fft_buffer::FftBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    fn make_render_buffer(
        size: usize,
        num_channels: usize,
    ) -> (BlockBuffer, SpectrumBuffer, FftBuffer) {
        let block_buffer = BlockBuffer::new(size, 1, num_channels);
        let spectrum_buffer = SpectrumBuffer::new(size, num_channels);
        let fft_buffer = FftBuffer::new(size, num_channels);
        (block_buffer, spectrum_buffer, fft_buffer)
    }

    #[test]
    fn creation_and_reset() {
        let config = EchoCanceller3Config::default();
        let estimator = ResidualEchoEstimator::new(&config, 1);
        // Noise floor should be initialized.
        for &v in &estimator.x2_noise_floor {
            assert_eq!(v, config.echo_model.min_noise_floor_power);
        }
    }

    #[test]
    fn nonlinear_estimate_produces_output() {
        let config = EchoCanceller3Config::default();
        let mut estimator = ResidualEchoEstimator::new(&config, 1);
        let aec_state = AecState::new(&config, 1);

        let size = 20;
        let (bb, sb, fb) = make_render_buffer(size, 1);
        let rb = RenderBuffer::new(&bb, &sb, &fb);

        let capture = [[0.0f32; FFT_LENGTH_BY_2]];
        let linear_out = [[0.0f32; FFT_LENGTH_BY_2]];
        let s2_linear = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        let y2 = [[1.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        let e2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        let mut r2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        let mut r2_unbounded = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]];

        // AecState defaults to usable_linear_estimate=false, so nonlinear path.
        estimator.estimate(
            &aec_state,
            &rb,
            &capture,
            &linear_out,
            &s2_linear,
            &y2,
            &e2,
            false,
            &mut r2,
            &mut r2_unbounded,
        );
        // R2 should be computed (may be zero since render buffer is empty).
        // The test mainly verifies no panics.
    }

    #[test]
    fn noise_gate_reduces_power() {
        let config = EchoCanceller3Config::default();
        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        // Set power below the noise gate.
        x2.fill(config.echo_model.noise_gate_power * 0.5);
        let original = x2;
        apply_noise_gate(&config.echo_model, &mut x2);
        // After noise gating, power should be reduced.
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert!(x2[k] <= original[k]);
        }
    }

    #[test]
    fn echo_path_gain_transparent_vs_normal() {
        let config = EchoCanceller3Config::default();
        let estimator = ResidualEchoEstimator::new(&config, 1);
        let aec_state = AecState::new(&config, 1);

        let normal_gain = estimator.get_echo_path_gain(&aec_state, true);
        // normal aec_state has transparent_mode_active=false
        let expected = config.ep_strength.default_gain * config.ep_strength.default_gain;
        assert!((normal_gain - expected).abs() < 1e-6);
    }
}
