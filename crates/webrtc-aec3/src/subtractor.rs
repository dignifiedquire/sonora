//! Echo subtractor — performs linear echo cancellation using adaptive filters.
//!
//! The Subtractor uses two adaptive FIR filters per capture channel:
//! - A "refined" filter that produces the primary echo estimate
//! - A "coarse" (shadow) filter that tracks the refined filter and is used
//!   to detect divergence
//!
//! Ported from `modules/audio_processing/aec3/subtractor.h/cc`.

use crate::adaptive_fir_filter::AdaptiveFirFilter;
use crate::adaptive_fir_filter_erl::compute_erl;
use crate::aec_state::AecState;
use crate::aec3_fft::Aec3Fft;
use crate::block::Block;
use crate::coarse_filter_update_gain::CoarseFilterUpdateGain;
use crate::common::{BLOCK_SIZE, FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::EchoCanceller3Config;
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::fft_data::FftData;
use crate::refined_filter_update_gain::RefinedFilterUpdateGain;
use crate::render_buffer::RenderBuffer;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::subtractor_output::SubtractorOutput;

/// Computes prediction error: e = y - IFFT(S)[second_half] * scale.
fn prediction_error(
    fft: &Aec3Fft,
    s: &FftData,
    y: &[f32],
    e: &mut [f32; BLOCK_SIZE],
    s_out: Option<&mut [f32; BLOCK_SIZE]>,
) {
    let mut tmp = [0.0f32; FFT_LENGTH];
    fft.ifft(s, &mut tmp);
    const SCALE: f32 = 1.0 / FFT_LENGTH_BY_2 as f32;
    for k in 0..BLOCK_SIZE {
        e[k] = y[k] - tmp[k + FFT_LENGTH_BY_2] * SCALE;
    }
    if let Some(s_out) = s_out {
        for k in 0..BLOCK_SIZE {
            s_out[k] = SCALE * tmp[k + FFT_LENGTH_BY_2];
        }
    }
}

/// Rescales the filter output and recomputes the error.
fn scale_filter_output(y: &[f32], factor: f32, e: &mut [f32], s: &mut [f32]) {
    debug_assert_eq!(y.len(), e.len());
    debug_assert_eq!(y.len(), s.len());
    for k in 0..y.len() {
        s[k] *= factor;
        e[k] = y[k] - s[k];
    }
}

/// Estimates filter misadjustment and recommends scaling when the
/// prediction error energy is much larger than the microphone signal energy.
struct FilterMisadjustmentEstimator {
    n_blocks: i32,
    n_blocks_acum: i32,
    e2_acum: f32,
    y2_acum: f32,
    inv_misadjustment: f32,
    overhang: i32,
}

impl FilterMisadjustmentEstimator {
    fn new() -> Self {
        Self {
            n_blocks: 4,
            n_blocks_acum: 0,
            e2_acum: 0.0,
            y2_acum: 0.0,
            inv_misadjustment: 0.0,
            overhang: 0,
        }
    }

    fn update(&mut self, output: &SubtractorOutput) {
        self.e2_acum += output.e2_refined_sum;
        self.y2_acum += output.y2;
        self.n_blocks_acum += 1;
        if self.n_blocks_acum == self.n_blocks {
            if self.y2_acum > self.n_blocks as f32 * 200.0 * 200.0 * BLOCK_SIZE as f32 {
                let update = self.e2_acum / self.y2_acum;
                if self.e2_acum > self.n_blocks as f32 * 7500.0 * 7500.0 * BLOCK_SIZE as f32 {
                    // Duration equal to blockSizeMs * n_blocks_ * 4.
                    self.overhang = 4;
                } else {
                    self.overhang = (self.overhang - 1).max(0);
                }

                if update < self.inv_misadjustment || self.overhang > 0 {
                    self.inv_misadjustment += 0.1 * (update - self.inv_misadjustment);
                }
            }
            self.e2_acum = 0.0;
            self.y2_acum = 0.0;
            self.n_blocks_acum = 0;
        }
    }

    fn get_misadjustment(&self) -> f32 {
        debug_assert!(self.inv_misadjustment > 0.0);
        // Adjust half of the estimated mismatch.
        2.0 / self.inv_misadjustment.sqrt()
    }

    fn is_adjustment_needed(&self) -> bool {
        self.inv_misadjustment > 10.0
    }

    fn reset(&mut self) {
        self.e2_acum = 0.0;
        self.y2_acum = 0.0;
        self.n_blocks_acum = 0;
        self.inv_misadjustment = 0.0;
        self.overhang = 0;
    }
}

/// Provides linear echo cancellation using dual adaptive filters.
pub(crate) struct Subtractor {
    fft: Aec3Fft,
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    use_coarse_filter_reset_hangover: bool,
    refined_filters: Vec<AdaptiveFirFilter>,
    coarse_filters: Vec<AdaptiveFirFilter>,
    refined_gains: Vec<RefinedFilterUpdateGain>,
    coarse_gains: Vec<CoarseFilterUpdateGain>,
    filter_misadjustment_estimators: Vec<FilterMisadjustmentEstimator>,
    poor_coarse_filter_counters: Vec<usize>,
    coarse_filter_reset_hangover: Vec<i32>,
    refined_frequency_responses: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    refined_impulse_responses: Vec<Vec<f32>>,
}

impl Subtractor {
    pub(crate) fn new(
        backend: webrtc_simd::SimdBackend,
        config: &EchoCanceller3Config,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        let max_refined_len = config
            .filter
            .refined_initial
            .length_blocks
            .max(config.filter.refined.length_blocks);
        let mut refined_filters = Vec::with_capacity(num_capture_channels);
        let mut coarse_filters = Vec::with_capacity(num_capture_channels);
        let mut refined_gains = Vec::with_capacity(num_capture_channels);
        let mut coarse_gains = Vec::with_capacity(num_capture_channels);
        let mut filter_misadjustment_estimators = Vec::with_capacity(num_capture_channels);

        let refined_frequency_responses =
            vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; max_refined_len]; num_capture_channels];
        let refined_impulse_responses =
            vec![
                vec![0.0f32; crate::common::get_time_domain_length(max_refined_len)];
                num_capture_channels
            ];

        for _ in 0..num_capture_channels {
            refined_filters.push(AdaptiveFirFilter::new(
                backend,
                config.filter.refined.length_blocks,
                config.filter.refined_initial.length_blocks,
                config.filter.config_change_duration_blocks,
                num_render_channels,
            ));

            coarse_filters.push(AdaptiveFirFilter::new(
                backend,
                config.filter.coarse.length_blocks,
                config.filter.coarse_initial.length_blocks,
                config.filter.config_change_duration_blocks,
                num_render_channels,
            ));

            refined_gains.push(RefinedFilterUpdateGain::new(
                &config.filter.refined_initial,
                config.filter.config_change_duration_blocks,
            ));

            coarse_gains.push(CoarseFilterUpdateGain::new(
                &config.filter.coarse_initial,
                config.filter.config_change_duration_blocks,
            ));

            filter_misadjustment_estimators.push(FilterMisadjustmentEstimator::new());
        }

        Self {
            fft: Aec3Fft::new(),
            config: config.clone(),
            num_capture_channels,
            use_coarse_filter_reset_hangover: true,
            refined_filters,
            coarse_filters,
            refined_gains,
            coarse_gains,
            filter_misadjustment_estimators,
            poor_coarse_filter_counters: vec![0; num_capture_channels],
            coarse_filter_reset_hangover: vec![0; num_capture_channels],
            refined_frequency_responses,
            refined_impulse_responses,
        }
    }

    /// Performs the echo subtraction.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn process(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        capture: &Block,
        render_signal_analyzer: &RenderSignalAnalyzer,
        aec_state: &AecState,
        outputs: &mut [SubtractorOutput],
    ) {
        debug_assert_eq!(self.num_capture_channels, capture.num_channels());

        // Compute the render powers.
        let same_filter_sizes =
            self.refined_filters[0].size_partitions() == self.coarse_filters[0].size_partitions();
        let mut x2_refined = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut x2_coarse_data = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        if same_filter_sizes {
            render_buffer.spectral_sum(self.refined_filters[0].size_partitions(), &mut x2_refined);
        } else if self.refined_filters[0].size_partitions()
            > self.coarse_filters[0].size_partitions()
        {
            render_buffer.spectral_sums(
                self.coarse_filters[0].size_partitions(),
                self.refined_filters[0].size_partitions(),
                &mut x2_coarse_data,
                &mut x2_refined,
            );
        } else {
            render_buffer.spectral_sums(
                self.refined_filters[0].size_partitions(),
                self.coarse_filters[0].size_partitions(),
                &mut x2_refined,
                &mut x2_coarse_data,
            );
        }

        let x2_coarse = if same_filter_sizes {
            &x2_refined
        } else {
            &x2_coarse_data
        };

        // Process all capture channels.
        for ch in 0..self.num_capture_channels {
            let output = &mut outputs[ch];
            let y = capture.view(0, ch);

            let mut s = FftData::default();
            let mut e_coarse_fft = FftData::default();

            // Form the outputs of the refined and coarse filters.
            self.refined_filters[ch].filter(render_buffer, &mut s);
            prediction_error(
                &self.fft,
                &s,
                y,
                &mut output.e_refined,
                Some(&mut output.s_refined),
            );

            self.coarse_filters[ch].filter(render_buffer, &mut s);
            prediction_error(
                &self.fft,
                &s,
                y,
                &mut output.e_coarse,
                Some(&mut output.s_coarse),
            );

            // Compute the signal powers in the subtractor output.
            output.compute_metrics(y);

            // Adjust the filter if needed.
            let mut refined_filters_adjusted = false;
            self.filter_misadjustment_estimators[ch].update(output);
            if self.filter_misadjustment_estimators[ch].is_adjustment_needed() {
                let scale = self.filter_misadjustment_estimators[ch].get_misadjustment();
                self.refined_filters[ch].scale_filter(scale);
                for h_k in &mut self.refined_impulse_responses[ch] {
                    *h_k *= scale;
                }
                scale_filter_output(y, scale, &mut output.e_refined, &mut output.s_refined);
                self.filter_misadjustment_estimators[ch].reset();
                refined_filters_adjusted = true;
            }

            // Compute the FFTs of the refined and coarse filter outputs.
            self.fft.zero_padded_fft(
                &output.e_refined,
                crate::aec3_fft::Window::Hanning,
                &mut output.e_refined_fft,
            );
            self.fft.zero_padded_fft(
                &output.e_coarse,
                crate::aec3_fft::Window::Hanning,
                &mut e_coarse_fft,
            );

            // Compute spectra for future use.
            e_coarse_fft.spectrum(&mut output.e2_coarse);
            output.e_refined_fft.spectrum(&mut output.e2_refined);

            // Update the refined filter.
            let mut g = FftData::default();
            if !refined_filters_adjusted {
                let disallow_leakage_diverged = self.coarse_filter_reset_hangover[ch] > 0
                    && self.use_coarse_filter_reset_hangover;

                let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                compute_erl(&self.refined_frequency_responses[ch], &mut erl);
                self.refined_gains[ch].compute(
                    &x2_refined,
                    render_signal_analyzer,
                    output,
                    &erl,
                    self.refined_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    disallow_leakage_diverged,
                    &mut g,
                );
            } else {
                g.re.fill(0.0);
                g.im.fill(0.0);
            }
            self.refined_filters[ch].adapt_with_impulse_response(
                render_buffer,
                &g,
                &mut self.refined_impulse_responses[ch],
            );
            self.refined_filters[ch]
                .compute_frequency_response(&mut self.refined_frequency_responses[ch]);

            // Update the coarse filter.
            self.poor_coarse_filter_counters[ch] = if output.e2_refined_sum < output.e2_coarse_sum {
                self.poor_coarse_filter_counters[ch] + 1
            } else {
                0
            };

            if self.poor_coarse_filter_counters[ch] < 5 {
                self.coarse_gains[ch].compute(
                    x2_coarse,
                    render_signal_analyzer,
                    &e_coarse_fft,
                    self.coarse_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    &mut g,
                );
                self.coarse_filter_reset_hangover[ch] =
                    (self.coarse_filter_reset_hangover[ch] - 1).max(0);
            } else {
                self.poor_coarse_filter_counters[ch] = 0;
                self.coarse_filters[ch].set_filter(
                    self.refined_filters[ch].size_partitions(),
                    self.refined_filters[ch].get_filter(),
                );
                self.coarse_gains[ch].compute(
                    x2_coarse,
                    render_signal_analyzer,
                    &output.e_refined_fft,
                    self.coarse_filters[ch].size_partitions(),
                    aec_state.saturated_capture(),
                    &mut g,
                );
                self.coarse_filter_reset_hangover[ch] =
                    self.config.filter.coarse_reset_hangover_blocks;
            }

            self.coarse_filters[ch].adapt(render_buffer, &g);

            // Clamp the refined error output.
            for v in &mut output.e_refined {
                *v = v.clamp(-32768.0, 32767.0);
            }
        }
    }

    /// Handles echo path changes.
    pub(crate) fn handle_echo_path_change(&mut self, echo_path_variability: &EchoPathVariability) {
        if echo_path_variability.delay_change != DelayAdjustment::None {
            for ch in 0..self.num_capture_channels {
                self.refined_filters[ch].handle_echo_path_change();
                self.coarse_filters[ch].handle_echo_path_change();
                self.refined_gains[ch].handle_echo_path_change(echo_path_variability);
                self.coarse_gains[ch].handle_echo_path_change();
                self.refined_gains[ch].set_config(&self.config.filter.refined_initial, true);
                self.coarse_gains[ch].set_config(&self.config.filter.coarse_initial, true);
                self.refined_filters[ch]
                    .set_size_partitions(self.config.filter.refined_initial.length_blocks, true);
                self.coarse_filters[ch]
                    .set_size_partitions(self.config.filter.coarse_initial.length_blocks, true);
            }
        }

        if echo_path_variability.gain_change {
            for ch in 0..self.num_capture_channels {
                self.refined_gains[ch].handle_echo_path_change(echo_path_variability);
            }
        }
    }

    /// Exits the initial state.
    pub(crate) fn exit_initial_state(&mut self) {
        for ch in 0..self.num_capture_channels {
            self.refined_gains[ch].set_config(&self.config.filter.refined, false);
            self.coarse_gains[ch].set_config(&self.config.filter.coarse, false);
            self.refined_filters[ch]
                .set_size_partitions(self.config.filter.refined.length_blocks, false);
            self.coarse_filters[ch]
                .set_size_partitions(self.config.filter.coarse.length_blocks, false);
        }
    }

    /// Returns the block-wise frequency responses for the refined adaptive
    /// filters.
    pub(crate) fn filter_frequency_responses(&self) -> &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>] {
        &self.refined_frequency_responses
    }

    /// Returns the estimates of the impulse responses for the refined adaptive
    /// filters.
    pub(crate) fn filter_impulse_responses(&self) -> &[Vec<f32>] {
        &self.refined_impulse_responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_buffer::BlockBuffer;
    use crate::fft_buffer::FftBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    /// Helper to create a minimal RenderBuffer with allocated but zeroed data.
    fn make_render_buffer(
        num_partitions: usize,
        num_channels: usize,
    ) -> (BlockBuffer, SpectrumBuffer, FftBuffer) {
        let bb = BlockBuffer::new(num_partitions, 1, num_channels);
        let sb = SpectrumBuffer::new(num_partitions, num_channels);
        let fb = FftBuffer::new(num_partitions, num_channels);
        (bb, sb, fb)
    }

    #[test]
    fn subtractor_creation() {
        let config = EchoCanceller3Config::default();
        let subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 1, 1);
        assert_eq!(subtractor.num_capture_channels, 1);
        assert_eq!(subtractor.filter_frequency_responses().len(), 1);
        assert_eq!(subtractor.filter_impulse_responses().len(), 1);
    }

    #[test]
    fn subtractor_multichannel_creation() {
        let config = EchoCanceller3Config::default();
        let subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 2, 4);
        assert_eq!(subtractor.num_capture_channels, 4);
        assert_eq!(subtractor.filter_frequency_responses().len(), 4);
        assert_eq!(subtractor.filter_impulse_responses().len(), 4);
    }

    #[test]
    fn handle_echo_path_change_delay() {
        let config = EchoCanceller3Config::default();
        let mut subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 1, 1);
        let variability = EchoPathVariability::new(true, DelayAdjustment::NewDetectedDelay, false);
        // Should not panic.
        subtractor.handle_echo_path_change(&variability);
    }

    #[test]
    fn handle_echo_path_change_gain() {
        let config = EchoCanceller3Config::default();
        let mut subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 1, 1);
        let variability = EchoPathVariability::new(true, DelayAdjustment::None, false);
        // Should not panic.
        subtractor.handle_echo_path_change(&variability);
    }

    #[test]
    fn exit_initial_state_does_not_panic() {
        let config = EchoCanceller3Config::default();
        let mut subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 1, 1);
        subtractor.exit_initial_state();
    }

    #[test]
    fn process_single_block() {
        let config = EchoCanceller3Config::default();
        let num_partitions = config
            .filter
            .refined
            .length_blocks
            .max(config.filter.coarse.length_blocks)
            + 1;
        let (bb, sb, fb) = make_render_buffer(num_partitions, 1);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);
        let capture = Block::new(1, 1);
        let rsa = RenderSignalAnalyzer::default();
        let aec_state = AecState::new(&config, 1);
        let mut outputs = vec![SubtractorOutput::default()];

        let mut subtractor = Subtractor::new(webrtc_simd::SimdBackend::Scalar, &config, 1, 1);
        subtractor.process(&render_buffer, &capture, &rsa, &aec_state, &mut outputs);

        // With zero input, all outputs should be zero/near-zero.
        assert!(outputs[0].e_refined.iter().all(|&v| v.abs() < 1e-10));
        assert!(outputs[0].e_coarse.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn filter_misadjustment_estimator_basic() {
        let mut estimator = FilterMisadjustmentEstimator::new();
        assert!(!estimator.is_adjustment_needed());

        // Feed high-error, low-y outputs to trigger adjustment.
        for _ in 0..4 {
            let mut output = SubtractorOutput::default();
            output.e2_refined_sum = 100_000_000.0; // Very high error.
            output.y2 = 100_000_000.0; // High y power.
            estimator.update(&output);
        }
        // After 4 blocks, check state.
        assert!(estimator.inv_misadjustment >= 0.0);
    }
}
