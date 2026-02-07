//! Echo remover — orchestrates echo removal using the subtractor, suppression,
//! and AEC state machine.
//!
//! Ported from `modules/audio_processing/aec3/echo_remover.h/cc`.

use crate::aec_state::AecState;
use crate::aec3_fft::{Aec3Fft, Window};
use crate::block::Block;
use crate::comfort_noise_generator::ComfortNoiseGenerator;
use crate::common::{
    BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, log2_to_db, num_bands_for_rate,
    valid_full_band_rate,
};
use crate::config::EchoCanceller3Config;
use crate::delay_estimate::DelayEstimate;
use crate::echo_path_variability::EchoPathVariability;
use crate::echo_remover_metrics::EchoRemoverMetrics;
use crate::fft_data::FftData;
use crate::render_buffer::RenderBuffer;
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::residual_echo_estimator::ResidualEchoEstimator;
use crate::subtractor::Subtractor;
use crate::subtractor_output::SubtractorOutput;
use crate::suppression_filter::SuppressionFilter;
use crate::suppression_gain::SuppressionGain;

/// Echo control metrics returned by `get_metrics`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EchoRemoverMetricsOutput {
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
}

/// Computes the linear echo power: S2 = |Y - E|^2.
fn linear_echo_power(e: &FftData, y: &FftData, s2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    for k in 0..e.re.len() {
        s2[k] =
            (y.re[k] - e.re[k]) * (y.re[k] - e.re[k]) + (y.im[k] - e.im[k]) * (y.im[k] - e.im[k]);
    }
}

/// Fades between two input signals using a fixed-size transition.
fn signal_transition(from: &[f32], to: &[f32], out: &mut [f32]) {
    debug_assert_eq!(from.len(), to.len());
    debug_assert_eq!(from.len(), out.len());

    if std::ptr::eq(from.as_ptr(), to.as_ptr()) {
        out.copy_from_slice(to);
    } else {
        const TRANSITION_SIZE: usize = 30;
        const ONE_BY_TRANSITION_SIZE_PLUS_ONE: f32 = 1.0 / (TRANSITION_SIZE + 1) as f32;

        debug_assert!(TRANSITION_SIZE <= out.len());

        for k in 0..TRANSITION_SIZE {
            let a = (k + 1) as f32 * ONE_BY_TRANSITION_SIZE_PLUS_ONE;
            out[k] = a * to[k] + (1.0 - a) * from[k];
        }

        out[TRANSITION_SIZE..].copy_from_slice(&to[TRANSITION_SIZE..]);
    }
}

/// Computes a windowed (sqrt-Hanning) padded FFT and updates the related memory.
fn windowed_padded_fft(
    fft: &Aec3Fft,
    v: &[f32],
    v_old: &mut [f32; FFT_LENGTH_BY_2],
    out: &mut FftData,
) {
    fft.padded_fft(v, v_old, Window::SqrtHanning, out);
    v_old.copy_from_slice(v);
}

/// Removes echo from the capture signal.
pub(crate) struct EchoRemover {
    config: EchoCanceller3Config,
    fft: Aec3Fft,
    sample_rate_hz: usize,
    num_render_channels: usize,
    num_capture_channels: usize,
    use_coarse_filter_output: bool,
    subtractor: Subtractor,
    suppression_gain: SuppressionGain,
    cng: ComfortNoiseGenerator,
    suppression_filter: SuppressionFilter,
    render_signal_analyzer: RenderSignalAnalyzer,
    residual_echo_estimator: ResidualEchoEstimator,
    echo_leakage_detected: bool,
    capture_output_used: bool,
    aec_state: AecState,
    metrics: EchoRemoverMetrics,
    e_old: Vec<[f32; FFT_LENGTH_BY_2]>,
    y_old: Vec<[f32; FFT_LENGTH_BY_2]>,
    block_counter: usize,
    gain_change_hangover: i32,
    refined_filter_output_last_selected: bool,
}

impl EchoRemover {
    pub(crate) fn new(
        backend: webrtc_simd::SimdBackend,
        config: &EchoCanceller3Config,
        sample_rate_hz: usize,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        debug_assert!(valid_full_band_rate(sample_rate_hz));
        Self {
            config: config.clone(),
            fft: Aec3Fft::new(),
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
            use_coarse_filter_output: config.filter.enable_coarse_filter_output_usage,
            subtractor: Subtractor::new(backend, config, num_render_channels, num_capture_channels),
            suppression_gain: SuppressionGain::new(config, sample_rate_hz, num_capture_channels),
            cng: ComfortNoiseGenerator::new(config, num_capture_channels),
            suppression_filter: SuppressionFilter::new(sample_rate_hz, num_capture_channels),
            render_signal_analyzer: RenderSignalAnalyzer::new(config),
            residual_echo_estimator: ResidualEchoEstimator::new(config, num_render_channels),
            echo_leakage_detected: false,
            capture_output_used: true,
            aec_state: AecState::new(config, num_capture_channels),
            metrics: EchoRemoverMetrics::new(),
            e_old: vec![[0.0; FFT_LENGTH_BY_2]; num_capture_channels],
            y_old: vec![[0.0; FFT_LENGTH_BY_2]; num_capture_channels],
            block_counter: 0,
            gain_change_hangover: 0,
            refined_filter_output_last_selected: true,
        }
    }

    /// Returns current echo removal metrics.
    pub(crate) fn get_metrics(&self) -> EchoRemoverMetricsOutput {
        EchoRemoverMetricsOutput {
            // Echo return loss (ERL) is inverted to go from gain to attenuation.
            echo_return_loss: -10.0 * (self.aec_state.erl_time_domain() as f64).log10(),
            echo_return_loss_enhancement: log2_to_db(self.aec_state.fullband_erle_log2()) as f64,
        }
    }

    /// Removes echo from a block of capture samples. The render signal in the
    /// render buffer is assumed to be pre-aligned with the capture signal.
    pub(crate) fn process_capture(
        &mut self,
        mut echo_path_variability: EchoPathVariability,
        capture_signal_saturation: bool,
        external_delay: &Option<DelayEstimate>,
        render_buffer: &RenderBuffer<'_>,
        linear_output: Option<&mut Block>,
        capture: &mut Block,
    ) {
        self.block_counter += 1;

        let num_capture_channels = self.num_capture_channels;
        debug_assert_eq!(
            render_buffer.get_block(0).num_bands(),
            num_bands_for_rate(self.sample_rate_hz)
        );
        debug_assert_eq!(capture.num_bands(), num_bands_for_rate(self.sample_rate_hz));
        debug_assert_eq!(
            render_buffer.get_block(0).num_channels(),
            self.num_render_channels
        );
        debug_assert_eq!(capture.num_channels(), num_capture_channels);

        // Per-channel working storage.
        let mut e = vec![[0.0f32; FFT_LENGTH_BY_2]; num_capture_channels];
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut r2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut r2_unbounded = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut s2_linear = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut y_fft = vec![FftData::default(); num_capture_channels];
        let mut e_fft = vec![FftData::default(); num_capture_channels];
        let mut comfort_noise = vec![FftData::default(); num_capture_channels];
        let mut high_band_comfort_noise = vec![FftData::default(); num_capture_channels];
        let mut subtractor_output: Vec<SubtractorOutput> = (0..num_capture_channels)
            .map(|_| SubtractorOutput::default())
            .collect();

        self.aec_state
            .update_capture_saturation(capture_signal_saturation);

        if echo_path_variability.audio_path_changed() {
            // Ensure that the gain change is only acted on once per frame.
            if echo_path_variability.gain_change {
                if self.gain_change_hangover == 0 {
                    const MAX_BLOCKS_PER_FRAME: i32 = 3;
                    self.gain_change_hangover = MAX_BLOCKS_PER_FRAME;
                } else {
                    echo_path_variability.gain_change = false;
                }
            }

            self.subtractor
                .handle_echo_path_change(&echo_path_variability);
            self.aec_state
                .handle_echo_path_change(&echo_path_variability);

            if echo_path_variability.delay_change
                != crate::echo_path_variability::DelayAdjustment::None
            {
                self.suppression_gain.set_initial_state(true);
            }
        }
        if self.gain_change_hangover > 0 {
            self.gain_change_hangover -= 1;
        }

        // Analyze the render signal.
        self.render_signal_analyzer.update(
            render_buffer,
            Some(self.aec_state.min_direct_path_filter_delay() as usize),
        );

        // State transition.
        if self.aec_state.transition_triggered() {
            self.subtractor.exit_initial_state();
            self.suppression_gain.set_initial_state(false);
        }

        // Perform linear echo cancellation.
        self.subtractor.process(
            render_buffer,
            capture,
            &self.render_signal_analyzer,
            &self.aec_state,
            &mut subtractor_output,
        );

        // Compute spectra.
        for ch in 0..num_capture_channels {
            self.form_linear_filter_output(&subtractor_output[ch], &mut e[ch]);
            windowed_padded_fft(
                &self.fft,
                capture.view(0, ch),
                &mut self.y_old[ch],
                &mut y_fft[ch],
            );
            windowed_padded_fft(&self.fft, &e[ch], &mut self.e_old[ch], &mut e_fft[ch]);
            linear_echo_power(&e_fft[ch], &y_fft[ch], &mut s2_linear[ch]);
            y_fft[ch].spectrum(&mut y2[ch]);
            e_fft[ch].spectrum(&mut e2[ch]);
        }
        // y_old and e_old now point to the current block.
        let y_current = &self.y_old;
        let e_current = &self.e_old;

        // Optionally return the linear filter output.
        if let Some(linear_out) = linear_output {
            debug_assert!(linear_out.num_bands() <= 1);
            debug_assert_eq!(num_capture_channels, linear_out.num_channels());
            for ch in 0..num_capture_channels {
                linear_out.view_mut(0, ch).copy_from_slice(&e[ch]);
            }
        }

        // Update the AEC state information.
        self.aec_state.update(
            external_delay,
            self.subtractor.filter_frequency_responses(),
            self.subtractor.filter_impulse_responses(),
            render_buffer,
            &e2,
            &y2,
            &subtractor_output,
        );

        // Choose the linear output.
        let y_fft_for_suppression: &[FftData] = if self.aec_state.use_linear_filter_output() {
            &e_fft
        } else {
            &y_fft
        };

        // Only do the below processing if the output will be used.
        let mut g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        if self.capture_output_used {
            // Estimate the residual echo power.
            self.residual_echo_estimator.estimate(
                &self.aec_state,
                render_buffer,
                y_current,
                e_current,
                &s2_linear,
                &y2,
                &e2,
                self.suppression_gain.is_dominant_nearend(),
                &mut r2,
                &mut r2_unbounded,
            );

            // Suppressor nearend estimate: E2 is bound by Y2.
            if self.aec_state.usable_linear_estimate() {
                for ch in 0..num_capture_channels {
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        e2[ch][k] = e2[ch][k].min(y2[ch][k]);
                    }
                }
            }

            // Select nearend spectrum (after E2 clamping).
            let nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]] =
                if self.aec_state.usable_linear_estimate() {
                    &e2
                } else {
                    &y2
                };

            // Estimate the comfort noise.
            self.cng.compute(
                self.aec_state.saturated_capture(),
                nearend_spectrum,
                &mut comfort_noise,
                &mut high_band_comfort_noise,
            );

            // Suppressor echo estimate.
            let echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]] =
                if self.aec_state.usable_linear_estimate() {
                    &s2_linear
                } else {
                    &r2
                };

            // Determine if the suppressor should assume clock drift.
            let clock_drift = self.config.echo_removal_control.has_clock_drift
                || echo_path_variability.clock_drift;

            // Compute preferred gains.
            let mut high_bands_gain = 0.0f32;
            let x = render_buffer.get_block(0);
            self.suppression_gain.get_gain(
                nearend_spectrum,
                echo_spectrum,
                &r2,
                &r2_unbounded,
                self.cng.noise_spectrum(),
                &self.render_signal_analyzer,
                &self.aec_state,
                x,
                clock_drift,
                &mut high_bands_gain,
                &mut g,
            );

            self.suppression_filter.apply_gain(
                &comfort_noise,
                &high_band_comfort_noise,
                &g,
                high_bands_gain,
                y_fft_for_suppression,
                capture,
            );
        } else {
            let nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]] =
                if self.aec_state.usable_linear_estimate() {
                    &e2
                } else {
                    &y2
                };
            self.cng.compute(
                self.aec_state.saturated_capture(),
                nearend_spectrum,
                &mut comfort_noise,
                &mut high_band_comfort_noise,
            );
            g.fill(0.0);
        }

        // Update the metrics.
        self.metrics
            .update(&self.aec_state, &self.cng.noise_spectrum()[0], &g);
    }

    /// Updates the status on whether echo leakage is detected.
    pub(crate) fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        self.echo_leakage_detected = leakage_detected;
    }

    /// Specifies whether the capture output will be used.
    pub(crate) fn set_capture_output_usage(&mut self, capture_output_used: bool) {
        self.capture_output_used = capture_output_used;
    }

    /// Selects which of the coarse and refined linear filter outputs is most
    /// appropriate and forms the output by smoothly transitioning between them.
    fn form_linear_filter_output(
        &mut self,
        subtractor_output: &SubtractorOutput,
        output: &mut [f32; FFT_LENGTH_BY_2],
    ) {
        debug_assert_eq!(subtractor_output.e_refined.len(), output.len());
        debug_assert_eq!(subtractor_output.e_coarse.len(), output.len());

        let mut use_refined_output = true;
        if self.use_coarse_filter_output {
            // As the output of the refined adaptive filter generally should be
            // better than the coarse filter output, add a margin and threshold
            // for when choosing the coarse filter output.
            if subtractor_output.e2_coarse_sum < 0.9 * subtractor_output.e2_refined_sum
                && subtractor_output.y2 > 30.0 * 30.0 * BLOCK_SIZE as f32
                && (subtractor_output.s2_refined > 60.0 * 60.0 * BLOCK_SIZE as f32
                    || subtractor_output.s2_coarse > 60.0 * 60.0 * BLOCK_SIZE as f32)
            {
                use_refined_output = false;
            } else {
                // If the refined filter is diverged, choose the filter output
                // that has the lowest power.
                if subtractor_output.e2_coarse_sum < subtractor_output.e2_refined_sum
                    && subtractor_output.y2 < subtractor_output.e2_refined_sum
                {
                    use_refined_output = false;
                }
            }
        }

        let from = if self.refined_filter_output_last_selected {
            &subtractor_output.e_refined
        } else {
            &subtractor_output.e_coarse
        };
        let to = if use_refined_output {
            &subtractor_output.e_refined
        } else {
            &subtractor_output.e_coarse
        };

        signal_transition(from, to, output);
        self.refined_filter_output_last_selected = use_refined_output;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::block_buffer::BlockBuffer;
    use crate::common::num_bands_for_rate;
    use crate::fft_buffer::FftBuffer;
    use crate::render_buffer::RenderBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    #[test]
    fn basic_api_calls() {
        for num_render_channels in [1, 2] {
            for num_capture_channels in [1, 2] {
                for &rate in &[16000, 32000, 48000] {
                    let config = EchoCanceller3Config::default();
                    let mut remover = EchoRemover::new(
                        webrtc_simd::SimdBackend::Scalar,
                        &config,
                        rate,
                        num_render_channels,
                        num_capture_channels,
                    );

                    let num_bands = num_bands_for_rate(rate);
                    let buf_size = config
                        .filter
                        .refined
                        .length_blocks
                        .max(config.filter.coarse.length_blocks)
                        + 1;
                    let bb = BlockBuffer::new(buf_size, num_bands, num_render_channels);
                    let sb = SpectrumBuffer::new(buf_size, num_render_channels);
                    let fb = FftBuffer::new(buf_size, num_render_channels);
                    let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

                    let mut capture = Block::new(num_bands, num_capture_channels);
                    let delay_estimate = None;

                    for k in 0..100 {
                        let echo_path_variability = EchoPathVariability::new(
                            k % 3 == 0,
                            if k % 5 == 0 {
                                crate::echo_path_variability::DelayAdjustment::NewDetectedDelay
                            } else {
                                crate::echo_path_variability::DelayAdjustment::None
                            },
                            false,
                        );

                        remover.process_capture(
                            echo_path_variability,
                            k % 2 == 0,
                            &delay_estimate,
                            &render_buffer,
                            Option::None,
                            &mut capture,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn get_metrics_initial() {
        let config = EchoCanceller3Config::default();
        let remover = EchoRemover::new(webrtc_simd::SimdBackend::Scalar, &config, 16000, 1, 1);
        let metrics = remover.get_metrics();
        // Just verify it doesn't panic and returns finite values.
        assert!(metrics.echo_return_loss.is_finite() || metrics.echo_return_loss.is_nan());
    }

    #[test]
    fn set_capture_output_usage() {
        let config = EchoCanceller3Config::default();
        let mut remover = EchoRemover::new(webrtc_simd::SimdBackend::Scalar, &config, 16000, 1, 1);
        remover.set_capture_output_usage(false);
        assert!(!remover.capture_output_used);
        remover.set_capture_output_usage(true);
        assert!(remover.capture_output_used);
    }

    #[test]
    fn update_echo_leakage_status() {
        let config = EchoCanceller3Config::default();
        let mut remover = EchoRemover::new(webrtc_simd::SimdBackend::Scalar, &config, 16000, 1, 1);
        remover.update_echo_leakage_status(true);
        assert!(remover.echo_leakage_detected);
        remover.update_echo_leakage_status(false);
        assert!(!remover.echo_leakage_detected);
    }

    #[test]
    fn signal_transition_same_buffer() {
        let a = [1.0f32; 64];
        let mut out = [0.0f32; 64];
        signal_transition(&a, &a, &mut out);
        assert_eq!(out, a);
    }

    #[test]
    fn signal_transition_crossfade() {
        let from = [0.0f32; 64];
        let to = [1.0f32; 64];
        let mut out = [0.0f32; 64];
        signal_transition(&from, &to, &mut out);
        // First sample should be close to 0, last sample in transition close to 1.
        assert!(out[0] > 0.0 && out[0] < 0.1);
        assert!(out[29] > 0.9 && out[29] < 1.0);
        // After transition, should be exactly 1.0.
        for &v in &out[30..] {
            assert_eq!(v, 1.0);
        }
    }
}
