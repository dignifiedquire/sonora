//! AEC state machine — central state tracking for echo cancellation.
//!
//! Coordinates filter analysis, ERL/ERLE estimation, echo audibility,
//! reverb estimation, subtractor output analysis, and transparent mode
//! detection.
//!
//! Ported from `modules/audio_processing/aec3/aec_state.h/cc`.

use crate::block::Block;
use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, NUM_BLOCKS_PER_SECOND};
use crate::config::EchoCanceller3Config;
use crate::delay_estimate::DelayEstimate;
use crate::echo_audibility::EchoAudibility;
use crate::echo_path_variability::EchoPathVariability;
use crate::erl_estimator::ErlEstimator;
use crate::erle_estimator::ErleEstimator;
use crate::filter_analyzer::FilterAnalyzer;
use crate::render_buffer::RenderBuffer;
use crate::reverb_model::ReverbModel;
use crate::reverb_model_estimator::ReverbModelEstimator;
use crate::spectrum_buffer::SpectrumBuffer;
use crate::subtractor_output::SubtractorOutput;
use crate::subtractor_output_analyzer::SubtractorOutputAnalyzer;
use crate::transparent_mode::TransparentMode;

/// Computes the average render spectrum with reverb contribution.
fn compute_avg_render_reverb(
    spectrum_buffer: &SpectrumBuffer,
    delay_blocks: i32,
    reverb_decay: f32,
    reverb_model: &mut ReverbModel,
    reverb_power_spectrum: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let num_render_channels = spectrum_buffer.buffer[0].len();
    let idx_at_delay = spectrum_buffer
        .index
        .offset_index(spectrum_buffer.index.read, delay_blocks);
    let idx_past = spectrum_buffer.index.inc_index(idx_at_delay);

    if num_render_channels > 1 {
        let mut x2_past = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let normalizer = 1.0 / num_render_channels as f32;
        for ch in 0..num_render_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2_past[k] += spectrum_buffer.buffer[idx_past][ch][k];
            }
        }
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            x2_past[k] *= normalizer;
        }
        reverb_model.update_reverb_no_freq_shaping(&x2_past, 1.0, reverb_decay);

        let mut x2_at_delay = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for ch in 0..num_render_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                x2_at_delay[k] += spectrum_buffer.buffer[idx_at_delay][ch][k];
            }
        }
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            x2_at_delay[k] *= normalizer;
        }

        let reverb_power = reverb_model.reverb();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            reverb_power_spectrum[k] = x2_at_delay[k] + reverb_power[k];
        }
    } else {
        reverb_model.update_reverb_no_freq_shaping(
            &spectrum_buffer.buffer[idx_past][0],
            1.0,
            reverb_decay,
        );

        let reverb_power = reverb_model.reverb();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            reverb_power_spectrum[k] = spectrum_buffer.buffer[idx_at_delay][0][k] + reverb_power[k];
        }
    }
}

// --- Nested state classes ---

/// Controls the transition from the initial state parameter set.
struct InitialState {
    conservative_initial_phase: bool,
    initial_state_seconds: f32,
    transition_triggered: bool,
    initial_state: bool,
    strong_not_saturated_render_blocks: usize,
}

impl InitialState {
    fn new(config: &EchoCanceller3Config) -> Self {
        let mut s = Self {
            conservative_initial_phase: config.filter.conservative_initial_phase,
            initial_state_seconds: config.filter.initial_state_seconds,
            transition_triggered: false,
            initial_state: true,
            strong_not_saturated_render_blocks: 0,
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.initial_state = true;
        self.strong_not_saturated_render_blocks = 0;
    }

    fn update(&mut self, active_render: bool, saturated_capture: bool) {
        self.strong_not_saturated_render_blocks += if active_render && !saturated_capture {
            1
        } else {
            0
        };

        let prev_initial_state = self.initial_state;
        if self.conservative_initial_phase {
            self.initial_state =
                self.strong_not_saturated_render_blocks < 5 * NUM_BLOCKS_PER_SECOND;
        } else {
            self.initial_state = (self.strong_not_saturated_render_blocks as f32)
                < self.initial_state_seconds * NUM_BLOCKS_PER_SECOND as f32;
        }

        self.transition_triggered = !self.initial_state && prev_initial_state;
    }

    fn initial_state_active(&self) -> bool {
        self.initial_state
    }

    fn transition_triggered(&self) -> bool {
        self.transition_triggered
    }
}

/// Manages the direct-path delay relative to the beginning of the filter.
struct FilterDelay {
    delay_headroom_blocks: i32,
    filter_delays_blocks: Vec<i32>,
    min_filter_delay: i32,
    external_delay: Option<DelayEstimate>,
}

impl FilterDelay {
    fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let delay_headroom_blocks = (config.delay.delay_headroom_samples / BLOCK_SIZE) as i32;
        Self {
            delay_headroom_blocks,
            filter_delays_blocks: vec![delay_headroom_blocks; num_capture_channels],
            min_filter_delay: delay_headroom_blocks,
            external_delay: None,
        }
    }

    fn external_delay_reported(&self) -> bool {
        self.external_delay.is_some()
    }

    fn external_delay_blocks(&self) -> Option<DelayEstimate> {
        self.external_delay
    }

    fn direct_path_filter_delays(&self) -> &[i32] {
        &self.filter_delays_blocks
    }

    fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_filter_delay
    }

    fn update(
        &mut self,
        analyzer_filter_delay_estimates_blocks: &[i32],
        external_delay: &Option<DelayEstimate>,
        blocks_with_proper_filter_adaptation: usize,
    ) {
        if let Some(ext) = external_delay {
            if self.external_delay.is_none() || self.external_delay.unwrap().delay != ext.delay {
                self.external_delay = Some(*ext);
            }
        }

        let delay_estimator_may_not_have_converged =
            blocks_with_proper_filter_adaptation < 2 * NUM_BLOCKS_PER_SECOND;
        if delay_estimator_may_not_have_converged && self.external_delay.is_some() {
            let delay_guess = self.delay_headroom_blocks;
            self.filter_delays_blocks.fill(delay_guess);
        } else {
            debug_assert_eq!(
                self.filter_delays_blocks.len(),
                analyzer_filter_delay_estimates_blocks.len()
            );
            self.filter_delays_blocks
                .copy_from_slice(analyzer_filter_delay_estimates_blocks);
        }

        self.min_filter_delay = *self
            .filter_delays_blocks
            .iter()
            .min()
            .unwrap_or(&self.delay_headroom_blocks);
    }
}

/// Analyzes the quality of the linear filter to decide if it is usable.
struct FilteringQualityAnalyzer {
    use_linear_filter: bool,
    overall_usable_linear_estimates: bool,
    filter_update_blocks_since_reset: usize,
    filter_update_blocks_since_start: usize,
    convergence_seen: bool,
    usable_linear_filter_estimates: Vec<bool>,
}

impl FilteringQualityAnalyzer {
    fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            use_linear_filter: config.filter.use_linear_filter,
            overall_usable_linear_estimates: false,
            filter_update_blocks_since_reset: 0,
            filter_update_blocks_since_start: 0,
            convergence_seen: false,
            usable_linear_filter_estimates: vec![false; num_capture_channels],
        }
    }

    fn linear_filter_usable(&self) -> bool {
        self.overall_usable_linear_estimates
    }

    fn usable_linear_filter_outputs(&self) -> &[bool] {
        &self.usable_linear_filter_estimates
    }

    fn reset(&mut self) {
        self.usable_linear_filter_estimates.fill(false);
        self.overall_usable_linear_estimates = false;
        self.filter_update_blocks_since_reset = 0;
    }

    fn update(
        &mut self,
        active_render: bool,
        transparent_mode: bool,
        saturated_capture: bool,
        external_delay: &Option<DelayEstimate>,
        any_filter_converged: bool,
    ) {
        let filter_update = active_render && !saturated_capture;
        self.filter_update_blocks_since_reset += if filter_update { 1 } else { 0 };
        self.filter_update_blocks_since_start += if filter_update { 1 } else { 0 };

        self.convergence_seen = self.convergence_seen || any_filter_converged;

        let sufficient_data_to_converge_at_startup =
            self.filter_update_blocks_since_start as f32 > NUM_BLOCKS_PER_SECOND as f32 * 0.4;
        let sufficient_data_to_converge_at_reset = sufficient_data_to_converge_at_startup
            && self.filter_update_blocks_since_reset as f32 > NUM_BLOCKS_PER_SECOND as f32 * 0.2;

        self.overall_usable_linear_estimates =
            sufficient_data_to_converge_at_startup && sufficient_data_to_converge_at_reset;

        self.overall_usable_linear_estimates = self.overall_usable_linear_estimates
            && (external_delay.is_some() || self.convergence_seen);

        self.overall_usable_linear_estimates =
            self.overall_usable_linear_estimates && !transparent_mode;

        if self.use_linear_filter {
            self.usable_linear_filter_estimates
                .fill(self.overall_usable_linear_estimates);
        }
    }
}

/// Detects whether the echo is saturated.
struct SaturationDetector {
    saturated_echo: bool,
}

impl SaturationDetector {
    fn new() -> Self {
        Self {
            saturated_echo: false,
        }
    }

    fn saturated_echo(&self) -> bool {
        self.saturated_echo
    }

    fn update(
        &mut self,
        x: &Block,
        saturated_capture: bool,
        usable_linear_estimate: bool,
        subtractor_output: &[SubtractorOutput],
        echo_path_gain: f32,
    ) {
        self.saturated_echo = false;
        if !saturated_capture {
            return;
        }

        if usable_linear_estimate {
            const SATURATION_THRESHOLD: f32 = 20000.0;
            for output in subtractor_output {
                self.saturated_echo = self.saturated_echo
                    || output.s_refined_max_abs > SATURATION_THRESHOLD
                    || output.s_coarse_max_abs > SATURATION_THRESHOLD;
            }
        } else {
            let mut max_sample = 0.0f32;
            for ch in 0..x.num_channels() {
                let x_ch = x.view(0, ch);
                for &sample in x_ch {
                    max_sample = max_sample.max(sample.abs());
                }
            }

            const MARGIN: f32 = 10.0;
            let peak_echo_amplitude = max_sample * echo_path_gain * MARGIN;
            self.saturated_echo = self.saturated_echo || peak_echo_amplitude > 32000.0;
        }
    }
}

// --- Main AecState ---

/// Central state machine for the echo canceller.
pub(crate) struct AecState {
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    // Field trial flags — use defaults (no field trials).
    deactivate_initial_state_reset_at_echo_path_change: bool,
    full_reset_at_echo_path_change: bool,
    subtractor_analyzer_reset_at_echo_path_change: bool,

    initial_state: InitialState,
    delay_state: FilterDelay,
    transparent_state: Option<TransparentMode>,
    filter_quality_state: FilteringQualityAnalyzer,
    saturation_detector: SaturationDetector,

    erl_estimator: ErlEstimator,
    erle_estimator: ErleEstimator,
    strong_not_saturated_render_blocks: usize,
    blocks_with_active_render: usize,
    capture_signal_saturation: bool,
    filter_analyzer: FilterAnalyzer,
    echo_audibility: EchoAudibility,
    reverb_model_estimator: ReverbModelEstimator,
    avg_render_reverb: ReverbModel,
    subtractor_output_analyzer: SubtractorOutputAnalyzer,
}

impl AecState {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        Self {
            config: config.clone(),
            num_capture_channels,
            // Without field trials, these are the defaults:
            deactivate_initial_state_reset_at_echo_path_change: false,
            full_reset_at_echo_path_change: true,
            subtractor_analyzer_reset_at_echo_path_change: true,

            initial_state: InitialState::new(config),
            delay_state: FilterDelay::new(config, num_capture_channels),
            transparent_state: TransparentMode::create(config),
            filter_quality_state: FilteringQualityAnalyzer::new(config, num_capture_channels),
            saturation_detector: SaturationDetector::new(),

            erl_estimator: ErlEstimator::new(2 * NUM_BLOCKS_PER_SECOND),
            erle_estimator: ErleEstimator::new(
                2 * NUM_BLOCKS_PER_SECOND,
                config,
                num_capture_channels,
            ),
            strong_not_saturated_render_blocks: 0,
            blocks_with_active_render: 0,
            capture_signal_saturation: false,
            filter_analyzer: FilterAnalyzer::new(config, num_capture_channels),
            echo_audibility: EchoAudibility::new(
                config.echo_audibility.use_stationarity_properties_at_init,
            ),
            reverb_model_estimator: ReverbModelEstimator::new(config, num_capture_channels),
            avg_render_reverb: ReverbModel::new(),
            subtractor_output_analyzer: SubtractorOutputAnalyzer::new(num_capture_channels),
        }
    }

    /// Returns whether the echo subtractor can be used to determine the
    /// residual echo.
    pub(crate) fn usable_linear_estimate(&self) -> bool {
        self.filter_quality_state.linear_filter_usable() && self.config.filter.use_linear_filter
    }

    /// Returns whether the echo subtractor output should be used as output.
    pub(crate) fn use_linear_filter_output(&self) -> bool {
        self.filter_quality_state.linear_filter_usable() && self.config.filter.use_linear_filter
    }

    /// Returns whether the render signal is currently active.
    pub(crate) fn active_render(&self) -> bool {
        self.blocks_with_active_render > 200
    }

    /// Gets the residual echo scaling.
    pub(crate) fn get_residual_echo_scaling(
        &self,
        residual_scaling: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        let filter_has_had_time_to_converge = if self.config.filter.conservative_initial_phase {
            self.strong_not_saturated_render_blocks as f32 >= 1.5 * NUM_BLOCKS_PER_SECOND as f32
        } else {
            self.strong_not_saturated_render_blocks as f32 >= 0.8 * NUM_BLOCKS_PER_SECOND as f32
        };
        self.echo_audibility
            .get_residual_echo_scaling(filter_has_had_time_to_converge, residual_scaling);
    }

    /// Returns whether stationarity properties should be used.
    pub(crate) fn use_stationarity_properties(&self) -> bool {
        self.config.echo_audibility.use_stationarity_properties
    }

    /// Returns the ERLE estimate.
    pub(crate) fn erle(&self, onset_compensated: bool) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        self.erle_estimator.erle(onset_compensated)
    }

    /// Returns the unbounded ERLE estimate.
    pub(crate) fn erle_unbounded(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        self.erle_estimator.erle_unbounded()
    }

    /// Returns the fullband ERLE estimate in log2 units.
    pub(crate) fn fullband_erle_log2(&self) -> f32 {
        self.erle_estimator.fullband_erle_log2()
    }

    /// Returns the ERL.
    pub(crate) fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.erl_estimator.erl()
    }

    /// Returns the time-domain ERL.
    pub(crate) fn erl_time_domain(&self) -> f32 {
        self.erl_estimator.erl_time_domain()
    }

    /// Returns the minimum direct path filter delay in blocks.
    pub(crate) fn min_direct_path_filter_delay(&self) -> i32 {
        self.delay_state.min_direct_path_filter_delay()
    }

    /// Returns whether the capture signal is saturated.
    pub(crate) fn saturated_capture(&self) -> bool {
        self.capture_signal_saturation
    }

    /// Returns whether the echo signal is saturated.
    pub(crate) fn saturated_echo(&self) -> bool {
        self.saturation_detector.saturated_echo()
    }

    /// Updates the capture signal saturation.
    pub(crate) fn update_capture_saturation(&mut self, capture_signal_saturation: bool) {
        self.capture_signal_saturation = capture_signal_saturation;
    }

    /// Returns whether transparent mode is active.
    pub(crate) fn transparent_mode_active(&self) -> bool {
        self.transparent_state.as_ref().is_some_and(|t| t.active())
    }

    /// Takes appropriate action at an echo path change.
    pub(crate) fn handle_echo_path_change(&mut self, echo_path_variability: &EchoPathVariability) {
        if self.full_reset_at_echo_path_change
            && echo_path_variability.delay_change
                != crate::echo_path_variability::DelayAdjustment::None
        {
            self.filter_analyzer.reset();
            self.capture_signal_saturation = false;
            self.strong_not_saturated_render_blocks = 0;
            self.blocks_with_active_render = 0;
            if !self.deactivate_initial_state_reset_at_echo_path_change {
                self.initial_state.reset();
            }
            if let Some(ref mut ts) = self.transparent_state {
                ts.reset();
            }
            self.erle_estimator.reset(true);
            self.erl_estimator.reset();
            self.filter_quality_state.reset();
        } else if echo_path_variability.gain_change {
            self.erle_estimator.reset(false);
        }

        if self.subtractor_analyzer_reset_at_echo_path_change {
            self.subtractor_output_analyzer.handle_echo_path_change();
        }
    }

    /// Returns the reverb decay. The parameter `mild` indicates which
    /// exponential decay to return.
    pub(crate) fn reverb_decay(&self, mild: bool) -> f32 {
        self.reverb_model_estimator.reverb_decay(mild)
    }

    /// Returns the frequency response of the reverberant echo.
    pub(crate) fn get_reverb_frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        self.reverb_model_estimator.get_reverb_frequency_response()
    }

    /// Returns whether the transition from the initial state has been
    /// triggered.
    pub(crate) fn transition_triggered(&self) -> bool {
        self.initial_state.transition_triggered()
    }

    /// Returns the filter length in blocks.
    pub(crate) fn filter_length_blocks(&self) -> usize {
        self.filter_analyzer.filter_length_blocks()
    }

    /// Returns the external delay estimate.
    pub(crate) fn external_delay_blocks(&self) -> Option<DelayEstimate> {
        self.delay_state.external_delay_blocks()
    }

    /// Updates the AEC state with new data.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn update(
        &mut self,
        external_delay: &Option<DelayEstimate>,
        adaptive_filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        adaptive_filter_impulse_responses: &[Vec<f32>],
        render_buffer: &RenderBuffer<'_>,
        e2_refined: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        subtractor_output: &[SubtractorOutput],
    ) {
        debug_assert_eq!(self.num_capture_channels, y2.len());
        debug_assert_eq!(self.num_capture_channels, subtractor_output.len());
        debug_assert_eq!(
            self.num_capture_channels,
            adaptive_filter_frequency_responses.len()
        );
        debug_assert_eq!(
            self.num_capture_channels,
            adaptive_filter_impulse_responses.len()
        );

        // Analyze the filter outputs and filters.
        let (any_filter_converged, any_coarse_filter_converged, all_filters_diverged) =
            self.subtractor_output_analyzer.update(subtractor_output);

        let mut any_filter_consistent = false;
        let mut max_echo_path_gain = 0.0f32;
        self.filter_analyzer.update(
            adaptive_filter_impulse_responses,
            render_buffer,
            &mut any_filter_consistent,
            &mut max_echo_path_gain,
        );

        // Estimate the direct path delay of the filter.
        if self.config.filter.use_linear_filter {
            self.delay_state.update(
                self.filter_analyzer.filter_delays_blocks(),
                external_delay,
                self.strong_not_saturated_render_blocks,
            );
        }

        let aligned_render_block =
            render_buffer.get_block(-self.delay_state.min_direct_path_filter_delay());

        // Update render counters.
        let mut active_render = false;
        for ch in 0..aligned_render_block.num_channels() {
            let block = aligned_render_block.view(0, ch);
            let render_energy: f32 = block.iter().map(|&v| v * v).sum();
            if render_energy
                > (self.config.render_levels.active_render_limit
                    * self.config.render_levels.active_render_limit)
                    * FFT_LENGTH_BY_2 as f32
            {
                active_render = true;
                break;
            }
        }
        self.blocks_with_active_render += if active_render { 1 } else { 0 };
        self.strong_not_saturated_render_blocks += if active_render && !self.saturated_capture() {
            1
        } else {
            0
        };

        let mut avg_render_spectrum_with_reverb = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        compute_avg_render_reverb(
            render_buffer.get_spectrum_buffer(),
            self.delay_state.min_direct_path_filter_delay(),
            self.reverb_decay(false),
            &mut self.avg_render_reverb,
            &mut avg_render_spectrum_with_reverb,
        );

        if self.config.echo_audibility.use_stationarity_properties {
            self.echo_audibility.update(
                render_buffer,
                self.avg_render_reverb.reverb(),
                self.delay_state.min_direct_path_filter_delay(),
                self.delay_state.external_delay_reported(),
            );
        }

        // Update the ERL and ERLE measures.
        if self.initial_state.transition_triggered() {
            self.erle_estimator.reset(false);
        }

        self.erle_estimator.update(
            render_buffer,
            adaptive_filter_frequency_responses,
            &avg_render_spectrum_with_reverb,
            y2,
            e2_refined,
            self.subtractor_output_analyzer.converged_filters(),
        );

        self.erl_estimator.update(
            self.subtractor_output_analyzer.converged_filters(),
            render_buffer.spectrum(self.delay_state.min_direct_path_filter_delay()),
            y2,
        );

        // Detect and flag echo saturation.
        if self.config.ep_strength.echo_can_saturate {
            self.saturation_detector.update(
                aligned_render_block,
                self.saturated_capture(),
                self.usable_linear_estimate(),
                subtractor_output,
                max_echo_path_gain,
            );
        } else {
            debug_assert!(!self.saturation_detector.saturated_echo());
        }

        // Update the decision on whether to use the initial state parameter set.
        self.initial_state
            .update(active_render, self.saturated_capture());

        // Detect whether the transparent mode should be activated.
        let saturated_capture = self.saturated_capture();
        if let Some(ref mut ts) = self.transparent_state {
            ts.update(
                self.delay_state.min_direct_path_filter_delay(),
                any_filter_consistent,
                any_filter_converged,
                any_coarse_filter_converged,
                all_filters_diverged,
                active_render,
                saturated_capture,
            );
        }

        // Analyze the quality of the filter.
        self.filter_quality_state.update(
            active_render,
            self.transparent_mode_active(),
            self.saturated_capture(),
            external_delay,
            any_filter_converged,
        );

        // Update the reverb estimate.
        let stationary_block = self.config.echo_audibility.use_stationarity_properties
            && self.echo_audibility.is_block_stationary();

        self.reverb_model_estimator.update(
            self.filter_analyzer.get_adjusted_filters(),
            adaptive_filter_frequency_responses,
            self.erle_estimator.get_inst_linear_quality_estimates(),
            self.delay_state.direct_path_filter_delays(),
            self.filter_quality_state.usable_linear_filter_outputs(),
            stationary_block,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_not_usable() {
        let config = EchoCanceller3Config::default();
        let state = AecState::new(&config, 1);
        assert!(!state.usable_linear_estimate());
        assert!(!state.active_render());
        assert!(!state.saturated_capture());
        assert!(!state.saturated_echo());
        assert!(!state.transparent_mode_active());
    }

    #[test]
    fn capture_saturation_flag() {
        let config = EchoCanceller3Config::default();
        let mut state = AecState::new(&config, 1);
        assert!(!state.saturated_capture());
        state.update_capture_saturation(true);
        assert!(state.saturated_capture());
        state.update_capture_saturation(false);
        assert!(!state.saturated_capture());
    }
}
