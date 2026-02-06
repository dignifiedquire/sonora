//! AEC state machine — stub for use by Subtractor and ResidualEchoEstimator.
//!
//! This is a forward declaration / stub for the full implementation in Step 18.
//! Only the methods needed by current consumers are provided here.
//!
//! Ported from `modules/audio_processing/aec3/aec_state.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::EchoCanceller3Config;
use crate::delay_estimate::DelayEstimate;
use crate::echo_path_variability::EchoPathVariability;
use crate::render_buffer::RenderBuffer;
use crate::subtractor_output::SubtractorOutput;

/// Central state machine for the echo canceller.
///
/// This is a stub — the full implementation is in Step 18.
pub(crate) struct AecState {
    config: EchoCanceller3Config,
    saturated_capture: bool,
    saturated_echo: bool,
    usable_linear_estimate: bool,
    transparent_mode_active: bool,
    use_stationarity_properties: bool,
    min_direct_path_filter_delay: i32,
    filter_length_blocks: usize,
    erle: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_unbounded: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_onset: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    reverb_decay: f32,
    reverb_frequency_response: [f32; FFT_LENGTH_BY_2_PLUS_1],
    residual_echo_scaling: [f32; FFT_LENGTH_BY_2_PLUS_1],
    external_delay: Option<DelayEstimate>,
}

impl AecState {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let erle_init = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        Self {
            config: config.clone(),
            saturated_capture: false,
            saturated_echo: false,
            usable_linear_estimate: false,
            transparent_mode_active: false,
            use_stationarity_properties: config.echo_audibility.use_stationarity_properties,
            min_direct_path_filter_delay: 0,
            filter_length_blocks: config.filter.refined.length_blocks,
            erle: vec![erle_init; num_capture_channels],
            erle_unbounded: vec![erle_init; num_capture_channels],
            erle_onset: vec![erle_init; num_capture_channels],
            reverb_decay: config.ep_strength.default_len.abs(),
            reverb_frequency_response: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            residual_echo_scaling: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            external_delay: None,
        }
    }

    /// Returns whether the capture signal is saturated.
    pub(crate) fn saturated_capture(&self) -> bool {
        self.saturated_capture
    }

    /// Returns whether the echo is saturated.
    pub(crate) fn saturated_echo(&self) -> bool {
        self.saturated_echo
    }

    /// Returns whether a usable linear estimate is available.
    pub(crate) fn usable_linear_estimate(&self) -> bool {
        self.usable_linear_estimate
    }

    /// Returns whether transparent mode is active.
    pub(crate) fn transparent_mode_active(&self) -> bool {
        self.transparent_mode_active
    }

    /// Returns whether stationarity properties should be used.
    pub(crate) fn use_stationarity_properties(&self) -> bool {
        self.use_stationarity_properties
    }

    /// Returns the minimum direct path filter delay in blocks.
    pub(crate) fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_direct_path_filter_delay
    }

    /// Returns the filter length in blocks.
    pub(crate) fn filter_length_blocks(&self) -> usize {
        self.filter_length_blocks
    }

    /// Returns the ERLE estimate. When `onset_compensated` is true, returns
    /// the onset-compensated ERLE.
    pub(crate) fn erle(&self, onset_compensated: bool) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if onset_compensated {
            &self.erle_onset
        } else {
            &self.erle
        }
    }

    /// Returns the unbounded ERLE estimate.
    pub(crate) fn erle_unbounded(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.erle_unbounded
    }

    /// Returns the reverb decay.
    pub(crate) fn reverb_decay(&self, mild: bool) -> f32 {
        // Stub — in full implementation, delegates to ReverbModelEstimator.
        if mild {
            self.config.ep_strength.nearend_len.abs()
        } else {
            self.reverb_decay
        }
    }

    /// Returns the reverb frequency response.
    pub(crate) fn get_reverb_frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.reverb_frequency_response
    }

    /// Fills `scaling` with the residual echo scaling factors.
    pub(crate) fn get_residual_echo_scaling(&self, scaling: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        scaling.copy_from_slice(&self.residual_echo_scaling);
    }

    /// Returns the external delay estimate.
    pub(crate) fn external_delay_blocks(&self) -> Option<DelayEstimate> {
        self.external_delay
    }

    /// Handles an echo path change event.
    pub(crate) fn handle_echo_path_change(&mut self, _echo_path_variability: &EchoPathVariability) {
        // Stub — full implementation in Step 18.
    }

    /// Updates the state with new data.
    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    pub(crate) fn update(
        &mut self,
        _delay_estimate: &Option<DelayEstimate>,
        _filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        _filter_impulse_responses: &[Vec<f32>],
        _render_buffer: &RenderBuffer<'_>,
        _e2_refined: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        _y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        _subtractor_output: &[SubtractorOutput],
    ) {
        // Stub — full implementation in Step 18.
    }
}
