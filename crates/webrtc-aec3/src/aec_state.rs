//! AEC state machine — stub for use by Subtractor.
//!
//! This is a forward declaration / stub for the full implementation in Step 18.
//! Only the methods needed by the Subtractor are provided here.
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
    saturated_capture: bool,
    min_direct_path_filter_delay: i32,
}

impl AecState {
    pub(crate) fn new(_config: &EchoCanceller3Config, _num_capture_channels: usize) -> Self {
        Self {
            saturated_capture: false,
            min_direct_path_filter_delay: 0,
        }
    }

    /// Returns whether the capture signal is saturated.
    pub(crate) fn saturated_capture(&self) -> bool {
        self.saturated_capture
    }

    /// Returns the minimum direct path filter delay in blocks.
    pub(crate) fn min_direct_path_filter_delay(&self) -> i32 {
        self.min_direct_path_filter_delay
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
