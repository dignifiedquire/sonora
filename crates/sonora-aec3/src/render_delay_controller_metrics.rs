//! Metrics for the render delay controller.
//!
//! In the C++ implementation, this reports WebRTC histogram metrics. We keep
//! the state tracking logic but skip the actual metric reporting.
//!
//! Ported from
//! `modules/audio_processing/aec3/render_delay_controller_metrics.h/cc`.

use crate::clockdrift_detector::ClockdriftLevel;
use crate::common::{BLOCK_SIZE, METRICS_REPORTING_INTERVAL_BLOCKS, NUM_BLOCKS_PER_SECOND};

/// Handles the reporting of metrics for the render delay controller.
pub(crate) struct RenderDelayControllerMetrics {
    delay_blocks: usize,
    reliable_delay_estimate_counter: i32,
    delay_change_counter: i32,
    call_counter: i32,
    initial_call_counter: i32,
    initial_update: bool,
}

impl RenderDelayControllerMetrics {
    pub(crate) fn new() -> Self {
        Self {
            delay_blocks: 0,
            reliable_delay_estimate_counter: 0,
            delay_change_counter: 0,
            call_counter: 0,
            initial_call_counter: 0,
            initial_update: true,
        }
    }

    /// Updates the metrics with new data.
    pub(crate) fn update(
        &mut self,
        delay_samples: Option<usize>,
        _buffer_delay_blocks: Option<usize>,
        _clockdrift: ClockdriftLevel,
    ) {
        self.call_counter += 1;

        if !self.initial_update {
            let delay_blocks;
            if let Some(samples) = delay_samples {
                self.reliable_delay_estimate_counter += 1;
                // Add an offset by 1 (metric is halved before reporting) to
                // reserve 0 for absent delay.
                delay_blocks = samples / BLOCK_SIZE + 2;
            } else {
                delay_blocks = 0;
            }

            if delay_blocks != self.delay_blocks {
                self.delay_change_counter += 1;
                self.delay_blocks = delay_blocks;
            }
        } else {
            self.initial_call_counter += 1;
            if self.initial_call_counter == 5 * NUM_BLOCKS_PER_SECOND as i32 {
                self.initial_update = false;
            }
        }

        if self.call_counter == METRICS_REPORTING_INTERVAL_BLOCKS as i32 {
            // In the C++ code, histogram metrics are reported here.
            // We skip them in the Rust port.
            self.call_counter = 0;
            self.reset_metrics();
        }
    }

    fn reset_metrics(&mut self) {
        self.delay_change_counter = 0;
        self.reliable_delay_estimate_counter = 0;
    }
}
