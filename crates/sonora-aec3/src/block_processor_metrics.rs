//! Block processor metrics — tracks render buffer underruns, overruns, and
//! periodically reports them.
//!
//! Ported from `modules/audio_processing/aec3/block_processor_metrics.h/cc`.

use crate::common::METRICS_REPORTING_INTERVAL_BLOCKS;

/// Handles the reporting of metrics for the block processor.
pub(crate) struct BlockProcessorMetrics {
    capture_block_counter: usize,
    metrics_reported: bool,
    render_buffer_underruns: usize,
    render_buffer_overruns: usize,
    buffer_render_calls: usize,
}

impl BlockProcessorMetrics {
    pub(crate) fn new() -> Self {
        Self {
            capture_block_counter: 0,
            metrics_reported: false,
            render_buffer_underruns: 0,
            render_buffer_overruns: 0,
            buffer_render_calls: 0,
        }
    }

    /// Updates the metric with new capture data.
    pub(crate) fn update_capture(&mut self, underrun: bool) {
        self.capture_block_counter += 1;
        if underrun {
            self.render_buffer_underruns += 1;
        }

        if self.capture_block_counter == METRICS_REPORTING_INTERVAL_BLOCKS {
            self.metrics_reported = true;

            // RTC_HISTOGRAM reporting of underrun/overrun categories skipped
            // (no metrics system in Rust port).

            self.reset_metrics();
            self.capture_block_counter = 0;
        } else {
            self.metrics_reported = false;
        }
    }

    /// Updates the metric with new render data.
    pub(crate) fn update_render(&mut self, overrun: bool) {
        self.buffer_render_calls += 1;
        if overrun {
            self.render_buffer_overruns += 1;
        }
    }

    /// Returns true if the metrics have just been reported.
    pub(crate) fn metrics_reported(&self) -> bool {
        self.metrics_reported
    }

    fn reset_metrics(&mut self) {
        self.render_buffer_underruns = 0;
        self.render_buffer_overruns = 0;
        self.buffer_render_calls = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_usage() {
        let mut metrics = BlockProcessorMetrics::new();

        for _ in 0..3 {
            for _ in 0..METRICS_REPORTING_INTERVAL_BLOCKS - 1 {
                metrics.update_render(false);
                metrics.update_render(false);
                metrics.update_capture(false);
                assert!(!metrics.metrics_reported());
            }
            metrics.update_capture(false);
            assert!(metrics.metrics_reported());
        }
    }
}
