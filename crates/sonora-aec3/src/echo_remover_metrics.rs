//! Echo remover quality metrics — tracks and reports ERL, ERLE, and saturation.
//!
//! Ported from `modules/audio_processing/aec3/echo_remover_metrics.h/cc`.

use crate::aec_state::AecState;
use crate::common::{
    FFT_LENGTH_BY_2_PLUS_1, METRICS_COLLECTION_BLOCKS, METRICS_REPORTING_INTERVAL_BLOCKS,
};

/// Metric tracking value, floor, and ceiling in the dB domain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DbMetric {
    pub sum_value: f32,
    pub floor_value: f32,
    pub ceil_value: f32,
}

impl Default for DbMetric {
    fn default() -> Self {
        Self {
            sum_value: 0.0,
            floor_value: 0.0,
            ceil_value: 0.0,
        }
    }
}

impl DbMetric {
    pub(crate) fn new(sum_value: f32, floor_value: f32, ceil_value: f32) -> Self {
        Self {
            sum_value,
            floor_value,
            ceil_value,
        }
    }

    /// Updates the metric with an instantaneous value.
    pub(crate) fn update_instant(&mut self, value: f32) {
        self.sum_value = value;
        self.floor_value = self.floor_value.min(value);
        self.ceil_value = self.ceil_value.max(value);
    }
}

/// Handles the reporting of metrics for the echo remover.
#[derive(Debug)]
pub(crate) struct EchoRemoverMetrics {
    block_counter: usize,
    erl_time_domain: DbMetric,
    erle_time_domain: DbMetric,
    saturated_capture: bool,
    metrics_reported: bool,
}

impl EchoRemoverMetrics {
    pub(crate) fn new() -> Self {
        let mut s = Self {
            block_counter: 0,
            erl_time_domain: DbMetric::default(),
            erle_time_domain: DbMetric::default(),
            saturated_capture: false,
            metrics_reported: false,
        };
        s.reset_metrics();
        s
    }

    /// Updates the metric with new data.
    pub(crate) fn update(
        &mut self,
        aec_state: &AecState,
        _comfort_noise_spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        _suppressor_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        self.metrics_reported = false;
        self.block_counter += 1;
        if self.block_counter <= METRICS_COLLECTION_BLOCKS {
            self.erl_time_domain
                .update_instant(aec_state.erl_time_domain());
            self.erle_time_domain
                .update_instant(aec_state.fullband_erle_log2());
            self.saturated_capture = self.saturated_capture || aec_state.saturated_capture();
        } else {
            // Report the metrics over several frames to lower the computational
            // impact of the logarithms.
            match self.block_counter {
                n if n == METRICS_COLLECTION_BLOCKS + 1 => {
                    // Would report UsableLinearEstimate, FilterDelay, CaptureSaturation
                    // via histogram. Skipped in Rust port (no metrics system).
                }
                n if n == METRICS_COLLECTION_BLOCKS + 2 => {
                    // Would report ERL value/max/min. Skipped.
                }
                n if n == METRICS_COLLECTION_BLOCKS + 3 => {
                    // Would report ERLE value/max/min. Skipped.
                    self.metrics_reported = true;
                    debug_assert_eq!(METRICS_REPORTING_INTERVAL_BLOCKS, self.block_counter);
                    self.block_counter = 0;
                    self.reset_metrics();
                }
                _ => {
                    debug_assert!(false, "unexpected block counter state");
                }
            }
        }
    }

    fn reset_metrics(&mut self) {
        self.erl_time_domain = DbMetric::new(0.0, 10000.0, 0.0);
        self.erle_time_domain = DbMetric::new(0.0, 0.0, 1000.0);
        self.saturated_capture = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn db_metric_default() {
        let metric = DbMetric::default();
        assert_eq!(metric.sum_value, 0.0);
        assert_eq!(metric.floor_value, 0.0);
        assert_eq!(metric.ceil_value, 0.0);
    }

    #[test]
    fn db_metric_constructor() {
        let metric = DbMetric::new(1.0, 2.0, 3.0);
        assert_eq!(metric.sum_value, 1.0);
        assert_eq!(metric.floor_value, 2.0);
        assert_eq!(metric.ceil_value, 3.0);
    }

    #[test]
    fn db_metric_update_instant() {
        let mut metric = DbMetric::new(0.0, 20.0, -20.0);
        let min_value = -77.0f32;
        let max_value = 33.0f32;
        let last_value = (min_value + max_value) / 2.0;
        let mut value = min_value;
        while value <= max_value {
            metric.update_instant(value);
            value += 1.0;
        }
        metric.update_instant(last_value);
        assert!((metric.sum_value - last_value).abs() < 1e-4);
        assert!((metric.ceil_value - max_value).abs() < 1e-4);
        assert!((metric.floor_value - min_value).abs() < 1e-4);
    }
}
