//! Aggregates lag estimates from the matched filter into a single reliable
//! combined lag estimate.
//!
//! Ported from `modules/audio_processing/aec3/matched_filter_lag_aggregator.h/cc`.

use crate::common::{BLOCK_SIZE_LOG2, MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS};
use crate::config::{Delay, DelaySelectionThresholds};
use crate::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::matched_filter::LagEstimate;

const PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED: i32 = -1;
const HISTOGRAM_DATA_SIZE: usize = 250;

fn get_down_sampling_block_size_log2(down_sampling_factor: usize) -> i32 {
    let mut dsf = down_sampling_factor >> 1;
    let mut dsf_log2: i32 = 0;
    while dsf > 0 {
        dsf_log2 += 1;
        dsf >>= 1;
    }
    let bsl2 = BLOCK_SIZE_LOG2 as i32;
    if bsl2 > dsf_log2 { bsl2 - dsf_log2 } else { 0 }
}

/// Aggregates lag estimates from matched filters into a histogram and finds
/// the best candidate.
struct HighestPeakAggregator {
    histogram: Vec<i32>,
    histogram_data: [i32; HISTOGRAM_DATA_SIZE],
    histogram_data_index: usize,
    candidate: i32,
}

impl HighestPeakAggregator {
    fn new(max_filter_lag: usize) -> Self {
        Self {
            histogram: vec![0i32; max_filter_lag + 1],
            histogram_data: [0i32; HISTOGRAM_DATA_SIZE],
            histogram_data_index: 0,
            candidate: -1,
        }
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data.fill(0);
        self.histogram_data_index = 0;
    }

    fn aggregate(&mut self, lag: i32) {
        let old_val = self.histogram_data[self.histogram_data_index] as usize;
        debug_assert!(old_val < self.histogram.len());
        self.histogram[old_val] -= 1;

        self.histogram_data[self.histogram_data_index] = lag;

        let new_val = self.histogram_data[self.histogram_data_index] as usize;
        debug_assert!(new_val < self.histogram.len());
        self.histogram[new_val] += 1;

        self.histogram_data_index = (self.histogram_data_index + 1) % self.histogram_data.len();

        // Find the index with the maximum histogram count.
        self.candidate = self
            .histogram
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(i, _)| i as i32)
            .unwrap_or(-1);
    }

    fn candidate(&self) -> i32 {
        self.candidate
    }

    fn histogram(&self) -> &[i32] {
        &self.histogram
    }
}

/// Aggregates pre-echo lag estimates using a histogram with weighted
/// penalization.
struct PreEchoLagAggregator {
    block_size_log2: i32,
    histogram_data: [i32; HISTOGRAM_DATA_SIZE],
    histogram: Vec<i32>,
    histogram_data_index: usize,
    pre_echo_candidate: i32,
    number_updates: i32,
}

impl PreEchoLagAggregator {
    fn new(max_filter_lag: usize, down_sampling_factor: usize) -> Self {
        let block_size_log2 = get_down_sampling_block_size_log2(down_sampling_factor);
        let hist_size = ((max_filter_lag + 1) * down_sampling_factor) >> BLOCK_SIZE_LOG2;
        let mut aggregator = Self {
            block_size_log2,
            histogram_data: [PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED; HISTOGRAM_DATA_SIZE],
            histogram: vec![0i32; hist_size],
            histogram_data_index: 0,
            pre_echo_candidate: 0,
            number_updates: 0,
        };
        aggregator.reset();
        aggregator
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.histogram_data
            .fill(PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED);
        self.histogram_data_index = 0;
        self.pre_echo_candidate = 0;
    }

    fn aggregate(&mut self, pre_echo_lag: i32) {
        let mut pre_echo_block_size = pre_echo_lag >> self.block_size_log2;
        debug_assert!(
            pre_echo_block_size >= 0 && (pre_echo_block_size as usize) < self.histogram.len()
        );
        pre_echo_block_size = pre_echo_block_size.clamp(0, self.histogram.len() as i32 - 1);

        // Remove the oldest point from the histogram.
        let old_val = self.histogram_data[self.histogram_data_index];
        if old_val != PRE_ECHO_HISTOGRAM_DATA_NOT_UPDATED {
            self.histogram[old_val as usize] -= 1;
        }

        self.histogram_data[self.histogram_data_index] = pre_echo_block_size;
        self.histogram[pre_echo_block_size as usize] += 1;
        self.histogram_data_index = (self.histogram_data_index + 1) % self.histogram_data.len();

        let num_blocks_per_second = 250i32;
        let pre_echo_candidate_block_size;
        if self.number_updates < num_blocks_per_second * 2 {
            self.number_updates += 1;
            let mut penalization_per_delay = 1.0f32;
            let mut max_histogram_value = -1.0f32;
            let mut best_idx = 0usize;
            let window = MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS;

            let mut start = 0;
            while start + window <= self.histogram.len() {
                let chunk = &self.histogram[start..start + window];
                let (local_max_idx, local_max_val) =
                    chunk.iter().enumerate().max_by_key(|(_, v)| *v).unwrap();
                let weighted = *local_max_val as f32 * penalization_per_delay;
                if weighted > max_histogram_value {
                    max_histogram_value = weighted;
                    best_idx = start + local_max_idx;
                }
                penalization_per_delay *= 0.7;
                start += window;
            }
            pre_echo_candidate_block_size = best_idx as i32;
        } else {
            pre_echo_candidate_block_size = self
                .histogram
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| *v)
                .map(|(i, _)| i as i32)
                .unwrap_or(0);
        }

        self.pre_echo_candidate = pre_echo_candidate_block_size << self.block_size_log2;
    }

    fn pre_echo_candidate(&self) -> i32 {
        self.pre_echo_candidate
    }
}

/// Aggregates lag estimates produced by the MatchedFilter class into a single
/// reliable combined lag estimate.
pub(crate) struct MatchedFilterLagAggregator {
    significant_candidate_found: bool,
    thresholds: DelaySelectionThresholds,
    headroom: i32,
    highest_peak_aggregator: HighestPeakAggregator,
    pre_echo_lag_aggregator: Option<PreEchoLagAggregator>,
}

impl MatchedFilterLagAggregator {
    pub(crate) fn new(max_filter_lag: usize, delay_config: &Delay) -> Self {
        let headroom =
            delay_config.delay_headroom_samples as i32 / delay_config.down_sampling_factor as i32;
        let pre_echo_lag_aggregator = if delay_config.detect_pre_echo {
            Some(PreEchoLagAggregator::new(
                max_filter_lag,
                delay_config.down_sampling_factor,
            ))
        } else {
            None
        };

        debug_assert!(
            delay_config.delay_selection_thresholds.initial
                <= delay_config.delay_selection_thresholds.converged
        );

        Self {
            significant_candidate_found: false,
            thresholds: delay_config.delay_selection_thresholds.clone(),
            headroom,
            highest_peak_aggregator: HighestPeakAggregator::new(max_filter_lag),
            pre_echo_lag_aggregator,
        }
    }

    pub(crate) fn reset(&mut self, hard_reset: bool) {
        self.highest_peak_aggregator.reset();
        if let Some(ref mut pre_echo) = self.pre_echo_lag_aggregator {
            pre_echo.reset();
        }
        if hard_reset {
            self.significant_candidate_found = false;
        }
    }

    pub(crate) fn aggregate(&mut self, lag_estimate: Option<LagEstimate>) -> Option<DelayEstimate> {
        if let (Some(est), Some(pre_echo)) = (lag_estimate, &mut self.pre_echo_lag_aggregator) {
            pre_echo.aggregate((est.pre_echo_lag as i32 - self.headroom).max(0));
        }

        if let Some(est) = lag_estimate {
            let lag_with_headroom = (est.lag as i32 - self.headroom).max(0);
            self.highest_peak_aggregator.aggregate(lag_with_headroom);
            let histogram = self.highest_peak_aggregator.histogram();
            let candidate = self.highest_peak_aggregator.candidate();

            self.significant_candidate_found = self.significant_candidate_found
                || histogram[candidate as usize] > self.thresholds.converged as i32;

            if histogram[candidate as usize] > self.thresholds.converged as i32
                || (histogram[candidate as usize] > self.thresholds.initial as i32
                    && !self.significant_candidate_found)
            {
                let quality = if self.significant_candidate_found {
                    DelayEstimateQuality::Refined
                } else {
                    DelayEstimateQuality::Coarse
                };
                let reported_delay = if let Some(ref pre_echo) = self.pre_echo_lag_aggregator {
                    pre_echo.pre_echo_candidate() as usize
                } else {
                    candidate as usize
                };
                return Some(DelayEstimate::new(quality, reported_delay));
            }
        }

        None
    }

    /// Returns whether a reliable delay estimate has been found.
    pub(crate) fn reliable_delay_found(&self) -> bool {
        self.significant_candidate_found
    }

    /// Returns the delay candidate computed from the highest peak.
    pub(crate) fn get_delay_at_highest_peak(&self) -> i32 {
        self.highest_peak_aggregator.candidate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Delay;

    const NUM_LAGS_BEFORE_DETECTION: usize = 26;

    #[test]
    fn lag_estimate_invariance_required_for_aggregated_lag() {
        let delay_config = Delay::default();
        let mut aggregator = MatchedFilterLagAggregator::new(100, &delay_config);

        let mut aggregated_lag = None;
        for _ in 0..NUM_LAGS_BEFORE_DETECTION {
            aggregated_lag = aggregator.aggregate(Some(LagEstimate::new(10, 10)));
        }
        assert!(aggregated_lag.is_some());

        for k in 0..NUM_LAGS_BEFORE_DETECTION * 100 {
            aggregated_lag = aggregator.aggregate(Some(LagEstimate::new(k % 100, k % 100)));
        }
        assert!(aggregated_lag.is_none());

        for k in 0..NUM_LAGS_BEFORE_DETECTION * 100 {
            aggregated_lag = aggregator.aggregate(Some(LagEstimate::new(k % 100, k % 100)));
            assert!(aggregated_lag.is_none());
        }
    }
}
