//! Echo path delay estimator.
//!
//! Estimates the delay of the echo path using matched filtering and lag
//! aggregation.
//!
//! Ported from `modules/audio_processing/aec3/echo_path_delay_estimator.h/cc`.

use crate::alignment_mixer::AlignmentMixer;
use crate::block::Block;
use crate::clockdrift_detector::{ClockdriftDetector, ClockdriftLevel};
use crate::common::{
    BLOCK_SIZE, MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS, NUM_BLOCKS_PER_SECOND,
};
use crate::config::EchoCanceller3Config;
use crate::decimator::Decimator;
use crate::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::matched_filter::MatchedFilter;
use crate::matched_filter_lag_aggregator::MatchedFilterLagAggregator;

/// Estimates the delay of the echo path.
pub(crate) struct EchoPathDelayEstimator {
    down_sampling_factor: usize,
    sub_block_size: usize,
    capture_mixer: AlignmentMixer,
    capture_decimator: Decimator,
    matched_filter: MatchedFilter,
    matched_filter_lag_aggregator: MatchedFilterLagAggregator,
    old_aggregated_lag: Option<DelayEstimate>,
    consistent_estimate_counter: usize,
    clockdrift_detector: ClockdriftDetector,
}

impl EchoPathDelayEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let down_sampling_factor = config.delay.down_sampling_factor;
        let sub_block_size = if down_sampling_factor != 0 {
            BLOCK_SIZE / down_sampling_factor
        } else {
            BLOCK_SIZE
        };

        let excitation_limit = if config.delay.down_sampling_factor == 8 {
            config.render_levels.poor_excitation_render_limit_ds8
        } else {
            config.render_levels.poor_excitation_render_limit
        };

        let matched_filter = MatchedFilter::new(
            sub_block_size,
            MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS,
            config.delay.num_filters,
            MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS,
            excitation_limit,
            config.delay.delay_estimate_smoothing,
            config.delay.delay_estimate_smoothing_delay_found,
            config.delay.delay_candidate_detection_threshold,
            config.delay.detect_pre_echo,
        );

        let matched_filter_lag_aggregator =
            MatchedFilterLagAggregator::new(matched_filter.get_max_filter_lag(), &config.delay);

        Self {
            down_sampling_factor,
            sub_block_size,
            capture_mixer: AlignmentMixer::new(
                num_capture_channels,
                &config.delay.capture_alignment_mixing,
            ),
            capture_decimator: Decimator::new(down_sampling_factor),
            matched_filter,
            matched_filter_lag_aggregator,
            old_aggregated_lag: None,
            consistent_estimate_counter: 0,
            clockdrift_detector: ClockdriftDetector::new(),
        }
    }

    /// Resets the estimation. If `reset_delay_confidence` is true, the reset
    /// behavior is as if the call is restarted.
    pub(crate) fn reset(&mut self, reset_delay_confidence: bool) {
        self.reset_internal(true, reset_delay_confidence);
    }

    /// Produces a delay estimate if one is available.
    pub(crate) fn estimate_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &Block,
    ) -> Option<DelayEstimate> {
        let mut downmixed_capture = [0.0f32; BLOCK_SIZE];
        self.capture_mixer
            .produce_output(capture, &mut downmixed_capture);

        let mut downsampled_capture_data = [0.0f32; BLOCK_SIZE];
        let downsampled_capture = &mut downsampled_capture_data[..self.sub_block_size];
        self.capture_decimator
            .decimate(&downmixed_capture, downsampled_capture);

        self.matched_filter.update(
            render_buffer,
            downsampled_capture,
            self.matched_filter_lag_aggregator.reliable_delay_found(),
        );

        let mut aggregated_matched_filter_lag = self
            .matched_filter_lag_aggregator
            .aggregate(self.matched_filter.get_best_lag_estimate());

        // Run clockdrift detection.
        if let Some(lag) = &aggregated_matched_filter_lag
            && lag.quality == DelayEstimateQuality::Refined
        {
            self.clockdrift_detector.update(
                self.matched_filter_lag_aggregator
                    .get_delay_at_highest_peak(),
            );
        }

        // Return the detected delay in samples as the aggregated matched filter
        // lag compensated by the down sampling factor.
        if let Some(lag) = &mut aggregated_matched_filter_lag {
            lag.delay *= self.down_sampling_factor;
        }

        if let (Some(old), Some(new)) = (&self.old_aggregated_lag, &aggregated_matched_filter_lag) {
            if old.delay == new.delay {
                self.consistent_estimate_counter += 1;
            } else {
                self.consistent_estimate_counter = 0;
            }
        } else {
            self.consistent_estimate_counter = 0;
        }
        self.old_aggregated_lag = aggregated_matched_filter_lag;

        const NUM_BLOCKS_PER_SECOND_BY_2: usize = NUM_BLOCKS_PER_SECOND / 2;
        if self.consistent_estimate_counter > NUM_BLOCKS_PER_SECOND_BY_2 {
            self.reset_internal(false, false);
        }

        aggregated_matched_filter_lag
    }

    /// Returns the level of detected clock drift.
    pub(crate) fn clockdrift(&self) -> ClockdriftLevel {
        self.clockdrift_detector.clockdrift_level()
    }

    fn reset_internal(&mut self, reset_lag_aggregator: bool, reset_delay_confidence: bool) {
        if reset_lag_aggregator {
            self.matched_filter_lag_aggregator
                .reset(reset_delay_confidence);
        }
        self.matched_filter.reset(reset_lag_aggregator);
        self.old_aggregated_lag = None;
        self.consistent_estimate_counter = 0;
    }
}
