//! Adaptive filter property analyzer.
//!
//! Analyzes the properties of the adaptive filter to determine delay, gain,
//! and consistency of the echo path estimate.
//!
//! Ported from `modules/audio_processing/aec3/filter_analyzer.h/cc`.

use crate::block::Block;
use crate::common::{BLOCK_SIZE, BLOCK_SIZE_LOG2, FFT_LENGTH_BY_2, NUM_BLOCKS_PER_SECOND};
use crate::config::EchoCanceller3Config;
use crate::render_buffer::RenderBuffer;

/// Finds the index of the peak (largest squared value) in the filter.
fn find_peak_index(
    filter_time_domain: &[f32],
    peak_index_in: usize,
    start_sample: usize,
    end_sample: usize,
) -> usize {
    let mut peak_index_out = peak_index_in;
    let mut max_h2 = filter_time_domain[peak_index_out] * filter_time_domain[peak_index_out];
    for k in start_sample..=end_sample {
        let tmp = filter_time_domain[k] * filter_time_domain[k];
        if tmp > max_h2 {
            peak_index_out = k;
            max_h2 = tmp;
        }
    }
    peak_index_out
}

/// Region of filter samples to analyze in the current call.
#[derive(Debug, Clone, Copy)]
struct FilterRegion {
    start_sample: usize,
    end_sample: usize,
}

/// Detects whether the filter impulse response shape has been consistent
/// over time.
struct ConsistentFilterDetector {
    significant_peak: bool,
    filter_floor_accum: f32,
    filter_secondary_peak: f32,
    filter_floor_low_limit: usize,
    filter_floor_high_limit: usize,
    active_render_threshold: f32,
    consistent_estimate_counter: usize,
    consistent_delay_reference: i32,
}

impl ConsistentFilterDetector {
    fn new(config: &EchoCanceller3Config) -> Self {
        let mut det = Self {
            significant_peak: false,
            filter_floor_accum: 0.0,
            filter_secondary_peak: 0.0,
            filter_floor_low_limit: 0,
            filter_floor_high_limit: 0,
            active_render_threshold: config.render_levels.active_render_limit
                * config.render_levels.active_render_limit
                * FFT_LENGTH_BY_2 as f32,
            consistent_estimate_counter: 0,
            consistent_delay_reference: -10,
        };
        det.reset();
        det
    }

    fn reset(&mut self) {
        self.significant_peak = false;
        self.filter_floor_accum = 0.0;
        self.filter_secondary_peak = 0.0;
        self.filter_floor_low_limit = 0;
        self.filter_floor_high_limit = 0;
        self.consistent_estimate_counter = 0;
        self.consistent_delay_reference = -10;
    }

    fn detect(
        &mut self,
        filter_to_analyze: &[f32],
        region: &FilterRegion,
        x_block: &Block,
        peak_index: usize,
        delay_blocks: i32,
    ) -> bool {
        if region.start_sample == 0 {
            self.filter_floor_accum = 0.0;
            self.filter_secondary_peak = 0.0;
            self.filter_floor_low_limit = if peak_index < 64 { 0 } else { peak_index - 64 };
            self.filter_floor_high_limit = if peak_index > filter_to_analyze.len() - 129 {
                0
            } else {
                peak_index + 128
            };
        }

        let mut filter_floor_accum = self.filter_floor_accum;
        let mut filter_secondary_peak = self.filter_secondary_peak;

        for k in region.start_sample..(region.end_sample + 1).min(self.filter_floor_low_limit) {
            let abs_h = filter_to_analyze[k].abs();
            filter_floor_accum += abs_h;
            filter_secondary_peak = filter_secondary_peak.max(abs_h);
        }

        for k in self.filter_floor_high_limit.max(region.start_sample)..=region.end_sample {
            let abs_h = filter_to_analyze[k].abs();
            filter_floor_accum += abs_h;
            filter_secondary_peak = filter_secondary_peak.max(abs_h);
        }

        self.filter_floor_accum = filter_floor_accum;
        self.filter_secondary_peak = filter_secondary_peak;

        if region.end_sample == filter_to_analyze.len() - 1 {
            let filter_floor = self.filter_floor_accum
                / (self.filter_floor_low_limit + filter_to_analyze.len()
                    - self.filter_floor_high_limit) as f32;

            let abs_peak = filter_to_analyze[peak_index].abs();
            self.significant_peak =
                abs_peak > 10.0 * filter_floor && abs_peak > 2.0 * self.filter_secondary_peak;
        }

        if self.significant_peak {
            let mut active_render_block = false;
            for ch in 0..x_block.num_channels() {
                let x_channel = x_block.view(0, ch);
                let x_energy: f32 = x_channel.iter().map(|&v| v * v).sum();
                if x_energy > self.active_render_threshold {
                    active_render_block = true;
                    break;
                }
            }

            if self.consistent_delay_reference == delay_blocks {
                if active_render_block {
                    self.consistent_estimate_counter += 1;
                }
            } else {
                self.consistent_estimate_counter = 0;
                self.consistent_delay_reference = delay_blocks;
            }
        }

        self.consistent_estimate_counter as f32 > 1.5 * NUM_BLOCKS_PER_SECOND as f32
    }
}

/// Per-channel filter analysis state.
struct FilterAnalysisState {
    gain: f32,
    peak_index: usize,
    filter_length_blocks: usize,
    consistent_estimate: bool,
    consistent_filter_detector: ConsistentFilterDetector,
}

impl FilterAnalysisState {
    fn new(config: &EchoCanceller3Config) -> Self {
        let mut state = Self {
            gain: config.ep_strength.default_gain,
            peak_index: 0,
            filter_length_blocks: config.filter.refined_initial.length_blocks,
            consistent_estimate: false,
            consistent_filter_detector: ConsistentFilterDetector::new(config),
        };
        state.reset(config.ep_strength.default_gain);
        state
    }

    fn reset(&mut self, default_gain: f32) {
        self.peak_index = 0;
        self.gain = default_gain;
        self.consistent_filter_detector.reset();
    }
}

/// Analyzes the properties of the adaptive filter.
pub(crate) struct FilterAnalyzer {
    bounded_erl: bool,
    default_gain: f32,
    h_highpass: Vec<Vec<f32>>,
    blocks_since_reset: usize,
    region: FilterRegion,
    filter_analysis_states: Vec<FilterAnalysisState>,
    filter_delays_blocks: Vec<i32>,
    min_filter_delay_blocks: i32,
}

impl FilterAnalyzer {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let time_domain_len =
            crate::common::get_time_domain_length(config.filter.refined.length_blocks);
        let mut analyzer = Self {
            bounded_erl: config.ep_strength.bounded_erl,
            default_gain: config.ep_strength.default_gain,
            h_highpass: vec![vec![0.0; time_domain_len]; num_capture_channels],
            blocks_since_reset: 0,
            region: FilterRegion {
                start_sample: 0,
                end_sample: 0,
            },
            filter_analysis_states: (0..num_capture_channels)
                .map(|_| FilterAnalysisState::new(config))
                .collect(),
            filter_delays_blocks: vec![0i32; num_capture_channels],
            min_filter_delay_blocks: 0,
        };
        analyzer.reset();
        analyzer
    }

    /// Resets the analysis.
    pub(crate) fn reset(&mut self) {
        self.blocks_since_reset = 0;
        self.reset_region();
        for state in &mut self.filter_analysis_states {
            state.reset(self.default_gain);
        }
        self.filter_delays_blocks.fill(0);
    }

    /// Updates the estimates with new input data.
    pub(crate) fn update(
        &mut self,
        filters_time_domain: &[Vec<f32>],
        render_buffer: &RenderBuffer<'_>,
        any_filter_consistent: &mut bool,
        max_echo_path_gain: &mut f32,
    ) {
        debug_assert_eq!(filters_time_domain.len(), self.filter_analysis_states.len());
        debug_assert_eq!(filters_time_domain.len(), self.h_highpass.len());

        self.blocks_since_reset += 1;
        self.set_region_to_analyze(filters_time_domain[0].len());
        self.analyze_region(filters_time_domain, render_buffer);

        // Aggregate the results for all capture channels.
        let st_ch0 = &self.filter_analysis_states[0];
        *any_filter_consistent = st_ch0.consistent_estimate;
        *max_echo_path_gain = st_ch0.gain;
        self.min_filter_delay_blocks = self.filter_delays_blocks[0];
        for ch in 1..filters_time_domain.len() {
            let st_ch = &self.filter_analysis_states[ch];
            *any_filter_consistent = *any_filter_consistent || st_ch.consistent_estimate;
            *max_echo_path_gain = max_echo_path_gain.max(st_ch.gain);
            self.min_filter_delay_blocks = self
                .min_filter_delay_blocks
                .min(self.filter_delays_blocks[ch]);
        }
    }

    /// Returns the delay in blocks for each filter.
    pub(crate) fn filter_delays_blocks(&self) -> &[i32] {
        &self.filter_delays_blocks
    }

    /// Returns the minimum delay of all filters in terms of blocks.
    pub(crate) fn min_filter_delay_blocks(&self) -> i32 {
        self.min_filter_delay_blocks
    }

    /// Returns the number of blocks for the current used filter.
    pub(crate) fn filter_length_blocks(&self) -> usize {
        self.filter_analysis_states[0].filter_length_blocks
    }

    /// Returns the preprocessed filter.
    pub(crate) fn get_adjusted_filters(&self) -> &[Vec<f32>] {
        &self.h_highpass
    }

    /// Sets the region of the filter to analyze. Public for testing.
    pub(crate) fn set_region_to_analyze(&mut self, filter_size: usize) {
        const NUMBER_BLOCKS_TO_UPDATE: usize = 1;
        let r = &mut self.region;
        r.start_sample = if r.end_sample >= filter_size - 1 {
            0
        } else {
            r.end_sample + 1
        };
        r.end_sample =
            (r.start_sample + NUMBER_BLOCKS_TO_UPDATE * BLOCK_SIZE - 1).min(filter_size - 1);

        debug_assert!(r.start_sample < filter_size);
        debug_assert!(r.end_sample < filter_size);
        debug_assert!(r.start_sample <= r.end_sample);
    }

    fn analyze_region(
        &mut self,
        filters_time_domain: &[Vec<f32>],
        render_buffer: &RenderBuffer<'_>,
    ) {
        self.pre_process_filters(filters_time_domain);

        let one_by_block_size: f32 = 1.0 / BLOCK_SIZE as f32;
        for ch in 0..filters_time_domain.len() {
            debug_assert!(self.region.start_sample < filters_time_domain[ch].len());
            debug_assert!(self.region.end_sample < filters_time_domain[ch].len());
            debug_assert_eq!(self.h_highpass[ch].len(), filters_time_domain[ch].len());
            debug_assert!(!self.h_highpass[ch].is_empty());

            let st_ch = &mut self.filter_analysis_states[ch];
            st_ch.peak_index = st_ch.peak_index.min(self.h_highpass[ch].len() - 1);

            st_ch.peak_index = find_peak_index(
                &self.h_highpass[ch],
                st_ch.peak_index,
                self.region.start_sample,
                self.region.end_sample,
            );
            self.filter_delays_blocks[ch] = (st_ch.peak_index >> BLOCK_SIZE_LOG2) as i32;

            let h_highpass_ch = &self.h_highpass[ch];
            let delay_blocks = self.filter_delays_blocks[ch];
            Self::update_filter_gain(
                h_highpass_ch,
                st_ch,
                self.blocks_since_reset,
                self.bounded_erl,
            );

            st_ch.filter_length_blocks =
                (filters_time_domain[ch].len() as f32 * one_by_block_size) as usize;

            let region = self.region;
            let x_block = render_buffer.get_block(-delay_blocks);
            st_ch.consistent_estimate = st_ch.consistent_filter_detector.detect(
                h_highpass_ch,
                &region,
                x_block,
                st_ch.peak_index,
                delay_blocks,
            );
        }
    }

    fn update_filter_gain(
        filter_time_domain: &[f32],
        st: &mut FilterAnalysisState,
        blocks_since_reset: usize,
        bounded_erl: bool,
    ) {
        let sufficient_time_to_converge = blocks_since_reset > 5 * NUM_BLOCKS_PER_SECOND;

        if sufficient_time_to_converge && st.consistent_estimate {
            st.gain = filter_time_domain[st.peak_index].abs();
        } else {
            // TODO(peah): Verify whether this check against a float is ok.
            if st.gain != 0.0 {
                st.gain = st.gain.max(filter_time_domain[st.peak_index].abs());
            }
        }

        if bounded_erl && st.gain != 0.0 {
            st.gain = st.gain.max(0.01);
        }
    }

    fn pre_process_filters(&mut self, filters_time_domain: &[Vec<f32>]) {
        // Minimum phase high-pass filter with cutoff frequency at about 600 Hz.
        const H: [f32; 3] = [0.792_974_2, -0.360_721_28, -0.470_477_66];

        for ch in 0..filters_time_domain.len() {
            debug_assert!(self.region.start_sample < filters_time_domain[ch].len());
            debug_assert!(self.region.end_sample < filters_time_domain[ch].len());

            self.h_highpass[ch].resize(filters_time_domain[ch].len(), 0.0);

            // Clear the region first.
            for k in self.region.start_sample..=self.region.end_sample {
                self.h_highpass[ch][k] = 0.0;
            }

            let region_end = self.region.end_sample;
            let start = (H.len() - 1).max(self.region.start_sample);
            for k in start..=region_end {
                let mut tmp = self.h_highpass[ch][k];
                for (j, &h_coeff) in H.iter().enumerate() {
                    tmp += filters_time_domain[ch][k - j] * h_coeff;
                }
                self.h_highpass[ch][k] = tmp;
            }
        }
    }

    fn reset_region(&mut self) {
        self.region.start_sample = 0;
        self.region.end_sample = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the filter analyzer handles filter resizes properly.
    #[test]
    fn filter_resize() {
        let c = EchoCanceller3Config::default();
        let mut filter = vec![0.0f32; 65];
        for num_capture_channels in [1, 2, 4] {
            let mut fa = FilterAnalyzer::new(&c, num_capture_channels);
            fa.set_region_to_analyze(filter.len());
            fa.set_region_to_analyze(filter.len());
            filter.resize(32, 0.0);
            fa.set_region_to_analyze(filter.len());
        }
    }
}
