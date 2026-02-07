//! Estimation of the late reverb decay (T60).
//!
//! Ported from `modules/audio_processing/aec3/reverb_decay_estimator.h/cc`.

use crate::common::{FFT_LENGTH_BY_2, fast_approx_log2f};
use crate::config::EchoCanceller3Config;

const EARLY_REVERB_MIN_SIZE_BLOCKS: usize = 3;
const BLOCKS_PER_SECTION: usize = 6;

/// Linear regression approach assumes symmetric index around 0.
const EARLY_REVERB_FIRST_POINT_AT_LINEAR_REGRESSORS: f32 =
    -0.5 * BLOCKS_PER_SECTION as f32 * FFT_LENGTH_BY_2 as f32 + 0.5;

/// Averages the values in a block of size `FFT_LENGTH_BY_2`.
fn block_average(v: &[f32], block_index: usize) -> f32 {
    let one_by_fft_length_by_2 = 1.0 / FFT_LENGTH_BY_2 as f32;
    let i = block_index * FFT_LENGTH_BY_2;
    debug_assert!(v.len() >= i + FFT_LENGTH_BY_2);
    let sum: f32 = v[i..i + FFT_LENGTH_BY_2].iter().sum();
    sum * one_by_fft_length_by_2
}

/// Analyzes the gain in a block.
fn analyze_block_gain(
    h2: &[f32; FFT_LENGTH_BY_2],
    floor_gain: f32,
    previous_gain: &mut f32,
) -> (bool, bool) {
    let gain = block_average(h2, 0).max(1e-32);
    let block_adapting = *previous_gain > 1.1 * gain || *previous_gain < 0.9 * gain;
    let decaying_gain = gain > floor_gain;
    *previous_gain = gain;
    (block_adapting, decaying_gain)
}

/// Arithmetic sum of `2 * sum(i^2)` for `i` from 0.5 to `(N-1)/2`.
const fn symmetric_arithmetic_sum(n: i32) -> f32 {
    n as f32 * (n as f32 * n as f32 - 1.0) * (1.0 / 12.0)
}

/// Returns the peak energy of an impulse response block.
fn block_energy_peak(h: &[f32], peak_block: usize) -> f32 {
    debug_assert!(h.len() >= (peak_block + 1) * FFT_LENGTH_BY_2);
    let start = peak_block * FFT_LENGTH_BY_2;
    let end = start + FFT_LENGTH_BY_2;
    let peak_value = h[start..end]
        .iter()
        .copied()
        .max_by(|a, b| (a * a).partial_cmp(&(b * b)).unwrap())
        .unwrap_or(0.0);
    peak_value * peak_value
}

/// Returns the average energy of an impulse response block.
fn block_energy_average(h: &[f32], block_index: usize) -> f32 {
    debug_assert!(h.len() >= (block_index + 1) * FFT_LENGTH_BY_2);
    let one_by_fft_length_by_2 = 1.0 / FFT_LENGTH_BY_2 as f32;
    let start = block_index * FFT_LENGTH_BY_2;
    let end = start + FFT_LENGTH_BY_2;
    let sum_sq: f32 = h[start..end].iter().map(|&x| x * x).sum();
    sum_sq * one_by_fft_length_by_2
}

/// Estimates the decay of the late reverb using linear regression.
struct LateReverbLinearRegressor {
    nz: f32,
    nn: f32,
    count: f32,
    n_total: i32,
    n: i32,
}

impl LateReverbLinearRegressor {
    fn new() -> Self {
        Self {
            nz: 0.0,
            nn: 0.0,
            count: 0.0,
            n_total: 0,
            n: 0,
        }
    }

    /// Resets the estimator to receive `num_data_points` data points.
    fn reset(&mut self, num_data_points: i32) {
        debug_assert!(num_data_points >= 0);
        debug_assert!(num_data_points % 2 == 0);
        let n = num_data_points;
        self.nz = 0.0;
        self.nn = symmetric_arithmetic_sum(n);
        self.count = if n > 0 { -n as f32 * 0.5 + 0.5 } else { 0.0 };
        self.n_total = n;
        self.n = 0;
    }

    /// Accumulates a data point.
    fn accumulate(&mut self, z: f32) {
        self.nz += self.count * z;
        self.count += 1.0;
        self.n += 1;
    }

    /// Returns whether an estimate is available.
    fn estimate_available(&self) -> bool {
        self.n == self.n_total && self.n_total != 0
    }

    /// Returns the linear regression slope estimate.
    fn estimate(&self) -> f32 {
        debug_assert!(self.estimate_available());
        if self.nn == 0.0 {
            return 0.0;
        }
        self.nz / self.nn
    }
}

/// Identifies the length of the early reverb from the linear filter by
/// dividing the impulse response into overlapping sections and computing the
/// tilt of each section via linear regression.
struct EarlyReverbLengthEstimator {
    numerators_smooth: Vec<f32>,
    numerators: Vec<f32>,
    coefficients_counter: usize,
    block_counter: usize,
    n_sections: usize,
}

impl EarlyReverbLengthEstimator {
    fn new(max_blocks: usize) -> Self {
        let len = max_blocks.saturating_sub(BLOCKS_PER_SECTION);
        Self {
            numerators_smooth: vec![0.0; len],
            numerators: vec![0.0; len],
            coefficients_counter: 0,
            block_counter: 0,
            n_sections: 0,
        }
    }

    /// Resets the estimator.
    fn reset(&mut self) {
        self.coefficients_counter = 0;
        self.numerators.fill(0.0);
        self.block_counter = 0;
    }

    /// Accumulates estimation data.
    fn accumulate(&mut self, value: f32, smoothing: f32) {
        // Each section is composed of BLOCKS_PER_SECTION blocks and each
        // section overlaps with the next one in (BLOCKS_PER_SECTION - 1)
        // blocks.
        let first_section_index = self.block_counter.saturating_sub(BLOCKS_PER_SECTION - 1);
        let last_section_index = self.block_counter.min(self.numerators.len() - 1);
        let x_value =
            self.coefficients_counter as f32 + EARLY_REVERB_FIRST_POINT_AT_LINEAR_REGRESSORS;
        let value_to_inc = FFT_LENGTH_BY_2 as f32 * value;
        let mut value_to_add =
            x_value * value + (self.block_counter - last_section_index) as f32 * value_to_inc;

        let mut section = last_section_index as i64;
        while section >= first_section_index as i64 {
            self.numerators[section as usize] += value_to_add;
            value_to_add += value_to_inc;
            section -= 1;
        }

        self.coefficients_counter += 1;
        if self.coefficients_counter == FFT_LENGTH_BY_2 {
            if self.block_counter >= BLOCKS_PER_SECTION - 1 {
                let section = self.block_counter - (BLOCKS_PER_SECTION - 1);
                debug_assert!(section < self.numerators.len());
                debug_assert!(section < self.numerators_smooth.len());
                self.numerators_smooth[section] +=
                    smoothing * (self.numerators[section] - self.numerators_smooth[section]);
                self.n_sections = section + 1;
            }
            self.block_counter += 1;
            self.coefficients_counter = 0;
        }
    }

    /// Estimates the size in blocks of the early reverb.
    fn estimate(&self) -> usize {
        const N: f32 = BLOCKS_PER_SECTION as f32 * FFT_LENGTH_BY_2 as f32;
        let nn = symmetric_arithmetic_sum(N as i32);
        // log2(1.1) * nn / FFT_LENGTH_BY_2
        let numerator_11: f32 = 0.137_503_52 * nn / FFT_LENGTH_BY_2 as f32;
        // log2(0.8) * nn / FFT_LENGTH_BY_2
        let numerator_08: f32 = -0.321_928_1 * nn / FFT_LENGTH_BY_2 as f32;
        const NUM_SECTIONS_TO_ANALYZE: usize = 9;

        if self.n_sections < NUM_SECTIONS_TO_ANALYZE {
            return 0;
        }

        debug_assert!(self.n_sections <= self.numerators_smooth.len());
        let min_numerator_tail = self.numerators_smooth[NUM_SECTIONS_TO_ANALYZE..self.n_sections]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        let mut early_reverb_size_minus_1 = 0;
        for k in 0..NUM_SECTIONS_TO_ANALYZE {
            if (self.numerators_smooth[k] > numerator_11)
                || (self.numerators_smooth[k] < numerator_08
                    && self.numerators_smooth[k] < 0.9 * min_numerator_tail)
            {
                early_reverb_size_minus_1 = k;
            }
        }

        if early_reverb_size_minus_1 == 0 {
            0
        } else {
            early_reverb_size_minus_1 + 1
        }
    }
}

/// Estimates the decay of the late reverb from the adaptive filter.
pub(crate) struct ReverbDecayEstimator {
    filter_length_blocks: usize,
    filter_length_coefficients: usize,
    use_adaptive_echo_decay: bool,
    late_reverb_decay_estimator: LateReverbLinearRegressor,
    early_reverb_estimator: EarlyReverbLengthEstimator,
    late_reverb_start: usize,
    late_reverb_end: usize,
    block_to_analyze: usize,
    estimation_region_candidate_size: usize,
    estimation_region_identified: bool,
    previous_gains: Vec<f32>,
    decay: f32,
    mild_decay: f32,
    tail_gain: f32,
    smoothing_constant: f32,
}

impl ReverbDecayEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config) -> Self {
        let filter_length_blocks = config.filter.refined.length_blocks;
        debug_assert!(filter_length_blocks > EARLY_REVERB_MIN_SIZE_BLOCKS);
        Self {
            filter_length_blocks,
            filter_length_coefficients: filter_length_blocks * FFT_LENGTH_BY_2,
            use_adaptive_echo_decay: config.ep_strength.default_len < 0.0,
            late_reverb_decay_estimator: LateReverbLinearRegressor::new(),
            early_reverb_estimator: EarlyReverbLengthEstimator::new(
                filter_length_blocks - EARLY_REVERB_MIN_SIZE_BLOCKS,
            ),
            late_reverb_start: EARLY_REVERB_MIN_SIZE_BLOCKS,
            late_reverb_end: EARLY_REVERB_MIN_SIZE_BLOCKS,
            block_to_analyze: 0,
            estimation_region_candidate_size: 0,
            estimation_region_identified: false,
            previous_gains: vec![0.0; filter_length_blocks],
            decay: config.ep_strength.default_len.abs(),
            mild_decay: config.ep_strength.nearend_len.abs(),
            tail_gain: 0.0,
            smoothing_constant: 0.0,
        }
    }

    /// Updates the decay estimate.
    pub(crate) fn update(
        &mut self,
        filter: &[f32],
        filter_quality: Option<f32>,
        filter_delay_blocks: i32,
        usable_linear_filter: bool,
        stationary_signal: bool,
    ) {
        if stationary_signal {
            return;
        }

        let filter_size = filter.len() as i32;
        let mut estimation_feasible = filter_delay_blocks
            <= self.filter_length_blocks as i32 - EARLY_REVERB_MIN_SIZE_BLOCKS as i32 - 1;
        estimation_feasible =
            estimation_feasible && filter_size == self.filter_length_coefficients as i32;
        estimation_feasible = estimation_feasible && filter_delay_blocks > 0;
        estimation_feasible = estimation_feasible && usable_linear_filter;

        if !estimation_feasible {
            self.reset_decay_estimation();
            return;
        }

        if !self.use_adaptive_echo_decay {
            return;
        }

        let new_smoothing = filter_quality.map_or(0.0, |q| q * 0.2);
        self.smoothing_constant = new_smoothing.max(self.smoothing_constant);
        if self.smoothing_constant == 0.0 {
            return;
        }

        if self.block_to_analyze < self.filter_length_blocks {
            self.analyze_filter(filter);
            self.block_to_analyze += 1;
        } else {
            self.estimate_decay(filter, filter_delay_blocks as usize);
        }
    }

    /// Returns the decay for the exponential model. When `mild` is true,
    /// returns a milder decay (unless adaptive echo decay is used).
    pub(crate) fn decay(&self, mild: bool) -> f32 {
        if self.use_adaptive_echo_decay {
            self.decay
        } else if mild {
            self.mild_decay
        } else {
            self.decay
        }
    }

    fn reset_decay_estimation(&mut self) {
        self.early_reverb_estimator.reset();
        self.late_reverb_decay_estimator.reset(0);
        self.block_to_analyze = 0;
        self.estimation_region_candidate_size = 0;
        self.estimation_region_identified = false;
        self.smoothing_constant = 0.0;
        self.late_reverb_start = 0;
        self.late_reverb_end = 0;
    }

    fn estimate_decay(&mut self, filter: &[f32], peak_block: usize) {
        debug_assert_eq!(0, filter.len() % FFT_LENGTH_BY_2);

        // Reset the block analysis counter.
        self.block_to_analyze =
            (peak_block + EARLY_REVERB_MIN_SIZE_BLOCKS).min(self.filter_length_blocks);

        let first_reverb_gain = block_energy_average(filter, self.block_to_analyze);
        let h_size_blocks = filter.len() >> crate::common::FFT_LENGTH_BY_2_LOG2;
        self.tail_gain = block_energy_average(filter, h_size_blocks - 1);
        let peak_energy = block_energy_peak(filter, peak_block);
        let sufficient_reverb_decay = first_reverb_gain > 4.0 * self.tail_gain;
        let valid_filter = first_reverb_gain > 2.0 * self.tail_gain && peak_energy < 100.0;

        // Estimate the size of the regions with early and late reflections.
        let size_early_reverb = self.early_reverb_estimator.estimate();
        let size_late_reverb = (self.estimation_region_candidate_size as i64
            - size_early_reverb as i64)
            .max(0) as usize;

        // Only update if the late reverb region is sufficiently large.
        if size_late_reverb >= 5 {
            if valid_filter && self.late_reverb_decay_estimator.estimate_available() {
                let mut decay = 2.0f32
                    .powf(self.late_reverb_decay_estimator.estimate() * FFT_LENGTH_BY_2 as f32);
                const MAX_DECAY: f32 = 0.95;
                const MIN_DECAY: f32 = 0.02;
                decay = decay.max(0.97 * self.decay);
                decay = decay.min(MAX_DECAY);
                decay = decay.max(MIN_DECAY);
                self.decay += self.smoothing_constant * (decay - self.decay);
            }

            self.late_reverb_decay_estimator
                .reset((size_late_reverb * FFT_LENGTH_BY_2) as i32);
            self.late_reverb_start = peak_block + EARLY_REVERB_MIN_SIZE_BLOCKS + size_early_reverb;
            self.late_reverb_end =
                self.block_to_analyze + self.estimation_region_candidate_size - 1;
        } else {
            self.late_reverb_decay_estimator.reset(0);
            self.late_reverb_start = 0;
            self.late_reverb_end = 0;
        }

        // Reset variables for identification of the reverb decay estimation region.
        self.estimation_region_identified = !(valid_filter && sufficient_reverb_decay);
        self.estimation_region_candidate_size = 0;

        // Stop estimation until another good filter is received.
        self.smoothing_constant = 0.0;

        // Reset early reflections detector.
        self.early_reverb_estimator.reset();
    }

    fn analyze_filter(&mut self, filter: &[f32]) {
        let start = self.block_to_analyze * FFT_LENGTH_BY_2;
        let h = &filter[start..start + FFT_LENGTH_BY_2];

        // Compute squared filter coefficients for the block.
        let mut h2 = [0.0f32; FFT_LENGTH_BY_2];
        for (out, &val) in h2.iter_mut().zip(h.iter()) {
            *out = val * val;
        }

        // Map out the region for estimating the reverb decay.
        let (adapting, above_noise_floor) = analyze_block_gain(
            &h2,
            self.tail_gain,
            &mut self.previous_gains[self.block_to_analyze],
        );

        // Count consecutive "good" filter sections.
        self.estimation_region_identified =
            self.estimation_region_identified || adapting || !above_noise_floor;
        if !self.estimation_region_identified {
            self.estimation_region_candidate_size += 1;
        }

        // Accumulate data for reverb decay estimation and early reflections.
        if self.block_to_analyze <= self.late_reverb_end {
            if self.block_to_analyze >= self.late_reverb_start {
                for &h2_k in &h2 {
                    let h2_log2 = fast_approx_log2f(h2_k + 1e-10);
                    self.late_reverb_decay_estimator.accumulate(h2_log2);
                    self.early_reverb_estimator
                        .accumulate(h2_log2, self.smoothing_constant);
                }
            } else {
                for &h2_k in &h2 {
                    let h2_log2 = fast_approx_log2f(h2_k + 1e-10);
                    self.early_reverb_estimator
                        .accumulate(h2_log2, self.smoothing_constant);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_returns_configured_value() {
        let config = EchoCanceller3Config::default();
        let estimator = ReverbDecayEstimator::new(&config);
        let expected = config.ep_strength.default_len.abs();
        assert!((estimator.decay(false) - expected).abs() < 1e-6);
    }

    #[test]
    fn mild_decay_returns_nearend_len() {
        let config = EchoCanceller3Config::default();
        let estimator = ReverbDecayEstimator::new(&config);
        let expected = config.ep_strength.nearend_len.abs();
        assert!((estimator.decay(true) - expected).abs() < 1e-6);
    }

    #[test]
    fn stationary_signal_skips_update() {
        let config = EchoCanceller3Config::default();
        let mut estimator = ReverbDecayEstimator::new(&config);
        let filter = vec![0.0f32; config.filter.refined.length_blocks * FFT_LENGTH_BY_2];
        let initial_decay = estimator.decay(false);
        estimator.update(&filter, Some(1.0), 5, true, true);
        assert_eq!(estimator.decay(false), initial_decay);
    }

    #[test]
    fn linear_regressor_basic() {
        let mut reg = LateReverbLinearRegressor::new();
        reg.reset(4);
        // Feed in a linear sequence: z = count * slope + offset
        // With symmetric indices around 0: -1.5, -0.5, 0.5, 1.5
        // Feed z values that form a known slope.
        reg.accumulate(1.0);
        reg.accumulate(2.0);
        reg.accumulate(3.0);
        reg.accumulate(4.0);
        assert!(reg.estimate_available());
        // The slope should be positive.
        assert!(reg.estimate() > 0.0);
    }

    #[test]
    fn early_reverb_estimator_returns_zero_with_insufficient_sections() {
        let estimator = EarlyReverbLengthEstimator::new(10);
        assert_eq!(estimator.estimate(), 0);
    }
}
