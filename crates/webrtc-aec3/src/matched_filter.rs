//! Matched filter for delay estimation via cross-correlation.
//!
//! Produces recursively updated cross-correlation estimates for several signal
//! shifts where the intra-shift spacing is uniform. The core algorithm is NLMS
//! (Normalized Least Mean Squares).
//!
//! Ported from `modules/audio_processing/aec3/matched_filter.h/cc`.

use crate::common::BLOCK_SIZE;
use crate::downsampled_render_buffer::DownsampledRenderBuffer;
use webrtc_simd::SimdBackend;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;

/// Subsample rate for computing accumulated error (pre-echo detection).
const ACCUMULATED_ERROR_SUB_SAMPLE_RATE: usize = 4;

/// Smoothing constant for accumulated error increases.
const SMOOTH_CONSTANT_INCREASES: f32 = 0.015;

/// Pre-echo detection threshold.
const PRE_ECHO_THRESHOLD: f32 = 0.5;

/// Lag estimate from the matched filter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LagEstimate {
    pub lag: usize,
    pub pre_echo_lag: usize,
}

impl LagEstimate {
    pub(crate) fn new(lag: usize, pre_echo_lag: usize) -> Self {
        Self { lag, pre_echo_lag }
    }
}

/// Scalar implementation of the matched filter core.
///
/// Performs NLMS cross-correlation of filter `h` with render signal `x` and
/// capture signal `y`. Optionally computes accumulated error for pre-echo
/// detection.
pub(crate) fn matched_filter_core(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
    compute_accumulated_error: bool,
    accumulated_error: &mut [f32],
) {
    if compute_accumulated_error {
        accumulated_error.fill(0.0);
    }

    let x_size = x.len();
    let h_size = h.len();

    // Process for all samples in the sub-block.
    for &y_i in y {
        // Apply the matched filter as filter * x, and compute x * x.
        let mut x2_sum = 0.0f32;
        let mut s = 0.0f32;
        let mut x_index = x_start_index;

        if compute_accumulated_error {
            for k in 0..h_size {
                x2_sum += x[x_index] * x[x_index];
                s += h[k] * x[x_index];
                x_index = if x_index < x_size - 1 { x_index + 1 } else { 0 };
                // Every 4 samples, record the accumulated error.
                if (k + 1) & 0b11 == 0 {
                    let idx = k >> 2;
                    let e = y_i - s;
                    accumulated_error[idx] += e * e;
                }
            }
        } else {
            for k in 0..h_size {
                x2_sum += x[x_index] * x[x_index];
                s += h[k] * x[x_index];
                x_index = if x_index < x_size - 1 { x_index + 1 } else { 0 };
            }
        }

        // Compute the matched filter error.
        let e = y_i - s;
        let saturation = y_i >= 32000.0 || y_i <= -32000.0;
        *error_sum += e * e;

        // Update the matched filter estimate in an NLMS manner.
        if x2_sum > x2_sum_threshold && !saturation {
            debug_assert!(x2_sum > 0.0);
            let alpha = smoothing * e / x2_sum;

            // filter = filter + smoothing * (y - filter * x) * x / (x * x)
            let mut x_index2 = x_start_index;
            for k in 0..h_size {
                h[k] += alpha * x[x_index2];
                x_index2 = if x_index2 < x_size - 1 {
                    x_index2 + 1
                } else {
                    0
                };
            }
            *filters_updated = true;
        }

        x_start_index = if x_start_index > 0 {
            x_start_index - 1
        } else {
            x_size - 1
        };
    }
}

/// SIMD-dispatched matched filter core.
///
/// Selects the best available implementation based on `backend`.
/// Falls back to scalar when no SIMD path matches.
#[allow(clippy::too_many_arguments, reason = "matches C++ function signature")]
pub(crate) fn matched_filter_core_dispatch(
    backend: SimdBackend,
    x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
    compute_accumulated_error: bool,
    accumulated_error: &mut [f32],
    scratch_memory: &mut [f32],
) {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx2 => {
            if compute_accumulated_error {
                // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma.
                unsafe {
                    avx2::matched_filter_core_accumulated_error(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                        accumulated_error,
                        scratch_memory,
                    );
                }
            } else {
                // SAFETY: detect_backend() only returns Avx2 after confirming avx2+fma.
                unsafe {
                    avx2::matched_filter_core(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                    );
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Sse2 => {
            if compute_accumulated_error {
                // SAFETY: detect_backend() only returns Sse2 after confirming sse2.
                unsafe {
                    sse2::matched_filter_core_accumulated_error(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                        accumulated_error,
                        scratch_memory,
                    );
                }
            } else {
                // SAFETY: detect_backend() only returns Sse2 after confirming sse2.
                unsafe {
                    sse2::matched_filter_core(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                    );
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            if compute_accumulated_error {
                // SAFETY: NEON is always available on aarch64.
                unsafe {
                    neon::matched_filter_core_accumulated_error(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                        accumulated_error,
                        scratch_memory,
                    );
                }
            } else {
                // SAFETY: NEON is always available on aarch64.
                unsafe {
                    neon::matched_filter_core(
                        x_start_index,
                        x2_sum_threshold,
                        smoothing,
                        x,
                        y,
                        h,
                        filters_updated,
                        error_sum,
                    );
                }
            }
        }
        _ => {
            matched_filter_core(
                x_start_index,
                x2_sum_threshold,
                smoothing,
                x,
                y,
                h,
                filters_updated,
                error_sum,
                compute_accumulated_error,
                accumulated_error,
            );
        }
    }
}

/// Find the index of the element with the largest squared value.
///
/// Uses even/odd tracking for better compiler optimization, matching the C++
/// implementation.
pub(crate) fn max_square_peak_index(h: &[f32]) -> usize {
    if h.len() < 2 {
        return 0;
    }

    let mut max_element1 = h[0] * h[0];
    let mut max_element2 = h[1] * h[1];
    let mut lag_estimate1: usize = 0;
    let mut lag_estimate2: usize = 1;
    let last_index = h.len() - 1;

    // Track even and odd max elements separately.
    let mut k = 2;
    while k < last_index {
        let element1 = h[k] * h[k];
        let element2 = h[k + 1] * h[k + 1];
        if element1 > max_element1 {
            max_element1 = element1;
            lag_estimate1 = k;
        }
        if element2 > max_element2 {
            max_element2 = element2;
            lag_estimate2 = k + 1;
        }
        k += 2;
    }

    if max_element2 > max_element1 {
        max_element1 = max_element2;
        lag_estimate1 = lag_estimate2;
    }

    // In case of odd h size, check the last element.
    let last_element = h[last_index] * h[last_index];
    if last_element > max_element1 {
        return last_index;
    }
    lag_estimate1
}

fn update_accumulated_error(
    instantaneous: &[f32],
    accumulated: &mut [f32],
    one_over_error_sum_anchor: f32,
) {
    for (acc, &inst) in accumulated.iter_mut().zip(instantaneous.iter()) {
        let error_norm = inst * one_over_error_sum_anchor;
        if error_norm < *acc {
            *acc = error_norm;
        } else {
            *acc += SMOOTH_CONSTANT_INCREASES * (error_norm - *acc);
        }
    }
}

fn compute_pre_echo_lag(
    accumulated_error: &[f32],
    lag: usize,
    alignment_shift_winner: usize,
) -> usize {
    debug_assert!(lag >= alignment_shift_winner);
    let mut pre_echo_lag_estimate = lag - alignment_shift_winner;
    let maximum_pre_echo_lag =
        (pre_echo_lag_estimate / ACCUMULATED_ERROR_SUB_SAMPLE_RATE).min(accumulated_error.len());

    for k in (0..maximum_pre_echo_lag as i32).rev() {
        let k = k as usize;
        if accumulated_error[k] > PRE_ECHO_THRESHOLD {
            break;
        }
        pre_echo_lag_estimate = (k + 1) * ACCUMULATED_ERROR_SUB_SAMPLE_RATE - 1;
    }
    pre_echo_lag_estimate + alignment_shift_winner
}

/// Matched filter for delay estimation.
///
/// Produces recursively updated cross-correlation estimates for several signal
/// shifts where the intra-shift spacing is uniform.
pub(crate) struct MatchedFilter {
    backend: SimdBackend,
    sub_block_size: usize,
    filter_intra_lag_shift: usize,
    filters: Vec<Vec<f32>>,
    accumulated_error: Vec<Vec<f32>>,
    instantaneous_accumulated_error: Vec<f32>,
    scratch_memory: Vec<f32>,
    reported_lag_estimate: Option<LagEstimate>,
    winner_lag: Option<usize>,
    last_detected_best_lag_filter: i32,
    excitation_limit: f32,
    smoothing_fast: f32,
    smoothing_slow: f32,
    matching_filter_threshold: f32,
    detect_pre_echo: bool,
    number_pre_echo_updates: i32,
}

impl MatchedFilter {
    #[allow(
        clippy::too_many_arguments,
        reason = "matches C++ constructor signature"
    )]
    pub(crate) fn new(
        backend: SimdBackend,
        sub_block_size: usize,
        window_size_sub_blocks: usize,
        num_matched_filters: usize,
        alignment_shift_sub_blocks: usize,
        excitation_limit: f32,
        smoothing_fast: f32,
        smoothing_slow: f32,
        matching_filter_threshold: f32,
        detect_pre_echo: bool,
    ) -> Self {
        debug_assert!(window_size_sub_blocks > 0);
        debug_assert!(BLOCK_SIZE % sub_block_size == 0);
        debug_assert!(sub_block_size % 4 == 0);

        let filter_intra_lag_shift = alignment_shift_sub_blocks * sub_block_size;
        let filter_size = window_size_sub_blocks * sub_block_size;

        let (accumulated_error, instantaneous_accumulated_error) = if detect_pre_echo {
            let acc_size = filter_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE;
            (
                vec![vec![1.0f32; acc_size]; num_matched_filters],
                vec![0.0f32; acc_size],
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Always allocate scratch_memory — SIMD paths use it to linearize
        // the circular buffer even outside the accumulated-error path.
        let scratch_memory = vec![0.0f32; filter_size];

        Self {
            backend,
            sub_block_size,
            filter_intra_lag_shift,
            filters: vec![vec![0.0f32; filter_size]; num_matched_filters],
            accumulated_error,
            instantaneous_accumulated_error,
            scratch_memory,
            reported_lag_estimate: None,
            winner_lag: None,
            last_detected_best_lag_filter: -1,
            excitation_limit,
            smoothing_fast,
            smoothing_slow,
            matching_filter_threshold,
            detect_pre_echo,
            number_pre_echo_updates: 0,
        }
    }

    /// Resets the matched filter.
    pub(crate) fn reset(&mut self, full_reset: bool) {
        for f in &mut self.filters {
            f.fill(0.0);
        }
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        if full_reset {
            for e in &mut self.accumulated_error {
                e.fill(1.0);
            }
            self.number_pre_echo_updates = 0;
        }
    }

    /// Updates the correlation with the values in the capture buffer.
    pub(crate) fn update(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        capture: &[f32],
        use_slow_smoothing: bool,
    ) {
        debug_assert_eq!(self.sub_block_size, capture.len());

        let smoothing = if use_slow_smoothing {
            self.smoothing_slow
        } else {
            self.smoothing_fast
        };

        let x2_sum_threshold =
            self.filters[0].len() as f32 * self.excitation_limit * self.excitation_limit;

        // Compute anchor for the matched filter error.
        let error_sum_anchor: f32 = capture.iter().map(|&y| y * y).sum();

        // Apply all matched filters.
        let mut winner_error_sum = error_sum_anchor;
        self.winner_lag = None;
        self.reported_lag_estimate = None;
        let mut alignment_shift: usize = 0;
        let mut previous_lag_estimate: Option<usize> = None;
        let num_filters = self.filters.len();
        let mut winner_index: i32 = -1;

        for n in 0..num_filters {
            let mut error_sum = 0.0f32;
            let mut filters_updated = false;
            let compute_pre_echo =
                self.detect_pre_echo && n as i32 == self.last_detected_best_lag_filter;

            let x_start_index = (render_buffer.read + alignment_shift + self.sub_block_size - 1)
                % render_buffer.buffer.len();

            matched_filter_core_dispatch(
                self.backend,
                x_start_index,
                x2_sum_threshold,
                smoothing,
                &render_buffer.buffer,
                capture,
                &mut self.filters[n],
                &mut filters_updated,
                &mut error_sum,
                compute_pre_echo,
                &mut self.instantaneous_accumulated_error,
                &mut self.scratch_memory,
            );

            // Estimate the lag as the peak of the matched filter.
            let lag_estimate = max_square_peak_index(&self.filters[n]);
            let reliable = lag_estimate > 2
                && lag_estimate < self.filters[n].len() - 10
                && error_sum < self.matching_filter_threshold * error_sum_anchor;

            let lag = lag_estimate + alignment_shift;
            if filters_updated && reliable && error_sum < winner_error_sum {
                winner_error_sum = error_sum;
                winner_index = n as i32;
                // In case 2 matched filters return the same winner candidate
                // (overlap region), choose the one with the smaller index.
                if previous_lag_estimate == Some(lag) {
                    self.winner_lag = previous_lag_estimate;
                    winner_index = n as i32 - 1;
                } else {
                    self.winner_lag = Some(lag);
                }
            }
            previous_lag_estimate = Some(lag);
            alignment_shift += self.filter_intra_lag_shift;
        }

        if winner_index != -1 {
            let winner_lag = self
                .winner_lag
                .expect("winner_lag must be set when winner_index != -1");
            self.reported_lag_estimate = Some(LagEstimate::new(winner_lag, winner_lag));

            if self.detect_pre_echo && self.last_detected_best_lag_filter == winner_index {
                const ENERGY_THRESHOLD: f32 = 1.0;
                if error_sum_anchor > ENERGY_THRESHOLD {
                    update_accumulated_error(
                        &self.instantaneous_accumulated_error,
                        &mut self.accumulated_error[winner_index as usize],
                        1.0 / error_sum_anchor,
                    );
                    self.number_pre_echo_updates += 1;
                }
                if self.number_pre_echo_updates >= 50 {
                    let pre_echo_lag = compute_pre_echo_lag(
                        &self.accumulated_error[winner_index as usize],
                        winner_lag,
                        winner_index as usize * self.filter_intra_lag_shift,
                    );
                    if let Some(ref mut est) = self.reported_lag_estimate {
                        est.pre_echo_lag = pre_echo_lag;
                    }
                } else if let Some(ref mut est) = self.reported_lag_estimate {
                    est.pre_echo_lag = winner_lag;
                }
            }
            self.last_detected_best_lag_filter = winner_index;
        }
    }

    /// Returns the current lag estimate.
    pub(crate) fn get_best_lag_estimate(&self) -> Option<LagEstimate> {
        self.reported_lag_estimate
    }

    /// Returns the maximum filter lag.
    pub(crate) fn get_max_filter_lag(&self) -> usize {
        self.filters.len() * self.filter_intra_lag_shift + self.filters[0].len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG random for deterministic tests (matching C++ Random(42U)).
    struct TestRandom {
        state: u32,
    }

    impl TestRandom {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            // Same constants as webrtc::Random (Musl LCG).
            self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            // Uniform in [-32767, 32767].
            let val = self.next_u32();
            ((val as i32 % 65535) - 32767) as f32
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32();
            }
        }
    }

    /// Verifies that max_square_peak_index finds the correct peak for
    /// various lengths and positions.
    #[test]
    fn max_square_peak_index_correctness() {
        // Single element.
        assert_eq!(max_square_peak_index(&[5.0]), 0);

        // Two elements.
        assert_eq!(max_square_peak_index(&[1.0, 2.0]), 1);
        assert_eq!(max_square_peak_index(&[3.0, 2.0]), 0);
        assert_eq!(max_square_peak_index(&[-3.0, 2.0]), 0);

        // Peak at various positions.
        for length in 3..130 {
            for peak_pos in 0..length {
                let mut h = vec![1.0f32; length];
                h[peak_pos] = 100.0;
                assert_eq!(
                    max_square_peak_index(&h),
                    peak_pos,
                    "length={length}, peak_pos={peak_pos}"
                );
                // Negative peak (squared should still find it).
                h[peak_pos] = -100.0;
                assert_eq!(
                    max_square_peak_index(&h),
                    peak_pos,
                    "length={length}, peak_pos={peak_pos} (negative)"
                );
            }
        }
    }

    /// Verifies max_square_peak_index with random data.
    #[test]
    fn max_square_peak_index_random() {
        let mut rng = TestRandom::new(42);

        for length in 1..128 {
            let mut y = vec![0.0f32; length];
            for _ in 0..64 {
                rng.fill(&mut y);
                // Make all values distinct to avoid tie-breaking issues.
                for (i, v) in y.iter_mut().enumerate() {
                    *v += i as f32 * 0.001;
                }

                let result = max_square_peak_index(&y);

                // Verify result is actually the max.
                let result_sq = y[result] * y[result];
                for (i, &v) in y.iter().enumerate() {
                    assert!(
                        result_sq >= v * v,
                        "length={length}, result_idx={result}, result_sq={result_sq}, \
                         idx={i}, val_sq={}",
                        v * v
                    );
                }
            }
        }
    }

    /// Verifies the matched filter core scalar function works correctly.
    ///
    /// Creates a scenario where y[i] = x[(x_start_index + delay)] for each
    /// sample, and verifies the filter peak converges to the delay tap.
    #[test]
    fn matched_filter_core_converges() {
        let mut rng = TestRandom::new(42);
        let h_size = 64;
        let x_size = 200;
        let sub_block_size = 16;
        let delay = 10usize;

        let mut x = vec![0.0f32; x_size];
        rng.fill(&mut x);

        let mut h = vec![0.0f32; h_size];
        let smoothing = 0.5f32;
        // Low threshold so the filter updates.
        let x2_sum_threshold = h_size as f32 * 1.0;

        let mut x_start_index = 50usize;
        for _ in 0..500 {
            // Create y: for each sample i, y[i] = x at the position the
            // filter core will read at tap `delay`.
            // The core processes y[0..sub_block_size]. For sample i, it reads
            // x starting at x_start_index (for i=0), then x_start_index-1
            // (for i=1), etc. At tap k, it reads x[(x_start_index + k) % x_size].
            // So y[i] should be x[(x_start_index - i + delay) % x_size].
            let mut y = vec![0.0f32; sub_block_size];
            for (i, sample) in y.iter_mut().enumerate() {
                let idx = (x_start_index + x_size - i + delay) % x_size;
                *sample = x[idx];
            }

            let mut filters_updated = false;
            let mut error_sum = 0.0f32;
            let mut accumulated_error = vec![0.0f32; h_size / 4];

            matched_filter_core(
                x_start_index,
                x2_sum_threshold,
                smoothing,
                &x,
                &y,
                &mut h,
                &mut filters_updated,
                &mut error_sum,
                false,
                &mut accumulated_error,
            );

            x_start_index = (x_start_index + sub_block_size) % x_size;
        }

        // The filter peak should be at the delay tap.
        let peak = max_square_peak_index(&h);
        assert_eq!(peak, delay, "Filter peak should match the delay");
    }

    /// Verifies that the accumulated error path produces non-zero results.
    #[test]
    fn matched_filter_core_accumulated_error() {
        let mut rng = TestRandom::new(123);
        let h_size = 32;
        let x_size = 200;
        let sub_block_size = 8;

        let mut x = vec![0.0f32; x_size];
        rng.fill(&mut x);
        let mut y = vec![0.0f32; sub_block_size];
        rng.fill(&mut y);

        let mut h = vec![0.0f32; h_size];
        let mut accumulated_error = vec![0.0f32; h_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE];

        let mut filters_updated = false;
        let mut error_sum = 0.0f32;

        matched_filter_core(
            0,
            0.0, // very low threshold so update happens
            0.5,
            &x,
            &y,
            &mut h,
            &mut filters_updated,
            &mut error_sum,
            true,
            &mut accumulated_error,
        );

        // At least some accumulated error values should be non-zero.
        let has_nonzero = accumulated_error.iter().any(|&v| v > 0.0);
        assert!(has_nonzero, "Accumulated error should have non-zero values");
    }

    /// Verifies that MatchedFilter reset clears filters.
    #[test]
    fn matched_filter_reset() {
        let mut mf = MatchedFilter::new(
            SimdBackend::Scalar,
            16, // sub_block_size
            32, // window_size_sub_blocks
            10, // num_matched_filters
            24, // alignment_shift_sub_blocks
            150.0,
            0.7,  // smoothing_fast
            0.3,  // smoothing_slow
            0.1,  // matching_filter_threshold
            true, // detect_pre_echo
        );

        // Pollute filters.
        mf.filters[0][0] = 42.0;
        mf.filters[5][100] = 99.0;
        mf.winner_lag = Some(42);
        mf.reported_lag_estimate = Some(LagEstimate::new(42, 42));

        mf.reset(true);

        // All filters should be zeroed.
        for f in &mf.filters {
            assert!(f.iter().all(|&v| v == 0.0));
        }
        // Accumulated error should be reset to 1.0.
        for e in &mf.accumulated_error {
            assert!(e.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
        }
        assert!(mf.winner_lag.is_none());
        assert!(mf.reported_lag_estimate.is_none());
    }

    /// Verifies get_max_filter_lag calculation.
    #[test]
    fn matched_filter_max_lag() {
        let mf = MatchedFilter::new(
            SimdBackend::Scalar,
            16, // sub_block_size
            32, // window_size_sub_blocks
            10, // num_matched_filters
            24, // alignment_shift_sub_blocks
            150.0,
            0.7,
            0.3,
            0.1,
            false,
        );
        // filter_intra_lag_shift = 24 * 16 = 384
        // filter_size = 32 * 16 = 512
        // max_lag = 10 * 384 + 512 = 4352
        assert_eq!(mf.get_max_filter_lag(), 4352);
    }

    /// Verifies that SIMD matched_filter_core_dispatch produces the same
    /// results as the scalar path for various filter sizes and start indices.
    #[test]
    fn matched_filter_core_simd_matches_scalar() {
        let backend = webrtc_simd::detect_backend();
        if backend == SimdBackend::Scalar {
            // Nothing to compare — skip.
            return;
        }

        let mut rng = TestRandom::new(42);

        // Test several filter sizes (must be divisible by 16 for AVX2 accumulated error).
        for h_size in [16, 32, 64, 128, 256, 512] {
            let x_size = h_size * 3;
            let sub_block_size = 16;

            let mut x = vec![0.0f32; x_size];
            rng.fill(&mut x);
            let mut y = vec![0.0f32; sub_block_size];
            rng.fill(&mut y);

            // Test with various start indices, including near wraparound.
            for x_start_index in [0, 1, h_size / 2, x_size - 1, x_size - h_size / 2] {
                // --- Without accumulated error ---
                let mut h_scalar = vec![0.0f32; h_size];
                rng.fill(&mut h_scalar);
                let mut h_simd = h_scalar.clone();

                let mut updated_scalar = false;
                let mut updated_simd = false;
                let mut error_sum_scalar = 0.0f32;
                let mut error_sum_simd = 0.0f32;
                let mut acc_err_scalar = vec![0.0f32; h_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE];
                let mut acc_err_simd = acc_err_scalar.clone();
                let mut scratch = vec![0.0f32; h_size];

                matched_filter_core(
                    x_start_index,
                    1.0,
                    0.5,
                    &x,
                    &y,
                    &mut h_scalar,
                    &mut updated_scalar,
                    &mut error_sum_scalar,
                    false,
                    &mut acc_err_scalar,
                );

                matched_filter_core_dispatch(
                    backend,
                    x_start_index,
                    1.0,
                    0.5,
                    &x,
                    &y,
                    &mut h_simd,
                    &mut updated_simd,
                    &mut error_sum_simd,
                    false,
                    &mut acc_err_simd,
                    &mut scratch,
                );

                assert_eq!(
                    updated_scalar, updated_simd,
                    "filters_updated mismatch for h_size={h_size}, x_start={x_start_index}"
                );
                let err_scale = error_sum_scalar.abs().max(1.0);
                assert!(
                    (error_sum_scalar - error_sum_simd).abs() / err_scale < 1e-3,
                    "error_sum mismatch: scalar={error_sum_scalar}, simd={error_sum_simd}, \
                     h_size={h_size}, x_start={x_start_index}"
                );
                for k in 0..h_size {
                    let abs_err = (h_scalar[k] - h_simd[k]).abs();
                    let scale = h_scalar[k].abs().max(1.0);
                    assert!(
                        abs_err / scale < 1e-3,
                        "h mismatch at {k}: scalar={}, simd={}, h_size={h_size}, \
                         x_start={x_start_index}",
                        h_scalar[k],
                        h_simd[k],
                    );
                }

                // --- With accumulated error ---
                let mut h_scalar2 = vec![0.0f32; h_size];
                rng.fill(&mut h_scalar2);
                let mut h_simd2 = h_scalar2.clone();

                let mut updated_scalar2 = false;
                let mut updated_simd2 = false;
                let mut error_sum_scalar2 = 0.0f32;
                let mut error_sum_simd2 = 0.0f32;
                let mut acc_err_scalar2 = vec![0.0f32; h_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE];
                let mut acc_err_simd2 = acc_err_scalar2.clone();
                let mut scratch2 = vec![0.0f32; h_size];

                matched_filter_core(
                    x_start_index,
                    1.0,
                    0.5,
                    &x,
                    &y,
                    &mut h_scalar2,
                    &mut updated_scalar2,
                    &mut error_sum_scalar2,
                    true,
                    &mut acc_err_scalar2,
                );

                matched_filter_core_dispatch(
                    backend,
                    x_start_index,
                    1.0,
                    0.5,
                    &x,
                    &y,
                    &mut h_simd2,
                    &mut updated_simd2,
                    &mut error_sum_simd2,
                    true,
                    &mut acc_err_simd2,
                    &mut scratch2,
                );

                assert_eq!(
                    updated_scalar2, updated_simd2,
                    "filters_updated mismatch (acc_error) for h_size={h_size}, \
                     x_start={x_start_index}"
                );
                let err_scale2 = error_sum_scalar2.abs().max(1.0);
                assert!(
                    (error_sum_scalar2 - error_sum_simd2).abs() / err_scale2 < 1e-3,
                    "error_sum mismatch (acc_error): scalar={error_sum_scalar2}, \
                     simd={error_sum_simd2}, h_size={h_size}, x_start={x_start_index}"
                );
                for k in 0..h_size {
                    let abs_err = (h_scalar2[k] - h_simd2[k]).abs();
                    let scale = h_scalar2[k].abs().max(1.0);
                    assert!(
                        abs_err / scale < 1e-3,
                        "h mismatch (acc_error) at {k}: scalar={}, simd={}, h_size={h_size}, \
                         x_start={x_start_index}",
                        h_scalar2[k],
                        h_simd2[k],
                    );
                }
                for k in 0..acc_err_scalar2.len() {
                    let abs_err = (acc_err_scalar2[k] - acc_err_simd2[k]).abs();
                    let scale = acc_err_scalar2[k].abs().max(1.0);
                    assert!(
                        abs_err / scale < 1e-3,
                        "accumulated_error mismatch at {k}: scalar={}, simd={}, \
                         h_size={h_size}, x_start={x_start_index}",
                        acc_err_scalar2[k],
                        acc_err_simd2[k],
                    );
                }
            }
        }
    }

    /// Verifies that matched_filter_core_dispatch with scalar backend produces
    /// identical results to calling matched_filter_core directly.
    #[test]
    fn matched_filter_core_dispatch_scalar_identical() {
        let mut rng = TestRandom::new(99);
        let h_size = 64;
        let x_size = 200;
        let sub_block_size = 16;

        let mut x = vec![0.0f32; x_size];
        rng.fill(&mut x);
        let mut y = vec![0.0f32; sub_block_size];
        rng.fill(&mut y);

        let mut h_direct = vec![0.0f32; h_size];
        rng.fill(&mut h_direct);
        let mut h_dispatch = h_direct.clone();

        let mut updated_direct = false;
        let mut updated_dispatch = false;
        let mut error_sum_direct = 0.0f32;
        let mut error_sum_dispatch = 0.0f32;
        let mut acc_err_direct = vec![0.0f32; h_size / ACCUMULATED_ERROR_SUB_SAMPLE_RATE];
        let mut acc_err_dispatch = acc_err_direct.clone();
        let mut scratch = vec![0.0f32; h_size];

        matched_filter_core(
            50,
            1.0,
            0.5,
            &x,
            &y,
            &mut h_direct,
            &mut updated_direct,
            &mut error_sum_direct,
            false,
            &mut acc_err_direct,
        );

        matched_filter_core_dispatch(
            SimdBackend::Scalar,
            50,
            1.0,
            0.5,
            &x,
            &y,
            &mut h_dispatch,
            &mut updated_dispatch,
            &mut error_sum_dispatch,
            false,
            &mut acc_err_dispatch,
            &mut scratch,
        );

        assert_eq!(updated_direct, updated_dispatch);
        assert_eq!(error_sum_direct, error_sum_dispatch);
        assert_eq!(h_direct, h_dispatch);
    }
}
