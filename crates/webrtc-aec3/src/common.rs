//! AEC3 constants and utility functions.
//!
//! Ported from `modules/audio_processing/aec3/aec3_common.h/cc`.

pub(crate) const NUM_BLOCKS_PER_SECOND: usize = 250;

pub(crate) const METRICS_REPORTING_INTERVAL_BLOCKS: usize = 10 * NUM_BLOCKS_PER_SECOND;
pub(crate) const METRICS_COMPUTATION_BLOCKS: usize = 3;
pub(crate) const METRICS_COLLECTION_BLOCKS: usize =
    METRICS_REPORTING_INTERVAL_BLOCKS - METRICS_COMPUTATION_BLOCKS;

pub(crate) const FFT_LENGTH_BY_2: usize = 64;
pub(crate) const FFT_LENGTH_BY_2_PLUS_1: usize = FFT_LENGTH_BY_2 + 1;
pub(crate) const FFT_LENGTH_BY_2_MINUS_1: usize = FFT_LENGTH_BY_2 - 1;
pub(crate) const FFT_LENGTH: usize = 2 * FFT_LENGTH_BY_2;
pub(crate) const FFT_LENGTH_BY_2_LOG2: usize = 6;

pub const RENDER_TRANSFER_QUEUE_SIZE_FRAMES: usize = 100;

pub const MAX_NUM_BANDS: usize = 3;
pub const FRAME_SIZE: usize = 160;
pub const SUB_FRAME_LENGTH: usize = FRAME_SIZE / 2;

pub const BLOCK_SIZE: usize = FFT_LENGTH_BY_2;
pub(crate) const BLOCK_SIZE_LOG2: usize = FFT_LENGTH_BY_2_LOG2;
pub(crate) const BLOCK_SIZE_MS: usize = FFT_LENGTH_BY_2 * 1000 / 16000;

pub(crate) const EXTENDED_BLOCK_SIZE: usize = 2 * FFT_LENGTH_BY_2;
pub(crate) const MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS: usize = 32;
pub(crate) const MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS: usize =
    MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS * 3 / 4;

/// Returns the number of frequency bands for the given sample rate.
pub const fn num_bands_for_rate(sample_rate_hz: usize) -> usize {
    sample_rate_hz / 16000
}

/// Returns whether the given sample rate is a valid full-band rate.
pub const fn valid_full_band_rate(sample_rate_hz: usize) -> bool {
    matches!(sample_rate_hz, 16000 | 32000 | 48000)
}

/// Returns the time-domain length corresponding to a filter length in blocks.
pub(crate) const fn get_time_domain_length(filter_length_blocks: usize) -> usize {
    filter_length_blocks * FFT_LENGTH_BY_2
}

/// Returns the required downsampled buffer size for matched filtering.
pub(crate) const fn get_down_sampled_buffer_size(
    down_sampling_factor: usize,
    num_matched_filters: usize,
) -> usize {
    BLOCK_SIZE / down_sampling_factor
        * (MATCHED_FILTER_ALIGNMENT_SHIFT_SIZE_SUB_BLOCKS * num_matched_filters
            + MATCHED_FILTER_WINDOW_SIZE_SUB_BLOCKS
            + 1)
}

/// Returns the render delay buffer size.
pub(crate) const fn get_render_delay_buffer_size(
    down_sampling_factor: usize,
    num_matched_filters: usize,
    filter_length_blocks: usize,
) -> usize {
    get_down_sampled_buffer_size(down_sampling_factor, num_matched_filters)
        / (BLOCK_SIZE / down_sampling_factor)
        + filter_length_blocks
        + 1
}

/// Fast approximate log2 of a float.
///
/// Reinterprets the float bits to extract the exponent for a rough log2.
pub(crate) fn fast_approx_log2f(input: f32) -> f32 {
    debug_assert!(input > 0.0);
    let bits = input.to_bits();
    let out = bits as f32;
    out * 1.192_092_9e-7 - 126.942_695
}

/// Converts a log2-domain power quantity to decibels.
pub(crate) fn log2_to_db(in_log2: f32) -> f32 {
    3.010_299_956_639_812 * in_log2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_log2_consistent() {
        assert_eq!(1 << BLOCK_SIZE_LOG2, BLOCK_SIZE);
    }

    #[test]
    fn fft_length_by2_log2_consistent() {
        assert_eq!(1 << FFT_LENGTH_BY_2_LOG2, FFT_LENGTH_BY_2);
    }

    #[test]
    fn num_bands_for_known_rates() {
        assert_eq!(num_bands_for_rate(16000), 1);
        assert_eq!(num_bands_for_rate(32000), 2);
        assert_eq!(num_bands_for_rate(48000), 3);
    }

    #[test]
    fn valid_rates() {
        assert!(valid_full_band_rate(16000));
        assert!(valid_full_band_rate(32000));
        assert!(valid_full_band_rate(48000));
        assert!(!valid_full_band_rate(8001));
    }

    #[test]
    fn fast_approx_log2f_reasonable() {
        // log2(1.0) = 0.0
        let v = fast_approx_log2f(1.0);
        assert!(v.abs() < 0.1, "log2(1.0) = {v}, expected ~0.0");

        // log2(2.0) = 1.0
        let v = fast_approx_log2f(2.0);
        assert!((v - 1.0).abs() < 0.1, "log2(2.0) = {v}, expected ~1.0");

        // log2(1024.0) = 10.0
        let v = fast_approx_log2f(1024.0);
        assert!((v - 10.0).abs() < 0.1, "log2(1024.0) = {v}, expected ~10.0");
    }

    #[test]
    fn log2_to_db_conversion() {
        // 1 in log2 domain = ~3.01 dB
        let db = log2_to_db(1.0);
        assert!((db - 3.0103).abs() < 0.001);
    }
}
