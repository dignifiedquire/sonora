//! Subtractor output data structure.
//!
//! Stores the values returned from the echo subtractor for a single capture
//! channel, including time-domain signals, frequency-domain data, and
//! computed power metrics.
//!
//! Ported from `modules/audio_processing/aec3/subtractor_output.h/cc`.

use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1};
use crate::fft_data::FftData;

/// Output from the echo subtractor for a single capture channel.
pub(crate) struct SubtractorOutput {
    pub s_refined: [f32; BLOCK_SIZE],
    pub s_coarse: [f32; BLOCK_SIZE],
    pub e_refined: [f32; BLOCK_SIZE],
    pub e_coarse: [f32; BLOCK_SIZE],
    pub e_refined_fft: FftData,
    pub e2_refined: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub e2_coarse: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub s2_refined: f32,
    pub s2_coarse: f32,
    pub e2_refined_sum: f32,
    pub e2_coarse_sum: f32,
    pub y2: f32,
    pub s_refined_max_abs: f32,
    pub s_coarse_max_abs: f32,
}

impl Default for SubtractorOutput {
    fn default() -> Self {
        Self {
            s_refined: [0.0; BLOCK_SIZE],
            s_coarse: [0.0; BLOCK_SIZE],
            e_refined: [0.0; BLOCK_SIZE],
            e_coarse: [0.0; BLOCK_SIZE],
            e_refined_fft: FftData::default(),
            e2_refined: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            e2_coarse: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            s2_refined: 0.0,
            s2_coarse: 0.0,
            e2_refined_sum: 0.0,
            e2_coarse_sum: 0.0,
            y2: 0.0,
            s_refined_max_abs: 0.0,
            s_coarse_max_abs: 0.0,
        }
    }
}

impl SubtractorOutput {
    /// Resets all fields to zero.
    pub(crate) fn reset(&mut self) {
        self.s_refined.fill(0.0);
        self.s_coarse.fill(0.0);
        self.e_refined.fill(0.0);
        self.e_coarse.fill(0.0);
        self.e_refined_fft.clear();
        self.e2_refined.fill(0.0);
        self.e2_coarse.fill(0.0);
        self.e2_refined_sum = 0.0;
        self.e2_coarse_sum = 0.0;
        self.s2_refined = 0.0;
        self.s2_coarse = 0.0;
        self.y2 = 0.0;
    }

    /// Updates the power metrics from the signal data.
    pub(crate) fn compute_metrics(&mut self, y: &[f32]) {
        self.y2 = y.iter().map(|&v| v * v).sum();
        self.e2_refined_sum = self.e_refined.iter().map(|&v| v * v).sum();
        self.e2_coarse_sum = self.e_coarse.iter().map(|&v| v * v).sum();
        self.s2_refined = self.s_refined.iter().map(|&v| v * v).sum();
        self.s2_coarse = self.s_coarse.iter().map(|&v| v * v).sum();

        self.s_refined_max_abs = self
            .s_refined
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));

        self.s_coarse_max_abs = self
            .s_coarse
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_everything() {
        let mut out = SubtractorOutput::default();
        out.s_refined.fill(1.0);
        out.e_refined.fill(2.0);
        out.y2 = 42.0;
        out.reset();
        assert_eq!(out.y2, 0.0);
        assert!(out.s_refined.iter().all(|&v| v == 0.0));
        assert!(out.e_refined.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn compute_metrics_sums_of_squares() {
        let mut out = SubtractorOutput::default();
        out.s_refined.fill(2.0);
        out.s_coarse.fill(3.0);
        out.e_refined.fill(1.0);
        out.e_coarse.fill(0.5);

        let y = [4.0f32; BLOCK_SIZE];
        out.compute_metrics(&y);

        assert!((out.y2 - 16.0 * BLOCK_SIZE as f32).abs() < 1e-4);
        assert!((out.s2_refined - 4.0 * BLOCK_SIZE as f32).abs() < 1e-4);
        assert!((out.s2_coarse - 9.0 * BLOCK_SIZE as f32).abs() < 1e-4);
        assert!((out.e2_refined_sum - 1.0 * BLOCK_SIZE as f32).abs() < 1e-4);
        assert!((out.e2_coarse_sum - 0.25 * BLOCK_SIZE as f32).abs() < 1e-4);
        assert!((out.s_refined_max_abs - 2.0).abs() < 1e-6);
        assert!((out.s_coarse_max_abs - 3.0).abs() < 1e-6);
    }

    #[test]
    fn compute_metrics_max_abs_handles_negatives() {
        let mut out = SubtractorOutput::default();
        out.s_refined[0] = -5.0;
        out.s_refined[1] = 3.0;
        out.s_coarse[0] = 2.0;
        out.s_coarse[1] = -7.0;

        let y = [0.0f32; BLOCK_SIZE];
        out.compute_metrics(&y);

        assert!((out.s_refined_max_abs - 5.0).abs() < 1e-6);
        assert!((out.s_coarse_max_abs - 7.0).abs() < 1e-6);
    }
}
