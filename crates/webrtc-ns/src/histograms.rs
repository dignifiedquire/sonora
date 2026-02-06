//! Histogram tracking for signal features.
//!
//! Maintains histograms of LRT, spectral flatness, and spectral difference
//! features. These are used by the prior signal model estimator to determine
//! feature thresholds.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/histograms.cc`

use crate::config::{BIN_SIZE_LRT, BIN_SIZE_SPEC_DIFF, BIN_SIZE_SPEC_FLAT};
use crate::signal_model::SignalModel;

/// Number of bins in each histogram.
pub const HISTOGRAM_SIZE: usize = 1000;

/// Histograms for three signal features.
#[derive(Debug, Clone)]
pub struct Histograms {
    lrt: [i32; HISTOGRAM_SIZE],
    spectral_flatness: [i32; HISTOGRAM_SIZE],
    spectral_diff: [i32; HISTOGRAM_SIZE],
}

impl Histograms {
    /// Create zero-initialized histograms.
    pub fn new() -> Self {
        Self {
            lrt: [0; HISTOGRAM_SIZE],
            spectral_flatness: [0; HISTOGRAM_SIZE],
            spectral_diff: [0; HISTOGRAM_SIZE],
        }
    }

    /// Clear all histograms to zero.
    pub fn clear(&mut self) {
        self.lrt.fill(0);
        self.spectral_flatness.fill(0);
        self.spectral_diff.fill(0);
    }

    /// Update histograms from the current signal features.
    pub fn update(&mut self, features: &SignalModel) {
        // Update LRT histogram.
        let one_by_bin_size_lrt = 1.0 / BIN_SIZE_LRT;
        if features.lrt < HISTOGRAM_SIZE as f32 * BIN_SIZE_LRT && features.lrt >= 0.0 {
            self.lrt[(one_by_bin_size_lrt * features.lrt) as usize] += 1;
        }

        // Update spectral flatness histogram.
        let one_by_bin_size_spec_flat = 1.0 / BIN_SIZE_SPEC_FLAT;
        if features.spectral_flatness < HISTOGRAM_SIZE as f32 * BIN_SIZE_SPEC_FLAT
            && features.spectral_flatness >= 0.0
        {
            self.spectral_flatness
                [(features.spectral_flatness * one_by_bin_size_spec_flat) as usize] += 1;
        }

        // Update spectral difference histogram.
        let one_by_bin_size_spec_diff = 1.0 / BIN_SIZE_SPEC_DIFF;
        if features.spectral_diff < HISTOGRAM_SIZE as f32 * BIN_SIZE_SPEC_DIFF
            && features.spectral_diff >= 0.0
        {
            self.spectral_diff[(features.spectral_diff * one_by_bin_size_spec_diff) as usize] += 1;
        }
    }

    /// Access the LRT histogram.
    pub fn lrt(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.lrt
    }

    /// Access the spectral flatness histogram.
    pub fn spectral_flatness(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.spectral_flatness
    }

    /// Access the spectral difference histogram.
    pub fn spectral_diff(&self) -> &[i32; HISTOGRAM_SIZE] {
        &self.spectral_diff
    }
}

impl Default for Histograms {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zeroed() {
        let h = Histograms::new();
        assert!(h.lrt().iter().all(|&v| v == 0));
        assert!(h.spectral_flatness().iter().all(|&v| v == 0));
        assert!(h.spectral_diff().iter().all(|&v| v == 0));
    }

    #[test]
    fn update_increments_correct_bins() {
        let mut h = Histograms::new();
        let features = SignalModel {
            lrt: 0.5,                // bin = 0.5 / 0.1 = 5
            spectral_flatness: 0.25, // bin = 0.25 / 0.05 = 5
            spectral_diff: 1.0,      // bin = 1.0 / 0.1 = 10
            ..SignalModel::new()
        };
        h.update(&features);

        assert_eq!(h.lrt()[5], 1);
        assert_eq!(h.spectral_flatness()[5], 1);
        assert_eq!(h.spectral_diff()[10], 1);

        // Other bins remain zero.
        assert_eq!(h.lrt()[0], 0);
        assert_eq!(h.lrt()[6], 0);
    }

    #[test]
    fn update_accumulates() {
        let mut h = Histograms::new();
        let features = SignalModel {
            lrt: 0.5,
            spectral_flatness: 0.25,
            spectral_diff: 1.0,
            ..SignalModel::new()
        };
        h.update(&features);
        h.update(&features);
        h.update(&features);

        assert_eq!(h.lrt()[5], 3);
    }

    #[test]
    fn out_of_range_ignored() {
        let mut h = Histograms::new();
        let features = SignalModel {
            lrt: -1.0,
            spectral_flatness: 1000.0,
            spectral_diff: -0.1,
            ..SignalModel::new()
        };
        h.update(&features);

        assert!(h.lrt().iter().all(|&v| v == 0));
        assert!(h.spectral_flatness().iter().all(|&v| v == 0));
        assert!(h.spectral_diff().iter().all(|&v| v == 0));
    }

    #[test]
    fn clear_resets() {
        let mut h = Histograms::new();
        let features = SignalModel::new();
        h.update(&features);
        h.clear();
        assert!(h.lrt().iter().all(|&v| v == 0));
    }
}
