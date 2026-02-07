//! Estimator of prior signal model parameters from feature histograms.
//!
//! Analyzes histograms of LRT, spectral flatness, and spectral difference
//! features to determine thresholds and feature weights for the prior
//! signal model.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/prior_signal_model_estimator.cc`

use crate::config::{
    BIN_SIZE_LRT, BIN_SIZE_SPEC_DIFF, BIN_SIZE_SPEC_FLAT, FEATURE_UPDATE_WINDOW_SIZE,
};
use crate::histograms::{HISTOGRAM_SIZE, Histograms};
use crate::prior_signal_model::PriorSignalModel;

/// Finds the first of the two largest peaks in a histogram.
///
/// Returns `(peak_position, peak_weight)`. If the two largest peaks are
/// close together and the secondary peak is at least half the primary,
/// they are merged.
fn find_first_of_two_largest_peaks(bin_size: f32, histogram: &[i32; HISTOGRAM_SIZE]) -> (f32, i32) {
    let mut peak_value = 0;
    let mut secondary_peak_value = 0;
    let mut peak_position = 0.0f32;
    let mut secondary_peak_position = 0.0f32;
    let mut peak_weight = 0;
    let mut secondary_peak_weight = 0;

    // Identify the two largest peaks.
    for (i, &hist_val) in histogram.iter().enumerate() {
        let bin_mid = (i as f32 + 0.5) * bin_size;
        if hist_val > peak_value {
            // Found new "first" peak candidate.
            secondary_peak_value = peak_value;
            secondary_peak_weight = peak_weight;
            secondary_peak_position = peak_position;

            peak_value = hist_val;
            peak_weight = hist_val;
            peak_position = bin_mid;
        } else if hist_val > secondary_peak_value {
            // Found new "second" peak candidate.
            secondary_peak_value = hist_val;
            secondary_peak_weight = hist_val;
            secondary_peak_position = bin_mid;
        }
    }

    // Merge the peaks if they are close.
    if (secondary_peak_position - peak_position).abs() < 2.0 * bin_size
        && secondary_peak_weight as f32 > 0.5 * peak_weight as f32
    {
        peak_weight += secondary_peak_weight;
        peak_position = 0.5 * (peak_position + secondary_peak_position);
    }

    (peak_position, peak_weight)
}

/// Updates the LRT threshold from the LRT histogram.
///
/// Returns `(prior_model_lrt, low_lrt_fluctuations)`.
fn update_lrt(lrt_histogram: &[i32; HISTOGRAM_SIZE]) -> (f32, bool) {
    let mut average = 0.0f32;
    let mut average_compl = 0.0f32;
    let mut average_squared = 0.0f32;
    let mut count = 0;

    for (i, &hist_val) in lrt_histogram.iter().enumerate().take(10) {
        let bin_mid = (i as f32 + 0.5) * BIN_SIZE_LRT;
        average += hist_val as f32 * bin_mid;
        count += hist_val;
    }
    if count > 0 {
        average /= count as f32;
    }

    for (i, &hist_val) in lrt_histogram.iter().enumerate() {
        let bin_mid = (i as f32 + 0.5) * BIN_SIZE_LRT;
        average_squared += hist_val as f32 * bin_mid * bin_mid;
        average_compl += hist_val as f32 * bin_mid;
    }
    const ONE_FEATURE_UPDATE_WINDOW_SIZE: f32 = 1.0 / FEATURE_UPDATE_WINDOW_SIZE as f32;
    average_squared *= ONE_FEATURE_UPDATE_WINDOW_SIZE;
    average_compl *= ONE_FEATURE_UPDATE_WINDOW_SIZE;

    // Fluctuation limit of LRT feature.
    let low_lrt_fluctuations = average_squared - average * average_compl < 0.05;

    // Get threshold for LRT feature.
    const MAX_LRT: f32 = 1.0;
    const MIN_LRT: f32 = 0.2;
    let prior_model_lrt = if low_lrt_fluctuations {
        // Very low fluctuation, so likely noise.
        MAX_LRT
    } else {
        (1.2 * average).clamp(MIN_LRT, MAX_LRT)
    };

    (prior_model_lrt, low_lrt_fluctuations)
}

/// Estimator that derives prior signal model parameters from feature histograms.
#[derive(Debug)]
pub struct PriorSignalModelEstimator {
    prior_model: PriorSignalModel,
}

impl PriorSignalModelEstimator {
    pub fn new(lrt_initial_value: f32) -> Self {
        Self {
            prior_model: PriorSignalModel::new(lrt_initial_value),
        }
    }

    /// Update the prior model from the accumulated histograms.
    pub fn update(&mut self, histograms: &Histograms) {
        let (lrt, low_lrt_fluctuations) = update_lrt(histograms.lrt());

        self.prior_model.lrt = lrt;

        // For spectral flatness and spectral difference: compute the main peaks
        // of the histograms.
        let (spectral_flatness_peak_position, spectral_flatness_peak_weight) =
            find_first_of_two_largest_peaks(BIN_SIZE_SPEC_FLAT, histograms.spectral_flatness());

        let (spectral_diff_peak_position, spectral_diff_peak_weight) =
            find_first_of_two_largest_peaks(BIN_SIZE_SPEC_DIFF, histograms.spectral_diff());

        // Reject if weight of peaks is not large enough, or peak value too small.
        // Peak limit for spectral flatness (varies between 0 and 1).
        let use_spec_flat = spectral_flatness_peak_weight as f32 >= 0.3 * 500.0
            && spectral_flatness_peak_position >= 0.6;

        // Reject if weight of peaks is not large enough or if fluctuation of the
        // LRT feature are very low, indicating a noise state.
        let use_spec_diff =
            spectral_diff_peak_weight as f32 >= 0.3 * 500.0 && !low_lrt_fluctuations;

        // Update the model.
        self.prior_model.template_diff_threshold =
            (1.2 * spectral_diff_peak_position).clamp(0.16, 1.0);

        let one_by_feature_sum =
            1.0 / (1.0 + use_spec_flat as i32 as f32 + use_spec_diff as i32 as f32);
        self.prior_model.lrt_weighting = one_by_feature_sum;

        if use_spec_flat {
            self.prior_model.flatness_threshold =
                (0.9 * spectral_flatness_peak_position).clamp(0.1, 0.95);
            self.prior_model.flatness_weighting = one_by_feature_sum;
        } else {
            self.prior_model.flatness_weighting = 0.0;
        }

        if use_spec_diff {
            self.prior_model.difference_weighting = one_by_feature_sum;
        } else {
            self.prior_model.difference_weighting = 0.0;
        }
    }

    /// Returns the estimated prior model.
    pub fn prior_model(&self) -> &PriorSignalModel {
        &self.prior_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_model::SignalModel;

    #[test]
    fn initial_prior_model() {
        let est = PriorSignalModelEstimator::new(0.5);
        let m = est.prior_model();
        assert_eq!(m.lrt, 0.5);
        assert_eq!(m.lrt_weighting, 1.0);
        assert_eq!(m.flatness_weighting, 0.0);
        assert_eq!(m.difference_weighting, 0.0);
    }

    #[test]
    fn update_with_empty_histograms() {
        let mut est = PriorSignalModelEstimator::new(0.5);
        let histograms = Histograms::default();
        est.update(&histograms);
        // With empty histograms, LRT average is 0, so low_lrt_fluctuations=true → lrt=1.0.
        assert_eq!(est.prior_model().lrt, 1.0);
        // With empty histograms, peaks have zero weight → features rejected.
        assert_eq!(est.prior_model().lrt_weighting, 1.0);
        assert_eq!(est.prior_model().flatness_weighting, 0.0);
        assert_eq!(est.prior_model().difference_weighting, 0.0);
    }

    #[test]
    fn find_peaks_single_peak() {
        let mut hist = [0i32; HISTOGRAM_SIZE];
        hist[50] = 100;
        let (pos, weight) = find_first_of_two_largest_peaks(0.1, &hist);
        assert!((pos - 5.05).abs() < 0.01); // (50 + 0.5) * 0.1
        assert_eq!(weight, 100);
    }

    #[test]
    fn find_peaks_two_close_peaks_merge() {
        let mut hist = [0i32; HISTOGRAM_SIZE];
        hist[50] = 100;
        hist[51] = 80; // close and > 0.5 * 100 → merged
        let (pos, weight) = find_first_of_two_largest_peaks(0.1, &hist);
        assert_eq!(weight, 180);
        // Merged position = average of the two bin midpoints.
        let expected_pos = 0.5 * (50.5 * 0.1 + 51.5 * 0.1);
        assert!((pos - expected_pos).abs() < 0.01);
    }

    #[test]
    fn find_peaks_two_distant_peaks_no_merge() {
        let mut hist = [0i32; HISTOGRAM_SIZE];
        hist[10] = 100;
        hist[50] = 80;
        let (pos, weight) = find_first_of_two_largest_peaks(0.1, &hist);
        // Not merged because they're far apart.
        assert_eq!(weight, 100);
        assert!((pos - 1.05).abs() < 0.01); // (10 + 0.5) * 0.1
    }

    #[test]
    fn update_enables_features_with_sufficient_data() {
        let mut est = PriorSignalModelEstimator::new(0.5);
        let mut histograms = Histograms::default();

        // Build histograms with enough weight and appropriate peak positions
        // to enable both spectral flatness and spectral diff features.
        let features = SignalModel {
            lrt: 0.5,
            spectral_flatness: 0.7, // > 0.6 threshold
            spectral_diff: 0.5,
            ..SignalModel::default()
        };
        for _ in 0..200 {
            histograms.update(&features);
        }

        est.update(&histograms);

        // With 200 updates at lrt=0.5, count > 0.3*500=150, so features should be enabled.
        // spectral_flatness peak at ~0.7 > 0.6 → use_spec_flat=true
        let m = est.prior_model();
        assert!(m.flatness_weighting > 0.0, "flatness should be enabled");
        // Weights should sum to 1.
        let sum = m.lrt_weighting + m.flatness_weighting + m.difference_weighting;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "weights should sum to 1, got {sum}"
        );
    }
}
