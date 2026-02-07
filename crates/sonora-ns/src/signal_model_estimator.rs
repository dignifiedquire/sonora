//! Signal model feature extraction and estimation.
//!
//! Extracts three features from the signal spectrum — log-likelihood ratio (LRT),
//! spectral flatness, and spectral difference — and periodically updates the
//! prior signal model from accumulated feature histograms.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/signal_model_estimator.cc`

use crate::config::{FEATURE_UPDATE_WINDOW_SIZE, FFT_SIZE_BY_2_PLUS_1, LTR_FEATURE_THR};
use crate::fast_math::{exp_approximation, log_approximation};
use crate::histograms::Histograms;
use crate::prior_signal_model::PriorSignalModel;
use crate::prior_signal_model_estimator::PriorSignalModelEstimator;
use crate::signal_model::SignalModel;

const ONE_BY_FFT_SIZE_BY_2_PLUS_1: f32 = 1.0 / FFT_SIZE_BY_2_PLUS_1 as f32;

/// Computes the spectral difference between the signal and conservative noise.
///
/// `spectral_diff = var(signal) - cov(signal, noise)^2 / var(noise)`
fn compute_spectral_diff(
    conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectral_sum: f32,
    diff_normalization: f32,
) -> f32 {
    // Compute average quantities.
    let mut noise_average = 0.0f32;
    for &n in conservative_noise_spectrum {
        noise_average += n;
    }
    noise_average *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    let signal_average = signal_spectral_sum * ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    // Compute variance and covariance quantities.
    let mut covariance = 0.0f32;
    let mut noise_variance = 0.0f32;
    let mut signal_variance = 0.0f32;
    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        let signal_diff = signal_spectrum[i] - signal_average;
        let noise_diff = conservative_noise_spectrum[i] - noise_average;
        covariance += signal_diff * noise_diff;
        noise_variance += noise_diff * noise_diff;
        signal_variance += signal_diff * signal_diff;
    }
    covariance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    noise_variance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    signal_variance *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    // Update of average magnitude spectrum.
    let spectral_diff = signal_variance - (covariance * covariance) / (noise_variance + 0.0001);
    // Normalize.
    spectral_diff / (diff_normalization + 0.0001)
}

/// Updates the spectral flatness feature.
fn update_spectral_flatness(
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectral_sum: f32,
    spectral_flatness: &mut f32,
) {
    // Compute log of ratio of the geometric to arithmetic mean
    // (handle the log(0) separately).
    const AVERAGING: f32 = 0.3;

    for &s in &signal_spectrum[1..] {
        if s == 0.0 {
            *spectral_flatness -= AVERAGING * *spectral_flatness;
            return;
        }
    }

    let mut avg_spect_flatness_num = 0.0f32;
    for &s in &signal_spectrum[1..] {
        avg_spect_flatness_num += log_approximation(s);
    }

    let mut avg_spect_flatness_denom = signal_spectral_sum - signal_spectrum[0];
    avg_spect_flatness_denom *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;
    avg_spect_flatness_num *= ONE_BY_FFT_SIZE_BY_2_PLUS_1;

    let spectral_tmp = exp_approximation(avg_spect_flatness_num) / avg_spect_flatness_denom;

    // Time-avg update of spectral flatness feature.
    *spectral_flatness += AVERAGING * (spectral_tmp - *spectral_flatness);
}

/// Updates the log-likelihood ratio (LRT) measures.
fn update_spectral_lrt(
    prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
    post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
    avg_log_lrt: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    lrt: &mut f32,
) {
    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        let tmp1 = 1.0 + 2.0 * prior_snr[i];
        let tmp2 = 2.0 * prior_snr[i] / (tmp1 + 0.0001);
        let bessel_tmp = (post_snr[i] + 1.0) * tmp2;
        avg_log_lrt[i] += 0.5 * (bessel_tmp - log_approximation(tmp1) - avg_log_lrt[i]);
    }

    let mut log_lrt_time_avg_k_sum = 0.0f32;
    for &v in avg_log_lrt.iter() {
        log_lrt_time_avg_k_sum += v;
    }
    *lrt = log_lrt_time_avg_k_sum * ONE_BY_FFT_SIZE_BY_2_PLUS_1;
}

/// Extracts signal features and periodically updates the prior model.
#[derive(Debug)]
pub struct SignalModelEstimator {
    diff_normalization: f32,
    signal_energy_sum: f32,
    histograms: Histograms,
    histogram_analysis_counter: i32,
    prior_model_estimator: PriorSignalModelEstimator,
    features: SignalModel,
}

impl Default for SignalModelEstimator {
    fn default() -> Self {
        Self {
            diff_normalization: 0.0,
            signal_energy_sum: 0.0,
            histograms: Histograms::default(),
            histogram_analysis_counter: FEATURE_UPDATE_WINDOW_SIZE,
            prior_model_estimator: PriorSignalModelEstimator::new(LTR_FEATURE_THR),
            features: SignalModel::default(),
        }
    }
}

impl SignalModelEstimator {
    /// Compute signal normalization during the initial startup phase.
    pub fn adjust_normalization(&mut self, num_analyzed_frames: i32, signal_energy: f32) {
        self.diff_normalization *= num_analyzed_frames as f32;
        self.diff_normalization += signal_energy;
        self.diff_normalization /= num_analyzed_frames as f32 + 1.0;
    }

    /// Update the signal model features.
    pub fn update(
        &mut self,
        prior_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        post_snr: &[f32; FFT_SIZE_BY_2_PLUS_1],
        conservative_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        signal_spectral_sum: f32,
        signal_energy: f32,
    ) {
        // Compute spectral flatness on input spectrum.
        update_spectral_flatness(
            signal_spectrum,
            signal_spectral_sum,
            &mut self.features.spectral_flatness,
        );

        // Compute difference of input spectrum with learned/estimated noise spectrum.
        let spectral_diff = compute_spectral_diff(
            conservative_noise_spectrum,
            signal_spectrum,
            signal_spectral_sum,
            self.diff_normalization,
        );
        // Compute time-avg update of difference feature.
        self.features.spectral_diff += 0.3 * (spectral_diff - self.features.spectral_diff);

        self.signal_energy_sum += signal_energy;

        // Compute histograms for parameter decisions (thresholds and weights for
        // features). Parameters are extracted periodically.
        self.histogram_analysis_counter -= 1;
        if self.histogram_analysis_counter > 0 {
            self.histograms.update(&self.features);
        } else {
            // Compute model parameters.
            self.prior_model_estimator.update(&self.histograms);

            // Clear histograms for next update.
            self.histograms.clear();

            self.histogram_analysis_counter = FEATURE_UPDATE_WINDOW_SIZE;

            // Update every window:
            // Compute normalization for the spectral difference for next estimation.
            self.signal_energy_sum /= FEATURE_UPDATE_WINDOW_SIZE as f32;
            self.diff_normalization = 0.5 * (self.signal_energy_sum + self.diff_normalization);
            self.signal_energy_sum = 0.0;
        }

        // Compute the LRT.
        update_spectral_lrt(
            prior_snr,
            post_snr,
            &mut self.features.avg_log_lrt,
            &mut self.features.lrt,
        );
    }

    /// Returns the current prior signal model.
    pub fn prior_model(&self) -> &PriorSignalModel {
        self.prior_model_estimator.prior_model()
    }

    /// Returns the current signal model features.
    pub fn model(&self) -> &SignalModel {
        &self.features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let est = SignalModelEstimator::default();
        assert_eq!(est.diff_normalization, 0.0);
        assert_eq!(est.signal_energy_sum, 0.0);
        assert_eq!(est.histogram_analysis_counter, FEATURE_UPDATE_WINDOW_SIZE);
    }

    #[test]
    fn adjust_normalization_accumulates() {
        let mut est = SignalModelEstimator::default();
        est.adjust_normalization(0, 100.0);
        assert_eq!(est.diff_normalization, 100.0); // (0*0 + 100) / 1 = 100
        est.adjust_normalization(1, 200.0);
        assert_eq!(est.diff_normalization, 150.0); // (1*100 + 200) / 2 = 150
    }

    #[test]
    fn update_decrements_histogram_counter() {
        let mut est = SignalModelEstimator::default();
        let prior_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let cons_noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        est.update(&prior_snr, &post_snr, &cons_noise, &signal, sum, sum);
        assert_eq!(
            est.histogram_analysis_counter,
            FEATURE_UPDATE_WINDOW_SIZE - 1
        );
    }

    #[test]
    fn histogram_resets_after_window() {
        let mut est = SignalModelEstimator::default();
        let prior_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let cons_noise = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();

        // Run through a full window.
        for _ in 0..FEATURE_UPDATE_WINDOW_SIZE {
            est.update(&prior_snr, &post_snr, &cons_noise, &signal, sum, sum);
        }
        // Counter should be reset.
        assert_eq!(est.histogram_analysis_counter, FEATURE_UPDATE_WINDOW_SIZE);
    }

    #[test]
    fn compute_spectral_diff_identical_signals() {
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let noise = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let sum: f32 = signal.iter().sum();
        // When signal == noise (constant), variance and covariance are both 0.
        let diff = compute_spectral_diff(&noise, &signal, sum, 1.0);
        assert!(diff.abs() < 0.01, "diff should be ~0, got {diff}");
    }

    #[test]
    fn update_spectral_lrt_basic() {
        let prior_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let post_snr = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let mut avg_log_lrt = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        let mut lrt = 0.0f32;

        update_spectral_lrt(&prior_snr, &post_snr, &mut avg_log_lrt, &mut lrt);

        // With prior_snr=1: tmp1=3, tmp2=2/3.0001, bessel=(1+1)*tmp2≈1.333
        // avg_log_lrt updated from 0 toward 0.5*(bessel - log(3) - 0)
        assert!(lrt != 0.0, "lrt should be updated from 0");
    }
}
