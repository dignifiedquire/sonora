//! Signal model for feature tracking.
//!
//! Holds the signal features (LRT, spectral flatness, spectral difference)
//! that are used by the speech probability estimator.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/signal_model.h`

use crate::config::{FFT_SIZE_BY_2_PLUS_1, LTR_FEATURE_THR};

/// Signal model containing extracted features.
#[derive(Debug, Clone)]
pub struct SignalModel {
    /// Log-likelihood ratio test statistic.
    pub lrt: f32,
    /// Spectral difference measure.
    pub spectral_diff: f32,
    /// Spectral flatness measure.
    pub spectral_flatness: f32,
    /// Time-smoothed log LRT per frequency bin.
    pub avg_log_lrt: [f32; FFT_SIZE_BY_2_PLUS_1],
}

impl SignalModel {
    pub fn new() -> Self {
        const SF_FEATURE_THR: f32 = 0.5;
        Self {
            lrt: LTR_FEATURE_THR,
            spectral_flatness: SF_FEATURE_THR,
            spectral_diff: SF_FEATURE_THR,
            avg_log_lrt: [LTR_FEATURE_THR; FFT_SIZE_BY_2_PLUS_1],
        }
    }
}

impl Default for SignalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let m = SignalModel::new();
        assert_eq!(m.lrt, 0.5);
        assert_eq!(m.spectral_flatness, 0.5);
        assert_eq!(m.spectral_diff, 0.5);
        assert!(m.avg_log_lrt.iter().all(|&v| v == 0.5));
        assert_eq!(m.avg_log_lrt.len(), 129);
    }
}
