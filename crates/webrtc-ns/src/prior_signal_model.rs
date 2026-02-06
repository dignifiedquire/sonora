//! Prior signal model parameters.
//!
//! Stores the prior model parameters that combine LRT, spectral flatness,
//! and spectral difference features for speech probability estimation.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/prior_signal_model.h`

/// Prior signal model with weighted feature combination.
#[derive(Debug, Clone, Copy)]
pub struct PriorSignalModel {
    /// Log-likelihood ratio threshold.
    pub lrt: f32,
    /// Spectral flatness threshold.
    pub flatness_threshold: f32,
    /// Template difference threshold.
    pub template_diff_threshold: f32,
    /// Weight for LRT feature.
    pub lrt_weighting: f32,
    /// Weight for spectral flatness feature.
    pub flatness_weighting: f32,
    /// Weight for spectral difference feature.
    pub difference_weighting: f32,
}

impl PriorSignalModel {
    /// Create with the given initial LRT value.
    pub fn new(lrt_initial_value: f32) -> Self {
        Self {
            lrt: lrt_initial_value,
            flatness_threshold: 0.5,
            template_diff_threshold: 0.5,
            lrt_weighting: 1.0,
            flatness_weighting: 0.0,
            difference_weighting: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let m = PriorSignalModel::new(1.0);
        assert_eq!(m.lrt, 1.0);
        assert_eq!(m.flatness_threshold, 0.5);
        assert_eq!(m.template_diff_threshold, 0.5);
        assert_eq!(m.lrt_weighting, 1.0);
        assert_eq!(m.flatness_weighting, 0.0);
        assert_eq!(m.difference_weighting, 0.0);
    }

    #[test]
    fn weightings_sum_to_one_initially() {
        let m = PriorSignalModel::new(0.0);
        let sum = m.lrt_weighting + m.flatness_weighting + m.difference_weighting;
        assert_eq!(sum, 1.0);
    }
}
