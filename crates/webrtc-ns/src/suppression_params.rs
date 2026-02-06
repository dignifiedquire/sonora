//! Level-dependent suppression parameters.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/suppression_params.cc`

use crate::config::SuppressionLevel;

/// Parameters that control the noise suppression behavior at a given level.
#[derive(Debug, Clone, Copy)]
pub struct SuppressionParams {
    /// Over-subtraction factor for noise estimate (higher = more aggressive).
    pub over_subtraction_factor: f32,
    /// Minimum gain applied to attenuated bins (sets the floor).
    pub minimum_attenuating_gain: f32,
    /// Whether to use adaptive attenuation adjustment.
    pub use_attenuation_adjustment: bool,
}

/// 6 dB suppression.
const K6DB: SuppressionParams = SuppressionParams {
    over_subtraction_factor: 1.0,
    minimum_attenuating_gain: 0.5,
    use_attenuation_adjustment: false,
};

/// 12 dB suppression.
const K12DB: SuppressionParams = SuppressionParams {
    over_subtraction_factor: 1.0,
    minimum_attenuating_gain: 0.25,
    use_attenuation_adjustment: true,
};

/// 18 dB suppression.
const K18DB: SuppressionParams = SuppressionParams {
    over_subtraction_factor: 1.1,
    minimum_attenuating_gain: 0.125,
    use_attenuation_adjustment: true,
};

/// 21 dB (20.9 dB) suppression.
const K21DB: SuppressionParams = SuppressionParams {
    over_subtraction_factor: 1.25,
    minimum_attenuating_gain: 0.09,
    use_attenuation_adjustment: true,
};

impl SuppressionParams {
    /// Get the suppression parameters for the given level.
    pub const fn for_level(level: SuppressionLevel) -> &'static Self {
        match level {
            SuppressionLevel::K6dB => &K6DB,
            SuppressionLevel::K12dB => &K12DB,
            SuppressionLevel::K18dB => &K18DB,
            SuppressionLevel::K21dB => &K21DB,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k6db_params() {
        let p = SuppressionParams::for_level(SuppressionLevel::K6dB);
        assert_eq!(p.over_subtraction_factor, 1.0);
        assert_eq!(p.minimum_attenuating_gain, 0.5);
        assert!(!p.use_attenuation_adjustment);
    }

    #[test]
    fn k12db_params() {
        let p = SuppressionParams::for_level(SuppressionLevel::K12dB);
        assert_eq!(p.over_subtraction_factor, 1.0);
        assert_eq!(p.minimum_attenuating_gain, 0.25);
        assert!(p.use_attenuation_adjustment);
    }

    #[test]
    fn k18db_params() {
        let p = SuppressionParams::for_level(SuppressionLevel::K18dB);
        assert_eq!(p.over_subtraction_factor, 1.1);
        assert_eq!(p.minimum_attenuating_gain, 0.125);
        assert!(p.use_attenuation_adjustment);
    }

    #[test]
    fn k21db_params() {
        let p = SuppressionParams::for_level(SuppressionLevel::K21dB);
        assert_eq!(p.over_subtraction_factor, 1.25);
        assert_eq!(p.minimum_attenuating_gain, 0.09);
        assert!(p.use_attenuation_adjustment);
    }

    #[test]
    fn gain_decreases_with_level() {
        let levels = [
            SuppressionLevel::K6dB,
            SuppressionLevel::K12dB,
            SuppressionLevel::K18dB,
            SuppressionLevel::K21dB,
        ];
        let gains: Vec<f32> = levels
            .iter()
            .map(|l| SuppressionParams::for_level(*l).minimum_attenuating_gain)
            .collect();
        for w in gains.windows(2) {
            assert!(w[0] > w[1], "gain should decrease: {} > {}", w[0], w[1]);
        }
    }
}
