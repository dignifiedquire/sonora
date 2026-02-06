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

impl SuppressionParams {
    /// Create suppression parameters for the given level.
    pub fn new(level: SuppressionLevel) -> Self {
        match level {
            SuppressionLevel::K6dB => Self {
                over_subtraction_factor: 1.0,
                minimum_attenuating_gain: 0.5, // 6 dB attenuation
                use_attenuation_adjustment: false,
            },
            SuppressionLevel::K12dB => Self {
                over_subtraction_factor: 1.0,
                minimum_attenuating_gain: 0.25, // 12 dB attenuation
                use_attenuation_adjustment: true,
            },
            SuppressionLevel::K18dB => Self {
                over_subtraction_factor: 1.1,
                minimum_attenuating_gain: 0.125, // 18 dB attenuation
                use_attenuation_adjustment: true,
            },
            SuppressionLevel::K21dB => Self {
                over_subtraction_factor: 1.25,
                minimum_attenuating_gain: 0.09, // 20.9 dB attenuation
                use_attenuation_adjustment: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k6db_params() {
        let p = SuppressionParams::new(SuppressionLevel::K6dB);
        assert_eq!(p.over_subtraction_factor, 1.0);
        assert_eq!(p.minimum_attenuating_gain, 0.5);
        assert!(!p.use_attenuation_adjustment);
    }

    #[test]
    fn k12db_params() {
        let p = SuppressionParams::new(SuppressionLevel::K12dB);
        assert_eq!(p.over_subtraction_factor, 1.0);
        assert_eq!(p.minimum_attenuating_gain, 0.25);
        assert!(p.use_attenuation_adjustment);
    }

    #[test]
    fn k18db_params() {
        let p = SuppressionParams::new(SuppressionLevel::K18dB);
        assert_eq!(p.over_subtraction_factor, 1.1);
        assert_eq!(p.minimum_attenuating_gain, 0.125);
        assert!(p.use_attenuation_adjustment);
    }

    #[test]
    fn k21db_params() {
        let p = SuppressionParams::new(SuppressionLevel::K21dB);
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
            .map(|l| SuppressionParams::new(*l).minimum_attenuating_gain)
            .collect();
        for w in gains.windows(2) {
            assert!(w[0] > w[1], "gain should decrease: {} > {}", w[0], w[1]);
        }
    }
}
