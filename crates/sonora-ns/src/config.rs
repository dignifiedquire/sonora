//! Noise suppression configuration and common constants.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/ns_config.h`
//!             `webrtc/modules/audio_processing/ns/ns_common.h`

/// FFT size used by the noise suppressor (256-point).
pub const FFT_SIZE: usize = 256;

/// Number of unique frequency bins (FFT_SIZE / 2 + 1).
pub const FFT_SIZE_BY_2_PLUS_1: usize = FFT_SIZE / 2 + 1;

/// Audio frame size in samples (160 = 10ms at 16kHz).
pub const NS_FRAME_SIZE: usize = 160;

/// Overlap between consecutive FFT frames.
pub const OVERLAP_SIZE: usize = FFT_SIZE - NS_FRAME_SIZE;

/// Number of blocks in the short startup phase.
pub const SHORT_STARTUP_PHASE_BLOCKS: i32 = 50;

/// Number of blocks in the long startup phase.
pub const LONG_STARTUP_PHASE_BLOCKS: i32 = 200;

/// Feature update window size in frames.
pub const FEATURE_UPDATE_WINDOW_SIZE: i32 = 500;

/// Threshold for the LRT feature.
pub const LTR_FEATURE_THR: f32 = 0.5;

/// Bin size for LRT histogram.
pub const BIN_SIZE_LRT: f32 = 0.1;

/// Bin size for spectral flatness histogram.
pub const BIN_SIZE_SPEC_FLAT: f32 = 0.05;

/// Bin size for spectral difference histogram.
pub const BIN_SIZE_SPEC_DIFF: f32 = 0.1;

/// Target suppression level for the noise suppressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SuppressionLevel {
    /// 6 dB suppression.
    K6dB,
    /// 12 dB suppression (default).
    #[default]
    K12dB,
    /// 18 dB suppression.
    K18dB,
    /// 21 dB suppression.
    K21dB,
}

/// Configuration for the noise suppressor.
#[derive(Debug, Clone, Copy)]
pub struct NsConfig {
    /// Target suppression level.
    pub target_level: SuppressionLevel,
}

impl Default for NsConfig {
    fn default() -> Self {
        Self {
            target_level: SuppressionLevel::K12dB,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_suppression_level() {
        let config = NsConfig::default();
        assert_eq!(config.target_level, SuppressionLevel::K12dB);
    }

    #[test]
    fn constants_match_cpp() {
        assert_eq!(FFT_SIZE, 256);
        assert_eq!(FFT_SIZE_BY_2_PLUS_1, 129);
        assert_eq!(NS_FRAME_SIZE, 160);
        assert_eq!(OVERLAP_SIZE, 96);
        assert_eq!(SHORT_STARTUP_PHASE_BLOCKS, 50);
        assert_eq!(LONG_STARTUP_PHASE_BLOCKS, 200);
        assert_eq!(FEATURE_UPDATE_WINDOW_SIZE, 500);
    }
}
