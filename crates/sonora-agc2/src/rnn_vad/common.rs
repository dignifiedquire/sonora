//! RNN VAD common constants.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/common.h`.

use std::f64::consts::PI;

pub const K_PI: f64 = PI;

pub const SAMPLE_RATE_24K_HZ: i32 = 24000;
pub const FRAME_SIZE_10MS_24K_HZ: usize = SAMPLE_RATE_24K_HZ as usize / 100;
pub const FRAME_SIZE_20MS_24K_HZ: usize = FRAME_SIZE_10MS_24K_HZ * 2;

// Pitch buffer.
pub const MIN_PITCH_24K_HZ: usize = SAMPLE_RATE_24K_HZ as usize / 800; // 0.00125 s.
pub const MAX_PITCH_24K_HZ: usize = (SAMPLE_RATE_24K_HZ as f64 / 62.5) as usize; // 0.016 s.
pub const BUF_SIZE_24K_HZ: usize = MAX_PITCH_24K_HZ + FRAME_SIZE_20MS_24K_HZ;
const _: () = assert!(BUF_SIZE_24K_HZ & 1 == 0, "The buffer size must be even.");

// 24 kHz analysis.
pub const INITIAL_MIN_PITCH_24K_HZ: usize = 3 * MIN_PITCH_24K_HZ;
const _: () = assert!(MIN_PITCH_24K_HZ < INITIAL_MIN_PITCH_24K_HZ);
const _: () = assert!(INITIAL_MIN_PITCH_24K_HZ < MAX_PITCH_24K_HZ);
const _: () = assert!(MAX_PITCH_24K_HZ > INITIAL_MIN_PITCH_24K_HZ);
/// Number of (inverted) lags during the initial pitch search phase at 24 kHz.
pub const INITIAL_NUM_LAGS_24K_HZ: usize = MAX_PITCH_24K_HZ - INITIAL_MIN_PITCH_24K_HZ;
/// Number of (inverted) lags during the pitch search refinement phase at 24 kHz.
pub const REFINE_NUM_LAGS_24K_HZ: usize = MAX_PITCH_24K_HZ + 1;
const _: () = assert!(
    REFINE_NUM_LAGS_24K_HZ > INITIAL_NUM_LAGS_24K_HZ,
    "The refinement step must search the pitch in an extended pitch range."
);

// 12 kHz analysis.
pub const SAMPLE_RATE_12K_HZ: i32 = 12000;
pub const FRAME_SIZE_10MS_12K_HZ: usize = SAMPLE_RATE_12K_HZ as usize / 100;
pub const FRAME_SIZE_20MS_12K_HZ: usize = FRAME_SIZE_10MS_12K_HZ * 2;
pub const BUF_SIZE_12K_HZ: usize = BUF_SIZE_24K_HZ / 2;
pub const INITIAL_MIN_PITCH_12K_HZ: usize = INITIAL_MIN_PITCH_24K_HZ / 2;
pub const MAX_PITCH_12K_HZ: usize = MAX_PITCH_24K_HZ / 2;
const _: () = assert!(MAX_PITCH_12K_HZ > INITIAL_MIN_PITCH_12K_HZ);
/// The inverted lags for the pitch interval are in range [0, `NUM_LAGS_12K_HZ`].
pub const NUM_LAGS_12K_HZ: usize = MAX_PITCH_12K_HZ - INITIAL_MIN_PITCH_12K_HZ;

// 48 kHz constants.
pub const MIN_PITCH_48K_HZ: usize = MIN_PITCH_24K_HZ * 2;
pub const MAX_PITCH_48K_HZ: usize = MAX_PITCH_24K_HZ * 2;

// Spectral features.
pub const NUM_BANDS: usize = 22;
pub const NUM_LOWER_BANDS: usize = 6;
const _: () = assert!(0 < NUM_LOWER_BANDS && NUM_LOWER_BANDS < NUM_BANDS);
pub const CEPSTRAL_COEFFS_HISTORY_SIZE: usize = 8;
const _: () = assert!(
    CEPSTRAL_COEFFS_HISTORY_SIZE > 2,
    "The history size must at least be 3 to compute first and second derivatives."
);

pub const FEATURE_VECTOR_SIZE: usize = 42;

/// Number of higher-band cepstral coefficients in the feature vector.
pub const NUM_HIGHER_BANDS: usize = NUM_BANDS - NUM_LOWER_BANDS;

/// 42-element feature vector fed to the VAD RNN.
///
/// Layout matches `[f32; 42]` via `#[repr(C)]`. Named fields replace the
/// manual index arithmetic that the C++ code uses with `ArrayView<float, 42>`.
///
/// Use `bytemuck::cast_ref::<_, [f32; FEATURE_VECTOR_SIZE]>()` for zero-copy
/// conversion to a flat slice when needed (e.g. feeding the RNN input layer).
#[derive(Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Copy)]
#[repr(C)]
pub struct FeatureVector {
    /// Average lower-band cepstral coefficients.
    pub average: [f32; NUM_LOWER_BANDS],
    /// Higher-band cepstral coefficients.
    pub higher_bands_cepstrum: [f32; NUM_HIGHER_BANDS],
    /// First derivative of lower-band cepstral coefficients.
    pub first_derivative: [f32; NUM_LOWER_BANDS],
    /// Second derivative of lower-band cepstral coefficients.
    pub second_derivative: [f32; NUM_LOWER_BANDS],
    /// Cross-correlation between reference and lagged frames per band.
    pub bands_cross_correlation: [f32; NUM_LOWER_BANDS],
    /// Normalized pitch period.
    pub pitch_period: f32,
    /// Spectral variability.
    pub spectral_variability: f32,
}

impl Default for FeatureVector {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl FeatureVector {
    /// Views the feature vector as a flat slice of floats (zero-copy).
    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(bytemuck::bytes_of(self))
    }
}
