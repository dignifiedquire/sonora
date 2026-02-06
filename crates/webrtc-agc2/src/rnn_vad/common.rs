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
