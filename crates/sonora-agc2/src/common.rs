//! AGC2 common constants and audio utility functions.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/agc2_common.h`
//! and `webrtc/common_audio/include/audio_util.h`.

pub const MIN_FLOAT_S16_VALUE: f32 = -32768.0;
pub const MAX_FLOAT_S16_VALUE: f32 = 32767.0;
pub const MAX_ABS_FLOAT_S16_VALUE: f32 = 32768.0;

/// Minimum audio level in dBFS scale for S16 samples.
pub const MIN_LEVEL_DBFS: f32 = -90.31;

pub const FRAME_DURATION_MS: i32 = 10;
pub const SUB_FRAMES_IN_FRAME: i32 = 20;
pub const MAXIMAL_NUMBER_OF_SAMPLES_PER_CHANNEL: usize = 480;

// Adaptive digital gain applier settings.

/// At what limiter levels should we start decreasing the adaptive digital gain.
pub const LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS: f32 = -1.0;

/// Number of milliseconds to wait to periodically reset the VAD.
pub const VAD_RESET_PERIOD_MS: i32 = 1500;

/// Speech probability threshold to detect speech activity.
pub const VAD_CONFIDENCE_THRESHOLD: f32 = 0.95;

/// Minimum number of adjacent speech frames having a sufficiently high speech
/// probability to reliably detect speech activity.
pub const ADJACENT_SPEECH_FRAMES_THRESHOLD: i32 = 12;

/// Number of milliseconds of speech frames to observe to make the estimator
/// confident.
pub const LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS: f32 = 400.0;
pub const LEVEL_ESTIMATOR_LEAK_FACTOR: f32 = 1.0 - 1.0 / LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS;

// Saturation Protector settings.
pub const SATURATION_PROTECTOR_INITIAL_HEADROOM_DB: f32 = 20.0;
pub const SATURATION_PROTECTOR_BUFFER_SIZE: usize = 4;

// Number of interpolation points for each region of the limiter.
// These values have been tuned to limit the interpolated gain curve error given
// the limiter parameters and allowing a maximum error of +/- 32768^-1.
pub const INTERPOLATED_GAIN_CURVE_KNEE_POINTS: usize = 22;
pub const INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS: usize = 10;
pub const INTERPOLATED_GAIN_CURVE_TOTAL_POINTS: usize =
    INTERPOLATED_GAIN_CURVE_KNEE_POINTS + INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS;

// Limiter parameters (from agc2_testing_common.h).
pub const LIMITER_MAX_INPUT_LEVEL_DB_FS: f64 = 1.0;
pub const LIMITER_KNEE_SMOOTHNESS_DB: f64 = 1.0;
pub const LIMITER_COMPRESSION_RATIO: f64 = 5.0;

// Audio utility functions ported from common_audio/include/audio_util.h.

/// Converts a dB value to a linear ratio: `10^(v/20)`.
pub fn db_to_ratio(v: f32) -> f32 {
    10.0_f32.powf(v / 20.0)
}

/// Converts a dBFS value to a float S16 linear value.
pub fn dbfs_to_float_s16(v: f32) -> f32 {
    db_to_ratio(v) * MAX_ABS_FLOAT_S16_VALUE
}

/// Converts a float S16 linear value to dBFS.
pub fn float_s16_to_dbfs(v: f32) -> f32 {
    debug_assert!(v >= 0.0);
    // kMinDbfs is equal to -20.0 * log10(-limits_int16::min())
    const MIN_DBFS: f32 = -90.309;
    if v <= 1.0 {
        return MIN_DBFS;
    }
    // Equal to 20 * log10(v / (-limits_int16::min()))
    20.0 * v.log10() + MIN_DBFS
}

/// Converts a dBFS value to a float S16 linear value (f64 version).
pub fn dbfs_to_float_s16_f64(v: f64) -> f64 {
    10.0_f64.powf(v / 20.0) * MAX_ABS_FLOAT_S16_VALUE as f64
}

/// Converts a float S16 linear value to dBFS (f64 version).
pub fn float_s16_to_dbfs_f64(v: f64) -> f64 {
    debug_assert!(v >= 0.0);
    const MIN_DBFS: f64 = -90.308_998_699_194_36;
    if v <= 1.0 {
        return MIN_DBFS;
    }
    20.0 * v.log10() + MIN_DBFS
}
