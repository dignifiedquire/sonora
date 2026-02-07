//! Limiter dB gain curve.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/limiter_db_gain_curve.h/.cc`.

// Used by interpolated_gain_curve and limiter.
#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{
    LIMITER_COMPRESSION_RATIO, LIMITER_KNEE_SMOOTHNESS_DB, LIMITER_MAX_INPUT_LEVEL_DB_FS,
    MAX_ABS_FLOAT_S16_VALUE, dbfs_to_float_s16_f64, float_s16_to_dbfs_f64,
};

/// A limiter gain curve (in dB scale) with four regions:
/// identity (linear), knee (quadratic polynomial), compression (linear),
/// saturation (linear).
pub(crate) struct LimiterDbGainCurve {
    max_input_level_linear: f64,
    knee_start_dbfs: f64,
    knee_start_linear: f64,
    limiter_start_dbfs: f64,
    limiter_start_linear: f64,
    /// Coefficients `[a, b, c]` of the knee region polynomial `ax^2 + bx + c`.
    knee_region_polynomial: [f64; 3],
    gain_curve_limiter_d1: f64,
    gain_curve_limiter_d2: f64,
    gain_curve_limiter_i1: f64,
    gain_curve_limiter_i2: f64,
}

fn compute_knee_start(
    max_input_level_db: f64,
    knee_smoothness_db: f64,
    compression_ratio: f64,
) -> f64 {
    debug_assert!(
        (compression_ratio - 1.0) * knee_smoothness_db / (2.0 * compression_ratio)
            < max_input_level_db
    );
    -knee_smoothness_db / 2.0 - max_input_level_db / (compression_ratio - 1.0)
}

fn compute_knee_region_polynomial(
    knee_start_dbfs: f64,
    knee_smoothness_db: f64,
    compression_ratio: f64,
) -> [f64; 3] {
    let a = (1.0 - compression_ratio) / (2.0 * knee_smoothness_db * compression_ratio);
    let b = 1.0 - 2.0 * a * knee_start_dbfs;
    let c = a * knee_start_dbfs * knee_start_dbfs;
    [a, b, c]
}

fn compute_limiter_d1(max_input_level_db: f64, compression_ratio: f64) -> f64 {
    (10.0_f64.powf(-max_input_level_db / (20.0 * compression_ratio)) * (1.0 - compression_ratio)
        / compression_ratio)
        / MAX_ABS_FLOAT_S16_VALUE as f64
}

const fn compute_limiter_d2(compression_ratio: f64) -> f64 {
    (1.0 - 2.0 * compression_ratio) / compression_ratio
}

fn compute_limiter_i2(
    max_input_level_db: f64,
    compression_ratio: f64,
    gain_curve_limiter_i1: f64,
) -> f64 {
    debug_assert!(gain_curve_limiter_i1 != 0.0);
    10.0_f64.powf(-max_input_level_db / (20.0 * compression_ratio))
        / gain_curve_limiter_i1
        / (MAX_ABS_FLOAT_S16_VALUE as f64).powf(gain_curve_limiter_i1 - 1.0)
}

impl Default for LimiterDbGainCurve {
    fn default() -> Self {
        let max_input_level_db = LIMITER_MAX_INPUT_LEVEL_DB_FS;
        let knee_smoothness_db = LIMITER_KNEE_SMOOTHNESS_DB;
        let compression_ratio = LIMITER_COMPRESSION_RATIO;

        let max_input_level_linear = dbfs_to_float_s16_f64(max_input_level_db);
        let knee_start_dbfs =
            compute_knee_start(max_input_level_db, knee_smoothness_db, compression_ratio);
        let knee_start_linear = dbfs_to_float_s16_f64(knee_start_dbfs);
        let limiter_start_dbfs = knee_start_dbfs + knee_smoothness_db;
        let limiter_start_linear = dbfs_to_float_s16_f64(limiter_start_dbfs);
        let knee_region_polynomial =
            compute_knee_region_polynomial(knee_start_dbfs, knee_smoothness_db, compression_ratio);
        let gain_curve_limiter_d1 = compute_limiter_d1(max_input_level_db, compression_ratio);
        let gain_curve_limiter_d2 = compute_limiter_d2(compression_ratio);
        let gain_curve_limiter_i1 = 1.0 / compression_ratio;
        let gain_curve_limiter_i2 =
            compute_limiter_i2(max_input_level_db, compression_ratio, gain_curve_limiter_i1);

        debug_assert!(knee_smoothness_db > 0.0);
        debug_assert!(compression_ratio > 1.0);
        debug_assert!(max_input_level_db >= knee_start_dbfs + knee_smoothness_db);

        Self {
            max_input_level_linear,
            knee_start_dbfs,
            knee_start_linear,
            limiter_start_dbfs,
            limiter_start_linear,
            knee_region_polynomial,
            gain_curve_limiter_d1,
            gain_curve_limiter_d2,
            gain_curve_limiter_i1,
            gain_curve_limiter_i2,
        }
    }
}

impl LimiterDbGainCurve {
    pub(crate) fn max_input_level_db(&self) -> f64 {
        LIMITER_MAX_INPUT_LEVEL_DB_FS
    }

    pub(crate) fn max_input_level_linear(&self) -> f64 {
        self.max_input_level_linear
    }

    pub(crate) fn knee_start_linear(&self) -> f64 {
        self.knee_start_linear
    }

    pub(crate) fn limiter_start_linear(&self) -> f64 {
        self.limiter_start_linear
    }

    /// Returns the output level in dBFS given an input level in dBFS.
    pub(crate) fn get_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        if input_level_dbfs < self.knee_start_dbfs {
            input_level_dbfs
        } else if input_level_dbfs < self.limiter_start_dbfs {
            self.get_knee_region_output_level_dbfs(input_level_dbfs)
        } else {
            self.get_compressor_region_output_level_dbfs(input_level_dbfs)
        }
    }

    /// Returns the gain (linear scale) for a given input level (linear scale).
    pub(crate) fn get_gain_linear(&self, input_level_linear: f64) -> f64 {
        if input_level_linear < self.knee_start_linear {
            return 1.0;
        }
        dbfs_to_float_s16_f64(self.get_output_level_dbfs(float_s16_to_dbfs_f64(input_level_linear)))
            / input_level_linear
    }

    /// Computes the first derivative of `get_gain_linear()` at `x`.
    /// Beyond-knee region only.
    pub(crate) fn get_gain_first_derivative_linear(&self, x: f64) -> f64 {
        debug_assert!(x >= self.limiter_start_linear - 1e-7 * MAX_ABS_FLOAT_S16_VALUE as f64);
        self.gain_curve_limiter_d1
            * (x / MAX_ABS_FLOAT_S16_VALUE as f64).powf(self.gain_curve_limiter_d2)
    }

    /// Computes the integral of `get_gain_linear()` in the range `[x0, x1]`.
    /// Beyond-knee region only.
    pub(crate) fn get_gain_integral_linear(&self, x0: f64, x1: f64) -> f64 {
        debug_assert!(x0 <= x1);
        debug_assert!(x0 >= self.limiter_start_linear);
        let limiter_integral =
            |x: f64| self.gain_curve_limiter_i2 * x.powf(self.gain_curve_limiter_i1);
        limiter_integral(x1) - limiter_integral(x0)
    }

    fn get_knee_region_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        self.knee_region_polynomial[0] * input_level_dbfs * input_level_dbfs
            + self.knee_region_polynomial[1] * input_level_dbfs
            + self.knee_region_polynomial[2]
    }

    fn get_compressor_region_output_level_dbfs(&self, input_level_dbfs: f64) -> f64 {
        (input_level_dbfs - LIMITER_MAX_INPUT_LEVEL_DB_FS) / LIMITER_COMPRESSION_RATIO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct() {
        let _l = LimiterDbGainCurve::default();
    }

    #[test]
    fn gain_curve_should_be_monotone() {
        let l = LimiterDbGainCurve::default();
        let mut last_output_level = None;
        let mut level = -90.0_f64;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            if let Some(last) = last_output_level {
                assert!(
                    last <= current_output_level,
                    "not monotone at level {level}: {last} > {current_output_level}"
                );
            }
            last_output_level = Some(current_output_level);
            level += 0.5;
        }
    }

    #[test]
    fn gain_curve_should_be_continuous() {
        let l = LimiterDbGainCurve::default();
        let mut last_output_level = None;
        const MAX_DELTA: f64 = 0.5;
        let mut level = -90.0_f64;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            if let Some(last) = last_output_level {
                assert!(
                    current_output_level <= last + MAX_DELTA,
                    "not continuous at level {level}"
                );
            }
            last_output_level = Some(current_output_level);
            level += 0.5;
        }
    }

    #[test]
    fn output_gain_should_be_less_than_full_scale() {
        let l = LimiterDbGainCurve::default();
        let mut level = -90.0_f64;
        while level <= l.max_input_level_db() {
            let current_output_level = l.get_output_level_dbfs(level);
            assert!(
                current_output_level <= 0.0,
                "output {current_output_level} > 0 at level {level}"
            );
            level += 0.5;
        }
    }
}
