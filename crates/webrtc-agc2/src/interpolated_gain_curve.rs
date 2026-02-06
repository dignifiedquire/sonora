//! Interpolated gain curve using piecewise-linear under-approximation.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/interpolated_gain_curve.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{INTERPOLATED_GAIN_CURVE_KNEE_POINTS, INTERPOLATED_GAIN_CURVE_TOTAL_POINTS};

/// Defined as `DbfsToLinear(kLimiterMaxInputLevelDbFs)`.
const MAX_INPUT_LEVEL_LINEAR: f32 = 36_766.3;

/// Input level scaling factor (32768.0).
pub(crate) const INPUT_LEVEL_SCALING_FACTOR: f32 = 32768.0;

/// Region of the gain curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GainCurveRegion {
    Identity = 0,
    Knee = 1,
    Limiter = 2,
    Saturation = 3,
}

/// Statistics tracking for gain curve lookups.
#[derive(Debug, Clone)]
pub(crate) struct Stats {
    pub look_ups_identity_region: usize,
    pub look_ups_knee_region: usize,
    pub look_ups_limiter_region: usize,
    pub look_ups_saturation_region: usize,
    pub available: bool,
    pub region: GainCurveRegion,
    pub region_duration_frames: i64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            look_ups_identity_region: 0,
            look_ups_knee_region: 0,
            look_ups_limiter_region: 0,
            look_ups_saturation_region: 0,
            available: false,
            region: GainCurveRegion::Identity,
            region_duration_frames: 0,
        }
    }
}

/// Interpolated gain curve using piecewise-linear under-approximation to avoid
/// saturation.
#[derive(Default)]
pub(crate) struct InterpolatedGainCurve {
    stats: Stats,
}

// Pre-computed approximation parameters (x boundaries, slopes m, intercepts q).
// These are static constexpr in the C++ header.
const APPROXIMATION_PARAMS_X: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
    30057.297, 30148.986, 30240.676, 30424.053, 30607.43, 30790.807, 30974.184, 31157.56, 31340.94,
    31524.316, 31707.693, 31891.07, 32074.447, 32257.824, 32441.201, 32624.58, 32807.957,
    32991.332, 33174.71, 33358.09, 33541.465, 33724.844, 33819.535, 34009.54, 34200.06, 34389.816,
    34674.49, 35054.375, 35434.863, 35814.816, 36195.168, 36575.03,
];

const APPROXIMATION_PARAMS_M: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
    -3.515_235_7e-07,
    -1.050_251_6e-06,
    -2.085_213_7e-06,
    -3.443_004_7e-06,
    -4.773_849_5e-06,
    -6.077_376e-06,
    -7.353_258e-6,
    -8.601_22e-06,
    -9.821_013e-06,
    -1.101_243_4e-05,
    -1.217_532_6e-05,
    -1.330_956_9e-05,
    -1.441_507_5e-05,
    -1.549_179_3e-05,
    -1.653_970_7e-05,
    -1.755_882_8e-05,
    -1.854_918_4e-05,
    -1.951_086_8e-05,
    -2.044_398e-05,
    -2.134_862_7e-05,
    -2.222_497e-5,
    -2.265_374_7e-05,
    -2.242_571e-5,
    -2.220_122e-05,
    -2.198_021e-05,
    -2.176_260_2e-05,
    -2.133_731_7e-05,
    -2.092_482e-5,
    -2.052_459_6e-05,
    -2.013_615_4e-05,
    -1.975_903e-5,
    -1.939_277_9e-05,
];

const APPROXIMATION_PARAMS_Q: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS] = [
    1.010_565_9,
    1.031_631_8,
    1.062_929_7,
    1.104_239_2,
    1.144_973,
    1.185_109_6,
    1.224_629,
    1.263_512_5,
    1.301_742,
    1.339_300_6,
    1.376_173_3,
    1.412_345_5,
    1.447_804,
    1.482_536_6,
    1.516_532_2,
    1.549_780_6,
    1.582_272_2,
    1.613_999_4,
    1.644_955,
    1.675_132_4,
    1.704_526_2,
    1.718_986_6,
    1.711_274_5,
    1.703_639_7,
    1.696_081_2,
    1.688_597_7,
    1.673_851_1,
    1.659_391_3,
    1.645_209_4,
    1.631_297_5,
    1.617_647_4,
    1.604_251_7,
];

impl InterpolatedGainCurve {
    pub(crate) fn get_stats(&self) -> &Stats {
        &self.stats
    }

    /// Given a non-negative input level (linear scale), returns a scalar gain
    /// factor to apply. Levels above `kLimiterMaxInputLevelDbFs` will be
    /// reduced to 0 dBFS after applying this gain.
    pub(crate) fn look_up_gain_to_apply(&mut self, input_level: f32) -> f32 {
        self.update_stats(input_level);

        if input_level <= APPROXIMATION_PARAMS_X[0] {
            // Identity region.
            return 1.0;
        }

        if input_level >= MAX_INPUT_LEVEL_LINEAR {
            // Saturation region: clamp to clipping level.
            return 32768.0 / input_level;
        }

        // Knee and limiter regions: find the linear piece index via binary search.
        let index = match APPROXIMATION_PARAMS_X
            .binary_search_by(|x| x.partial_cmp(&input_level).unwrap())
        {
            Ok(i) => i.saturating_sub(1),
            Err(i) => i.saturating_sub(1),
        };

        debug_assert!(index < APPROXIMATION_PARAMS_M.len());
        debug_assert!(APPROXIMATION_PARAMS_X[index] <= input_level);

        // Piecewise linear interpolation.
        let gain = APPROXIMATION_PARAMS_M[index] * input_level + APPROXIMATION_PARAMS_Q[index];
        debug_assert!(gain >= 0.0);
        gain
    }

    fn update_stats(&mut self, input_level: f32) {
        self.stats.available = true;

        let region = if input_level < APPROXIMATION_PARAMS_X[0] {
            self.stats.look_ups_identity_region += 1;
            GainCurveRegion::Identity
        } else if input_level < APPROXIMATION_PARAMS_X[INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1] {
            self.stats.look_ups_knee_region += 1;
            GainCurveRegion::Knee
        } else if input_level < MAX_INPUT_LEVEL_LINEAR {
            self.stats.look_ups_limiter_region += 1;
            GainCurveRegion::Limiter
        } else {
            self.stats.look_ups_saturation_region += 1;
            GainCurveRegion::Saturation
        };

        if region == self.stats.region {
            self.stats.region_duration_frames += 1;
        } else {
            self.stats.region_duration_frames = 0;
            self.stats.region = region;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{MAX_ABS_FLOAT_S16_VALUE, dbfs_to_float_s16};
    use crate::limiter_db_gain_curve::LimiterDbGainCurve;

    const LEVEL_EPSILON: f64 = 1e-2 * MAX_ABS_FLOAT_S16_VALUE as f64;
    const TOLERANCE: f32 = 1.0 / 32768.0;

    /// Returns evenly spaced points over `[l, r]`.
    fn lin_space(l: f64, r: f64, num_points: usize) -> Vec<f64> {
        assert!(num_points >= 2);
        let step = (r - l) / (num_points - 1) as f64;
        let mut points = Vec::with_capacity(num_points);
        points.push(l);
        for i in 1..num_points - 1 {
            points.push(l + i as f64 * step);
        }
        points.push(r);
        points
    }

    #[test]
    fn create_use() {
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            LEVEL_EPSILON,
            dbfs_to_float_s16(limiter.max_input_level_db() as f32 + 1.0) as f64,
            500,
        );
        for level in levels {
            assert!(igc.look_up_gain_to_apply(level as f32) >= 0.0);
        }
    }

    #[test]
    fn check_valid_output() {
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(LEVEL_EPSILON, limiter.max_input_level_linear() * 2.0, 500);
        for level in levels {
            let gain = igc.look_up_gain_to_apply(level as f32);
            assert!(
                (0.0..=1.0).contains(&gain),
                "gain {gain} out of range at level {level}"
            );
        }
    }

    #[test]
    fn check_monotonicity() {
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            LEVEL_EPSILON,
            limiter.max_input_level_linear() + LEVEL_EPSILON + 0.5,
            500,
        );
        let mut prev_gain = igc.look_up_gain_to_apply(0.0);
        for level in levels {
            let gain = igc.look_up_gain_to_apply(level as f32);
            assert!(
                prev_gain >= gain,
                "not monotone at level {level}: {prev_gain} < {gain}"
            );
            prev_gain = gain;
        }
    }

    #[test]
    fn check_approximation() {
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            LEVEL_EPSILON,
            limiter.max_input_level_linear() - LEVEL_EPSILON,
            500,
        );
        for level in levels {
            let diff = (limiter.get_gain_linear(level) as f32
                - igc.look_up_gain_to_apply(level as f32))
            .abs();
            assert!(
                diff < TOLERANCE,
                "approximation error {diff} at level {level}"
            );
        }
    }

    #[test]
    fn check_region_boundaries() {
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();

        igc.look_up_gain_to_apply(LEVEL_EPSILON as f32);
        igc.look_up_gain_to_apply((limiter.knee_start_linear() + LEVEL_EPSILON) as f32);
        igc.look_up_gain_to_apply((limiter.limiter_start_linear() + LEVEL_EPSILON) as f32);
        igc.look_up_gain_to_apply((limiter.max_input_level_linear() + LEVEL_EPSILON) as f32);

        let stats = igc.get_stats();
        assert_eq!(1, stats.look_ups_identity_region);
        assert_eq!(1, stats.look_ups_knee_region);
        assert_eq!(1, stats.look_ups_limiter_region);
        assert_eq!(1, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_identity_region() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(LEVEL_EPSILON, limiter.knee_start_linear(), NUM_STEPS);
        for level in &levels {
            assert_eq!(1.0, igc.look_up_gain_to_apply(*level as f32));
        }

        let stats = igc.get_stats();
        assert_eq!(NUM_STEPS - 1, stats.look_ups_identity_region);
        assert_eq!(1, stats.look_ups_knee_region);
        assert_eq!(0, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_knee() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            limiter.knee_start_linear() + LEVEL_EPSILON,
            limiter.limiter_start_linear(),
            NUM_STEPS,
        );
        for level in &levels {
            assert!(
                igc.look_up_gain_to_apply(*level as f32)
                    <= limiter.get_gain_linear(*level) as f32 + 1e-7,
                "over-approximation at level {level}"
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(NUM_STEPS - 1, stats.look_ups_knee_region);
        assert_eq!(1, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_beyond_knee() {
        const NUM_STEPS: usize = 10;
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            limiter.limiter_start_linear() + LEVEL_EPSILON,
            limiter.max_input_level_linear() - LEVEL_EPSILON,
            NUM_STEPS,
        );
        for level in &levels {
            assert!(
                igc.look_up_gain_to_apply(*level as f32)
                    <= limiter.get_gain_linear(*level) as f32 + 1e-7,
                "over-approximation at level {level}"
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(0, stats.look_ups_knee_region);
        assert_eq!(NUM_STEPS, stats.look_ups_limiter_region);
        assert_eq!(0, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_no_over_approximation_with_saturation() {
        const NUM_STEPS: usize = 3;
        let mut igc = InterpolatedGainCurve::default();
        let limiter = LimiterDbGainCurve::default();
        let levels = lin_space(
            limiter.max_input_level_linear() + LEVEL_EPSILON,
            limiter.max_input_level_linear() + LEVEL_EPSILON + 0.5,
            NUM_STEPS,
        );
        for level in &levels {
            assert!(
                igc.look_up_gain_to_apply(*level as f32) <= limiter.get_gain_linear(*level) as f32,
                "over-approximation at level {level}"
            );
        }

        let stats = igc.get_stats();
        assert_eq!(0, stats.look_ups_identity_region);
        assert_eq!(0, stats.look_ups_knee_region);
        assert_eq!(0, stats.look_ups_limiter_region);
        assert_eq!(NUM_STEPS, stats.look_ups_saturation_region);
    }

    #[test]
    fn check_approximation_params() {
        let params = compute_interpolated_gain_curve_approximation_params();
        let igc = InterpolatedGainCurve::default();

        for i in 0..INTERPOLATED_GAIN_CURVE_TOTAL_POINTS {
            assert!(
                (APPROXIMATION_PARAMS_X[i] - params.x[i]).abs() < 0.9,
                "x[{i}]: {} vs {}",
                APPROXIMATION_PARAMS_X[i],
                params.x[i]
            );
            assert!(
                (APPROXIMATION_PARAMS_M[i] - params.m[i]).abs() < 0.00001,
                "m[{i}]: {} vs {}",
                APPROXIMATION_PARAMS_M[i],
                params.m[i]
            );
            assert!(
                (APPROXIMATION_PARAMS_Q[i] - params.q[i]).abs() < 0.001,
                "q[{i}]: {} vs {}",
                APPROXIMATION_PARAMS_Q[i],
                params.q[i]
            );
        }
        // Silence unused variable warning.
        let _ = igc;
    }

    /// Computed approximation parameters for verification.
    struct InterpolatedParameters {
        x: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
        m: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
        q: [f32; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
    }

    /// Recomputes the interpolated gain curve parameters from the analytical
    /// `LimiterDbGainCurve`. Port of `compute_interpolated_gain_curve.cc`.
    fn compute_interpolated_gain_curve_approximation_params() -> InterpolatedParameters {
        use crate::common::{
            INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS, INTERPOLATED_GAIN_CURVE_KNEE_POINTS,
        };

        let limiter = LimiterDbGainCurve::default();

        let mut params = InterpolatedParameters {
            x: [0.0; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
            m: [0.0; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
            q: [0.0; INTERPOLATED_GAIN_CURVE_TOTAL_POINTS],
        };

        // Knee region: equally spaced points with an extra point at the start.
        let knee_points = lin_space(
            limiter.knee_start_linear(),
            limiter.limiter_start_linear(),
            INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1,
        );
        params.x[0] = knee_points[0] as f32;
        params.x[1] = ((knee_points[0] + knee_points[1]) / 2.0) as f32;
        for (i, &kp) in knee_points.iter().enumerate().skip(1) {
            params.x[i + 1] = kp as f32;
        }

        // Compute (m, q) for knee region linear pieces.
        for i in 0..INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1 {
            let x0 = params.x[i] as f64;
            let x1 = params.x[i + 1] as f64;
            let y0 = limiter.get_gain_linear(x0);
            let y1 = limiter.get_gain_linear(x1);
            params.m[i] = ((y1 - y0) / (x1 - x0)) as f32;
            params.q[i] = (y0 - params.m[i] as f64 * x0) as f32;
        }

        // Beyond-knee region: sample using greedy error minimization.
        let samples = sample_limiter_region(&limiter);

        // First beyond-knee piece uses the last knee point as boundary.
        let (m, q) = compute_linear_approximation_params(
            &limiter,
            params.x[INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1] as f64,
        );
        params.m[INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1] = m as f32;
        params.q[INTERPOLATED_GAIN_CURVE_KNEE_POINTS - 1] = q as f32;

        for (i, &sample) in samples.iter().enumerate() {
            let (m, q) = compute_linear_approximation_params(&limiter, sample);
            params.m[i + INTERPOLATED_GAIN_CURVE_KNEE_POINTS] = m as f32;
            params.q[i + INTERPOLATED_GAIN_CURVE_KNEE_POINTS] = q as f32;
        }

        // Find intersection points between adjacent linear pieces.
        for i in INTERPOLATED_GAIN_CURVE_KNEE_POINTS
            ..INTERPOLATED_GAIN_CURVE_KNEE_POINTS + INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS
        {
            params.x[i] = ((params.q[i - 1] as f64 - params.q[i] as f64)
                / (params.m[i] as f64 - params.m[i - 1] as f64)) as f32;
        }

        params
    }

    fn compute_linear_approximation_params(limiter: &LimiterDbGainCurve, x: f64) -> (f64, f64) {
        let m = limiter.get_gain_first_derivative_linear(x);
        let q = limiter.get_gain_linear(x) - m * x;
        (m, q)
    }

    fn limiter_under_approximation_negative_error(
        limiter: &LimiterDbGainCurve,
        x0: f64,
        x1: f64,
    ) -> f64 {
        let area_limiter = limiter.get_gain_integral_linear(x0, x1);
        let area_interpolated = compute_area_under_piecewise_linear_approximation(limiter, x0, x1);
        area_limiter - area_interpolated
    }

    fn compute_area_under_piecewise_linear_approximation(
        limiter: &LimiterDbGainCurve,
        x0: f64,
        x1: f64,
    ) -> f64 {
        let (m0, q0) = compute_linear_approximation_params(limiter, x0);
        let (m1, q1) = compute_linear_approximation_params(limiter, x1);
        let x_split = (q0 - q1) / (m1 - m0);

        let area =
            |xl: f64, xr: f64, m: f64, q: f64| xr * (m * xr / 2.0 + q) - xl * (m * xl / 2.0 + q);
        area(x0, x_split, m0, q0) + area(x_split, x1, m1, q1)
    }

    fn sample_limiter_region(limiter: &LimiterDbGainCurve) -> Vec<f64> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct Interval {
            x0: f64,
            x1: f64,
            error: f64,
        }

        impl Eq for Interval {}

        impl PartialOrd for Interval {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Interval {
            fn cmp(&self, other: &Self) -> Ordering {
                self.error.total_cmp(&other.error)
            }
        }

        let mut heap = BinaryHeap::new();
        let start = limiter.limiter_start_linear();
        let end = limiter.max_input_level_linear();
        heap.push(Interval {
            x0: start,
            x1: end,
            error: limiter_under_approximation_negative_error(limiter, start, end),
        });

        use crate::common::INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS;
        while heap.len() < INTERPOLATED_GAIN_CURVE_BEYOND_KNEE_POINTS {
            let interval = heap.pop().unwrap();
            let x_split = (interval.x0 + interval.x1) / 2.0;
            heap.push(Interval {
                x0: interval.x0,
                x1: x_split,
                error: limiter_under_approximation_negative_error(limiter, interval.x0, x_split),
            });
            heap.push(Interval {
                x0: x_split,
                x1: interval.x1,
                error: limiter_under_approximation_negative_error(limiter, x_split, interval.x1),
            });
        }

        let mut samples: Vec<f64> = heap.into_iter().map(|i| i.x1).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples
    }
}
