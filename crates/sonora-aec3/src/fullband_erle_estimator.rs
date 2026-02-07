//! Fullband ERLE estimator.
//!
//! Estimates the echo return loss enhancement using the energy of all frequency
//! bands. Includes instantaneous ERLE tracking and linear filter quality
//! estimation.
//!
//! Ported from `modules/audio_processing/aec3/fullband_erle_estimator.h/cc`.

use crate::common::{FFT_LENGTH_BY_2_PLUS_1, fast_approx_log2f};
use crate::config::Erle as ErleConfig;

const EPSILON: f32 = 1e-3;
const X2_BAND_ENERGY_THRESHOLD: f32 = 44015068.0;
const BLOCKS_TO_HOLD_ERLE: i32 = 100;
const POINTS_TO_ACCUMULATE: i32 = 6;

/// Instantaneous ERLE estimator with quality tracking.
struct ErleInstantaneous {
    clamp_inst_quality_to_zero: bool,
    clamp_inst_quality_to_one: bool,
    erle_log2: Option<f32>,
    inst_quality_estimate: f32,
    max_erle_log2: f32,
    min_erle_log2: f32,
    y2_acum: f32,
    e2_acum: f32,
    num_points: i32,
}

impl ErleInstantaneous {
    fn new(config: &ErleConfig) -> Self {
        let mut s = Self {
            clamp_inst_quality_to_zero: config.clamp_quality_estimate_to_zero,
            clamp_inst_quality_to_one: config.clamp_quality_estimate_to_one,
            erle_log2: None,
            inst_quality_estimate: 0.0,
            max_erle_log2: 0.0,
            min_erle_log2: 0.0,
            y2_acum: 0.0,
            e2_acum: 0.0,
            num_points: 0,
        };
        s.reset();
        s
    }

    /// Updates the estimator with a new point. Returns `true` if the
    /// instantaneous ERLE was updated due to having enough points.
    fn update(&mut self, y2_sum: f32, e2_sum: f32) -> bool {
        let mut update_estimates = false;
        self.e2_acum += e2_sum;
        self.y2_acum += y2_sum;
        self.num_points += 1;
        if self.num_points == POINTS_TO_ACCUMULATE {
            if self.e2_acum > 0.0 {
                update_estimates = true;
                self.erle_log2 = Some(fast_approx_log2f(self.y2_acum / self.e2_acum + EPSILON));
            }
            self.num_points = 0;
            self.e2_acum = 0.0;
            self.y2_acum = 0.0;
        }

        if update_estimates {
            self.update_max_min();
            self.update_quality_estimate();
        }
        update_estimates
    }

    /// Resets the instantaneous ERLE estimator to its initial state.
    fn reset(&mut self) {
        self.reset_accumulators();
        self.max_erle_log2 = -10.0; // -30 dB
        self.min_erle_log2 = 33.0; // 100 dB
        self.inst_quality_estimate = 0.0;
    }

    /// Resets the members related with an instantaneous estimate.
    fn reset_accumulators(&mut self) {
        self.erle_log2 = None;
        self.inst_quality_estimate = 0.0;
        self.num_points = 0;
        self.e2_acum = 0.0;
        self.y2_acum = 0.0;
    }

    /// Returns the instantaneous ERLE in log2 units.
    fn get_inst_erle_log2(&self) -> Option<f32> {
        self.erle_log2
    }

    /// Gets an indication between 0 and 1 of the performance of the linear
    /// filter for the current time instant.
    fn get_quality_estimate(&self) -> Option<f32> {
        self.erle_log2.map(|_| {
            let mut value = self.inst_quality_estimate;
            if self.clamp_inst_quality_to_zero {
                value = value.max(0.0);
            }
            if self.clamp_inst_quality_to_one {
                value = value.min(1.0);
            }
            value
        })
    }

    fn update_max_min(&mut self) {
        debug_assert!(self.erle_log2.is_some());
        let erle = self.erle_log2.unwrap();
        // Forget factor, approx 1 dB every 3 sec.
        self.max_erle_log2 -= 0.0004;
        self.max_erle_log2 = self.max_erle_log2.max(erle);
        self.min_erle_log2 += 0.0004;
        self.min_erle_log2 = self.min_erle_log2.min(erle);
    }

    fn update_quality_estimate(&mut self) {
        let alpha = 0.07;
        let mut quality_estimate = 0.0;
        debug_assert!(self.erle_log2.is_some());
        let erle = self.erle_log2.unwrap();
        if self.max_erle_log2 > self.min_erle_log2 {
            quality_estimate =
                (erle - self.min_erle_log2) / (self.max_erle_log2 - self.min_erle_log2);
        }
        if quality_estimate > self.inst_quality_estimate {
            self.inst_quality_estimate = quality_estimate;
        } else {
            self.inst_quality_estimate += alpha * (quality_estimate - self.inst_quality_estimate);
        }
    }
}

/// Estimates the echo return loss enhancement using the energy of all
/// frequency bands.
pub(crate) struct FullBandErleEstimator {
    min_erle_log2: f32,
    #[allow(dead_code, reason = "will be used in later steps")]
    max_erle_lf_log2: f32,
    hold_counters_instantaneous_erle: Vec<i32>,
    erle_time_domain_log2: Vec<f32>,
    instantaneous_erle: Vec<ErleInstantaneous>,
    linear_filters_qualities: Vec<Option<f32>>,
}

impl FullBandErleEstimator {
    pub(crate) fn new(config: &ErleConfig, num_capture_channels: usize) -> Self {
        let min_erle_log2 = fast_approx_log2f(config.min + EPSILON);
        let max_erle_lf_log2 = fast_approx_log2f(config.max_l + EPSILON);
        let mut s = Self {
            min_erle_log2,
            max_erle_lf_log2,
            hold_counters_instantaneous_erle: vec![0; num_capture_channels],
            erle_time_domain_log2: vec![min_erle_log2; num_capture_channels],
            instantaneous_erle: (0..num_capture_channels)
                .map(|_| ErleInstantaneous::new(config))
                .collect(),
            linear_filters_qualities: vec![None; num_capture_channels],
        };
        s.reset();
        s
    }

    /// Resets the ERLE estimator.
    pub(crate) fn reset(&mut self) {
        for inst in &mut self.instantaneous_erle {
            inst.reset();
        }
        self.update_quality_estimates();
        self.erle_time_domain_log2.fill(self.min_erle_log2);
        self.hold_counters_instantaneous_erle.fill(0);
    }

    /// Updates the ERLE estimator.
    pub(crate) fn update(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        for ch in 0..y2.len() {
            if converged_filters[ch] {
                let x2_sum: f32 = x2.iter().sum();
                if x2_sum > X2_BAND_ENERGY_THRESHOLD * x2.len() as f32 {
                    let y2_sum: f32 = y2[ch].iter().sum();
                    let e2_sum: f32 = e2[ch].iter().sum();
                    if self.instantaneous_erle[ch].update(y2_sum, e2_sum) {
                        self.hold_counters_instantaneous_erle[ch] = BLOCKS_TO_HOLD_ERLE;
                        let inst_erle = self.instantaneous_erle[ch].get_inst_erle_log2().unwrap();
                        self.erle_time_domain_log2[ch] +=
                            0.05 * (inst_erle - self.erle_time_domain_log2[ch]);
                        self.erle_time_domain_log2[ch] =
                            self.erle_time_domain_log2[ch].max(self.min_erle_log2);
                    }
                }
            }
            self.hold_counters_instantaneous_erle[ch] -= 1;
            if self.hold_counters_instantaneous_erle[ch] == 0 {
                self.instantaneous_erle[ch].reset_accumulators();
            }
        }
        self.update_quality_estimates();
    }

    /// Returns the fullband ERLE estimates in log2 units.
    pub(crate) fn fullband_erle_log2(&self) -> f32 {
        self.erle_time_domain_log2
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(self.min_erle_log2)
    }

    /// Returns an estimation of the current linear filter quality.
    pub(crate) fn get_inst_linear_quality_estimates(&self) -> &[Option<f32>] {
        &self.linear_filters_qualities
    }

    fn update_quality_estimates(&mut self) {
        for ch in 0..self.instantaneous_erle.len() {
            self.linear_filters_qualities[ch] = self.instantaneous_erle[ch].get_quality_estimate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Erle as ErleConfig;

    #[test]
    fn creation_and_reset() {
        let config = ErleConfig::default();
        let est = FullBandErleEstimator::new(&config, 2);
        let min_log2 = fast_approx_log2f(config.min + EPSILON);
        assert!((est.fullband_erle_log2() - min_log2).abs() < 0.001);
        assert_eq!(est.get_inst_linear_quality_estimates().len(), 2);
    }

    #[test]
    fn erle_increases_with_echo() {
        let config = ErleConfig::default();
        let mut est = FullBandErleEstimator::new(&config, 1);
        let min_log2 = fast_approx_log2f(config.min + EPSILON);

        // Create spectra where echo is present: Y2 >> E2.
        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let erle_target = 10.0f32;
        let mut y2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            y2[0][k] = x2[k] * 9.0;
            e2[0][k] = y2[0][k] / erle_target;
        }
        let converged = [true];

        for _ in 0..200 {
            est.update(&x2, &y2, &e2, &converged);
        }

        // ERLE should have increased above the minimum.
        assert!(est.fullband_erle_log2() > min_log2 + 0.5);
    }

    #[test]
    fn quality_estimate_available_after_updates() {
        let config = ErleConfig::default();
        let mut est = FullBandErleEstimator::new(&config, 1);

        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let mut y2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            y2[0][k] = x2[k] * 9.0;
            e2[0][k] = y2[0][k] / 10.0;
        }
        let converged = [true];

        // Before enough points, quality is None.
        assert!(est.get_inst_linear_quality_estimates()[0].is_none());

        // After POINTS_TO_ACCUMULATE updates, quality should become available.
        for _ in 0..(POINTS_TO_ACCUMULATE + 1) {
            est.update(&x2, &y2, &e2, &converged);
        }
        assert!(est.get_inst_linear_quality_estimates()[0].is_some());
    }

    #[test]
    fn unconverged_filter_does_not_update() {
        let config = ErleConfig::default();
        let mut est = FullBandErleEstimator::new(&config, 1);
        let initial = est.fullband_erle_log2();

        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let mut y2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = [[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            y2[0][k] = x2[k] * 9.0;
            e2[0][k] = y2[0][k] / 10.0;
        }
        let converged = [false]; // NOT converged

        for _ in 0..100 {
            est.update(&x2, &y2, &e2, &converged);
        }
        assert!((est.fullband_erle_log2() - initial).abs() < 0.001);
    }
}
