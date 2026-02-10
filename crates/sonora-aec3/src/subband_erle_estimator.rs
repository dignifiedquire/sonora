//! Subband ERLE estimator.
//!
//! Estimates the echo return loss enhancement for each frequency subband,
//! with optional onset detection and compensation.
//!
//! Ported from `modules/audio_processing/aec3/subband_erle_estimator.h/cc`.

use crate::common::{
    BLOCKS_TO_HOLD_ERLE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, X2_BAND_ENERGY_THRESHOLD,
};
use crate::config::EchoCanceller3Config;
const BLOCKS_FOR_ONSET_DETECTION: i32 = BLOCKS_TO_HOLD_ERLE + 150;
const POINTS_TO_ACCUMULATE: i32 = 6;

fn set_max_erle_bands(max_erle_l: f32, max_erle_h: f32) -> [f32; FFT_LENGTH_BY_2_PLUS_1] {
    let mut max_erle = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
    max_erle[..FFT_LENGTH_BY_2 / 2].fill(max_erle_l);
    max_erle[FFT_LENGTH_BY_2 / 2..].fill(max_erle_h);
    max_erle
}

#[derive(Debug)]
struct AccumulatedSpectra {
    y2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    e2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    low_render_energy: Vec<[bool; FFT_LENGTH_BY_2_PLUS_1]>,
    num_points: Vec<i32>,
}

impl AccumulatedSpectra {
    fn new(num_capture_channels: usize) -> Self {
        Self {
            y2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            e2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            low_render_energy: vec![[false; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            num_points: vec![0; num_capture_channels],
        }
    }
}

/// Estimates the echo return loss enhancement for each frequency subband.
#[derive(Debug)]
pub(crate) struct SubbandErleEstimator {
    use_onset_detection: bool,
    min_erle: f32,
    max_erle: [f32; FFT_LENGTH_BY_2_PLUS_1],
    use_min_erle_during_onsets: bool,
    accum_spectra: AccumulatedSpectra,
    erle: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_onset_compensated: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_unbounded: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_during_onsets: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    coming_onset: Vec<[bool; FFT_LENGTH_BY_2_PLUS_1]>,
    hold_counters: Vec<[i32; FFT_LENGTH_BY_2_PLUS_1]>,
}

impl SubbandErleEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        // Default to true (kill switch not enabled).
        let use_min_erle_during_onsets = true;

        let mut s = Self {
            use_onset_detection: config.erle.onset_detection,
            min_erle: config.erle.min,
            max_erle: set_max_erle_bands(config.erle.max_l, config.erle.max_h),
            use_min_erle_during_onsets,
            accum_spectra: AccumulatedSpectra::new(num_capture_channels),
            erle: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            erle_onset_compensated: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            erle_unbounded: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            erle_during_onsets: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            coming_onset: vec![[false; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            hold_counters: vec![[0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
        };
        s.reset();
        s
    }

    /// Resets the ERLE estimator.
    pub(crate) fn reset(&mut self) {
        let num_capture_channels = self.erle.len();
        for ch in 0..num_capture_channels {
            self.erle[ch].fill(self.min_erle);
            self.erle_onset_compensated[ch].fill(self.min_erle);
            self.erle_unbounded[ch].fill(self.min_erle);
            self.erle_during_onsets[ch].fill(self.min_erle);
            self.coming_onset[ch].fill(true);
            self.hold_counters[ch].fill(0);
        }
        self.reset_accumulated_spectra();
    }

    /// Updates the ERLE estimate.
    pub(crate) fn update(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        self.update_accumulated_spectra(x2, y2, e2, converged_filters);
        self.update_bands(converged_filters);

        if self.use_onset_detection {
            self.decrease_erle_per_band_for_low_render_signals();
        }

        let num_capture_channels = self.erle.len();
        for ch in 0..num_capture_channels {
            self.erle[ch][0] = self.erle[ch][1];
            self.erle[ch][FFT_LENGTH_BY_2] = self.erle[ch][FFT_LENGTH_BY_2 - 1];

            self.erle_onset_compensated[ch][0] = self.erle_onset_compensated[ch][1];
            self.erle_onset_compensated[ch][FFT_LENGTH_BY_2] =
                self.erle_onset_compensated[ch][FFT_LENGTH_BY_2 - 1];

            self.erle_unbounded[ch][0] = self.erle_unbounded[ch][1];
            self.erle_unbounded[ch][FFT_LENGTH_BY_2] = self.erle_unbounded[ch][FFT_LENGTH_BY_2 - 1];
        }
    }

    /// Returns the ERLE estimate.
    pub(crate) fn erle(&self, onset_compensated: bool) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if onset_compensated && self.use_onset_detection {
            &self.erle_onset_compensated
        } else {
            &self.erle
        }
    }

    /// Returns the non-capped ERLE estimate.
    pub(crate) fn erle_unbounded(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.erle_unbounded
    }

    fn update_bands(&mut self, converged_filters: &[bool]) {
        let num_capture_channels = self.accum_spectra.y2.len();
        for (ch, &converged) in converged_filters
            .iter()
            .enumerate()
            .take(num_capture_channels)
        {
            if !converged {
                continue;
            }

            if self.accum_spectra.num_points[ch] != POINTS_TO_ACCUMULATE {
                continue;
            }

            let mut new_erle = [0.0f32; FFT_LENGTH_BY_2];
            let mut is_erle_updated = [false; FFT_LENGTH_BY_2];

            for k in 1..FFT_LENGTH_BY_2 {
                if self.accum_spectra.e2[ch][k] > 0.0 {
                    new_erle[k] = self.accum_spectra.y2[ch][k] / self.accum_spectra.e2[ch][k];
                    is_erle_updated[k] = true;
                }
            }

            if self.use_onset_detection {
                for k in 1..FFT_LENGTH_BY_2 {
                    if is_erle_updated[k] && !self.accum_spectra.low_render_energy[ch][k] {
                        if self.coming_onset[ch][k] {
                            self.coming_onset[ch][k] = false;
                            if !self.use_min_erle_during_onsets {
                                let alpha = if new_erle[k] < self.erle_during_onsets[ch][k] {
                                    0.3
                                } else {
                                    0.15
                                };
                                self.erle_during_onsets[ch][k] = (self.erle_during_onsets[ch][k]
                                    + alpha * (new_erle[k] - self.erle_during_onsets[ch][k]))
                                    .clamp(self.min_erle, self.max_erle[k]);
                            }
                        }
                        self.hold_counters[ch][k] = BLOCKS_FOR_ONSET_DETECTION;
                    }
                }
            }

            for k in 1..FFT_LENGTH_BY_2 {
                if is_erle_updated[k] {
                    let low_render_energy = self.accum_spectra.low_render_energy[ch][k];
                    update_erle_band(
                        &mut self.erle[ch][k],
                        new_erle[k],
                        low_render_energy,
                        self.min_erle,
                        self.max_erle[k],
                    );
                    if self.use_onset_detection {
                        update_erle_band(
                            &mut self.erle_onset_compensated[ch][k],
                            new_erle[k],
                            low_render_energy,
                            self.min_erle,
                            self.max_erle[k],
                        );
                    }

                    // Virtually unbounded ERLE.
                    const UNBOUNDED_ERLE_MAX: f32 = 100000.0;
                    update_erle_band(
                        &mut self.erle_unbounded[ch][k],
                        new_erle[k],
                        low_render_energy,
                        self.min_erle,
                        UNBOUNDED_ERLE_MAX,
                    );
                }
            }
        }
    }

    fn decrease_erle_per_band_for_low_render_signals(&mut self) {
        let num_capture_channels = self.accum_spectra.y2.len();
        for ch in 0..num_capture_channels {
            for k in 1..FFT_LENGTH_BY_2 {
                self.hold_counters[ch][k] -= 1;
                if self.hold_counters[ch][k] <= (BLOCKS_FOR_ONSET_DETECTION - BLOCKS_TO_HOLD_ERLE) {
                    if self.erle_onset_compensated[ch][k] > self.erle_during_onsets[ch][k] {
                        self.erle_onset_compensated[ch][k] = self.erle_during_onsets[ch][k]
                            .max(0.97 * self.erle_onset_compensated[ch][k]);
                        debug_assert!(self.min_erle <= self.erle_onset_compensated[ch][k]);
                    }
                    if self.hold_counters[ch][k] <= 0 {
                        self.coming_onset[ch][k] = true;
                        self.hold_counters[ch][k] = 0;
                    }
                }
            }
        }
    }

    fn reset_accumulated_spectra(&mut self) {
        for ch in 0..self.erle_during_onsets.len() {
            self.accum_spectra.y2[ch].fill(0.0);
            self.accum_spectra.e2[ch].fill(0.0);
            self.accum_spectra.num_points[ch] = 0;
            self.accum_spectra.low_render_energy[ch].fill(false);
        }
    }

    fn update_accumulated_spectra(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        let num_capture_channels = y2.len();
        for ch in 0..num_capture_channels {
            if !converged_filters[ch] {
                continue;
            }

            if self.accum_spectra.num_points[ch] == POINTS_TO_ACCUMULATE {
                self.accum_spectra.num_points[ch] = 0;
                self.accum_spectra.y2[ch].fill(0.0);
                self.accum_spectra.e2[ch].fill(0.0);
                self.accum_spectra.low_render_energy[ch].fill(false);
            }

            for (((ay2_k, &y2_k), (ae2_k, &e2_k)), (lre_k, &x2_k)) in self.accum_spectra.y2[ch]
                .iter_mut()
                .zip(y2[ch].iter())
                .zip(self.accum_spectra.e2[ch].iter_mut().zip(e2[ch].iter()))
                .zip(
                    self.accum_spectra.low_render_energy[ch]
                        .iter_mut()
                        .zip(x2.iter()),
                )
            {
                *ay2_k += y2_k;
                *ae2_k += e2_k;
                *lre_k = *lre_k || x2_k < X2_BAND_ENERGY_THRESHOLD;
            }

            self.accum_spectra.num_points[ch] += 1;
        }
    }
}

fn update_erle_band(
    erle: &mut f32,
    new_erle: f32,
    low_render_energy: bool,
    min_erle: f32,
    max_erle: f32,
) {
    let alpha = if new_erle < *erle {
        if low_render_energy { 0.0 } else { 0.1 }
    } else {
        0.05
    };
    *erle = (*erle + alpha * (new_erle - *erle)).clamp(min_erle, max_erle);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(onset: bool) -> EchoCanceller3Config {
        let mut cfg = EchoCanceller3Config::default();
        cfg.erle.onset_detection = onset;
        cfg
    }

    #[test]
    fn erle_starts_at_min() {
        let config = make_config(true);
        let est = SubbandErleEstimator::new(&config, 1);
        let erle = est.erle(false);
        for &v in erle[0].iter() {
            assert!((v - config.erle.min).abs() < 0.001);
        }
    }

    #[test]
    fn erle_increases_with_echo_signal() {
        let config = make_config(false);
        let mut est = SubbandErleEstimator::new(&config, 1);

        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let erle_target = 10.0f32;
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for ((&x2_k, y2_k), e2_k) in x2.iter().zip(y2[0].iter_mut()).zip(e2[0].iter_mut()) {
            *y2_k = x2_k * 9.0;
            *e2_k = *y2_k / erle_target;
        }
        let converged = vec![true];

        for _ in 0..1000 {
            est.update(&x2, &y2, &e2, &converged);
        }

        let erle = est.erle(false);
        for (k, &erle_k) in erle[0][1..FFT_LENGTH_BY_2].iter().enumerate() {
            let k = k + 1;
            assert!(
                erle_k > config.erle.min + 0.1,
                "ERLE at bin {k} = {} should be above {}",
                erle_k,
                config.erle.min
            );
        }
    }

    #[test]
    fn erle_bounded_by_max() {
        let config = make_config(false);
        let mut est = SubbandErleEstimator::new(&config, 1);

        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for ((&x2_k, y2_k), e2_k) in x2.iter().zip(y2[0].iter_mut()).zip(e2[0].iter_mut()) {
            *y2_k = x2_k * 100.0;
            *e2_k = 1.0;
        }
        let converged = vec![true];

        for _ in 0..2000 {
            est.update(&x2, &y2, &e2, &converged);
        }

        let erle = est.erle(false);
        let max_erle = set_max_erle_bands(config.erle.max_l, config.erle.max_h);
        for (k, (&erle_k, &max_k)) in erle[0].iter().zip(max_erle.iter()).enumerate() {
            assert!(
                erle_k <= max_k + 0.001,
                "ERLE at bin {k} = {} exceeds max {}",
                erle_k,
                max_k
            );
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let config = make_config(true);
        let mut est = SubbandErleEstimator::new(&config, 1);

        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for ((&x2_k, y2_k), e2_k) in x2.iter().zip(y2[0].iter_mut()).zip(e2[0].iter_mut()) {
            *y2_k = x2_k * 9.0;
            *e2_k = *y2_k / 10.0;
        }
        let converged = vec![true];

        for _ in 0..100 {
            est.update(&x2, &y2, &e2, &converged);
        }

        est.reset();
        let erle = est.erle(false);
        for &v in erle[0].iter() {
            assert!((v - config.erle.min).abs() < 0.001);
        }
    }
}
