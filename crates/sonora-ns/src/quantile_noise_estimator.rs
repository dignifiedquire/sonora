//! Quantile-based noise spectrum estimation.
//!
//! Tracks noise level using a quantile estimator with multiple simultaneous
//! estimates at different update rates, providing robust noise floor tracking
//! even during speech.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/quantile_noise_estimator.cc`

use crate::config::{FFT_SIZE_BY_2_PLUS_1, LONG_STARTUP_PHASE_BLOCKS};
use crate::fast_math::{exp_approximation_batch, log_approximation_batch};

/// Number of simultaneous quantile estimates.
const SIMULT: usize = 3;

/// Quantile-based noise spectrum estimator.
///
/// Maintains [`SIMULT`] simultaneous quantile trackers at staggered update
/// intervals. Each tracker estimates the 25th percentile of the log-spectrum
/// distribution. When a tracker's counter expires, its estimate is promoted
/// to the output noise spectrum.
#[derive(Debug)]
pub struct QuantileNoiseEstimator {
    /// Density estimates, shape `[SIMULT][FFT_SIZE_BY_2_PLUS_1]` flattened.
    density: [f32; SIMULT * FFT_SIZE_BY_2_PLUS_1],
    /// Log-domain quantile estimates, shape `[SIMULT][FFT_SIZE_BY_2_PLUS_1]` flattened.
    log_quantile: [f32; SIMULT * FFT_SIZE_BY_2_PLUS_1],
    /// Current noise spectrum estimate (linear domain).
    quantile: [f32; FFT_SIZE_BY_2_PLUS_1],
    /// Per-tracker frame counters.
    counter: [i32; SIMULT],
    /// Total number of updates performed.
    num_updates: i32,
}

impl Default for QuantileNoiseEstimator {
    fn default() -> Self {
        let one_by_simult = 1.0 / SIMULT as f32;
        let mut counter = [0i32; SIMULT];
        for (i, c) in counter.iter_mut().enumerate() {
            *c = (LONG_STARTUP_PHASE_BLOCKS as f32 * (i as f32 + 1.0) * one_by_simult).floor()
                as i32;
        }

        Self {
            density: [0.3; SIMULT * FFT_SIZE_BY_2_PLUS_1],
            log_quantile: [8.0; SIMULT * FFT_SIZE_BY_2_PLUS_1],
            quantile: [0.0; FFT_SIZE_BY_2_PLUS_1],
            counter,
            num_updates: 1,
        }
    }
}

impl QuantileNoiseEstimator {
    /// Estimate the noise spectrum from the current signal spectrum.
    ///
    /// Updates the internal quantile trackers and writes the noise estimate
    /// into `noise_spectrum`.
    pub fn estimate(
        &mut self,
        signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
        noise_spectrum: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    ) {
        let mut log_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        log_approximation_batch(signal_spectrum, &mut log_spectrum);

        let mut quantile_index_to_return: i32 = -1;

        // Loop over simultaneous estimates.
        for s in 0..SIMULT {
            let k = s * FFT_SIZE_BY_2_PLUS_1;
            let one_by_counter_plus_1 = 1.0 / (self.counter[s] as f32 + 1.0);

            for (i, &log_spec_i) in log_spectrum.iter().enumerate() {
                let j = k + i;

                // Update log quantile estimate.
                let delta = if self.density[j] > 1.0 {
                    40.0 / self.density[j]
                } else {
                    40.0
                };

                let multiplier = delta * one_by_counter_plus_1;
                if log_spec_i > self.log_quantile[j] {
                    self.log_quantile[j] += 0.25 * multiplier;
                } else {
                    self.log_quantile[j] -= 0.75 * multiplier;
                }

                // Update density estimate.
                const WIDTH: f32 = 0.01;
                const ONE_BY_WIDTH_PLUS_2: f32 = 1.0 / (2.0 * WIDTH);
                if (log_spec_i - self.log_quantile[j]).abs() < WIDTH {
                    self.density[j] = (self.counter[s] as f32 * self.density[j]
                        + ONE_BY_WIDTH_PLUS_2)
                        * one_by_counter_plus_1;
                }
            }

            if self.counter[s] >= LONG_STARTUP_PHASE_BLOCKS {
                self.counter[s] = 0;
                if self.num_updates >= LONG_STARTUP_PHASE_BLOCKS {
                    quantile_index_to_return = k as i32;
                }
            }

            self.counter[s] += 1;
        }

        // Sequentially update the noise during startup.
        if self.num_updates < LONG_STARTUP_PHASE_BLOCKS {
            // Use the last "s" to get noise during startup that differs from zero.
            quantile_index_to_return = (FFT_SIZE_BY_2_PLUS_1 * (SIMULT - 1)) as i32;
            self.num_updates += 1;
        }

        if quantile_index_to_return >= 0 {
            let start = quantile_index_to_return as usize;
            exp_approximation_batch(
                &self.log_quantile[start..start + FFT_SIZE_BY_2_PLUS_1],
                &mut self.quantile,
            );
        }

        noise_spectrum.copy_from_slice(&self.quantile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let qne = QuantileNoiseEstimator::default();
        assert_eq!(qne.num_updates, 1);
        assert_eq!(qne.quantile, [0.0; FFT_SIZE_BY_2_PLUS_1]);
        // Counters should be staggered fractions of LONG_STARTUP_PHASE_BLOCKS
        assert_eq!(qne.counter[0], 66); // floor(200 * 1/3)
        assert_eq!(qne.counter[1], 133); // floor(200 * 2/3)
        assert_eq!(qne.counter[2], 200); // floor(200 * 3/3)
    }

    #[test]
    fn estimate_produces_nonzero_after_first_call() {
        let mut qne = QuantileNoiseEstimator::default();
        let signal = [1.0f32; FFT_SIZE_BY_2_PLUS_1];
        let mut noise = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        qne.estimate(&signal, &mut noise);
        // After first estimate, noise should be nonzero (startup path)
        assert!(noise.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn estimate_converges_on_constant_signal() {
        let mut qne = QuantileNoiseEstimator::default();
        let signal = [10.0f32; FFT_SIZE_BY_2_PLUS_1];
        let mut noise = [0.0f32; FFT_SIZE_BY_2_PLUS_1];

        // Run through the full startup phase and beyond.
        for _ in 0..300 {
            qne.estimate(&signal, &mut noise);
        }

        // Noise estimate should converge toward the signal level.
        for &n in &noise {
            assert!(
                (n - 10.0).abs() < 5.0,
                "noise {n} should be close to signal level 10.0"
            );
        }
    }

    #[test]
    fn estimate_tracks_noise_floor() {
        let mut qne = QuantileNoiseEstimator::default();
        let mut noise = [0.0f32; FFT_SIZE_BY_2_PLUS_1];

        // Feed alternating low and high levels (simulating speech + noise).
        // The quantile estimator (25th percentile) should track the lower level.
        for frame in 0..400 {
            let level = if frame % 4 == 0 { 100.0 } else { 1.0 };
            let signal = [level; FFT_SIZE_BY_2_PLUS_1];
            qne.estimate(&signal, &mut noise);
        }

        // Noise estimate should be closer to the low level than the high level.
        let avg_noise: f32 = noise.iter().sum::<f32>() / noise.len() as f32;
        assert!(
            avg_noise < 50.0,
            "avg noise {avg_noise} should track the noise floor, not the peaks"
        );
    }
}
