//! Echo Return Loss (ERL) estimator.
//!
//! Estimates the echo return loss based on the signal spectra using maximum
//! statistics. The ERL is estimated per frequency bin and also as a fullband
//! time-domain quantity.
//!
//! Ported from `modules/audio_processing/aec3/erl_estimator.h/cc`.

use crate::common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_MINUS_1, FFT_LENGTH_BY_2_PLUS_1};

const MIN_ERL: f32 = 0.01;
const MAX_ERL: f32 = 1000.0;

/// Estimates the echo return loss based on the signal spectra.
pub(crate) struct ErlEstimator {
    startup_phase_length_blocks: usize,
    erl: [f32; FFT_LENGTH_BY_2_PLUS_1],
    hold_counters: [i32; FFT_LENGTH_BY_2_MINUS_1],
    erl_time_domain: f32,
    hold_counter_time_domain: i32,
    blocks_since_reset: usize,
}

impl ErlEstimator {
    pub(crate) fn new(startup_phase_length_blocks: usize) -> Self {
        Self {
            startup_phase_length_blocks,
            erl: [MAX_ERL; FFT_LENGTH_BY_2_PLUS_1],
            hold_counters: [0; FFT_LENGTH_BY_2_MINUS_1],
            erl_time_domain: MAX_ERL,
            hold_counter_time_domain: 0,
            blocks_since_reset: 0,
        }
    }

    /// Resets the ERL estimation.
    pub(crate) fn reset(&mut self) {
        self.blocks_since_reset = 0;
    }

    /// Updates the ERL estimate.
    pub(crate) fn update(
        &mut self,
        converged_filters: &[bool],
        render_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        capture_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    ) {
        let num_capture_channels = converged_filters.len();
        debug_assert_eq!(capture_spectra.len(), num_capture_channels);

        // Corresponds to WGN of power -46 dBFS.
        const X2_MIN: f32 = 44015068.0;

        let first_converged = converged_filters.iter().position(|&c| c);
        let any_filter_converged = first_converged.is_some();

        self.blocks_since_reset += 1;
        if self.blocks_since_reset < self.startup_phase_length_blocks || !any_filter_converged {
            return;
        }

        // Use the maximum spectrum across capture and render channels.
        let mut max_capture_spectrum = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        if num_capture_channels == 1 {
            max_capture_spectrum.copy_from_slice(&capture_spectra[0]);
        } else {
            let first = first_converged.unwrap();
            max_capture_spectrum.copy_from_slice(&capture_spectra[first]);
            for ch in (first + 1)..num_capture_channels {
                if !converged_filters[ch] {
                    continue;
                }
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    max_capture_spectrum[k] = max_capture_spectrum[k].max(capture_spectra[ch][k]);
                }
            }
        }

        let num_render_channels = render_spectra.len();
        let mut max_render_spectrum = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        max_render_spectrum.copy_from_slice(&render_spectra[0]);
        for ch in 1..num_render_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                max_render_spectrum[k] = max_render_spectrum[k].max(render_spectra[ch][k]);
            }
        }

        let x2 = &max_render_spectrum;
        let y2 = &max_capture_spectrum;

        // Update the estimates in a maximum statistics manner.
        for k in 1..FFT_LENGTH_BY_2 {
            if x2[k] > X2_MIN {
                let new_erl = y2[k] / x2[k];
                if new_erl < self.erl[k] {
                    self.hold_counters[k - 1] = 1000;
                    self.erl[k] += 0.1 * (new_erl - self.erl[k]);
                    self.erl[k] = self.erl[k].max(MIN_ERL);
                }
            }
        }

        for counter in &mut self.hold_counters {
            *counter -= 1;
        }
        for k in 1..FFT_LENGTH_BY_2 {
            if self.hold_counters[k - 1] <= 0 {
                self.erl[k] = MAX_ERL.min(2.0 * self.erl[k]);
            }
        }

        self.erl[0] = self.erl[1];
        self.erl[FFT_LENGTH_BY_2] = self.erl[FFT_LENGTH_BY_2 - 1];

        // Compute ERL over all frequency bins.
        let x2_sum: f32 = x2.iter().sum();
        if x2_sum > X2_MIN * FFT_LENGTH_BY_2_PLUS_1 as f32 {
            let y2_sum: f32 = y2.iter().sum();
            let new_erl = y2_sum / x2_sum;
            if new_erl < self.erl_time_domain {
                self.hold_counter_time_domain = 1000;
                self.erl_time_domain += 0.1 * (new_erl - self.erl_time_domain);
                self.erl_time_domain = self.erl_time_domain.max(MIN_ERL);
            }
        }

        self.hold_counter_time_domain -= 1;
        if self.hold_counter_time_domain <= 0 {
            self.erl_time_domain = MAX_ERL.min(2.0 * self.erl_time_domain);
        }
    }

    /// Returns the most recent ERL estimate.
    pub(crate) fn erl(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.erl
    }

    /// Returns the time-domain ERL estimate.
    pub(crate) fn erl_time_domain(&self) -> f32 {
        self.erl_time_domain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_erl(erl: &[f32; FFT_LENGTH_BY_2_PLUS_1], erl_time_domain: f32, reference: f32) {
        for &v in erl.iter() {
            assert!(
                (v - reference).abs() < 0.001,
                "ERL bin {v} != reference {reference}"
            );
        }
        assert!(
            (erl_time_domain - reference).abs() < 0.001,
            "ERL time domain {erl_time_domain} != reference {reference}"
        );
    }

    #[test]
    fn estimates() {
        for &num_render_channels in &[1usize, 2, 8] {
            for &num_capture_channels in &[1usize, 2, 8] {
                let mut x2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_render_channels];
                let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
                let mut converged_filters = vec![false; num_capture_channels];
                let converged_idx = num_capture_channels - 1;
                converged_filters[converged_idx] = true;

                let mut estimator = ErlEstimator::new(0);

                // Verifies that the ERL estimate is properly reduced to lower values.
                for x2_ch in &mut x2 {
                    x2_ch.fill(500.0 * 1000.0 * 1000.0);
                }
                y2[converged_idx].fill(10.0 * x2[0][0]);
                for _ in 0..200 {
                    estimator.update(&converged_filters, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 10.0);

                // Verifies that the ERL is not immediately increased when the ERL in
                // the data increases.
                y2[converged_idx].fill(10000.0 * x2[0][0]);
                for _ in 0..998 {
                    estimator.update(&converged_filters, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 10.0);

                // Verifies that the rate of increase is 3 dB.
                estimator.update(&converged_filters, &x2, &y2);
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 20.0);

                // Verifies that the maximum ERL is achieved when there are no low ERL
                // estimates.
                for _ in 0..1000 {
                    estimator.update(&converged_filters, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 1000.0);

                // Verifies that the ERL estimate is not updated for low-level signals.
                for x2_ch in &mut x2 {
                    x2_ch.fill(1000.0 * 1000.0);
                }
                y2[converged_idx].fill(10.0 * x2[0][0]);
                for _ in 0..200 {
                    estimator.update(&converged_filters, &x2, &y2);
                }
                verify_erl(estimator.erl(), estimator.erl_time_domain(), 1000.0);
            }
        }
    }
}
