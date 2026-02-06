//! Stationarity estimation for the render signal spectrum.
//!
//! Ported from `modules/audio_processing/aec3/stationarity_estimator.h/cc`.

use crate::common::{FFT_LENGTH_BY_2_PLUS_1, NUM_BLOCKS_PER_SECOND};
use crate::spectrum_buffer::SpectrumBuffer;

const MIN_NOISE_POWER: f32 = 10.0;
const HANGOVER_BLOCKS: i32 = (NUM_BLOCKS_PER_SECOND / 20) as i32;
const N_BLOCKS_AVERAGE_INIT_PHASE: usize = 20;
const N_BLOCKS_INITIAL_PHASE: usize = NUM_BLOCKS_PER_SECOND * 2;
const WINDOW_LENGTH: usize = 13;

/// Noise power spectrum estimator.
struct NoiseSpectrum {
    noise_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1],
    block_counter: usize,
}

impl NoiseSpectrum {
    fn new() -> Self {
        let mut s = Self {
            noise_spectrum: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            block_counter: 0,
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.block_counter = 0;
        self.noise_spectrum.fill(MIN_NOISE_POWER);
    }

    fn spectrum(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.noise_spectrum
    }

    fn power(&self, band: usize) -> f32 {
        debug_assert!(band < self.noise_spectrum.len());
        self.noise_spectrum[band]
    }

    fn update(&mut self, spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]]) {
        let num_render_channels = spectrum.len();

        let avg_spectrum: [f32; FFT_LENGTH_BY_2_PLUS_1];
        if num_render_channels == 1 {
            avg_spectrum = spectrum[0];
        } else {
            let mut data = spectrum[0];
            let one_by_num_channels = 1.0 / num_render_channels as f32;
            for ch in 1..num_render_channels {
                for k in 1..FFT_LENGTH_BY_2_PLUS_1 {
                    data[k] += spectrum[ch][k];
                }
            }
            for k in 1..FFT_LENGTH_BY_2_PLUS_1 {
                data[k] *= one_by_num_channels;
            }
            avg_spectrum = data;
        }

        self.block_counter += 1;
        let alpha = self.get_alpha();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if self.block_counter <= N_BLOCKS_AVERAGE_INIT_PHASE {
                self.noise_spectrum[k] +=
                    (1.0 / N_BLOCKS_AVERAGE_INIT_PHASE as f32) * avg_spectrum[k];
            } else {
                self.noise_spectrum[k] =
                    self.update_band_by_smoothing(avg_spectrum[k], self.noise_spectrum[k], alpha);
            }
        }
    }

    fn get_alpha(&self) -> f32 {
        const ALPHA: f32 = 0.004;
        const ALPHA_INIT: f32 = 0.04;
        const TILT_ALPHA: f32 = (ALPHA_INIT - ALPHA) / N_BLOCKS_INITIAL_PHASE as f32;

        if self.block_counter > N_BLOCKS_INITIAL_PHASE + N_BLOCKS_AVERAGE_INIT_PHASE {
            ALPHA
        } else {
            // During the initial averaging phase (block_counter <=
            // N_BLOCKS_AVERAGE_INIT_PHASE), alpha is not used. Guard
            // against underflow.
            let elapsed = self
                .block_counter
                .saturating_sub(N_BLOCKS_AVERAGE_INIT_PHASE);
            ALPHA_INIT - TILT_ALPHA * elapsed as f32
        }
    }

    fn update_band_by_smoothing(&self, power_band: f32, power_band_noise: f32, alpha: f32) -> f32 {
        let mut power_band_noise_updated = power_band_noise;
        if power_band_noise < power_band {
            debug_assert!(power_band > 0.0);
            let mut alpha_inc = alpha * (power_band_noise / power_band);
            if self.block_counter > N_BLOCKS_INITIAL_PHASE {
                if 10.0 * power_band_noise < power_band {
                    alpha_inc *= 0.1;
                }
            }
            power_band_noise_updated += alpha_inc * (power_band - power_band_noise);
        } else {
            power_band_noise_updated += alpha * (power_band - power_band_noise);
            power_band_noise_updated = power_band_noise_updated.max(MIN_NOISE_POWER);
        }
        power_band_noise_updated
    }
}

/// Estimates whether the render signal is stationary.
pub(crate) struct StationarityEstimator {
    noise: NoiseSpectrum,
    hangovers: [i32; FFT_LENGTH_BY_2_PLUS_1],
    stationarity_flags: [bool; FFT_LENGTH_BY_2_PLUS_1],
}

impl StationarityEstimator {
    pub(crate) fn new() -> Self {
        let mut s = Self {
            noise: NoiseSpectrum::new(),
            hangovers: [0; FFT_LENGTH_BY_2_PLUS_1],
            stationarity_flags: [false; FFT_LENGTH_BY_2_PLUS_1],
        };
        s.reset();
        s
    }

    /// Resets the estimator state.
    pub(crate) fn reset(&mut self) {
        self.noise.reset();
        self.hangovers.fill(0);
        self.stationarity_flags.fill(false);
    }

    /// Updates the noise estimator (useful before delay is known).
    pub(crate) fn update_noise_estimator(&mut self, spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]]) {
        self.noise.update(spectrum);
    }

    /// Updates the stationarity flags.
    pub(crate) fn update_stationarity_flags(
        &mut self,
        spectrum_buffer: &SpectrumBuffer,
        render_reverb_contribution_spectrum: &[f32],
        idx_current: usize,
        num_lookahead: usize,
    ) {
        let mut indexes = [0usize; WINDOW_LENGTH];
        let num_lookahead_bounded = num_lookahead.min(WINDOW_LENGTH - 1);

        let idx = if num_lookahead_bounded < WINDOW_LENGTH - 1 {
            let num_lookback = (WINDOW_LENGTH - 1) - num_lookahead_bounded;
            spectrum_buffer
                .index
                .offset_index(idx_current, num_lookback as i32)
        } else {
            idx_current
        };

        indexes[0] = idx;
        for k in 1..WINDOW_LENGTH {
            indexes[k] = spectrum_buffer.index.dec_index(indexes[k - 1]);
        }

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            self.stationarity_flags[k] = self.estimate_band_stationarity(
                spectrum_buffer,
                render_reverb_contribution_spectrum,
                &indexes,
                k,
            );
        }
        self.update_hangover();
        self.smooth_stationary_per_freq();
    }

    /// Returns true if the given band is stationary.
    pub(crate) fn is_band_stationary(&self, band: usize) -> bool {
        self.stationarity_flags[band] && self.hangovers[band] == 0
    }

    /// Returns true if the current block is estimated as stationary.
    pub(crate) fn is_block_stationary(&self) -> bool {
        let mut acum = 0.0f32;
        for band in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if self.is_band_stationary(band) {
                acum += 1.0;
            }
        }
        acum * (1.0 / FFT_LENGTH_BY_2_PLUS_1 as f32) > 0.75
    }

    fn estimate_band_stationarity(
        &self,
        spectrum_buffer: &SpectrumBuffer,
        average_reverb: &[f32],
        indexes: &[usize; WINDOW_LENGTH],
        band: usize,
    ) -> bool {
        const THR_STATIONARITY: f32 = 10.0;
        let mut acum_power = 0.0f32;
        let num_render_channels = spectrum_buffer.buffer[0].len();
        let one_by_num_channels = 1.0 / num_render_channels as f32;
        for &idx in indexes {
            for ch in 0..num_render_channels {
                acum_power += spectrum_buffer.buffer[idx][ch][band] * one_by_num_channels;
            }
        }
        acum_power += average_reverb[band];
        let noise = WINDOW_LENGTH as f32 * self.noise.power(band);
        debug_assert!(noise > 0.0);
        acum_power < THR_STATIONARITY * noise
    }

    fn are_all_bands_stationary(&self) -> bool {
        self.stationarity_flags.iter().all(|&b| b)
    }

    fn update_hangover(&mut self) {
        let reduce_hangover = self.are_all_bands_stationary();
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            if !self.stationarity_flags[k] {
                self.hangovers[k] = HANGOVER_BLOCKS;
            } else if reduce_hangover {
                self.hangovers[k] = (self.hangovers[k] - 1).max(0);
            }
        }
    }

    fn smooth_stationary_per_freq(&mut self) {
        let mut smoothed = [false; FFT_LENGTH_BY_2_PLUS_1];
        for k in 1..FFT_LENGTH_BY_2_PLUS_1 - 1 {
            smoothed[k] = self.stationarity_flags[k - 1]
                && self.stationarity_flags[k]
                && self.stationarity_flags[k + 1];
        }
        smoothed[0] = smoothed[1];
        smoothed[FFT_LENGTH_BY_2_PLUS_1 - 1] = smoothed[FFT_LENGTH_BY_2_PLUS_1 - 2];
        self.stationarity_flags = smoothed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_not_stationary() {
        let est = StationarityEstimator::new();
        assert!(!est.is_block_stationary());
    }

    #[test]
    fn noise_spectrum_initializes_to_min_power() {
        let ns = NoiseSpectrum::new();
        for &v in ns.spectrum() {
            assert_eq!(v, MIN_NOISE_POWER);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut est = StationarityEstimator::new();
        est.hangovers.fill(5);
        est.stationarity_flags.fill(true);
        est.reset();
        for &h in &est.hangovers {
            assert_eq!(h, 0);
        }
        for &f in &est.stationarity_flags {
            assert!(!f);
        }
    }

    #[test]
    fn noise_update_increases_from_initial() {
        let mut ns = NoiseSpectrum::new();
        let spectrum = [[100.0f32; FFT_LENGTH_BY_2_PLUS_1]];
        // During init phase, noise accumulates.
        for _ in 0..N_BLOCKS_AVERAGE_INIT_PHASE {
            ns.update(&spectrum);
        }
        // Noise should have increased from the initial MIN_NOISE_POWER.
        assert!(ns.power(1) > MIN_NOISE_POWER);
    }
}
