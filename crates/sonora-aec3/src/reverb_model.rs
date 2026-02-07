//! Exponential reverberant model applied over power spectra.
//!
//! Ported from `modules/audio_processing/aec3/reverb_model.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;

/// Describes an exponential reverberant model that can be applied over power
/// spectra.
pub(crate) struct ReverbModel {
    reverb: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl ReverbModel {
    pub(crate) fn new() -> Self {
        Self {
            reverb: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    /// Resets the state.
    pub(crate) fn reset(&mut self) {
        self.reverb.fill(0.0);
    }

    /// Returns the reverb power spectrum.
    pub(crate) fn reverb(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.reverb
    }

    /// Updates the reverb estimate with uniform frequency scaling.
    ///
    /// Before applying the exponential reverberant model, the input power
    /// spectrum is pre-scaled by a single scalar `power_spectrum_scaling`.
    pub(crate) fn update_reverb_no_freq_shaping(
        &mut self,
        power_spectrum: &[f32],
        power_spectrum_scaling: f32,
        reverb_decay: f32,
    ) {
        if reverb_decay > 0.0 {
            for (k, rev) in self
                .reverb
                .iter_mut()
                .enumerate()
                .take(power_spectrum.len())
            {
                *rev = (*rev + power_spectrum[k] * power_spectrum_scaling) * reverb_decay;
            }
        }
    }

    /// Updates the reverb estimate with per-frequency scaling.
    ///
    /// A different scaling is applied per frequency bin.
    pub(crate) fn update_reverb(
        &mut self,
        power_spectrum: &[f32],
        power_spectrum_scaling: &[f32],
        reverb_decay: f32,
    ) {
        if reverb_decay > 0.0 {
            for (k, rev) in self
                .reverb
                .iter_mut()
                .enumerate()
                .take(power_spectrum.len())
            {
                *rev = (*rev + power_spectrum[k] * power_spectrum_scaling[k]) * reverb_decay;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_reverb_is_zero() {
        let model = ReverbModel::new();
        for &v in model.reverb() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut model = ReverbModel::new();
        let spectrum = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        model.update_reverb_no_freq_shaping(&spectrum, 1.0, 0.9);
        model.reset();
        for &v in model.reverb() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn no_update_when_decay_is_zero() {
        let mut model = ReverbModel::new();
        let spectrum = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        model.update_reverb_no_freq_shaping(&spectrum, 1.0, 0.0);
        for &v in model.reverb() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn reverb_accumulates_with_decay() {
        let mut model = ReverbModel::new();
        let spectrum = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let decay = 0.5;
        // First update: reverb = (0 + 1*1) * 0.5 = 0.5
        model.update_reverb_no_freq_shaping(&spectrum, 1.0, decay);
        assert!((model.reverb()[0] - 0.5).abs() < 1e-6);
        // Second update: reverb = (0.5 + 1*1) * 0.5 = 0.75
        model.update_reverb_no_freq_shaping(&spectrum, 1.0, decay);
        assert!((model.reverb()[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn per_frequency_scaling_applied() {
        let mut model = ReverbModel::new();
        let spectrum = [2.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut scaling = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        scaling[0] = 1.0;
        scaling[1] = 0.5;
        model.update_reverb(&spectrum, &scaling, 1.0);
        assert!((model.reverb()[0] - 2.0).abs() < 1e-6);
        assert!((model.reverb()[1] - 1.0).abs() < 1e-6);
        assert_eq!(model.reverb()[2], 0.0);
    }
}
