//! Internal helpers for spectral feature computation.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/spectral_features_internal.cc`.

use super::common::{FRAME_SIZE_20MS_24K_HZ, NUM_BANDS};
use std::f64::consts::PI;
use std::ptr;

/// Number of Opus bands at 24 kHz (last 3 Opus bands are beyond Nyquist,
/// but band #19 gets contributions from band #18 due to the symmetric
/// triangular filter with peak response at 12 kHz).
pub const OPUS_BANDS_24K_HZ: usize = 20;

/// Number of FFT frequency bins covered by each band in the Opus scale at
/// 24 kHz for 20 ms frames.
pub const OPUS_SCALE_NUM_BINS_24K_HZ_20MS: [usize; OPUS_BANDS_24K_HZ - 1] = [
    4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48,
];

/// Weights for each FFT coefficient for each Opus band (Nyquist excluded).
#[allow(clippy::excessive_precision, reason = "values from C++ source")]
const OPUS_BAND_WEIGHTS_24K_HZ_20MS: [f32; FRAME_SIZE_20MS_24K_HZ / 2] = [
    // Band 0
    0.0, 0.25, 0.5, 0.75, // Band 1
    0.0, 0.25, 0.5, 0.75, // Band 2
    0.0, 0.25, 0.5, 0.75, // Band 3
    0.0, 0.25, 0.5, 0.75, // Band 4
    0.0, 0.25, 0.5, 0.75, // Band 5
    0.0, 0.25, 0.5, 0.75, // Band 6
    0.0, 0.25, 0.5, 0.75, // Band 7
    0.0, 0.25, 0.5, 0.75, // Band 8
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, // Band 9
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, // Band 10
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, // Band 11
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, // Band 12
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
    0.8125, 0.875, 0.9375, // Band 13
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
    0.8125, 0.875, 0.9375, // Band 14
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75,
    0.8125, 0.875, 0.9375, // Band 15
    0.0, 0.0416667, 0.0833333, 0.125, 0.166667, 0.208333, 0.25, 0.291667, 0.333333, 0.375,
    0.416667, 0.458333, 0.5, 0.541667, 0.583333, 0.625, 0.666667, 0.708333, 0.75, 0.791667,
    0.833333, 0.875, 0.916667, 0.958333, // Band 16
    0.0, 0.0416667, 0.0833333, 0.125, 0.166667, 0.208333, 0.25, 0.291667, 0.333333, 0.375,
    0.416667, 0.458333, 0.5, 0.541667, 0.583333, 0.625, 0.666667, 0.708333, 0.75, 0.791667,
    0.833333, 0.875, 0.916667, 0.958333, // Band 17
    0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375,
    0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875,
    0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875, // Band 18
    0.0, 0.0208333, 0.0416667, 0.0625, 0.0833333, 0.104167, 0.125, 0.145833, 0.166667, 0.1875,
    0.208333, 0.229167, 0.25, 0.270833, 0.291667, 0.3125, 0.333333, 0.354167, 0.375, 0.395833,
    0.416667, 0.4375, 0.458333, 0.479167, 0.5, 0.520833, 0.541667, 0.5625, 0.583333, 0.604167,
    0.625, 0.645833, 0.666667, 0.6875, 0.708333, 0.729167, 0.75, 0.770833, 0.791667, 0.8125,
    0.833333, 0.854167, 0.875, 0.895833, 0.916667, 0.9375, 0.958333, 0.979167,
];

/// Computes band-wise spectral correlations using triangular filters in the
/// Opus scale.
#[derive(Debug)]
pub struct SpectralCorrelator {
    weights: Vec<f32>,
}

impl Default for SpectralCorrelator {
    fn default() -> Self {
        Self {
            weights: OPUS_BAND_WEIGHTS_24K_HZ_20MS.to_vec(),
        }
    }
}

impl SpectralCorrelator {
    /// Computes band-wise spectral auto-correlation.
    pub fn compute_auto_correlation(&self, x: &[f32], auto_corr: &mut [f32; OPUS_BANDS_24K_HZ]) {
        self.compute_cross_correlation(x, x, auto_corr);
    }

    /// Computes band-wise spectral cross-correlation.
    ///
    /// `x` and `y` must have size `FRAME_SIZE_20MS_24K_HZ` and be encoded as
    /// interleaved real-complex FFT coefficients where `x[1] = y[1] = 0`.
    pub fn compute_cross_correlation(
        &self,
        x: &[f32],
        y: &[f32],
        cross_corr: &mut [f32; OPUS_BANDS_24K_HZ],
    ) {
        debug_assert_eq!(x.len(), FRAME_SIZE_20MS_24K_HZ);
        debug_assert_eq!(y.len(), FRAME_SIZE_20MS_24K_HZ);
        debug_assert_eq!(x[1], 0.0, "The Nyquist coefficient must be zeroed.");
        debug_assert_eq!(y[1], 0.0, "The Nyquist coefficient must be zeroed.");

        let mut k = 0_usize; // Next Fourier coefficient index.
        cross_corr[0] = 0.0;
        for (i, &num_bins) in OPUS_SCALE_NUM_BINS_24K_HZ_20MS.iter().enumerate() {
            cross_corr[i + 1] = 0.0;
            for _ in 0..num_bins {
                let v = x[2 * k] * y[2 * k] + x[2 * k + 1] * y[2 * k + 1];
                let tmp = self.weights[k] * v;
                cross_corr[i] += v - tmp;
                cross_corr[i + 1] += tmp;
                k += 1;
            }
        }
        cross_corr[0] *= 2.0; // The first band only gets half contribution.
        debug_assert_eq!(k, FRAME_SIZE_20MS_24K_HZ / 2);
    }
}

/// Computes the smoothed log magnitude spectrum from band energies.
pub fn compute_smoothed_log_magnitude_spectrum(
    bands_energy: &[f32],
    log_bands_energy: &mut [f32; NUM_BANDS],
) {
    debug_assert!(bands_energy.len() <= NUM_BANDS);
    const ONE_BY_HUNDRED: f32 = 1e-2;
    const LOG_ONE_BY_HUNDRED: f32 = -2.0;

    let mut log_max = LOG_ONE_BY_HUNDRED;
    let mut follow = LOG_ONE_BY_HUNDRED;

    let smooth = |x: f32, log_max: &mut f32, follow: &mut f32| -> f32 {
        let x = x.max(*log_max - 7.0).max(*follow - 1.5);
        *log_max = log_max.max(x);
        *follow = follow.max(x).max(*follow - 1.5);
        x
    };

    // Smoothing over the bands for which the band energy is defined.
    for (i, &energy) in bands_energy.iter().enumerate() {
        log_bands_energy[i] = smooth((ONE_BY_HUNDRED + energy).log10(), &mut log_max, &mut follow);
    }
    // Smoothing over the remaining bands (zero energy).
    for lbe in log_bands_energy
        .iter_mut()
        .take(NUM_BANDS)
        .skip(bands_energy.len())
    {
        *lbe = smooth(LOG_ONE_BY_HUNDRED, &mut log_max, &mut follow);
    }
}

/// Creates a DCT table for arrays having size equal to `NUM_BANDS`.
pub fn compute_dct_table() -> [f32; NUM_BANDS * NUM_BANDS] {
    let mut dct_table = [0.0_f32; NUM_BANDS * NUM_BANDS];
    let k = (0.5_f64).sqrt();
    for i in 0..NUM_BANDS {
        for j in 0..NUM_BANDS {
            dct_table[i * NUM_BANDS + j] =
                ((i as f64 + 0.5) * j as f64 * PI / NUM_BANDS as f64).cos() as f32;
        }
        dct_table[i * NUM_BANDS] *= k as f32;
    }
    dct_table
}

/// Computes DCT for `input` given a pre-computed DCT table.
///
/// In-place computation is not allowed. `output` can be smaller than `input`
/// to compute only the first DCT coefficients.
pub fn compute_dct(input: &[f32], dct_table: &[f32; NUM_BANDS * NUM_BANDS], output: &mut [f32]) {
    // DCT scaling factor: sqrt(2 / NUM_BANDS).
    #[allow(clippy::excessive_precision, reason = "value from C++ source")]
    const DCT_SCALING_FACTOR: f32 = 0.301511345;

    debug_assert!(!ptr::eq(input.as_ptr(), output.as_ptr()));
    debug_assert!(input.len() <= NUM_BANDS);
    debug_assert!(!output.is_empty());
    debug_assert!(output.len() <= input.len());

    for (i, out) in output.iter_mut().enumerate() {
        *out = 0.0;
        for (j, &inp) in input.iter().enumerate() {
            *out += inp * dct_table[j * NUM_BANDS + i];
        }
        *out *= DCT_SCALING_FACTOR;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_scale_boundaries_sum_to_half_frame() {
        let total_bins: usize = OPUS_SCALE_NUM_BINS_24K_HZ_20MS.iter().sum();
        assert_eq!(total_bins, FRAME_SIZE_20MS_24K_HZ / 2);
    }

    #[test]
    fn spectral_correlator_valid_output() {
        let correlator = SpectralCorrelator::default();
        // Create a simple test signal: unit energy at each frequency bin.
        let mut x = vec![0.0_f32; FRAME_SIZE_20MS_24K_HZ];
        for i in 0..FRAME_SIZE_20MS_24K_HZ / 2 {
            x[2 * i] = 1.0; // Real part.
        }
        x[1] = 0.0; // Nyquist coefficient must be zero.

        let mut auto_corr = [0.0_f32; OPUS_BANDS_24K_HZ];
        correlator.compute_auto_correlation(&x, &mut auto_corr);

        // Auto-correlation of a signal with itself must be non-negative.
        for (i, &ac) in auto_corr.iter().enumerate() {
            assert!(ac >= 0.0, "auto_corr[{i}] = {ac} < 0");
        }
    }

    #[test]
    fn smoothed_log_magnitude_spectrum_basic() {
        let bands_energy = [1.0_f32; OPUS_BANDS_24K_HZ];
        let mut log_bands_energy = [0.0_f32; NUM_BANDS];
        compute_smoothed_log_magnitude_spectrum(&bands_energy, &mut log_bands_energy);

        // log10(1e-2 + 1.0) ≈ 0.00432
        for (i, &lbe) in log_bands_energy.iter().enumerate().take(OPUS_BANDS_24K_HZ) {
            assert!(
                lbe > -0.1 && lbe < 0.1,
                "log_bands_energy[{i}] = {lbe}, expected ~0"
            );
        }
    }

    #[test]
    fn dct_of_constant_input() {
        let dct_table = compute_dct_table();
        let input = [1.0_f32; NUM_BANDS];
        let mut output = [0.0_f32; NUM_BANDS];
        compute_dct(&input, &dct_table, &mut output);

        // DC coefficient should be non-zero, higher coefficients should be ~0.
        assert!(output[0].abs() > 0.1, "DC coefficient should be non-zero");
        for (i, &o) in output.iter().enumerate().skip(1) {
            assert!(
                o.abs() < 1e-5,
                "output[{i}] = {o}, expected ~0 for constant input"
            );
        }
    }
}
