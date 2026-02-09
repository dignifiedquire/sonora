//! Spectral feature extraction for the RNN VAD.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/spectral_features.cc`.

use super::common::{
    CEPSTRAL_COEFFS_HISTORY_SIZE, FRAME_SIZE_20MS_24K_HZ, NUM_BANDS, NUM_HIGHER_BANDS,
    NUM_LOWER_BANDS,
};
use super::ring_buffer::RingBuffer;
use super::spectral_features_internal::{
    OPUS_BANDS_24K_HZ, SpectralCorrelator, compute_dct, compute_dct_table,
    compute_smoothed_log_magnitude_spectrum,
};
use super::symmetric_matrix_buffer::SymmetricMatrixBuffer;
use sonora_fft::pffft::{FftType, Pffft, PffftBuffer};
use std::f32::consts::FRAC_PI_2;

const SILENCE_THRESHOLD: f32 = 0.04;

/// Output of spectral feature extraction for a non-silent frame.
pub(crate) struct SpectralFeaturesOutput {
    pub higher_bands_cepstrum: [f32; NUM_HIGHER_BANDS],
    pub average: [f32; NUM_LOWER_BANDS],
    pub first_derivative: [f32; NUM_LOWER_BANDS],
    pub second_derivative: [f32; NUM_LOWER_BANDS],
    pub bands_cross_correlation: [f32; NUM_LOWER_BANDS],
    pub variability: f32,
}

/// Spectral feature extractor for 20 ms frames at 24 kHz.
#[derive(Debug)]
pub(crate) struct SpectralFeaturesExtractor {
    half_window: Vec<f32>,
    fft: Pffft,
    fft_buffer: PffftBuffer,
    reference_frame_fft: PffftBuffer,
    lagged_frame_fft: PffftBuffer,
    spectral_correlator: SpectralCorrelator,
    reference_frame_bands_energy: [f32; OPUS_BANDS_24K_HZ],
    lagged_frame_bands_energy: [f32; OPUS_BANDS_24K_HZ],
    bands_cross_corr: [f32; OPUS_BANDS_24K_HZ],
    dct_table: [f32; NUM_BANDS * NUM_BANDS],
    cepstral_coeffs_ring_buf: RingBuffer<NUM_BANDS, CEPSTRAL_COEFFS_HISTORY_SIZE>,
    cepstral_diffs_buf: SymmetricMatrixBuffer<CEPSTRAL_COEFFS_HISTORY_SIZE>,
}

impl Default for SpectralFeaturesExtractor {
    fn default() -> Self {
        let scaling = 1.0 / FRAME_SIZE_20MS_24K_HZ as f32;
        let half_window = compute_scaled_half_vorbis_window(scaling);
        let fft = Pffft::new(FRAME_SIZE_20MS_24K_HZ, FftType::Real);
        let fft_buffer = fft.create_buffer();
        let reference_frame_fft = fft.create_buffer();
        let lagged_frame_fft = fft.create_buffer();

        Self {
            half_window,
            fft,
            fft_buffer,
            reference_frame_fft,
            lagged_frame_fft,
            spectral_correlator: SpectralCorrelator::default(),
            reference_frame_bands_energy: [0.0; OPUS_BANDS_24K_HZ],
            lagged_frame_bands_energy: [0.0; OPUS_BANDS_24K_HZ],
            bands_cross_corr: [0.0; OPUS_BANDS_24K_HZ],
            dct_table: compute_dct_table(),
            cepstral_coeffs_ring_buf: RingBuffer::default(),
            cepstral_diffs_buf: SymmetricMatrixBuffer::default(),
        }
    }
}

impl SpectralFeaturesExtractor {
    /// Resets internal state.
    #[cfg(test)]
    pub(crate) fn reset(&mut self) {
        self.cepstral_coeffs_ring_buf.reset();
        self.cepstral_diffs_buf.reset();
    }

    /// Analyzes a pair of reference and lagged frames, detects silence and
    /// computes features.
    ///
    /// Returns `None` if silence is detected, otherwise returns the computed
    /// spectral features.
    pub(crate) fn check_silence_compute_features(
        &mut self,
        reference_frame: &[f32],
        lagged_frame: &[f32],
    ) -> Option<SpectralFeaturesOutput> {
        debug_assert_eq!(reference_frame.len(), FRAME_SIZE_20MS_24K_HZ);
        debug_assert_eq!(lagged_frame.len(), FRAME_SIZE_20MS_24K_HZ);

        // Compute the Opus band energies for the reference frame.
        self.compute_windowed_forward_fft(reference_frame, true);
        self.spectral_correlator.compute_auto_correlation(
            self.reference_frame_fft.as_slice(),
            &mut self.reference_frame_bands_energy,
        );

        // Check if the reference frame has silence.
        let tot_energy: f32 = self.reference_frame_bands_energy.iter().sum();
        if tot_energy < SILENCE_THRESHOLD {
            return None;
        }

        // Compute the Opus band energies for the lagged frame.
        self.compute_windowed_forward_fft(lagged_frame, false);
        self.spectral_correlator.compute_auto_correlation(
            self.lagged_frame_fft.as_slice(),
            &mut self.lagged_frame_bands_energy,
        );

        // Log of the band energies for the reference frame.
        let mut log_bands_energy = [0.0_f32; NUM_BANDS];
        compute_smoothed_log_magnitude_spectrum(
            &self.reference_frame_bands_energy,
            &mut log_bands_energy,
        );

        // Reference frame cepstrum.
        let mut cepstrum = [0.0_f32; NUM_BANDS];
        compute_dct(&log_bands_energy, &self.dct_table, &mut cepstrum);
        // Ad-hoc correction terms for the first two cepstral coefficients.
        cepstrum[0] -= 12.0;
        cepstrum[1] -= 4.0;

        // Update the ring buffer and the cepstral difference stats.
        self.cepstral_coeffs_ring_buf.push(&cepstrum);
        self.update_cepstral_difference_stats(&cepstrum);

        // Compute remaining features.
        let mut average = [0.0_f32; NUM_LOWER_BANDS];
        let mut first_derivative = [0.0_f32; NUM_LOWER_BANDS];
        let mut second_derivative = [0.0_f32; NUM_LOWER_BANDS];
        self.compute_avg_and_derivatives(
            &mut average,
            &mut first_derivative,
            &mut second_derivative,
        );

        let mut bands_cross_correlation = [0.0_f32; NUM_LOWER_BANDS];
        self.compute_normalized_cepstral_correlation(&mut bands_cross_correlation);

        let mut higher_bands_cepstrum = [0.0_f32; NUM_HIGHER_BANDS];
        higher_bands_cepstrum.copy_from_slice(&cepstrum[NUM_LOWER_BANDS..]);

        Some(SpectralFeaturesOutput {
            higher_bands_cepstrum,
            average,
            first_derivative,
            second_derivative,
            bands_cross_correlation,
            variability: self.compute_variability(),
        })
    }

    /// Applies windowing and computes forward FFT.
    fn compute_windowed_forward_fft(&mut self, frame: &[f32], is_reference: bool) {
        debug_assert_eq!(frame.len(), FRAME_SIZE_20MS_24K_HZ);
        let half_size = FRAME_SIZE_20MS_24K_HZ / 2;

        let buf = self.fft_buffer.as_mut_slice();
        for i in 0..half_size {
            let j = FRAME_SIZE_20MS_24K_HZ - 1 - i;
            buf[i] = frame[i] * self.half_window[i];
            buf[j] = frame[j] * self.half_window[i];
        }

        let output = if is_reference {
            &mut self.reference_frame_fft
        } else {
            &mut self.lagged_frame_fft
        };
        self.fft.forward(&self.fft_buffer, output, true);
        // Set the Nyquist frequency coefficient to zero.
        output.as_mut_slice()[1] = 0.0;
    }

    /// Computes average and first/second derivatives of cepstral coefficients.
    fn compute_avg_and_derivatives(
        &self,
        average: &mut [f32],
        first_derivative: &mut [f32],
        second_derivative: &mut [f32],
    ) {
        let curr = self.cepstral_coeffs_ring_buf.get_array_view(0);
        let prev1 = self.cepstral_coeffs_ring_buf.get_array_view(1);
        let prev2 = self.cepstral_coeffs_ring_buf.get_array_view(2);

        for i in 0..NUM_LOWER_BANDS {
            // Average, kernel: [1, 1, 1].
            average[i] = curr[i] + prev1[i] + prev2[i];
            // First derivative, kernel: [1, 0, -1].
            first_derivative[i] = curr[i] - prev2[i];
            // Second derivative, Laplacian kernel: [1, -2, 1].
            second_derivative[i] = curr[i] - 2.0 * prev1[i] + prev2[i];
        }
    }

    /// Computes normalized cepstral correlation between reference and lagged frames.
    fn compute_normalized_cepstral_correlation(&mut self, bands_cross_corr_out: &mut [f32]) {
        self.spectral_correlator.compute_cross_correlation(
            self.reference_frame_fft.as_slice(),
            self.lagged_frame_fft.as_slice(),
            &mut self.bands_cross_corr,
        );

        // Normalize.
        for i in 0..OPUS_BANDS_24K_HZ {
            self.bands_cross_corr[i] /= (0.001
                + self.reference_frame_bands_energy[i] * self.lagged_frame_bands_energy[i])
                .sqrt();
        }

        // Cepstrum.
        compute_dct(
            &self.bands_cross_corr,
            &self.dct_table,
            bands_cross_corr_out,
        );
        // Ad-hoc correction terms for the first two cepstral coefficients.
        bands_cross_corr_out[0] -= 1.3;
        bands_cross_corr_out[1] -= 0.9;
    }

    /// Computes cepstral variability score.
    fn compute_variability(&self) -> f32 {
        let mut variability = 0.0_f32;
        for delay1 in 0..CEPSTRAL_COEFFS_HISTORY_SIZE {
            let mut min_dist = f32::MAX;
            for delay2 in 0..CEPSTRAL_COEFFS_HISTORY_SIZE {
                if delay1 == delay2 {
                    continue;
                }
                min_dist = min_dist.min(self.cepstral_diffs_buf.get_value(delay1, delay2));
            }
            variability += min_dist;
        }
        // Normalize (based on training set stats).
        variability / CEPSTRAL_COEFFS_HISTORY_SIZE as f32 - 2.1
    }

    /// Updates cepstral difference stats in the symmetric matrix buffer.
    fn update_cepstral_difference_stats(&mut self, new_cepstral_coeffs: &[f32; NUM_BANDS]) {
        let mut distances = [0.0_f32; CEPSTRAL_COEFFS_HISTORY_SIZE - 1];
        for (i, dist) in distances.iter_mut().enumerate() {
            let delay = i + 1;
            let old_coeffs = self.cepstral_coeffs_ring_buf.get_array_view(delay);
            *dist = 0.0;
            for k in 0..NUM_BANDS {
                let c = new_cepstral_coeffs[k] - old_coeffs[k];
                *dist += c * c;
            }
        }
        self.cepstral_diffs_buf.push(&distances);
    }
}

/// Computes the first half of the Vorbis window with scaling.
fn compute_scaled_half_vorbis_window(scaling: f32) -> Vec<f32> {
    let half_size = FRAME_SIZE_20MS_24K_HZ / 2;
    let mut half_window = vec![0.0_f32; half_size];
    for (i, w) in half_window.iter_mut().enumerate() {
        let sin_arg = FRAC_PI_2 * (i as f32 + 0.5) / half_size as f32;
        *w = scaling * (FRAC_PI_2 * sin_arg.sin() * sin_arg.sin()).sin();
    }
    half_window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_detection_on_zero_frame() {
        let mut extractor = SpectralFeaturesExtractor::default();
        let frame = [0.0_f32; FRAME_SIZE_20MS_24K_HZ];

        let result = extractor.check_silence_compute_features(&frame, &frame);
        assert!(result.is_none(), "Zero frame should be detected as silence");
    }

    #[test]
    fn non_silence_frame_produces_features() {
        use std::f32::consts::TAU;
        let mut extractor = SpectralFeaturesExtractor::default();
        // Create a non-trivial frame with enough energy.
        let mut frame = [0.0_f32; FRAME_SIZE_20MS_24K_HZ];
        for (i, s) in frame.iter_mut().enumerate() {
            *s = (TAU * 440.0 * i as f32 / 24000.0).sin();
        }

        // Feed multiple frames to fill the cepstral history.
        let mut last_output = None;
        for _ in 0..CEPSTRAL_COEFFS_HISTORY_SIZE + 1 {
            let result = extractor.check_silence_compute_features(&frame, &frame);
            assert!(result.is_some(), "Non-zero frame should not be silence");
            last_output = result;
        }

        // After enough frames, features should have finite values.
        let features = last_output.unwrap();
        assert!(features.average.iter().all(|v| v.is_finite()));
        assert!(features.first_derivative.iter().all(|v| v.is_finite()));
        assert!(features.second_derivative.iter().all(|v| v.is_finite()));
        assert!(
            features
                .bands_cross_correlation
                .iter()
                .all(|v| v.is_finite())
        );
        assert!(features.variability.is_finite());
    }

    #[test]
    fn constant_input_zero_derivative() {
        let mut extractor = SpectralFeaturesExtractor::default();
        // Create a constant non-zero frame (amplitude must exceed silence threshold).
        let frame = [1.0_f32; FRAME_SIZE_20MS_24K_HZ];

        // Feed the same frame multiple times.
        let mut last_output = None;
        for _ in 0..CEPSTRAL_COEFFS_HISTORY_SIZE + 1 {
            let result = extractor.check_silence_compute_features(&frame, &frame);
            if result.is_some() {
                last_output = result;
            }
        }

        // With constant input, derivatives should be zero.
        let features = last_output.expect("expected non-silent output");
        for (i, &deriv) in features.first_derivative.iter().enumerate() {
            assert!(
                deriv.abs() < 1e-5,
                "first_derivative[{i}] = {deriv}, expected ~0"
            );
        }
        for (i, &deriv) in features.second_derivative.iter().enumerate() {
            assert!(
                deriv.abs() < 1e-5,
                "second_derivative[{i}] = {deriv}, expected ~0"
            );
        }
    }
}
