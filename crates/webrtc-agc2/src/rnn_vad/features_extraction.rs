//! Feature extractor to feed the VAD RNN.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/features_extraction.cc`.

use super::common::{
    BUF_SIZE_24K_HZ, FRAME_SIZE_10MS_24K_HZ, FRAME_SIZE_20MS_24K_HZ, FeatureVector,
    MAX_PITCH_24K_HZ,
};
use super::lp_residual::{
    NUM_LPC_COEFFICIENTS, compute_and_post_process_lpc_coefficients, compute_lp_residual,
};
use super::pitch_search::PitchEstimator;
use super::sequence_buffer::SequenceBuffer;
use super::spectral_features::SpectralFeaturesExtractor;
use webrtc_simd::SimdBackend;

/// Feature extractor to feed the VAD RNN.
///
/// Processes 10 ms frames at 24 kHz and produces a 42-element feature vector.
#[derive(Debug)]
pub struct FeaturesExtractor {
    pitch_buf_24k_hz: SequenceBuffer<BUF_SIZE_24K_HZ, FRAME_SIZE_10MS_24K_HZ>,
    lp_residual: Vec<f32>,
    pitch_estimator: PitchEstimator,
    spectral_features_extractor: SpectralFeaturesExtractor,
    pitch_period_48k_hz: i32,
}

impl FeaturesExtractor {
    /// Creates a new feature extractor.
    pub fn new(backend: SimdBackend) -> Self {
        Self {
            pitch_buf_24k_hz: SequenceBuffer::default(),
            lp_residual: vec![0.0; BUF_SIZE_24K_HZ],
            pitch_estimator: PitchEstimator::new(backend),
            spectral_features_extractor: SpectralFeaturesExtractor::default(),
            pitch_period_48k_hz: 0,
        }
    }

    /// Resets internal state.
    pub fn reset(&mut self) {
        self.pitch_buf_24k_hz.reset();
        self.spectral_features_extractor.reset();
    }

    /// Analyzes the samples, computes the feature vector and returns `true` if
    /// silence is detected.
    ///
    /// When silence is detected, `feature_vector` is partially written and
    /// must not be used to feed the VAD RNN.
    pub fn check_silence_compute_features(
        &mut self,
        samples: &[f32],
        feature_vector: &mut FeatureVector,
    ) -> bool {
        debug_assert_eq!(samples.len(), FRAME_SIZE_10MS_24K_HZ);

        // Feed buffer with samples (HPF disabled, matching C++ default).
        let samples_arr: &[f32; FRAME_SIZE_10MS_24K_HZ] = samples.try_into().unwrap();
        self.pitch_buf_24k_hz.push(samples_arr);

        // Extract the LP residual.
        let pitch_buf_view = self.pitch_buf_24k_hz.get_buffer_view();
        let mut lpc_coeffs = [0.0_f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(pitch_buf_view, &mut lpc_coeffs);
        compute_lp_residual(&lpc_coeffs, pitch_buf_view, &mut self.lp_residual);

        // Estimate pitch on the LP-residual and write the normalized pitch
        // period (normalization based on training data stats).
        self.pitch_period_48k_hz = self.pitch_estimator.estimate(&self.lp_residual);
        feature_vector.pitch_period = 0.01 * (self.pitch_period_48k_hz - 300) as f32;

        // Extract lagged frame (according to the estimated pitch period).
        debug_assert!(self.pitch_period_48k_hz / 2 <= MAX_PITCH_24K_HZ as i32);
        let lag_offset = MAX_PITCH_24K_HZ - self.pitch_period_48k_hz as usize / 2;
        let lagged_frame = &pitch_buf_view[lag_offset..lag_offset + FRAME_SIZE_20MS_24K_HZ];

        // Reference frame is the most recent 20 ms.
        let reference_frame: &[f32; FRAME_SIZE_20MS_24K_HZ] =
            self.pitch_buf_24k_hz.get_most_recent_values_view();

        // Analyze reference and lagged frames, check silence, write features.
        self.spectral_features_extractor
            .check_silence_compute_features(
                reference_frame,
                lagged_frame,
                &mut feature_vector.higher_bands_cepstrum,
                &mut feature_vector.average,
                &mut feature_vector.first_derivative,
                &mut feature_vector.second_derivative,
                &mut feature_vector.bands_cross_correlation,
                &mut feature_vector.spectral_variability,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::super::common::SAMPLE_RATE_24K_HZ;
    use super::*;
    use std::f32::consts::TAU;

    /// Number of 10 ms frames required to fill the pitch buffer.
    const NUM_TEST_DATA_FRAMES: usize = BUF_SIZE_24K_HZ.div_ceil(FRAME_SIZE_10MS_24K_HZ);
    const NUM_TEST_DATA_SIZE: usize = NUM_TEST_DATA_FRAMES * FRAME_SIZE_10MS_24K_HZ;

    fn pitch_is_valid(pitch_hz: f32) -> bool {
        use super::super::common::INITIAL_MIN_PITCH_24K_HZ;
        let pitch_period = (SAMPLE_RATE_24K_HZ as f32 / pitch_hz) as usize;
        (INITIAL_MIN_PITCH_24K_HZ..=MAX_PITCH_24K_HZ).contains(&pitch_period)
    }

    fn create_pure_tone(amplitude: f32, freq_hz: f32, dst: &mut [f32]) {
        for (i, s) in dst.iter_mut().enumerate() {
            *s = amplitude * (TAU * freq_hz * i as f32 / SAMPLE_RATE_24K_HZ as f32).sin();
        }
    }

    /// Feeds `features_extractor` with `samples` splitting it in 10 ms frames.
    fn feed_test_data(
        features_extractor: &mut FeaturesExtractor,
        samples: &[f32],
        feature_vector: &mut FeatureVector,
    ) -> bool {
        let mut is_silence = true;
        let num_frames = samples.len() / FRAME_SIZE_10MS_24K_HZ;
        for i in 0..num_frames {
            let start = i * FRAME_SIZE_10MS_24K_HZ;
            let end = start + FRAME_SIZE_10MS_24K_HZ;
            is_silence = features_extractor
                .check_silence_compute_features(&samples[start..end], feature_vector);
        }
        is_silence
    }

    #[test]
    fn feature_extraction_low_high_pitch() {
        let amplitude = 1000.0_f32;
        let low_pitch_hz = 150.0_f32;
        let high_pitch_hz = 250.0_f32;
        assert!(pitch_is_valid(low_pitch_hz));
        assert!(pitch_is_valid(high_pitch_hz));

        let backend = webrtc_simd::detect_backend();
        let mut features_extractor = FeaturesExtractor::new(backend);
        let mut samples = vec![0.0_f32; NUM_TEST_DATA_SIZE];
        let mut feature_vector = FeatureVector::default();

        // Low frequency tone → high period.
        create_pure_tone(amplitude, low_pitch_hz, &mut samples);
        assert!(!feed_test_data(
            &mut features_extractor,
            &samples,
            &mut feature_vector
        ));
        let high_pitch_period = feature_vector.pitch_period;

        // High frequency tone → low period.
        features_extractor.reset();
        create_pure_tone(amplitude, high_pitch_hz, &mut samples);
        assert!(!feed_test_data(
            &mut features_extractor,
            &samples,
            &mut feature_vector
        ));
        let low_pitch_period = feature_vector.pitch_period;

        assert!(
            low_pitch_period < high_pitch_period,
            "Expected low_pitch_period ({low_pitch_period}) < high_pitch_period ({high_pitch_period})"
        );
    }
}
