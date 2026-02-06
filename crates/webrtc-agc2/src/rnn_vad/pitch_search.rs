//! Pitch estimator for the RNN VAD.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/pitch_search.cc`.

use super::auto_correlation::AutoCorrelationCalculator;
use super::common::{BUF_SIZE_24K_HZ, MAX_PITCH_48K_HZ, NUM_LAGS_12K_HZ, REFINE_NUM_LAGS_24K_HZ};
use super::pitch_search_internal::{
    PitchInfo, compute_extended_pitch_period_48k_hz, compute_pitch_period_12k_hz,
    compute_pitch_period_48k_hz, compute_sliding_frame_square_energies_24k_hz, decimate_2x,
};
use webrtc_simd::SimdBackend;

/// Pitch estimator.
#[derive(Debug)]
pub struct PitchEstimator {
    backend: SimdBackend,
    last_pitch_48k_hz: PitchInfo,
    auto_corr_calculator: AutoCorrelationCalculator,
    y_energy_24k_hz: Vec<f32>,
    pitch_buffer_12k_hz: Vec<f32>,
    auto_correlation_12k_hz: Vec<f32>,
}

impl PitchEstimator {
    /// Creates a new pitch estimator.
    pub fn new(backend: SimdBackend) -> Self {
        Self {
            backend,
            last_pitch_48k_hz: PitchInfo::default(),
            auto_corr_calculator: AutoCorrelationCalculator::default(),
            y_energy_24k_hz: vec![0.0; REFINE_NUM_LAGS_24K_HZ],
            pitch_buffer_12k_hz: vec![0.0; BUF_SIZE_24K_HZ / 2],
            auto_correlation_12k_hz: vec![0.0; NUM_LAGS_12K_HZ],
        }
    }

    /// Returns the estimated pitch period at 48 kHz.
    pub fn estimate(&mut self, pitch_buffer: &[f32]) -> i32 {
        debug_assert_eq!(pitch_buffer.len(), BUF_SIZE_24K_HZ);

        // Perform the initial pitch search at 12 kHz.
        decimate_2x(pitch_buffer, &mut self.pitch_buffer_12k_hz);
        self.auto_corr_calculator
            .compute_on_pitch_buffer(&self.pitch_buffer_12k_hz, &mut self.auto_correlation_12k_hz);
        let mut pitch_periods = compute_pitch_period_12k_hz(
            &self.pitch_buffer_12k_hz,
            &self.auto_correlation_12k_hz,
            self.backend,
        );
        // Adapt inverted lags from 12 to 24 kHz.
        pitch_periods.best *= 2;
        pitch_periods.second_best *= 2;

        // Refine from 12 kHz to 48 kHz.
        compute_sliding_frame_square_energies_24k_hz(
            pitch_buffer,
            &mut self.y_energy_24k_hz,
            self.backend,
        );

        let pitch_lag_48k_hz = compute_pitch_period_48k_hz(
            pitch_buffer,
            &self.y_energy_24k_hz,
            pitch_periods,
            self.backend,
        );

        self.last_pitch_48k_hz = compute_extended_pitch_period_48k_hz(
            pitch_buffer,
            &self.y_energy_24k_hz,
            MAX_PITCH_48K_HZ as i32 - pitch_lag_48k_hz,
            self.last_pitch_48k_hz,
            self.backend,
        );

        self.last_pitch_48k_hz.period
    }

    /// Returns the last pitch strength (for testing).
    pub fn last_pitch_strength(&self) -> f32 {
        self.last_pitch_48k_hz.strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    fn test_resources_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/resources/audio_processing/agc2/rnn_vad")
    }

    /// Reads chunks of float data from a binary file.
    struct ChunksReader {
        data: Vec<f32>,
        offset: usize,
        chunk_size: usize,
    }

    impl ChunksReader {
        fn open(path: &Path, chunk_size: usize) -> Self {
            let mut file = File::open(path)
                .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Self {
                data,
                offset: 0,
                chunk_size,
            }
        }

        fn num_chunks(&self) -> usize {
            self.data.len() / self.chunk_size
        }

        fn read_chunk(&mut self, buf: &mut [f32]) -> bool {
            if self.offset + buf.len() > self.data.len() {
                return false;
            }
            buf.copy_from_slice(&self.data[self.offset..self.offset + buf.len()]);
            self.offset += buf.len();
            true
        }

        fn read_value(&mut self) -> Option<f32> {
            if self.offset >= self.data.len() {
                return None;
            }
            let val = self.data[self.offset];
            self.offset += 1;
            Some(val)
        }
    }

    #[test]
    fn pitch_search_within_tolerance() {
        let pitch_info_size = 2; // period + strength
        let chunk_size = BUF_SIZE_24K_HZ + pitch_info_size;

        // Use ARM64-specific reference data on ARM64 platforms.
        let resource_name = if cfg!(target_arch = "aarch64") {
            "pitch_lp_res_arm64.dat"
        } else {
            "pitch_lp_res.dat"
        };

        let path = test_resources_dir().join(resource_name);
        let mut reader = ChunksReader::open(&path, chunk_size);
        let num_frames = reader.num_chunks().min(300); // Max 3 s.

        let backend = webrtc_simd::detect_backend();
        let mut pitch_estimator = PitchEstimator::new(backend);
        let mut lp_residual = vec![0.0_f32; BUF_SIZE_24K_HZ];

        for i in 0..num_frames {
            assert!(
                reader.read_chunk(&mut lp_residual),
                "Failed to read LP residual at frame {i}"
            );
            let expected_pitch_period = reader.read_value().unwrap();
            let expected_pitch_strength = reader.read_value().unwrap();

            let pitch_period = pitch_estimator.estimate(&lp_residual);
            assert_eq!(
                expected_pitch_period, pitch_period as f32,
                "Pitch period mismatch at frame {i}"
            );
            assert!(
                (expected_pitch_strength - pitch_estimator.last_pitch_strength()).abs() < 15e-6,
                "Pitch strength mismatch at frame {i}: expected {expected_pitch_strength}, got {}",
                pitch_estimator.last_pitch_strength()
            );
        }
    }
}
