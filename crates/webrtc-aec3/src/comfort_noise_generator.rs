//! Comfort noise generator — synthesizes comfort noise to fill in suppressed
//! regions.
//!
//! Ported from `modules/audio_processing/aec3/comfort_noise_generator.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::EchoCanceller3Config;
use crate::fft_data::FftData;
use crate::vector_math::VectorMath;

const FFT_LENGTH_BY_2: usize = crate::common::FFT_LENGTH_BY_2;

/// Table of sqrt(2) * sin(2*pi*i/32).
const SQRT2: f32 = std::f32::consts::SQRT_2;

#[allow(clippy::excessive_precision, reason = "exact upstream table values")]
const K_SQRT2_SIN: [f32; 32] = [
    0.0000000, 0.2758994, 0.5411961, 0.7856950, 1.0000000, 1.1758756, 1.3065630, 1.3870398, SQRT2,
    1.3870398, 1.3065630, 1.1758756, 1.0000000, 0.7856950, 0.5411961, 0.2758994, 0.0000000,
    -0.2758994, -0.5411961, -0.7856950, -1.0000000, -1.1758756, -1.3065630, -1.3870398, -SQRT2,
    -1.3870398, -1.3065630, -1.1758756, -1.0000000, -0.7856950, -0.5411961, -0.2758994,
];

/// Computes the noise floor value that matches a WGN input of noise_floor_dbfs.
fn get_noise_floor_factor(noise_floor_dbfs: f32) -> f32 {
    // kdBfsNormalization = 20.f*log10(32768.f).
    const K_DBFS_NORMALIZATION: f32 = 90.308_99;
    64.0 * 10.0f32.powf((K_DBFS_NORMALIZATION + noise_floor_dbfs) * 0.1)
}

/// Generates comfort noise for a single channel from the noise power spectrum.
fn generate_comfort_noise(
    n2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    seed: &mut u32,
    lower_band_noise: &mut FftData,
    upper_band_noise: &mut FftData,
    vector_math: &VectorMath,
) {
    // Compute square root spectrum.
    let mut n = *n2;
    vector_math.sqrt(&mut n);

    // Compute the noise level for the upper bands.
    // C++ uses integer division: kFftLengthBy2Plus1 / 2 + 1 = 65/2 + 1 = 33
    const K_FFT_LENGTH_BY_2_PLUS_1_BY_2: usize = FFT_LENGTH_BY_2_PLUS_1 / 2;
    const K_ONE_BY_NUM_BANDS: f32 = 1.0 / (K_FFT_LENGTH_BY_2_PLUS_1_BY_2 + 1) as f32;
    let high_band_noise_level: f32 =
        n[K_FFT_LENGTH_BY_2_PLUS_1_BY_2..].iter().sum::<f32>() * K_ONE_BY_NUM_BANDS;

    // The analysis and synthesis windowing cause loss of power when
    // cross-fading the noise where frames are completely uncorrelated
    // (generated with random phase), hence the factor sqrt(2).
    lower_band_noise.re[0] = 0.0;
    lower_band_noise.re[FFT_LENGTH_BY_2] = 0.0;
    upper_band_noise.re[0] = 0.0;
    upper_band_noise.re[FFT_LENGTH_BY_2] = 0.0;

    const INDEX_MASK: u32 = 32 - 1;

    for k in 1..FFT_LENGTH_BY_2 {
        // Generate a random 31-bit integer.
        *seed = seed.wrapping_mul(69069).wrapping_add(1) & (0x8000_0000 - 1);
        // Convert to a 5-bit index.
        let i = (*seed >> 26) as usize;

        // y = sqrt(2) * sin(a)
        let x = K_SQRT2_SIN[i];
        // x = sqrt(2) * cos(a) = sqrt(2) * sin(a + pi/2)
        let y = K_SQRT2_SIN[(i + 8) & INDEX_MASK as usize];

        // Form low-frequency noise via spectral shaping.
        lower_band_noise.re[k] = n[k] * x;
        lower_band_noise.im[k] = n[k] * y;

        // Form the high-frequency noise via simple levelling.
        upper_band_noise.re[k] = high_band_noise_level * x;
        upper_band_noise.im[k] = high_band_noise_level * y;
    }
}

/// Generates the comfort noise.
pub(crate) struct ComfortNoiseGenerator {
    vector_math: VectorMath,
    seed: u32,
    num_capture_channels: usize,
    noise_floor: f32,
    n2_initial: Option<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    y2_smoothed: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    n2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    n2_counter: i32,
}

impl ComfortNoiseGenerator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let backend = webrtc_simd::detect_backend();
        Self {
            vector_math: VectorMath::new(backend),
            seed: 42,
            num_capture_channels,
            noise_floor: get_noise_floor_factor(config.comfort_noise.noise_floor_dbfs),
            n2_initial: Some(vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels]),
            y2_smoothed: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            n2: vec![[1.0e6; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            n2_counter: 0,
        }
    }

    /// Computes the comfort noise.
    pub(crate) fn compute(
        &mut self,
        saturated_capture: bool,
        capture_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        lower_band_noise: &mut [FftData],
        upper_band_noise: &mut [FftData],
    ) {
        let y2 = capture_spectrum;

        if !saturated_capture {
            // Smooth Y2.
            for ch in 0..self.num_capture_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.y2_smoothed[ch][k] += 0.1 * (y2[ch][k] - self.y2_smoothed[ch][k]);
                }
            }

            if self.n2_counter > 50 {
                // Update N2 from Y2_smoothed.
                for ch in 0..self.num_capture_channels {
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        let a = self.n2[ch][k];
                        let b = self.y2_smoothed[ch][k];
                        self.n2[ch][k] = if b < a {
                            (0.9 * b + 0.1 * a) * 1.0002
                        } else {
                            a * 1.0002
                        };
                    }
                }
            }

            if let Some(ref mut n2_initial) = self.n2_initial {
                self.n2_counter += 1;
                if self.n2_counter == 1000 {
                    self.n2_initial = None;
                } else {
                    // Compute the N2_initial from N2.
                    for ch in 0..self.num_capture_channels {
                        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                            let a = self.n2[ch][k];
                            let b = n2_initial[ch][k];
                            n2_initial[ch][k] = if a > b { b + 0.001 * (a - b) } else { a };
                        }
                    }
                }
            }

            for ch in 0..self.num_capture_channels {
                for n in &mut self.n2[ch] {
                    *n = n.max(self.noise_floor);
                }
                if let Some(ref mut n2_initial) = self.n2_initial {
                    for n in &mut n2_initial[ch] {
                        *n = n.max(self.noise_floor);
                    }
                }
            }
        }

        // Choose N2 estimate to use.
        for ch in 0..self.num_capture_channels {
            let n2_ch = if let Some(ref n2_initial) = self.n2_initial {
                &n2_initial[ch]
            } else {
                &self.n2[ch]
            };
            generate_comfort_noise(
                n2_ch,
                &mut self.seed,
                &mut lower_band_noise[ch],
                &mut upper_band_noise[ch],
                &self.vector_math,
            );
        }
    }

    /// Returns the estimate of the background noise spectrum.
    pub(crate) fn noise_spectrum(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        &self.n2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn power(n: &FftData) -> f32 {
        let mut n2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        n.spectrum(&mut n2);
        n2.iter().sum::<f32>() / n2.len() as f32
    }

    #[test]
    fn correct_level() {
        const NUM_CHANNELS: usize = 5;
        let config = EchoCanceller3Config::default();
        let mut cng = ComfortNoiseGenerator::new(&config, NUM_CHANNELS);

        let mut n2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; NUM_CHANNELS];
        let mut n_lower = vec![FftData::default(); NUM_CHANNELS];
        let mut n_upper = vec![FftData::default(); NUM_CHANNELS];

        for ch in 0..NUM_CHANNELS {
            n2[ch].fill(1000.0 * 1000.0 / (ch + 1) as f32);
        }

        // Ensure instantaneous update to nonzero noise.
        cng.compute(false, &n2, &mut n_lower, &mut n_upper);

        for ch in 0..NUM_CHANNELS {
            assert!(power(&n_lower[ch]) > 0.0);
            assert!(power(&n_upper[ch]) > 0.0);
        }

        for _ in 0..10000 {
            cng.compute(false, &n2, &mut n_lower, &mut n_upper);
        }

        for ch in 0..NUM_CHANNELS {
            let expected = 2.0 * n2[ch][0];
            let tolerance = n2[ch][0] / 10.0;
            assert!(
                (expected - power(&n_lower[ch])).abs() < tolerance,
                "ch {ch}: lower power {} not near expected {expected}",
                power(&n_lower[ch])
            );
            assert!(
                (expected - power(&n_upper[ch])).abs() < tolerance,
                "ch {ch}: upper power {} not near expected {expected}",
                power(&n_upper[ch])
            );
        }
    }
}
