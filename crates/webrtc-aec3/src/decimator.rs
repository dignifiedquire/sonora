//! Signal decimator with anti-aliasing and noise reduction filtering.
//!
//! Ported from `modules/audio_processing/aec3/decimator.h/cc`.

use crate::cascaded_biquad_filter::{BiQuadCoefficients, CascadedBiQuadFilter};
use crate::common::BLOCK_SIZE;

// signal.ellip(6, 1, 40, 1800/8000, 'lowpass', output='sos')
const LOW_PASS_FILTER_DS4: [BiQuadCoefficients; 3] = [
    BiQuadCoefficients {
        b: [0.018_091_987_7, 0.003_209_613_63, 0.018_091_987_7],
        a: [-1.518_319_5, 0.633_165_865],
    },
    BiQuadCoefficients {
        b: [1.0, -1.245_504_59, 1.0],
        a: [-1.497_842_54, 0.853_586_692],
    },
    BiQuadCoefficients {
        b: [1.0, -1.422_168_1, 1.0],
        a: [-1.497_912_82, 0.969_572_384],
    },
];

// signal.cheby1(1, 6, [1000/8000, 2000/8000], 'bandpass', output='sos')
// repeated 5 times.
const BAND_PASS_FILTER_DS8: [BiQuadCoefficients; 5] = [BiQuadCoefficients {
    b: [0.103_304_783, 0.0, -0.103_304_783],
    a: [-1.520_363, 0.793_390_435],
}; 5];

// signal.butter(2, 1000/8000.0, 'highpass', output='sos')
const HIGH_PASS_FILTER: [BiQuadCoefficients; 1] = [BiQuadCoefficients {
    b: [0.757_076_375, -1.514_152_75, 0.757_076_375],
    a: [-1.454_243_59, 0.574_061_915],
}];

/// Downsamples a 64-sample block by factor 4 or 8 with anti-aliasing.
pub(crate) struct Decimator {
    down_sampling_factor: usize,
    anti_aliasing_filter: CascadedBiQuadFilter,
    noise_reduction_filter: CascadedBiQuadFilter,
}

impl Decimator {
    pub(crate) fn new(down_sampling_factor: usize) -> Self {
        debug_assert!(down_sampling_factor == 4 || down_sampling_factor == 8);
        let anti_aliasing_filter = if down_sampling_factor == 4 {
            CascadedBiQuadFilter::new(&LOW_PASS_FILTER_DS4)
        } else {
            CascadedBiQuadFilter::new(&BAND_PASS_FILTER_DS8)
        };
        let noise_reduction_filter = if down_sampling_factor == 8 {
            CascadedBiQuadFilter::new(&[]) // pass-through
        } else {
            CascadedBiQuadFilter::new(&HIGH_PASS_FILTER)
        };
        Self {
            down_sampling_factor,
            anti_aliasing_filter,
            noise_reduction_filter,
        }
    }

    /// Decimates a `BLOCK_SIZE`-sample input into an output of
    /// `BLOCK_SIZE / down_sampling_factor` samples.
    pub(crate) fn decimate(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(BLOCK_SIZE, input.len());
        debug_assert_eq!(BLOCK_SIZE / self.down_sampling_factor, output.len());

        let mut x = [0.0f32; BLOCK_SIZE];
        self.anti_aliasing_filter.process(input, &mut x);
        self.noise_reduction_filter.process_in_place(&mut x);

        for (j, out) in output.iter_mut().enumerate() {
            *out = x[j * self.down_sampling_factor];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const NUM_STARTUP_BLOCKS: usize = 50;
    const NUM_BLOCKS: usize = 1000;

    fn produce_decimated_sinusoidal_output_power(
        sample_rate_hz: f32,
        down_sampling_factor: usize,
        sinusoidal_frequency_hz: f32,
    ) -> (f32, f32) {
        let total_samples = BLOCK_SIZE * NUM_BLOCKS;
        let sub_block_size = BLOCK_SIZE / down_sampling_factor;

        let input: Vec<f32> = (0..total_samples)
            .map(|k| {
                32767.0 * (2.0 * PI * sinusoidal_frequency_hz * k as f32 / sample_rate_hz).sin()
            })
            .collect();

        let mut decimator = Decimator::new(down_sampling_factor);
        let mut output = vec![0.0f32; sub_block_size * NUM_BLOCKS];

        for k in 0..NUM_BLOCKS {
            let mut sub_block = vec![0.0f32; sub_block_size];
            decimator.decimate(&input[k * BLOCK_SIZE..(k + 1) * BLOCK_SIZE], &mut sub_block);
            output[k * sub_block_size..(k + 1) * sub_block_size].copy_from_slice(&sub_block);
        }

        let input_eval = &input[NUM_STARTUP_BLOCKS * BLOCK_SIZE..];
        let output_eval = &output[NUM_STARTUP_BLOCKS * sub_block_size..];

        let input_power: f32 =
            input_eval.iter().map(|x| x * x).sum::<f32>() / input_eval.len() as f32;
        let output_power: f32 =
            output_eval.iter().map(|x| x * x).sum::<f32>() / output_eval.len() as f32;

        (input_power, output_power)
    }

    #[test]
    fn no_leakage_from_upper_frequencies() {
        for &rate in &[16000, 32000, 48000] {
            for &dsf in &[4usize, 8] {
                let freq = 3.0 / 8.0 * rate as f32;
                let (input_power, output_power) =
                    produce_decimated_sinusoidal_output_power(rate as f32, dsf, freq);
                assert!(
                    output_power < 0.0001 * input_power,
                    "rate={rate}, dsf={dsf}: output_power={output_power}, \
                     input_power={input_power}, ratio={}",
                    output_power / input_power
                );
            }
        }
    }
}
