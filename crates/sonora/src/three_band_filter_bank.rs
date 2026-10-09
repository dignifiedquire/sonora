//! 3-band FIR filter bank with DCT modulation.
//!
//! Splits a 480-sample (48 kHz, 10 ms) frame into three 160-sample sub-bands
//! (0–8 kHz, 8–16 kHz, 16–24 kHz) and can merge them back.
//!
//! Ported from `modules/audio_processing/three_band_filter_bank.h/cc`.

use sonora_simd::NATIVE_FMA;

const SQRT_3: f32 = 1.732_050_8;

const SPARSITY: usize = 4;
const STRIDE_LOG2: usize = 2;
const STRIDE: usize = 1 << STRIDE_LOG2;
const NUM_ZERO_FILTERS: usize = 2;
const FILTER_SIZE: usize = 4;
const MEMORY_SIZE: usize = FILTER_SIZE * STRIDE - 1; // 15

/// Number of frequency bands.
pub const NUM_BANDS: usize = 3;
/// Full-band frame size (480 samples = 48 kHz × 10 ms).
pub const FULL_BAND_SIZE: usize = 480;
/// Split-band frame size (160 samples per band).
pub const SPLIT_BAND_SIZE: usize = FULL_BAND_SIZE / NUM_BANDS;

const NUM_NON_ZERO_FILTERS: usize = SPARSITY * NUM_BANDS - NUM_ZERO_FILTERS; // 10
const SUB_SAMPLING: usize = NUM_BANDS;
const ZERO_FILTER_INDEX_1: usize = 3;
const ZERO_FILTER_INDEX_2: usize = 9;

#[rustfmt::skip]
const FILTER_COEFFS: [[f32; FILTER_SIZE]; NUM_NON_ZERO_FILTERS] = [
    [-0.00047749, -0.00496888, 0.16547118,  0.00425496],
    [-0.00173287, -0.01585778, 0.14989004,  0.00994113],
    [-0.00304815, -0.02536082, 0.12154542,  0.01157993],
    [-0.00346946, -0.02587886, 0.04760441,  0.00607594],
    [-0.00154717, -0.01136076, 0.01387458,  0.00186353],
    [ 0.00186353,  0.01387458,-0.01136076, -0.00154717],
    [ 0.00607594,  0.04760441,-0.02587886, -0.00346946],
    [ 0.00983212,  0.08543175,-0.02982767, -0.00383509],
    [ 0.00994113,  0.14989004,-0.01585778, -0.00173287],
    [ 0.00425496,  0.16547118,-0.00496888, -0.00047749],
];

#[rustfmt::skip]
const DCT_MODULATION: [[f32; NUM_BANDS]; NUM_NON_ZERO_FILTERS] = [
    [ 2.0,     2.0,    2.0],
    [ SQRT_3,  0.0,   -SQRT_3],
    [ 1.0,    -2.0,    1.0],
    [-1.0,     2.0,   -1.0],
    [-SQRT_3,  0.0,    SQRT_3],
    [-2.0,    -2.0,   -2.0],
    [-SQRT_3,  0.0,    SQRT_3],
    [-1.0,     2.0,   -1.0],
    [ 1.0,    -2.0,    1.0],
    [ SQRT_3,  0.0,   -SQRT_3],
];

/// Polyphase filter core: filters `input` through `filter` with shift `in_shift`,
/// using and updating `state`.
///
/// If `FMA`, the unrolled 4-tap sums of Parts 1 and 3 use [`f32::mul_add`];
/// otherwise they use plain arithmetic, in the C++ operation order. C++ starts
/// that sum from `0.0`; leaving it out changes only the sign of an all-zero
/// sum, which `analysis` and `synthesis` lose when they add the result into
/// zeroed buffers. The public methods pass [`NATIVE_FMA`]; the tests run both
/// forms.
///
/// Direct port of C++ `FilterCore` in `three_band_filter_bank.cc`.
// LLVM's inline cost for this body sits at its threshold, so without
// `inline(always)` whether `analysis` and `synthesis` inline it depends on
// the codegen-unit partition (with Cargo's default 16 units it did on
// aarch64 and did not on x86_64).
#[inline(always)]
fn filter_core<const FMA: bool>(
    filter: &[f32; FILTER_SIZE],
    input: &[f32; SPLIT_BAND_SIZE],
    in_shift: usize,
    output: &mut [f32; SPLIT_BAND_SIZE],
    state: &mut [f32; MEMORY_SIZE],
) {
    debug_assert!(in_shift < STRIDE);

    // Zero-initialize output (matches C++ std::fill).
    output.fill(0.0);

    let f0 = filter[0];
    let f1 = filter[1];
    let f2 = filter[2];
    let f3 = filter[3];

    // Part 1: samples that depend entirely on state (0..in_shift, at most 3 iterations).
    #[allow(clippy::needless_range_loop, reason = "index used in arithmetic")]
    for k in 0..in_shift {
        let j = MEMORY_SIZE + k - in_shift;
        output[k] = if FMA {
            f0.mul_add(
                state[j],
                f1.mul_add(
                    state[j - STRIDE],
                    f2.mul_add(state[j - 2 * STRIDE], f3 * state[j - 3 * STRIDE]),
                ),
            )
        } else {
            f0 * state[j]
                + f1 * state[j - STRIDE]
                + f2 * state[j - 2 * STRIDE]
                + f3 * state[j - 3 * STRIDE]
        };
    }

    // Part 2: transition samples (partially from input, partially from state).
    // Matches C++ loop structure with loop_limit = min(kFilterSize, 1 + (shift >> kStrideLog2)).
    #[allow(clippy::needless_range_loop, reason = "index used in arithmetic")]
    for k in in_shift..(FILTER_SIZE * STRIDE) {
        let shift = k - in_shift;
        let loop_limit = (1 + (shift >> STRIDE_LOG2)).min(FILTER_SIZE);

        // Taps sourced from input.
        for i in 0..loop_limit {
            output[k] += input[shift - i * STRIDE] * filter[i];
        }

        // Taps sourced from state.
        let j_base = MEMORY_SIZE + shift - loop_limit * STRIDE;
        for i in loop_limit..FILTER_SIZE {
            output[k] += state[j_base - (i - loop_limit) * STRIDE] * filter[i];
        }
    }

    // Part 3: samples fully within input (hottest path — 144 of 160 iterations).
    // All 4 taps read from input at fixed offsets.
    #[allow(clippy::needless_range_loop, reason = "index used in arithmetic")]
    for k in (FILTER_SIZE * STRIDE)..SPLIT_BAND_SIZE {
        let base = k - in_shift;
        output[k] = if FMA {
            f0.mul_add(
                input[base],
                f1.mul_add(
                    input[base - STRIDE],
                    f2.mul_add(input[base - 2 * STRIDE], f3 * input[base - 3 * STRIDE]),
                ),
            )
        } else {
            f0 * input[base]
                + f1 * input[base - STRIDE]
                + f2 * input[base - 2 * STRIDE]
                + f3 * input[base - 3 * STRIDE]
        };
    }

    // Update state from end of input.
    state.copy_from_slice(&input[SPLIT_BAND_SIZE - MEMORY_SIZE..]);
}

/// 3-band QMF filter bank for analysis and synthesis.
#[derive(Debug)]
pub struct ThreeBandFilterBank {
    state_analysis: [[f32; MEMORY_SIZE]; NUM_NON_ZERO_FILTERS],
    state_synthesis: [[f32; MEMORY_SIZE]; NUM_NON_ZERO_FILTERS],
}

impl Default for ThreeBandFilterBank {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreeBandFilterBank {
    pub fn new() -> Self {
        Self {
            state_analysis: [[0.0; MEMORY_SIZE]; NUM_NON_ZERO_FILTERS],
            state_synthesis: [[0.0; MEMORY_SIZE]; NUM_NON_ZERO_FILTERS],
        }
    }

    /// Splits a 480-sample fullband frame into 3 × 160-sample sub-bands.
    pub fn analysis(
        &mut self,
        input: &[f32; FULL_BAND_SIZE],
        output: &mut [[f32; SPLIT_BAND_SIZE]; NUM_BANDS],
    ) {
        self.analysis_with::<NATIVE_FMA>(input, output);
    }

    /// [`Self::analysis`], with [`filter_core`] in the form `FMA` selects.
    /// Always inlined, so that LLVM optimizes this body as part of
    /// `analysis`.
    #[inline(always)]
    fn analysis_with<const FMA: bool>(
        &mut self,
        input: &[f32; FULL_BAND_SIZE],
        output: &mut [[f32; SPLIT_BAND_SIZE]; NUM_BANDS],
    ) {
        // Initialize output to zero.
        for band in output.iter_mut() {
            band.fill(0.0);
        }

        for downsampling_index in 0..SUB_SAMPLING {
            // Downsample: pick every SUB_SAMPLING-th sample with offset.
            let mut in_subsampled = [0.0f32; SPLIT_BAND_SIZE];
            for k in 0..SPLIT_BAND_SIZE {
                in_subsampled[k] =
                    input[(SUB_SAMPLING - 1) - downsampling_index + SUB_SAMPLING * k];
            }

            for in_shift in 0..STRIDE {
                // Choose filter, skip zero filters.
                let index = downsampling_index + in_shift * SUB_SAMPLING;
                if index == ZERO_FILTER_INDEX_1 || index == ZERO_FILTER_INDEX_2 {
                    continue;
                }
                let filter_index = if index < ZERO_FILTER_INDEX_1 {
                    index
                } else if index < ZERO_FILTER_INDEX_2 {
                    index - 1
                } else {
                    index - 2
                };

                let filter = &FILTER_COEFFS[filter_index];
                let dct_mod = &DCT_MODULATION[filter_index];

                // Filter.
                let mut out_subsampled = [0.0f32; SPLIT_BAND_SIZE];
                filter_core::<FMA>(
                    filter,
                    &in_subsampled,
                    in_shift,
                    &mut out_subsampled,
                    &mut self.state_analysis[filter_index],
                );

                // Band-modulate and accumulate.
                for band in 0..NUM_BANDS {
                    let mod_val = dct_mod[band];
                    for n in 0..SPLIT_BAND_SIZE {
                        output[band][n] += mod_val * out_subsampled[n];
                    }
                }
            }
        }
    }

    /// Merges 3 × 160-sample sub-bands into a 480-sample fullband frame.
    pub fn synthesis(
        &mut self,
        input: &[[f32; SPLIT_BAND_SIZE]; NUM_BANDS],
        output: &mut [f32; FULL_BAND_SIZE],
    ) {
        self.synthesis_with::<NATIVE_FMA>(input, output);
    }

    /// [`Self::synthesis`], with [`filter_core`] in the form `FMA` selects.
    /// Always inlined, so that LLVM optimizes this body as part of
    /// `synthesis`.
    #[inline(always)]
    fn synthesis_with<const FMA: bool>(
        &mut self,
        input: &[[f32; SPLIT_BAND_SIZE]; NUM_BANDS],
        output: &mut [f32; FULL_BAND_SIZE],
    ) {
        output.fill(0.0);

        for upsampling_index in 0..SUB_SAMPLING {
            for in_shift in 0..STRIDE {
                // Choose filter, skip zero filters.
                let index = upsampling_index + in_shift * SUB_SAMPLING;
                if index == ZERO_FILTER_INDEX_1 || index == ZERO_FILTER_INDEX_2 {
                    continue;
                }
                let filter_index = if index < ZERO_FILTER_INDEX_1 {
                    index
                } else if index < ZERO_FILTER_INDEX_2 {
                    index - 1
                } else {
                    index - 2
                };

                let filter = &FILTER_COEFFS[filter_index];
                let dct_mod = &DCT_MODULATION[filter_index];

                // Prepare filter input by modulating the banded input.
                let mut in_subsampled = [0.0f32; SPLIT_BAND_SIZE];
                for band in 0..NUM_BANDS {
                    let mod_val = dct_mod[band];
                    for n in 0..SPLIT_BAND_SIZE {
                        in_subsampled[n] += mod_val * input[band][n];
                    }
                }

                // Filter.
                let mut out_subsampled = [0.0f32; SPLIT_BAND_SIZE];
                filter_core::<FMA>(
                    filter,
                    &in_subsampled,
                    in_shift,
                    &mut out_subsampled,
                    &mut self.state_synthesis[filter_index],
                );

                // Upsample.
                let upsampling_scaling = SUB_SAMPLING as f32;
                for k in 0..SPLIT_BAND_SIZE {
                    output[upsampling_index + SUB_SAMPLING * k] +=
                        upsampling_scaling * out_subsampled[k];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_produces_output() {
        let mut fb = ThreeBandFilterBank::new();
        // Impulse input.
        let mut input = [0.0f32; FULL_BAND_SIZE];
        input[0] = 1.0;
        let mut output = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        fb.analysis(&input, &mut output);

        // At least one band should have non-zero output.
        let total_energy: f32 = output.iter().flat_map(|b| b.iter()).map(|x| x * x).sum();
        assert!(
            total_energy > 0.0,
            "output should be non-zero for impulse input"
        );
    }

    #[test]
    fn synthesis_produces_output() {
        let mut fb = ThreeBandFilterBank::new();
        // Put a signal in band 0.
        let mut input = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        input[0][0] = 1.0;
        let mut output = [0.0f32; FULL_BAND_SIZE];
        fb.synthesis(&input, &mut output);

        let total_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(total_energy > 0.0, "output should be non-zero");
    }

    #[test]
    fn analysis_synthesis_roundtrip() {
        use std::f32::consts::PI;

        let mut fb_analysis = ThreeBandFilterBank::new();
        let mut fb_synthesis = ThreeBandFilterBank::new();

        // Create a signal and process multiple frames to let filter settle.
        let num_frames = 20;
        let mut last_input = [0.0f32; FULL_BAND_SIZE];
        let mut last_output = [0.0f32; FULL_BAND_SIZE];

        for frame in 0..num_frames {
            let mut input = [0.0f32; FULL_BAND_SIZE];
            // Sine wave at ~1 kHz (well within band 0).
            for (i, sample) in input.iter_mut().enumerate() {
                let t = (frame * FULL_BAND_SIZE + i) as f32 / 48000.0;
                *sample = (2.0 * PI * 1000.0 * t).sin();
            }

            let mut bands = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
            fb_analysis.analysis(&input, &mut bands);

            let mut output = [0.0f32; FULL_BAND_SIZE];
            fb_synthesis.synthesis(&bands, &mut output);

            last_input = input;
            last_output = output;
        }

        // After many frames, the output should approximate the input
        // (with a fixed delay of 24 samples and ~9.5 dB SNR).
        // Just check that the output has significant energy.
        let input_energy: f32 = last_input.iter().map(|x| x * x).sum();
        let output_energy: f32 = last_output.iter().map(|x| x * x).sum();
        assert!(
            output_energy > input_energy * 0.05,
            "roundtrip should preserve most energy: input={input_energy}, output={output_energy}",
        );
    }

    /// Both forms of the unrolled Parts 1 and 3 of `filter_core`, on every
    /// target: the fused form must match the `mul_add` chain, and the plain
    /// form the C++ tap order, bit for bit.
    #[test]
    fn filter_core_matches_fused_and_plain_references() {
        use std::array::from_fn;

        let filter = &FILTER_COEFFS[1];
        let state: [f32; MEMORY_SIZE] = from_fn(|i| (i as f32 * 0.618_034).fract() - 0.5);
        let input: [f32; SPLIT_BAND_SIZE] = from_fn(|i| (i as f32 * 0.414_214).fract() - 0.5);
        // Sample n of the stream is history[MEMORY_SIZE + n]; n < 0 is state.
        let history: Vec<f32> = state.iter().chain(&input).copied().collect();

        let (mut part1_differs, mut part3_differs) = (false, false);
        // in_shift >= 1 so that Part 1 (state only) runs.
        for in_shift in 1..STRIDE {
            let mut fused_output = [0.0_f32; SPLIT_BAND_SIZE];
            filter_core::<true>(
                filter,
                &input,
                in_shift,
                &mut fused_output,
                &mut state.clone(),
            );
            let mut plain_output = [0.0_f32; SPLIT_BAND_SIZE];
            filter_core::<false>(
                filter,
                &input,
                in_shift,
                &mut plain_output,
                &mut state.clone(),
            );

            // Part 2 (taps split between state and input) is unchanged.
            let parts_1_and_3 = (0..in_shift).chain(FILTER_SIZE * STRIDE..SPLIT_BAND_SIZE);
            for k in parts_1_and_3 {
                let t: [f32; FILTER_SIZE] =
                    from_fn(|i| history[MEMORY_SIZE + k - in_shift - i * STRIDE]);
                let f = filter;
                let fused = f[0].mul_add(t[0], f[1].mul_add(t[1], f[2].mul_add(t[2], f[3] * t[3])));
                // C++ accumulates `out[k] += in[j] * filter[i]` for i = 0..3.
                let plain = f[0] * t[0] + f[1] * t[1] + f[2] * t[2] + f[3] * t[3];
                if fused.to_bits() != plain.to_bits() {
                    if k < in_shift {
                        part1_differs = true;
                    } else {
                        part3_differs = true;
                    }
                }
                assert_eq!(
                    fused_output[k].to_bits(),
                    fused.to_bits(),
                    "fused, in_shift {in_shift}, k {k}"
                );
                assert_eq!(
                    plain_output[k].to_bits(),
                    plain.to_bits(),
                    "plain, in_shift {in_shift}, k {k}"
                );
            }
        }
        assert!(part1_differs && part3_differs);
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Output bits of `analysis_with::<FMA>` on `input` and of
    /// `synthesis_with::<FMA>` on `bands`, each on a new filter bank.
    fn run_forms<const FMA: bool>(
        input: &[f32; FULL_BAND_SIZE],
        bands: &[[f32; SPLIT_BAND_SIZE]; NUM_BANDS],
    ) -> [Vec<u32>; 2] {
        let mut analysis = [[0.0_f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        ThreeBandFilterBank::new().analysis_with::<FMA>(input, &mut analysis);
        let mut synthesis = [0.0_f32; FULL_BAND_SIZE];
        ThreeBandFilterBank::new().synthesis_with::<FMA>(bands, &mut synthesis);
        [bits(analysis.as_flattened()), bits(&synthesis)]
    }

    /// `analysis` and `synthesis` run the [`NATIVE_FMA`] form.
    #[test]
    fn analysis_and_synthesis_use_native_fma() {
        use std::array::from_fn;

        let input: [f32; FULL_BAND_SIZE] = from_fn(|i| (i as f32 * 0.618_034).fract() - 0.5);
        let bands: [[f32; SPLIT_BAND_SIZE]; NUM_BANDS] =
            from_fn(|b| from_fn(|i| ((b * SPLIT_BAND_SIZE + i) as f32 * 0.414_214).fract() - 0.5));
        // The forms differ on these inputs, so the comparison below can fail.
        let [fused, plain] = [
            run_forms::<true>(&input, &bands),
            run_forms::<false>(&input, &bands),
        ];
        assert!(fused[0] != plain[0] && fused[1] != plain[1]);

        let mut analysis = [[0.0_f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        ThreeBandFilterBank::new().analysis(&input, &mut analysis);
        let mut synthesis = [0.0_f32; FULL_BAND_SIZE];
        ThreeBandFilterBank::new().synthesis(&bands, &mut synthesis);
        assert_eq!(
            [bits(analysis.as_flattened()), bits(&synthesis)],
            run_forms::<NATIVE_FMA>(&input, &bands)
        );
    }

    #[test]
    fn zero_input_produces_zero_output() {
        let mut fb = ThreeBandFilterBank::new();
        let input = [0.0f32; FULL_BAND_SIZE];
        let mut output = [[1.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        fb.analysis(&input, &mut output);

        for band in &output {
            for &s in band {
                assert_eq!(s, 0.0);
            }
        }
    }
}
