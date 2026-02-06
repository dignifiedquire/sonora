//! Audio sample format conversions matching WebRTC's `audio_util.h`.
//!
//! # Format conventions
//!
//! | Name      | Type    | Range                          |
//! |-----------|---------|--------------------------------|
//! | S16       | `i16`   | \[-32768, 32767\]              |
//! | Float     | `f32`   | \[-1.0, 1.0\]                  |
//! | FloatS16  | `f32`   | \[-32768.0, 32768.0\]          |
//! | Dbfs      | `f32`   | \[-90.31, 0\] (approx)         |

/// Absolute highest acceptable sample rate (Hz).
pub const MAX_SAMPLE_RATE_HZ: u32 = 384_000;

/// Maximum samples per channel in a 10 ms frame at [`MAX_SAMPLE_RATE_HZ`].
pub const MAX_SAMPLES_PER_CHANNEL_10MS: usize = MAX_SAMPLE_RATE_HZ as usize / 100;

const S16_TO_FLOAT_SCALING: f32 = 1.0 / 32768.0;
const MIN_DBFS: f32 = -90.309;
const MAX_ABS_FLOAT_S16: f32 = 32768.0;

// ── Scalar conversions ──────────────────────────────────────────────

/// Convert a single S16 sample to Float \[-1.0, 1.0\].
#[inline]
pub fn s16_to_float(v: i16) -> f32 {
    f32::from(v) * S16_TO_FLOAT_SCALING
}

/// Convert a single FloatS16 sample to S16, rounding to nearest.
#[inline]
pub fn float_s16_to_s16(v: f32) -> i16 {
    let v = v.clamp(-32768.0, 32767.0);
    (v + f32::copysign(0.5, v)) as i16
}

/// Convert a single Float \[-1.0, 1.0\] sample to S16.
#[inline]
pub fn float_to_s16(v: f32) -> i16 {
    let v = (v * 32768.0).clamp(-32768.0, 32767.0);
    (v + f32::copysign(0.5, v)) as i16
}

/// Convert a single Float \[-1.0, 1.0\] to FloatS16 \[-32768.0, 32768.0\].
#[inline]
pub fn float_to_float_s16(v: f32) -> f32 {
    v.clamp(-1.0, 1.0) * 32768.0
}

/// Convert a single FloatS16 \[-32768.0, 32768.0\] to Float \[-1.0, 1.0\].
#[inline]
pub fn float_s16_to_float(v: f32) -> f32 {
    v.clamp(-32768.0, 32768.0) * S16_TO_FLOAT_SCALING
}

/// Convert dBFS to FloatS16 scale.
#[inline]
pub fn dbfs_to_float_s16(v: f32) -> f32 {
    db_to_ratio(v) * MAX_ABS_FLOAT_S16
}

/// Convert FloatS16 (>= 0) to dBFS.
#[inline]
pub fn float_s16_to_dbfs(v: f32) -> f32 {
    debug_assert!(v >= 0.0);
    if v <= 1.0 {
        return MIN_DBFS;
    }
    20.0 * v.log10() + MIN_DBFS
}

/// Convert a dB value to a linear ratio.
#[inline]
pub fn db_to_ratio(v: f32) -> f32 {
    10.0_f32.powf(v / 20.0)
}

// ── Slice conversions ───────────────────────────────────────────────

/// Convert a slice of S16 samples to Float in-place into `dest`.
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn s16_to_float_slice(src: &[i16], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = s16_to_float(s);
    }
}

/// Convert a slice of Float samples to S16 in-place into `dest`.
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn float_to_s16_slice(src: &[f32], dest: &mut [i16]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = float_to_s16(s);
    }
}

/// Convert a slice of S16 to FloatS16 (just widen to f32).
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn s16_to_float_s16_slice(src: &[i16], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = f32::from(s);
    }
}

/// Convert a slice of FloatS16 to S16 with rounding.
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn float_s16_to_s16_slice(src: &[f32], dest: &mut [i16]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = float_s16_to_s16(s);
    }
}

/// Convert a slice of Float to FloatS16.
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn float_to_float_s16_slice(src: &[f32], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = float_to_float_s16(s);
    }
}

/// Convert a slice of FloatS16 to Float.
///
/// # Panics
///
/// Panics if `src` and `dest` have different lengths.
pub fn float_s16_to_float_slice(src: &[f32], dest: &mut [f32]) {
    assert_eq!(src.len(), dest.len(), "slice length mismatch");
    for (d, &s) in dest.iter_mut().zip(src) {
        *d = float_s16_to_float(s);
    }
}

/// Convert a slice of Float to FloatS16 in-place.
pub fn float_to_float_s16_slice_inplace(data: &mut [f32]) {
    for s in data.iter_mut() {
        *s = float_to_float_s16(*s);
    }
}

/// Convert a slice of FloatS16 to Float in-place.
pub fn float_s16_to_float_slice_inplace(data: &mut [f32]) {
    for s in data.iter_mut() {
        *s = float_s16_to_float(*s);
    }
}

// ── Interleave / deinterleave ───────────────────────────────────────

/// Deinterleave multi-channel audio into per-channel buffers.
///
/// `interleaved` contains `num_channels` samples per frame, `samples_per_channel`
/// frames total. Each entry in `deinterleaved` receives one channel's samples.
pub fn deinterleave<T: Copy>(
    interleaved: &[T],
    deinterleaved: &mut [Vec<T>],
    samples_per_channel: usize,
    num_channels: usize,
) {
    assert_eq!(
        interleaved.len(),
        samples_per_channel * num_channels,
        "interleaved length mismatch"
    );
    assert_eq!(deinterleaved.len(), num_channels, "channel count mismatch");

    for (ch, channel_buf) in deinterleaved.iter_mut().enumerate() {
        assert!(
            channel_buf.len() >= samples_per_channel,
            "channel {ch} buffer too short"
        );
        let mut idx = ch;
        for slot in channel_buf.iter_mut().take(samples_per_channel) {
            *slot = interleaved[idx];
            idx += num_channels;
        }
    }
}

/// Interleave per-channel buffers into a single interleaved buffer.
pub fn interleave<T: Copy>(
    deinterleaved: &[&[T]],
    interleaved: &mut [T],
    samples_per_channel: usize,
    num_channels: usize,
) {
    assert_eq!(
        interleaved.len(),
        samples_per_channel * num_channels,
        "interleaved length mismatch"
    );
    assert_eq!(deinterleaved.len(), num_channels, "channel count mismatch");

    for (ch, channel_buf) in deinterleaved.iter().enumerate() {
        assert!(
            channel_buf.len() >= samples_per_channel,
            "channel {ch} buffer too short"
        );
        let mut idx = ch;
        for j in 0..samples_per_channel {
            interleaved[idx] = channel_buf[j];
            idx += num_channels;
        }
    }
}

/// Downmix interleaved multi-channel i16 audio to mono by averaging.
///
/// Uses `i32` accumulator to avoid overflow.
pub fn downmix_interleaved_to_mono_i16(
    interleaved: &[i16],
    num_frames: usize,
    num_channels: usize,
    mono: &mut [i16],
) {
    assert!(num_channels > 0, "num_channels must be > 0");
    assert!(num_frames > 0, "num_frames must be > 0");
    assert_eq!(
        interleaved.len(),
        num_frames * num_channels,
        "interleaved length mismatch"
    );
    assert!(mono.len() >= num_frames, "mono buffer too short");

    for (slot, frame) in mono.iter_mut().zip(interleaved.chunks(num_channels)) {
        let acc: i32 = frame.iter().map(|&s| i32::from(s)).sum();
        *slot = (acc / num_channels as i32) as i16;
    }
}

/// Downmix interleaved multi-channel f32 audio to mono by averaging.
pub fn downmix_interleaved_to_mono_f32(
    interleaved: &[f32],
    num_frames: usize,
    num_channels: usize,
    mono: &mut [f32],
) {
    assert!(num_channels > 0, "num_channels must be > 0");
    assert!(num_frames > 0, "num_frames must be > 0");
    assert_eq!(
        interleaved.len(),
        num_frames * num_channels,
        "interleaved length mismatch"
    );
    assert!(mono.len() >= num_frames, "mono buffer too short");

    for (slot, frame) in mono.iter_mut().zip(interleaved.chunks(num_channels)) {
        let acc: f32 = frame.iter().sum();
        *slot = acc / num_channels as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── S16ToFloat ──────────────────────────────────────────────────

    #[test]
    fn s16_to_float_known_values() {
        let input: &[i16] = &[0, 1, -1, 16384, -16384, 32767, -32768];
        let output: Vec<f32> = input.iter().map(|&v| s16_to_float(v)).collect();

        // Matches C++ inline: v * (1.0 / 32768.0)
        assert_eq!(output[0], 0.0);
        assert!((output[1] - 1.0 / 32768.0).abs() < 1e-10);
        assert!((output[2] - (-1.0 / 32768.0)).abs() < 1e-10);
        assert!((output[3] - 0.5).abs() < 1e-7);
        assert_eq!(output[4], -0.5);
        assert!((output[5] - (32767.0 / 32768.0)).abs() < 1e-7);
        assert_eq!(output[6], -1.0);
    }

    // ── FloatS16ToS16 ───────────────────────────────────────────────

    #[test]
    fn float_s16_to_s16_known_values() {
        let input: &[f32] = &[0.0, 0.4, 0.5, -0.4, -0.5, 32768.0, -32769.0];
        let expected: &[i16] = &[0, 0, 1, 0, -1, 32767, -32768];
        let output: Vec<i16> = input.iter().map(|&v| float_s16_to_s16(v)).collect();
        assert_eq!(&output, expected);
    }

    // ── FloatToFloatS16 ─────────────────────────────────────────────

    #[test]
    fn float_to_float_s16_known_values() {
        let input: &[f32] = &[
            0.0,
            0.4 / 32768.0,
            0.6 / 32768.0,
            -0.4 / 32768.0,
            -0.6 / 32768.0,
            1.0,
            -1.0,
        ];
        let expected: &[f32] = &[0.0, 0.4, 0.6, -0.4, -0.6, 32768.0, -32768.0];
        let output: Vec<f32> = input.iter().map(|&v| float_to_float_s16(v)).collect();
        for (o, e) in output.iter().zip(expected) {
            assert!((o - e).abs() < 0.01, "expected {e}, got {o}");
        }
    }

    // ── FloatS16ToFloat ─────────────────────────────────────────────

    #[test]
    fn float_s16_to_float_known_values() {
        let input: &[f32] = &[0.0, 0.4, 0.6, -0.4, -0.6, 32767.0, -32768.0];
        let expected: &[f32] = &[
            0.0,
            0.4 / 32768.0,
            0.6 / 32768.0,
            -0.4 / 32768.0,
            -0.6 / 32768.0,
            1.0,
            -1.0,
        ];
        let output: Vec<f32> = input.iter().map(|&v| float_s16_to_float(v)).collect();
        for (o, e) in output.iter().zip(expected) {
            assert!((o - e).abs() < 0.01, "expected {e}, got {o}");
        }
    }

    // ── DbfsToFloatS16 ─────────────────────────────────────────────

    #[test]
    fn dbfs_to_float_s16_known_values() {
        let input: &[f32] = &[-90.0, -70.0, -30.0, -20.0, -10.0, -5.0, -1.0, 0.0, 1.0];
        let expected: &[f32] = &[
            1.036_215_2,
            10.362_151,
            1_036.215_1,
            3_276.8,
            10_362.151,
            18_426.8,
            29_204.512,
            32_768.0,
            36_766.3,
        ];
        for (&i, &e) in input.iter().zip(expected) {
            let o = dbfs_to_float_s16(i);
            assert!(
                (o - e).abs() < 0.01,
                "dbfs_to_float_s16({i}): expected {e}, got {o}"
            );
        }
    }

    // ── FloatS16ToDbfs ──────────────────────────────────────────────

    #[test]
    fn float_s16_to_dbfs_known_values() {
        let input: &[f32] = &[
            1.036_215_1,
            10.362_151,
            1_036.215_1,
            3_276.8,
            10_362.151,
            18_426.8,
            29_204.511,
            32_768.0,
            36_766.3,
        ];
        let expected: &[f32] = &[-90.0, -70.0, -30.0, -20.0, -10.0, -5.0, -1.0, 0.0, 1.0];
        for (&i, &e) in input.iter().zip(expected) {
            let o = float_s16_to_dbfs(i);
            assert!(
                (o - e).abs() < 0.01,
                "float_s16_to_dbfs({i}): expected {e}, got {o}"
            );
        }
    }

    // ── Slice conversions ───────────────────────────────────────────

    #[test]
    fn s16_to_float_slice_roundtrip() {
        let input: &[i16] = &[0, 100, -100, 32767, -32768];
        let mut float_buf = vec![0.0_f32; input.len()];
        s16_to_float_slice(input, &mut float_buf);

        // Each value should be in [-1.0, 1.0]
        for &v in &float_buf {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn float_s16_to_s16_slice_matches_scalar() {
        let input: &[f32] = &[0.0, 0.4, 0.5, -0.4, -0.5, 32768.0, -32769.0];
        let mut output = vec![0_i16; input.len()];
        float_s16_to_s16_slice(input, &mut output);
        let expected: Vec<i16> = input.iter().map(|&v| float_s16_to_s16(v)).collect();
        assert_eq!(output, expected);
    }

    // ── Interleave / deinterleave ───────────────────────────────────

    #[test]
    fn interleaving_stereo() {
        let interleaved: &[i16] = &[2, 3, 4, 9, 8, 27, 16, 81];
        let samples_per_channel = 4;
        let num_channels = 2;

        let mut deint = vec![vec![0_i16; samples_per_channel]; num_channels];
        deinterleave(interleaved, &mut deint, samples_per_channel, num_channels);

        assert_eq!(&deint[0], &[2, 4, 8, 16]);
        assert_eq!(&deint[1], &[3, 9, 27, 81]);

        let refs: Vec<&[i16]> = deint.iter().map(|v| v.as_slice()).collect();
        let mut reinterleaved = vec![0_i16; interleaved.len()];
        interleave(&refs, &mut reinterleaved, samples_per_channel, num_channels);
        assert_eq!(&reinterleaved, interleaved);
    }

    #[test]
    fn interleaving_mono_is_identity() {
        let interleaved: &[i16] = &[1, 2, 3, 4, 5];
        let samples_per_channel = 5;
        let num_channels = 1;

        let mut deint = vec![vec![0_i16; samples_per_channel]; num_channels];
        deinterleave(interleaved, &mut deint, samples_per_channel, num_channels);
        assert_eq!(&deint[0], interleaved);

        let refs: Vec<&[i16]> = deint.iter().map(|v| v.as_slice()).collect();
        let mut reinterleaved = vec![0_i16; interleaved.len()];
        interleave(&refs, &mut reinterleaved, samples_per_channel, num_channels);
        assert_eq!(&reinterleaved, interleaved);
    }

    // ── Downmix ─────────────────────────────────────────────────────

    #[test]
    fn downmix_mono_is_identity() {
        let interleaved: &[i16] = &[1, 2, -1, -3];
        let mut mono = vec![0_i16; 4];
        downmix_interleaved_to_mono_i16(interleaved, 4, 1, &mut mono);
        assert_eq!(&mono, interleaved);
    }

    #[test]
    fn downmix_stereo() {
        let interleaved: &[i16] = &[10, 20, -10, -30];
        let mut mono = vec![0_i16; 2];
        downmix_interleaved_to_mono_i16(interleaved, 2, 2, &mut mono);
        assert_eq!(&mono, &[15, -20]);
    }

    #[test]
    fn downmix_three_channels() {
        let interleaved: &[i16] = &[30000, 30000, 24001, -5, -10, -20, -30000, -30999, -30000];
        let mut mono = vec![0_i16; 3];
        downmix_interleaved_to_mono_i16(interleaved, 3, 3, &mut mono);
        assert_eq!(&mono, &[28000, -11, -30333]);
    }

    // ── FloatToS16 ──────────────────────────────────────────────────

    #[test]
    fn float_to_s16_known_values() {
        let input: &[f32] = &[0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5];
        let output: Vec<i16> = input.iter().map(|&v| float_to_s16(v)).collect();
        // 0.0 -> 0, 1.0 -> clamped to 32767, -1.0 -> -32768
        // 0.5 -> 16384, -0.5 -> -16384, 1.5 -> clamped to 32767, -1.5 -> clamped to -32768
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 32767);
        assert_eq!(output[2], -32768);
        assert_eq!(output[3], 16384);
        assert_eq!(output[4], -16384);
        assert_eq!(output[5], 32767);
        assert_eq!(output[6], -32768);
    }

    // ── Additional slice conversion tests ───────────────────────────

    #[test]
    fn float_to_s16_slice_matches_scalar() {
        let input: &[f32] = &[0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = vec![0_i16; input.len()];
        float_to_s16_slice(input, &mut output);
        let expected: Vec<i16> = input.iter().map(|&v| float_to_s16(v)).collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn s16_to_float_s16_slice_widens() {
        let input: &[i16] = &[0, 1, -1, 32767, -32768];
        let mut output = vec![0.0_f32; input.len()];
        s16_to_float_s16_slice(input, &mut output);
        assert_eq!(output, &[0.0, 1.0, -1.0, 32767.0, -32768.0]);
    }

    #[test]
    fn float_to_float_s16_slice_matches_scalar() {
        let input: &[f32] = &[0.0, 0.5, -0.5, 1.0, -1.0];
        let mut output = vec![0.0_f32; input.len()];
        float_to_float_s16_slice(input, &mut output);
        let expected: Vec<f32> = input.iter().map(|&v| float_to_float_s16(v)).collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn float_s16_to_float_slice_matches_scalar() {
        let input: &[f32] = &[0.0, 16384.0, -16384.0, 32767.0, -32768.0];
        let mut output = vec![0.0_f32; input.len()];
        float_s16_to_float_slice(input, &mut output);
        let expected: Vec<f32> = input.iter().map(|&v| float_s16_to_float(v)).collect();
        assert_eq!(output, expected);
    }

    // ── Downmix f32 ─────────────────────────────────────────────────

    #[test]
    fn downmix_f32_mono_is_identity() {
        let interleaved: &[f32] = &[0.1, 0.2, -0.1, -0.3];
        let mut mono = vec![0.0_f32; 4];
        downmix_interleaved_to_mono_f32(interleaved, 4, 1, &mut mono);
        assert_eq!(&mono, interleaved);
    }

    #[test]
    fn downmix_f32_stereo() {
        let interleaved: &[f32] = &[0.2, 0.4, -0.6, -0.8];
        let mut mono = vec![0.0_f32; 2];
        downmix_interleaved_to_mono_f32(interleaved, 2, 2, &mut mono);
        assert!((mono[0] - 0.3).abs() < 1e-7);
        assert!((mono[1] - (-0.7)).abs() < 1e-7);
    }

    // ── S16 -> Float -> S16 roundtrip ───────────────────────────────

    #[test]
    fn s16_float_roundtrip() {
        // Converting S16 -> Float -> S16 should be lossless for most values
        for v in [-32768_i16, -16384, -1, 0, 1, 16384, 32767] {
            let f = s16_to_float(v);
            let back = float_to_s16(f);
            assert_eq!(v, back, "roundtrip failed for {v}");
        }
    }
}
