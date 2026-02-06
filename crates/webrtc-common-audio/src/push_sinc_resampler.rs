//! Push-based wrapper around [`SincResampler`].
//!
//! Port of WebRTC's `PushSincResampler` — converts the pull-based
//! [`SincResampler`] into a push API suitable for streaming audio.
//!
//! # How it works
//!
//! The caller provides a complete input buffer and gets resampled output.
//! Internally, the first call performs a priming pass to fill the sinc
//! kernel delay line, ensuring subsequent calls produce correct output
//! with minimal latency (half-kernel delay rather than full-frame delay).
//!
//! # Example
//!
//! ```
//! use webrtc_common_audio::push_sinc_resampler::PushSincResampler;
//!
//! // 48kHz → 16kHz, 10ms frames
//! let mut resampler = PushSincResampler::new(480, 160);
//! let input = vec![0.0_f32; 480];
//! let mut output = vec![0.0_f32; 160];
//! resampler.resample(&input, &mut output);
//! ```

use std::mem;

use crate::audio_util;
use crate::sinc_resampler::{KERNEL_SIZE, SincResampler, SincResamplerCallback};

/// A [`SincResamplerCallback`] that reads from a slice, providing silence
/// on the first call to prime the delay line.
struct SliceCallback<'a> {
    source: &'a [f32],
    first_pass: bool,
}

impl SincResamplerCallback for SliceCallback<'_> {
    fn run(&mut self, frames: usize, destination: &mut [f32]) {
        if self.first_pass {
            destination[..frames].fill(0.0);
            self.first_pass = false;
            return;
        }
        assert_eq!(
            self.source.len(),
            frames,
            "callback requested {frames} frames but source has {}",
            self.source.len()
        );
        destination[..frames].copy_from_slice(&self.source[..frames]);
    }
}

/// Push-based single-channel resampler.
///
/// Wraps [`SincResampler`] with automatic priming and a push-style API.
#[derive(Debug)]
pub struct PushSincResampler {
    resampler: SincResampler,
    float_buffer: Vec<f32>,
    destination_frames: usize,
    first_pass: bool,
}

impl PushSincResampler {
    /// Create a new push resampler.
    ///
    /// - `source_frames`: number of input frames per call (e.g. 480 for 10ms at 48kHz)
    /// - `destination_frames`: number of output frames per call
    ///
    /// The sample rate ratio is inferred from `source_frames / destination_frames`.
    pub fn new(source_frames: usize, destination_frames: usize) -> Self {
        assert!(source_frames > 0);
        assert!(destination_frames > 0);
        let ratio = source_frames as f64 / destination_frames as f64;
        Self {
            resampler: SincResampler::new(ratio, source_frames),
            float_buffer: Vec::new(),
            destination_frames,
            first_pass: true,
        }
    }

    /// Resample `f32` audio. Returns the number of output frames written
    /// (always equal to `destination_frames`).
    ///
    /// # Panics
    ///
    /// - `source.len()` must equal the `source_frames` given at construction.
    /// - `destination.len()` must be at least `destination_frames`.
    pub fn resample(&mut self, source: &[f32], destination: &mut [f32]) -> usize {
        assert_eq!(source.len(), self.resampler.request_frames());
        assert!(destination.len() >= self.destination_frames);

        let mut cb = SliceCallback {
            source,
            first_pass: self.first_pass,
        };

        // On the first pass, prime the SincResampler buffer so that all later
        // calls result in exactly one callback. ChunkSize() is the exact
        // output count that triggers a single Run() for `source_frames` input.
        if self.first_pass {
            let chunk = self.resampler.chunk_size();
            self.resampler.resample(chunk, destination, &mut cb);
        }

        self.resampler
            .resample(self.destination_frames, destination, &mut cb);

        if self.first_pass {
            self.first_pass = false;
        }

        self.destination_frames
    }

    /// Resample `i16` audio. Internally converts to float, resamples, and
    /// converts back. Returns the number of output frames written.
    ///
    /// # Panics
    ///
    /// - `source.len()` must equal the `source_frames` given at construction.
    /// - `destination.len()` must be at least `destination_frames`.
    pub fn resample_i16(&mut self, source: &[i16], destination: &mut [i16]) -> usize {
        assert_eq!(source.len(), self.resampler.request_frames());
        assert!(destination.len() >= self.destination_frames);

        // Convert i16 → f32 for the sinc resampler.
        let float_source: Vec<f32> = source.iter().map(|&s| s as f32).collect();

        // Take float_buffer out to avoid double-borrow of self.
        let mut float_buf = mem::take(&mut self.float_buffer);
        if float_buf.len() < self.destination_frames {
            float_buf.resize(self.destination_frames, 0.0);
        }

        self.resample(&float_source, &mut float_buf);

        audio_util::float_s16_to_s16_slice(
            &float_buf[..self.destination_frames],
            &mut destination[..self.destination_frames],
        );

        // Put it back.
        self.float_buffer = float_buf;
        self.destination_frames
    }

    /// Algorithmic delay in seconds due to the sinc kernel.
    pub fn algorithmic_delay_seconds(source_rate_hz: u32) -> f32 {
        1.0 / source_rate_hz as f32 * KERNEL_SIZE as f32 / 2.0
    }

    /// Number of destination frames per call.
    pub fn destination_frames(&self) -> usize {
        self.destination_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passthrough() {
        let frames = 480;
        let mut r = PushSincResampler::new(frames, frames);
        let input: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0_f32; frames];

        // Feed a few blocks to let the filter settle after priming.
        for _ in 0..3 {
            let n = r.resample(&input, &mut output);
            assert_eq!(n, frames);
        }

        // Identity resampler introduces half-kernel delay (16 samples).
        // Verify the output reproduces the input waveform with a fixed delay.
        // Since we feed the same (non-periodic) block repeatedly, the output
        // will have a discontinuity at frame boundaries — that's expected.
        // Instead, check that the middle section tracks the input shape.
        let delay = 16_usize; // half kernel
        let check_start = delay;
        let check_end = frames - delay;
        let max_diff = output[check_start..check_end]
            .iter()
            .zip(input[..check_end - check_start].iter())
            .map(|(o, i)| (o - i).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 0.1,
            "output should track delayed input, max_diff={max_diff}"
        );
    }

    #[test]
    fn downsample_48k_to_16k() {
        let src_frames = 480; // 10ms at 48kHz
        let dst_frames = 160; // 10ms at 16kHz
        let mut r = PushSincResampler::new(src_frames, dst_frames);

        let input: Vec<f32> = (0..src_frames).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut output = vec![0.0_f32; dst_frames];
        for _ in 0..3 {
            let n = r.resample(&input, &mut output);
            assert_eq!(n, dst_frames);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01, "output should have signal energy");
    }

    #[test]
    fn upsample_16k_to_48k() {
        let src_frames = 160;
        let dst_frames = 480;
        let mut r = PushSincResampler::new(src_frames, dst_frames);

        let input: Vec<f32> = (0..src_frames).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut output = vec![0.0_f32; dst_frames];
        for _ in 0..3 {
            let n = r.resample(&input, &mut output);
            assert_eq!(n, dst_frames);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01, "output should have signal energy");
    }

    #[test]
    fn i16_resample() {
        let src_frames = 480;
        let dst_frames = 160;
        let mut r = PushSincResampler::new(src_frames, dst_frames);

        let input: Vec<i16> = (0..src_frames)
            .map(|i| (1000.0 * (i as f32 * 0.02).sin()) as i16)
            .collect();
        let mut output = vec![0_i16; dst_frames];
        for _ in 0..3 {
            let n = r.resample_i16(&input, &mut output);
            assert_eq!(n, dst_frames);
        }
        let has_signal = output.iter().any(|&v| v != 0);
        assert!(has_signal, "i16 resampler should produce output");
    }

    #[test]
    fn algorithmic_delay() {
        let delay = PushSincResampler::algorithmic_delay_seconds(48000);
        // KERNEL_SIZE/2 = 16 samples at 48kHz ≈ 0.000333s
        assert!((delay - 16.0 / 48000.0).abs() < 1e-6);
    }
}
