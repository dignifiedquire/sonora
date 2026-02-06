//! Multi-channel push-based audio resampler.
//!
//! Port of WebRTC's `PushResampler` — wraps [`PushSincResampler`] to handle
//! multi-channel (interleaved) audio. Each channel gets its own resampler
//! instance.
//!
//! # Usage
//!
//! ```
//! use webrtc_common_audio::push_resampler::PushResampler;
//!
//! // 48kHz stereo → 16kHz stereo (10ms frames)
//! let mut resampler = PushResampler::<f32>::new(480, 160, 2);
//! let input = vec![0.0_f32; 480 * 2]; // interleaved stereo
//! let mut output = vec![0.0_f32; 160 * 2];
//! resampler.resample(&input, &mut output);
//! ```

use crate::audio_util;
use crate::push_sinc_resampler::PushSincResampler;

/// Maximum number of channels supported.
const MAX_NUMBER_OF_CHANNELS: usize = 8;

/// Trait for sample types supported by `PushResampler`.
pub trait Sample: Copy + Default + 'static {
    /// Resample a single channel using the given `PushSincResampler`.
    fn resample_channel(
        resampler: &mut PushSincResampler,
        source: &[Self],
        destination: &mut [Self],
    ) -> usize;
}

impl Sample for f32 {
    fn resample_channel(
        resampler: &mut PushSincResampler,
        source: &[Self],
        destination: &mut [Self],
    ) -> usize {
        resampler.resample(source, destination)
    }
}

impl Sample for i16 {
    fn resample_channel(
        resampler: &mut PushSincResampler,
        source: &[Self],
        destination: &mut [Self],
    ) -> usize {
        resampler.resample_i16(source, destination)
    }
}

/// Multi-channel push-based resampler.
///
/// Handles deinterleaving, per-channel resampling, and re-interleaving.
/// For mono signals, the deinterleave/interleave steps are skipped.
#[derive(Debug)]
pub struct PushResampler<T: Sample> {
    resamplers: Vec<PushSincResampler>,
    source_buf: Vec<T>,
    dest_buf: Vec<T>,
    src_samples_per_channel: usize,
    dst_samples_per_channel: usize,
    num_channels: usize,
}

impl<T: Sample> PushResampler<T> {
    /// Create a new multi-channel push resampler.
    ///
    /// - `src_samples_per_channel`: input frames per channel (e.g. 480 for 10ms at 48kHz)
    /// - `dst_samples_per_channel`: output frames per channel
    /// - `num_channels`: number of audio channels (1..=8)
    pub fn new(
        src_samples_per_channel: usize,
        dst_samples_per_channel: usize,
        num_channels: usize,
    ) -> Self {
        assert!(src_samples_per_channel > 0);
        assert!(dst_samples_per_channel > 0);
        assert!(num_channels > 0);
        assert!(
            num_channels <= MAX_NUMBER_OF_CHANNELS,
            "max {MAX_NUMBER_OF_CHANNELS} channels supported"
        );
        assert!(
            src_samples_per_channel <= audio_util::MAX_SAMPLES_PER_CHANNEL_10MS,
            "source frames exceed maximum"
        );
        assert!(
            dst_samples_per_channel <= audio_util::MAX_SAMPLES_PER_CHANNEL_10MS,
            "destination frames exceed maximum"
        );

        let resamplers = (0..num_channels)
            .map(|_| PushSincResampler::new(src_samples_per_channel, dst_samples_per_channel))
            .collect();

        Self {
            resamplers,
            source_buf: vec![T::default(); src_samples_per_channel * num_channels],
            dest_buf: vec![T::default(); dst_samples_per_channel * num_channels],
            src_samples_per_channel,
            dst_samples_per_channel,
            num_channels,
        }
    }

    /// Resample interleaved audio.
    ///
    /// - `src`: interleaved input, length must be `src_samples_per_channel * num_channels`
    /// - `dst`: interleaved output, length must be at least `dst_samples_per_channel * num_channels`
    pub fn resample(&mut self, src: &[T], dst: &mut [T]) {
        let expected_src_len = self.src_samples_per_channel * self.num_channels;
        let expected_dst_len = self.dst_samples_per_channel * self.num_channels;
        assert_eq!(
            src.len(),
            expected_src_len,
            "source length must be {} (got {})",
            expected_src_len,
            src.len()
        );
        assert!(
            dst.len() >= expected_dst_len,
            "destination length must be at least {} (got {})",
            expected_dst_len,
            dst.len()
        );

        // Fast path: matching rates just copies.
        if self.src_samples_per_channel == self.dst_samples_per_channel {
            dst[..expected_dst_len].copy_from_slice(&src[..expected_src_len]);
            return;
        }

        if self.num_channels == 1 {
            // Mono: skip deinterleave/interleave.
            T::resample_channel(&mut self.resamplers[0], src, dst);
            return;
        }

        // Deinterleave into source_buf (channel-planar layout).
        deinterleave_to_planar(
            src,
            &mut self.source_buf,
            self.src_samples_per_channel,
            self.num_channels,
        );

        // Resample each channel.
        for ch in 0..self.num_channels {
            let src_start = ch * self.src_samples_per_channel;
            let src_end = src_start + self.src_samples_per_channel;
            let dst_start = ch * self.dst_samples_per_channel;
            let dst_end = dst_start + self.dst_samples_per_channel;

            // We need non-overlapping borrows, so split the dest buffer.
            let src_slice = &self.source_buf[src_start..src_end];

            // Copy source to a temp so we can borrow dest_buf mutably.
            // (source_buf and dest_buf are separate fields, so this is fine.)
            let n = T::resample_channel(
                &mut self.resamplers[ch],
                src_slice,
                &mut self.dest_buf[dst_start..dst_end],
            );
            debug_assert_eq!(n, self.dst_samples_per_channel);
        }

        // Re-interleave from dest_buf to output.
        interleave_from_planar(
            &self.dest_buf,
            dst,
            self.dst_samples_per_channel,
            self.num_channels,
        );
    }

    /// Resample a single (mono) channel without interleaving.
    ///
    /// # Panics
    ///
    /// Panics if the resampler was created with more than 1 channel.
    pub fn resample_mono(&mut self, src: &[T], dst: &mut [T]) {
        assert_eq!(self.num_channels, 1, "resample_mono requires 1 channel");
        assert_eq!(src.len(), self.src_samples_per_channel);
        assert!(dst.len() >= self.dst_samples_per_channel);

        if self.src_samples_per_channel == self.dst_samples_per_channel {
            dst[..self.dst_samples_per_channel]
                .copy_from_slice(&src[..self.src_samples_per_channel]);
        } else {
            T::resample_channel(&mut self.resamplers[0], src, dst);
        }
    }
}

/// Deinterleave from interleaved layout to channel-planar layout.
///
/// `planar` layout: `[ch0_frame0..ch0_frameN | ch1_frame0..ch1_frameN | ...]`
fn deinterleave_to_planar<T: Copy>(
    interleaved: &[T],
    planar: &mut [T],
    frames_per_channel: usize,
    num_channels: usize,
) {
    for ch in 0..num_channels {
        let dst_start = ch * frames_per_channel;
        for frame in 0..frames_per_channel {
            planar[dst_start + frame] = interleaved[frame * num_channels + ch];
        }
    }
}

/// Interleave from channel-planar layout to interleaved layout.
fn interleave_from_planar<T: Copy>(
    planar: &[T],
    interleaved: &mut [T],
    frames_per_channel: usize,
    num_channels: usize,
) {
    for ch in 0..num_channels {
        let src_start = ch * frames_per_channel;
        for frame in 0..frames_per_channel {
            interleaved[frame * num_channels + ch] = planar[src_start + frame];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_identity() {
        let frames = 480;
        let mut r = PushResampler::<f32>::new(frames, frames, 1);
        let input: Vec<f32> = (0..frames).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0_f32; frames];
        r.resample(&input, &mut output);
        // Same rate = copy.
        assert_eq!(&input, &output);
    }

    #[test]
    fn stereo_identity() {
        let frames = 480;
        let mut r = PushResampler::<f32>::new(frames, frames, 2);
        let input: Vec<f32> = (0..frames * 2).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0_f32; frames * 2];
        r.resample(&input, &mut output);
        assert_eq!(&input, &output);
    }

    #[test]
    fn mono_downsample() {
        let mut r = PushResampler::<f32>::new(480, 160, 1);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut output = vec![0.0_f32; 160];
        // Run a few blocks to settle.
        for _ in 0..3 {
            r.resample(&input, &mut output);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01, "output should have signal energy");
    }

    #[test]
    fn stereo_downsample() {
        let mut r = PushResampler::<f32>::new(480, 160, 2);
        // Interleaved stereo: L=sine, R=cosine
        let input: Vec<f32> = (0..480)
            .flat_map(|i| {
                let t = i as f32 * 0.02;
                [t.sin(), t.cos()]
            })
            .collect();
        let mut output = vec![0.0_f32; 160 * 2];
        for _ in 0..3 {
            r.resample(&input, &mut output);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01, "stereo output should have signal energy");
    }

    #[test]
    fn i16_mono_downsample() {
        let mut r = PushResampler::<i16>::new(480, 160, 1);
        let input: Vec<i16> = (0..480)
            .map(|i| (5000.0 * (i as f32 * 0.02).sin()) as i16)
            .collect();
        let mut output = vec![0_i16; 160];
        for _ in 0..3 {
            r.resample(&input, &mut output);
        }
        let has_signal = output.iter().any(|&v| v.unsigned_abs() > 10);
        assert!(has_signal, "i16 output should have signal");
    }

    #[test]
    fn mono_upsample() {
        let mut r = PushResampler::<f32>::new(160, 480, 1);
        let input: Vec<f32> = (0..160).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut output = vec![0.0_f32; 480];
        for _ in 0..3 {
            r.resample(&input, &mut output);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01, "upsampled output should have signal energy");
    }

    #[test]
    fn resample_mono_api() {
        let mut r = PushResampler::<f32>::new(480, 160, 1);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut output = vec![0.0_f32; 160];
        for _ in 0..3 {
            r.resample_mono(&input, &mut output);
        }
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(energy > 0.01);
    }

    #[test]
    fn deinterleave_interleave_roundtrip() {
        let frames = 4;
        let channels = 3;
        let interleaved: Vec<f32> = (0..frames * channels).map(|i| i as f32).collect();
        let mut planar = vec![0.0_f32; frames * channels];
        let mut result = vec![0.0_f32; frames * channels];

        deinterleave_to_planar(&interleaved, &mut planar, frames, channels);
        // Check planar layout: ch0=[0,3,6,9], ch1=[1,4,7,10], ch2=[2,5,8,11]
        assert_eq!(&planar[0..4], &[0.0, 3.0, 6.0, 9.0]);
        assert_eq!(&planar[4..8], &[1.0, 4.0, 7.0, 10.0]);
        assert_eq!(&planar[8..12], &[2.0, 5.0, 8.0, 11.0]);

        interleave_from_planar(&planar, &mut result, frames, channels);
        assert_eq!(&interleaved, &result);
    }
}
