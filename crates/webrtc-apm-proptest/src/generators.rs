//! Audio buffer generators for property-based testing.
//!
//! Provides both strategy functions (for use with `#[strategy(...)]`) and
//! `Arbitrary`-deriving structs for common audio test inputs.

use proptest::prelude::*;
use test_strategy::Arbitrary;

/// A valid WebRTC sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Arbitrary)]
pub enum SampleRate {
    #[weight(1)]
    Hz8000,
    #[weight(1)]
    Hz16000,
    #[weight(1)]
    Hz32000,
    #[weight(1)]
    Hz48000,
}

impl SampleRate {
    pub fn hz(self) -> u32 {
        match self {
            Self::Hz8000 => 8000,
            Self::Hz16000 => 16000,
            Self::Hz32000 => 32000,
            Self::Hz48000 => 48000,
        }
    }

    /// Number of samples in a 10ms frame at this rate.
    pub fn frame_size(self) -> usize {
        (self.hz() / 100) as usize
    }
}

/// A valid channel count for WebRTC audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Arbitrary)]
pub enum ChannelCount {
    #[weight(1)]
    Mono,
    #[weight(1)]
    Stereo,
}

impl ChannelCount {
    pub fn count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// A mono f32 audio frame with its sample rate.
#[derive(Debug, Clone, Arbitrary)]
pub struct MonoFrameF32 {
    pub sample_rate: SampleRate,
    #[strategy(audio_frame_f32(#sample_rate.hz()))]
    pub samples: Vec<f32>,
}

/// A mono i16 audio frame with its sample rate.
#[derive(Debug, Clone, Arbitrary)]
pub struct MonoFrameI16 {
    pub sample_rate: SampleRate,
    #[strategy(audio_frame_i16(#sample_rate.hz()))]
    pub samples: Vec<i16>,
}

/// An interleaved multi-channel f32 audio frame.
#[derive(Debug, Clone, Arbitrary)]
pub struct MultiChannelFrameF32 {
    pub sample_rate: SampleRate,
    pub channels: ChannelCount,
    #[strategy(audio_frame_multichannel_f32(#sample_rate.hz(), #channels.count()))]
    pub samples: Vec<f32>,
}

/// Generate a mono audio frame at a given sample rate (~10ms frame).
pub fn audio_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate interleaved stereo audio (~10ms frame).
pub fn stereo_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize * 2;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate a mono i16 audio frame at a given sample rate (~10ms frame).
pub fn audio_frame_i16(sample_rate: u32) -> impl Strategy<Value = Vec<i16>> {
    let frame_size = (sample_rate / 100) as usize;
    proptest::collection::vec(i16::MIN..=i16::MAX, frame_size..=frame_size)
}

/// Generate multi-channel audio frames (~10ms).
pub fn audio_frame_multichannel_f32(
    sample_rate: u32,
    channels: usize,
) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize * channels;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate FIR filter coefficients.
pub fn fir_coefficients(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..=1.0f32, 1..=max_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn frame_f32_correct_length(#[strategy(audio_frame_f32(16000))] frame: Vec<f32>) {
        assert_eq!(frame.len(), 160);
        for &s in &frame {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[proptest]
    fn frame_f32_48k_length(#[strategy(audio_frame_f32(48000))] frame: Vec<f32>) {
        assert_eq!(frame.len(), 480);
    }

    #[proptest]
    fn frame_i16_correct_length(#[strategy(audio_frame_i16(8000))] frame: Vec<i16>) {
        assert_eq!(frame.len(), 80);
    }

    #[proptest]
    fn stereo_frame_correct_length(#[strategy(stereo_frame_f32(16000))] frame: Vec<f32>) {
        assert_eq!(frame.len(), 320);
    }

    #[proptest]
    fn mono_frame_struct_consistent(frame: MonoFrameF32) {
        assert_eq!(frame.samples.len(), frame.sample_rate.frame_size());
        for &s in &frame.samples {
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[proptest]
    fn mono_frame_i16_struct_consistent(frame: MonoFrameI16) {
        assert_eq!(frame.samples.len(), frame.sample_rate.frame_size());
    }

    #[proptest]
    fn multichannel_frame_struct_consistent(frame: MultiChannelFrameF32) {
        let expected = frame.sample_rate.frame_size() * frame.channels.count();
        assert_eq!(frame.samples.len(), expected);
    }

    #[proptest]
    fn sample_rate_valid_values(sr: SampleRate) {
        assert!([8000, 16000, 32000, 48000].contains(&sr.hz()));
    }

    #[proptest]
    fn fir_coefficients_valid_range(#[strategy(fir_coefficients(64))] coeffs: Vec<f32>) {
        assert!(!coeffs.is_empty());
        assert!(coeffs.len() <= 64);
        for &c in &coeffs {
            assert!((-1.0..=1.0).contains(&c));
        }
    }
}
