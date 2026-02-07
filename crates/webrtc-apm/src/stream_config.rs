//! Stream configuration for audio processing.
//!
//! Ported from `api/audio/audio_processing.h` (StreamConfig class).

/// Configuration describing an audio stream's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    sample_rate_hz: usize,
    num_channels: usize,
    num_frames: usize,
    /// The original signed sample rate, preserved for format validation.
    /// Negative rates indicate an invalid/uninterpretable format.
    sample_rate_hz_signed: i32,
}

impl StreamConfig {
    /// Create a new stream configuration.
    pub fn new(sample_rate_hz: usize, num_channels: usize) -> Self {
        Self {
            sample_rate_hz,
            num_channels,
            num_frames: sample_rate_hz / 100,
            sample_rate_hz_signed: sample_rate_hz as i32,
        }
    }

    /// Create a new stream configuration from a signed sample rate.
    ///
    /// Negative rates are preserved for format validation but treated
    /// as zero for frame calculations.
    pub fn from_signed(sample_rate_hz: i32, num_channels: usize) -> Self {
        let rate_usize = if sample_rate_hz < 0 {
            0
        } else {
            sample_rate_hz as usize
        };
        Self {
            sample_rate_hz: rate_usize,
            num_channels,
            num_frames: rate_usize / 100,
            sample_rate_hz_signed: sample_rate_hz,
        }
    }

    /// The sampling rate in Hz.
    #[inline]
    pub fn sample_rate_hz(&self) -> usize {
        self.sample_rate_hz
    }

    /// The original signed sampling rate in Hz.
    ///
    /// Negative values indicate an invalid format.
    #[inline]
    pub fn sample_rate_hz_signed(&self) -> i32 {
        self.sample_rate_hz_signed
    }

    /// The number of channels.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// The number of frames per 10ms chunk.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Total number of samples (channels × frames).
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.num_channels * self.num_frames
    }
}
