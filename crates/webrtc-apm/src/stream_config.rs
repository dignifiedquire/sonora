//! Stream configuration for audio processing.
//!
//! Ported from `api/audio/audio_processing.h` (StreamConfig class).

/// Configuration describing an audio stream's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    sample_rate_hz: usize,
    num_channels: usize,
    num_frames: usize,
}

impl StreamConfig {
    /// Create a new stream configuration.
    pub fn new(sample_rate_hz: usize, num_channels: usize) -> Self {
        Self {
            sample_rate_hz,
            num_channels,
            num_frames: sample_rate_hz / 100,
        }
    }

    /// The sampling rate in Hz.
    #[inline]
    pub fn sample_rate_hz(&self) -> usize {
        self.sample_rate_hz
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
