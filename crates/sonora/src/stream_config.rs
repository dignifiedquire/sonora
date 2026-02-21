//! Stream configuration for audio processing.
//!
//! Ported from `api/audio/audio_processing.h` (StreamConfig class).

use std::num::{NonZeroU16, NonZeroU32};

/// Minimum supported sample rate in Hz.
pub const MIN_SAMPLE_RATE_HZ: u32 = 8_000;
/// Maximum supported sample rate in Hz.
pub const MAX_SAMPLE_RATE_HZ: u32 = 384_000;

/// Error returned when creating a [`CheckedStreamConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamConfigError {
    /// Sample rate is outside the supported range.
    UnsupportedSampleRate { sample_rate_hz: u32 },
    /// Sample rate does not map to an integer number of 10ms frames.
    Non10msAlignedSampleRate { sample_rate_hz: u32 },
}

impl std::fmt::Display for StreamConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UnsupportedSampleRate { sample_rate_hz } => write!(
                f,
                "unsupported sample rate {sample_rate_hz}; expected {MIN_SAMPLE_RATE_HZ}..={MAX_SAMPLE_RATE_HZ}",
            ),
            Self::Non10msAlignedSampleRate { sample_rate_hz } => write!(
                f,
                "sample rate {sample_rate_hz} is not aligned to 10ms frames (must be divisible by 100)",
            ),
        }
    }
}

impl std::error::Error for StreamConfigError {}

/// Configuration describing an audio stream's properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    sample_rate_hz: u32,
    num_channels: u16,
}

impl StreamConfig {
    /// Create a new stream configuration.
    ///
    /// This constructor is intentionally unchecked for backward compatibility.
    /// For a type-safe API that prevents invalid values, use
    /// [`CheckedStreamConfig::new`].
    pub const fn new(sample_rate_hz: u32, num_channels: u16) -> Self {
        Self {
            sample_rate_hz,
            num_channels,
        }
    }

    /// The sampling rate in Hz.
    #[inline]
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// The number of channels.
    #[inline]
    pub fn num_channels(&self) -> u16 {
        self.num_channels
    }

    /// The number of frames per 10ms chunk.
    #[inline]
    pub fn num_frames(&self) -> usize {
        self.sample_rate_hz as usize / 100
    }

    /// Total number of samples (channels × frames).
    #[inline]
    pub fn num_samples(&self) -> usize {
        self.num_channels as usize * self.num_frames()
    }
}

/// Validated stream configuration that makes common invalid states impossible.
///
/// Invariants:
/// - `sample_rate_hz` is in `8000..=384000`.
/// - `sample_rate_hz` is divisible by `100` (exact 10ms frames).
/// - `num_channels` is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedStreamConfig {
    sample_rate_hz: NonZeroU32,
    num_channels: NonZeroU16,
}

impl CheckedStreamConfig {
    /// Create a validated stream configuration.
    pub fn new(
        sample_rate_hz: u32,
        num_channels: NonZeroU16,
    ) -> Result<Self, StreamConfigError> {
        if !(MIN_SAMPLE_RATE_HZ..=MAX_SAMPLE_RATE_HZ).contains(&sample_rate_hz) {
            return Err(StreamConfigError::UnsupportedSampleRate { sample_rate_hz });
        }
        if sample_rate_hz % 100 != 0 {
            return Err(StreamConfigError::Non10msAlignedSampleRate { sample_rate_hz });
        }
        Ok(Self {
            sample_rate_hz: NonZeroU32::new(sample_rate_hz)
                .expect("sample rate range validation guarantees non-zero"),
            num_channels,
        })
    }

    /// The sampling rate in Hz.
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate_hz.get()
    }

    /// The non-zero number of channels.
    pub const fn num_channels(self) -> NonZeroU16 {
        self.num_channels
    }

    /// Convert to legacy [`StreamConfig`].
    pub const fn into_stream_config(self) -> StreamConfig {
        StreamConfig::new(self.sample_rate_hz(), self.num_channels.get())
    }
}

impl From<CheckedStreamConfig> for StreamConfig {
    fn from(value: CheckedStreamConfig) -> Self {
        value.into_stream_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_stream_config_rejects_unsupported_sample_rate() {
        let err = CheckedStreamConfig::new(7_900, NonZeroU16::new(1).unwrap()).unwrap_err();
        assert_eq!(
            err,
            StreamConfigError::UnsupportedSampleRate {
                sample_rate_hz: 7_900,
            }
        );
    }

    #[test]
    fn checked_stream_config_rejects_non_10ms_aligned_rate() {
        let err = CheckedStreamConfig::new(44_101, NonZeroU16::new(1).unwrap()).unwrap_err();
        assert_eq!(
            err,
            StreamConfigError::Non10msAlignedSampleRate {
                sample_rate_hz: 44_101,
            }
        );
    }

    #[test]
    fn checked_stream_config_accepts_valid_values() {
        let checked = CheckedStreamConfig::new(48_000, NonZeroU16::new(2).unwrap()).unwrap();
        assert_eq!(checked.sample_rate_hz(), 48_000);
        assert_eq!(checked.num_channels().get(), 2);

        let legacy: StreamConfig = checked.into();
        assert_eq!(legacy.sample_rate_hz(), 48_000);
        assert_eq!(legacy.num_channels(), 2);
        assert_eq!(legacy.num_frames(), 480);
        assert_eq!(legacy.num_samples(), 960);
    }
}
