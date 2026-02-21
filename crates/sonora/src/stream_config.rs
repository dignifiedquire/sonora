//! Stream configuration for audio processing.
//!
//! Ported from `api/audio/audio_processing.h` (StreamConfig class).

use std::num::NonZeroU16;

/// Minimum supported sample rate in Hz.
pub const MIN_SAMPLE_RATE_HZ: u32 = 8_000;
/// Maximum supported sample rate in Hz.
pub const MAX_SAMPLE_RATE_HZ: u32 = 384_000;

/// Well-known sample rates used by the checked API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SampleRate {
    Hz8000 = 8_000,
    Hz16000 = 16_000,
    Hz32000 = 32_000,
    Hz48000 = 48_000,
}

impl SampleRate {
    /// Return this sample rate as an integer in Hz.
    pub const fn as_hz(self) -> u32 {
        self as u32
    }
}

/// Error returned when creating a [`CheckedStreamConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamConfigError {
    /// Sample rate is not one of the supported enum variants.
    UnsupportedSampleRate { sample_rate_hz: u32 },
}

impl std::fmt::Display for StreamConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UnsupportedSampleRate { sample_rate_hz } => write!(
                f,
                "unsupported sample rate {sample_rate_hz}; expected one of 8000, 16000, 32000, 48000",
            ),
        }
    }
}

impl std::error::Error for StreamConfigError {}

impl TryFrom<u32> for SampleRate {
    type Error = StreamConfigError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            8_000 => Ok(Self::Hz8000),
            16_000 => Ok(Self::Hz16000),
            32_000 => Ok(Self::Hz32000),
            48_000 => Ok(Self::Hz48000),
            sample_rate_hz => Err(StreamConfigError::UnsupportedSampleRate { sample_rate_hz }),
        }
    }
}

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
/// - sample rate must be one of [`SampleRate`] variants.
/// - channel count is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedStreamConfig {
    sample_rate: SampleRate,
    num_channels: NonZeroU16,
}

impl CheckedStreamConfig {
    /// Create a validated stream configuration.
    pub const fn new(sample_rate: SampleRate, num_channels: NonZeroU16) -> Self {
        Self {
            sample_rate,
            num_channels,
        }
    }

    /// Create a validated stream configuration from an integer sample rate.
    pub fn try_from_hz(
        sample_rate_hz: u32,
        num_channels: NonZeroU16,
    ) -> Result<Self, StreamConfigError> {
        Ok(Self::new(SampleRate::try_from(sample_rate_hz)?, num_channels))
    }

    /// The checked sample-rate enum.
    pub const fn sample_rate(self) -> SampleRate {
        self.sample_rate
    }

    /// The sampling rate in Hz.
    pub const fn sample_rate_hz(self) -> u32 {
        self.sample_rate.as_hz()
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
    fn sample_rate_try_from_hz_rejects_unsupported_rate() {
        let err = SampleRate::try_from(7_900).unwrap_err();
        assert_eq!(
            err,
            StreamConfigError::UnsupportedSampleRate {
                sample_rate_hz: 7_900,
            }
        );
    }

    #[test]
    fn checked_stream_config_try_from_hz_rejects_unsupported_rate() {
        let err = CheckedStreamConfig::try_from_hz(44_100, NonZeroU16::new(1).unwrap()).unwrap_err();
        assert_eq!(
            err,
            StreamConfigError::UnsupportedSampleRate {
                sample_rate_hz: 44_100,
            }
        );
    }

    #[test]
    fn checked_stream_config_accepts_valid_values() {
        let checked = CheckedStreamConfig::new(SampleRate::Hz48000, NonZeroU16::new(2).unwrap());
        assert_eq!(checked.sample_rate(), SampleRate::Hz48000);
        assert_eq!(checked.sample_rate_hz(), 48_000);
        assert_eq!(checked.num_channels().get(), 2);

        let legacy: StreamConfig = checked.into();
        assert_eq!(legacy.sample_rate_hz(), 48_000);
        assert_eq!(legacy.num_channels(), 2);
        assert_eq!(legacy.num_frames(), 480);
        assert_eq!(legacy.num_samples(), 960);
    }
}
