//! Audio processing configuration.
//!
//! Ported from `AudioProcessing::Config` in `api/audio/audio_processing.h`.

/// Top-level configuration for the audio processing pipeline.
///
/// This config is intended to be used during setup, and to enable/disable
/// top-level processing effects. Use during processing may cause undesired
/// submodule resets, affecting audio quality. Use [`RuntimeSetting`] for
/// runtime configuration changes.
///
/// All components are disabled (`None`) by default. Setting a component to
/// `Some(...)` enables it and triggers memory allocation and initialization.
///
/// # Example
///
/// ```
/// use sonora::Config;
/// use sonora::config::{EchoCanceller, NoiseSuppression, NoiseSuppressionLevel};
///
/// let config = Config {
///     echo_canceller: Some(EchoCanceller::default()),
///     noise_suppression: Some(NoiseSuppression {
///         level: NoiseSuppressionLevel::High,
///         ..Default::default()
///     }),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Pipeline processing properties.
    pub pipeline: Pipeline,
    /// Pre-amplifier settings. Amplifies the capture signal before any other
    /// processing. Set to `Some(...)` to enable.
    pub pre_amplifier: Option<PreAmplifier>,
    /// Capture-level adjustment settings. Should not be used together with
    /// [`PreAmplifier`]. Set to `Some(...)` to enable.
    pub capture_level_adjustment: Option<CaptureLevelAdjustment>,
    /// High-pass filter settings. Set to `Some(...)` to enable.
    pub high_pass_filter: Option<HighPassFilter>,
    /// Echo canceller (AEC3) settings. Set to `Some(...)` to enable.
    pub echo_canceller: Option<EchoCanceller>,
    /// Noise suppression settings. Set to `Some(...)` to enable.
    pub noise_suppression: Option<NoiseSuppression>,
    /// Automatic Gain Controller 2 (AGC2) settings. Combines input volume
    /// control, adaptive digital gain, fixed digital gain, and a limiter.
    /// Set to `Some(...)` to enable.
    pub gain_controller2: Option<GainController2>,
}

/// Maximum internal processing rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxProcessingRate {
    /// 32 kHz internal processing rate.
    Rate32kHz,
    /// 48 kHz internal processing rate.
    Rate48kHz,
}

impl MaxProcessingRate {
    /// Returns the rate in Hz.
    pub(crate) fn as_hz(self) -> u32 {
        match self {
            Self::Rate32kHz => 32000,
            Self::Rate48kHz => 48000,
        }
    }
}

/// Pipeline processing properties.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Maximum allowed processing rate used internally.
    pub maximum_internal_processing_rate: MaxProcessingRate,
    /// Allow multi-channel processing of render audio.
    pub multi_channel_render: bool,
    /// Allow multi-channel processing of capture audio when AEC3 is active.
    pub multi_channel_capture: bool,
    /// How to downmix multi-channel capture audio to mono.
    pub capture_downmix_method: DownmixMethod,
}

/// Ways to downmix a multi-channel track to mono.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownmixMethod {
    /// Average across channels.
    AverageChannels,
    /// Use the first channel.
    UseFirstChannel,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            maximum_internal_processing_rate: MaxProcessingRate::Rate32kHz,
            multi_channel_render: false,
            multi_channel_capture: false,
            capture_downmix_method: DownmixMethod::AverageChannels,
        }
    }
}

/// Pre-amplifier settings. Amplifies the capture signal before any other
/// processing.
#[derive(Debug, Clone)]
pub struct PreAmplifier {
    /// Linear gain factor applied to the capture signal (default: 1.0).
    pub fixed_gain_factor: f32,
}

impl Default for PreAmplifier {
    fn default() -> Self {
        Self {
            fixed_gain_factor: 1.0,
        }
    }
}

/// General level adjustment in the capture pipeline. Should not be used
/// together with the legacy [`PreAmplifier`].
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureLevelAdjustment {
    /// Linear gain factor applied before any processing (default: 1.0).
    pub pre_gain_factor: f32,
    /// Linear gain factor applied after all processing (default: 1.0).
    pub post_gain_factor: f32,
    /// Analog mic gain emulation settings. Set to `Some(...)` to enable.
    pub analog_mic_gain_emulation: Option<AnalogMicGainEmulation>,
}

impl Default for CaptureLevelAdjustment {
    fn default() -> Self {
        Self {
            pre_gain_factor: 1.0,
            post_gain_factor: 1.0,
            analog_mic_gain_emulation: None,
        }
    }
}

/// Analog microphone gain emulation settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalogMicGainEmulation {
    /// Initial analog gain level to use for the emulated analog gain.
    /// Range: `0..=255` (default: 255).
    pub initial_level: u8,
}

impl Default for AnalogMicGainEmulation {
    fn default() -> Self {
        Self { initial_level: 255 }
    }
}

/// High-pass filter settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighPassFilter {
    /// When true, the filter operates on the full-band signal rather than
    /// only the split band (default: true).
    pub apply_in_full_band: bool,
}

impl Default for HighPassFilter {
    fn default() -> Self {
        Self {
            apply_in_full_band: true,
        }
    }
}

/// Echo canceller (AEC3) settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoCanceller {
    /// Enforce the highpass filter to be on (default: true). Has no effect
    /// in mobile mode.
    pub enforce_high_pass_filtering: bool,
}

impl Default for EchoCanceller {
    fn default() -> Self {
        Self {
            enforce_high_pass_filtering: true,
        }
    }
}

/// Background noise suppression settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseSuppression {
    /// Aggressiveness level for noise suppression (default: `Moderate`).
    pub level: NoiseSuppressionLevel,
    /// When true and linear AEC output is available, noise suppression
    /// analyzes the linear AEC output instead of the regular signal.
    pub analyze_linear_aec_output_when_available: bool,
}

/// Noise suppression aggressiveness level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSuppressionLevel {
    /// Low suppression (~6 dB).
    Low,
    /// Moderate suppression (~12 dB, default).
    Moderate,
    /// High suppression (~18 dB).
    High,
    /// Very high suppression (~21 dB).
    VeryHigh,
}

impl Default for NoiseSuppression {
    fn default() -> Self {
        Self {
            level: NoiseSuppressionLevel::Moderate,
            analyze_linear_aec_output_when_available: false,
        }
    }
}

/// Automatic Gain Controller 2 (AGC2) settings.
///
/// AGC2 brings the captured audio signal to the desired level by combining
/// three controllers (input volume, adaptive digital, and fixed digital)
/// and a limiter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GainController2 {
    /// Enable the input volume controller. Adjusts the input volume applied
    /// when audio is captured (e.g., microphone volume on a soundcard).
    pub input_volume_controller: bool,
    /// Adaptive digital controller settings. Set to `Some(...)` to enable.
    /// Adjusts and applies a digital gain after echo cancellation and
    /// noise suppression.
    pub adaptive_digital: Option<AdaptiveDigital>,
    /// Applies a fixed digital gain after the adaptive digital controller
    /// and before the limiter.
    pub fixed_digital: FixedDigital,
}

/// Adaptive digital controller settings within AGC2.
///
/// Adjusts and applies a digital gain after echo cancellation and after
/// noise suppression.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDigital {
    /// Headroom in dB (default: 5.0).
    pub headroom_db: f32,
    /// Maximum gain in dB (default: 50.0).
    pub max_gain_db: f32,
    /// Initial gain in dB (default: 15.0).
    pub initial_gain_db: f32,
    /// Maximum gain change rate in dB/second (default: 6.0).
    pub max_gain_change_db_per_second: f32,
    /// Maximum output noise level in dBFS (default: -50.0).
    pub max_output_noise_level_dbfs: f32,
}

impl Default for AdaptiveDigital {
    fn default() -> Self {
        Self {
            headroom_db: 5.0,
            max_gain_db: 50.0,
            initial_gain_db: 15.0,
            max_gain_change_db_per_second: 6.0,
            max_output_noise_level_dbfs: -50.0,
        }
    }
}

/// Fixed digital controller settings within AGC2.
///
/// Applies a fixed digital gain after the adaptive digital controller
/// and before the limiter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FixedDigital {
    /// Fixed gain in dB (default: 0.0). Setting a value greater than zero
    /// turns the limiter into a compressor that first applies a fixed gain.
    pub gain_db: f32,
}

/// Runtime settings that can be applied without reinitialization.
///
/// These are enqueued and applied at the next call to
/// [`AudioProcessing::process_stream_f32()`](crate::AudioProcessing::process_stream_f32)
/// or [`AudioProcessing::process_stream_i16()`](crate::AudioProcessing::process_stream_i16).
#[derive(Debug, Clone)]
pub enum RuntimeSetting {
    /// Capture pre-gain linear factor.
    CapturePreGain(f32),
    /// Capture post-gain linear factor.
    CapturePostGain(f32),
    /// Fixed post-gain in dB. Must be in the range `0.0..=90.0`.
    CaptureFixedPostGain(f32),
    /// Playout (render) volume change. The value is the unnormalized volume.
    PlayoutVolumeChange(i32),
    /// Notifies that the playout (render) audio device has changed.
    PlayoutAudioDeviceChange(PlayoutAudioDeviceInfo),
    /// Whether the capture output is used. When false, some components may
    /// optimize by skipping work.
    CaptureOutputUsed(bool),
}

/// Play-out audio device properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayoutAudioDeviceInfo {
    /// Identifies the audio device.
    pub id: i32,
    /// Maximum volume of the audio device.
    pub max_volume: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_all_disabled() {
        let config = Config::default();
        assert_eq!(
            config.pipeline.maximum_internal_processing_rate,
            MaxProcessingRate::Rate32kHz
        );
        assert!(!config.pipeline.multi_channel_render);
        assert!(!config.pipeline.multi_channel_capture);
        assert_eq!(
            config.pipeline.capture_downmix_method,
            DownmixMethod::AverageChannels
        );
        assert!(config.pre_amplifier.is_none());
        assert!(config.capture_level_adjustment.is_none());
        assert!(config.high_pass_filter.is_none());
        assert!(config.echo_canceller.is_none());
        assert!(config.noise_suppression.is_none());
        assert!(config.gain_controller2.is_none());
    }

    #[test]
    fn default_sub_configs_match_upstream() {
        let pre_amp = PreAmplifier::default();
        assert_eq!(pre_amp.fixed_gain_factor, 1.0);

        let cla = CaptureLevelAdjustment::default();
        assert_eq!(cla.pre_gain_factor, 1.0);
        assert_eq!(cla.post_gain_factor, 1.0);
        assert!(cla.analog_mic_gain_emulation.is_none());

        let amge = AnalogMicGainEmulation::default();
        assert_eq!(amge.initial_level, 255);

        let hpf = HighPassFilter::default();
        assert!(hpf.apply_in_full_band);

        let ec = EchoCanceller::default();
        assert!(ec.enforce_high_pass_filtering);

        let ns = NoiseSuppression::default();
        assert_eq!(ns.level, NoiseSuppressionLevel::Moderate);
        assert!(!ns.analyze_linear_aec_output_when_available);

        let gc2 = GainController2::default();
        assert!(!gc2.input_volume_controller);
        assert!(gc2.adaptive_digital.is_none());
        assert_eq!(gc2.fixed_digital.gain_db, 0.0);

        let ad = AdaptiveDigital::default();
        assert_eq!(ad.headroom_db, 5.0);
        assert_eq!(ad.max_gain_db, 50.0);
        assert_eq!(ad.initial_gain_db, 15.0);
        assert_eq!(ad.max_gain_change_db_per_second, 6.0);
        assert_eq!(ad.max_output_noise_level_dbfs, -50.0);
    }

    #[test]
    fn capture_level_adjustment_equality() {
        let a = CaptureLevelAdjustment::default();
        let mut b = a.clone();
        assert_eq!(a, b);
        b.pre_gain_factor = 2.0;
        assert_ne!(a, b);
    }

    #[test]
    fn max_processing_rate_as_hz() {
        assert_eq!(MaxProcessingRate::Rate32kHz.as_hz(), 32000);
        assert_eq!(MaxProcessingRate::Rate48kHz.as_hz(), 48000);
    }
}
