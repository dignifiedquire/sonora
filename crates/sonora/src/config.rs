//! Audio processing configuration.
//!
//! Ported from `AudioProcessing::Config` in `api/audio/audio_processing.h`.

/// Top-level configuration for the audio processing pipeline.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Pipeline processing properties.
    pub pipeline: Pipeline,
    /// Pre-amplifier settings.
    pub pre_amplifier: PreAmplifier,
    /// Capture-level adjustment settings.
    pub capture_level_adjustment: CaptureLevelAdjustment,
    /// High-pass filter settings.
    pub high_pass_filter: HighPassFilter,
    /// Echo canceller settings.
    pub echo_canceller: EchoCanceller,
    /// Noise suppression settings.
    pub noise_suppression: NoiseSuppression,
    /// AGC2 settings.
    pub gain_controller2: GainController2,
}

/// Pipeline processing properties.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Maximum allowed processing rate used internally.
    /// May only be set to 32000 or 48000; other values are treated as 48000.
    pub maximum_internal_processing_rate: i32,
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
            maximum_internal_processing_rate: 32000,
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
    pub enabled: bool,
    pub fixed_gain_factor: f32,
}

impl Default for PreAmplifier {
    fn default() -> Self {
        Self {
            enabled: false,
            fixed_gain_factor: 1.0,
        }
    }
}

/// General level adjustment in the capture pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureLevelAdjustment {
    pub enabled: bool,
    /// Scales the signal before any processing is done.
    pub pre_gain_factor: f32,
    /// Scales the signal after all processing is done.
    pub post_gain_factor: f32,
    /// Analog mic gain emulation settings.
    pub analog_mic_gain_emulation: AnalogMicGainEmulation,
}

impl Default for CaptureLevelAdjustment {
    fn default() -> Self {
        Self {
            enabled: false,
            pre_gain_factor: 1.0,
            post_gain_factor: 1.0,
            analog_mic_gain_emulation: AnalogMicGainEmulation::default(),
        }
    }
}

/// Analog microphone gain emulation settings.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalogMicGainEmulation {
    pub enabled: bool,
    /// Initial analog gain level (range: 0..=255).
    pub initial_level: i32,
}

impl Default for AnalogMicGainEmulation {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_level: 255,
        }
    }
}

/// High-pass filter settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighPassFilter {
    pub enabled: bool,
    pub apply_in_full_band: bool,
}

impl Default for HighPassFilter {
    fn default() -> Self {
        Self {
            enabled: false,
            apply_in_full_band: true,
        }
    }
}

/// Echo canceller settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoCanceller {
    pub enabled: bool,
    /// Enforce high-pass filtering (no effect in mobile mode).
    pub enforce_high_pass_filtering: bool,
}

impl Default for EchoCanceller {
    fn default() -> Self {
        Self {
            enabled: false,
            enforce_high_pass_filtering: true,
        }
    }
}

/// Noise suppression settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseSuppression {
    pub enabled: bool,
    pub level: NoiseSuppressionLevel,
    pub analyze_linear_aec_output_when_available: bool,
}

/// Noise suppression aggressiveness level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSuppressionLevel {
    Low,
    Moderate,
    High,
    VeryHigh,
}

impl Default for NoiseSuppression {
    fn default() -> Self {
        Self {
            enabled: false,
            level: NoiseSuppressionLevel::Moderate,
            analyze_linear_aec_output_when_available: false,
        }
    }
}

/// AGC2 (Automatic Gain Controller 2) settings.
#[derive(Debug, Clone, PartialEq)]
pub struct GainController2 {
    pub enabled: bool,
    /// Input volume controller settings.
    pub input_volume_controller: InputVolumeControllerConfig,
    /// Adaptive digital controller settings.
    pub adaptive_digital: AdaptiveDigital,
    /// Fixed digital controller settings.
    pub fixed_digital: FixedDigital,
}

impl Default for GainController2 {
    fn default() -> Self {
        Self {
            enabled: false,
            input_volume_controller: InputVolumeControllerConfig::default(),
            adaptive_digital: AdaptiveDigital::default(),
            fixed_digital: FixedDigital::default(),
        }
    }
}

/// Input volume controller settings within AGC2.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InputVolumeControllerConfig {
    pub enabled: bool,
}

/// Adaptive digital controller settings within AGC2.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDigital {
    pub enabled: bool,
    pub headroom_db: f32,
    pub max_gain_db: f32,
    pub initial_gain_db: f32,
    pub max_gain_change_db_per_second: f32,
    pub max_output_noise_level_dbfs: f32,
}

impl Default for AdaptiveDigital {
    fn default() -> Self {
        Self {
            enabled: false,
            headroom_db: 5.0,
            max_gain_db: 50.0,
            initial_gain_db: 15.0,
            max_gain_change_db_per_second: 6.0,
            max_output_noise_level_dbfs: -50.0,
        }
    }
}

/// Fixed digital controller settings within AGC2.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FixedDigital {
    /// Fixed gain in dB. Setting > 0 turns the limiter into a compressor.
    pub gain_db: f32,
}

/// Runtime settings that can be applied without reinitialization.
#[derive(Debug, Clone)]
pub enum RuntimeSetting {
    /// Capture pre-gain factor.
    CapturePreGain(f32),
    /// Capture post-gain factor.
    CapturePostGain(f32),
    /// Fixed post-gain in dB (range: 0..=90).
    CaptureFixedPostGain(f32),
    /// Playout volume change (unnormalized).
    PlayoutVolumeChange(i32),
    /// Playout audio device change.
    PlayoutAudioDeviceChange(PlayoutAudioDeviceInfo),
    /// Whether the capture output is used.
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
    fn default_config_matches_upstream() {
        let config = Config::default();
        assert_eq!(config.pipeline.maximum_internal_processing_rate, 32000);
        assert!(!config.pipeline.multi_channel_render);
        assert!(!config.pipeline.multi_channel_capture);
        assert_eq!(
            config.pipeline.capture_downmix_method,
            DownmixMethod::AverageChannels
        );
        assert!(!config.pre_amplifier.enabled);
        assert_eq!(config.pre_amplifier.fixed_gain_factor, 1.0);
        assert!(!config.capture_level_adjustment.enabled);
        assert_eq!(config.capture_level_adjustment.pre_gain_factor, 1.0);
        assert_eq!(config.capture_level_adjustment.post_gain_factor, 1.0);
        assert!(
            !config
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .enabled
        );
        assert_eq!(
            config
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .initial_level,
            255
        );
        assert!(!config.high_pass_filter.enabled);
        assert!(config.high_pass_filter.apply_in_full_band);
        assert!(!config.echo_canceller.enabled);
        assert!(config.echo_canceller.enforce_high_pass_filtering);
        assert!(!config.noise_suppression.enabled);
        assert_eq!(
            config.noise_suppression.level,
            NoiseSuppressionLevel::Moderate
        );
        assert!(!config.gain_controller2.enabled);
        assert!(!config.gain_controller2.input_volume_controller.enabled);
        assert!(!config.gain_controller2.adaptive_digital.enabled);
        assert_eq!(config.gain_controller2.adaptive_digital.headroom_db, 5.0);
        assert_eq!(config.gain_controller2.adaptive_digital.max_gain_db, 50.0);
        assert_eq!(
            config.gain_controller2.adaptive_digital.initial_gain_db,
            15.0
        );
        assert_eq!(
            config
                .gain_controller2
                .adaptive_digital
                .max_gain_change_db_per_second,
            6.0
        );
        assert_eq!(
            config
                .gain_controller2
                .adaptive_digital
                .max_output_noise_level_dbfs,
            -50.0
        );
        assert_eq!(config.gain_controller2.fixed_digital.gain_db, 0.0);
    }

    #[test]
    fn capture_level_adjustment_equality() {
        let a = CaptureLevelAdjustment::default();
        let mut b = a.clone();
        assert_eq!(a, b);
        b.pre_gain_factor = 2.0;
        assert_ne!(a, b);
    }
}
