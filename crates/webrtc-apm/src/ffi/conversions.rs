//! Bidirectional conversions between C API types and Rust types.

use crate::config::{
    AdaptiveDigital, AnalogMicGainEmulation, CaptureLevelAdjustment, Config, DownmixMethod,
    EchoCanceller, FixedDigital, GainController2, HighPassFilter, InputVolumeControllerConfig,
    NoiseSuppression, NoiseSuppressionLevel, Pipeline, PreAmplifier,
};
use crate::stats::AudioProcessingStats;
use crate::stream_config::StreamConfig;

use super::types::{
    WapConfig, WapDownmixMethod, WapNoiseSuppressionLevel, WapStats, WapStreamConfig,
};

// ---------------------------------------------------------------------------
// WapConfig <-> Config
// ---------------------------------------------------------------------------

impl WapConfig {
    /// Convert from flat C config to nested Rust [`Config`].
    pub(crate) fn to_rust(self) -> Config {
        Config {
            pipeline: Pipeline {
                maximum_internal_processing_rate: self.pipeline_maximum_internal_processing_rate,
                multi_channel_render: self.pipeline_multi_channel_render,
                multi_channel_capture: self.pipeline_multi_channel_capture,
                capture_downmix_method: self.pipeline_capture_downmix_method.to_rust(),
            },
            pre_amplifier: PreAmplifier {
                enabled: self.pre_amplifier_enabled,
                fixed_gain_factor: self.pre_amplifier_fixed_gain_factor,
            },
            capture_level_adjustment: CaptureLevelAdjustment {
                enabled: self.capture_level_adjustment_enabled,
                pre_gain_factor: self.capture_level_adjustment_pre_gain_factor,
                post_gain_factor: self.capture_level_adjustment_post_gain_factor,
                analog_mic_gain_emulation: AnalogMicGainEmulation {
                    enabled: self.analog_mic_gain_emulation_enabled,
                    initial_level: self.analog_mic_gain_emulation_initial_level,
                },
            },
            high_pass_filter: HighPassFilter {
                enabled: self.high_pass_filter_enabled,
                apply_in_full_band: self.high_pass_filter_apply_in_full_band,
            },
            echo_canceller: EchoCanceller {
                enabled: self.echo_canceller_enabled,
                enforce_high_pass_filtering: self.echo_canceller_enforce_high_pass_filtering,
            },
            noise_suppression: NoiseSuppression {
                enabled: self.noise_suppression_enabled,
                level: self.noise_suppression_level.to_rust(),
                analyze_linear_aec_output_when_available: self
                    .noise_suppression_analyze_linear_aec_output_when_available,
            },
            gain_controller2: GainController2 {
                enabled: self.gain_controller2_enabled,
                input_volume_controller: InputVolumeControllerConfig {
                    enabled: self.gain_controller2_input_volume_controller_enabled,
                },
                adaptive_digital: AdaptiveDigital {
                    enabled: self.gain_controller2_adaptive_digital_enabled,
                    headroom_db: self.gain_controller2_adaptive_digital_headroom_db,
                    max_gain_db: self.gain_controller2_adaptive_digital_max_gain_db,
                    initial_gain_db: self.gain_controller2_adaptive_digital_initial_gain_db,
                    max_gain_change_db_per_second: self
                        .gain_controller2_adaptive_digital_max_gain_change_db_per_second,
                    max_output_noise_level_dbfs: self
                        .gain_controller2_adaptive_digital_max_output_noise_level_dbfs,
                },
                fixed_digital: FixedDigital {
                    gain_db: self.gain_controller2_fixed_digital_gain_db,
                },
            },
        }
    }

    /// Convert from nested Rust [`Config`] to flat C config.
    pub(crate) fn from_rust(config: &Config) -> Self {
        Self {
            pipeline_maximum_internal_processing_rate: config
                .pipeline
                .maximum_internal_processing_rate,
            pipeline_multi_channel_render: config.pipeline.multi_channel_render,
            pipeline_multi_channel_capture: config.pipeline.multi_channel_capture,
            pipeline_capture_downmix_method: WapDownmixMethod::from_rust(
                config.pipeline.capture_downmix_method,
            ),

            pre_amplifier_enabled: config.pre_amplifier.enabled,
            pre_amplifier_fixed_gain_factor: config.pre_amplifier.fixed_gain_factor,

            capture_level_adjustment_enabled: config.capture_level_adjustment.enabled,
            capture_level_adjustment_pre_gain_factor: config
                .capture_level_adjustment
                .pre_gain_factor,
            capture_level_adjustment_post_gain_factor: config
                .capture_level_adjustment
                .post_gain_factor,
            analog_mic_gain_emulation_enabled: config
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .enabled,
            analog_mic_gain_emulation_initial_level: config
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .initial_level,

            high_pass_filter_enabled: config.high_pass_filter.enabled,
            high_pass_filter_apply_in_full_band: config.high_pass_filter.apply_in_full_band,

            echo_canceller_enabled: config.echo_canceller.enabled,
            echo_canceller_enforce_high_pass_filtering: config
                .echo_canceller
                .enforce_high_pass_filtering,

            noise_suppression_enabled: config.noise_suppression.enabled,
            noise_suppression_level: WapNoiseSuppressionLevel::from_rust(
                config.noise_suppression.level,
            ),
            noise_suppression_analyze_linear_aec_output_when_available: config
                .noise_suppression
                .analyze_linear_aec_output_when_available,

            gain_controller2_enabled: config.gain_controller2.enabled,
            gain_controller2_fixed_digital_gain_db: config.gain_controller2.fixed_digital.gain_db,
            gain_controller2_adaptive_digital_enabled: config
                .gain_controller2
                .adaptive_digital
                .enabled,
            gain_controller2_adaptive_digital_headroom_db: config
                .gain_controller2
                .adaptive_digital
                .headroom_db,
            gain_controller2_adaptive_digital_max_gain_db: config
                .gain_controller2
                .adaptive_digital
                .max_gain_db,
            gain_controller2_adaptive_digital_initial_gain_db: config
                .gain_controller2
                .adaptive_digital
                .initial_gain_db,
            gain_controller2_adaptive_digital_max_gain_change_db_per_second: config
                .gain_controller2
                .adaptive_digital
                .max_gain_change_db_per_second,
            gain_controller2_adaptive_digital_max_output_noise_level_dbfs: config
                .gain_controller2
                .adaptive_digital
                .max_output_noise_level_dbfs,
            gain_controller2_input_volume_controller_enabled: config
                .gain_controller2
                .input_volume_controller
                .enabled,
        }
    }
}

// ---------------------------------------------------------------------------
// Enum conversions
// ---------------------------------------------------------------------------

impl WapNoiseSuppressionLevel {
    pub(crate) fn to_rust(self) -> NoiseSuppressionLevel {
        match self {
            Self::Low => NoiseSuppressionLevel::Low,
            Self::Moderate => NoiseSuppressionLevel::Moderate,
            Self::High => NoiseSuppressionLevel::High,
            Self::VeryHigh => NoiseSuppressionLevel::VeryHigh,
        }
    }

    pub(crate) fn from_rust(level: NoiseSuppressionLevel) -> Self {
        match level {
            NoiseSuppressionLevel::Low => Self::Low,
            NoiseSuppressionLevel::Moderate => Self::Moderate,
            NoiseSuppressionLevel::High => Self::High,
            NoiseSuppressionLevel::VeryHigh => Self::VeryHigh,
        }
    }
}

impl WapDownmixMethod {
    pub(crate) fn to_rust(self) -> DownmixMethod {
        match self {
            Self::AverageChannels => DownmixMethod::AverageChannels,
            Self::UseFirstChannel => DownmixMethod::UseFirstChannel,
        }
    }

    pub(crate) fn from_rust(method: DownmixMethod) -> Self {
        match method {
            DownmixMethod::AverageChannels => Self::AverageChannels,
            DownmixMethod::UseFirstChannel => Self::UseFirstChannel,
        }
    }
}

// ---------------------------------------------------------------------------
// WapStreamConfig -> StreamConfig
// ---------------------------------------------------------------------------

impl WapStreamConfig {
    pub(crate) fn to_rust(self) -> StreamConfig {
        StreamConfig::from_signed(
            self.sample_rate_hz,
            if self.num_channels < 0 {
                0
            } else {
                self.num_channels as usize
            },
        )
    }
}

// ---------------------------------------------------------------------------
// AudioProcessingStats -> WapStats
// ---------------------------------------------------------------------------

impl WapStats {
    pub(crate) fn from_rust(stats: &AudioProcessingStats) -> Self {
        Self {
            has_echo_return_loss: stats.echo_return_loss.is_some(),
            echo_return_loss: stats.echo_return_loss.unwrap_or(0.0),

            has_echo_return_loss_enhancement: stats.echo_return_loss_enhancement.is_some(),
            echo_return_loss_enhancement: stats.echo_return_loss_enhancement.unwrap_or(0.0),

            has_divergent_filter_fraction: stats.divergent_filter_fraction.is_some(),
            divergent_filter_fraction: stats.divergent_filter_fraction.unwrap_or(0.0),

            has_delay_median_ms: stats.delay_median_ms.is_some(),
            delay_median_ms: stats.delay_median_ms.unwrap_or(0),

            has_delay_standard_deviation_ms: stats.delay_standard_deviation_ms.is_some(),
            delay_standard_deviation_ms: stats.delay_standard_deviation_ms.unwrap_or(0),

            has_residual_echo_likelihood: stats.residual_echo_likelihood.is_some(),
            residual_echo_likelihood: stats.residual_echo_likelihood.unwrap_or(0.0),

            has_residual_echo_likelihood_recent_max: stats
                .residual_echo_likelihood_recent_max
                .is_some(),
            residual_echo_likelihood_recent_max: stats
                .residual_echo_likelihood_recent_max
                .unwrap_or(0.0),

            has_delay_ms: stats.delay_ms.is_some(),
            delay_ms: stats.delay_ms.unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip_default() {
        let rust_config = Config::default();
        let c_config = WapConfig::from_rust(&rust_config);
        let roundtrip = c_config.to_rust();

        assert_eq!(
            rust_config.pipeline.maximum_internal_processing_rate,
            roundtrip.pipeline.maximum_internal_processing_rate
        );
        assert_eq!(
            rust_config.pipeline.multi_channel_render,
            roundtrip.pipeline.multi_channel_render
        );
        assert_eq!(
            rust_config.pipeline.multi_channel_capture,
            roundtrip.pipeline.multi_channel_capture
        );
        assert_eq!(
            rust_config.pipeline.capture_downmix_method,
            roundtrip.pipeline.capture_downmix_method
        );
        assert_eq!(
            rust_config.pre_amplifier.enabled,
            roundtrip.pre_amplifier.enabled
        );
        assert_eq!(
            rust_config.pre_amplifier.fixed_gain_factor,
            roundtrip.pre_amplifier.fixed_gain_factor
        );
        assert_eq!(
            rust_config.capture_level_adjustment,
            roundtrip.capture_level_adjustment
        );
        assert_eq!(rust_config.high_pass_filter, roundtrip.high_pass_filter);
        assert_eq!(rust_config.echo_canceller, roundtrip.echo_canceller);
        assert_eq!(
            rust_config.noise_suppression.enabled,
            roundtrip.noise_suppression.enabled
        );
        assert_eq!(
            rust_config.noise_suppression.level,
            roundtrip.noise_suppression.level
        );
        assert_eq!(rust_config.gain_controller2, roundtrip.gain_controller2);
    }

    #[test]
    fn config_roundtrip_all_enabled() {
        let mut rust_config = Config::default();
        rust_config.echo_canceller.enabled = true;
        rust_config.noise_suppression.enabled = true;
        rust_config.noise_suppression.level = NoiseSuppressionLevel::VeryHigh;
        rust_config.high_pass_filter.enabled = true;
        rust_config.gain_controller2.enabled = true;
        rust_config.gain_controller2.adaptive_digital.enabled = true;
        rust_config.gain_controller2.adaptive_digital.headroom_db = 3.0;
        rust_config.gain_controller2.adaptive_digital.max_gain_db = 40.0;
        rust_config.gain_controller2.fixed_digital.gain_db = 6.0;
        rust_config.pre_amplifier.enabled = true;
        rust_config.pre_amplifier.fixed_gain_factor = 2.5;
        rust_config.capture_level_adjustment.enabled = true;
        rust_config.capture_level_adjustment.pre_gain_factor = 1.5;
        rust_config.capture_level_adjustment.post_gain_factor = 0.8;
        rust_config
            .capture_level_adjustment
            .analog_mic_gain_emulation
            .enabled = true;
        rust_config
            .capture_level_adjustment
            .analog_mic_gain_emulation
            .initial_level = 128;
        rust_config.pipeline.maximum_internal_processing_rate = 48000;
        rust_config.pipeline.multi_channel_render = true;
        rust_config.pipeline.multi_channel_capture = true;
        rust_config.pipeline.capture_downmix_method = DownmixMethod::UseFirstChannel;

        let c_config = WapConfig::from_rust(&rust_config);
        let roundtrip = c_config.to_rust();

        assert!(roundtrip.echo_canceller.enabled);
        assert!(roundtrip.noise_suppression.enabled);
        assert_eq!(
            roundtrip.noise_suppression.level,
            NoiseSuppressionLevel::VeryHigh
        );
        assert!(roundtrip.high_pass_filter.enabled);
        assert!(roundtrip.gain_controller2.enabled);
        assert!(roundtrip.gain_controller2.adaptive_digital.enabled);
        assert_eq!(roundtrip.gain_controller2.adaptive_digital.headroom_db, 3.0);
        assert_eq!(
            roundtrip.gain_controller2.adaptive_digital.max_gain_db,
            40.0
        );
        assert_eq!(roundtrip.gain_controller2.fixed_digital.gain_db, 6.0);
        assert!(roundtrip.pre_amplifier.enabled);
        assert_eq!(roundtrip.pre_amplifier.fixed_gain_factor, 2.5);
        assert!(roundtrip.capture_level_adjustment.enabled);
        assert_eq!(roundtrip.capture_level_adjustment.pre_gain_factor, 1.5);
        assert_eq!(roundtrip.capture_level_adjustment.post_gain_factor, 0.8);
        assert!(
            roundtrip
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .enabled
        );
        assert_eq!(
            roundtrip
                .capture_level_adjustment
                .analog_mic_gain_emulation
                .initial_level,
            128
        );
        assert_eq!(roundtrip.pipeline.maximum_internal_processing_rate, 48000);
        assert!(roundtrip.pipeline.multi_channel_render);
        assert!(roundtrip.pipeline.multi_channel_capture);
        assert_eq!(
            roundtrip.pipeline.capture_downmix_method,
            DownmixMethod::UseFirstChannel
        );
    }

    #[test]
    fn noise_suppression_level_roundtrip() {
        for (c_level, rust_level) in [
            (WapNoiseSuppressionLevel::Low, NoiseSuppressionLevel::Low),
            (
                WapNoiseSuppressionLevel::Moderate,
                NoiseSuppressionLevel::Moderate,
            ),
            (WapNoiseSuppressionLevel::High, NoiseSuppressionLevel::High),
            (
                WapNoiseSuppressionLevel::VeryHigh,
                NoiseSuppressionLevel::VeryHigh,
            ),
        ] {
            assert_eq!(c_level.to_rust(), rust_level);
            assert_eq!(WapNoiseSuppressionLevel::from_rust(rust_level), c_level);
        }
    }

    #[test]
    fn downmix_method_roundtrip() {
        for (c_method, rust_method) in [
            (
                WapDownmixMethod::AverageChannels,
                DownmixMethod::AverageChannels,
            ),
            (
                WapDownmixMethod::UseFirstChannel,
                DownmixMethod::UseFirstChannel,
            ),
        ] {
            assert_eq!(c_method.to_rust(), rust_method);
            assert_eq!(WapDownmixMethod::from_rust(rust_method), c_method);
        }
    }

    #[test]
    fn stats_conversion_all_none() {
        let stats = AudioProcessingStats::default();
        let c_stats = WapStats::from_rust(&stats);
        assert!(!c_stats.has_echo_return_loss);
        assert!(!c_stats.has_echo_return_loss_enhancement);
        assert!(!c_stats.has_divergent_filter_fraction);
        assert!(!c_stats.has_delay_median_ms);
        assert!(!c_stats.has_delay_standard_deviation_ms);
        assert!(!c_stats.has_residual_echo_likelihood);
        assert!(!c_stats.has_residual_echo_likelihood_recent_max);
        assert!(!c_stats.has_delay_ms);
    }

    #[test]
    fn stats_conversion_with_values() {
        let stats = AudioProcessingStats {
            echo_return_loss: Some(10.5),
            echo_return_loss_enhancement: Some(20.3),
            divergent_filter_fraction: None,
            delay_median_ms: Some(42),
            delay_standard_deviation_ms: None,
            residual_echo_likelihood: Some(0.1),
            residual_echo_likelihood_recent_max: Some(0.5),
            delay_ms: Some(30),
        };
        let c_stats = WapStats::from_rust(&stats);
        assert!(c_stats.has_echo_return_loss);
        assert_eq!(c_stats.echo_return_loss, 10.5);
        assert!(c_stats.has_echo_return_loss_enhancement);
        assert_eq!(c_stats.echo_return_loss_enhancement, 20.3);
        assert!(!c_stats.has_divergent_filter_fraction);
        assert!(c_stats.has_delay_median_ms);
        assert_eq!(c_stats.delay_median_ms, 42);
        assert!(!c_stats.has_delay_standard_deviation_ms);
        assert!(c_stats.has_residual_echo_likelihood);
        assert_eq!(c_stats.residual_echo_likelihood, 0.1);
        assert!(c_stats.has_residual_echo_likelihood_recent_max);
        assert_eq!(c_stats.residual_echo_likelihood_recent_max, 0.5);
        assert!(c_stats.has_delay_ms);
        assert_eq!(c_stats.delay_ms, 30);
    }
}
