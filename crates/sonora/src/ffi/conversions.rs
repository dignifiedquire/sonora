//! Bidirectional conversions between C API types and Rust types.

use crate::config::{
    AdaptiveDigital, AnalogMicGainEmulation, CaptureLevelAdjustment, Config, DownmixMethod,
    EchoCanceller, FixedDigital, GainController2, HighPassFilter, MaxProcessingRate,
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
                maximum_internal_processing_rate: if self.pipeline_maximum_internal_processing_rate
                    == 32000
                {
                    MaxProcessingRate::Rate32kHz
                } else {
                    MaxProcessingRate::Rate48kHz
                },
                multi_channel_render: self.pipeline_multi_channel_render,
                multi_channel_capture: self.pipeline_multi_channel_capture,
                capture_downmix_method: self.pipeline_capture_downmix_method.to_rust(),
            },
            pre_amplifier: if self.pre_amplifier_enabled {
                Some(PreAmplifier {
                    fixed_gain_factor: self.pre_amplifier_fixed_gain_factor,
                })
            } else {
                None
            },
            capture_level_adjustment: if self.capture_level_adjustment_enabled {
                Some(CaptureLevelAdjustment {
                    pre_gain_factor: self.capture_level_adjustment_pre_gain_factor,
                    post_gain_factor: self.capture_level_adjustment_post_gain_factor,
                    analog_mic_gain_emulation: if self.analog_mic_gain_emulation_enabled {
                        Some(AnalogMicGainEmulation {
                            initial_level: self
                                .analog_mic_gain_emulation_initial_level
                                .clamp(0, 255) as u8,
                        })
                    } else {
                        None
                    },
                })
            } else {
                None
            },
            high_pass_filter: if self.high_pass_filter_enabled {
                Some(HighPassFilter {
                    apply_in_full_band: self.high_pass_filter_apply_in_full_band,
                })
            } else {
                None
            },
            echo_canceller: if self.echo_canceller_enabled {
                Some(EchoCanceller {
                    enforce_high_pass_filtering: self.echo_canceller_enforce_high_pass_filtering,
                })
            } else {
                None
            },
            noise_suppression: if self.noise_suppression_enabled {
                Some(NoiseSuppression {
                    level: self.noise_suppression_level.to_rust(),
                    analyze_linear_aec_output_when_available: self
                        .noise_suppression_analyze_linear_aec_output_when_available,
                })
            } else {
                None
            },
            gain_controller2: if self.gain_controller2_enabled {
                Some(GainController2 {
                    input_volume_controller: self.gain_controller2_input_volume_controller_enabled,
                    adaptive_digital: if self.gain_controller2_adaptive_digital_enabled {
                        Some(AdaptiveDigital {
                            headroom_db: self.gain_controller2_adaptive_digital_headroom_db,
                            max_gain_db: self.gain_controller2_adaptive_digital_max_gain_db,
                            initial_gain_db: self.gain_controller2_adaptive_digital_initial_gain_db,
                            max_gain_change_db_per_second: self
                                .gain_controller2_adaptive_digital_max_gain_change_db_per_second,
                            max_output_noise_level_dbfs: self
                                .gain_controller2_adaptive_digital_max_output_noise_level_dbfs,
                        })
                    } else {
                        None
                    },
                    fixed_digital: FixedDigital {
                        gain_db: self.gain_controller2_fixed_digital_gain_db,
                    },
                })
            } else {
                None
            },
        }
    }

    /// Convert from nested Rust [`Config`] to flat C config.
    pub(crate) fn from_rust(config: &Config) -> Self {
        let (pre_amplifier_enabled, pre_amplifier_fixed_gain_factor) = match &config.pre_amplifier {
            Some(pa) => (true, pa.fixed_gain_factor),
            None => (false, PreAmplifier::default().fixed_gain_factor),
        };

        let (
            capture_level_adjustment_enabled,
            capture_level_adjustment_pre_gain_factor,
            capture_level_adjustment_post_gain_factor,
            analog_mic_gain_emulation_enabled,
            analog_mic_gain_emulation_initial_level,
        ) = match &config.capture_level_adjustment {
            Some(cla) => {
                let (emu_enabled, emu_level) = match &cla.analog_mic_gain_emulation {
                    Some(amge) => (true, i32::from(amge.initial_level)),
                    None => (
                        false,
                        i32::from(AnalogMicGainEmulation::default().initial_level),
                    ),
                };
                (
                    true,
                    cla.pre_gain_factor,
                    cla.post_gain_factor,
                    emu_enabled,
                    emu_level,
                )
            }
            None => {
                let defaults = CaptureLevelAdjustment::default();
                (
                    false,
                    defaults.pre_gain_factor,
                    defaults.post_gain_factor,
                    false,
                    i32::from(AnalogMicGainEmulation::default().initial_level),
                )
            }
        };

        let (high_pass_filter_enabled, high_pass_filter_apply_in_full_band) =
            match &config.high_pass_filter {
                Some(hpf) => (true, hpf.apply_in_full_band),
                None => (false, HighPassFilter::default().apply_in_full_band),
            };

        let (echo_canceller_enabled, echo_canceller_enforce_high_pass_filtering) =
            match &config.echo_canceller {
                Some(ec) => (true, ec.enforce_high_pass_filtering),
                None => (false, EchoCanceller::default().enforce_high_pass_filtering),
            };

        let (
            noise_suppression_enabled,
            noise_suppression_level,
            noise_suppression_analyze_linear_aec_output_when_available,
        ) = match &config.noise_suppression {
            Some(ns) => (
                true,
                WapNoiseSuppressionLevel::from_rust(ns.level),
                ns.analyze_linear_aec_output_when_available,
            ),
            None => {
                let defaults = NoiseSuppression::default();
                (
                    false,
                    WapNoiseSuppressionLevel::from_rust(defaults.level),
                    defaults.analyze_linear_aec_output_when_available,
                )
            }
        };

        let (
            gain_controller2_enabled,
            gain_controller2_input_volume_controller_enabled,
            gain_controller2_adaptive_digital_enabled,
            gain_controller2_adaptive_digital_headroom_db,
            gain_controller2_adaptive_digital_max_gain_db,
            gain_controller2_adaptive_digital_initial_gain_db,
            gain_controller2_adaptive_digital_max_gain_change_db_per_second,
            gain_controller2_adaptive_digital_max_output_noise_level_dbfs,
            gain_controller2_fixed_digital_gain_db,
        ) = match &config.gain_controller2 {
            Some(gc2) => {
                let (ad_enabled, ad_headroom, ad_max_gain, ad_initial, ad_change, ad_noise) =
                    match &gc2.adaptive_digital {
                        Some(ad) => (
                            true,
                            ad.headroom_db,
                            ad.max_gain_db,
                            ad.initial_gain_db,
                            ad.max_gain_change_db_per_second,
                            ad.max_output_noise_level_dbfs,
                        ),
                        None => {
                            let defaults = AdaptiveDigital::default();
                            (
                                false,
                                defaults.headroom_db,
                                defaults.max_gain_db,
                                defaults.initial_gain_db,
                                defaults.max_gain_change_db_per_second,
                                defaults.max_output_noise_level_dbfs,
                            )
                        }
                    };
                (
                    true,
                    gc2.input_volume_controller,
                    ad_enabled,
                    ad_headroom,
                    ad_max_gain,
                    ad_initial,
                    ad_change,
                    ad_noise,
                    gc2.fixed_digital.gain_db,
                )
            }
            None => {
                let defaults = AdaptiveDigital::default();
                (
                    false,
                    false,
                    false,
                    defaults.headroom_db,
                    defaults.max_gain_db,
                    defaults.initial_gain_db,
                    defaults.max_gain_change_db_per_second,
                    defaults.max_output_noise_level_dbfs,
                    FixedDigital::default().gain_db,
                )
            }
        };

        Self {
            pipeline_maximum_internal_processing_rate: config
                .pipeline
                .maximum_internal_processing_rate
                .as_hz() as i32,
            pipeline_multi_channel_render: config.pipeline.multi_channel_render,
            pipeline_multi_channel_capture: config.pipeline.multi_channel_capture,
            pipeline_capture_downmix_method: WapDownmixMethod::from_rust(
                config.pipeline.capture_downmix_method,
            ),

            pre_amplifier_enabled,
            pre_amplifier_fixed_gain_factor,

            capture_level_adjustment_enabled,
            capture_level_adjustment_pre_gain_factor,
            capture_level_adjustment_post_gain_factor,
            analog_mic_gain_emulation_enabled,
            analog_mic_gain_emulation_initial_level,

            high_pass_filter_enabled,
            high_pass_filter_apply_in_full_band,

            echo_canceller_enabled,
            echo_canceller_enforce_high_pass_filtering,

            noise_suppression_enabled,
            noise_suppression_level,
            noise_suppression_analyze_linear_aec_output_when_available,

            gain_controller2_enabled,
            gain_controller2_fixed_digital_gain_db,
            gain_controller2_adaptive_digital_enabled,
            gain_controller2_adaptive_digital_headroom_db,
            gain_controller2_adaptive_digital_max_gain_db,
            gain_controller2_adaptive_digital_initial_gain_db,
            gain_controller2_adaptive_digital_max_gain_change_db_per_second,
            gain_controller2_adaptive_digital_max_output_noise_level_dbfs,
            gain_controller2_input_volume_controller_enabled,
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
        assert!(roundtrip.pre_amplifier.is_none());
        assert!(roundtrip.capture_level_adjustment.is_none());
        assert!(roundtrip.high_pass_filter.is_none());
        assert!(roundtrip.echo_canceller.is_none());
        assert!(roundtrip.noise_suppression.is_none());
        assert!(roundtrip.gain_controller2.is_none());
    }

    #[test]
    fn config_roundtrip_all_enabled() {
        let rust_config = Config {
            echo_canceller: Some(EchoCanceller {
                enforce_high_pass_filtering: true,
            }),
            noise_suppression: Some(NoiseSuppression {
                level: NoiseSuppressionLevel::VeryHigh,
                ..Default::default()
            }),
            high_pass_filter: Some(HighPassFilter {
                apply_in_full_band: true,
            }),
            gain_controller2: Some(GainController2 {
                adaptive_digital: Some(AdaptiveDigital {
                    headroom_db: 3.0,
                    max_gain_db: 40.0,
                    ..Default::default()
                }),
                fixed_digital: FixedDigital { gain_db: 6.0 },
                ..Default::default()
            }),
            pre_amplifier: Some(PreAmplifier {
                fixed_gain_factor: 2.5,
            }),
            capture_level_adjustment: Some(CaptureLevelAdjustment {
                pre_gain_factor: 1.5,
                post_gain_factor: 0.8,
                analog_mic_gain_emulation: Some(AnalogMicGainEmulation { initial_level: 128 }),
            }),
            pipeline: Pipeline {
                maximum_internal_processing_rate: MaxProcessingRate::Rate48kHz,
                multi_channel_render: true,
                multi_channel_capture: true,
                capture_downmix_method: DownmixMethod::UseFirstChannel,
            },
        };

        let c_config = WapConfig::from_rust(&rust_config);
        let roundtrip = c_config.to_rust();

        assert!(roundtrip.echo_canceller.is_some());
        let ec = roundtrip.echo_canceller.unwrap();
        assert!(ec.enforce_high_pass_filtering);

        assert!(roundtrip.noise_suppression.is_some());
        let ns = roundtrip.noise_suppression.unwrap();
        assert_eq!(ns.level, NoiseSuppressionLevel::VeryHigh);

        assert!(roundtrip.high_pass_filter.is_some());

        let gc2 = roundtrip.gain_controller2.unwrap();
        assert!(gc2.adaptive_digital.is_some());
        let ad = gc2.adaptive_digital.unwrap();
        assert_eq!(ad.headroom_db, 3.0);
        assert_eq!(ad.max_gain_db, 40.0);
        assert_eq!(gc2.fixed_digital.gain_db, 6.0);

        let pa = roundtrip.pre_amplifier.unwrap();
        assert_eq!(pa.fixed_gain_factor, 2.5);

        let cla = roundtrip.capture_level_adjustment.unwrap();
        assert_eq!(cla.pre_gain_factor, 1.5);
        assert_eq!(cla.post_gain_factor, 0.8);
        let amge = cla.analog_mic_gain_emulation.unwrap();
        assert_eq!(amge.initial_level, 128);

        assert_eq!(
            roundtrip.pipeline.maximum_internal_processing_rate,
            MaxProcessingRate::Rate48kHz
        );
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
