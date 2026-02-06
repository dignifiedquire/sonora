//! AEC3 configuration.
//!
//! Ported from `api/audio/echo_canceller3_config.h/cc`.

/// Configuration for the Echo Canceller 3.
#[derive(Debug, Clone)]
pub struct EchoCanceller3Config {
    pub buffering: Buffering,
    pub delay: Delay,
    pub filter: Filter,
    pub erle: Erle,
    pub ep_strength: EpStrength,
    pub echo_audibility: EchoAudibility,
    pub render_levels: RenderLevels,
    pub echo_removal_control: EchoRemovalControl,
    pub echo_model: EchoModel,
    pub comfort_noise: ComfortNoise,
    pub suppressor: Suppressor,
    pub multi_channel: MultiChannel,
}

impl Default for EchoCanceller3Config {
    fn default() -> Self {
        Self {
            buffering: Buffering::default(),
            delay: Delay::default(),
            filter: Filter::default(),
            erle: Erle::default(),
            ep_strength: EpStrength::default(),
            echo_audibility: EchoAudibility::default(),
            render_levels: RenderLevels::default(),
            echo_removal_control: EchoRemovalControl::default(),
            echo_model: EchoModel::default(),
            comfort_noise: ComfortNoise::default(),
            suppressor: Suppressor::default(),
            multi_channel: MultiChannel::default(),
        }
    }
}

impl EchoCanceller3Config {
    /// Validates and clamps config parameters to reasonable ranges.
    /// Returns `true` if no changes were needed.
    pub fn validate(&mut self) -> bool {
        let mut ok = true;

        if self.delay.down_sampling_factor != 4 && self.delay.down_sampling_factor != 8 {
            self.delay.down_sampling_factor = 4;
            ok = false;
        }

        ok &= limit_usize(&mut self.delay.default_delay, 0, 5000);
        ok &= limit_usize(&mut self.delay.num_filters, 0, 5000);
        ok &= limit_usize(&mut self.delay.delay_headroom_samples, 0, 5000);
        ok &= limit_usize(&mut self.delay.hysteresis_limit_blocks, 0, 5000);
        ok &= limit_usize(&mut self.delay.fixed_capture_delay_samples, 0, 5000);
        ok &= limit_f32(&mut self.delay.delay_estimate_smoothing, 0.0, 1.0);
        ok &= limit_f32(
            &mut self.delay.delay_candidate_detection_threshold,
            0.0,
            1.0,
        );
        ok &= limit_i32(&mut self.delay.delay_selection_thresholds.initial, 1, 250);
        ok &= limit_i32(&mut self.delay.delay_selection_thresholds.converged, 1, 250);

        ok &= floor_limit_usize(&mut self.filter.refined.length_blocks, 1);
        ok &= limit_f32(&mut self.filter.refined.leakage_converged, 0.0, 1000.0);
        ok &= limit_f32(&mut self.filter.refined.leakage_diverged, 0.0, 1000.0);
        ok &= limit_f32(&mut self.filter.refined.error_floor, 0.0, 1000.0);
        ok &= limit_f32(&mut self.filter.refined.error_ceil, 0.0, 100_000_000.0);
        ok &= limit_f32(&mut self.filter.refined.noise_gate, 0.0, 100_000_000.0);

        ok &= floor_limit_usize(&mut self.filter.refined_initial.length_blocks, 1);
        ok &= limit_f32(
            &mut self.filter.refined_initial.leakage_converged,
            0.0,
            1000.0,
        );
        ok &= limit_f32(
            &mut self.filter.refined_initial.leakage_diverged,
            0.0,
            1000.0,
        );
        ok &= limit_f32(&mut self.filter.refined_initial.error_floor, 0.0, 1000.0);
        ok &= limit_f32(
            &mut self.filter.refined_initial.error_ceil,
            0.0,
            100_000_000.0,
        );
        ok &= limit_f32(
            &mut self.filter.refined_initial.noise_gate,
            0.0,
            100_000_000.0,
        );

        if self.filter.refined.length_blocks < self.filter.refined_initial.length_blocks {
            self.filter.refined_initial.length_blocks = self.filter.refined.length_blocks;
            ok = false;
        }

        ok &= floor_limit_usize(&mut self.filter.coarse.length_blocks, 1);
        ok &= limit_f32(&mut self.filter.coarse.rate, 0.0, 1.0);
        ok &= limit_f32(&mut self.filter.coarse.noise_gate, 0.0, 100_000_000.0);

        ok &= floor_limit_usize(&mut self.filter.coarse_initial.length_blocks, 1);
        ok &= limit_f32(&mut self.filter.coarse_initial.rate, 0.0, 1.0);
        ok &= limit_f32(
            &mut self.filter.coarse_initial.noise_gate,
            0.0,
            100_000_000.0,
        );

        if self.filter.coarse.length_blocks < self.filter.coarse_initial.length_blocks {
            self.filter.coarse_initial.length_blocks = self.filter.coarse.length_blocks;
            ok = false;
        }

        ok &= limit_usize(&mut self.filter.config_change_duration_blocks, 0, 100_000);
        ok &= limit_f32(&mut self.filter.initial_state_seconds, 0.0, 100.0);
        ok &= limit_i32(&mut self.filter.coarse_reset_hangover_blocks, 0, 250_000);

        ok &= limit_f32(&mut self.erle.min, 1.0, 100_000.0);
        ok &= limit_f32(&mut self.erle.max_l, 1.0, 100_000.0);
        ok &= limit_f32(&mut self.erle.max_h, 1.0, 100_000.0);
        if self.erle.min > self.erle.max_l || self.erle.min > self.erle.max_h {
            self.erle.min = self.erle.max_l.min(self.erle.max_h);
            ok = false;
        }
        ok &= limit_usize(
            &mut self.erle.num_sections,
            1,
            self.filter.refined.length_blocks,
        );

        ok &= limit_f32(&mut self.ep_strength.default_gain, 0.0, 1_000_000.0);
        ok &= limit_f32(&mut self.ep_strength.default_len, -1.0, 1.0);
        ok &= limit_f32(&mut self.ep_strength.nearend_len, -1.0, 1.0);

        let max_power = 32768.0f32 * 32768.0;
        ok &= limit_f32(&mut self.echo_audibility.low_render_limit, 0.0, max_power);
        ok &= limit_f32(
            &mut self.echo_audibility.normal_render_limit,
            0.0,
            max_power,
        );
        ok &= limit_f32(&mut self.echo_audibility.floor_power, 0.0, max_power);
        ok &= limit_f32(
            &mut self.echo_audibility.audibility_threshold_lf,
            0.0,
            max_power,
        );
        ok &= limit_f32(
            &mut self.echo_audibility.audibility_threshold_mf,
            0.0,
            max_power,
        );
        ok &= limit_f32(
            &mut self.echo_audibility.audibility_threshold_hf,
            0.0,
            max_power,
        );

        ok &= limit_f32(&mut self.render_levels.active_render_limit, 0.0, max_power);
        ok &= limit_f32(
            &mut self.render_levels.poor_excitation_render_limit,
            0.0,
            max_power,
        );
        ok &= limit_f32(
            &mut self.render_levels.poor_excitation_render_limit_ds8,
            0.0,
            max_power,
        );

        ok &= limit_usize(&mut self.echo_model.noise_floor_hold, 0, 1000);
        ok &= limit_f32(&mut self.echo_model.min_noise_floor_power, 0.0, 2_000_000.0);
        ok &= limit_f32(&mut self.echo_model.stationary_gate_slope, 0.0, 1_000_000.0);
        ok &= limit_f32(&mut self.echo_model.noise_gate_power, 0.0, 1_000_000.0);
        ok &= limit_f32(&mut self.echo_model.noise_gate_slope, 0.0, 1_000_000.0);
        ok &= limit_usize(&mut self.echo_model.render_pre_window_size, 0, 100);
        ok &= limit_usize(&mut self.echo_model.render_post_window_size, 0, 100);

        ok &= limit_f32(&mut self.comfort_noise.noise_floor_dbfs, -200.0, 0.0);

        ok &= limit_usize(&mut self.suppressor.nearend_average_blocks, 1, 5000);

        ok &= validate_tuning(&mut self.suppressor.normal_tuning);
        ok &= validate_tuning(&mut self.suppressor.nearend_tuning);

        ok &= limit_i32(&mut self.suppressor.last_permanent_lf_smoothing_band, 0, 64);
        ok &= limit_i32(&mut self.suppressor.last_lf_smoothing_band, 0, 64);
        ok &= limit_i32(&mut self.suppressor.last_lf_band, 0, 63);
        ok &= limit_i32(
            &mut self.suppressor.first_hf_band,
            self.suppressor.last_lf_band + 1,
            64,
        );

        ok &= limit_f32(
            &mut self.suppressor.dominant_nearend_detection.enr_threshold,
            0.0,
            1_000_000.0,
        );
        ok &= limit_f32(
            &mut self.suppressor.dominant_nearend_detection.snr_threshold,
            0.0,
            1_000_000.0,
        );
        ok &= limit_i32(
            &mut self.suppressor.dominant_nearend_detection.hold_duration,
            0,
            10_000,
        );
        ok &= limit_i32(
            &mut self.suppressor.dominant_nearend_detection.trigger_threshold,
            0,
            10_000,
        );

        ok &= limit_usize(
            &mut self
                .suppressor
                .subband_nearend_detection
                .nearend_average_blocks,
            1,
            1024,
        );
        ok &= limit_usize(
            &mut self.suppressor.subband_nearend_detection.subband1.low,
            0,
            65,
        );
        ok &= limit_usize(
            &mut self.suppressor.subband_nearend_detection.subband1.high,
            self.suppressor.subband_nearend_detection.subband1.low,
            65,
        );
        ok &= limit_usize(
            &mut self.suppressor.subband_nearend_detection.subband2.low,
            0,
            65,
        );
        ok &= limit_usize(
            &mut self.suppressor.subband_nearend_detection.subband2.high,
            self.suppressor.subband_nearend_detection.subband2.low,
            65,
        );
        ok &= limit_f32(
            &mut self.suppressor.subband_nearend_detection.nearend_threshold,
            0.0,
            1.0e24,
        );
        ok &= limit_f32(
            &mut self.suppressor.subband_nearend_detection.snr_threshold,
            0.0,
            1.0e24,
        );

        ok &= limit_f32(
            &mut self.suppressor.high_bands_suppression.enr_threshold,
            0.0,
            1_000_000.0,
        );
        ok &= limit_f32(
            &mut self.suppressor.high_bands_suppression.max_gain_during_echo,
            0.0,
            1.0,
        );
        ok &= limit_f32(
            &mut self
                .suppressor
                .high_bands_suppression
                .anti_howling_activation_threshold,
            0.0,
            max_power,
        );
        ok &= limit_f32(
            &mut self.suppressor.high_bands_suppression.anti_howling_gain,
            0.0,
            1.0,
        );

        ok &= limit_i32(
            &mut self
                .suppressor
                .high_frequency_suppression
                .limiting_gain_band,
            1,
            64,
        );
        ok &= limit_i32(
            &mut self
                .suppressor
                .high_frequency_suppression
                .bands_in_limiting_gain,
            0,
            64 - self
                .suppressor
                .high_frequency_suppression
                .limiting_gain_band,
        );

        ok &= limit_f32(&mut self.suppressor.floor_first_increase, 0.0, 1_000_000.0);

        ok
    }

    /// Creates the default configuration tuned for multichannel.
    pub fn create_default_multichannel_config() -> Self {
        let mut cfg = Self::default();
        cfg.filter.coarse.length_blocks = 11;
        cfg.filter.coarse.rate = 0.95;
        cfg.filter.coarse_initial.length_blocks = 11;
        cfg.filter.coarse_initial.rate = 0.95;
        cfg.suppressor.normal_tuning.max_dec_factor_lf = 0.35;
        cfg.suppressor.normal_tuning.max_inc_factor = 1.5;
        cfg
    }
}

fn validate_tuning(t: &mut Tuning) -> bool {
    let mut ok = true;
    ok &= limit_f32(&mut t.mask_lf.enr_transparent, 0.0, 100.0);
    ok &= limit_f32(&mut t.mask_lf.enr_suppress, 0.0, 100.0);
    ok &= limit_f32(&mut t.mask_lf.emr_transparent, 0.0, 100.0);
    ok &= limit_f32(&mut t.mask_hf.enr_transparent, 0.0, 100.0);
    ok &= limit_f32(&mut t.mask_hf.enr_suppress, 0.0, 100.0);
    ok &= limit_f32(&mut t.mask_hf.emr_transparent, 0.0, 100.0);
    ok &= limit_f32(&mut t.max_inc_factor, 0.0, 100.0);
    ok &= limit_f32(&mut t.max_dec_factor_lf, 0.0, 100.0);
    ok
}

fn limit_f32(value: &mut f32, min: f32, max: f32) -> bool {
    let clamped = value.clamp(min, max);
    let clamped = if clamped.is_finite() { clamped } else { min };
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}

fn limit_usize(value: &mut usize, min: usize, max: usize) -> bool {
    let clamped = (*value).clamp(min, max);
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}

fn limit_i32(value: &mut i32, min: i32, max: i32) -> bool {
    let clamped = (*value).clamp(min, max);
    let unchanged = *value == clamped;
    *value = clamped;
    unchanged
}

fn floor_limit_usize(value: &mut usize, min: usize) -> bool {
    if *value < min {
        *value = min;
        false
    } else {
        true
    }
}

// --- Sub-config structs ---

#[derive(Debug, Clone)]
pub struct Buffering {
    pub excess_render_detection_interval_blocks: usize,
    pub max_allowed_excess_render_blocks: usize,
}

impl Default for Buffering {
    fn default() -> Self {
        Self {
            excess_render_detection_interval_blocks: 250,
            max_allowed_excess_render_blocks: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DelaySelectionThresholds {
    pub initial: i32,
    pub converged: i32,
}

#[derive(Debug, Clone)]
pub struct AlignmentMixing {
    pub downmix: bool,
    pub adaptive_selection: bool,
    pub activity_power_threshold: f32,
    pub prefer_first_two_channels: bool,
}

#[derive(Debug, Clone)]
pub struct Delay {
    pub default_delay: usize,
    pub down_sampling_factor: usize,
    pub num_filters: usize,
    pub delay_headroom_samples: usize,
    pub hysteresis_limit_blocks: usize,
    pub fixed_capture_delay_samples: usize,
    pub delay_estimate_smoothing: f32,
    pub delay_estimate_smoothing_delay_found: f32,
    pub delay_candidate_detection_threshold: f32,
    pub delay_selection_thresholds: DelaySelectionThresholds,
    pub use_external_delay_estimator: bool,
    pub log_warning_on_delay_changes: bool,
    pub render_alignment_mixing: AlignmentMixing,
    pub capture_alignment_mixing: AlignmentMixing,
    pub detect_pre_echo: bool,
}

impl Default for Delay {
    fn default() -> Self {
        Self {
            default_delay: 5,
            down_sampling_factor: 4,
            num_filters: 5,
            delay_headroom_samples: 32,
            hysteresis_limit_blocks: 1,
            fixed_capture_delay_samples: 0,
            delay_estimate_smoothing: 0.7,
            delay_estimate_smoothing_delay_found: 0.7,
            delay_candidate_detection_threshold: 0.2,
            delay_selection_thresholds: DelaySelectionThresholds {
                initial: 5,
                converged: 20,
            },
            use_external_delay_estimator: false,
            log_warning_on_delay_changes: false,
            render_alignment_mixing: AlignmentMixing {
                downmix: false,
                adaptive_selection: true,
                activity_power_threshold: 10000.0,
                prefer_first_two_channels: true,
            },
            capture_alignment_mixing: AlignmentMixing {
                downmix: false,
                adaptive_selection: true,
                activity_power_threshold: 10000.0,
                prefer_first_two_channels: false,
            },
            detect_pre_echo: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefinedConfiguration {
    pub length_blocks: usize,
    pub leakage_converged: f32,
    pub leakage_diverged: f32,
    pub error_floor: f32,
    pub error_ceil: f32,
    pub noise_gate: f32,
}

#[derive(Debug, Clone)]
pub struct CoarseConfiguration {
    pub length_blocks: usize,
    pub rate: f32,
    pub noise_gate: f32,
}

#[derive(Debug, Clone)]
pub struct Filter {
    pub refined: RefinedConfiguration,
    pub coarse: CoarseConfiguration,
    pub refined_initial: RefinedConfiguration,
    pub coarse_initial: CoarseConfiguration,
    pub config_change_duration_blocks: usize,
    pub initial_state_seconds: f32,
    pub coarse_reset_hangover_blocks: i32,
    pub conservative_initial_phase: bool,
    pub enable_coarse_filter_output_usage: bool,
    pub use_linear_filter: bool,
    pub high_pass_filter_echo_reference: bool,
    pub export_linear_aec_output: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            refined: RefinedConfiguration {
                length_blocks: 13,
                leakage_converged: 0.00005,
                leakage_diverged: 0.05,
                error_floor: 0.001,
                error_ceil: 2.0,
                noise_gate: 20_075_344.0,
            },
            coarse: CoarseConfiguration {
                length_blocks: 13,
                rate: 0.7,
                noise_gate: 20_075_344.0,
            },
            refined_initial: RefinedConfiguration {
                length_blocks: 12,
                leakage_converged: 0.005,
                leakage_diverged: 0.5,
                error_floor: 0.001,
                error_ceil: 2.0,
                noise_gate: 20_075_344.0,
            },
            coarse_initial: CoarseConfiguration {
                length_blocks: 12,
                rate: 0.9,
                noise_gate: 20_075_344.0,
            },
            config_change_duration_blocks: 250,
            initial_state_seconds: 2.5,
            coarse_reset_hangover_blocks: 25,
            conservative_initial_phase: false,
            enable_coarse_filter_output_usage: true,
            use_linear_filter: true,
            high_pass_filter_echo_reference: false,
            export_linear_aec_output: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Erle {
    pub min: f32,
    pub max_l: f32,
    pub max_h: f32,
    pub onset_detection: bool,
    pub num_sections: usize,
    pub clamp_quality_estimate_to_zero: bool,
    pub clamp_quality_estimate_to_one: bool,
}

impl Default for Erle {
    fn default() -> Self {
        Self {
            min: 1.0,
            max_l: 4.0,
            max_h: 1.5,
            onset_detection: true,
            num_sections: 1,
            clamp_quality_estimate_to_zero: true,
            clamp_quality_estimate_to_one: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpStrength {
    pub default_gain: f32,
    pub default_len: f32,
    pub nearend_len: f32,
    pub echo_can_saturate: bool,
    pub bounded_erl: bool,
    pub erle_onset_compensation_in_dominant_nearend: bool,
    pub use_conservative_tail_frequency_response: bool,
}

impl Default for EpStrength {
    fn default() -> Self {
        Self {
            default_gain: 1.0,
            default_len: 0.83,
            nearend_len: 0.83,
            echo_can_saturate: true,
            bounded_erl: false,
            erle_onset_compensation_in_dominant_nearend: false,
            use_conservative_tail_frequency_response: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EchoAudibility {
    pub low_render_limit: f32,
    pub normal_render_limit: f32,
    pub floor_power: f32,
    pub audibility_threshold_lf: f32,
    pub audibility_threshold_mf: f32,
    pub audibility_threshold_hf: f32,
    pub use_stationarity_properties: bool,
    pub use_stationarity_properties_at_init: bool,
}

impl Default for EchoAudibility {
    fn default() -> Self {
        Self {
            low_render_limit: 4.0 * 64.0,
            normal_render_limit: 64.0,
            floor_power: 2.0 * 64.0,
            audibility_threshold_lf: 10.0,
            audibility_threshold_mf: 10.0,
            audibility_threshold_hf: 10.0,
            use_stationarity_properties: false,
            use_stationarity_properties_at_init: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderLevels {
    pub active_render_limit: f32,
    pub poor_excitation_render_limit: f32,
    pub poor_excitation_render_limit_ds8: f32,
    pub render_power_gain_db: f32,
}

impl Default for RenderLevels {
    fn default() -> Self {
        Self {
            active_render_limit: 100.0,
            poor_excitation_render_limit: 150.0,
            poor_excitation_render_limit_ds8: 20.0,
            render_power_gain_db: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EchoRemovalControl {
    pub has_clock_drift: bool,
    pub linear_and_stable_echo_path: bool,
}

impl Default for EchoRemovalControl {
    fn default() -> Self {
        Self {
            has_clock_drift: false,
            linear_and_stable_echo_path: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EchoModel {
    pub noise_floor_hold: usize,
    pub min_noise_floor_power: f32,
    pub stationary_gate_slope: f32,
    pub noise_gate_power: f32,
    pub noise_gate_slope: f32,
    pub render_pre_window_size: usize,
    pub render_post_window_size: usize,
    pub model_reverb_in_nonlinear_mode: bool,
}

impl Default for EchoModel {
    fn default() -> Self {
        Self {
            noise_floor_hold: 50,
            min_noise_floor_power: 1_638_400.0,
            stationary_gate_slope: 10.0,
            noise_gate_power: 27509.42,
            noise_gate_slope: 0.3,
            render_pre_window_size: 1,
            render_post_window_size: 1,
            model_reverb_in_nonlinear_mode: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComfortNoise {
    pub noise_floor_dbfs: f32,
}

impl Default for ComfortNoise {
    fn default() -> Self {
        Self {
            noise_floor_dbfs: -96.03406,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaskingThresholds {
    pub enr_transparent: f32,
    pub enr_suppress: f32,
    pub emr_transparent: f32,
}

#[derive(Debug, Clone)]
pub struct Tuning {
    pub mask_lf: MaskingThresholds,
    pub mask_hf: MaskingThresholds,
    pub max_inc_factor: f32,
    pub max_dec_factor_lf: f32,
}

#[derive(Debug, Clone)]
pub struct DominantNearendDetection {
    pub enr_threshold: f32,
    pub enr_exit_threshold: f32,
    pub snr_threshold: f32,
    pub hold_duration: i32,
    pub trigger_threshold: i32,
    pub use_during_initial_phase: bool,
    pub use_unbounded_echo_spectrum: bool,
}

impl Default for DominantNearendDetection {
    fn default() -> Self {
        Self {
            enr_threshold: 0.25,
            enr_exit_threshold: 10.0,
            snr_threshold: 30.0,
            hold_duration: 50,
            trigger_threshold: 12,
            use_during_initial_phase: true,
            use_unbounded_echo_spectrum: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubbandRegion {
    pub low: usize,
    pub high: usize,
}

#[derive(Debug, Clone)]
pub struct SubbandNearendDetection {
    pub nearend_average_blocks: usize,
    pub subband1: SubbandRegion,
    pub subband2: SubbandRegion,
    pub nearend_threshold: f32,
    pub snr_threshold: f32,
}

impl Default for SubbandNearendDetection {
    fn default() -> Self {
        Self {
            nearend_average_blocks: 1,
            subband1: SubbandRegion { low: 1, high: 1 },
            subband2: SubbandRegion { low: 1, high: 1 },
            nearend_threshold: 1.0,
            snr_threshold: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighBandsSuppression {
    pub enr_threshold: f32,
    pub max_gain_during_echo: f32,
    pub anti_howling_activation_threshold: f32,
    pub anti_howling_gain: f32,
}

impl Default for HighBandsSuppression {
    fn default() -> Self {
        Self {
            enr_threshold: 1.0,
            max_gain_during_echo: 1.0,
            anti_howling_activation_threshold: 400.0,
            anti_howling_gain: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighFrequencySuppression {
    pub limiting_gain_band: i32,
    pub bands_in_limiting_gain: i32,
}

impl Default for HighFrequencySuppression {
    fn default() -> Self {
        Self {
            limiting_gain_band: 16,
            bands_in_limiting_gain: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Suppressor {
    pub nearend_average_blocks: usize,
    pub normal_tuning: Tuning,
    pub nearend_tuning: Tuning,
    pub lf_smoothing_during_initial_phase: bool,
    pub last_permanent_lf_smoothing_band: i32,
    pub last_lf_smoothing_band: i32,
    pub last_lf_band: i32,
    pub first_hf_band: i32,
    pub dominant_nearend_detection: DominantNearendDetection,
    pub subband_nearend_detection: SubbandNearendDetection,
    pub use_subband_nearend_detection: bool,
    pub high_bands_suppression: HighBandsSuppression,
    pub high_frequency_suppression: HighFrequencySuppression,
    pub floor_first_increase: f32,
    pub conservative_hf_suppression: bool,
}

impl Default for Suppressor {
    fn default() -> Self {
        Self {
            nearend_average_blocks: 4,
            normal_tuning: Tuning {
                mask_lf: MaskingThresholds {
                    enr_transparent: 0.3,
                    enr_suppress: 0.4,
                    emr_transparent: 0.3,
                },
                mask_hf: MaskingThresholds {
                    enr_transparent: 0.07,
                    enr_suppress: 0.1,
                    emr_transparent: 0.3,
                },
                max_inc_factor: 2.0,
                max_dec_factor_lf: 0.25,
            },
            nearend_tuning: Tuning {
                mask_lf: MaskingThresholds {
                    enr_transparent: 1.09,
                    enr_suppress: 1.1,
                    emr_transparent: 0.3,
                },
                mask_hf: MaskingThresholds {
                    enr_transparent: 0.1,
                    enr_suppress: 0.3,
                    emr_transparent: 0.3,
                },
                max_inc_factor: 2.0,
                max_dec_factor_lf: 0.25,
            },
            lf_smoothing_during_initial_phase: true,
            last_permanent_lf_smoothing_band: 0,
            last_lf_smoothing_band: 5,
            last_lf_band: 5,
            first_hf_band: 8,
            dominant_nearend_detection: DominantNearendDetection::default(),
            subband_nearend_detection: SubbandNearendDetection::default(),
            use_subband_nearend_detection: false,
            high_bands_suppression: HighBandsSuppression::default(),
            high_frequency_suppression: HighFrequencySuppression::default(),
            floor_first_increase: 0.00001,
            conservative_hf_suppression: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultiChannel {
    pub detect_stereo_content: bool,
    pub stereo_detection_threshold: f32,
    pub stereo_detection_timeout_threshold_seconds: i32,
    pub stereo_detection_hysteresis_seconds: f32,
}

impl Default for MultiChannel {
    fn default() -> Self {
        Self {
            detect_stereo_content: true,
            stereo_detection_threshold: 0.0,
            stereo_detection_timeout_threshold_seconds: 300,
            stereo_detection_hysteresis_seconds: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let mut cfg = EchoCanceller3Config::default();
        assert!(cfg.validate());
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let mut cfg = EchoCanceller3Config::default();
        cfg.delay.down_sampling_factor = 3; // invalid, must be 4 or 8
        cfg.erle.min = 200_000.0; // above max of 100_000
        assert!(!cfg.validate());
        assert_eq!(cfg.delay.down_sampling_factor, 4);
        // erle.min gets clamped to 100_000 first, but then the
        // `min > max_l || min > max_h` check clamps it further to
        // min(max_l=4.0, max_h=1.5) = 1.5.
        assert!((cfg.erle.min - 1.5).abs() < 0.01);
    }

    #[test]
    fn multichannel_config_differs_from_default() {
        let def = EchoCanceller3Config::default();
        let mc = EchoCanceller3Config::create_default_multichannel_config();
        assert_eq!(mc.filter.coarse.length_blocks, 11);
        assert_ne!(
            def.filter.coarse.length_blocks,
            mc.filter.coarse.length_blocks
        );
    }
}
