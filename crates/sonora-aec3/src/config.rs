//! AEC3 configuration.
//!
//! Ported from `api/audio/echo_canceller3_config.h/cc`.

/// Configuration for the Echo Canceller 3.
///
/// This is a detailed internal configuration with many tuning parameters.
/// Most users should not need to modify these values — the defaults match
/// the upstream C++ WebRTC configuration. Use [`validate()`](Self::validate) to
/// clamp all parameters to reasonable ranges.
#[derive(Debug, Clone, Default)]
pub struct EchoCanceller3Config {
    /// Render buffering and excess detection settings.
    pub buffering: Buffering,
    /// Delay estimation and alignment settings.
    pub delay: Delay,
    /// Adaptive filter configuration.
    pub filter: Filter,
    /// Echo Return Loss Enhancement (ERLE) estimation settings.
    pub erle: Erle,
    /// Echo path strength estimation settings.
    pub ep_strength: EpStrength,
    /// Echo audibility detection settings.
    pub echo_audibility: EchoAudibility,
    /// Render signal power thresholds.
    pub render_levels: RenderLevels,
    /// Echo removal control settings.
    pub echo_removal_control: EchoRemovalControl,
    /// Echo model parameters.
    pub echo_model: EchoModel,
    /// Comfort noise generation settings.
    pub comfort_noise: ComfortNoise,
    /// Suppression filter settings.
    pub suppressor: Suppressor,
    /// Multi-channel processing settings.
    pub multi_channel: MultiChannel,
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

/// Render buffer excess detection settings.
#[derive(Debug, Clone)]
pub struct Buffering {
    /// Interval in blocks between excess render detection checks (default: 250).
    pub excess_render_detection_interval_blocks: usize,
    /// Maximum allowed excess render blocks before triggering correction (default: 8).
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

/// Thresholds for delay estimator convergence detection.
#[derive(Debug, Clone)]
pub struct DelaySelectionThresholds {
    /// Threshold used during the initial phase before convergence (default: 5).
    pub initial: i32,
    /// Threshold used after the delay estimator has converged (default: 20).
    pub converged: i32,
}

/// Multichannel alignment mixing strategy.
#[derive(Debug, Clone)]
pub struct AlignmentMixing {
    /// Whether to downmix multiple channels to mono for alignment.
    pub downmix: bool,
    /// Whether to adaptively select the best channel for alignment.
    pub adaptive_selection: bool,
    /// Power threshold for considering a channel as active (default: 10000.0).
    pub activity_power_threshold: f32,
    /// Whether to prefer the first two channels when selecting alignment reference.
    pub prefer_first_two_channels: bool,
}

/// Delay estimation and alignment parameters.
#[derive(Debug, Clone)]
pub struct Delay {
    /// Default delay in blocks before estimation converges (default: 5).
    pub default_delay: usize,
    /// Down-sampling factor for the delay estimator; must be 4 or 8 (default: 4).
    pub down_sampling_factor: usize,
    /// Number of correlator filters used for delay estimation (default: 5).
    pub num_filters: usize,
    /// Extra headroom in samples added to the estimated delay (default: 32).
    pub delay_headroom_samples: usize,
    /// Hysteresis in blocks before accepting a new delay estimate (default: 1).
    pub hysteresis_limit_blocks: usize,
    /// Fixed capture delay override in samples; 0 means use estimation (default: 0).
    pub fixed_capture_delay_samples: usize,
    /// Smoothing factor for delay estimates in [0, 1] (default: 0.7).
    pub delay_estimate_smoothing: f32,
    /// Smoothing factor for delay estimates after delay is found (default: 0.7).
    pub delay_estimate_smoothing_delay_found: f32,
    /// Correlation threshold for detecting a delay candidate (default: 0.2).
    pub delay_candidate_detection_threshold: f32,
    /// Convergence thresholds for delay selection.
    pub delay_selection_thresholds: DelaySelectionThresholds,
    /// Whether to use an externally provided delay estimate.
    pub use_external_delay_estimator: bool,
    /// Whether to log warnings when the delay estimate changes.
    pub log_warning_on_delay_changes: bool,
    /// Alignment mixing settings for the render signal.
    pub render_alignment_mixing: AlignmentMixing,
    /// Alignment mixing settings for the capture signal.
    pub capture_alignment_mixing: AlignmentMixing,
    /// Whether to detect and compensate for pre-echo artifacts.
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

/// Configuration for the refined (main) adaptive filter.
#[derive(Debug, Clone)]
pub struct RefinedConfiguration {
    /// Filter length in blocks (default: 13, initial: 12).
    pub length_blocks: usize,
    /// Leakage factor when the filter has converged (default: 0.00005).
    pub leakage_converged: f32,
    /// Leakage factor when the filter has diverged (default: 0.05).
    pub leakage_diverged: f32,
    /// Minimum error floor to prevent division by zero (default: 0.001).
    pub error_floor: f32,
    /// Maximum error ceiling to limit adaptation (default: 2.0).
    pub error_ceil: f32,
    /// Power threshold below which adaptation is gated (default: 20075344.0).
    pub noise_gate: f32,
}

/// Configuration for the coarse (shadow) adaptive filter.
#[derive(Debug, Clone)]
pub struct CoarseConfiguration {
    /// Filter length in blocks (default: 13, initial: 12).
    pub length_blocks: usize,
    /// Adaptation step-size rate in [0, 1] (default: 0.7, initial: 0.9).
    pub rate: f32,
    /// Power threshold below which adaptation is gated (default: 20075344.0).
    pub noise_gate: f32,
}

/// Adaptive filter adaptation settings.
#[derive(Debug, Clone)]
pub struct Filter {
    /// Refined (main) adaptive filter configuration.
    pub refined: RefinedConfiguration,
    /// Coarse (shadow) adaptive filter configuration.
    pub coarse: CoarseConfiguration,
    /// Refined filter configuration used during the initial phase.
    pub refined_initial: RefinedConfiguration,
    /// Coarse filter configuration used during the initial phase.
    pub coarse_initial: CoarseConfiguration,
    /// Duration in blocks for transitioning between config changes (default: 250).
    pub config_change_duration_blocks: usize,
    /// Duration in seconds of the initial adaptation phase (default: 2.5).
    pub initial_state_seconds: f32,
    /// Hangover in blocks after a coarse filter reset (default: 25).
    pub coarse_reset_hangover_blocks: i32,
    /// Whether to use a conservative strategy during the initial phase.
    pub conservative_initial_phase: bool,
    /// Whether to allow using the coarse filter output for echo subtraction.
    pub enable_coarse_filter_output_usage: bool,
    /// Whether to use the linear adaptive filter for echo removal.
    pub use_linear_filter: bool,
    /// Whether to high-pass filter the echo reference signal.
    pub high_pass_filter_echo_reference: bool,
    /// Whether to export the linear AEC output for external use.
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

/// Echo Return Loss Enhancement (ERLE) estimation parameters.
#[derive(Debug, Clone)]
pub struct Erle {
    /// Minimum ERLE value in linear scale (default: 1.0).
    pub min: f32,
    /// Maximum ERLE for LF bands in linear scale (default: 4.0).
    pub max_l: f32,
    /// Maximum ERLE for HF bands in linear scale (default: 1.5).
    pub max_h: f32,
    /// Whether to use onset detection to reset ERLE estimates.
    pub onset_detection: bool,
    /// Number of frequency sections for ERLE estimation (default: 1).
    pub num_sections: usize,
    /// Whether to clamp the filter quality estimate at zero.
    pub clamp_quality_estimate_to_zero: bool,
    /// Whether to clamp the filter quality estimate at one.
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

/// Echo path strength and suppression gain parameters.
#[derive(Debug, Clone)]
pub struct EpStrength {
    /// Default echo path gain applied to the suppressor (default: 1.0).
    pub default_gain: f32,
    /// Echo path tail length as a fraction in [-1, 1] (default: 0.83).
    pub default_len: f32,
    /// Echo path tail length during dominant nearend in [-1, 1] (default: 0.83).
    pub nearend_len: f32,
    /// Whether the echo path can introduce saturation/clipping.
    pub echo_can_saturate: bool,
    /// Whether to bound the ERL estimate.
    pub bounded_erl: bool,
    /// Whether to compensate ERLE onset during dominant nearend detection.
    pub erle_onset_compensation_in_dominant_nearend: bool,
    /// Whether to use a conservative tail frequency response estimate.
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

/// Echo audibility detection parameters.
#[derive(Debug, Clone)]
pub struct EchoAudibility {
    /// Render power threshold for low-activity detection (default: 256.0).
    pub low_render_limit: f32,
    /// Render power threshold for normal-activity detection (default: 64.0).
    pub normal_render_limit: f32,
    /// Minimum floor power for audibility computation (default: 128.0).
    pub floor_power: f32,
    /// Audibility threshold for LF bands (default: 10.0).
    pub audibility_threshold_lf: f32,
    /// Audibility threshold for mid-frequency bands (default: 10.0).
    pub audibility_threshold_mf: f32,
    /// Audibility threshold for HF bands (default: 10.0).
    pub audibility_threshold_hf: f32,
    /// Whether to use signal stationarity properties for audibility detection.
    pub use_stationarity_properties: bool,
    /// Whether to use stationarity properties during the initial phase.
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

/// Render signal level thresholds.
#[derive(Debug, Clone)]
pub struct RenderLevels {
    /// Power threshold above which the render signal is considered active (default: 100.0).
    pub active_render_limit: f32,
    /// Power threshold below which render excitation is considered poor (default: 150.0).
    pub poor_excitation_render_limit: f32,
    /// Poor excitation threshold for 8x down-sampled signals (default: 20.0).
    pub poor_excitation_render_limit_ds8: f32,
    /// Gain in dB applied to the render power estimate (default: 0.0).
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

/// Selects the transparent mode algorithm for AEC3.
///
/// Transparent mode detects scenarios where no echo is present (e.g. headset
/// use) and reduces suppression accordingly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TransparentModeType {
    /// Counter-based heuristic (the default).
    #[default]
    Legacy,
    /// Two-state Hidden Markov Model classifier.
    ///
    /// Uses Bayesian inference on filter convergence observations to estimate
    /// the probability of being in a "transparent" (no-echo) state. Generally
    /// more responsive than Legacy mode.
    Hmm,
}

/// Top-level echo removal control settings.
#[derive(Debug, Clone, Default)]
pub struct EchoRemovalControl {
    /// Whether the render and capture clocks are drifting relative to each other.
    pub has_clock_drift: bool,
    /// Whether the echo path is linear and stable (e.g. loopback scenarios).
    pub linear_and_stable_echo_path: bool,
    /// Which transparent mode algorithm to use.
    pub transparent_mode: TransparentModeType,
}

/// Echo and noise model parameters.
#[derive(Debug, Clone)]
pub struct EchoModel {
    /// Number of blocks to hold the noise floor estimate (default: 50).
    pub noise_floor_hold: usize,
    /// Minimum noise floor power level (default: 1638400.0).
    pub min_noise_floor_power: f32,
    /// Slope of the stationarity gate function (default: 10.0).
    pub stationary_gate_slope: f32,
    /// Power threshold for the noise gate (default: 27509.42).
    pub noise_gate_power: f32,
    /// Slope of the noise gate transition (default: 0.3).
    pub noise_gate_slope: f32,
    /// Number of blocks before the current block in the render window (default: 1).
    pub render_pre_window_size: usize,
    /// Number of blocks after the current block in the render window (default: 1).
    pub render_post_window_size: usize,
    /// Whether to model reverb in nonlinear processing mode.
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

/// Comfort noise generation settings.
#[derive(Debug, Clone)]
pub struct ComfortNoise {
    /// Noise floor level in dBFS for comfort noise injection (default: -96.03).
    pub noise_floor_dbfs: f32,
}

impl Default for ComfortNoise {
    fn default() -> Self {
        Self {
            noise_floor_dbfs: -96.03406,
        }
    }
}

/// Suppression masking thresholds based on ENR and EMR.
#[derive(Debug, Clone)]
pub struct MaskingThresholds {
    /// ENR threshold below which the signal is treated as transparent (no suppression).
    pub enr_transparent: f32,
    /// ENR threshold above which full suppression is applied.
    pub enr_suppress: f32,
    /// EMR threshold below which the signal is treated as transparent.
    pub emr_transparent: f32,
}

/// Suppressor tuning with LF/HF masking thresholds and gain limits.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// Masking thresholds for LF bands.
    pub mask_lf: MaskingThresholds,
    /// Masking thresholds for HF bands.
    pub mask_hf: MaskingThresholds,
    /// Maximum gain increase factor per block (default: 2.0).
    pub max_inc_factor: f32,
    /// Maximum gain decrease factor for LF bands per block (default: 0.25).
    pub max_dec_factor_lf: f32,
}

/// Dominant nearend speech detection parameters.
#[derive(Debug, Clone)]
pub struct DominantNearendDetection {
    /// ENR threshold to enter nearend-dominant state (default: 0.25).
    pub enr_threshold: f32,
    /// ENR threshold to exit nearend-dominant state (default: 10.0).
    pub enr_exit_threshold: f32,
    /// SNR threshold for nearend detection (default: 30.0).
    pub snr_threshold: f32,
    /// Number of blocks to hold the nearend-dominant state (default: 50).
    pub hold_duration: i32,
    /// Number of bands that must exceed the threshold to trigger (default: 12).
    pub trigger_threshold: i32,
    /// Whether to use nearend detection during the initial adaptation phase.
    pub use_during_initial_phase: bool,
    /// Whether to use an unbounded echo spectrum estimate for detection.
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

/// A frequency subband range specified by low and high bin indices.
#[derive(Debug, Clone)]
pub struct SubbandRegion {
    /// Lower frequency bin index (inclusive).
    pub low: usize,
    /// Upper frequency bin index (inclusive).
    pub high: usize,
}

/// Subband-based nearend speech detection parameters.
#[derive(Debug, Clone)]
pub struct SubbandNearendDetection {
    /// Number of blocks to average for nearend power estimation (default: 1).
    pub nearend_average_blocks: usize,
    /// First subband region for nearend detection.
    pub subband1: SubbandRegion,
    /// Second subband region for nearend detection.
    pub subband2: SubbandRegion,
    /// Nearend power threshold for detection (default: 1.0).
    pub nearend_threshold: f32,
    /// SNR threshold for subband nearend detection (default: 1.0).
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

/// High-band suppression and anti-howling settings.
#[derive(Debug, Clone)]
pub struct HighBandsSuppression {
    /// ENR threshold for activating high-band suppression (default: 1.0).
    pub enr_threshold: f32,
    /// Maximum gain applied to high bands during echo (default: 1.0).
    pub max_gain_during_echo: f32,
    /// Power threshold to activate anti-howling protection (default: 400.0).
    pub anti_howling_activation_threshold: f32,
    /// Gain applied when anti-howling is active (default: 1.0).
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

/// HF gain limiting parameters.
#[derive(Debug, Clone)]
pub struct HighFrequencySuppression {
    /// Starting band index for HF gain limiting (default: 16).
    pub limiting_gain_band: i32,
    /// Number of bands over which HF gain limiting is applied (default: 1).
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

/// Top-level suppressor configuration.
#[derive(Debug, Clone)]
pub struct Suppressor {
    /// Number of blocks to average for nearend power estimation (default: 4).
    pub nearend_average_blocks: usize,
    /// Tuning parameters used during normal (non-nearend) operation.
    pub normal_tuning: Tuning,
    /// Tuning parameters used during dominant nearend conditions.
    pub nearend_tuning: Tuning,
    /// Whether to apply LF gain smoothing during the initial adaptation phase.
    pub lf_smoothing_during_initial_phase: bool,
    /// Last band index with permanent LF gain smoothing (default: 0).
    pub last_permanent_lf_smoothing_band: i32,
    /// Last band index with LF gain smoothing (default: 5).
    pub last_lf_smoothing_band: i32,
    /// Last band index considered as LF (default: 5).
    pub last_lf_band: i32,
    /// First band index considered as HF (default: 8).
    pub first_hf_band: i32,
    /// Dominant nearend speech detection settings.
    pub dominant_nearend_detection: DominantNearendDetection,
    /// Subband-based nearend detection settings.
    pub subband_nearend_detection: SubbandNearendDetection,
    /// Whether to use subband nearend detection instead of dominant nearend detection.
    pub use_subband_nearend_detection: bool,
    /// High-band suppression and anti-howling settings.
    pub high_bands_suppression: HighBandsSuppression,
    /// HF gain limiting settings.
    pub high_frequency_suppression: HighFrequencySuppression,
    /// Initial suppression gain floor increase step (default: 0.00001).
    pub floor_first_increase: f32,
    /// Whether to apply conservative suppression in HF bands.
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

/// Multichannel and stereo content detection settings.
#[derive(Debug, Clone)]
pub struct MultiChannel {
    /// Whether to detect stereo content and adapt processing accordingly.
    pub detect_stereo_content: bool,
    /// Power difference threshold for stereo detection (default: 0.0).
    pub stereo_detection_threshold: f32,
    /// Timeout in seconds before resetting stereo detection (default: 300).
    pub stereo_detection_timeout_threshold_seconds: i32,
    /// Hysteresis duration in seconds for stereo detection state changes (default: 2.0).
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
}
