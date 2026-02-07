//! C-compatible type definitions for the audio processing C API.
//!
//! All types here are `#[repr(C)]` and are safe to pass across FFI boundaries.

use crate::AudioProcessing;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/// Error codes returned by C API functions.
///
/// `0` = success, negative = error.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WapError {
    /// Operation succeeded.
    None = 0,
    /// Null pointer passed to a function that requires non-null.
    NullPointer = -1,
    /// Internal error (panic caught at FFI boundary).
    Internal = -2,
    /// Bad sample rate.
    BadSampleRate = -3,
    /// Bad number of channels.
    BadNumberChannels = -4,
    /// A stream parameter was out of range and was clamped.
    BadStreamParameter = -5,
    /// Invalid data length.
    BadDataLength = -6,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Noise suppression aggressiveness level.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WapNoiseSuppressionLevel {
    Low = 0,
    Moderate = 1,
    High = 2,
    VeryHigh = 3,
}

/// Downmix method for multi-channel capture.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WapDownmixMethod {
    AverageChannels = 0,
    UseFirstChannel = 1,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Flat configuration struct for the audio processing pipeline.
///
/// Obtain a default-initialized instance via `wap_config_default()`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WapConfig {
    // -- Pipeline --
    pub pipeline_maximum_internal_processing_rate: i32,
    pub pipeline_multi_channel_render: bool,
    pub pipeline_multi_channel_capture: bool,
    pub pipeline_capture_downmix_method: WapDownmixMethod,

    // -- Pre-amplifier --
    pub pre_amplifier_enabled: bool,
    pub pre_amplifier_fixed_gain_factor: f32,

    // -- Capture level adjustment --
    pub capture_level_adjustment_enabled: bool,
    pub capture_level_adjustment_pre_gain_factor: f32,
    pub capture_level_adjustment_post_gain_factor: f32,
    pub analog_mic_gain_emulation_enabled: bool,
    pub analog_mic_gain_emulation_initial_level: i32,

    // -- High-pass filter --
    pub high_pass_filter_enabled: bool,
    pub high_pass_filter_apply_in_full_band: bool,

    // -- Echo canceller --
    pub echo_canceller_enabled: bool,
    pub echo_canceller_enforce_high_pass_filtering: bool,

    // -- Noise suppression --
    pub noise_suppression_enabled: bool,
    pub noise_suppression_level: WapNoiseSuppressionLevel,
    pub noise_suppression_analyze_linear_aec_output_when_available: bool,

    // -- Gain controller 2 --
    pub gain_controller2_enabled: bool,
    pub gain_controller2_fixed_digital_gain_db: f32,
    pub gain_controller2_adaptive_digital_enabled: bool,
    pub gain_controller2_adaptive_digital_headroom_db: f32,
    pub gain_controller2_adaptive_digital_max_gain_db: f32,
    pub gain_controller2_adaptive_digital_initial_gain_db: f32,
    pub gain_controller2_adaptive_digital_max_gain_change_db_per_second: f32,
    pub gain_controller2_adaptive_digital_max_output_noise_level_dbfs: f32,
    pub gain_controller2_input_volume_controller_enabled: bool,
}

// ---------------------------------------------------------------------------
// Stream configuration
// ---------------------------------------------------------------------------

/// Audio stream configuration (sample rate and channel count).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WapStreamConfig {
    pub sample_rate_hz: i32,
    pub num_channels: i32,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Audio processing statistics.
///
/// Each statistic has a `has_*` boolean. When `false`, the corresponding
/// value field is meaningless.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WapStats {
    pub has_echo_return_loss: bool,
    pub echo_return_loss: f64,

    pub has_echo_return_loss_enhancement: bool,
    pub echo_return_loss_enhancement: f64,

    pub has_divergent_filter_fraction: bool,
    pub divergent_filter_fraction: f64,

    pub has_delay_median_ms: bool,
    pub delay_median_ms: i32,

    pub has_delay_standard_deviation_ms: bool,
    pub delay_standard_deviation_ms: i32,

    pub has_residual_echo_likelihood: bool,
    pub residual_echo_likelihood: f64,

    pub has_residual_echo_likelihood_recent_max: bool,
    pub residual_echo_likelihood_recent_max: f64,

    pub has_delay_ms: bool,
    pub delay_ms: i32,
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to the audio processing engine.
///
/// Created via `wap_create()` or `wap_create_with_config()`.
/// Destroyed via `wap_destroy()`.
///
/// **NOT thread-safe**: all calls on the same handle must be serialized.
pub struct WapAudioProcessing {
    pub(crate) inner: AudioProcessing,
}
