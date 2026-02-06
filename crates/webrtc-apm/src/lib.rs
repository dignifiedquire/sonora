//! WebRTC Audio Processing Module — Rust port.
//!
//! Provides echo cancellation, noise suppression, automatic gain control,
//! and other audio processing capabilities.

pub(crate) mod audio_buffer;
pub(crate) mod audio_samples_scaler;
pub(crate) mod capture_levels_adjuster;
pub(crate) mod echo_detector;
pub(crate) mod high_pass_filter;
pub(crate) mod input_volume_controller;
pub(crate) mod residual_echo_detector;
pub(crate) mod rms_level;
pub(crate) mod splitting_filter;
pub(crate) mod stream_config;
pub(crate) mod three_band_filter_bank;
