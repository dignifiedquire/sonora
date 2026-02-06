//! WebRTC Audio Processing Module — Rust port.
//!
//! Provides echo cancellation, noise suppression, automatic gain control,
//! and other audio processing capabilities.

pub(crate) mod audio_buffer;
pub(crate) mod splitting_filter;
pub(crate) mod stream_config;
pub(crate) mod three_band_filter_bank;
