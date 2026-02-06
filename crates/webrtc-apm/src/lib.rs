//! WebRTC Audio Processing Module — Rust port.
//!
//! Provides echo cancellation, noise suppression, automatic gain control,
//! and other audio processing capabilities.

pub(crate) mod splitting_filter;
pub(crate) mod three_band_filter_bank;
