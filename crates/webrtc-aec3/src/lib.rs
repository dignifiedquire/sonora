//! WebRTC Echo Canceller 3 (AEC3) — Rust port.
//!
//! This crate provides a pure-Rust implementation of WebRTC's modern echo
//! canceller (AEC3), ported from the C++ source at
//! `modules/audio_processing/aec3/`.

pub(crate) mod aec3_fft;
pub(crate) mod block;
pub(crate) mod block_buffer;
pub(crate) mod circular_buffer;
pub(crate) mod common;
pub(crate) mod config;
pub(crate) mod delay_estimate;
pub(crate) mod echo_path_variability;
pub(crate) mod fft_buffer;
pub(crate) mod fft_data;
pub(crate) mod spectrum_buffer;
