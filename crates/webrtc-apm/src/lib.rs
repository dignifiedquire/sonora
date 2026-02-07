//! WebRTC Audio Processing Module — Rust port.
//!
//! Provides echo cancellation, noise suppression, automatic gain control,
//! and other audio processing capabilities.
//!
//! # Quick Start
//!
//! ```ignore
//! use webrtc_apm::{AudioProcessing, Config, StreamConfig};
//!
//! let mut config = Config::default();
//! config.echo_canceller.enabled = true;
//! config.noise_suppression.enabled = true;
//!
//! let mut apm = AudioProcessing::builder().config(config).build();
//! let stream = StreamConfig::new(16000, 1);
//!
//! // For each ~10 ms audio frame:
//! // 1. Feed far-end (render) audio:
//! // apm.process_reverse_stream_f32(&src, &stream, &stream, &mut dest)?;
//! // 2. Process near-end (capture) audio:
//! // apm.process_stream_f32(&src, &stream, &stream, &mut dest)?;
//! ```

pub(crate) mod audio_buffer;
pub(crate) mod audio_converter;
mod audio_processing;
pub(crate) mod audio_processing_impl;
pub(crate) mod audio_samples_scaler;
pub(crate) mod capture_levels_adjuster;
pub mod config;
pub(crate) mod config_selector;
pub(crate) mod echo_canceller3;
pub(crate) mod echo_detector;
pub mod ffi;
pub(crate) mod gain_controller2;
pub(crate) mod high_pass_filter;
pub(crate) mod input_volume_controller;
pub(crate) mod residual_echo_detector;
pub(crate) mod rms_level;
pub(crate) mod splitting_filter;
pub mod stats;
pub(crate) mod stream_config;
pub(crate) mod submodule_states;
pub(crate) mod swap_queue;
pub(crate) mod three_band_filter_bank;

// Public re-exports.
pub use audio_processing::{AudioProcessing, AudioProcessingBuilder, Error};
pub use config::{Config, RuntimeSetting};
pub use stats::AudioProcessingStats;
pub use stream_config::StreamConfig;
