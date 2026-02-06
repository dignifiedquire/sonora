//! Automatic Gain Control 2 (AGC2) for WebRTC Audio Processing.
//!
//! Contains adaptive digital gain control with RNN-based voice activity
//! detection, limiter, and clipping prediction. This is the modern AGC
//! used by the default audio processing pipeline.

pub(crate) mod adaptive_digital_gain_controller;
pub(crate) mod biquad_filter;
pub mod common;
pub(crate) mod fixed_digital_level_estimator;
pub(crate) mod gain_applier;
pub(crate) mod interpolated_gain_curve;
pub(crate) mod limiter;
pub(crate) mod limiter_db_gain_curve;
pub(crate) mod noise_level_estimator;
pub mod rnn_vad;
pub(crate) mod saturation_protector;
pub(crate) mod saturation_protector_buffer;
pub(crate) mod speech_level_estimator;
pub(crate) mod speech_probability_buffer;
pub(crate) mod vad_wrapper;
