//! Automatic Gain Control 2 (AGC2) for WebRTC Audio Processing.
//!
//! Contains adaptive digital gain control with RNN-based voice activity
//! detection, limiter, and clipping prediction. This is the modern AGC
//! used by the default audio processing pipeline.

pub mod adaptive_digital_gain_controller;
pub mod biquad_filter;
pub mod clipping_predictor;
#[allow(dead_code, reason = "consumed by clipping_predictor")]
pub mod clipping_predictor_level_buffer;
pub mod common;
pub mod fixed_digital_level_estimator;
pub mod gain_applier;
pub mod interpolated_gain_curve;
pub mod limiter;
pub mod limiter_db_gain_curve;
pub mod noise_level_estimator;
pub mod rnn_vad;
pub mod saturation_protector;
pub mod saturation_protector_buffer;
pub mod speech_level_estimator;
pub mod speech_probability_buffer;
pub mod vad_wrapper;
