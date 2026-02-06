//! Automatic Gain Control 2 (AGC2) for WebRTC Audio Processing.
//!
//! Contains adaptive digital gain control with RNN-based voice activity
//! detection, limiter, and clipping prediction. This is the modern AGC
//! used by the default audio processing pipeline.

pub mod common;
pub mod rnn_vad;
pub(crate) mod vad_wrapper;
