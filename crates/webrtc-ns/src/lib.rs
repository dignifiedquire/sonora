//! Noise Suppression for WebRTC Audio Processing.
//!
//! Implements Wiener filtering based on noise estimation with
//! configurable suppression levels.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/`

pub mod config;
pub mod fast_math;
pub mod histograms;
pub mod ns_fft;
pub mod prior_signal_model;
pub mod signal_model;
pub mod suppression_params;
