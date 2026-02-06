//! Noise Suppression for WebRTC Audio Processing.
//!
//! Implements Wiener filtering based on noise estimation with
//! configurable suppression levels.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/`

pub mod config;
pub mod fast_math;
pub mod histograms;
pub mod noise_estimator;
pub mod noise_suppressor;
pub mod ns_fft;
pub mod prior_signal_model;
pub mod prior_signal_model_estimator;
pub mod quantile_noise_estimator;
pub mod signal_model;
pub mod signal_model_estimator;
pub mod speech_probability_estimator;
pub mod suppression_params;
pub mod wiener_filter;
