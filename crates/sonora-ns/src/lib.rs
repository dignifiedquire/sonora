//! Noise Suppression for WebRTC Audio Processing.
//!
//! Implements Wiener filtering based on noise estimation with
//! configurable suppression levels.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/`

pub mod config;
pub(crate) mod fast_math;
pub(crate) mod histograms;
pub(crate) mod noise_estimator;
pub mod noise_suppressor;
pub(crate) mod ns_fft;
pub(crate) mod prior_signal_model;
pub(crate) mod prior_signal_model_estimator;
pub(crate) mod quantile_noise_estimator;
pub(crate) mod signal_model;
pub(crate) mod signal_model_estimator;
pub(crate) mod speech_probability_estimator;
pub(crate) mod suppression_params;
pub(crate) mod wiener_filter;
