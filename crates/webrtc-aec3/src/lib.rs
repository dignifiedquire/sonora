//! WebRTC Echo Canceller 3 (AEC3) — Rust port.
//!
//! This crate provides a pure-Rust implementation of WebRTC's modern echo
//! canceller (AEC3), ported from the C++ source at
//! `modules/audio_processing/aec3/`.

pub(crate) mod adaptive_fir_filter;
pub(crate) mod adaptive_fir_filter_erl;
pub(crate) mod aec3_fft;
pub(crate) mod aec_state;
pub(crate) mod alignment_mixer;
pub(crate) mod block;
pub(crate) mod block_buffer;
pub(crate) mod block_framer;
pub(crate) mod cascaded_biquad_filter;
pub(crate) mod circular_buffer;
pub(crate) mod clockdrift_detector;
pub(crate) mod coarse_filter_update_gain;
pub(crate) mod comfort_noise_generator;
pub(crate) mod common;
pub(crate) mod config;
pub(crate) mod decimator;
pub(crate) mod delay_estimate;
pub(crate) mod downsampled_render_buffer;
pub(crate) mod echo_audibility;
pub(crate) mod echo_path_delay_estimator;
pub(crate) mod echo_path_variability;
pub(crate) mod echo_remover;
pub(crate) mod echo_remover_metrics;
pub(crate) mod erl_estimator;
pub(crate) mod erle_estimator;
pub(crate) mod fft_buffer;
pub(crate) mod fft_data;
pub(crate) mod filter_analyzer;
pub(crate) mod frame_blocker;
pub(crate) mod fullband_erle_estimator;
pub(crate) mod matched_filter;
pub(crate) mod matched_filter_lag_aggregator;
pub(crate) mod moving_average;
pub(crate) mod multi_channel_content_detector;
pub(crate) mod nearend_detector;
pub(crate) mod refined_filter_update_gain;
pub(crate) mod render_buffer;
pub(crate) mod render_delay_buffer;
pub(crate) mod render_delay_controller;
pub(crate) mod render_delay_controller_metrics;
pub(crate) mod render_signal_analyzer;
pub(crate) mod residual_echo_estimator;
pub(crate) mod reverb_decay_estimator;
pub(crate) mod reverb_frequency_response;
pub(crate) mod reverb_model;
pub(crate) mod reverb_model_estimator;
pub(crate) mod signal_dependent_erle_estimator;
pub(crate) mod spectrum_buffer;
pub(crate) mod stationarity_estimator;
pub(crate) mod subband_erle_estimator;
pub(crate) mod subtractor;
pub(crate) mod subtractor_output;
pub(crate) mod subtractor_output_analyzer;
pub(crate) mod suppression_filter;
pub(crate) mod suppression_gain;
pub(crate) mod transparent_mode;
pub(crate) mod vector_math;
