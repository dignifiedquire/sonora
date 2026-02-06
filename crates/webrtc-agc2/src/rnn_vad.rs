//! RNN-based Voice Activity Detector for AGC2.
//!
//! A neural network-based VAD that uses spectral features and pitch
//! analysis to estimate speech probability. Operates at 24kHz internally.

pub mod activations;
pub mod auto_correlation;
pub mod common;
pub mod fc_layer;
pub mod gru_layer;
pub mod lp_residual;
pub mod pitch_search;
pub mod pitch_search_internal;
pub mod ring_buffer;
pub mod sequence_buffer;
pub mod spectral_features;
pub mod spectral_features_internal;
pub mod symmetric_matrix_buffer;
pub mod vector_math;
pub mod weights;
