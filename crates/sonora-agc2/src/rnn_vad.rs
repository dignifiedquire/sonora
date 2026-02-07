//! RNN-based Voice Activity Detector for AGC2.
//!
//! A neural network-based VAD that uses spectral features and pitch
//! analysis to estimate speech probability. Operates at 24kHz internally.

pub(crate) mod activations;
pub(crate) mod auto_correlation;
pub(crate) mod common;
pub(crate) mod fc_layer;
pub(crate) mod features_extraction;
pub(crate) mod gru_layer;
pub(crate) mod lp_residual;
pub(crate) mod pitch_search;
pub(crate) mod pitch_search_internal;
pub(crate) mod ring_buffer;
pub(crate) mod rnn;
pub(crate) mod sequence_buffer;
pub(crate) mod spectral_features;
pub(crate) mod spectral_features_internal;
pub(crate) mod symmetric_matrix_buffer;
pub(crate) mod vector_math;
pub(crate) mod weights;
