//! RNN-based Voice Activity Detector for AGC2.
//!
//! A neural network-based VAD that uses spectral features and pitch
//! analysis to estimate speech probability. Operates at 24kHz internally.

pub mod activations;
pub mod common;
pub mod ring_buffer;
pub mod sequence_buffer;
pub mod symmetric_matrix_buffer;
pub mod vector_math;
pub mod weights;
