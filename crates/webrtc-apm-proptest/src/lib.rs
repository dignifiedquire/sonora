//! Property-based test framework for WebRTC Audio Processing.
//!
//! Provides audio buffer generators and comparison utilities for
//! verifying the Rust port against the C++ reference implementation.
//!
//! # Usage
//!
//! ```ignore
//! use webrtc_apm_proptest::generators::*;
//! use test_strategy::proptest;
//!
//! #[proptest]
//! fn my_test(#[strategy(audio_frame_f32(16000))] frame: Vec<f32>) {
//!     assert_eq!(frame.len(), 160);
//! }
//! ```

pub mod comparison;
pub mod generators;

pub use proptest;
pub use test_strategy;
