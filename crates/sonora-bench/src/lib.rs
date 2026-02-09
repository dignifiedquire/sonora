//! C++ comparison testing and benchmarking infrastructure for sonora.
//!
//! Provides audio buffer generators and comparison utilities for
//! verifying the Rust port against the C++ reference implementation.

pub mod comparison;
pub mod generators;

pub use proptest;
pub use test_strategy;
