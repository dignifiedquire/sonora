//! C-compatible FFI layer for the audio processing pipeline.
//!
//! This module exposes `extern "C"` functions and `#[repr(C)]` types that
//! allow C and C++ consumers to use the Rust audio processing engine.
//!
//! # Symbol prefix
//!
//! - Functions: `wap_*`
//! - Types: `Wap*`
//!
//! # Thread safety
//!
//! **NOT thread-safe.** All calls on the same [`WapAudioProcessing`] handle
//! must be serialized by the caller, matching the C++ API contract.

pub mod types;

mod conversions;
pub mod functions;
mod panic_guard;
