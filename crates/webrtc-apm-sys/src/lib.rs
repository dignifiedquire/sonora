//! C++ FFI bindings to WebRTC Audio Processing for comparison testing.
//!
//! Uses cxx to provide safe Rust access to the C++ AudioProcessing
//! implementation. Used exclusively for property-based testing to verify
//! that the Rust port produces identical output to the C++ reference.
//!
//! # Usage
//!
//! This crate requires the C++ library to be pre-built:
//!
//! ```bash
//! meson setup builddir -Dtests=enabled
//! ninja -C builddir
//! ```
//!
//! Then enable the `cxx-bridge` feature:
//!
//! ```bash
//! cargo test -p webrtc-apm-sys --features cxx-bridge
//! ```
//!
//! # Design
//!
//! Since `AudioProcessing` is an abstract C++ class with virtual methods,
//! we use a thin C++ shim layer (`cpp/shim.h`, `cpp/shim.cc`) that wraps
//! the virtual interface into concrete free functions that cxx can bridge.
//!
//! Shim functions are added incrementally as each porting phase needs
//! comparison testing for new components.

#[cfg(feature = "cxx-bridge")]
mod bridge;

#[cfg(feature = "cxx-bridge")]
pub use bridge::*;
