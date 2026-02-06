//! C++ FFI bindings to WebRTC Audio Processing for comparison testing.
//!
//! Uses cxx to provide safe Rust access to the C++ AudioProcessing
//! implementation. Used exclusively for property-based testing to verify
//! that the Rust port produces identical output to the C++ reference.
