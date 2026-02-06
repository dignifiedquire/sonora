//! FFT implementations for WebRTC Audio Processing.
//!
//! Three FFT algorithms ported from C/C++:
//!
//! - [`ooura_fft`] — fixed 128-point real FFT (used by AEC3)
//! - [`Fft4g`](fft4g::Fft4g) — variable-size real FFT, power-of-2 (used by NS, VAD)
//! - [`Pffft`](pffft::Pffft) — variable-size real/complex FFT, composite sizes (used by AGC2 RNN-VAD)

#![deny(unsafe_code)]

pub mod fft4g;
pub mod ooura_fft;
pub mod pffft;
