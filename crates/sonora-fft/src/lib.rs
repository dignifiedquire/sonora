//! FFT implementations for WebRTC Audio Processing.
//!
//! Three FFT algorithms ported from C/C++:
//!
//! - [`ooura_fft`] — fixed 128-point real FFT (used by AEC3)
//! - [`Fft4g`](fft4g::Fft4g) — variable-size real FFT, power-of-2 (used by NS, VAD)
//! - [`Pffft`](pffft::Pffft) — variable-size real/complex FFT, composite sizes (used by AGC2 RNN-VAD)

// SIMD modules require unsafe for intrinsics; safe wrappers are provided.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod fft4g;
pub mod ooura_fft;
#[cfg(target_arch = "aarch64")]
pub(crate) mod ooura_fft_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod ooura_fft_sse2;
pub mod pffft;
