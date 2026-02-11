#![doc = include_str!("../README.md")]
// SIMD modules require unsafe for intrinsics; safe wrappers are provided.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod fft4g;
pub mod ooura_fft;
#[cfg(target_arch = "aarch64")]
pub(crate) mod ooura_fft_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod ooura_fft_sse2;
pub mod pffft;
