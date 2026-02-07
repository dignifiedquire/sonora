# Changelog

## 0.1.0 (unreleased)

Initial release. Pure Rust port of WebRTC Audio Processing (M145).

- Echo cancellation (AEC3) with SIMD acceleration (SSE2, AVX2, NEON)
- Noise suppression with multi-band Wiener filtering
- Automatic gain control (AGC2) with RNN-based voice activity detection
- High-pass filter for DC offset removal
- Resampling and channel conversion
- C API via cbindgen for FFI integration
- Float (deinterleaved) and i16 (interleaved) processing
