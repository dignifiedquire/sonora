# Changelog

## Unreleased

The port now tracks M145 plus 15 later upstream WebRTC changes, up to M156. `cpp/NEWS` lists them. Three of them (multi-channel defaults, shared comfort noise, joint coarse/refined filter choice) were already in 0.2.0.

- Input volume controller: the default config is now upstream's production config (upstream d9b92fe1bc). With `input_volume_controller` enabled, `recommended_stream_analog_level()` targets a speech level in [-50, -12] dBFS instead of [-30, -18] dBFS, updates the volume from the speech level at most once every 100 frames, and also lowers it when clipping is predicted. The controller does not modify samples itself, but its recommendation reaches the output whenever it is applied: through `capture_level_adjustment.analog_mic_gain_emulation`, or when the caller passes it back with `set_stream_analog_level()`, where each volume change resets AGC2's adaptive digital level estimate and, with the echo canceller enabled, signals an echo path gain change to AEC3. So the output changes for callers that apply the recommendation.
- AEC3: ports the render buffer headroom fix for underruns (d460e60e19). Two further changes are internal and change no behaviour: the code that applies suppressor config updates (e10cd19640) is ported, but nothing in the port triggers it, and the code for three field-trial kill switches that the port never enabled is removed.
- Floating-point environment: like upstream, each `process_*` call now sets hardware flush-to-zero for its duration (MXCSR FTZ and DAZ on x86/x86_64 with SSE, FPCR.FZ on aarch64 with NEON). This targets the AEC3 slowdown on x86 while the render reference is silent (#34). Code that a call reaches, such as `tracing` subscribers and the global allocator, runs in that mode, and the output can differ by tiny amounts on near-silent input. Upstream sets the x86 bits only in clang builds, so on x86 a caller that replaces a GCC-built C++ library gets FTZ/DAZ where it had IEEE gradual underflow. Changing the floating-point environment is formally undefined behavior in Rust, and upstream's `DenormalDisabler`, nih-plug and the `no_denormals` crate also change it; sonora sets flush-to-zero only on the calling thread and restores that thread's setting after each call, but a thread started during a call can inherit flush-to-zero and keep it (on Linux, for example). Other targets do not flush and log "Denormal disabler unsupported" when the processor is built. The "Floating-point environment" section on `AudioProcessing` has the details.

## 0.1.0 (unreleased)

Initial release. Pure Rust port of WebRTC Audio Processing (M145).

- Echo cancellation (AEC3) with SIMD acceleration (SSE2, AVX2, NEON)
- Noise suppression with multi-band Wiener filtering
- Automatic gain control (AGC2) with RNN-based voice activity detection
- High-pass filter for DC offset removal
- Resampling and channel conversion
- C API via cbindgen for FFI integration
- Float (deinterleaved) and i16 (interleaved) processing
