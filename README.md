# Sonora

[![CI][ci-image]][ci-link]
[![C++ Validation][cpp-image]][cpp-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Pure Rust implementation of [WebRTC] audio processing, providing echo cancellation,
noise suppression, and automatic gain control.

Ported from the [WebRTC Native Code][webrtc-src] (M145) audio processing module.

## Crates

| Crate | Description | Crates.io | Documentation |
|-------|-------------|:---------:|:-------------:|
| [`sonora`] | Full audio processing pipeline | [![crates.io](https://img.shields.io/crates/v/sonora.svg)](https://crates.io/crates/sonora) | [![docs.rs](https://docs.rs/sonora/badge.svg)](https://docs.rs/sonora) |
| [`sonora-ffi`] | C API (FFI) for integration with C/C++ projects | [![crates.io](https://img.shields.io/crates/v/sonora-ffi.svg)](https://crates.io/crates/sonora-ffi) | [![docs.rs](https://docs.rs/sonora-ffi/badge.svg)](https://docs.rs/sonora-ffi) |
| [`sonora-aec3`] | Echo Canceller 3 (AEC3) | [![crates.io](https://img.shields.io/crates/v/sonora-aec3.svg)](https://crates.io/crates/sonora-aec3) | [![docs.rs](https://docs.rs/sonora-aec3/badge.svg)](https://docs.rs/sonora-aec3) |
| [`sonora-agc2`] | Automatic Gain Control with RNN VAD | [![crates.io](https://img.shields.io/crates/v/sonora-agc2.svg)](https://crates.io/crates/sonora-agc2) | [![docs.rs](https://docs.rs/sonora-agc2/badge.svg)](https://docs.rs/sonora-agc2) |
| [`sonora-ns`] | Noise Suppression | [![crates.io](https://img.shields.io/crates/v/sonora-ns.svg)](https://crates.io/crates/sonora-ns) | [![docs.rs](https://docs.rs/sonora-ns/badge.svg)](https://docs.rs/sonora-ns) |
| [`sonora-common-audio`] | Audio DSP primitives (resamplers, filters) | [![crates.io](https://img.shields.io/crates/v/sonora-common-audio.svg)](https://crates.io/crates/sonora-common-audio) | [![docs.rs](https://docs.rs/sonora-common-audio/badge.svg)](https://docs.rs/sonora-common-audio) |
| [`sonora-simd`] | SIMD operations (SSE2, AVX2, NEON) | [![crates.io](https://img.shields.io/crates/v/sonora-simd.svg)](https://crates.io/crates/sonora-simd) | [![docs.rs](https://docs.rs/sonora-simd/badge.svg)](https://docs.rs/sonora-simd) |
| [`sonora-fft`] | FFT implementations (Ooura, PFFFT) | [![crates.io](https://img.shields.io/crates/v/sonora-fft.svg)](https://crates.io/crates/sonora-fft) | [![docs.rs](https://docs.rs/sonora-fft/badge.svg)](https://docs.rs/sonora-fft) |

## Features

- **Echo Cancellation (AEC3)** -- adaptive filter-based echo canceller with delay estimation
- **Noise Suppression** -- Wiener filter-based noise reduction with voice activity detection
- **Automatic Gain Control (AGC2)** -- RNN VAD-based gain controller with limiter
- **High-Pass Filter** -- DC offset removal
- **C API** -- cbindgen-generated C header for FFI integration (via [`sonora-ffi`])

## Quick Start

```rust
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora::config::{EchoCanceller, NoiseSuppression, GainController2};

let config = Config {
    echo_canceller: Some(EchoCanceller::default()),
    noise_suppression: Some(NoiseSuppression::default()),
    gain_controller2: Some(GainController2::default()),
    ..Default::default()
};

let mut apm = AudioProcessing::builder()
    .config(config)
    .capture_config(StreamConfig::new(48000, 1))
    .render_config(StreamConfig::new(48000, 1))
    .build();

// Process 10ms frames (48kHz * 10ms = 480 samples)
let src = vec![0.0f32; 480];
let mut dest = vec![0.0f32; 480];
apm.process_capture_f32(&[&src], &mut [&mut dest]).unwrap();
```

## Supported Platforms

| Platform | Architecture | SIMD Backend | CI Status |
|----------|-------------|--------------|-----------|
| Linux (Ubuntu) | x86_64 | SSE2, AVX2 | Build, test, clippy, fmt, docs |
| macOS | ARM64 (Apple Silicon) | NEON | Build, test |
| Windows | x86_64 | SSE2, AVX2 | Build, test |
| Android | aarch64 | NEON | Cross-compile check |
| Android | x86_64 | SSE2, AVX2 | Cross-compile check |

Runtime feature detection is used for AVX2 on x86_64. SSE2 is assumed available on all x86_64 targets. NEON is used on AArch64. A scalar fallback is provided for all other architectures.

### C++ Integration

The C++ reference test suite (WebRTC M145, 2400+ tests) is validated on Ubuntu x86_64 with the Rust backend linked via the `sonora-sys` FFI bridge.

## Benchmarks

Full pipeline processing a 10 ms frame with AEC3 + noise suppression + AGC2 enabled.
Measured on Apple M4 Max (NEON backend), Rust 1.85, `-C target-cpu=native`:

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| 16 kHz mono | 4.2 us | 4.0 us | 1.07x |
| 48 kHz mono | 13.3 us | 10.8 us | 1.24x |

See [BENCHMARKS.md](BENCHMARKS.md) for per-component comparisons, profiling breakdown, and instructions for reproducing.

## Development

This project uses [cargo-make](https://github.com/sagiegurari/cargo-make) for task automation.

```bash
cargo install cargo-make
```

### Quick start

    cargo make ci              # Format, lint, test, docs
    cargo make bench           # Rust pipeline benchmarks
    cargo make check           # Type-check all crates (including excluded)
    cargo make clippy          # Lint all crates (including excluded)

### C++ comparison (optional)

    cargo make setup           # Install system deps + build C++ library
    cargo make cpp-bench       # Run Rust vs C++ comparison benchmarks
    cargo make cpp-test        # Run comparison tests
    cargo make cpp-validate    # Run 2400+ C++ test suite with Rust backend

See `Makefile.toml` for the full list of tasks.

## Minimum Supported Rust Version

The minimum supported Rust version is **1.91**.

## License

All crates in this repository are licensed under [BSD-3-Clause](LICENSE).

## History

The audio processing code in this project has a long lineage:

1. **Google's libwebrtc** -- The original AudioProcessing module was developed as part of the [WebRTC Native Code][webrtc-src] project at Google, providing production-grade echo cancellation, noise suppression, and gain control for real-time communication.

2. **PulseAudio's webrtc-audio-processing** -- [Arun Raghavan](https://github.com/arunraghavan) and contributors extracted the AudioProcessing module into a [standalone library][pulseaudio-wap] with a Meson build system, making it usable outside of the full WebRTC stack. This packaging is used by PulseAudio, PipeWire, and other Linux audio projects.

3. **M145 upgrade and test expansion** -- With AI assistance (Claude, Anthropic), the C++ codebase was updated to WebRTC M145 (branch-heads/7632), the full upstream test suite (2400+ tests) was ported, and the build was upgraded to C++20 with modern abseil-cpp.

4. **Sonora: AI-assisted Rust port** -- The C++ implementation was ported to pure Rust with the assistance of Claude (Anthropic), producing this crate. The port includes hand-written SIMD (SSE2, AVX2, NEON), a pure-Rust FFT, and a C API for FFI integration. The Rust implementation is validated against the C++ reference via property-based testing.

[//]: # (badges)

[ci-image]: https://github.com/dignifiedquire/sonora/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/dignifiedquire/sonora/actions/workflows/ci.yml?query=branch:main
[cpp-image]: https://github.com/dignifiedquire/sonora/actions/workflows/cpp-validation.yml/badge.svg?branch=main
[cpp-link]: https://github.com/dignifiedquire/sonora/actions/workflows/cpp-validation.yml?query=branch:main
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (crates)

[`sonora`]: ./crates/sonora
[`sonora-ffi`]: ./crates/sonora-ffi
[`sonora-aec3`]: ./crates/sonora-aec3
[`sonora-agc2`]: ./crates/sonora-agc2
[`sonora-ns`]: ./crates/sonora-ns
[`sonora-common-audio`]: ./crates/sonora-common-audio
[`sonora-simd`]: ./crates/sonora-simd
[`sonora-fft`]: ./crates/sonora-fft

[//]: # (general links)

[WebRTC]: https://webrtc.org
[webrtc-src]: https://webrtc.googlesource.com/src/
[pulseaudio-wap]: https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing/
