# Sonora

[![CI][ci-image]][ci-link]
[![C++ Validation][cpp-image]][cpp-link]
[![Coverage][codecov-image]][codecov-link]
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

Run the minimal echo cancellation demo:

```bash
cargo run -p sonora --example simple
```

More examples in [`crates/sonora/examples/`](crates/sonora/examples/):

| Example | Description | Command |
|---------|-------------|---------|
| [`simple`](crates/sonora/examples/simple.rs) | Synthetic AEC round-trip | `cargo run -p sonora --example simple` |
| [`karaoke`](crates/sonora/examples/karaoke.rs) | Mic loopback with echo cancellation | `cargo run -p sonora --features examples --example karaoke` |
| [`recording`](crates/sonora/examples/recording.rs) | Record & process to WAV | `cargo run -p sonora --features examples --example recording -- --duration 5 --ns --agc` |

The `karaoke` and `recording` examples require the `examples` feature which pulls in [cpal], [hound], and other audio I/O dependencies. These examples are based on the [tonarino/webrtc-audio-processing examples][tonarino-examples], ported from PortAudio to cpal.

[cpal]: https://crates.io/crates/cpal
[hound]: https://crates.io/crates/hound
[tonarino-examples]: https://github.com/tonarino/webrtc-audio-processing/tree/main/examples

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

## Related Projects

- **[tonarino/webrtc-audio-processing]** -- Rust bindings to the C++ WebRTC AudioProcessing module. Sonora's examples are based on theirs. If you need the battle-tested C++ implementation with a Rust wrapper, use tonarino; if you want a pure-Rust solution with no C++ dependency, use sonora.
- **[pulseaudio-wap]** -- The original extracted C++ code

## History

1. **Google's libwebrtc** -- The original AudioProcessing module is developed as part of the [WebRTC Native Code][webrtc-src] project at Google.

2. **PulseAudio's webrtc-audio-processing** -- [Arun Raghavan](https://github.com/arunraghavan) and contributors extracted the AudioProcessing module into a [standalone library][pulseaudio-wap] with a Meson build system, making it usable outside of the full WebRTC stack. This packaging is used by PulseAudio, PipeWire, and other Linux audio projects.

3. **M145 upgrade and test expansion** -- The C++ codebase was updated to WebRTC M145 (branch-heads/7632), the full upstream test suite (2400+ tests) was ported, and the build was upgraded to C++20. (AI assisted)

4. **Sonora:Rust port** -- The C++ implementation was ported to pure Rust, producing this set of crates. The port includes the full SIMD (SSE2, AVX2, NEON) optimizations and the FFTs, as well as a C API for FFI integration. The full C++ test suite passes against this Rust version. (AI assisted)

## License

All crates in this repository are licensed under [BSD-3-Clause](LICENSE).


[//]: # (badges)

[ci-image]: https://github.com/dignifiedquire/sonora/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/dignifiedquire/sonora/actions/workflows/ci.yml?query=branch:main
[cpp-image]: https://github.com/dignifiedquire/sonora/actions/workflows/cpp-validation.yml/badge.svg?branch=main
[cpp-link]: https://github.com/dignifiedquire/sonora/actions/workflows/cpp-validation.yml?query=branch:main
[codecov-image]: https://codecov.io/gh/dignifiedquire/sonora/branch/main/graph/badge.svg
[codecov-link]: https://app.codecov.io/gh/dignifiedquire/sonora/tree/main/crates
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
[tonarino/webrtc-audio-processing]: https://github.com/tonarino/webrtc-audio-processing
