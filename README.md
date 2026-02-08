# Sonora

A pure Rust implementation of WebRTC audio processing, providing echo cancellation, noise suppression, and automatic gain control.

Ported from the [WebRTC Native Code](https://webrtc.googlesource.com/src/) (M145) audio processing module.

## Features

- **Echo Cancellation (AEC3)** — adaptive filter-based echo canceller with delay estimation
- **Noise Suppression** — Wiener filter-based noise reduction with voice activity detection
- **Automatic Gain Control (AGC2)** — RNN VAD-based gain controller with limiter
- **High-Pass Filter** — DC offset removal
- **C API** — cbindgen-generated C header for FFI integration (feature-gated)

## Quick Start

```rust
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora::config::{EchoCanceller, NoiseSuppression, GainController2};

let config = Config {
    echo_canceller: EchoCanceller { enabled: true, ..Default::default() },
    noise_suppression: NoiseSuppression { enabled: true, ..Default::default() },
    gain_controller2: GainController2 { enabled: true, ..Default::default() },
    ..Default::default()
};

let mut apm = AudioProcessing::builder().config(config).build();
let stream = StreamConfig::new(48000, 1);

// Process 10ms frames (48kHz * 10ms = 480 samples)
let src = vec![0.0f32; 480];
let mut dest = vec![0.0f32; 480];
apm.process_stream_f32(&[&src], &stream, &stream, &mut [&mut dest]).unwrap();
```

## Crates

| Crate | Description |
|-------|-------------|
| [`sonora`](crates/sonora) | Main crate — full audio processing pipeline |
| [`sonora-aec3`](crates/sonora-aec3) | Echo canceller (AEC3) |
| [`sonora-agc2`](crates/sonora-agc2) | Automatic gain control with RNN VAD |
| [`sonora-ns`](crates/sonora-ns) | Noise suppression |
| [`sonora-common-audio`](crates/sonora-common-audio) | Audio utilities (resampler, filters, buffers) |
| [`sonora-simd`](crates/sonora-simd) | SIMD operations (SSE2, AVX2, NEON) |
| [`sonora-fft`](crates/sonora-fft) | FFT implementations (Ooura, PFFFT) |
| [`sonora-sys`](crates/sonora-sys) | C++ FFI bridge for comparison testing |

## Benchmarks

Full pipeline processing a 10 ms frame with AEC3 + noise suppression + AGC2 enabled.

Measured on Apple M4 Max (NEON backend), Rust 1.85, `-C target-cpu=native`:

### Rust vs C++ comparison

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| 16 kHz mono (all) | 4.2 us | 4.0 us | 1.07x |
| 16 kHz mono (NS only) | 3.6 us | 3.6 us | 1.01x |
| 16 kHz mono (EC only) | 1.4 us | 1.2 us | 1.15x |
| 16 kHz mono (AGC2 only) | 418 ns | 475 ns | 0.88x |
| 48 kHz mono (all) | 13.3 us | 10.8 us | 1.24x |

### Profiling breakdown (48 kHz mono, all components)

| Component | Rust self% | C++ self% | Notes |
|-----------|-----------|-----------|-------|
| Sinc resampler (NEON convolution) | 19.0% | 18.5% | At parity |
| High-pass filter (cascaded biquad) | 17.6% | 17.6% | At parity |
| Band split/merge (3-band + QMF) | 20.5% | 17.8% | Rust ~15% more |
| Noise suppression (analyze+process) | 9.5% | 5.4% | Inlining differences |
| FFT (Ooura fft4g) | 8.6% | 6.6% | Unchecked indexing approach applied |
| Pipeline orchestration | 3.8% | ~0% | Rust Option unwrap overhead |
| Sinc resampler scaffolding | 8.6% | 5.9% | Loop/callback overhead |

Key findings:
- SIMD (NEON) convolution is at parity with C++
- Noise suppression is at parity after FMA (`mul_add`) and unchecked FFT indexing
- The remaining gap is spread across many small overheads: Rust's stack probes for large stack frames, pipeline orchestration cost, and dynamic dispatch through trait objects
- No single "smoking gun" — optimization requires incremental improvements across many hot paths

### Component benchmarks (Rust)

| Benchmark | Time |
|-----------|------|
| `process_stream` 16 kHz mono | 4.3 us |
| `process_stream` 48 kHz mono | 13.3 us |
| `process_stream` 48 kHz stereo | 19.5 us |
| Noise suppressor (analyze + process) | 1.1 us |
| PFFFT forward 128-point | 239 ns |
| PFFFT forward 256-point | 487 ns |
| PFFFT forward 512-point | 1.2 us |

All times are per 10 ms frame. Reproduce with:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p sonora --bench pipeline
```

### Running the comparison

Requires building the C++ reference library first (needs meson, ninja, and abseil):

```bash
# Build and install the C++ library (release mode for fair comparison)
cd cpp
meson setup builddir --buildtype=release -Dtests=disabled -Dprefix=$(pwd)/install
ninja -C builddir install
cd ..

# Run the comparison benchmark
RUSTFLAGS="-C target-cpu=native" PKG_CONFIG_PATH=$(pwd)/cpp/install/lib/pkgconfig \
  cargo bench -p sonora --features cpp-comparison --bench cpp_comparison
```

## Supported Platforms

Sonora is tested on the following platforms in CI:

| Platform | Architecture | SIMD Backend | CI Status |
|----------|-------------|--------------|-----------|
| Linux (Ubuntu) | x86_64 | SSE2, AVX2 | Build, test, clippy, fmt, docs |
| macOS | ARM64 (Apple Silicon) | NEON | Build, test |
| Windows | x86_64 | SSE2, AVX2 | Build, test |
| Android | aarch64 | NEON | Cross-compile check |
| Android | x86_64 | SSE2, AVX2 | Cross-compile check |

### C++ Integration

The C++ reference test suite (WebRTC M145, 2400+ tests) is validated on Ubuntu x86_64 with the Rust backend linked via the `sonora-sys` FFI bridge. Tests requiring AGC1 (not ported) are disabled when the Rust backend is active.

### SIMD

Runtime feature detection is used for AVX2 on x86_64. SSE2 is assumed available on all x86_64 targets. NEON is used on AArch64. A scalar fallback is provided for all other architectures.

## MSRV

The minimum supported Rust version is **1.91**.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

## History

The audio processing code in this project has a long lineage:

1. **Google's libwebrtc** — The original AudioProcessing module was developed as part of the [WebRTC Native Code](https://webrtc.googlesource.com/src/) project at Google, providing production-grade echo cancellation, noise suppression, and gain control for real-time communication.

2. **PulseAudio's webrtc-audio-processing** — [Arun Raghavan](https://github.com/arunraghavan) and contributors extracted the AudioProcessing module into a [standalone library](https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing/) with a Meson build system, making it usable outside of the full WebRTC stack. This packaging is used by PulseAudio, PipeWire, and other Linux audio projects.

3. **M145 upgrade and test expansion** — With AI assistance (Claude, Anthropic), dignifiedquire updated the C++ codebase to WebRTC M145 (branch-heads/7632), ported the full upstream test suite (2400+ tests), and upgraded to C++20 with modern abseil-cpp.

4. **Sonora: AI-assisted Rust port** — The C++ implementation was ported to pure Rust with the assistance of Claude (Anthropic), producing this crate. The port includes hand-written SIMD (SSE2, AVX2, NEON), a pure-Rust FFT, and a C API for FFI integration. The Rust implementation is validated against the C++ reference via property-based testing.
