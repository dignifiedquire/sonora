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

## MSRV

The minimum supported Rust version is **1.91**.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

Based on code from the [WebRTC project](https://webrtc.org/), copyright The WebRTC Project Authors.
