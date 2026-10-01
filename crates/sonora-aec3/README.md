# sonora-aec3

[![crate][crate-image]][crate-link]
[![docs][docs-image]][docs-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Pure Rust implementation of [Echo Canceller 3 (AEC3)][AEC3] from WebRTC.

Adaptive filter-based acoustic echo canceller with automatic delay estimation,
render signal analysis, and echo path change detection. Operates in the
frequency domain using partitioned block processing.

Part of the [Sonora] audio processing library.

## Not ported: neural residual echo estimator

Upstream AEC3 can replace its heuristic residual echo estimate with a neural
network (`NeuralResidualEchoEstimator`, "ML-REE"). This crate does not port it:

- Upstream ships it disabled. An application must inject it through
  `BuiltinAudioProcessingBuilder::SetNeuralResidualEchoEstimator()`, the
  creator API was marked experimental at M145, and WebRTC's build keeps the
  default implementation out of ordinary targets.
- It runs inference with TensorFlow Lite, a C++ library, and needs a trained
  model. The WebRTC repository contains only a no-op model for testing.
- The C++ reference that sonora is validated against (PulseAudio's
  webrtc-audio-processing packaging) does not build it either.

Without it, AEC3 behaves like upstream AEC3 with no estimator injected, which
is upstream's default. Upstream switches the suppressor configuration when the
estimator turns on or off. Only the receiving side of that switch is ported:
the suppression gain can apply a new configuration, but the echo remover
always passes the fixed `suppressor` configuration and reports no change, so
the switch never happens.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/sonora-aec3.svg
[crate-link]: https://crates.io/crates/sonora-aec3
[docs-image]: https://docs.rs/sonora-aec3/badge.svg
[docs-link]: https://docs.rs/sonora-aec3/
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[AEC3]: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/
[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
