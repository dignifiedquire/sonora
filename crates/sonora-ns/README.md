# sonora-ns

[![crate][crate-image]][crate-link]
[![docs][docs-image]][docs-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Pure Rust implementation of [Noise Suppression][NS] from WebRTC.

Wiener filter-based noise reduction with multi-band processing, prior and
posterior SNR estimation, and voice activity detection. Supports four
suppression levels (low, moderate, high, very high).

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/sonora-ns.svg
[crate-link]: https://crates.io/crates/sonora-ns
[docs-image]: https://docs.rs/sonora-ns/badge.svg
[docs-link]: https://docs.rs/sonora-ns/
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[NS]: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/ns/
[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
