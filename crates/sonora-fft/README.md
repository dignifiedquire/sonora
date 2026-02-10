# sonora-fft

[![crate][crate-image]][crate-link]
[![docs][docs-image]][docs-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Pure Rust FFT implementations for the [Sonora] audio processing library.

Includes Ooura 128-point and general-purpose (fft4g) FFTs, plus a Rust port
of [PFFFT] (Pretty Fast FFT) for composite-size real and complex transforms.
Optimized for the specific sizes used in WebRTC audio processing (128, 256, 512).

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/sonora-fft.svg
[crate-link]: https://crates.io/crates/sonora-fft
[docs-image]: https://docs.rs/sonora-fft/badge.svg
[docs-link]: https://docs.rs/sonora-fft/
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[PFFFT]: https://bitbucket.org/jpommier/pffft/
[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
