# sonora-common-audio

[![crate][crate-image]][crate-link]
[![docs][docs-image]][docs-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Audio DSP primitives for the [Sonora] audio processing library.

Includes sinc resampler, push resampler, channel buffer, biquad filter,
ring buffer, and audio format conversion utilities. These building blocks
are shared across the echo canceller, noise suppressor, and gain controller.

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/sonora-common-audio.svg
[crate-link]: https://crates.io/crates/sonora-common-audio
[docs-image]: https://docs.rs/sonora-common-audio/badge.svg
[docs-link]: https://docs.rs/sonora-common-audio/
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
