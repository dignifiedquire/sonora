# sonora-simd

[![crate][crate-image]][crate-link]
[![docs][docs-image]][docs-link]
![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

SIMD abstraction layer for the [Sonora] audio processing library.

Provides accelerated vector operations for audio processing hot paths.
Supports SSE2 and AVX2 (x86_64), NEON (AArch64), with a scalar fallback
for all other architectures. AVX2 is detected at runtime via [cpufeatures].

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/sonora-simd.svg
[crate-link]: https://crates.io/crates/sonora-simd
[docs-image]: https://docs.rs/sonora-simd/badge.svg
[docs-link]: https://docs.rs/sonora-simd/
[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[Sonora]: https://github.com/dignifiedquire/sonora#readme
[cpufeatures]: https://docs.rs/cpufeatures
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
