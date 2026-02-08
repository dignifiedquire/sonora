# sonora-sys

![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

C++ FFI bindings to the WebRTC audio processing library for comparison testing.

Uses [cxx] to bridge between Rust and the C++ reference implementation
(built with Meson). This crate is used internally for property-based testing
and benchmarking against the upstream C++ code. Not published to crates.io.

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[cxx]: https://cxx.rs
[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
