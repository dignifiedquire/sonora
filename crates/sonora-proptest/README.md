# sonora-proptest

![BSD-3-Clause licensed][license-image]
![Rust Version][rustc-image]

Property-based testing utilities for the [Sonora] audio processing library.

Provides [proptest] strategies for generating audio frames, configurations,
and test fixtures. Used internally for cross-validating the Rust implementation
against the C++ reference. Not published to crates.io.

Part of the [Sonora] audio processing library.

## License

BSD-3-Clause. See [LICENSE] for details.

[//]: # (badges)

[license-image]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.91+-blue.svg

[//]: # (general links)

[proptest]: https://docs.rs/proptest
[Sonora]: https://github.com/dignifiedquire/sonora#readme
[LICENSE]: https://github.com/dignifiedquire/sonora/blob/main/LICENSE
