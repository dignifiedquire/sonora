# Benchmarks

Full pipeline processing a 10 ms frame with AEC3 + noise suppression + AGC2 enabled.

Measured on Apple M4 Max (NEON backend), Rust 1.85, `-C target-cpu=native`.

## Rust vs C++ Comparison

| Benchmark | Rust | C++ | Ratio |
|-----------|------|-----|-------|
| 16 kHz mono (all) | 4.2 us | 4.0 us | 1.07x |
| 16 kHz mono (NS only) | 3.6 us | 3.6 us | 1.01x |
| 16 kHz mono (EC only) | 1.4 us | 1.2 us | 1.15x |
| 16 kHz mono (AGC2 only) | 418 ns | 475 ns | 0.88x |
| 48 kHz mono (all) | 13.3 us | 10.8 us | 1.24x |

## Profiling Breakdown (48 kHz mono, all components)

| Component | Rust self% | C++ self% | Notes |
|-----------|-----------|-----------|-------|
| Sinc resampler (NEON convolution) | 19.0% | 18.5% | At parity |
| High-pass filter (cascaded biquad) | 17.6% | 17.6% | At parity |
| Band split/merge (3-band + QMF) | 20.5% | 17.8% | Rust ~15% more |
| Noise suppression (analyze+process) | 9.5% | 5.4% | Inlining differences |
| FFT (Ooura fft4g) | 8.6% | 6.6% | Unchecked indexing approach applied |
| Pipeline orchestration | 3.8% | ~0% | Rust Option unwrap overhead |
| Sinc resampler scaffolding | 8.6% | 5.9% | Loop/callback overhead |

## Component Benchmarks (Rust)

| Benchmark | Time |
|-----------|------|
| `process_stream` 16 kHz mono | 4.3 us |
| `process_stream` 48 kHz mono | 13.3 us |
| `process_stream` 48 kHz stereo | 19.5 us |
| Noise suppressor (analyze + process) | 1.1 us |
| PFFFT forward 128-point | 239 ns |
| PFFFT forward 256-point | 487 ns |
| PFFFT forward 512-point | 1.2 us |

All times are per 10 ms frame.

## Running Benchmarks

### Using cargo-make

```bash
cargo install cargo-make

cargo make bench          # Rust pipeline benchmarks
cargo make setup          # Install system deps + build C++ library
cargo make cpp-bench      # Rust vs C++ comparison benchmarks
```

### Manual commands

#### Rust pipeline benchmarks

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p sonora --bench pipeline
```

#### C++ comparison

Requires building the C++ reference library first (needs meson, ninja, and abseil):

```bash
# Build and install the C++ library (release mode for fair comparison)
cd cpp
meson setup builddir --buildtype=release -Dtests=disabled -Dprefix=$(pwd)/install
ninja -C builddir install
cd ..

# Run the comparison benchmark
RUSTFLAGS="-C target-cpu=native" PKG_CONFIG_PATH=$(pwd)/cpp/install/lib/pkgconfig \
  cargo bench --manifest-path crates/sonora-bench/Cargo.toml --bench cpp_comparison
```
