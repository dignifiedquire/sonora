//! Rust vs C++ comparison benchmarks.
//!
//! Requires the `cpp-comparison` feature and a pre-built C++ library:
//! ```bash
//! # Build C++ library first (in the cpp/ submodule)
//! cd cpp && meson setup builddir -Dprefix=$(pwd)/install && ninja -C builddir install
//! # Then run comparison benchmarks
//! PKG_CONFIG_PATH=cpp/install/lib/pkgconfig cargo bench -p sonora --features cpp-comparison --bench cpp_comparison
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sonora::config::{EchoCanceller, GainController2, NoiseSuppression};
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora_sys;

fn bench_process_stream_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("process_stream_comparison");

    let sample_rate = 48000i32;
    let channels = 1usize;
    let frames = (sample_rate as usize) / 100; // 10ms

    // Input data
    let src: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
    let mut dest = vec![0.0f32; frames];

    // --- Rust backend ---
    {
        let config = Config {
            echo_canceller: EchoCanceller {
                enabled: true,
                ..Default::default()
            },
            noise_suppression: NoiseSuppression {
                enabled: true,
                ..Default::default()
            },
            gain_controller2: GainController2 {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let stream = StreamConfig::new(sample_rate as usize, channels);
        let mut apm = AudioProcessing::builder().config(config).build();

        // Warm up
        for _ in 0..50 {
            let src_slice = [src.as_slice()];
            let mut dst_slice = [dest.as_mut_slice()];
            let _ = apm.process_stream_f32(&src_slice, &stream, &stream, &mut dst_slice);
        }

        group.bench_function("rust_48k_mono", |b| {
            b.iter(|| {
                let src_slice = [src.as_slice()];
                let mut dst_slice = [dest.as_mut_slice()];
                apm.process_stream_f32(black_box(&src_slice), &stream, &stream, &mut dst_slice)
                    .unwrap();
            });
        });
    }

    // --- C++ backend ---
    {
        let mut cpp_apm = sonora_sys::create_apm();

        // Apply matching config: EC + NS + AGC2
        sonora_sys::apply_config(
            cpp_apm.pin_mut(),
            true, // ec
            true, // ns
            1,    // ns_level (moderate)
            true, // agc2
        );

        // Warm up
        for _ in 0..50 {
            sonora_sys::process_stream_f32(
                cpp_apm.pin_mut(),
                &src,
                sample_rate,
                channels,
                sample_rate,
                channels,
                &mut dest,
            );
        }

        group.bench_function("cpp_48k_mono", |b| {
            b.iter(|| {
                sonora_sys::process_stream_f32(
                    cpp_apm.pin_mut(),
                    black_box(&src),
                    sample_rate,
                    channels,
                    sample_rate,
                    channels,
                    &mut dest,
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_process_stream_comparison);
criterion_main!(benches);
