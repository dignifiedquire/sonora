//! Rust vs C++ comparison benchmarks.
//!
//! Tests multiple sample rates, channel counts, and component configurations.
//!
//! Requires the `cpp-comparison` feature and a pre-built C++ library:
//! ```bash
//! cd cpp && meson setup builddir --buildtype=release -Dtests=disabled -Dprefix=$(pwd)/install
//! ninja -C builddir install && cd ..
//! RUSTFLAGS="-C target-cpu=native" PKG_CONFIG_PATH=$(pwd)/cpp/install/lib/pkgconfig \
//!   cargo bench -p sonora --features cpp-comparison --bench cpp_comparison
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sonora::config::{EchoCanceller, GainController2, HighPassFilter, NoiseSuppression};
use sonora::{AudioProcessing, Config, StreamConfig};
use std::slice;

// ── Helpers ──────────────────────────────────────────────────────────────────

struct ComponentConfig {
    name: &'static str,
    ec: bool,
    ns: bool,
    agc2: bool,
}

const CONFIGS: &[ComponentConfig] = &[
    ComponentConfig {
        name: "all",
        ec: true,
        ns: true,
        agc2: true,
    },
    ComponentConfig {
        name: "ec_only",
        ec: true,
        ns: false,
        agc2: false,
    },
    ComponentConfig {
        name: "ns_only",
        ec: false,
        ns: true,
        agc2: false,
    },
    ComponentConfig {
        name: "agc2_only",
        ec: false,
        ns: false,
        agc2: true,
    },
    ComponentConfig {
        name: "none",
        ec: false,
        ns: false,
        agc2: false,
    },
];

struct Format {
    name: &'static str,
    sample_rate: usize,
    channels: usize,
}

const FORMATS: &[Format] = &[
    Format {
        name: "16k_mono",
        sample_rate: 16000,
        channels: 1,
    },
    Format {
        name: "48k_mono",
        sample_rate: 48000,
        channels: 1,
    },
    Format {
        name: "48k_stereo",
        sample_rate: 48000,
        channels: 2,
    },
];

fn gen_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect()
}

fn make_rust_apm(cfg: &ComponentConfig, sample_rate: usize, channels: usize) -> AudioProcessing {
    let config = Config {
        echo_canceller: EchoCanceller {
            enabled: cfg.ec,
            ..Default::default()
        },
        noise_suppression: NoiseSuppression {
            enabled: cfg.ns,
            ..Default::default()
        },
        gain_controller2: GainController2 {
            enabled: cfg.agc2,
            ..Default::default()
        },
        high_pass_filter: HighPassFilter {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut apm = AudioProcessing::builder().config(config).build();
    let stream = StreamConfig::new(sample_rate, channels);
    let frames = sample_rate / 100;

    // Warm up
    let src_ch = gen_signal(frames);
    let src: Vec<&[f32]> = (0..channels).map(|_| src_ch.as_slice()).collect();
    let mut dst_ch = vec![0.0f32; frames];
    let mut dst: Vec<&mut [f32]> = (0..channels)
        .map(|_| unsafe { slice::from_raw_parts_mut(dst_ch.as_mut_ptr(), frames) })
        .collect();
    for _ in 0..50 {
        let _ = apm.process_stream_f32(&src, &stream, &stream, &mut dst);
    }
    apm
}

// ── Benchmark ────────────────────────────────────────────────────────────────

fn bench_comparison(c: &mut Criterion) {
    for fmt in FORMATS {
        let frames = fmt.sample_rate / 100;
        let stream = StreamConfig::new(fmt.sample_rate, fmt.channels);
        let src_ch = gen_signal(frames);

        for cfg in CONFIGS {
            let group_name = format!("{}/{}", fmt.name, cfg.name);
            let mut group = c.benchmark_group(&group_name);

            // ── Rust ──
            {
                let mut apm = make_rust_apm(cfg, fmt.sample_rate, fmt.channels);
                let src: Vec<&[f32]> = (0..fmt.channels).map(|_| src_ch.as_slice()).collect();

                if fmt.channels == 1 {
                    let mut dst = vec![0.0f32; frames];
                    group.bench_function("rust", |b| {
                        b.iter(|| {
                            let mut dst_slices = [dst.as_mut_slice()];
                            apm.process_stream_f32(
                                black_box(&src),
                                &stream,
                                &stream,
                                &mut dst_slices,
                            )
                            .unwrap();
                        });
                    });
                } else {
                    let mut dst_l = vec![0.0f32; frames];
                    let mut dst_r = vec![0.0f32; frames];
                    group.bench_function("rust", |b| {
                        b.iter(|| {
                            let mut dst_slices = [dst_l.as_mut_slice(), dst_r.as_mut_slice()];
                            apm.process_stream_f32(
                                black_box(&src),
                                &stream,
                                &stream,
                                &mut dst_slices,
                            )
                            .unwrap();
                        });
                    });
                }
            }

            // ── C++ ──
            {
                let mut cpp_apm = sonora_sys::create_apm();
                sonora_sys::apply_config(cpp_apm.pin_mut(), cfg.ec, cfg.ns, 1, cfg.agc2, false);
                let sr = fmt.sample_rate as i32;

                if fmt.channels == 1 {
                    let mut dst = vec![0.0f32; frames];
                    // Warm up
                    for _ in 0..50 {
                        sonora_sys::process_stream_f32(
                            cpp_apm.pin_mut(),
                            &src_ch,
                            sr,
                            1,
                            sr,
                            1,
                            &mut dst,
                        );
                    }
                    group.bench_function("cpp", |b| {
                        b.iter(|| {
                            sonora_sys::process_stream_f32(
                                cpp_apm.pin_mut(),
                                black_box(&src_ch),
                                sr,
                                1,
                                sr,
                                1,
                                &mut dst,
                            );
                        });
                    });
                } else {
                    let src_r = gen_signal(frames);
                    let mut dst_l = vec![0.0f32; frames];
                    let mut dst_r = vec![0.0f32; frames];
                    // Warm up
                    for _ in 0..50 {
                        sonora_sys::process_stream_f32_2ch(
                            cpp_apm.pin_mut(),
                            &src_ch,
                            &src_r,
                            sr,
                            &mut dst_l,
                            &mut dst_r,
                        );
                    }
                    group.bench_function("cpp", |b| {
                        b.iter(|| {
                            sonora_sys::process_stream_f32_2ch(
                                cpp_apm.pin_mut(),
                                black_box(&src_ch),
                                black_box(&src_r),
                                sr,
                                &mut dst_l,
                                &mut dst_r,
                            );
                        });
                    });
                }
            }

            group.finish();
        }
    }
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
