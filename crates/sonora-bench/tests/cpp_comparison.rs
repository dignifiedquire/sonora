//! Integration tests comparing Rust and C++ audio processing output.
//!
//! Per-component tests verify bit-level matching at each DSP stage.
//! Full-pipeline tests verify end-to-end equivalence.

use sonora::config::{EchoCanceller, GainController2, NoiseSuppression};
use sonora::internals::{self, FULL_BAND_SIZE, NUM_BANDS, SPLIT_BAND_SIZE};
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora_bench::comparison::compare_f32;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gen_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect()
}

// ── Per-component: ThreeBandFilterBank ────────────────────────────────────────

#[test]
fn filter_bank_analysis_matches_cpp() {
    let mut rust_bank = internals::ThreeBandFilterBank::new();
    let mut cpp_bank = sonora_sys::create_filter_bank();

    let input = gen_signal(FULL_BAND_SIZE);
    let input_arr: &[f32; FULL_BAND_SIZE] = input.as_slice().try_into().unwrap();

    // Process multiple frames to test state accumulation.
    for frame_idx in 0..10 {
        let mut rust_out = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        rust_bank.analysis(input_arr, &mut rust_out);

        let mut cpp_out = vec![0.0f32; FULL_BAND_SIZE]; // 3×160 packed
        sonora_sys::filter_bank_analysis(cpp_bank.pin_mut(), &input, &mut cpp_out);

        // Compare each band separately for clearer diagnostics.
        for band in 0..NUM_BANDS {
            let cpp_band = &cpp_out[band * SPLIT_BAND_SIZE..(band + 1) * SPLIT_BAND_SIZE];
            let result = compare_f32(&rust_out[band], cpp_band, 0.0);
            assert!(
                result.mismatches == 0,
                "filter_bank analysis band {band} frame {frame_idx}: {result}",
            );
        }
    }
}

#[test]
fn filter_bank_synthesis_matches_cpp() {
    let mut rust_bank = internals::ThreeBandFilterBank::new();
    let mut cpp_bank = sonora_sys::create_filter_bank();

    // Generate band data (3×160 samples).
    let band_data: Vec<f32> = (0..FULL_BAND_SIZE)
        .map(|i| (i as f32 * 0.007).sin() * 0.05)
        .collect();

    for frame_idx in 0..10 {
        // Rust: unpack into [[f32; 160]; 3]
        let mut rust_input = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        for band in 0..NUM_BANDS {
            rust_input[band]
                .copy_from_slice(&band_data[band * SPLIT_BAND_SIZE..(band + 1) * SPLIT_BAND_SIZE]);
        }
        let mut rust_out = [0.0f32; FULL_BAND_SIZE];
        rust_bank.synthesis(&rust_input, &mut rust_out);

        let mut cpp_out = vec![0.0f32; FULL_BAND_SIZE];
        sonora_sys::filter_bank_synthesis(cpp_bank.pin_mut(), &band_data, &mut cpp_out);

        let result = compare_f32(&rust_out, &cpp_out, 0.0);
        assert!(
            result.mismatches == 0,
            "filter_bank synthesis frame {frame_idx}: {result}",
        );
    }
}

#[test]
fn filter_bank_roundtrip_matches_cpp() {
    let mut rust_bank = internals::ThreeBandFilterBank::new();
    let mut cpp_bank = sonora_sys::create_filter_bank();

    let input = gen_signal(FULL_BAND_SIZE);
    let input_arr: &[f32; FULL_BAND_SIZE] = input.as_slice().try_into().unwrap();

    for frame_idx in 0..10 {
        // Analysis
        let mut rust_bands = [[0.0f32; SPLIT_BAND_SIZE]; NUM_BANDS];
        rust_bank.analysis(input_arr, &mut rust_bands);

        let mut cpp_bands = vec![0.0f32; FULL_BAND_SIZE];
        sonora_sys::filter_bank_analysis(cpp_bank.pin_mut(), &input, &mut cpp_bands);

        // Verify analysis matches
        for band in 0..NUM_BANDS {
            let cpp_band = &cpp_bands[band * SPLIT_BAND_SIZE..(band + 1) * SPLIT_BAND_SIZE];
            let result = compare_f32(&rust_bands[band], cpp_band, 0.0);
            assert!(
                result.mismatches == 0,
                "filter_bank roundtrip analysis band {band} frame {frame_idx}: {result}",
            );
        }

        // Synthesis (use Rust analysis output for both to isolate synthesis comparison)
        let rust_packed: Vec<f32> = rust_bands.iter().flatten().copied().collect();

        let mut rust_synth_bank = internals::ThreeBandFilterBank::new();
        let mut cpp_synth_bank = sonora_sys::create_filter_bank();

        let mut rust_out = [0.0f32; FULL_BAND_SIZE];
        rust_synth_bank.synthesis(&rust_bands, &mut rust_out);

        let mut cpp_out = vec![0.0f32; FULL_BAND_SIZE];
        sonora_sys::filter_bank_synthesis(cpp_synth_bank.pin_mut(), &rust_packed, &mut cpp_out);

        let result = compare_f32(&rust_out, &cpp_out, 0.0);
        assert!(
            result.mismatches == 0,
            "filter_bank roundtrip synthesis frame {frame_idx}: {result}",
        );
    }
}

// ── Per-component: HighPassFilter ────────────────────────────────────────────

#[test]
fn hpf_matches_cpp() {
    for &sample_rate in &[16000i32, 32000, 48000] {
        let frame_size = sample_rate as usize / 100;
        let mut rust_hpf = internals::HighPassFilter::new(sample_rate, 1);
        let mut cpp_hpf = sonora_sys::create_hpf(sample_rate, 1);

        let input = gen_signal(frame_size);

        for frame_idx in 0..20 {
            let mut rust_data = vec![input.clone()];
            rust_hpf.process_channels(&mut rust_data);

            let mut cpp_data = input.clone();
            sonora_sys::hpf_process(cpp_hpf.pin_mut(), &mut cpp_data);

            let result = compare_f32(&rust_data[0], &cpp_data, 0.0);
            assert!(
                result.mismatches == 0,
                "hpf {sample_rate}Hz frame {frame_idx}: {result}",
            );
        }
    }
}

// ── Full pipeline ────────────────────────────────────────────────────────────

struct ComponentConfig {
    name: &'static str,
    ec: bool,
    ns: bool,
    agc2: bool,
}

struct Format {
    name: &'static str,
    sample_rate: u32,
    channels: u16,
}

const CONFIGS: &[ComponentConfig] = &[
    ComponentConfig {
        name: "none",
        ec: false,
        ns: false,
        agc2: false,
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
        name: "all",
        ec: true,
        ns: true,
        agc2: true,
    },
];

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

const WARMUP_FRAMES: usize = 50;
const TEST_FRAMES: usize = 10;

fn make_rust_apm(cfg: &ComponentConfig) -> AudioProcessing {
    let config = Config {
        echo_canceller: if cfg.ec {
            Some(EchoCanceller::default())
        } else {
            None
        },
        noise_suppression: if cfg.ns {
            Some(NoiseSuppression::default())
        } else {
            None
        },
        gain_controller2: if cfg.agc2 {
            Some(GainController2::default())
        } else {
            None
        },
        high_pass_filter: None,
        ..Default::default()
    };
    AudioProcessing::builder().config(config).build()
}

#[test]
fn rust_cpp_pipeline_comparison() {
    let mut all_results: Vec<(String, sonora_bench::comparison::ComparisonResult)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for fmt in FORMATS {
        let stream = StreamConfig::new(fmt.sample_rate, fmt.channels);
        let frames_per_10ms = stream.num_frames();
        let sr = fmt.sample_rate as i32;
        let src_ch = gen_signal(frames_per_10ms);

        for cfg in CONFIGS {
            let label = format!("{}/{}", fmt.name, cfg.name);

            let mut rust_apm = make_rust_apm(cfg);
            let mut cpp_apm = sonora_sys::create_apm();
            sonora_sys::apply_config(cpp_apm.pin_mut(), cfg.ec, cfg.ns, 1, cfg.agc2, false);

            if fmt.channels == 1 {
                let mut rust_dst = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst = vec![0.0f32; frames_per_10ms];
                let mut worst = sonora_bench::comparison::ComparisonResult {
                    max_abs_diff: 0.0,
                    max_abs_diff_index: 0,
                    mean_abs_diff: 0.0,
                    mismatches: 0,
                    total: 0,
                };

                for frame_idx in 0..(WARMUP_FRAMES + TEST_FRAMES) {
                    rust_dst.fill(0.0);
                    cpp_dst.fill(0.0);

                    let src_slices = [src_ch.as_slice()];
                    let mut dst_slices = [rust_dst.as_mut_slice()];
                    let _ = rust_apm.process_capture_f32_with_config(
                        &src_slices,
                        &stream,
                        &stream,
                        &mut dst_slices,
                    );

                    sonora_sys::process_stream_f32(
                        cpp_apm.pin_mut(),
                        &src_ch,
                        sr,
                        1,
                        sr,
                        1,
                        &mut cpp_dst,
                    );

                    if frame_idx >= WARMUP_FRAMES {
                        let result = compare_f32(&rust_dst, &cpp_dst, 0.0);
                        if result.max_abs_diff > worst.max_abs_diff {
                            worst = result;
                        }
                    }
                }

                if worst.max_abs_diff > 0.0 {
                    failures.push(format!("{label}: {worst}"));
                }
                all_results.push((label, worst));
            } else {
                let src_r = gen_signal(frames_per_10ms);
                let mut rust_dst_l = vec![0.0f32; frames_per_10ms];
                let mut rust_dst_r = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst_l = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst_r = vec![0.0f32; frames_per_10ms];
                let mut worst_l = sonora_bench::comparison::ComparisonResult {
                    max_abs_diff: 0.0,
                    max_abs_diff_index: 0,
                    mean_abs_diff: 0.0,
                    mismatches: 0,
                    total: 0,
                };
                let mut worst_r = worst_l.clone();

                for frame_idx in 0..(WARMUP_FRAMES + TEST_FRAMES) {
                    rust_dst_l.fill(0.0);
                    rust_dst_r.fill(0.0);
                    cpp_dst_l.fill(0.0);
                    cpp_dst_r.fill(0.0);

                    let src_slices = [src_ch.as_slice(), src_r.as_slice()];
                    let mut dst_slices = [rust_dst_l.as_mut_slice(), rust_dst_r.as_mut_slice()];
                    let _ = rust_apm.process_capture_f32_with_config(
                        &src_slices,
                        &stream,
                        &stream,
                        &mut dst_slices,
                    );

                    sonora_sys::process_stream_f32_2ch(
                        cpp_apm.pin_mut(),
                        &src_ch,
                        &src_r,
                        sr,
                        &mut cpp_dst_l,
                        &mut cpp_dst_r,
                    );

                    if frame_idx >= WARMUP_FRAMES {
                        let rl = compare_f32(&rust_dst_l, &cpp_dst_l, 0.0);
                        let rr = compare_f32(&rust_dst_r, &cpp_dst_r, 0.0);
                        if rl.max_abs_diff > worst_l.max_abs_diff {
                            worst_l = rl;
                        }
                        if rr.max_abs_diff > worst_r.max_abs_diff {
                            worst_r = rr;
                        }
                    }
                }

                if worst_l.max_abs_diff > 0.0 {
                    failures.push(format!("{label}/L: {worst_l}"));
                }
                if worst_r.max_abs_diff > 0.0 {
                    failures.push(format!("{label}/R: {worst_r}"));
                }
                all_results.push((format!("{label}/L"), worst_l));
                all_results.push((format!("{label}/R"), worst_r));
            }
        }
    }

    // Print full report.
    eprintln!("\n=== Rust vs C++ divergence report ({WARMUP_FRAMES}+{TEST_FRAMES} frames) ===");
    for (label, result) in &all_results {
        if result.max_abs_diff > 0.0 {
            eprintln!("  DIFF  {label}: {result}");
        } else {
            eprintln!("  OK    {label}: bit-identical");
        }
    }
    eprintln!();

    if !failures.is_empty() {
        panic!(
            "Rust/C++ divergence in {} config(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
