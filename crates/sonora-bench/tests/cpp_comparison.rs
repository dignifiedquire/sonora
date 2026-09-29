//! Integration tests comparing Rust and C++ audio processing output.
//!
//! Per-component tests verify close matching at each DSP stage.
//! Full-pipeline tests verify end-to-end equivalence.
//!
//! On ARM (NEON) both Rust and C++ use fused multiply-add producing bit-identical
//! results. On x86 the Rust `mul_add` intrinsic and C++ scalar arithmetic may
//! diverge by a small amount due to different FMA contraction behaviour between
//! LLVM and GCC. The tolerances below accommodate this.

use sonora::config::{EchoCanceller, GainController2, NoiseSuppression, TransparentModeType};
use sonora::high_pass_filter::HighPassFilter;
use sonora::three_band_filter_bank::{
    FULL_BAND_SIZE, NUM_BANDS, SPLIT_BAND_SIZE, ThreeBandFilterBank,
};
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora_bench::comparison::compare_f32;

// ── Tolerances ───────────────────────────────────────────────────────────────

/// Per-component tolerance (filter bank, HPF). FIR stages diverge by ~1 ULP
/// (~1.5e-8), but IIR filters (biquad cascade in the HPF) accumulate small
/// differences across frames, reaching ~1e-5 after 20 frames at 48 kHz.
const COMPONENT_TOL: f32 = 5e-5;

/// Full-pipeline tolerance. Small per-sample differences compound through
/// multi-band splitting, EC, and NS at 48 kHz.
const PIPELINE_TOL: f32 = 0.2;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gen_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect()
}

// ── Per-component: ThreeBandFilterBank ────────────────────────────────────────

#[test]
fn filter_bank_analysis_matches_cpp() {
    let mut rust_bank = ThreeBandFilterBank::new();
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
            let result = compare_f32(&rust_out[band], cpp_band, COMPONENT_TOL);
            assert!(
                result.mismatches == 0,
                "filter_bank analysis band {band} frame {frame_idx}: {result}",
            );
        }
    }
}

#[test]
fn filter_bank_synthesis_matches_cpp() {
    let mut rust_bank = ThreeBandFilterBank::new();
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

        let result = compare_f32(&rust_out, &cpp_out, COMPONENT_TOL);
        assert!(
            result.mismatches == 0,
            "filter_bank synthesis frame {frame_idx}: {result}",
        );
    }
}

#[test]
fn filter_bank_roundtrip_matches_cpp() {
    let mut rust_bank = ThreeBandFilterBank::new();
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
            let result = compare_f32(&rust_bands[band], cpp_band, COMPONENT_TOL);
            assert!(
                result.mismatches == 0,
                "filter_bank roundtrip analysis band {band} frame {frame_idx}: {result}",
            );
        }

        // Synthesis (use Rust analysis output for both to isolate synthesis comparison)
        let rust_packed: Vec<f32> = rust_bands.iter().flatten().copied().collect();

        let mut rust_synth_bank = ThreeBandFilterBank::new();
        let mut cpp_synth_bank = sonora_sys::create_filter_bank();

        let mut rust_out = [0.0f32; FULL_BAND_SIZE];
        rust_synth_bank.synthesis(&rust_bands, &mut rust_out);

        let mut cpp_out = vec![0.0f32; FULL_BAND_SIZE];
        sonora_sys::filter_bank_synthesis(cpp_synth_bank.pin_mut(), &rust_packed, &mut cpp_out);

        let result = compare_f32(&rust_out, &cpp_out, COMPONENT_TOL);
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
        let mut rust_hpf = HighPassFilter::new(sample_rate, 1);
        let mut cpp_hpf = sonora_sys::create_hpf(sample_rate, 1);

        let input = gen_signal(frame_size);

        for frame_idx in 0..20 {
            let mut rust_data = vec![input.clone()];
            rust_hpf.process_channels(&mut rust_data);

            let mut cpp_data = input.clone();
            sonora_sys::hpf_process(cpp_hpf.pin_mut(), &mut cpp_data);

            let result = compare_f32(&rust_data[0], &cpp_data, COMPONENT_TOL);
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

                if worst.max_abs_diff > PIPELINE_TOL {
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

                if worst_l.max_abs_diff > PIPELINE_TOL {
                    failures.push(format!("{label}/L: {worst_l}"));
                }
                if worst_r.max_abs_diff > PIPELINE_TOL {
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

// ── Full pipeline: stereo echo with decorrelated channels ────────────────────

/// Frames for `stereo_echo_pipeline_matches_cpp`. AEC3 treats the render
/// signal as mono until it has seen 2 s of stereo content
/// (`stereo_detection_hysteresis_seconds`), so the run must be well past
/// 200 frames to reach the multi-channel render path.
const STEREO_ECHO_FRAMES: usize = 500;

/// Echo path delay in samples (10 ms at 48 kHz).
const ECHO_DELAY: usize = 480;

/// Tolerance for `stereo_echo_pipeline_matches_cpp`, set from measurements
/// instead of `PIPELINE_TOL`. The Rust/C++ max diff is 0.0050 (L) and 0.0056
/// (R) on macOS arm64 and on x86_64 Linux (GCC, `-march=native`). With Rust
/// render downmixed to mono (`multi_channel_render: false`, the default before
/// upstream 7c388cbabb) it grows to 0.055 (L) and 0.046 (R) on both, which
/// `PIPELINE_TOL` (0.2) would accept.
const STEREO_ECHO_TOL: f32 = 0.02;

/// Deterministic white noise in `[-amp, amp)` from a 32-bit LCG. Different
/// seeds give uncorrelated channels.
fn gen_noise(len: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// Full pipeline (EC + NS + AGC2) at 48 kHz stereo with a far-end signal and
/// different content on each channel.
///
/// `rust_cpp_pipeline_comparison` feeds no render signal and the same signal
/// to both channels, so it never reaches the multi-channel AEC3 paths. Here
/// the two render channels are independent noise, and each capture channel
/// is a different mix of the delayed render plus its own near-end noise.
///
/// This guards the multi-channel defaults of upstream 7c388cbabb: capture
/// downmixed to mono on either side fails the channel check, and Rust render
/// downmixed to mono exceeds `STEREO_ECHO_TOL`. The stimulus also runs
/// the shared comfort noise (297352a2fd) and the joint coarse/refined filter
/// choice (573e746914), but reverting either one in Rust changes the max diff
/// by less than 1e-5, so this test cannot detect those regressions.
#[test]
fn stereo_echo_pipeline_matches_cpp() {
    let stream = StreamConfig::new(48000, 2);
    let n = stream.num_frames();
    let sr = 48000i32;
    let total = n * STEREO_ECHO_FRAMES;

    let render_l = gen_noise(total, 1, 0.3);
    let render_r = gen_noise(total, 2, 0.3);
    let near_l = gen_noise(total, 3, 0.01);
    let near_r = gen_noise(total, 4, 0.01);
    let delayed = |x: &[f32], i: usize| {
        if i >= ECHO_DELAY {
            x[i - ECHO_DELAY]
        } else {
            0.0
        }
    };
    let capture_l: Vec<f32> = (0..total)
        .map(|i| 0.5 * delayed(&render_l, i) + 0.2 * delayed(&render_r, i) + near_l[i])
        .collect();
    let capture_r: Vec<f32> = (0..total)
        .map(|i| 0.2 * delayed(&render_l, i) + 0.5 * delayed(&render_r, i) + near_r[i])
        .collect();

    let mut rust_apm = make_rust_apm(&ComponentConfig {
        name: "all",
        ec: true,
        ns: true,
        agc2: true,
    });
    let mut cpp_apm = sonora_sys::create_apm();
    sonora_sys::apply_config(cpp_apm.pin_mut(), true, true, 1, true, false);

    let mut rust_rev_l = vec![0.0f32; n];
    let mut rust_rev_r = vec![0.0f32; n];
    let mut cpp_rev_l = vec![0.0f32; n];
    let mut cpp_rev_r = vec![0.0f32; n];
    let mut rust_dst_l = vec![0.0f32; n];
    let mut rust_dst_r = vec![0.0f32; n];
    let mut cpp_dst_l = vec![0.0f32; n];
    let mut cpp_dst_r = vec![0.0f32; n];
    let mut worst_l = sonora_bench::comparison::ComparisonResult {
        max_abs_diff: 0.0,
        max_abs_diff_index: 0,
        mean_abs_diff: 0.0,
        mismatches: 0,
        total: 0,
    };
    let mut worst_r = worst_l.clone();
    // Set when an output frame differs between channels. If capture were
    // downmixed to mono, both output channels would be identical.
    let mut rust_channels_differ = false;
    let mut cpp_channels_differ = false;

    for frame_idx in 0..STEREO_ECHO_FRAMES {
        let range = frame_idx * n..(frame_idx + 1) * n;

        let src_slices = [&render_l[range.clone()], &render_r[range.clone()]];
        let mut dst_slices = [rust_rev_l.as_mut_slice(), rust_rev_r.as_mut_slice()];
        rust_apm
            .process_render_f32_with_config(&src_slices, &stream, &stream, &mut dst_slices)
            .unwrap();
        let ret = sonora_sys::process_reverse_stream_f32_2ch(
            cpp_apm.pin_mut(),
            &render_l[range.clone()],
            &render_r[range.clone()],
            sr,
            &mut cpp_rev_l,
            &mut cpp_rev_r,
        );
        assert_eq!(
            ret, 0,
            "C++ ProcessReverseStream failed at frame {frame_idx}"
        );

        let src_slices = [&capture_l[range.clone()], &capture_r[range.clone()]];
        let mut dst_slices = [rust_dst_l.as_mut_slice(), rust_dst_r.as_mut_slice()];
        rust_apm
            .process_capture_f32_with_config(&src_slices, &stream, &stream, &mut dst_slices)
            .unwrap();
        let ret = sonora_sys::process_stream_f32_2ch(
            cpp_apm.pin_mut(),
            &capture_l[range.clone()],
            &capture_r[range],
            sr,
            &mut cpp_dst_l,
            &mut cpp_dst_r,
        );
        assert_eq!(ret, 0, "C++ ProcessStream failed at frame {frame_idx}");

        if frame_idx >= WARMUP_FRAMES {
            rust_channels_differ |= rust_dst_l != rust_dst_r;
            cpp_channels_differ |= cpp_dst_l != cpp_dst_r;
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

    eprintln!(
        "\n=== Stereo echo: Rust vs C++ ({WARMUP_FRAMES}+{} frames) ===",
        STEREO_ECHO_FRAMES - WARMUP_FRAMES
    );
    for (label, result) in [
        ("48k_stereo/echo/L", &worst_l),
        ("48k_stereo/echo/R", &worst_r),
    ] {
        if result.max_abs_diff > 0.0 {
            eprintln!("  DIFF  {label}: {result}");
        } else {
            eprintln!("  OK    {label}: bit-identical");
        }
    }

    assert!(
        rust_channels_differ && cpp_channels_differ,
        "output channels are identical (Rust: {}, C++: {}); capture was not processed as stereo",
        !rust_channels_differ,
        !cpp_channels_differ,
    );
    assert!(
        worst_l.max_abs_diff <= STEREO_ECHO_TOL && worst_r.max_abs_diff <= STEREO_ECHO_TOL,
        "stereo echo pipeline diverged:\n  L: {worst_l}\n  R: {worst_r}",
    );
}

// ── HMM transparent mode: Rust vs C++ ────────────────────────────────────────

#[test]
fn hmm_transparent_mode_matches_cpp() {
    let stream = StreamConfig::new(16000, 1);
    let frames_per_10ms = stream.num_frames();
    let sr = 16000i32;
    let src = gen_signal(frames_per_10ms);

    // Rust: EC with HMM transparent mode.
    let config = Config {
        echo_canceller: Some(EchoCanceller {
            transparent_mode: TransparentModeType::Hmm,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut rust_apm = AudioProcessing::builder().config(config).build();

    // C++: EC with HMM field trial enabled.
    let mut cpp_apm =
        sonora_sys::create_apm_with_field_trials("WebRTC-Aec3TransparentModeHmm/Enabled/");
    sonora_sys::apply_config(cpp_apm.pin_mut(), true, false, 1, false, false);

    let mut worst = sonora_bench::comparison::ComparisonResult {
        max_abs_diff: 0.0,
        max_abs_diff_index: 0,
        mean_abs_diff: 0.0,
        mismatches: 0,
        total: 0,
    };

    for frame_idx in 0..(WARMUP_FRAMES + TEST_FRAMES) {
        let mut rust_dst = vec![0.0f32; frames_per_10ms];
        let mut cpp_dst = vec![0.0f32; frames_per_10ms];

        let src_slices = [src.as_slice()];
        let mut dst_slices = [rust_dst.as_mut_slice()];
        let _ = rust_apm.process_capture_f32_with_config(
            &src_slices,
            &stream,
            &stream,
            &mut dst_slices,
        );

        sonora_sys::process_stream_f32(cpp_apm.pin_mut(), &src, sr, 1, sr, 1, &mut cpp_dst);

        if frame_idx >= WARMUP_FRAMES {
            let result = compare_f32(dst_slices[0], &cpp_dst, 0.0);
            if result.max_abs_diff > worst.max_abs_diff {
                worst = result;
            }
        }
    }

    eprintln!("\n=== HMM transparent mode: Rust vs C++ ===");
    if worst.max_abs_diff > 0.0 {
        eprintln!("  DIFF  16k_mono/ec_hmm: {worst}");
    } else {
        eprintln!("  OK    16k_mono/ec_hmm: bit-identical");
    }

    assert!(
        worst.max_abs_diff <= PIPELINE_TOL,
        "HMM transparent mode diverged: {worst}",
    );
}
