//! Integration tests comparing Rust and C++ audio processing output.
//!
//! Verifies that the Rust port produces near-identical output to the C++
//! reference implementation for every supported configuration.
//!
//! Requires the `cpp-comparison` feature and a pre-built C++ library.

#![cfg(feature = "cpp-comparison")]

use sonora::config::{EchoCanceller, GainController2, HighPassFilter, NoiseSuppression};
use sonora::{AudioProcessing, Config, StreamConfig};

// ── Test matrix ──────────────────────────────────────────────────────────────

struct ComponentConfig {
    name: &'static str,
    ec: bool,
    ns: bool,
    agc2: bool,
}

struct Format {
    name: &'static str,
    sample_rate: usize,
    channels: usize,
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

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gen_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect()
}

fn make_rust_apm(cfg: &ComponentConfig) -> AudioProcessing {
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
    AudioProcessing::builder().config(config).build()
}

// ── Divergence tracking ──────────────────────────────────────────────────────

/// Track the worst divergence seen across all frames for a given config.
struct DivergenceTracker {
    label: String,
    max_diff: f32,
    max_diff_idx: usize,
    max_diff_frame: usize,
    rust_val: f32,
    cpp_val: f32,
}

impl DivergenceTracker {
    fn new(label: String) -> Self {
        Self {
            label,
            max_diff: 0.0,
            max_diff_idx: 0,
            max_diff_frame: 0,
            rust_val: 0.0,
            cpp_val: 0.0,
        }
    }

    fn update(&mut self, rust_out: &[f32], cpp_out: &[f32], frame: usize) {
        for (i, (&r, &c)) in rust_out.iter().zip(cpp_out.iter()).enumerate() {
            let diff = (r - c).abs();
            if diff > self.max_diff {
                self.max_diff = diff;
                self.max_diff_idx = i;
                self.max_diff_frame = frame;
                self.rust_val = r;
                self.cpp_val = c;
            }
        }
    }

    fn report(&self) -> String {
        if self.max_diff > 0.0 {
            format!(
                "{}: max_diff={:.6e} at sample [{}] frame {} (rust={}, cpp={})",
                self.label,
                self.max_diff,
                self.max_diff_idx,
                self.max_diff_frame,
                self.rust_val,
                self.cpp_val,
            )
        } else {
            format!("{}: bit-identical", self.label)
        }
    }
}

// ── Test ─────────────────────────────────────────────────────────────────────

const WARMUP_FRAMES: usize = 50;
const TEST_FRAMES: usize = 100;

/// Tolerance for comparing Rust vs C++ output.
///
/// Small FP divergence is expected due to differences in SIMD intrinsic usage
/// and compiler-level FP operation reordering between the Rust and C++ builds.
///
/// This tolerance should be kept as tight as possible and investigated if it
/// needs to be raised.
const TOLERANCE: f32 = 1e-4;

#[test]
fn rust_cpp_output_comparison() {
    let mut trackers: Vec<DivergenceTracker> = Vec::new();

    for fmt in FORMATS {
        let frames_per_10ms = fmt.sample_rate / 100;
        let stream = StreamConfig::new(fmt.sample_rate, fmt.channels);
        let sr = fmt.sample_rate as i32;
        let src_ch = gen_signal(frames_per_10ms);

        for cfg in CONFIGS {
            let label = format!("{}/{}", fmt.name, cfg.name);

            let mut rust_apm = make_rust_apm(cfg);
            let mut cpp_apm = sonora_sys::create_apm();
            sonora_sys::apply_config(cpp_apm.pin_mut(), cfg.ec, cfg.ns, 1, cfg.agc2, false);

            if fmt.channels == 1 {
                let mut tracker = DivergenceTracker::new(label);
                let mut rust_dst = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst = vec![0.0f32; frames_per_10ms];

                for frame_idx in 0..(WARMUP_FRAMES + TEST_FRAMES) {
                    rust_dst.fill(0.0);
                    cpp_dst.fill(0.0);

                    let src_slices = [src_ch.as_slice()];
                    let mut dst_slices = [rust_dst.as_mut_slice()];
                    let _ =
                        rust_apm.process_stream_f32(&src_slices, &stream, &stream, &mut dst_slices);

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
                        tracker.update(&rust_dst, &cpp_dst, frame_idx);
                    }
                }
                trackers.push(tracker);
            } else {
                let src_r = gen_signal(frames_per_10ms);
                let mut tracker_l = DivergenceTracker::new(format!("{label}/L"));
                let mut tracker_r = DivergenceTracker::new(format!("{label}/R"));
                let mut rust_dst_l = vec![0.0f32; frames_per_10ms];
                let mut rust_dst_r = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst_l = vec![0.0f32; frames_per_10ms];
                let mut cpp_dst_r = vec![0.0f32; frames_per_10ms];

                for frame_idx in 0..(WARMUP_FRAMES + TEST_FRAMES) {
                    rust_dst_l.fill(0.0);
                    rust_dst_r.fill(0.0);
                    cpp_dst_l.fill(0.0);
                    cpp_dst_r.fill(0.0);

                    let src_slices = [src_ch.as_slice(), src_r.as_slice()];
                    let mut dst_slices = [rust_dst_l.as_mut_slice(), rust_dst_r.as_mut_slice()];
                    let _ =
                        rust_apm.process_stream_f32(&src_slices, &stream, &stream, &mut dst_slices);

                    sonora_sys::process_stream_f32_2ch(
                        cpp_apm.pin_mut(),
                        &src_ch,
                        &src_r,
                        sr,
                        &mut cpp_dst_l,
                        &mut cpp_dst_r,
                    );

                    if frame_idx >= WARMUP_FRAMES {
                        tracker_l.update(&rust_dst_l, &cpp_dst_l, frame_idx);
                        tracker_r.update(&rust_dst_r, &cpp_dst_r, frame_idx);
                    }
                }
                trackers.push(tracker_l);
                trackers.push(tracker_r);
            }
        }
    }

    // Print full divergence report
    eprintln!("\n=== Rust vs C++ divergence report ({WARMUP_FRAMES}+{TEST_FRAMES} frames) ===");
    for t in &trackers {
        eprintln!("  {}", t.report());
    }
    eprintln!();

    // Fail if any exceed tolerance
    let failures: Vec<_> = trackers.iter().filter(|t| t.max_diff > TOLERANCE).collect();
    if !failures.is_empty() {
        let msgs: Vec<_> = failures.iter().map(|t| t.report()).collect();
        panic!(
            "Rust/C++ divergence exceeds tolerance ({TOLERANCE}) in {} config(s):\n{}",
            failures.len(),
            msgs.join("\n")
        );
    }
}
