//! Per-component comparison tests for the Noise Suppressor.
//!
//! Verifies that the Rust NoiseSuppressor produces identical output
//! to the C++ reference implementation.
//!
//! Requires the `cpp-comparison` feature and a pre-built C++ library.

#![cfg(feature = "cpp-comparison")]

use sonora_ns::config::{NS_FRAME_SIZE, NsConfig, SuppressionLevel};
use sonora_ns::noise_suppressor::NoiseSuppressor;
use sonora_proptest::comparison::compare_f32;

fn gen_signal(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect()
}

#[test]
fn ns_analyze_process_matches_cpp() {
    let config = NsConfig {
        target_level: SuppressionLevel::K12dB,
    };
    let mut rust_ns = NoiseSuppressor::new(config);
    // C++ NsConfig::SuppressionLevel::k12dB = 1
    let mut cpp_ns = sonora_sys::create_ns(1, 16000, 1);

    let input = gen_signal(NS_FRAME_SIZE);

    for frame_idx in 0..20 {
        let input_arr: &[f32; NS_FRAME_SIZE] = input.as_slice().try_into().unwrap();

        // Analyze
        rust_ns.analyze(input_arr);
        sonora_sys::ns_analyze(cpp_ns.pin_mut(), &input);

        // Process
        let mut rust_frame: [f32; NS_FRAME_SIZE] = *input_arr;
        rust_ns.process(&mut rust_frame);

        let mut cpp_frame = input.clone();
        sonora_sys::ns_process(cpp_ns.pin_mut(), &mut cpp_frame);

        let result = compare_f32(&rust_frame, &cpp_frame, 0.0);
        if result.mismatches > 0 {
            eprintln!("ns frame {frame_idx}: {result}");
        }
        assert!(result.mismatches == 0, "ns frame {frame_idx}: {result}",);
    }
}

#[test]
fn ns_all_levels_match_cpp() {
    let levels = [
        (SuppressionLevel::K6dB, 0u8),
        (SuppressionLevel::K12dB, 1),
        (SuppressionLevel::K18dB, 2),
        (SuppressionLevel::K21dB, 3),
    ];

    let input = gen_signal(NS_FRAME_SIZE);
    let input_arr: &[f32; NS_FRAME_SIZE] = input.as_slice().try_into().unwrap();

    for (rust_level, cpp_level) in &levels {
        let config = NsConfig {
            target_level: *rust_level,
        };
        let mut rust_ns = NoiseSuppressor::new(config);
        let mut cpp_ns = sonora_sys::create_ns(*cpp_level, 16000, 1);

        let mut worst_diff = 0.0f32;

        for frame_idx in 0..20 {
            rust_ns.analyze(input_arr);
            sonora_sys::ns_analyze(cpp_ns.pin_mut(), &input);

            let mut rust_frame = *input_arr;
            rust_ns.process(&mut rust_frame);

            let mut cpp_frame = input.clone();
            sonora_sys::ns_process(cpp_ns.pin_mut(), &mut cpp_frame);

            let result = compare_f32(&rust_frame, &cpp_frame, 0.0);
            worst_diff = worst_diff.max(result.max_abs_diff);
        }

        if worst_diff > 0.0 {
            eprintln!("ns level {:?}: worst_diff={worst_diff:.6e}", rust_level);
        }
        assert!(
            worst_diff == 0.0,
            "ns level {:?}: worst_diff={worst_diff:.6e} (expected bit-identical)",
            rust_level,
        );
    }
}
