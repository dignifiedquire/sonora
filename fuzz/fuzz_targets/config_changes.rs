#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sonora::config::{EchoCanceller, GainController2, NoiseSuppression, NoiseSuppressionLevel};
use sonora::{AudioProcessing, Config, StreamConfig};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    configs: Vec<FuzzConfig>,
    /// Audio samples to process between config changes
    samples: Vec<f32>,
}

#[derive(Debug, Arbitrary)]
struct FuzzConfig {
    ec_enabled: bool,
    ns_enabled: bool,
    ns_level: u8,
    agc2_enabled: bool,
}

fn ns_level(idx: u8) -> NoiseSuppressionLevel {
    match idx % 4 {
        0 => NoiseSuppressionLevel::Low,
        1 => NoiseSuppressionLevel::Moderate,
        2 => NoiseSuppressionLevel::High,
        _ => NoiseSuppressionLevel::VeryHigh,
    }
}

/// Clamp to valid audio range [-1, 1], replacing NaN/inf with 0.
fn sanitize_sample(s: f32) -> f32 {
    if s.is_finite() {
        s.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fuzz_target!(|input: FuzzInput| {
    let rate = 16000u32;
    let frames = (rate / 100) as usize;

    if input.samples.len() < frames || input.configs.is_empty() {
        return;
    }

    let stream = StreamConfig::new(rate, 1);
    let mut apm = AudioProcessing::builder()
        .capture_config(stream)
        .render_config(stream)
        .build();
    let sanitized: Vec<f32> = input.samples[..frames]
        .iter()
        .copied()
        .map(sanitize_sample)
        .collect();
    let src_slices = [sanitized.as_slice()];
    let mut dest = vec![0.0f32; frames];

    for cfg in &input.configs {
        let config = Config {
            echo_canceller: if cfg.ec_enabled {
                Some(EchoCanceller::default())
            } else {
                None
            },
            noise_suppression: if cfg.ns_enabled {
                Some(NoiseSuppression {
                    level: ns_level(cfg.ns_level),
                    ..Default::default()
                })
            } else {
                None
            },
            gain_controller2: if cfg.agc2_enabled {
                Some(GainController2::default())
            } else {
                None
            },
            ..Default::default()
        };
        apm.apply_config(config);

        let mut dst_slices = [dest.as_mut_slice()];
        let _ = apm.process_capture_f32(&src_slices, &mut dst_slices);
    }
});
