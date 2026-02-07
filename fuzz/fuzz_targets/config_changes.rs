#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora::config::{
    EchoCanceller, GainController2, NoiseSuppression, NoiseSuppressionLevel,
};

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

fuzz_target!(|input: FuzzInput| {
    let rate = 16000usize;
    let frames = rate / 100;

    if input.samples.len() < frames || input.configs.is_empty() {
        return;
    }

    let stream = StreamConfig::new(rate, 1);
    let mut apm = AudioProcessing::new();
    let src = &input.samples[..frames];
    let src_slices = [src];
    let mut dest = vec![0.0f32; frames];

    for cfg in &input.configs {
        let config = Config {
            echo_canceller: EchoCanceller {
                enabled: cfg.ec_enabled,
                ..Default::default()
            },
            noise_suppression: NoiseSuppression {
                enabled: cfg.ns_enabled,
                level: ns_level(cfg.ns_level),
                ..Default::default()
            },
            gain_controller2: GainController2 {
                enabled: cfg.agc2_enabled,
                ..Default::default()
            },
            ..Default::default()
        };
        apm.apply_config(config);

        let mut dst_slices = [dest.as_mut_slice()];
        let _ = apm.process_stream_f32(&src_slices, &stream, &stream, &mut dst_slices);
    }
});
