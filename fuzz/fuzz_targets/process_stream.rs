#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sonora::{AudioProcessing, Config, StreamConfig};
use sonora::config::{EchoCanceller, GainController2, NoiseSuppression};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Sample rate index: 0=8k, 1=16k, 2=32k, 3=48k
    sample_rate_idx: u8,
    /// Number of channels (clamped to 1-2)
    channels: u8,
    /// Audio samples (will be clamped to frame size)
    samples: Vec<f32>,
}

fn sample_rate(idx: u8) -> usize {
    match idx % 4 {
        0 => 8000,
        1 => 16000,
        2 => 32000,
        _ => 48000,
    }
}

fuzz_target!(|input: FuzzInput| {
    let rate = sample_rate(input.sample_rate_idx);
    let channels = (input.channels % 2) as usize + 1;
    let frames = rate / 100; // 10ms

    if input.samples.len() < frames * channels {
        return;
    }

    let config = Config {
        echo_canceller: EchoCanceller { enabled: true, ..Default::default() },
        noise_suppression: NoiseSuppression { enabled: true, ..Default::default() },
        gain_controller2: GainController2 { enabled: true, ..Default::default() },
        ..Default::default()
    };
    let stream = StreamConfig::new(rate, channels);
    let mut apm = AudioProcessing::builder().config(config).build();

    // Build per-channel slices from flat data
    let src_slices: Vec<&[f32]> = (0..channels)
        .map(|ch| &input.samples[ch * frames..(ch + 1) * frames])
        .collect();
    let mut dest_data: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; frames]).collect();
    let mut dest_slices: Vec<&mut [f32]> = dest_data.iter_mut().map(|v| v.as_mut_slice()).collect();

    let _ = apm.process_stream_f32(&src_slices, &stream, &stream, &mut dest_slices);
});
