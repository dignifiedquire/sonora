#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sonora::config::{EchoCanceller, GainController2, NoiseSuppression};
use sonora::{AudioProcessing, Config, StreamConfig};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Sample rate index: 0=8k, 1=16k, 2=32k, 3=48k
    sample_rate_idx: u8,
    /// Number of channels (clamped to 1-2)
    channels: u8,
    /// Audio samples (will be clamped to frame size and [-1, 1])
    samples: Vec<f32>,
}

fn sample_rate(idx: u8) -> u32 {
    match idx % 4 {
        0 => 8000,
        1 => 16000,
        2 => 32000,
        _ => 48000,
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
    let rate = sample_rate(input.sample_rate_idx);
    let channels = (input.channels % 2) as u16 + 1;
    let frames = (rate / 100) as usize; // 10ms

    if input.samples.len() < frames * channels as usize {
        return;
    }

    let config = Config {
        echo_canceller: Some(EchoCanceller::default()),
        noise_suppression: Some(NoiseSuppression::default()),
        gain_controller2: Some(GainController2::default()),
        ..Default::default()
    };
    let stream = StreamConfig::new(rate, channels);
    let mut apm = AudioProcessing::builder()
        .config(config)
        .capture_config(stream)
        .render_config(stream)
        .build();

    // Build per-channel slices from sanitized flat data
    let sanitized: Vec<f32> = input.samples.iter().copied().map(sanitize_sample).collect();
    let src_slices: Vec<&[f32]> = (0..channels as usize)
        .map(|ch| &sanitized[ch * frames..(ch + 1) * frames])
        .collect();
    let mut dest_data: Vec<Vec<f32>> = (0..channels as usize)
        .map(|_| vec![0.0f32; frames])
        .collect();
    let mut dest_slices: Vec<&mut [f32]> = dest_data.iter_mut().map(|v| v.as_mut_slice()).collect();

    let _ = apm.process_capture_f32(&src_slices, &mut dest_slices);
});
