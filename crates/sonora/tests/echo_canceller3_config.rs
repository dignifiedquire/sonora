//! `AudioProcessingBuilder::echo_canceller3_config`: a caller's AEC3 tuning
//! reaches the echo canceller, also for stereo render, and the default path
//! is unchanged.

use sonora::config::{EchoCanceller, EchoCanceller3Config};
use sonora::{AudioProcessing, Config, StreamConfig};

const RATE: usize = 48_000;
const FRAME: usize = RATE / 100;

/// Deterministic white noise in [-0.5, 0.5] (xorshift).
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// Runs 4 s of far-end noise with a delayed echo plus near-end bursts
/// (double talk) through an APM and returns the processed capture signal.
fn run(apm: &mut AudioProcessing) -> Vec<f32> {
    let len = 4 * RATE;
    let far = noise(len, 1);
    let near = noise(len, 2);
    let mut output = Vec::with_capacity(len);
    let mut unused = [0.0f32; FRAME];
    let mut out = [0.0f32; FRAME];
    for f in 0..len / FRAME {
        let range = f * FRAME..(f + 1) * FRAME;
        let mic: Vec<f32> = range
            .clone()
            .map(|n| {
                let echo = if n >= 2400 { 0.3 * far[n - 2400] } else { 0.0 };
                // Near-end talk in every other half second.
                let talk = if (n / (RATE / 2)) % 2 == 1 {
                    0.2 * near[n]
                } else {
                    0.0
                };
                echo + talk
            })
            .collect();
        apm.process_render_f32(&[&far[range]], &mut [&mut unused])
            .unwrap();
        apm.process_capture_f32(&[&mic], &mut [&mut out]).unwrap();
        output.extend_from_slice(&out);
    }
    output
}

fn apm(aec3: Option<EchoCanceller3Config>) -> AudioProcessing {
    apm_with_render_channels(aec3, 1)
}

fn apm_with_render_channels(
    aec3: Option<EchoCanceller3Config>,
    render_channels: u16,
) -> AudioProcessing {
    let mut builder = AudioProcessing::builder()
        .config(Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Config::default()
        })
        .capture_config(StreamConfig::new(RATE as u32, 1))
        .render_config(StreamConfig::new(RATE as u32, render_channels));
    if let Some(aec3) = aec3 {
        builder = builder.echo_canceller3_config(aec3);
    }
    builder.build()
}

#[test]
fn default_config_gives_the_same_output_as_no_config() {
    let without = run(&mut apm(None));
    let with_default = run(&mut apm(Some(EchoCanceller3Config::default())));
    assert_eq!(without, with_default);
}

#[test]
fn custom_tuning_reaches_the_echo_canceller() {
    let default = run(&mut apm(None));
    let mut tuned_config = EchoCanceller3Config::default();
    // A much stronger assumed echo path: the suppressor removes more.
    tuned_config.ep_strength.default_gain = 10.0;
    let tuned = run(&mut apm(Some(tuned_config)));
    assert_ne!(default, tuned);
}

/// Runs 8 s of independent far-end noise on two render channels with a
/// delayed echo of both on one microphone, and returns the processed capture
/// signal of the last 2 s: by then AEC3 has detected stereo render (after 2 s
/// of differing channels) and processes with its multichannel config.
fn run_stereo(apm: &mut AudioProcessing) -> Vec<f32> {
    let len = 8 * RATE;
    let far = [noise(len, 3), noise(len, 4)];
    let mut output = Vec::with_capacity(2 * RATE);
    let mut unused = [[0.0f32; FRAME]; 2];
    let mut out = [0.0f32; FRAME];
    for f in 0..len / FRAME {
        let range = f * FRAME..(f + 1) * FRAME;
        let mic: Vec<f32> = range
            .clone()
            .map(|n| {
                if n >= 2400 {
                    0.3 * far[0][n - 2400] + 0.1 * far[1][n - 2400]
                } else {
                    0.0
                }
            })
            .collect();
        let [u0, u1] = &mut unused;
        apm.process_render_f32(
            &[&far[0][range.clone()], &far[1][range.clone()]],
            &mut [u0, u1],
        )
        .unwrap();
        apm.process_capture_f32(&[&mic], &mut [&mut out]).unwrap();
        if range.start >= len - 2 * RATE {
            output.extend_from_slice(&out);
        }
    }
    output
}

#[test]
fn custom_config_with_a_fixed_capture_delay_builds_and_processes() {
    // The default multichannel config has no fixed capture delay. Combining
    // the two used to trip ConfigSelector's compatibility check in debug
    // builds, even with mono render.
    let mut config = EchoCanceller3Config::default();
    config.delay.fixed_capture_delay_samples = 480;
    let output = run(&mut apm(Some(config)));
    assert!(output.iter().all(|s| s.is_finite()));
}

#[test]
fn custom_tuning_reaches_the_echo_canceller_with_stereo_render() {
    let default = run_stereo(&mut apm_with_render_channels(None, 2));
    let mut tuned_config = EchoCanceller3Config::default();
    tuned_config.ep_strength.default_gain = 10.0;
    let tuned = run_stereo(&mut apm_with_render_channels(Some(tuned_config), 2));
    assert_ne!(default, tuned);
}
