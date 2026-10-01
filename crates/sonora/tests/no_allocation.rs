//! Echo cancellation runs in real-time audio callbacks, where allocating
//! memory can block. These tests check that steady-state processing with the
//! echo canceller enabled does not allocate, through the float and int16
//! APIs, at the 16, 32 and 48 kHz processing rates (one, two and three
//! bands), with rate conversion, downmixing, stereo, an unused capture output,
//! a change of the echo delay and noise suppression. They count allocations
//! (`alloc` and `realloc`) of the current thread only, so tests running in
//! parallel do not interfere.
//!
//! Frees are not counted. With multichannel render, the echo canceller
//! rebuilds its block processor when it detects stereo render, and the new
//! comfort noise generator frees its initial noise estimate (one buffer of 260
//! bytes per capture channel) 1000 blocks (4 s) later, while allocations are
//! counted. Upstream frees the same buffer at the same point.
//!
//! Each test also checks that the echo canceller found and removed the echo,
//! so that a test cannot pass because the processing it covers was skipped.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use sonora::config::{EchoCanceller, MaxProcessingRate, NoiseSuppression, Pipeline};
use sonora::{AudioProcessing, Config, StreamConfig};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    if COUNTING.with(Cell::get) {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
    }
}

struct CountingAllocator;

// SAFETY: forwards every call unchanged to the system allocator and only
// counts calls.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Deterministic noise in [-1, 1] (xorshift64).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// The sample format passed to the processing calls.
#[derive(Clone, Copy, Debug)]
enum Format {
    /// Deinterleaved float (`process_*_f32`).
    F32,
    /// Interleaved int16 (`process_*_i16`), as the C API's
    /// `wap_process_stream_i16` uses.
    I16,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    rate: u32,
    /// The highest internal processing rate. At the default of 32 kHz, 44.1
    /// and 48 kHz streams are resampled to 32 kHz and split into two bands;
    /// processing in three bands needs 48 kHz.
    max_processing_rate: MaxProcessingRate,
    format: Format,
    capture_in_channels: u16,
    capture_out_channels: u16,
    render_channels: u16,
    capture_output_used: bool,
    /// Whether the echo delay changes from 60 ms to 100 ms after
    /// `DELAY_CHANGE_SECONDS`, so that the echo canceller has to find the new
    /// delay while allocations are counted.
    delay_change: bool,
    /// Whether noise suppression runs after the echo canceller.
    noise_suppression: bool,
}

impl Scenario {
    const fn mono(rate: u32, format: Format) -> Self {
        Self {
            rate,
            max_processing_rate: MaxProcessingRate::Rate32kHz,
            format,
            capture_in_channels: 1,
            capture_out_channels: 1,
            render_channels: 1,
            capture_output_used: true,
            delay_change: false,
            noise_suppression: false,
        }
    }
}

const MAX_CHANNELS: usize = 2;
const MAX_FRAME: usize = 480;
const SECONDS: usize = 15;
/// Lazy initialisation is allowed during the first 5 seconds.
const WARM_UP_SECONDS: usize = 5;
const DELAY_CHANGE_SECONDS: usize = 9;

/// Whether near-end talk is present in second `s` (every fourth second, so
/// that double talk occurs while allocations are counted).
fn near_end_active(s: usize) -> bool {
    s % 4 == 3
}

struct Outcome {
    allocations: usize,
    erle_db: f64,
    /// Echo removed from the first capture channel over the last 3 seconds
    /// without near-end talk: microphone energy over output energy.
    attenuation_db: f64,
    delay_ms: i32,
    processing_rate: usize,
}

fn run(scenario: Scenario) -> Outcome {
    let rate = scenario.rate as usize;
    let frame = rate / 100;
    let frames = SECONDS * 100;
    let samples = frames * frame;
    let cin = scenario.capture_in_channels as usize;
    let cout = scenario.capture_out_channels as usize;
    let rch = scenario.render_channels as usize;

    let mut noise = Noise(0x1234_5678_9abc_def1);
    // Far end with pauses, independent noise per channel so that stereo
    // render is detected as stereo. The pauses keep a low noise floor rather
    // than exact zeros: AEC3 treats render as stereo only after the channels
    // have differed for 2 s without a break, and identical silent channels in
    // each pause would restart that count, so stereo render would be
    // processed as mono.
    let far: Vec<Vec<f32>> = (0..rch)
        .map(|_| {
            (0..samples)
                .map(|n| {
                    let active = ((n as f32 / rate as f32) * 9.4).sin() > -0.3;
                    if active {
                        0.3 * noise.next()
                    } else {
                        0.001 * noise.next()
                    }
                })
                .collect()
        })
        .collect();
    // Each microphone picks up every far-end channel, with a different gain,
    // delayed by 60 ms (100 ms after the delay change), plus near-end talk
    // and a little noise.
    let mic: Vec<Vec<f32>> = (0..cin)
        .map(|c| {
            (0..samples)
                .map(|n| {
                    let delay_ms = if scenario.delay_change && n >= DELAY_CHANGE_SECONDS * rate {
                        100
                    } else {
                        60
                    };
                    let delay = delay_ms * rate / 1000;
                    let mut echo = 0.0;
                    if n >= delay {
                        for (r, far_r) in far.iter().enumerate() {
                            let gain = if r == c { 0.3 } else { 0.1 };
                            echo += gain * far_r[n - delay];
                        }
                    }
                    let near = if near_end_active(n / rate) {
                        0.2 * noise.next()
                    } else {
                        0.0
                    };
                    echo + near + 0.001 * noise.next()
                })
                .collect()
        })
        .collect();
    let interleave = |channels: &[Vec<f32>]| -> Vec<i16> {
        (0..samples)
            .flat_map(|n| channels.iter().map(move |c| (c[n] * 32767.0) as i16))
            .collect()
    };
    let far_i16 = interleave(&far);
    let mic_i16 = interleave(&mic);

    let capture_in = StreamConfig::new(scenario.rate, scenario.capture_in_channels);
    let capture_out = StreamConfig::new(scenario.rate, scenario.capture_out_channels);
    let render = StreamConfig::new(scenario.rate, scenario.render_channels);
    let mut apm = AudioProcessing::builder()
        .config(Config {
            echo_canceller: Some(EchoCanceller::default()),
            noise_suppression: scenario.noise_suppression.then(NoiseSuppression::default),
            pipeline: Pipeline {
                maximum_internal_processing_rate: scenario.max_processing_rate,
                ..Pipeline::default()
            },
            ..Config::default()
        })
        .capture_config(capture_in)
        .render_config(render)
        .build();
    if !scenario.capture_output_used {
        apm.set_capture_output_used(false);
    }

    let mut render_out = [[0.0f32; MAX_FRAME]; MAX_CHANNELS];
    let mut capture_out_f32 = [[0.0f32; MAX_FRAME]; MAX_CHANNELS];
    let mut render_out_i16 = [0i16; MAX_FRAME * MAX_CHANNELS];
    let mut capture_out_i16 = [0i16; MAX_FRAME * MAX_CHANNELS];
    let [r0, r1] = &mut render_out;
    let mut render_dest = [&mut r0[..frame], &mut r1[..frame]];
    let [c0, c1] = &mut capture_out_f32;
    let mut capture_dest = [&mut c0[..frame], &mut c1[..frame]];

    let mut mic_energy = 0.0f64;
    let mut out_energy = 0.0f64;
    for f in 0..frames {
        if f == WARM_UP_SECONDS * 100 {
            ALLOCATIONS.with(|a| a.set(0));
            COUNTING.with(|c| c.set(true));
        }
        let range = f * frame..(f + 1) * frame;
        let measure = f >= (SECONDS - 3) * 100 && !near_end_active(f / 100);
        match scenario.format {
            Format::F32 => {
                let far_src = [&far[0][range.clone()], &far[rch - 1][range.clone()]];
                let mic_src = [&mic[0][range.clone()], &mic[cin - 1][range.clone()]];
                apm.process_render_f32(&far_src[..rch], &mut render_dest[..rch])
                    .unwrap();
                apm.process_capture_f32_with_config(
                    &mic_src[..cin],
                    &capture_in,
                    &capture_out,
                    &mut capture_dest[..cout],
                )
                .unwrap();
                if measure {
                    for (m, o) in mic_src[0].iter().zip(capture_dest[0].iter()) {
                        mic_energy += f64::from(*m) * f64::from(*m);
                        out_energy += f64::from(*o) * f64::from(*o);
                    }
                }
            }
            Format::I16 => {
                let far_src = &far_i16[f * frame * rch..(f + 1) * frame * rch];
                let mic_src = &mic_i16[f * frame * cin..(f + 1) * frame * cin];
                apm.process_render_i16(far_src, &mut render_out_i16[..frame * rch])
                    .unwrap();
                apm.process_capture_i16_with_config(
                    mic_src,
                    &capture_in,
                    &capture_out,
                    &mut capture_out_i16[..frame * cout],
                )
                .unwrap();
                if measure {
                    for k in 0..frame {
                        let m = f64::from(mic_src[k * cin]);
                        let o = f64::from(capture_out_i16[k * cout]);
                        mic_energy += m * m;
                        out_energy += o * o;
                    }
                }
            }
        }
        let _ = apm.statistics();
    }
    COUNTING.with(|c| c.set(false));

    let stats = apm.statistics();
    Outcome {
        allocations: ALLOCATIONS.with(Cell::get),
        erle_db: stats.echo_return_loss_enhancement.unwrap(),
        attenuation_db: 10.0 * (mic_energy / out_energy.max(f64::MIN_POSITIVE)).log10(),
        delay_ms: stats.delay_ms.unwrap(),
        processing_rate: apm.proc_sample_rate_hz(),
    }
}

fn check(scenario: Scenario) {
    let outcome = run(scenario);
    // Streams above 32 kHz are processed at the maximum processing rate,
    // which decides whether the scenario covers two bands or three.
    let expected_rate = if scenario.rate <= 32_000 {
        scenario.rate
    } else {
        scenario.max_processing_rate.as_hz()
    };
    assert_eq!(
        outcome.processing_rate, expected_rate as usize,
        "processing rate: {scenario:?}"
    );
    assert_eq!(
        outcome.allocations, 0,
        "steady-state processing allocated: {scenario:?}"
    );

    // Without echo in the microphone signal the ERLE stays below 1 dB.
    // Without echo removal the attenuation still reaches about 6.4 dB: the
    // average of two microphones carries less echo than the first one, which
    // is the one measured (3.5 dB), and processing 48 kHz at 32 kHz drops the
    // noise above 16 kHz (2.9 dB). Noise suppression raises that to 7.3 dB.
    // Here both are above 15 dB, so 10 dB shows that the echo canceller ran
    // and converged.
    assert!(
        outcome.erle_db > 10.0,
        "ERLE is {:.1} dB: {scenario:?}",
        outcome.erle_db
    );
    if scenario.capture_output_used {
        assert!(
            outcome.attenuation_db > 10.0,
            "echo attenuated by only {:.1} dB: {scenario:?}",
            outcome.attenuation_db
        );
    }
    // The delay estimate must follow the echo, including after a change.
    let expected_delay_ms = if scenario.delay_change { 100 } else { 60 };
    assert!(
        (outcome.delay_ms - expected_delay_ms).abs() <= 10,
        "delay estimate {} ms, expected {expected_delay_ms} ms: {scenario:?}",
        outcome.delay_ms
    );
}

#[test]
fn f32_16k_mono() {
    check(Scenario::mono(16_000, Format::F32));
}

#[test]
fn f32_32k_mono() {
    check(Scenario::mono(32_000, Format::F32));
}

#[test]
fn f32_48k_mono() {
    check(Scenario::mono(48_000, Format::F32));
}

/// The only scenario that processes at 48 kHz, in three bands.
#[test]
fn f32_48k_mono_three_bands() {
    check(Scenario {
        max_processing_rate: MaxProcessingRate::Rate48kHz,
        ..Scenario::mono(48_000, Format::F32)
    });
}

#[test]
fn i16_16k_mono() {
    check(Scenario::mono(16_000, Format::I16));
}

#[test]
fn i16_48k_mono() {
    check(Scenario::mono(48_000, Format::I16));
}

#[test]
fn f32_48k_stereo_in_mono_out() {
    check(Scenario {
        capture_in_channels: 2,
        ..Scenario::mono(48_000, Format::F32)
    });
}

#[test]
fn f32_48k_stereo() {
    check(Scenario {
        capture_in_channels: 2,
        capture_out_channels: 2,
        render_channels: 2,
        ..Scenario::mono(48_000, Format::F32)
    });
}

/// 44.1 kHz is resampled to the 32 kHz processing rate and back, so this
/// covers the int16 downmix with resampling.
#[test]
fn i16_44k_stereo_in_mono_out() {
    check(Scenario {
        capture_in_channels: 2,
        ..Scenario::mono(44_100, Format::I16)
    });
}

/// Covers the int16 multi-channel conversions with resampling.
#[test]
fn i16_44k_stereo() {
    check(Scenario {
        capture_in_channels: 2,
        capture_out_channels: 2,
        render_channels: 2,
        ..Scenario::mono(44_100, Format::I16)
    });
}

/// Noise suppression runs in the same capture callback: here two
/// suppressors, one per capture channel, each over two bands.
#[test]
fn f32_48k_stereo_noise_suppression() {
    check(Scenario {
        capture_in_channels: 2,
        capture_out_channels: 2,
        render_channels: 2,
        noise_suppression: true,
        ..Scenario::mono(48_000, Format::F32)
    });
}

#[test]
fn f32_48k_capture_output_unused() {
    check(Scenario {
        capture_output_used: false,
        ..Scenario::mono(48_000, Format::F32)
    });
}

#[test]
fn f32_48k_delay_change() {
    check(Scenario {
        delay_change: true,
        ..Scenario::mono(48_000, Format::F32)
    });
}
