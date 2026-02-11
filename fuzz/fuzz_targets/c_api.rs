#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sonora_ffi::functions::*;
use sonora_ffi::types::*;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    sample_rate_idx: u8,
    channels: u8,
    operations: Vec<FuzzOp>,
    samples: Vec<f32>,
}

#[derive(Debug, Arbitrary)]
enum FuzzOp {
    ProcessF32,
    ProcessI16,
    ProcessReverseF32,
    ApplyConfig {
        ec: bool,
        ns: bool,
        ns_level: u8,
        agc2: bool,
    },
    SetAnalogLevel(i32),
    SetDelay(i32),
    GetStatistics,
    SetPreGain(f32),
    SetPostGain(f32),
}

fn sample_rate(idx: u8) -> i32 {
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
    let channels = ((input.channels % 2) as i32) + 1;
    let frames = (rate / 100) as usize;
    let total = frames * channels as usize;

    if input.samples.len() < total {
        return;
    }

    let apm = wap_create();
    if apm.is_null() {
        return;
    }

    let stream_config = WapStreamConfig {
        sample_rate_hz: rate,
        num_channels: channels,
    };

    // Prepare per-channel audio buffers with sanitized data
    let sanitized: Vec<f32> = input.samples.iter().copied().map(sanitize_sample).collect();
    let channel_bufs: Vec<Vec<f32>> = (0..channels as usize)
        .map(|ch| sanitized[ch * frames..(ch + 1) * frames].to_vec())
        .collect();
    let src_ptrs: Vec<*const f32> = channel_bufs.iter().map(|b| b.as_ptr()).collect();
    let mut dest_bufs: Vec<Vec<f32>> = (0..channels as usize)
        .map(|_| vec![0.0f32; frames])
        .collect();
    let dest_ptrs: Vec<*mut f32> = dest_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    for op in &input.operations {
        match op {
            FuzzOp::ProcessF32 => {
                let _ = unsafe {
                    wap_process_stream_f32(
                        apm,
                        src_ptrs.as_ptr(),
                        stream_config,
                        stream_config,
                        dest_ptrs.as_ptr(),
                    )
                };
            }
            FuzzOp::ProcessI16 => {
                let src_i16: Vec<i16> = sanitized[..total]
                    .iter()
                    .map(|&s| (s * 16384.0) as i16)
                    .collect();
                let mut dest_i16 = vec![0i16; total];
                let _ = unsafe {
                    wap_process_stream_i16(
                        apm,
                        src_i16.as_ptr(),
                        src_i16.len() as i32,
                        stream_config,
                        stream_config,
                        dest_i16.as_mut_ptr(),
                        dest_i16.len() as i32,
                    )
                };
            }
            FuzzOp::ProcessReverseF32 => {
                let _ = unsafe {
                    wap_process_reverse_stream_f32(
                        apm,
                        src_ptrs.as_ptr(),
                        stream_config,
                        stream_config,
                        dest_ptrs.as_ptr(),
                    )
                };
            }
            FuzzOp::ApplyConfig {
                ec,
                ns,
                ns_level,
                agc2,
            } => {
                let mut config = wap_config_default();
                config.echo_canceller_enabled = *ec;
                config.noise_suppression_enabled = *ns;
                config.noise_suppression_level = match ns_level % 4 {
                    0 => WapNoiseSuppressionLevel::Low,
                    1 => WapNoiseSuppressionLevel::Moderate,
                    2 => WapNoiseSuppressionLevel::High,
                    _ => WapNoiseSuppressionLevel::VeryHigh,
                };
                config.gain_controller2_enabled = *agc2;
                let _ = unsafe { wap_apply_config(apm, config) };
            }
            FuzzOp::SetAnalogLevel(level) => {
                let _ = unsafe { wap_set_stream_analog_level(apm, *level) };
            }
            FuzzOp::SetDelay(delay) => {
                let _ = unsafe { wap_set_stream_delay_ms(apm, *delay) };
            }
            FuzzOp::GetStatistics => {
                let mut stats = std::mem::MaybeUninit::<WapStats>::zeroed();
                let _ = unsafe { wap_get_statistics(apm, stats.as_mut_ptr()) };
            }
            FuzzOp::SetPreGain(gain) => {
                let _ = unsafe { wap_set_capture_pre_gain(apm, *gain) };
            }
            FuzzOp::SetPostGain(gain) => {
                let _ = unsafe { wap_set_capture_post_gain(apm, *gain) };
            }
        }
    }

    unsafe { wap_destroy(apm) };
});
