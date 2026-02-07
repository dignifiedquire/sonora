//! Exported `extern "C"` functions for the audio processing C API.
//!
//! # Symbol prefix
//!
//! All public symbols use the `wap_` prefix.

use std::ptr;
use std::slice;

use crate::AudioProcessing;
use crate::config::{Config, PlayoutAudioDeviceInfo, RuntimeSetting};
use crate::stream_config::StreamConfig;

use super::panic_guard::{ffi_guard, ffi_guard_ptr};
use super::types::{WapAudioProcessing, WapConfig, WapError, WapStats, WapStreamConfig};

// ─── Version ─────────────────────────────────────────────────────────

/// Returns a pointer to a static null-terminated version string.
///
/// The returned pointer is valid for the lifetime of the process.
#[unsafe(no_mangle)]
pub extern "C" fn wap_version() -> *const std::ffi::c_char {
    // Safety: the byte string is a static literal with a trailing NUL.
    c"0.1.0".as_ptr()
}

// ─── Lifecycle ───────────────────────────────────────────────────────

/// Returns a default-initialized configuration.
#[unsafe(no_mangle)]
pub extern "C" fn wap_config_default() -> WapConfig {
    WapConfig::from_rust(&Config::default())
}

/// Creates a new audio processing instance with default configuration.
///
/// Returns `NULL` on allocation failure or internal error.
/// The caller owns the returned pointer and must free it with
/// [`wap_destroy()`].
#[unsafe(no_mangle)]
pub extern "C" fn wap_create() -> *mut WapAudioProcessing {
    ffi_guard_ptr! {
        let apm = AudioProcessing::new();
        let boxed = Box::new(WapAudioProcessing { inner: apm });
        Box::into_raw(boxed)
    }
}

/// Creates a new audio processing instance with the given configuration.
///
/// Returns `NULL` on allocation failure or internal error.
/// The caller owns the returned pointer and must free it with
/// [`wap_destroy()`].
#[unsafe(no_mangle)]
pub extern "C" fn wap_create_with_config(config: WapConfig) -> *mut WapAudioProcessing {
    ffi_guard_ptr! {
        let rust_config = config.to_rust();
        let apm = AudioProcessing::builder().config(rust_config).build();
        let boxed = Box::new(WapAudioProcessing { inner: apm });
        Box::into_raw(boxed)
    }
}

/// Destroys an audio processing instance and frees its memory.
///
/// Passing `NULL` is a safe no-op. After this call the pointer is invalid.
#[unsafe(no_mangle)]
pub extern "C" fn wap_destroy(apm: *mut WapAudioProcessing) {
    if !apm.is_null() {
        // Safety: we created this pointer via Box::into_raw in wap_create/
        // wap_create_with_config, and the caller guarantees single ownership.
        let _ = unsafe { Box::from_raw(apm) };
    }
}

// ─── Configuration ───────────────────────────────────────────────────

/// Applies a new configuration to the audio processing instance.
///
/// Returns `WapError::NullPointer` if `apm` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_apply_config(apm: *mut WapAudioProcessing, config: WapConfig) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        // Safety: the caller guarantees the pointer is valid and not aliased.
        let apm = unsafe { &mut *apm };
        let rust_config = config.to_rust();
        apm.inner.apply_config(rust_config);
        WapError::None
    }
}

/// Retrieves the current configuration.
///
/// Returns `WapError::NullPointer` if `apm` or `config_out` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_get_config(
    apm: *const WapAudioProcessing,
    config_out: *mut WapConfig,
) -> WapError {
    ffi_guard! {
        if apm.is_null() || config_out.is_null() {
            return WapError::NullPointer;
        }
        // Safety: the caller guarantees the pointers are valid.
        let apm = unsafe { &*apm };
        let rust_config = apm.inner.get_config();
        let c_config = WapConfig::from_rust(rust_config);
        unsafe { ptr::write(config_out, c_config) };
        WapError::None
    }
}

// ─── Initialization ──────────────────────────────────────────────────

/// Initializes the processing pipeline with explicit stream configurations.
///
/// Sets sample rates and channel counts for all four audio paths (capture
/// input, capture output, reverse input, reverse output) atomically,
/// triggering a full reinitialisation of internal buffers and submodules.
///
/// Returns `WapError::NullPointer` if `apm` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_initialize(
    apm: *mut WapAudioProcessing,
    input_config: WapStreamConfig,
    output_config: WapStreamConfig,
    reverse_input_config: WapStreamConfig,
    reverse_output_config: WapStreamConfig,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        let in_cfg = input_config.to_rust();
        let out_cfg = output_config.to_rust();
        let rev_in_cfg = reverse_input_config.to_rust();
        let rev_out_cfg = reverse_output_config.to_rust();
        apm.inner.initialize(&in_cfg, &out_cfg, &rev_in_cfg, &rev_out_cfg);
        WapError::None
    }
}

// ─── Processing (float, deinterleaved) ──────────────────────────────

/// Maximum number of channels accepted to prevent unreasonable allocations.
const MAX_CHANNELS: usize = 8;

/// Reconstructs `&[&[f32]]` from C-style `*const *const f32`.
///
/// Returns `None` if any pointer is null or `num_channels` is out of range.
///
/// # Safety
///
/// Each channel pointer must point to at least `num_frames` valid `f32`
/// values. The caller must guarantee the pointers remain valid for the
/// duration of the call.
unsafe fn rebuild_channel_slices<'a>(
    data: *const *const f32,
    num_channels: usize,
    num_frames: usize,
) -> Option<Vec<&'a [f32]>> {
    // 0 channels is a format validation error, not a null-pointer error.
    // Return an empty vec so the caller can forward to handle_unsupported_formats
    // which will return the correct error code (BadNumberChannels).
    if num_channels == 0 {
        return Some(Vec::new());
    }
    if data.is_null() || num_channels > MAX_CHANNELS {
        return None;
    }
    // Safety: caller guarantees `data` points to `num_channels` pointers.
    let ptrs = unsafe { slice::from_raw_parts(data, num_channels) };
    let mut slices = Vec::with_capacity(num_channels);
    for &p in ptrs {
        if p.is_null() {
            return None;
        }
        // Safety: caller guarantees each pointer has `num_frames` valid f32s.
        slices.push(unsafe { slice::from_raw_parts(p, num_frames) });
    }
    Some(slices)
}

/// Reconstructs `&mut [&mut [f32]]` from C-style `*const *mut f32`.
///
/// # Safety
///
/// Same as [`rebuild_channel_slices`], plus the output buffers must not
/// alias the input buffers or each other.
unsafe fn rebuild_channel_slices_mut<'a>(
    data: *const *mut f32,
    num_channels: usize,
    num_frames: usize,
) -> Option<Vec<&'a mut [f32]>> {
    // 0 channels is a format validation error, not a null-pointer error.
    if num_channels == 0 {
        return Some(Vec::new());
    }
    if data.is_null() || num_channels > MAX_CHANNELS {
        return None;
    }
    // Safety: caller guarantees `data` points to `num_channels` pointers.
    let ptrs = unsafe { slice::from_raw_parts(data, num_channels) };
    let mut slices = Vec::with_capacity(num_channels);
    for &p in ptrs {
        if p.is_null() {
            return None;
        }
        // Safety: caller guarantees each pointer has `num_frames` valid f32s
        // and that output buffers do not alias.
        slices.push(unsafe { slice::from_raw_parts_mut(p, num_frames) });
    }
    Some(slices)
}

/// Converts a [`crate::Error`] to a [`WapError`].
fn rust_error_to_wap(err: crate::Error) -> WapError {
    match err {
        crate::Error::BadSampleRate => WapError::BadSampleRate,
        crate::Error::BadNumberChannels => WapError::BadNumberChannels,
        crate::Error::BadStreamParameter => WapError::BadStreamParameter,
    }
}

/// Processes a capture audio frame (float, deinterleaved).
///
/// - `src`: array of `input_config.num_channels` pointers, each pointing
///   to `input_config.sample_rate_hz / 100` samples.
/// - `dest`: array of `output_config.num_channels` pointers (output buffers).
///
/// Returns `WapError::None` on success.
#[unsafe(no_mangle)]
pub extern "C" fn wap_process_stream_f32(
    apm: *mut WapAudioProcessing,
    src: *const *const f32,
    input_config: WapStreamConfig,
    output_config: WapStreamConfig,
    dest: *const *mut f32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let in_cfg = input_config.to_rust();
        let out_cfg = output_config.to_rust();
        let in_frames = in_cfg.num_frames();
        let out_frames = out_cfg.num_frames();

        // Safety: caller guarantees valid pointers with correct dimensions.
        let Some(src_slices) = (unsafe {
            rebuild_channel_slices(src, in_cfg.num_channels(), in_frames)
        }) else {
            return WapError::NullPointer;
        };
        let Some(mut dest_slices) = (unsafe {
            rebuild_channel_slices_mut(dest, out_cfg.num_channels(), out_frames)
        }) else {
            return WapError::NullPointer;
        };

        let apm = unsafe { &mut *apm };
        let dest_refs: &mut [&mut [f32]] = &mut dest_slices;
        match apm.inner.process_stream_f32(&src_slices, &in_cfg, &out_cfg, dest_refs) {
            Ok(()) => WapError::None,
            Err(e) => rust_error_to_wap(e),
        }
    }
}

/// Processes a reverse (render / far-end) audio frame (float, deinterleaved).
#[unsafe(no_mangle)]
pub extern "C" fn wap_process_reverse_stream_f32(
    apm: *mut WapAudioProcessing,
    src: *const *const f32,
    input_config: WapStreamConfig,
    output_config: WapStreamConfig,
    dest: *const *mut f32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let in_cfg = input_config.to_rust();
        let out_cfg = output_config.to_rust();
        let in_frames = in_cfg.num_frames();
        let out_frames = out_cfg.num_frames();

        // Safety: caller guarantees valid pointers with correct dimensions.
        let Some(src_slices) = (unsafe {
            rebuild_channel_slices(src, in_cfg.num_channels(), in_frames)
        }) else {
            return WapError::NullPointer;
        };
        let Some(mut dest_slices) = (unsafe {
            rebuild_channel_slices_mut(dest, out_cfg.num_channels(), out_frames)
        }) else {
            return WapError::NullPointer;
        };

        let apm = unsafe { &mut *apm };
        let dest_refs: &mut [&mut [f32]] = &mut dest_slices;
        match apm.inner.process_reverse_stream_f32(&src_slices, &in_cfg, &out_cfg, dest_refs) {
            Ok(()) => WapError::None,
            Err(e) => rust_error_to_wap(e),
        }
    }
}

// ─── Processing (int16, interleaved) ─────────────────────────────────

/// Processes a capture audio frame (int16, interleaved).
///
/// - `src`: pointer to `num_frames * num_channels` interleaved i16 samples.
/// - `dest`: pointer to output buffer of same size.
/// - Input and output configs must have native rates (8k/16k/32k/48k) and
///   matching rates and channel counts.
#[unsafe(no_mangle)]
pub extern "C" fn wap_process_stream_i16(
    apm: *mut WapAudioProcessing,
    src: *const i16,
    src_len: i32,
    input_config: WapStreamConfig,
    output_config: WapStreamConfig,
    dest: *mut i16,
    dest_len: i32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() || src.is_null() || dest.is_null() {
            return WapError::NullPointer;
        }
        let in_cfg = input_config.to_rust();
        let out_cfg = output_config.to_rust();
        let expected_src_len = in_cfg.num_frames() * in_cfg.num_channels();
        let expected_dest_len = out_cfg.num_frames() * out_cfg.num_channels();

        if (src_len as usize) < expected_src_len || (dest_len as usize) < expected_dest_len {
            return WapError::BadDataLength;
        }

        // Safety: caller guarantees valid pointers with at least the
        // required number of samples.
        let src_slice = unsafe { slice::from_raw_parts(src, expected_src_len) };
        let dest_slice = unsafe { slice::from_raw_parts_mut(dest, expected_dest_len) };

        let apm = unsafe { &mut *apm };
        match apm.inner.process_stream_i16(src_slice, &in_cfg, &out_cfg, dest_slice) {
            Ok(()) => WapError::None,
            Err(e) => rust_error_to_wap(e),
        }
    }
}

/// Processes a reverse (render / far-end) audio frame (int16, interleaved).
#[unsafe(no_mangle)]
pub extern "C" fn wap_process_reverse_stream_i16(
    apm: *mut WapAudioProcessing,
    src: *const i16,
    src_len: i32,
    input_config: WapStreamConfig,
    output_config: WapStreamConfig,
    dest: *mut i16,
    dest_len: i32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() || src.is_null() || dest.is_null() {
            return WapError::NullPointer;
        }
        let in_cfg = input_config.to_rust();
        let out_cfg = output_config.to_rust();
        let expected_src_len = in_cfg.num_frames() * in_cfg.num_channels();
        let expected_dest_len = out_cfg.num_frames() * out_cfg.num_channels();

        if (src_len as usize) < expected_src_len || (dest_len as usize) < expected_dest_len {
            return WapError::BadDataLength;
        }

        // Safety: caller guarantees valid pointers.
        let src_slice = unsafe { slice::from_raw_parts(src, expected_src_len) };
        let dest_slice = unsafe { slice::from_raw_parts_mut(dest, expected_dest_len) };

        let apm = unsafe { &mut *apm };
        match apm.inner.process_reverse_stream_i16(src_slice, &in_cfg, &out_cfg, dest_slice) {
            Ok(()) => WapError::None,
            Err(e) => rust_error_to_wap(e),
        }
    }
}

// ─── Analog level (for AGC) ──────────────────────────────────────────

/// Sets the applied input volume (e.g. from the OS mixer).
///
/// Must be called before [`wap_process_stream_f32()`] if the input volume
/// controller is enabled. Value should be in range `[0, 255]`.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_stream_analog_level(
    apm: *mut WapAudioProcessing,
    level: i32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_stream_analog_level(level);
        WapError::None
    }
}

/// Returns the recommended analog level from AGC.
///
/// Should be called after [`wap_process_stream_f32()`] to obtain the
/// recommended new analog level. Returns 0 if `apm` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_recommended_stream_analog_level(apm: *const WapAudioProcessing) -> i32 {
    if apm.is_null() {
        return 0;
    }
    let apm = unsafe { &*apm };
    apm.inner.recommended_stream_analog_level()
}

// ─── Stream delay ────────────────────────────────────────────────────

/// Sets the delay in ms between render and capture.
///
/// The delay is clamped to `[0, 500]`. Returns `WapError::BadStreamParameter`
/// if clamping was necessary (processing still proceeds).
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_stream_delay_ms(apm: *mut WapAudioProcessing, delay: i32) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        match apm.inner.set_stream_delay_ms(delay) {
            Ok(()) => WapError::None,
            Err(e) => rust_error_to_wap(e),
        }
    }
}

/// Returns the current stream delay in ms. Returns 0 if `apm` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_stream_delay_ms(apm: *const WapAudioProcessing) -> i32 {
    if apm.is_null() {
        return 0;
    }
    let apm = unsafe { &*apm };
    apm.inner.stream_delay_ms()
}

// ─── Runtime settings ────────────────────────────────────────────────

/// Sets the capture pre-gain factor via runtime setting.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_capture_pre_gain(apm: *mut WapAudioProcessing, gain: f32) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_runtime_setting(RuntimeSetting::CapturePreGain(gain));
        WapError::None
    }
}

/// Sets the capture post-gain factor via runtime setting.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_capture_post_gain(apm: *mut WapAudioProcessing, gain: f32) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_runtime_setting(RuntimeSetting::CapturePostGain(gain));
        WapError::None
    }
}

/// Sets the fixed post-gain in dB via runtime setting (range: 0..=90).
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_capture_fixed_post_gain(
    apm: *mut WapAudioProcessing,
    gain_db: f32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_runtime_setting(RuntimeSetting::CaptureFixedPostGain(gain_db));
        WapError::None
    }
}

/// Notifies of a playout volume change via runtime setting.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_playout_volume(apm: *mut WapAudioProcessing, volume: i32) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_runtime_setting(RuntimeSetting::PlayoutVolumeChange(volume));
        WapError::None
    }
}

/// Notifies of a playout audio device change via runtime setting.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_playout_audio_device(
    apm: *mut WapAudioProcessing,
    device_id: i32,
    max_volume: i32,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        let info = PlayoutAudioDeviceInfo {
            id: device_id,
            max_volume,
        };
        apm.inner.set_runtime_setting(RuntimeSetting::PlayoutAudioDeviceChange(info));
        WapError::None
    }
}

/// Sets whether the capture output is used via runtime setting.
#[unsafe(no_mangle)]
pub extern "C" fn wap_set_capture_output_used(
    apm: *mut WapAudioProcessing,
    used: bool,
) -> WapError {
    ffi_guard! {
        if apm.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &mut *apm };
        apm.inner.set_runtime_setting(RuntimeSetting::CaptureOutputUsed(used));
        WapError::None
    }
}

// ─── Statistics ──────────────────────────────────────────────────────

/// Retrieves current processing statistics.
///
/// Returns `WapError::NullPointer` if `apm` or `stats_out` is null.
#[unsafe(no_mangle)]
pub extern "C" fn wap_get_statistics(
    apm: *const WapAudioProcessing,
    stats_out: *mut WapStats,
) -> WapError {
    ffi_guard! {
        if apm.is_null() || stats_out.is_null() {
            return WapError::NullPointer;
        }
        let apm = unsafe { &*apm };
        let rust_stats = apm.inner.get_statistics();
        let c_stats = WapStats::from_rust(&rust_stats);
        unsafe { ptr::write(stats_out, c_stats) };
        WapError::None
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::types::WapNoiseSuppressionLevel;

    #[test]
    fn version_returns_non_null() {
        let ptr = wap_version();
        assert!(!ptr.is_null());
        // Safety: wap_version returns a static NUL-terminated string.
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(cstr.to_str().unwrap(), "0.1.0");
    }

    #[test]
    fn create_and_destroy() {
        let apm = wap_create();
        assert!(!apm.is_null());
        wap_destroy(apm);
    }

    #[test]
    fn destroy_null_is_safe() {
        wap_destroy(ptr::null_mut());
    }

    #[test]
    fn create_with_config() {
        let mut config = wap_config_default();
        config.echo_canceller_enabled = true;
        config.noise_suppression_enabled = true;
        config.noise_suppression_level = WapNoiseSuppressionLevel::VeryHigh;

        let apm = wap_create_with_config(config);
        assert!(!apm.is_null());

        // Verify config was applied.
        let mut config_out = wap_config_default();
        let err = wap_get_config(apm, &mut config_out);
        assert_eq!(err, WapError::None);
        assert!(config_out.echo_canceller_enabled);
        assert!(config_out.noise_suppression_enabled);
        assert_eq!(
            config_out.noise_suppression_level,
            WapNoiseSuppressionLevel::VeryHigh
        );

        wap_destroy(apm);
    }

    #[test]
    fn config_default_matches_rust_default() {
        let c_config = wap_config_default();
        let rust_config = Config::default();
        let roundtrip = c_config.to_rust();
        assert_eq!(
            rust_config.echo_canceller.enabled,
            roundtrip.echo_canceller.enabled
        );
        assert_eq!(
            rust_config.noise_suppression.enabled,
            roundtrip.noise_suppression.enabled
        );
        assert_eq!(
            rust_config.high_pass_filter.enabled,
            roundtrip.high_pass_filter.enabled
        );
        assert_eq!(
            rust_config.gain_controller2.enabled,
            roundtrip.gain_controller2.enabled
        );
    }

    #[test]
    fn apply_config_null_returns_error() {
        let config = wap_config_default();
        let err = wap_apply_config(ptr::null_mut(), config);
        assert_eq!(err, WapError::NullPointer);
    }

    #[test]
    fn get_config_null_returns_error() {
        let err = wap_get_config(ptr::null(), ptr::null_mut());
        assert_eq!(err, WapError::NullPointer);
    }

    #[test]
    fn apply_and_get_config_roundtrip() {
        let apm = wap_create();
        assert!(!apm.is_null());

        // Apply a non-default config.
        let mut config = wap_config_default();
        config.echo_canceller_enabled = true;
        config.noise_suppression_enabled = true;
        config.high_pass_filter_enabled = true;
        config.gain_controller2_enabled = true;
        config.gain_controller2_fixed_digital_gain_db = 5.0;
        config.pre_amplifier_enabled = true;
        config.pre_amplifier_fixed_gain_factor = 2.5;

        let err = wap_apply_config(apm, config);
        assert_eq!(err, WapError::None);

        // Get config back.
        let mut config_out = wap_config_default();
        let err = wap_get_config(apm, &mut config_out);
        assert_eq!(err, WapError::None);
        assert!(config_out.echo_canceller_enabled);
        assert!(config_out.noise_suppression_enabled);
        assert!(config_out.high_pass_filter_enabled);
        assert!(config_out.gain_controller2_enabled);
        assert_eq!(config_out.gain_controller2_fixed_digital_gain_db, 5.0);
        assert!(config_out.pre_amplifier_enabled);
        assert_eq!(config_out.pre_amplifier_fixed_gain_factor, 2.5);

        wap_destroy(apm);
    }

    #[test]
    fn get_config_null_config_out_returns_error() {
        let apm = wap_create();
        assert!(!apm.is_null());
        let err = wap_get_config(apm, ptr::null_mut());
        assert_eq!(err, WapError::NullPointer);
        wap_destroy(apm);
    }

    // ─── Processing tests ────────────────────────────────────────

    #[test]
    fn process_stream_f32_null_apm() {
        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let err = wap_process_stream_f32(ptr::null_mut(), ptr::null(), config, config, ptr::null());
        assert_eq!(err, WapError::NullPointer);
    }

    #[test]
    fn process_stream_f32_silence_passthrough() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src_data = [0.0f32; 160];
        let src_ptrs: [*const f32; 1] = [src_data.as_ptr()];
        let mut dest_data = [1.0f32; 160];
        let dest_ptrs: [*mut f32; 1] = [dest_data.as_mut_ptr()];

        let err =
            wap_process_stream_f32(apm, src_ptrs.as_ptr(), config, config, dest_ptrs.as_ptr());
        assert_eq!(err, WapError::None);

        for &s in &dest_data {
            assert!(s.abs() < 1e-6, "expected silence, got {s}");
        }
        wap_destroy(apm);
    }

    #[test]
    fn process_stream_f32_no_processing_passthrough() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src_data: Vec<f32> = (0..160).map(|i| (i as f32 / 160.0) * 2.0 - 1.0).collect();
        let src_ptrs: [*const f32; 1] = [src_data.as_ptr()];
        let mut dest_data = vec![0.0f32; 160];
        let dest_ptrs: [*mut f32; 1] = [dest_data.as_mut_ptr()];

        let err =
            wap_process_stream_f32(apm, src_ptrs.as_ptr(), config, config, dest_ptrs.as_ptr());
        assert_eq!(err, WapError::None);

        for i in 0..160 {
            assert_eq!(src_data[i], dest_data[i], "sample {i} mismatch");
        }
        wap_destroy(apm);
    }

    #[test]
    fn process_reverse_stream_f32_works() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src_data = [0.5f32; 160];
        let src_ptrs: [*const f32; 1] = [src_data.as_ptr()];
        let mut dest_data = [0.0f32; 160];
        let dest_ptrs: [*mut f32; 1] = [dest_data.as_mut_ptr()];

        let err = wap_process_reverse_stream_f32(
            apm,
            src_ptrs.as_ptr(),
            config,
            config,
            dest_ptrs.as_ptr(),
        );
        assert_eq!(err, WapError::None);
        wap_destroy(apm);
    }

    #[test]
    fn process_stream_f32_stereo() {
        let mut c = wap_config_default();
        c.pipeline_multi_channel_capture = true;
        let apm = wap_create_with_config(c);
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 2,
        };
        let src_l = [0.1f32; 160];
        let src_r = [0.2f32; 160];
        let src_ptrs: [*const f32; 2] = [src_l.as_ptr(), src_r.as_ptr()];
        let mut dest_l = [0.0f32; 160];
        let mut dest_r = [0.0f32; 160];
        let dest_ptrs: [*mut f32; 2] = [dest_l.as_mut_ptr(), dest_r.as_mut_ptr()];

        let err =
            wap_process_stream_f32(apm, src_ptrs.as_ptr(), config, config, dest_ptrs.as_ptr());
        assert_eq!(err, WapError::None);
        wap_destroy(apm);
    }

    #[test]
    fn process_stream_i16_silence_passthrough() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src = [0i16; 160];
        let mut dest = [100i16; 160];

        let err = wap_process_stream_i16(
            apm,
            src.as_ptr(),
            160,
            config,
            config,
            dest.as_mut_ptr(),
            160,
        );
        assert_eq!(err, WapError::None);

        for &s in &dest {
            assert_eq!(s, 0, "expected silence");
        }
        wap_destroy(apm);
    }

    #[test]
    fn process_stream_i16_null_returns_error() {
        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let err = wap_process_stream_i16(
            ptr::null_mut(),
            ptr::null(),
            0,
            config,
            config,
            ptr::null_mut(),
            0,
        );
        assert_eq!(err, WapError::NullPointer);
    }

    #[test]
    fn process_stream_i16_bad_length() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src = [0i16; 10]; // Too short for 160 frames.
        let mut dest = [0i16; 160];

        let err = wap_process_stream_i16(
            apm,
            src.as_ptr(),
            10,
            config,
            config,
            dest.as_mut_ptr(),
            160,
        );
        assert_eq!(err, WapError::BadDataLength);
        wap_destroy(apm);
    }

    #[test]
    fn process_reverse_stream_i16_works() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let src = [100i16; 160];
        let mut dest = [0i16; 160];

        let err = wap_process_reverse_stream_i16(
            apm,
            src.as_ptr(),
            160,
            config,
            config,
            dest.as_mut_ptr(),
            160,
        );
        assert_eq!(err, WapError::None);
        wap_destroy(apm);
    }

    // ─── Analog level tests ─────────────────────────────────────

    #[test]
    fn set_and_get_analog_level() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let err = wap_set_stream_analog_level(apm, 128);
        assert_eq!(err, WapError::None);
        // Recommended level depends on processing state; just check it doesn't crash.
        let _level = wap_recommended_stream_analog_level(apm);

        wap_destroy(apm);
    }

    #[test]
    fn analog_level_null_safety() {
        let err = wap_set_stream_analog_level(ptr::null_mut(), 0);
        assert_eq!(err, WapError::NullPointer);
        assert_eq!(wap_recommended_stream_analog_level(ptr::null()), 0);
    }

    // ─── Stream delay tests ──────────────────────────────────────

    #[test]
    fn set_and_get_delay() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let err = wap_set_stream_delay_ms(apm, 50);
        assert_eq!(err, WapError::None);
        assert_eq!(wap_stream_delay_ms(apm), 50);

        wap_destroy(apm);
    }

    #[test]
    fn delay_clamped_high() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let err = wap_set_stream_delay_ms(apm, 600);
        assert_eq!(err, WapError::BadStreamParameter);
        assert_eq!(wap_stream_delay_ms(apm), 500);

        wap_destroy(apm);
    }

    #[test]
    fn delay_clamped_negative() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let err = wap_set_stream_delay_ms(apm, -10);
        assert_eq!(err, WapError::BadStreamParameter);
        assert_eq!(wap_stream_delay_ms(apm), 0);

        wap_destroy(apm);
    }

    #[test]
    fn delay_null_safety() {
        let err = wap_set_stream_delay_ms(ptr::null_mut(), 0);
        assert_eq!(err, WapError::NullPointer);
        assert_eq!(wap_stream_delay_ms(ptr::null()), 0);
    }

    // ─── Runtime settings tests ──────────────────────────────────

    #[test]
    fn runtime_settings_null_safety() {
        assert_eq!(
            wap_set_capture_pre_gain(ptr::null_mut(), 1.0),
            WapError::NullPointer
        );
        assert_eq!(
            wap_set_capture_post_gain(ptr::null_mut(), 1.0),
            WapError::NullPointer
        );
        assert_eq!(
            wap_set_capture_fixed_post_gain(ptr::null_mut(), 0.0),
            WapError::NullPointer
        );
        assert_eq!(
            wap_set_playout_volume(ptr::null_mut(), 0),
            WapError::NullPointer
        );
        assert_eq!(
            wap_set_playout_audio_device(ptr::null_mut(), 0, 0),
            WapError::NullPointer
        );
        assert_eq!(
            wap_set_capture_output_used(ptr::null_mut(), true),
            WapError::NullPointer
        );
    }

    #[test]
    fn runtime_settings_succeed() {
        let apm = wap_create();
        assert!(!apm.is_null());

        assert_eq!(wap_set_capture_pre_gain(apm, 2.0), WapError::None);
        assert_eq!(wap_set_capture_post_gain(apm, 1.5), WapError::None);
        assert_eq!(wap_set_capture_fixed_post_gain(apm, 10.0), WapError::None);
        assert_eq!(wap_set_playout_volume(apm, 128), WapError::None);
        assert_eq!(wap_set_playout_audio_device(apm, 1, 255), WapError::None);
        assert_eq!(wap_set_capture_output_used(apm, false), WapError::None);

        wap_destroy(apm);
    }

    // ─── Statistics tests ────────────────────────────────────────

    #[test]
    fn get_statistics_null_safety() {
        assert_eq!(
            wap_get_statistics(ptr::null(), ptr::null_mut()),
            WapError::NullPointer
        );
        let apm = wap_create();
        assert_eq!(
            wap_get_statistics(apm, ptr::null_mut()),
            WapError::NullPointer
        );
        wap_destroy(apm);
    }

    #[test]
    fn get_statistics_default() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let mut stats = WapStats {
            has_echo_return_loss: true,
            echo_return_loss: 999.0,
            has_echo_return_loss_enhancement: false,
            echo_return_loss_enhancement: 0.0,
            has_divergent_filter_fraction: false,
            divergent_filter_fraction: 0.0,
            has_delay_median_ms: false,
            delay_median_ms: 0,
            has_delay_standard_deviation_ms: false,
            delay_standard_deviation_ms: 0,
            has_residual_echo_likelihood: false,
            residual_echo_likelihood: 0.0,
            has_residual_echo_likelihood_recent_max: false,
            residual_echo_likelihood_recent_max: 0.0,
            has_delay_ms: false,
            delay_ms: 0,
        };
        let err = wap_get_statistics(apm, &mut stats);
        assert_eq!(err, WapError::None);
        // With no processing done, no stats should be available.
        assert!(!stats.has_echo_return_loss);
        assert!(!stats.has_echo_return_loss_enhancement);

        wap_destroy(apm);
    }

    // ─── Processing error tests (continued) ──────────────────────

    #[test]
    fn process_stream_f32_bad_rate() {
        let apm = wap_create();
        assert!(!apm.is_null());

        let config = WapStreamConfig {
            sample_rate_hz: 100, // Bad rate.
            num_channels: 1,
        };
        let src_data = [0.0f32; 1];
        let src_ptrs: [*const f32; 1] = [src_data.as_ptr()];
        let mut dest_data = [0.0f32; 1];
        let dest_ptrs: [*mut f32; 1] = [dest_data.as_mut_ptr()];

        let err =
            wap_process_stream_f32(apm, src_ptrs.as_ptr(), config, config, dest_ptrs.as_ptr());
        assert_eq!(err, WapError::BadSampleRate);
        wap_destroy(apm);
    }

    // ─── End-to-end integration tests ────────────────────────────

    /// Helper: generates a simple sine tone at the given frequency.
    fn generate_sine(num_frames: usize, freq_hz: f32, sample_rate: f32) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..num_frames)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin() * 0.5)
            .collect()
    }

    #[test]
    fn e2e_echo_cancellation_pipeline() {
        // Run the full AEC3 pipeline: feed render, then process capture.
        let mut config = wap_config_default();
        config.echo_canceller_enabled = true;
        config.high_pass_filter_enabled = true;
        let apm = wap_create_with_config(config);
        assert!(!apm.is_null());

        let stream_config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let num_frames = 160;

        // Simulate 50 frames of echo cancellation.
        for i in 0..50 {
            // Render (far-end) signal: sine tone.
            let render = generate_sine(num_frames, 440.0, 16000.0);
            let render_ptrs: [*const f32; 1] = [render.as_ptr()];
            let mut render_out = vec![0.0f32; num_frames];
            let render_out_ptrs: [*mut f32; 1] = [render_out.as_mut_ptr()];

            let err = wap_process_reverse_stream_f32(
                apm,
                render_ptrs.as_ptr(),
                stream_config,
                stream_config,
                render_out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "render frame {i}");

            // Set stream delay before capture processing.
            let err = wap_set_stream_delay_ms(apm, 10);
            assert_eq!(err, WapError::None);

            // Capture (near-end) signal: same sine (simulating echo).
            let capture = generate_sine(num_frames, 440.0, 16000.0);
            let capture_ptrs: [*const f32; 1] = [capture.as_ptr()];
            let mut capture_out = vec![0.0f32; num_frames];
            let capture_out_ptrs: [*mut f32; 1] = [capture_out.as_mut_ptr()];

            let err = wap_process_stream_f32(
                apm,
                capture_ptrs.as_ptr(),
                stream_config,
                stream_config,
                capture_out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "capture frame {i}");
        }

        // After convergence, check that stats are populated.
        let mut stats = WapStats {
            has_echo_return_loss: false,
            echo_return_loss: 0.0,
            has_echo_return_loss_enhancement: false,
            echo_return_loss_enhancement: 0.0,
            has_divergent_filter_fraction: false,
            divergent_filter_fraction: 0.0,
            has_delay_median_ms: false,
            delay_median_ms: 0,
            has_delay_standard_deviation_ms: false,
            delay_standard_deviation_ms: 0,
            has_residual_echo_likelihood: false,
            residual_echo_likelihood: 0.0,
            has_residual_echo_likelihood_recent_max: false,
            residual_echo_likelihood_recent_max: 0.0,
            has_delay_ms: false,
            delay_ms: 0,
        };
        let err = wap_get_statistics(apm, &mut stats);
        assert_eq!(err, WapError::None);
        // delay_ms should be populated after AEC processing.
        assert!(
            stats.has_delay_ms,
            "expected delay_ms stat after AEC3 processing"
        );

        wap_destroy(apm);
    }

    #[test]
    fn e2e_all_components_enabled() {
        // Enable every component and process audio without errors.
        let mut config = wap_config_default();
        config.high_pass_filter_enabled = true;
        config.echo_canceller_enabled = true;
        config.noise_suppression_enabled = true;
        config.noise_suppression_level = WapNoiseSuppressionLevel::High;
        config.gain_controller2_enabled = true;
        config.gain_controller2_adaptive_digital_enabled = true;
        config.capture_level_adjustment_enabled = true;
        config.capture_level_adjustment_pre_gain_factor = 1.0;
        config.capture_level_adjustment_post_gain_factor = 1.0;

        let apm = wap_create_with_config(config);
        assert!(!apm.is_null());

        let stream_config = WapStreamConfig {
            sample_rate_hz: 48000,
            num_channels: 1,
        };
        let num_frames = 480; // 10ms at 48kHz

        for i in 0..100 {
            // Feed render.
            let render = generate_sine(num_frames, 300.0, 48000.0);
            let render_ptrs: [*const f32; 1] = [render.as_ptr()];
            let mut render_out = vec![0.0f32; num_frames];
            let render_out_ptrs: [*mut f32; 1] = [render_out.as_mut_ptr()];

            let err = wap_process_reverse_stream_f32(
                apm,
                render_ptrs.as_ptr(),
                stream_config,
                stream_config,
                render_out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "render frame {i}");

            let err = wap_set_stream_delay_ms(apm, 20);
            assert_eq!(err, WapError::None);

            // Process capture with some noise-like content.
            let capture: Vec<f32> = (0..num_frames)
                .map(|j| {
                    let t = (i * num_frames + j) as f32 / 48000.0;
                    (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.3
                        + (2.0 * std::f32::consts::PI * 1500.0 * t).sin() * 0.1
                })
                .collect();
            let capture_ptrs: [*const f32; 1] = [capture.as_ptr()];
            let mut capture_out = vec![0.0f32; num_frames];
            let capture_out_ptrs: [*mut f32; 1] = [capture_out.as_mut_ptr()];

            let err = wap_process_stream_f32(
                apm,
                capture_ptrs.as_ptr(),
                stream_config,
                stream_config,
                capture_out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "capture frame {i}");

            // Output should be finite.
            for (j, &s) in capture_out.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "non-finite sample at frame {i} sample {j}: {s}"
                );
            }
        }

        wap_destroy(apm);
    }

    #[test]
    fn e2e_config_change_mid_stream() {
        // Start with no processing, then enable components mid-stream.
        let apm = wap_create();
        assert!(!apm.is_null());

        let stream_config = WapStreamConfig {
            sample_rate_hz: 16000,
            num_channels: 1,
        };
        let num_frames = 160;
        let src_data = generate_sine(num_frames, 440.0, 16000.0);
        let src_ptrs: [*const f32; 1] = [src_data.as_ptr()];

        // Process 10 frames with default config (all disabled).
        for i in 0..10 {
            let mut dest = vec![0.0f32; num_frames];
            let dest_ptrs: [*mut f32; 1] = [dest.as_mut_ptr()];
            let err = wap_process_stream_f32(
                apm,
                src_ptrs.as_ptr(),
                stream_config,
                stream_config,
                dest_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "pre-config frame {i}");
            // With no processing, output should equal input.
            for j in 0..num_frames {
                assert_eq!(
                    src_data[j], dest[j],
                    "passthrough mismatch at frame {i} sample {j}"
                );
            }
        }

        // Enable HPF + NS mid-stream.
        let mut config = wap_config_default();
        config.high_pass_filter_enabled = true;
        config.noise_suppression_enabled = true;
        config.noise_suppression_level = WapNoiseSuppressionLevel::VeryHigh;
        let err = wap_apply_config(apm, config);
        assert_eq!(err, WapError::None);

        // Verify config took effect.
        let mut readback = wap_config_default();
        let err = wap_get_config(apm, &mut readback);
        assert_eq!(err, WapError::None);
        assert!(readback.high_pass_filter_enabled);
        assert!(readback.noise_suppression_enabled);
        assert_eq!(
            readback.noise_suppression_level,
            WapNoiseSuppressionLevel::VeryHigh
        );

        // Process 20 more frames with NS enabled.
        for i in 0..20 {
            let mut dest = vec![0.0f32; num_frames];
            let dest_ptrs: [*mut f32; 1] = [dest.as_mut_ptr()];
            let err = wap_process_stream_f32(
                apm,
                src_ptrs.as_ptr(),
                stream_config,
                stream_config,
                dest_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "post-config frame {i}");
            // Output should still be finite.
            for &s in &dest {
                assert!(s.is_finite());
            }
        }

        // Now disable NS and enable AGC2.
        config.noise_suppression_enabled = false;
        config.gain_controller2_enabled = true;
        config.gain_controller2_adaptive_digital_enabled = true;
        let err = wap_apply_config(apm, config);
        assert_eq!(err, WapError::None);

        // Process 10 more frames.
        for i in 0..10 {
            let mut dest = vec![0.0f32; num_frames];
            let dest_ptrs: [*mut f32; 1] = [dest.as_mut_ptr()];
            let err = wap_process_stream_f32(
                apm,
                src_ptrs.as_ptr(),
                stream_config,
                stream_config,
                dest_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "agc2 frame {i}");
        }

        wap_destroy(apm);
    }

    #[test]
    fn e2e_i16_pipeline() {
        // End-to-end test of the int16 interleaved path at multiple rates.
        let mut config = wap_config_default();
        config.high_pass_filter_enabled = true;
        config.noise_suppression_enabled = true;
        let apm = wap_create_with_config(config);
        assert!(!apm.is_null());

        for &rate in &[8000i32, 16000, 32000, 48000] {
            let stream_config = WapStreamConfig {
                sample_rate_hz: rate,
                num_channels: 1,
            };
            let num_frames = (rate / 100) as usize;

            for i in 0..20 {
                // Generate simple i16 samples.
                let src: Vec<i16> = (0..num_frames)
                    .map(|j| {
                        let t = (i * num_frames + j) as f32 / rate as f32;
                        (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i16
                    })
                    .collect();
                let mut dest = vec![0i16; num_frames];

                let err = wap_process_stream_i16(
                    apm,
                    src.as_ptr(),
                    num_frames as i32,
                    stream_config,
                    stream_config,
                    dest.as_mut_ptr(),
                    num_frames as i32,
                );
                assert_eq!(err, WapError::None, "rate={rate} frame={i}");
            }
        }

        wap_destroy(apm);
    }

    #[test]
    fn e2e_stereo_capture_and_render() {
        // Test stereo processing with multi-channel enabled.
        let mut config = wap_config_default();
        config.pipeline_multi_channel_capture = true;
        config.pipeline_multi_channel_render = true;
        config.echo_canceller_enabled = true;
        config.noise_suppression_enabled = true;
        let apm = wap_create_with_config(config);
        assert!(!apm.is_null());

        let stream_config = WapStreamConfig {
            sample_rate_hz: 48000,
            num_channels: 2,
        };
        let num_frames = 480;

        for i in 0..30 {
            // Stereo render.
            let render_l = generate_sine(num_frames, 300.0, 48000.0);
            let render_r = generate_sine(num_frames, 500.0, 48000.0);
            let render_ptrs: [*const f32; 2] = [render_l.as_ptr(), render_r.as_ptr()];
            let mut out_l = vec![0.0f32; num_frames];
            let mut out_r = vec![0.0f32; num_frames];
            let out_ptrs: [*mut f32; 2] = [out_l.as_mut_ptr(), out_r.as_mut_ptr()];

            let err = wap_process_reverse_stream_f32(
                apm,
                render_ptrs.as_ptr(),
                stream_config,
                stream_config,
                out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "render frame {i}");

            let err = wap_set_stream_delay_ms(apm, 15);
            assert_eq!(err, WapError::None);

            // Stereo capture.
            let cap_l = generate_sine(num_frames, 300.0, 48000.0);
            let cap_r = generate_sine(num_frames, 500.0, 48000.0);
            let cap_ptrs: [*const f32; 2] = [cap_l.as_ptr(), cap_r.as_ptr()];
            let mut cap_out_l = vec![0.0f32; num_frames];
            let mut cap_out_r = vec![0.0f32; num_frames];
            let cap_out_ptrs: [*mut f32; 2] = [cap_out_l.as_mut_ptr(), cap_out_r.as_mut_ptr()];

            let err = wap_process_stream_f32(
                apm,
                cap_ptrs.as_ptr(),
                stream_config,
                stream_config,
                cap_out_ptrs.as_ptr(),
            );
            assert_eq!(err, WapError::None, "capture frame {i}");

            // Both channels should have finite output.
            for &s in cap_out_l.iter().chain(cap_out_r.iter()) {
                assert!(s.is_finite(), "non-finite output at frame {i}");
            }
        }

        wap_destroy(apm);
    }
}
