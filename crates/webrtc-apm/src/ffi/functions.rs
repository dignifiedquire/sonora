//! Exported `extern "C"` functions for the audio processing C API.
//!
//! # Symbol prefix
//!
//! All public symbols use the `wap_` prefix.

use std::ptr;
use std::slice;

use crate::AudioProcessing;
use crate::config::Config;
use crate::stream_config::StreamConfig;

use super::panic_guard::{ffi_guard, ffi_guard_ptr};
use super::types::{WapAudioProcessing, WapConfig, WapError, WapStreamConfig};

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
    if data.is_null() || num_channels == 0 || num_channels > MAX_CHANNELS {
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
    if data.is_null() || num_channels == 0 || num_channels > MAX_CHANNELS {
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
}
