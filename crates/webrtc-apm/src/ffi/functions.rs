//! Exported `extern "C"` functions for the audio processing C API.
//!
//! # Symbol prefix
//!
//! All public symbols use the `wap_` prefix.

use std::ptr;

use crate::AudioProcessing;
use crate::config::Config;

use super::panic_guard::{ffi_guard, ffi_guard_ptr};
use super::types::{WapAudioProcessing, WapConfig, WapError};

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
}
