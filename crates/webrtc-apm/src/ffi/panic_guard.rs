//! Panic guard macros for FFI boundary safety.
//!
//! Every `extern "C"` function must catch panics to prevent undefined
//! behaviour when unwinding crosses the FFI boundary.

/// Wrap an FFI function body that returns [`WapError`](super::types::WapError).
///
/// On panic the macro returns `WapError::Internal`.
///
/// # Example
///
/// ```ignore
/// extern "C" fn wap_do_thing(ptr: *mut WapAudioProcessing) -> WapError {
///     ffi_guard! {
///         // ... body ...
///         WapError::None
///     }
/// }
/// ```
macro_rules! ffi_guard {
    ($($body:tt)*) => {{
        use std::panic;
        use std::panic::AssertUnwindSafe;

        match panic::catch_unwind(AssertUnwindSafe(move || { $($body)* })) {
            Ok(result) => result,
            Err(_) => $crate::ffi::types::WapError::Internal,
        }
    }};
}

/// Wrap an FFI function body that returns a pointer (or pointer-like value).
///
/// On panic the macro returns [`std::ptr::null_mut()`].
///
/// # Example
///
/// ```ignore
/// extern "C" fn wap_create() -> *mut WapAudioProcessing {
///     ffi_guard_ptr! {
///         // ... body ...
///         Box::into_raw(boxed)
///     }
/// }
/// ```
macro_rules! ffi_guard_ptr {
    ($($body:tt)*) => {{
        use std::panic;
        use std::panic::AssertUnwindSafe;
        use std::ptr;

        match panic::catch_unwind(AssertUnwindSafe(move || { $($body)* })) {
            Ok(result) => result,
            Err(_) => ptr::null_mut(),
        }
    }};
}

pub(crate) use ffi_guard;
pub(crate) use ffi_guard_ptr;

#[cfg(test)]
mod tests {
    use crate::ffi::types::WapError;

    #[test]
    fn ffi_guard_returns_value_on_success() {
        let result: WapError = ffi_guard! { WapError::None };
        assert_eq!(result, WapError::None);
    }

    #[test]
    fn ffi_guard_returns_internal_on_panic() {
        let result: WapError = ffi_guard! {
            panic!("test panic");
        };
        assert_eq!(result, WapError::Internal);
    }

    #[test]
    fn ffi_guard_ptr_returns_pointer_on_success() {
        let mut value = 42i32;
        let ptr: *mut i32 = ffi_guard_ptr! { &raw mut value };
        assert!(!ptr.is_null());
    }

    #[test]
    fn ffi_guard_ptr_returns_null_on_panic() {
        let ptr: *mut i32 = ffi_guard_ptr! {
            panic!("test panic");
        };
        assert!(ptr.is_null());
    }
}
