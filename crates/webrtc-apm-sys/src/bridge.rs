//! cxx bridge module for C++ interop.

#[cxx::bridge(namespace = "webrtc_shim")]
mod ffi {
    unsafe extern "C++" {
        include!("webrtc-apm-sys/cpp/shim.h");

        type ApmHandle;

        /// Create a new AudioProcessing instance with default config.
        fn create_apm() -> UniquePtr<ApmHandle>;

        /// Process a single 10ms frame of interleaved i16 audio.
        /// Returns 0 on success, negative error code on failure.
        fn process_stream_i16(
            handle: Pin<&mut ApmHandle>,
            src: &[i16],
            input_sample_rate: i32,
            input_channels: usize,
            output_sample_rate: i32,
            output_channels: usize,
            dest: &mut [i16],
        ) -> i32;
    }
}

pub use ffi::*;
