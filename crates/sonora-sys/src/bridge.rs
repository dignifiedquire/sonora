//! cxx bridge module for C++ interop.

#[cxx::bridge(namespace = "webrtc_shim")]
mod ffi {
    unsafe extern "C++" {
        include!("sonora-sys/cpp/shim.h");

        type ApmHandle;

        /// Create a new AudioProcessing instance with default config.
        fn create_apm() -> UniquePtr<ApmHandle>;

        /// Apply audio processing config.
        fn apply_config(
            handle: Pin<&mut ApmHandle>,
            ec_enabled: bool,
            ns_enabled: bool,
            ns_level: u8,
            agc2_enabled: bool,
            hpf_enabled: bool,
        );

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

        /// Process a single 10ms frame of f32 audio (mono).
        /// Returns 0 on success, negative error code on failure.
        fn process_stream_f32(
            handle: Pin<&mut ApmHandle>,
            src: &[f32],
            input_sample_rate: i32,
            input_channels: usize,
            output_sample_rate: i32,
            output_channels: usize,
            dest: &mut [f32],
        ) -> i32;

        /// Process a single 10ms frame of f32 audio (stereo, separate L/R channels).
        /// Returns 0 on success, negative error code on failure.
        fn process_stream_f32_2ch(
            handle: Pin<&mut ApmHandle>,
            src_l: &[f32],
            src_r: &[f32],
            sample_rate: i32,
            dest_l: &mut [f32],
            dest_r: &mut [f32],
        ) -> i32;

        /// Process a single 10ms reverse stream frame of f32 audio (mono).
        /// Returns 0 on success, negative error code on failure.
        fn process_reverse_stream_f32(
            handle: Pin<&mut ApmHandle>,
            src: &[f32],
            input_sample_rate: i32,
            input_channels: usize,
            output_sample_rate: i32,
            output_channels: usize,
            dest: &mut [f32],
        ) -> i32;
    }
}

pub use ffi::*;
