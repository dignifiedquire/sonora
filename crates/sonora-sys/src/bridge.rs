//! cxx bridge module for C++ interop.

#[cxx::bridge(namespace = "webrtc_shim")]
mod ffi {
    unsafe extern "C++" {
        include!("sonora-sys/cpp/shim.h");

        // ── Full pipeline ────────────────────────────────────────────────────

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
        fn process_stream_f32_2ch(
            handle: Pin<&mut ApmHandle>,
            src_l: &[f32],
            src_r: &[f32],
            sample_rate: i32,
            dest_l: &mut [f32],
            dest_r: &mut [f32],
        ) -> i32;

        /// Process a single 10ms reverse stream frame of f32 audio (mono).
        fn process_reverse_stream_f32(
            handle: Pin<&mut ApmHandle>,
            src: &[f32],
            input_sample_rate: i32,
            input_channels: usize,
            output_sample_rate: i32,
            output_channels: usize,
            dest: &mut [f32],
        ) -> i32;

        // ── Per-component: ThreeBandFilterBank ───────────────────────────────

        type FilterBankHandle;

        /// Create a new ThreeBandFilterBank instance.
        fn create_filter_bank() -> UniquePtr<FilterBankHandle>;

        /// Analysis: split 480-sample fullband into 3×160-sample bands (packed).
        fn filter_bank_analysis(handle: Pin<&mut FilterBankHandle>, src: &[f32], dest: &mut [f32]);

        /// Synthesis: merge 3×160-sample bands (packed) into 480-sample fullband.
        fn filter_bank_synthesis(handle: Pin<&mut FilterBankHandle>, src: &[f32], dest: &mut [f32]);

        // ── Per-component: HighPassFilter ────────────────────────────────────

        type HpfHandle;

        /// Create a new HighPassFilter instance.
        fn create_hpf(sample_rate_hz: i32, num_channels: usize) -> UniquePtr<HpfHandle>;

        /// Process a single channel in-place.
        fn hpf_process(handle: Pin<&mut HpfHandle>, ch0: &mut [f32]);

        // ── Per-component: NoiseSuppressor ───────────────────────────────────

        type NsHandle;

        /// Create a new NoiseSuppressor instance.
        fn create_ns(level: u8, sample_rate_hz: i32, num_channels: usize) -> UniquePtr<NsHandle>;

        /// Analyze band0 (160 floats). Must be called before ns_process.
        fn ns_analyze(handle: Pin<&mut NsHandle>, band0: &[f32]);

        /// Process band0 in-place (160 floats).
        fn ns_process(handle: Pin<&mut NsHandle>, band0: &mut [f32]);
    }
}

pub use ffi::*;
