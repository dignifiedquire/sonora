//! Public audio processing API.
//!
//! Provides the user-facing [`AudioProcessing`] struct and
//! [`AudioProcessingBuilder`] for constructing configured instances.
//!
//! Ported from `AudioProcessing` / `AudioProcessingBuilderInterface` in
//! `api/audio/audio_processing.h`.

use crate::audio_processing_impl::AudioProcessingImpl;
use crate::config::{Config, RuntimeSetting};
use crate::stats::AudioProcessingStats;
use crate::stream_config::StreamConfig;

// ─── Error ───────────────────────────────────────────────────────────

/// Errors returned by audio processing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Bad sample rate (too low, too high, or negative).
    BadSampleRate,
    /// Bad number of channels (zero, or output channels don't match input).
    BadNumberChannels,
    /// A stream parameter (e.g. delay) was out of range and was clamped.
    BadStreamParameter,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadSampleRate => write!(f, "bad sample rate"),
            Self::BadNumberChannels => write!(f, "bad number of channels"),
            Self::BadStreamParameter => write!(f, "bad stream parameter (clamped)"),
        }
    }
}

impl std::error::Error for Error {}

// ─── Format validation ──────────────────────────────────────────────

/// Maximum supported sample rate.
const MAX_SAMPLE_RATE: usize = 384_000;

/// Minimum supported sample rate.
const MIN_SAMPLE_RATE: usize = 8_000;

/// Supported native sample rates for int16 processing.
const NATIVE_SAMPLE_RATES: [usize; 4] = [8000, 16000, 32000, 48000];

fn validate_stream_config(config: &StreamConfig) -> Result<(), Error> {
    if config.num_channels() == 0 {
        return Err(Error::BadNumberChannels);
    }
    let rate = config.sample_rate_hz();
    if rate < MIN_SAMPLE_RATE || rate > MAX_SAMPLE_RATE {
        return Err(Error::BadSampleRate);
    }
    Ok(())
}

fn validate_float_configs(
    input_config: &StreamConfig,
    output_config: &StreamConfig,
) -> Result<(), Error> {
    validate_stream_config(input_config)?;
    validate_stream_config(output_config)?;
    // Output must have 1 channel or the same number as input.
    let out_ch = output_config.num_channels();
    let in_ch = input_config.num_channels();
    if out_ch != 1 && out_ch != in_ch {
        return Err(Error::BadNumberChannels);
    }
    Ok(())
}

fn validate_i16_configs(
    input_config: &StreamConfig,
    output_config: &StreamConfig,
) -> Result<(), Error> {
    validate_stream_config(input_config)?;
    validate_stream_config(output_config)?;
    // int16 requires native rates.
    if !NATIVE_SAMPLE_RATES.contains(&input_config.sample_rate_hz()) {
        return Err(Error::BadSampleRate);
    }
    if !NATIVE_SAMPLE_RATES.contains(&output_config.sample_rate_hz()) {
        return Err(Error::BadSampleRate);
    }
    // int16 requires matching input/output rates and channels.
    if input_config.sample_rate_hz() != output_config.sample_rate_hz() {
        return Err(Error::BadSampleRate);
    }
    if input_config.num_channels() != output_config.num_channels() {
        return Err(Error::BadNumberChannels);
    }
    Ok(())
}

// ─── AudioProcessingBuilder ─────────────────────────────────────────

/// Builder for constructing an [`AudioProcessing`] instance.
///
/// # Example
/// ```ignore
/// use webrtc_apm::{AudioProcessing, Config};
///
/// let mut config = Config::default();
/// config.echo_canceller.enabled = true;
/// config.noise_suppression.enabled = true;
///
/// let apm = AudioProcessing::builder()
///     .config(config)
///     .build();
/// ```
pub struct AudioProcessingBuilder {
    config: Config,
}

impl AudioProcessingBuilder {
    fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set the initial configuration.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Build the [`AudioProcessing`] instance.
    pub fn build(self) -> AudioProcessing {
        AudioProcessing {
            inner: AudioProcessingImpl::with_config(self.config),
            stream_delay_ms: 0,
            was_stream_delay_set: false,
        }
    }
}

// ─── AudioProcessing ────────────────────────────────────────────────

/// Audio processing engine providing echo cancellation, noise suppression,
/// automatic gain control, and other audio processing capabilities.
///
/// # Usage
///
/// 1. Create an instance via [`AudioProcessing::builder()`] or
///    [`AudioProcessing::new()`].
/// 2. For each audio frame (~10 ms):
///    - Call [`process_reverse_stream_f32()`](AudioProcessing::process_reverse_stream_f32)
///      with the far-end (render/playback) audio.
///    - Call [`process_stream_f32()`](AudioProcessing::process_stream_f32)
///      with the near-end (capture/microphone) audio.
/// 3. Apply configuration changes via [`apply_config()`](AudioProcessing::apply_config).
///
/// Both f32 (deinterleaved) and i16 (interleaved) interfaces are provided.
pub struct AudioProcessing {
    inner: AudioProcessingImpl,
    stream_delay_ms: i32,
    was_stream_delay_set: bool,
}

impl AudioProcessing {
    /// Creates a new instance with default configuration.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder for constructing an instance with custom configuration.
    pub fn builder() -> AudioProcessingBuilder {
        AudioProcessingBuilder::new()
    }

    /// Applies a new configuration, selectively reinitializing submodules
    /// as needed.
    pub fn apply_config(&mut self, config: Config) {
        self.inner.apply_config(config);
    }

    /// Enqueues a runtime setting for the capture path.
    ///
    /// Runtime settings are applied at the next [`process_stream_f32()`]
    /// or [`process_stream_i16()`] call.
    pub fn set_runtime_setting(&mut self, setting: RuntimeSetting) {
        self.inner.set_runtime_setting(setting);
    }

    /// Returns current processing statistics.
    pub fn get_statistics(&self) -> AudioProcessingStats {
        self.inner.get_statistics()
    }

    /// Returns the last applied configuration.
    pub fn get_config(&self) -> &Config {
        self.inner.config()
    }

    // ─── Analog level (for AGC) ──────────────────────────────────

    /// Sets the applied input volume (e.g. from the OS mixer).
    ///
    /// Must be called before [`process_stream_f32()`] if the input volume
    /// controller is enabled. Value must be in range `[0, 255]`.
    pub fn set_stream_analog_level(&mut self, level: i32) {
        self.inner.set_applied_input_volume(level);
    }

    /// Returns the recommended analog level from AGC.
    ///
    /// Should be called after [`process_stream_f32()`] to obtain the
    /// recommended new analog level for the audio HAL.
    pub fn recommended_stream_analog_level(&self) -> i32 {
        self.inner.recommended_input_volume().unwrap_or(0)
    }

    // ─── Stream delay ────────────────────────────────────────────

    /// Sets the delay in ms between render and capture.
    ///
    /// The delay is clamped to `[0, 500]`. Returns `Err(BadStreamParameter)`
    /// if clamping was necessary (processing still proceeds).
    pub fn set_stream_delay_ms(&mut self, delay: i32) -> Result<(), Error> {
        self.was_stream_delay_set = true;
        let mut clamped = delay;
        let mut warning = false;
        if clamped < 0 {
            clamped = 0;
            warning = true;
        }
        if clamped > 500 {
            clamped = 500;
            warning = true;
        }
        self.stream_delay_ms = clamped;
        if warning {
            Err(Error::BadStreamParameter)
        } else {
            Ok(())
        }
    }

    /// Returns the current stream delay in ms.
    pub fn stream_delay_ms(&self) -> i32 {
        self.stream_delay_ms
    }

    // ─── Processing rate info ────────────────────────────────────

    /// The internal capture processing sample rate.
    pub fn proc_sample_rate_hz(&self) -> usize {
        self.inner.proc_sample_rate_hz()
    }

    // ─── Float (deinterleaved) processing ────────────────────────

    /// Processes a capture audio frame (float, deinterleaved).
    ///
    /// Each element of `src` / `dest` points to a channel buffer with
    /// `input_config.num_frames()` / `output_config.num_frames()` samples.
    /// Values should be in the range `[-1.0, 1.0]`.
    ///
    /// The output must have 1 channel or the same number as the input.
    pub fn process_stream_f32(
        &mut self,
        src: &[&[f32]],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [&mut [f32]],
    ) -> Result<(), Error> {
        validate_float_configs(input_config, output_config)?;
        self.inner
            .process_stream(src, input_config, output_config, dest);
        Ok(())
    }

    /// Processes a reverse (render / far-end) audio frame (float, deinterleaved).
    ///
    /// Each element of `src` / `dest` points to a channel buffer.
    pub fn process_reverse_stream_f32(
        &mut self,
        src: &[&[f32]],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [&mut [f32]],
    ) -> Result<(), Error> {
        validate_float_configs(input_config, output_config)?;
        self.inner
            .process_reverse_stream(src, input_config, output_config, dest);
        Ok(())
    }

    // ─── Int16 (interleaved) processing ──────────────────────────

    /// Processes a capture audio frame (int16, interleaved).
    ///
    /// Requires native sample rates (8k, 16k, 32k, 48k) and matching
    /// input/output rates and channel counts.
    pub fn process_stream_i16(
        &mut self,
        src: &[i16],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [i16],
    ) -> Result<(), Error> {
        validate_i16_configs(input_config, output_config)?;
        self.inner
            .process_stream_i16(src, input_config, output_config, dest);
        Ok(())
    }

    /// Processes a reverse (render / far-end) audio frame (int16, interleaved).
    ///
    /// Requires native sample rates (8k, 16k, 32k, 48k) and matching
    /// input/output rates and channel counts.
    pub fn process_reverse_stream_i16(
        &mut self,
        src: &[i16],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [i16],
    ) -> Result<(), Error> {
        validate_i16_configs(input_config, output_config)?;
        self.inner
            .process_reverse_stream_i16(src, input_config, output_config, dest);
        Ok(())
    }
}

impl Default for AudioProcessing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_creates_default_instance() {
        let apm = AudioProcessing::builder().build();
        assert_eq!(apm.stream_delay_ms(), 0);
        assert_eq!(apm.recommended_stream_analog_level(), 0);
    }

    #[test]
    fn builder_with_config() {
        let mut config = Config::default();
        config.echo_canceller.enabled = true;
        config.noise_suppression.enabled = true;
        let _apm = AudioProcessing::builder().config(config).build();
    }

    #[test]
    fn set_stream_delay_clamps_negative() {
        let mut apm = AudioProcessing::new();
        let result = apm.set_stream_delay_ms(-10);
        assert_eq!(result, Err(Error::BadStreamParameter));
        assert_eq!(apm.stream_delay_ms(), 0);
    }

    #[test]
    fn set_stream_delay_clamps_high() {
        let mut apm = AudioProcessing::new();
        let result = apm.set_stream_delay_ms(600);
        assert_eq!(result, Err(Error::BadStreamParameter));
        assert_eq!(apm.stream_delay_ms(), 500);
    }

    #[test]
    fn set_stream_delay_valid_range() {
        let mut apm = AudioProcessing::new();
        assert!(apm.set_stream_delay_ms(50).is_ok());
        assert_eq!(apm.stream_delay_ms(), 50);
    }

    #[test]
    fn process_stream_f32_validates_bad_rate() {
        let mut apm = AudioProcessing::new();
        let input_config = StreamConfig::new(100, 1); // Bad rate
        let output_config = StreamConfig::new(16000, 1);
        let src_data = [0.0f32; 1];
        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = [0.0f32; 160];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
        let result = apm.process_stream_f32(src, &input_config, &output_config, dest);
        assert_eq!(result, Err(Error::BadSampleRate));
    }

    #[test]
    fn process_stream_f32_validates_bad_channels() {
        let mut apm = AudioProcessing::new();
        let input_config = StreamConfig::new(16000, 0); // Bad channels
        let output_config = StreamConfig::new(16000, 1);
        let src: &[&[f32]] = &[];
        let mut dest_data = [0.0f32; 160];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
        let result = apm.process_stream_f32(src, &input_config, &output_config, dest);
        assert_eq!(result, Err(Error::BadNumberChannels));
    }

    #[test]
    fn process_stream_f32_validates_channel_mismatch() {
        let mut apm = AudioProcessing::new();
        let input_config = StreamConfig::new(16000, 2);
        let output_config = StreamConfig::new(16000, 3); // Not 1 and not 2
        let src_data = [0.0f32; 160];
        let src: &[&[f32]] = &[&src_data, &src_data];
        let mut dest0 = [0.0f32; 160];
        let mut dest1 = [0.0f32; 160];
        let mut dest2 = [0.0f32; 160];
        let dest: &mut [&mut [f32]] = &mut [&mut dest0, &mut dest1, &mut dest2];
        let result = apm.process_stream_f32(src, &input_config, &output_config, dest);
        assert_eq!(result, Err(Error::BadNumberChannels));
    }

    #[test]
    fn process_stream_i16_validates_non_native_rate() {
        let mut apm = AudioProcessing::new();
        let input_config = StreamConfig::new(44100, 1); // Not a native rate
        let output_config = StreamConfig::new(44100, 1);
        let src = [0i16; 441];
        let mut dest = [0i16; 441];
        let result = apm.process_stream_i16(&src, &input_config, &output_config, &mut dest);
        assert_eq!(result, Err(Error::BadSampleRate));
    }

    #[test]
    fn process_stream_f32_silence_passthrough() {
        let mut apm = AudioProcessing::new();
        let config = StreamConfig::new(16000, 1);
        let src_data = [0.0f32; 160];
        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = [1.0f32; 160]; // Fill with non-zero to verify overwrite
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
        let result = apm.process_stream_f32(src, &config, &config, dest);
        assert!(result.is_ok());
        // With all defaults (no processing enabled), silence in → silence out.
        for &sample in dest[0].iter() {
            assert!(sample.abs() < 1e-6, "expected silence, got {sample}",);
        }
    }

    #[test]
    fn process_stream_i16_silence_passthrough() {
        let mut apm = AudioProcessing::new();
        let config = StreamConfig::new(16000, 1);
        let src = [0i16; 160];
        let mut dest = [100i16; 160]; // Fill with non-zero to verify overwrite
        let result = apm.process_stream_i16(&src, &config, &config, &mut dest);
        assert!(result.is_ok());
        for &sample in dest.iter() {
            assert_eq!(sample, 0, "expected silence");
        }
    }

    #[test]
    fn process_reverse_stream_f32_passthrough() {
        let mut apm = AudioProcessing::new();
        let config = StreamConfig::new(16000, 1);
        let src_data = [0.5f32; 160];
        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = [0.0f32; 160];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
        let result = apm.process_reverse_stream_f32(src, &config, &config, dest);
        assert!(result.is_ok());
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", Error::BadSampleRate), "bad sample rate");
        assert_eq!(
            format!("{}", Error::BadNumberChannels),
            "bad number of channels"
        );
        assert_eq!(
            format!("{}", Error::BadStreamParameter),
            "bad stream parameter (clamped)"
        );
    }

    // ─── End-to-end tests ────────────────────────────────────────

    /// Helper: generate a ~10ms frame of sawtooth at the given rate.
    fn sawtooth_frame(sample_rate_hz: usize, num_channels: usize) -> Vec<f32> {
        let num_frames = sample_rate_hz / 100;
        let mut data = vec![0.0f32; num_frames * num_channels];
        for i in 0..num_frames {
            let sample = ((i % 100) as f32 / 100.0) * 2.0 - 1.0;
            for ch in 0..num_channels {
                data[i * num_channels + ch] = sample * 0.5;
            }
        }
        data
    }

    /// Helper: deinterleave a multichannel buffer into per-channel slices.
    fn deinterleave(data: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
        let num_frames = data.len() / num_channels;
        let mut channels = vec![vec![0.0f32; num_frames]; num_channels];
        for i in 0..num_frames {
            for ch in 0..num_channels {
                channels[ch][i] = data[i * num_channels + ch];
            }
        }
        channels
    }

    #[test]
    fn no_processing_when_all_disabled_float() {
        // Matches C++ test: NoProcessingWhenAllComponentsDisabledFloat.
        // With all components disabled, ProcessStream copies input to output.
        let mut apm = AudioProcessing::new();
        let config = StreamConfig::new(16000, 1);
        let num_frames = 160;
        let src_data: Vec<f32> = (0..num_frames)
            .map(|i| (i as f32 / num_frames as f32) * 2.0 - 1.0)
            .collect();
        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = vec![0.0f32; num_frames];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];

        apm.process_stream_f32(src, &config, &config, dest).unwrap();

        for i in 0..num_frames {
            assert_eq!(
                src_data[i], dest[0][i],
                "sample {i} mismatch: src={}, dest={}",
                src_data[i], dest[0][i]
            );
        }
    }

    #[test]
    fn no_processing_when_all_disabled_float_reverse() {
        let mut apm = AudioProcessing::new();
        let config = StreamConfig::new(16000, 1);
        let num_frames = 160;
        let src_data: Vec<f32> = (0..num_frames)
            .map(|i| (i as f32 / num_frames as f32) * 2.0 - 1.0)
            .collect();
        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = vec![0.0f32; num_frames];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];

        apm.process_reverse_stream_f32(src, &config, &config, dest)
            .unwrap();

        for i in 0..num_frames {
            assert_eq!(src_data[i], dest[0][i], "sample {i} mismatch");
        }
    }

    #[test]
    fn no_processing_when_all_disabled_i16() {
        // Matches C++ test: NoProcessingWhenAllComponentsDisabledInt.
        let mut apm = AudioProcessing::new();
        for &rate in &[8000usize, 16000, 32000, 48000] {
            let config = StreamConfig::new(rate, 1);
            let num_frames = rate / 100;
            let src: Vec<i16> = (0..num_frames)
                .map(|i| ((i as i32 * 200) % 30000 - 15000) as i16)
                .collect();
            let mut dest = vec![0i16; num_frames];

            apm.process_stream_i16(&src, &config, &config, &mut dest)
                .unwrap();

            for i in 0..num_frames {
                assert_eq!(src[i], dest[i], "rate={rate} sample {i} mismatch");
            }
        }
    }

    #[test]
    fn all_processing_disabled_by_default() {
        // Matches C++ test: AllProcessingDisabledByDefault.
        let apm = AudioProcessing::new();
        let config = apm.get_config();
        assert!(!config.echo_canceller.enabled);
        assert!(!config.high_pass_filter.enabled);
        assert!(!config.noise_suppression.enabled);
        assert!(!config.gain_controller2.enabled);
    }

    #[test]
    fn echo_canceller_processes_multiple_frames() {
        let mut config = Config::default();
        config.echo_canceller.enabled = true;
        let mut apm = AudioProcessing::builder().config(config).build();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        // Process 50 frames (500 ms) of render + capture.
        for _ in 0..50 {
            let render = vec![0.1f32; num_frames];
            let render_src: &[&[f32]] = &[&render];
            let mut render_dest = vec![0.0f32; num_frames];
            let render_dest: &mut [&mut [f32]] = &mut [&mut render_dest];
            apm.process_reverse_stream_f32(render_src, &stream, &stream, render_dest)
                .unwrap();

            let capture = vec![0.05f32; num_frames];
            let capture_src: &[&[f32]] = &[&capture];
            let mut capture_dest = vec![0.0f32; num_frames];
            let capture_dest: &mut [&mut [f32]] = &mut [&mut capture_dest];
            apm.process_stream_f32(capture_src, &stream, &stream, capture_dest)
                .unwrap();
        }
    }

    #[test]
    fn noise_suppression_processes_multiple_frames() {
        let mut config = Config::default();
        config.noise_suppression.enabled = true;
        config.noise_suppression.level = crate::config::NoiseSuppressionLevel::High;
        let mut apm = AudioProcessing::builder().config(config).build();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        for _ in 0..50 {
            let capture = sawtooth_frame(16000, 1);
            let channels = deinterleave(&capture, 1);
            let src: Vec<&[f32]> = channels.iter().map(|c| c.as_slice()).collect();
            let mut dest_data = vec![0.0f32; num_frames];
            let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
            apm.process_stream_f32(&src, &stream, &stream, dest)
                .unwrap();
        }
    }

    #[test]
    fn gain_controller2_processes_multiple_frames() {
        let mut config = Config::default();
        config.gain_controller2.enabled = true;
        config.gain_controller2.adaptive_digital.enabled = true;
        let mut apm = AudioProcessing::builder().config(config).build();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        for _ in 0..50 {
            let capture = sawtooth_frame(16000, 1);
            let channels = deinterleave(&capture, 1);
            let src: Vec<&[f32]> = channels.iter().map(|c| c.as_slice()).collect();
            let mut dest_data = vec![0.0f32; num_frames];
            let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
            apm.process_stream_f32(&src, &stream, &stream, dest)
                .unwrap();
        }
    }

    #[test]
    fn high_pass_filter_processes_multiple_frames() {
        let mut config = Config::default();
        config.high_pass_filter.enabled = true;
        let mut apm = AudioProcessing::builder().config(config).build();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        for _ in 0..50 {
            let capture = sawtooth_frame(16000, 1);
            let channels = deinterleave(&capture, 1);
            let src: Vec<&[f32]> = channels.iter().map(|c| c.as_slice()).collect();
            let mut dest_data = vec![0.0f32; num_frames];
            let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
            apm.process_stream_f32(&src, &stream, &stream, dest)
                .unwrap();
        }
    }

    #[test]
    fn all_components_enabled_processes_multiple_frames() {
        let mut config = Config::default();
        config.echo_canceller.enabled = true;
        config.noise_suppression.enabled = true;
        config.high_pass_filter.enabled = true;
        config.gain_controller2.enabled = true;
        config.gain_controller2.adaptive_digital.enabled = true;
        let mut apm = AudioProcessing::builder().config(config).build();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        for _ in 0..100 {
            // Render.
            let render = sawtooth_frame(16000, 1);
            let render_ch = deinterleave(&render, 1);
            let render_src: Vec<&[f32]> = render_ch.iter().map(|c| c.as_slice()).collect();
            let mut render_dest = vec![0.0f32; num_frames];
            let render_dest: &mut [&mut [f32]] = &mut [&mut render_dest];
            apm.process_reverse_stream_f32(&render_src, &stream, &stream, render_dest)
                .unwrap();

            // Capture.
            let capture = sawtooth_frame(16000, 1);
            let capture_ch = deinterleave(&capture, 1);
            let capture_src: Vec<&[f32]> = capture_ch.iter().map(|c| c.as_slice()).collect();
            let mut capture_dest = vec![0.0f32; num_frames];
            let capture_dest: &mut [&mut [f32]] = &mut [&mut capture_dest];
            apm.process_stream_f32(&capture_src, &stream, &stream, capture_dest)
                .unwrap();
        }
    }

    #[test]
    fn config_change_mid_stream() {
        let mut apm = AudioProcessing::new();
        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;
        let silence = vec![0.0f32; num_frames];
        let src: &[&[f32]] = &[&silence];
        let mut dest_data = vec![0.0f32; num_frames];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];

        // Process a few frames with default config.
        for _ in 0..10 {
            apm.process_stream_f32(src, &stream, &stream, dest).unwrap();
        }

        // Enable echo canceller mid-stream.
        let mut config = Config::default();
        config.echo_canceller.enabled = true;
        apm.apply_config(config);

        // Process more frames — should not crash.
        for _ in 0..10 {
            apm.process_stream_f32(src, &stream, &stream, dest).unwrap();
        }

        // Enable noise suppression mid-stream.
        let mut config = apm.get_config().clone();
        config.noise_suppression.enabled = true;
        apm.apply_config(config);

        for _ in 0..10 {
            apm.process_stream_f32(src, &stream, &stream, dest).unwrap();
        }
    }

    #[test]
    fn sample_rate_change_mid_stream() {
        let mut apm = AudioProcessing::new();

        // Process at 16 kHz.
        let stream_16k = StreamConfig::new(16000, 1);
        let src_16k = vec![0.0f32; 160];
        let src: &[&[f32]] = &[&src_16k];
        let mut dest_16k = vec![0.0f32; 160];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_16k];
        for _ in 0..5 {
            apm.process_stream_f32(src, &stream_16k, &stream_16k, dest)
                .unwrap();
        }

        // Switch to 48 kHz — should reinitialize automatically.
        let stream_48k = StreamConfig::new(48000, 1);
        let src_48k = vec![0.0f32; 480];
        let src: &[&[f32]] = &[&src_48k];
        let mut dest_48k = vec![0.0f32; 480];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_48k];
        for _ in 0..5 {
            apm.process_stream_f32(src, &stream_48k, &stream_48k, dest)
                .unwrap();
        }
    }

    #[test]
    fn pre_amplifier_applies_gain() {
        // Matches C++ test: PreAmplifier (simplified).
        let mut config = Config::default();
        config.pre_amplifier.enabled = true;
        config.pre_amplifier.fixed_gain_factor = 2.0;
        let mut apm = AudioProcessing::builder().config(config).build();

        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        // Run enough frames for the filter to settle.
        for _ in 0..20 {
            let src_data: Vec<f32> = (0..num_frames)
                .map(|i| ((i % 3) as f32 - 1.0) * 0.3)
                .collect();
            let src: &[&[f32]] = &[&src_data];
            let mut dest_data = vec![0.0f32; num_frames];
            let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
            apm.process_stream_f32(src, &stream, &stream, dest).unwrap();
        }

        // Process one more frame and check that the output has roughly 2x power.
        let src_data: Vec<f32> = (0..num_frames)
            .map(|i| ((i % 3) as f32 - 1.0) * 0.3)
            .collect();
        let input_power: f32 = src_data.iter().map(|&s| s * s).sum::<f32>() / num_frames as f32;

        let src: &[&[f32]] = &[&src_data];
        let mut dest_data = vec![0.0f32; num_frames];
        let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
        apm.process_stream_f32(src, &stream, &stream, dest).unwrap();

        let output_power: f32 = dest[0].iter().map(|&s| s * s).sum::<f32>() / num_frames as f32;

        // 2x gain → 4x power. Allow some tolerance for filter settling.
        assert!(
            output_power > input_power * 3.0,
            "expected ~4x power, got input={input_power}, output={output_power}",
        );
    }

    #[test]
    fn runtime_setting_capture_pre_gain() {
        let mut config = Config::default();
        config.pre_amplifier.enabled = true;
        config.pre_amplifier.fixed_gain_factor = 1.0;
        let mut apm = AudioProcessing::builder().config(config).build();

        // Change gain via runtime setting.
        apm.set_runtime_setting(RuntimeSetting::CapturePreGain(2.0));

        let stream = StreamConfig::new(16000, 1);
        let num_frames = 160;

        // Process enough frames for the setting to take effect.
        for _ in 0..20 {
            let src_data = vec![0.1f32; num_frames];
            let src: &[&[f32]] = &[&src_data];
            let mut dest_data = vec![0.0f32; num_frames];
            let dest: &mut [&mut [f32]] = &mut [&mut dest_data];
            apm.process_stream_f32(src, &stream, &stream, dest).unwrap();
        }

        // Verify the config was updated.
        let config = apm.get_config();
        assert_eq!(config.pre_amplifier.fixed_gain_factor, 2.0);
    }

    #[test]
    fn stereo_processing_does_not_crash() {
        let mut config = Config::default();
        config.echo_canceller.enabled = true;
        config.pipeline.multi_channel_render = true;
        config.pipeline.multi_channel_capture = true;
        let mut apm = AudioProcessing::builder().config(config).build();

        let stream = StreamConfig::new(16000, 2);
        let num_frames = 160;

        for _ in 0..20 {
            // Render (stereo).
            let render_l = vec![0.1f32; num_frames];
            let render_r = vec![0.05f32; num_frames];
            let render_src: &[&[f32]] = &[&render_l, &render_r];
            let mut render_dest_l = vec![0.0f32; num_frames];
            let mut render_dest_r = vec![0.0f32; num_frames];
            let render_dest: &mut [&mut [f32]] = &mut [&mut render_dest_l, &mut render_dest_r];
            apm.process_reverse_stream_f32(render_src, &stream, &stream, render_dest)
                .unwrap();

            // Capture (stereo).
            let capture_l = vec![0.05f32; num_frames];
            let capture_r = vec![0.02f32; num_frames];
            let capture_src: &[&[f32]] = &[&capture_l, &capture_r];
            let mut capture_dest_l = vec![0.0f32; num_frames];
            let mut capture_dest_r = vec![0.0f32; num_frames];
            let capture_dest: &mut [&mut [f32]] = &mut [&mut capture_dest_l, &mut capture_dest_r];
            apm.process_stream_f32(capture_src, &stream, &stream, capture_dest)
                .unwrap();
        }
    }
}
