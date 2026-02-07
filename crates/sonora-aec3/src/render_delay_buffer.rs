//! Render delay buffer — buffers incoming render blocks for extraction with a
//! specified delay.
//!
//! The C++ version is an abstract class with a factory (`Create`). We port the
//! single concrete implementation directly.
//!
//! Ported from `modules/audio_processing/aec3/render_delay_buffer.h/cc`.

use crate::aec3_fft::Aec3Fft;
use crate::alignment_mixer::AlignmentMixer;
use crate::block::Block;
use crate::block_buffer::BlockBuffer;
use crate::common::{
    BLOCK_SIZE, BLOCK_SIZE_MS, FFT_LENGTH_BY_2, get_down_sampled_buffer_size,
    get_render_delay_buffer_size, num_bands_for_rate,
};
use crate::config::EchoCanceller3Config;
use crate::decimator::Decimator;
use crate::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::fft_buffer::FftBuffer;
use crate::render_buffer::RenderBuffer;
use crate::spectrum_buffer::SpectrumBuffer;

/// Events that can occur during buffer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferingEvent {
    None,
    RenderUnderrun,
    RenderOverrun,
    ApiCallSkew,
}

/// Buffers incoming render blocks such that these may be extracted with a
/// specified delay.
pub(crate) struct RenderDelayBuffer {
    config: EchoCanceller3Config,
    render_linear_amplitude_gain: f32,
    down_sampling_factor: usize,
    sub_block_size: usize,
    blocks: BlockBuffer,
    spectra: SpectrumBuffer,
    ffts: FftBuffer,
    delay: Option<usize>,
    low_rate: DownsampledRenderBuffer,
    render_mixer: AlignmentMixer,
    render_decimator: Decimator,
    fft: Aec3Fft,
    render_ds: Vec<f32>,
    buffer_headroom: usize,
    render_activity: bool,
    last_call_was_render: bool,
    num_api_calls_in_a_row: i32,
    max_observed_jitter: i32,
    capture_call_counter: i64,
    render_call_counter: i64,
    render_activity_counter: usize,
    external_audio_buffer_delay: Option<i32>,
    external_audio_buffer_delay_verified_after_reset: bool,
    min_latency_blocks: usize,
    excess_render_detection_counter: usize,
}

impl RenderDelayBuffer {
    pub(crate) fn new(
        config: &EchoCanceller3Config,
        sample_rate_hz: usize,
        num_render_channels: usize,
    ) -> Self {
        let down_sampling_factor = config.delay.down_sampling_factor;
        let sub_block_size = if down_sampling_factor > 0 {
            BLOCK_SIZE / down_sampling_factor
        } else {
            BLOCK_SIZE
        };

        let buffer_size = get_render_delay_buffer_size(
            down_sampling_factor,
            config.delay.num_filters,
            config.filter.refined.length_blocks,
        );
        let num_bands = num_bands_for_rate(sample_rate_hz);

        let blocks = BlockBuffer::new(buffer_size, num_bands, num_render_channels);
        let spectra = SpectrumBuffer::new(buffer_size, num_render_channels);
        let ffts = FftBuffer::new(buffer_size, num_render_channels);

        let render_linear_amplitude_gain =
            10.0f32.powf(config.render_levels.render_power_gain_db / 20.0);

        let low_rate = DownsampledRenderBuffer::new(get_down_sampled_buffer_size(
            down_sampling_factor,
            config.delay.num_filters,
        ));

        let buffer_headroom = config.filter.refined.length_blocks;

        let mut rdb = Self {
            config: config.clone(),
            render_linear_amplitude_gain,
            down_sampling_factor,
            sub_block_size,
            blocks,
            spectra,
            ffts,
            delay: Some(config.delay.default_delay),
            low_rate,
            render_mixer: AlignmentMixer::new(
                num_render_channels,
                &config.delay.render_alignment_mixing,
            ),
            render_decimator: Decimator::new(down_sampling_factor),
            fft: Aec3Fft::new(),
            render_ds: vec![0.0f32; sub_block_size],
            buffer_headroom,
            render_activity: false,
            last_call_was_render: false,
            num_api_calls_in_a_row: 1,
            max_observed_jitter: 1,
            capture_call_counter: 0,
            render_call_counter: 0,
            render_activity_counter: 0,
            external_audio_buffer_delay: None,
            external_audio_buffer_delay_verified_after_reset: false,
            min_latency_blocks: 0,
            excess_render_detection_counter: 0,
        };
        rdb.reset();
        rdb
    }

    /// Resets the buffer alignment.
    pub(crate) fn reset(&mut self) {
        self.last_call_was_render = false;
        self.num_api_calls_in_a_row = 1;
        self.min_latency_blocks = 0;
        self.excess_render_detection_counter = 0;

        // Initialize the read index to one sub-block before the write index.
        self.low_rate.read = self
            .low_rate
            .offset_index(self.low_rate.write, self.sub_block_size as i32);

        // Check for any external audio buffer delay and whether it is feasible.
        if let Some(ext_delay) = self.external_audio_buffer_delay {
            let headroom = 2i32;
            let audio_buffer_delay_to_set = if ext_delay <= headroom {
                1usize
            } else {
                (ext_delay - headroom) as usize
            };
            let audio_buffer_delay_to_set = audio_buffer_delay_to_set.min(self.max_delay());

            self.apply_total_delay(audio_buffer_delay_to_set as i32);
            self.delay = Some(self.compute_delay() as usize);
            self.external_audio_buffer_delay_verified_after_reset = false;
        } else {
            self.apply_total_delay(self.config.delay.default_delay as i32);
            self.delay = None;
        }
    }

    /// Inserts a block into the buffer.
    pub(crate) fn insert(&mut self, block: &Block) -> BufferingEvent {
        self.render_call_counter += 1;
        if self.delay.is_some() {
            if !self.last_call_was_render {
                self.last_call_was_render = true;
                self.num_api_calls_in_a_row = 1;
            } else {
                self.num_api_calls_in_a_row += 1;
                if self.num_api_calls_in_a_row > self.max_observed_jitter {
                    self.max_observed_jitter = self.num_api_calls_in_a_row;
                }
            }
        }

        // Increase the write indices to where the new blocks should be written.
        let previous_write = self.blocks.index.write;
        self.increment_write_indices();

        // Allow overrun and do a reset when render overrun occurs due to more
        // render data being inserted than capture data is received.
        let event = if self.render_overrun() {
            BufferingEvent::RenderOverrun
        } else {
            BufferingEvent::None
        };

        // Detect and update render activity.
        if !self.render_activity {
            if self.detect_active_render(block.view(0, 0)) {
                self.render_activity_counter += 1;
            }
            self.render_activity = self.render_activity_counter >= 20;
        }

        // Insert the new render block into the specified position.
        self.insert_block(block, previous_write);

        if event != BufferingEvent::None {
            self.reset();
        }

        event
    }

    /// Called on capture blocks where `prepare_capture_processing` is not
    /// called.
    pub(crate) fn handle_skipped_capture_processing(&mut self) {
        self.capture_call_counter += 1;
    }

    /// Prepares the render buffers for processing another capture block.
    pub(crate) fn prepare_capture_processing(&mut self) -> BufferingEvent {
        let mut event = BufferingEvent::None;
        self.capture_call_counter += 1;

        if self.delay.is_some() {
            if self.last_call_was_render {
                self.last_call_was_render = false;
                self.num_api_calls_in_a_row = 1;
            } else {
                self.num_api_calls_in_a_row += 1;
                if self.num_api_calls_in_a_row > self.max_observed_jitter {
                    self.max_observed_jitter = self.num_api_calls_in_a_row;
                }
            }
        }

        if self.detect_excess_render_blocks() {
            self.reset();
            event = BufferingEvent::RenderOverrun;
        } else if self.render_underrun() {
            // Don't increment the read indices of the low rate buffer if there
            // is a render underrun.
            self.increment_read_indices();
            // Incrementing the buffer index without increasing the low rate
            // buffer index means that the delay is reduced by one.
            if let Some(d) = self.delay
                && d > 0
            {
                self.delay = Some(d - 1);
            }
            event = BufferingEvent::RenderUnderrun;
        } else {
            // Increment the read indices in the render buffers to point to the
            // most recent block to use in the capture processing.
            self.increment_low_rate_read_indices();
            self.increment_read_indices();
        }

        self.render_activity = if self.render_activity {
            self.render_activity_counter = 0;
            false
        } else {
            false
        };

        event
    }

    /// Sets the buffer delay and returns a bool indicating whether the delay
    /// changed.
    pub(crate) fn align_from_delay(&mut self, delay: usize) -> bool {
        if let Some(d) = self.delay {
            if !self.external_audio_buffer_delay_verified_after_reset
                && self.external_audio_buffer_delay.is_some()
            {
                self.external_audio_buffer_delay_verified_after_reset = true;
            }
            if d == delay {
                return false;
            }
        }
        self.delay = Some(delay);

        // Compute the total delay and limit the delay to the allowed range.
        let total_delay = self.map_delay_to_total_delay(delay);
        let total_delay = total_delay.max(0).min(self.max_delay() as i32);

        // Apply the delay to the buffers.
        self.apply_total_delay(total_delay);
        true
    }

    /// Sets the buffer delay from the most recently reported external delay.
    pub(crate) fn align_from_external_delay(&mut self) {
        if let Some(ext_delay) = self.external_audio_buffer_delay {
            let delay = self.render_call_counter - self.capture_call_counter + ext_delay as i64;
            let delay_with_headroom =
                delay - self.config.delay.delay_headroom_samples as i64 / BLOCK_SIZE as i64;
            self.apply_total_delay(delay_with_headroom as i32);
        }
    }

    /// Gets the buffer delay.
    pub(crate) fn delay(&self) -> usize {
        self.compute_delay() as usize
    }

    /// Gets the maximum delay.
    pub(crate) fn max_delay(&self) -> usize {
        self.blocks.buffer.len() - 1 - self.buffer_headroom
    }

    /// Returns a `RenderBuffer` view for the echo remover.
    pub(crate) fn get_render_buffer(&self) -> RenderBuffer<'_> {
        let mut rb = RenderBuffer::new(&self.blocks, &self.spectra, &self.ffts);
        rb.set_render_activity(self.render_activity);
        rb
    }

    /// Returns a reference to the downsampled render buffer.
    pub(crate) fn get_downsampled_render_buffer(&self) -> &DownsampledRenderBuffer {
        &self.low_rate
    }

    /// Computes the delay estimator offset for the given config.
    pub(crate) fn delay_estimator_offset(config: &EchoCanceller3Config) -> i32 {
        let delay_headroom_blocks = config.delay.delay_headroom_samples as i32 / BLOCK_SIZE as i32;
        let hysteresis = config.delay.hysteresis_limit_blocks as i32;
        delay_headroom_blocks + hysteresis
    }

    /// Provides an optional external estimate of the audio buffer delay.
    pub(crate) fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        const SAMPLE_RATE_FOR_FIXED_CAPTURE_DELAY: i32 = 16000;
        const NUM_SAMPLES_PER_MS: i32 = SAMPLE_RATE_FOR_FIXED_CAPTURE_DELAY / 1000;
        self.external_audio_buffer_delay = Some(
            (delay_ms * NUM_SAMPLES_PER_MS + self.config.delay.fixed_capture_delay_samples as i32)
                / (BLOCK_SIZE_MS as i32 * NUM_SAMPLES_PER_MS),
        );
    }

    /// Returns whether an external delay estimate has been reported.
    pub(crate) fn has_received_buffer_delay(&self) -> bool {
        self.external_audio_buffer_delay.is_some()
    }

    // --- Private methods ---

    fn map_delay_to_total_delay(&self, external_delay_blocks: usize) -> i32 {
        let latency_blocks = self.buffer_latency();
        latency_blocks + external_delay_blocks as i32
    }

    fn compute_delay(&self) -> i32 {
        let latency_blocks = self.buffer_latency();
        let size = self.spectra.index.size;
        let internal_delay = if self.spectra.index.read >= self.spectra.index.write {
            self.spectra.index.read - self.spectra.index.write
        } else {
            size + self.spectra.index.read - self.spectra.index.write
        };
        internal_delay as i32 - latency_blocks
    }

    fn apply_total_delay(&mut self, delay: i32) {
        self.blocks.index.read = self
            .blocks
            .index
            .offset_index(self.blocks.index.write, -delay);
        self.spectra.index.read = self
            .spectra
            .index
            .offset_index(self.spectra.index.write, delay);
        self.ffts.index.read = self.ffts.index.offset_index(self.ffts.index.write, delay);
    }

    fn insert_block(&mut self, block: &Block, previous_write: usize) {
        let write_idx = self.blocks.index.write;
        let num_bands = self.blocks.buffer[write_idx].num_bands();
        let num_render_channels = self.blocks.buffer[write_idx].num_channels();
        debug_assert_eq!(block.num_bands(), num_bands);
        debug_assert_eq!(block.num_channels(), num_render_channels);

        // Copy block data.
        for band in 0..num_bands {
            for ch in 0..num_render_channels {
                let src = block.view(band, ch);
                let dst = self.blocks.buffer[write_idx].view_mut(band, ch);
                dst.copy_from_slice(src);
            }
        }

        // Apply render linear amplitude gain if needed.
        if self.render_linear_amplitude_gain != 1.0 {
            for band in 0..num_bands {
                for ch in 0..num_render_channels {
                    let view = self.blocks.buffer[write_idx].view_mut(band, ch);
                    for sample in view.iter_mut() {
                        *sample *= self.render_linear_amplitude_gain;
                    }
                }
            }
        }

        // Downmix and decimate for the low-rate buffer.
        let mut downmixed_render = [0.0f32; BLOCK_SIZE];
        self.render_mixer
            .produce_output(&self.blocks.buffer[write_idx], &mut downmixed_render);
        self.render_decimator
            .decimate(&downmixed_render, &mut self.render_ds);

        // Copy decimated data into low-rate buffer in reverse order.
        let lr_write = self.low_rate.write;
        for (i, &v) in self.render_ds.iter().rev().enumerate() {
            let idx = (lr_write + i) % self.low_rate.size;
            self.low_rate.buffer[idx] = v;
        }

        // Compute FFT and spectrum for each channel.
        let fft_write = self.ffts.index.write;
        let spec_write = self.spectra.index.write;
        for channel in 0..num_render_channels {
            // Get the current and previous block data for PaddedFft.
            let x = self.blocks.buffer[write_idx].view(0, channel);
            let x_old = self.blocks.buffer[previous_write].view(0, channel);

            // Need to copy to avoid borrow issues with self.fft.
            let mut x_buf = [0.0f32; FFT_LENGTH_BY_2];
            let mut x_old_buf = [0.0f32; FFT_LENGTH_BY_2];
            x_buf.copy_from_slice(x);
            x_old_buf.copy_from_slice(x_old);

            self.fft.padded_fft_rect(
                &x_buf,
                &x_old_buf,
                &mut self.ffts.buffer[fft_write][channel],
            );
            self.ffts.buffer[fft_write][channel]
                .spectrum(&mut self.spectra.buffer[spec_write][channel]);
        }
    }

    fn detect_active_render(&self, x: &[f32]) -> bool {
        let x_energy: f32 = x.iter().map(|&v| v * v).sum();
        x_energy
            > (self.config.render_levels.active_render_limit
                * self.config.render_levels.active_render_limit)
                * FFT_LENGTH_BY_2 as f32
    }

    fn detect_excess_render_blocks(&mut self) -> bool {
        let mut excess_render_detected = false;
        let latency_blocks = self.buffer_latency() as usize;
        self.min_latency_blocks = self.min_latency_blocks.min(latency_blocks);

        self.excess_render_detection_counter += 1;
        if self.excess_render_detection_counter
            >= self
                .config
                .buffering
                .excess_render_detection_interval_blocks
        {
            excess_render_detected =
                self.min_latency_blocks > self.config.buffering.max_allowed_excess_render_blocks;
            self.min_latency_blocks = latency_blocks;
            self.excess_render_detection_counter = 0;
        }

        excess_render_detected
    }

    fn buffer_latency(&self) -> i32 {
        let lr = &self.low_rate;
        let latency_samples = (lr.size + lr.read - lr.write) % lr.size;
        let latency_blocks = latency_samples / self.sub_block_size;
        latency_blocks as i32
    }

    fn increment_write_indices(&mut self) {
        self.low_rate
            .update_write_index(-(self.sub_block_size as i32));
        self.blocks.index.inc_write();
        self.spectra.index.dec_write();
        self.ffts.index.dec_write();
    }

    fn increment_low_rate_read_indices(&mut self) {
        self.low_rate
            .update_read_index(-(self.sub_block_size as i32));
    }

    fn increment_read_indices(&mut self) {
        if self.blocks.index.read != self.blocks.index.write {
            self.blocks.index.inc_read();
            self.spectra.index.dec_read();
            self.ffts.index.dec_read();
        }
    }

    fn render_overrun(&self) -> bool {
        self.low_rate.read == self.low_rate.write
            || self.blocks.index.read == self.blocks.index.write
    }

    fn render_underrun(&self) -> bool {
        self.low_rate.read == self.low_rate.write
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_default_buffer(sample_rate_hz: usize, num_channels: usize) -> RenderDelayBuffer {
        let config = EchoCanceller3Config::default();
        RenderDelayBuffer::new(&config, sample_rate_hz, num_channels)
    }

    #[test]
    fn basic_insert_and_prepare() {
        for &rate in &[16000usize, 32000, 48000] {
            for &channels in &[1usize, 2] {
                let mut buf = create_default_buffer(rate, channels);
                let num_bands = num_bands_for_rate(rate);
                let block = Block::new(num_bands, channels);

                // Insert should succeed without overrun.
                let event = buf.insert(&block);
                assert_eq!(event, BufferingEvent::None, "rate={rate}, ch={channels}");

                // Prepare capture processing should succeed.
                let event = buf.prepare_capture_processing();
                assert_ne!(
                    event,
                    BufferingEvent::RenderOverrun,
                    "rate={rate}, ch={channels}"
                );
            }
        }
    }

    #[test]
    fn buffer_overflow() {
        // Matches C++ test: insert many blocks without capture processing to
        // trigger overflow detection.
        for &rate in &[16000usize, 32000, 48000] {
            for &channels in &[1usize, 2, 8] {
                let mut buf = create_default_buffer(rate, channels);
                let num_bands = num_bands_for_rate(rate);
                let block = Block::new(num_bands, channels);

                let mut overrun_detected = false;
                for _ in 0..1000 {
                    if buf.insert(&block) != BufferingEvent::None {
                        overrun_detected = true;
                        break;
                    }
                }
                assert!(
                    overrun_detected,
                    "Expected overrun for rate={rate}, ch={channels}"
                );
            }
        }
    }

    #[test]
    fn align_from_delay_returns_changed() {
        let mut buf = create_default_buffer(16000, 1);
        let block = Block::new(1, 1);

        // Prime the buffer.
        for _ in 0..10 {
            buf.insert(&block);
            buf.prepare_capture_processing();
        }

        // First alignment should return true (delay changed).
        assert!(buf.align_from_delay(2));
        // Same delay again should return false.
        assert!(!buf.align_from_delay(2));
        // Different delay should return true.
        assert!(buf.align_from_delay(5));
    }

    #[test]
    fn render_buffer_view_accessible() {
        let mut buf = create_default_buffer(16000, 1);
        let block = Block::new(1, 1);

        buf.insert(&block);
        buf.prepare_capture_processing();

        let render_buf = buf.get_render_buffer();
        // Should be able to access the block and spectrum.
        let _b = render_buf.get_block(0);
        let _s = render_buf.spectrum(0);
    }

    #[test]
    fn set_audio_buffer_delay() {
        let mut buf = create_default_buffer(16000, 1);
        assert!(!buf.has_received_buffer_delay());
        buf.set_audio_buffer_delay(20);
        assert!(buf.has_received_buffer_delay());
    }

    #[test]
    fn max_delay_respects_headroom() {
        let config = EchoCanceller3Config::default();
        let buf = RenderDelayBuffer::new(&config, 16000, 1);
        let expected_max = buf.blocks.buffer.len() - 1 - config.filter.refined.length_blocks;
        assert_eq!(buf.max_delay(), expected_max);
    }
}
