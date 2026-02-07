//! Echo audibility estimation based on render signal stationarity.
//!
//! Ported from `modules/audio_processing/aec3/echo_audibility.h/cc`.

use crate::block_buffer::BlockBuffer;
use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2_PLUS_1};
use crate::render_buffer::RenderBuffer;
use crate::spectrum_buffer::SpectrumBuffer;
use crate::stationarity_estimator::StationarityEstimator;

/// Estimates whether the echo is audible based on render signal stationarity.
pub(crate) struct EchoAudibility {
    render_spectrum_write_prev: Option<usize>,
    render_block_write_prev: usize,
    non_zero_render_seen: bool,
    use_render_stationarity_at_init: bool,
    render_stationarity: StationarityEstimator,
}

impl EchoAudibility {
    pub(crate) fn new(use_render_stationarity_at_init: bool) -> Self {
        let mut s = Self {
            render_spectrum_write_prev: None,
            render_block_write_prev: 0,
            non_zero_render_seen: false,
            use_render_stationarity_at_init,
            render_stationarity: StationarityEstimator::new(),
        };
        s.reset();
        s
    }

    /// Feeds new render data to the echo audibility estimator.
    pub(crate) fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        average_reverb: &[f32],
        min_channel_delay_blocks: i32,
        external_delay_seen: bool,
    ) {
        self.update_render_noise_estimator(
            render_buffer.get_spectrum_buffer(),
            render_buffer.get_block_buffer(),
            external_delay_seen,
        );

        if external_delay_seen || self.use_render_stationarity_at_init {
            self.update_render_stationarity_flags(
                render_buffer,
                average_reverb,
                min_channel_delay_blocks,
            );
        }
    }

    /// Gets the residual echo scaling.
    pub(crate) fn get_residual_echo_scaling(
        &self,
        filter_has_had_time_to_converge: bool,
        residual_scaling: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        for band in 0..residual_scaling.len() {
            if self.render_stationarity.is_band_stationary(band)
                && (filter_has_had_time_to_converge || self.use_render_stationarity_at_init)
            {
                residual_scaling[band] = 0.0;
            } else {
                residual_scaling[band] = 1.0;
            }
        }
    }

    /// Returns true if the current render block is estimated as stationary.
    pub(crate) fn is_block_stationary(&self) -> bool {
        self.render_stationarity.is_block_stationary()
    }

    fn reset(&mut self) {
        self.render_stationarity.reset();
        self.non_zero_render_seen = false;
        self.render_spectrum_write_prev = None;
    }

    fn update_render_stationarity_flags(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        average_reverb: &[f32],
        min_channel_delay_blocks: i32,
    ) {
        let spectrum_buffer = render_buffer.get_spectrum_buffer();
        let idx_at_delay = spectrum_buffer
            .index
            .offset_index(spectrum_buffer.index.read, min_channel_delay_blocks);

        let mut num_lookahead = render_buffer.headroom() as i32 - min_channel_delay_blocks + 1;
        num_lookahead = num_lookahead.max(0);

        self.render_stationarity.update_stationarity_flags(
            spectrum_buffer,
            average_reverb,
            idx_at_delay,
            num_lookahead as usize,
        );
    }

    fn update_render_noise_estimator(
        &mut self,
        spectrum_buffer: &SpectrumBuffer,
        block_buffer: &BlockBuffer,
        external_delay_seen: bool,
    ) {
        if self.render_spectrum_write_prev.is_none() {
            self.render_spectrum_write_prev = Some(spectrum_buffer.index.write);
            self.render_block_write_prev = block_buffer.index.write;
            return;
        }

        let render_spectrum_write_current = spectrum_buffer.index.write;

        if !self.non_zero_render_seen && !external_delay_seen {
            self.non_zero_render_seen = !self.is_render_too_low(block_buffer);
        }

        if self.non_zero_render_seen {
            let mut idx = self.render_spectrum_write_prev.unwrap();
            while idx != render_spectrum_write_current {
                self.render_stationarity
                    .update_noise_estimator(&spectrum_buffer.buffer[idx]);
                idx = spectrum_buffer.index.dec_index(idx);
            }
        }

        self.render_spectrum_write_prev = Some(render_spectrum_write_current);
    }

    fn is_render_too_low(&mut self, block_buffer: &BlockBuffer) -> bool {
        let num_render_channels = block_buffer.buffer[0].num_channels();
        let render_block_write_current = block_buffer.index.write;

        let too_low;
        if render_block_write_current == self.render_block_write_prev {
            too_low = true;
        } else {
            let mut found_low = false;
            let mut idx = self.render_block_write_prev;
            while idx != render_block_write_current {
                let mut max_abs_over_channels = 0.0f32;
                for ch in 0..num_render_channels {
                    let block = block_buffer.buffer[idx].view(0, ch);
                    let mut min_val = f32::MAX;
                    let mut max_val = f32::MIN;
                    for k in 0..BLOCK_SIZE {
                        min_val = min_val.min(block[k]);
                        max_val = max_val.max(block[k]);
                    }
                    let max_abs_channel = min_val.abs().max(max_val.abs());
                    max_abs_over_channels = max_abs_over_channels.max(max_abs_channel);
                }
                if max_abs_over_channels < 10.0 {
                    found_low = true;
                    break;
                }
                idx = block_buffer.index.inc_index(idx);
            }
            too_low = found_low;
        }

        self.render_block_write_prev = render_block_write_current;
        too_low
    }
}
