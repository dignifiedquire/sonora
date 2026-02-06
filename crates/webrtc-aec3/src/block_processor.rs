//! Block processor — block-level echo cancellation processing.
//!
//! Connects the render delay buffer, render delay controller, and echo remover
//! to provide full echo cancellation at the block level (64 samples per block).
//!
//! Ported from `modules/audio_processing/aec3/block_processor.h/cc`.

use crate::block::Block;
use crate::block_processor_metrics::BlockProcessorMetrics;
use crate::common::{num_bands_for_rate, valid_full_band_rate};
use crate::config::EchoCanceller3Config;
use crate::delay_estimate::DelayEstimate;
use crate::echo_path_variability::{DelayAdjustment, EchoPathVariability};
use crate::echo_remover::EchoRemover;
use crate::render_delay_buffer::{BufferingEvent, RenderDelayBuffer};
use crate::render_delay_controller::RenderDelayController;

/// Block-level echo cancellation processor.
pub(crate) struct BlockProcessor {
    config: EchoCanceller3Config,
    capture_properly_started: bool,
    render_properly_started: bool,
    sample_rate_hz: usize,
    render_buffer: RenderDelayBuffer,
    delay_controller: Option<RenderDelayController>,
    echo_remover: EchoRemover,
    metrics: BlockProcessorMetrics,
    render_event: BufferingEvent,
    capture_call_counter: usize,
    estimated_delay: Option<DelayEstimate>,
}

/// Metrics output from the block processor.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BlockProcessorMetricsOutput {
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
}

impl BlockProcessor {
    pub(crate) fn new(
        config: &EchoCanceller3Config,
        sample_rate_hz: usize,
        num_render_channels: usize,
        num_capture_channels: usize,
    ) -> Self {
        debug_assert!(valid_full_band_rate(sample_rate_hz));

        let render_buffer = RenderDelayBuffer::new(config, sample_rate_hz, num_render_channels);

        let delay_controller = if !config.delay.use_external_delay_estimator {
            Some(RenderDelayController::new(config, num_capture_channels))
        } else {
            None
        };

        let echo_remover = EchoRemover::new(
            config,
            sample_rate_hz,
            num_render_channels,
            num_capture_channels,
        );

        Self {
            config: config.clone(),
            capture_properly_started: false,
            render_properly_started: false,
            sample_rate_hz,
            render_buffer,
            delay_controller,
            echo_remover,
            metrics: BlockProcessorMetrics::new(),
            render_event: BufferingEvent::None,
            capture_call_counter: 0,
            estimated_delay: None,
        }
    }

    /// Returns current echo cancellation metrics.
    pub(crate) fn get_metrics(&self) -> BlockProcessorMetricsOutput {
        let echo_metrics = self.echo_remover.get_metrics();
        const BLOCK_SIZE_MS: i32 = 4;
        BlockProcessorMetricsOutput {
            echo_return_loss: echo_metrics.echo_return_loss,
            echo_return_loss_enhancement: echo_metrics.echo_return_loss_enhancement,
            delay_ms: self.render_buffer.delay() as i32 * BLOCK_SIZE_MS,
        }
    }

    /// Provides an optional external estimate of the audio buffer delay.
    pub(crate) fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.render_buffer.set_audio_buffer_delay(delay_ms);
    }

    /// Processes a block of capture data.
    pub(crate) fn process_capture(
        &mut self,
        echo_path_gain_change: bool,
        capture_signal_saturation: bool,
        linear_output: Option<&mut Block>,
        capture_block: &mut Block,
    ) {
        debug_assert_eq!(
            num_bands_for_rate(self.sample_rate_hz),
            capture_block.num_bands()
        );

        self.capture_call_counter += 1;

        if self.render_properly_started {
            if !self.capture_properly_started {
                self.capture_properly_started = true;
                self.render_buffer.reset();
                if let Some(ref mut dc) = self.delay_controller {
                    dc.reset(true);
                }
            }
        } else {
            // If no render data has yet arrived, do not process the capture signal.
            self.render_buffer.handle_skipped_capture_processing();
            return;
        }

        let mut echo_path_variability =
            EchoPathVariability::new(echo_path_gain_change, DelayAdjustment::None, false);

        if self.render_event == BufferingEvent::RenderOverrun && self.render_properly_started {
            echo_path_variability.delay_change = DelayAdjustment::BufferFlush;
            if let Some(ref mut dc) = self.delay_controller {
                dc.reset(true);
            }
        }
        self.render_event = BufferingEvent::None;

        // Update the render buffers and prepare for reading.
        let buffer_event = self.render_buffer.prepare_capture_processing();
        // Reset the delay controller at render buffer underrun.
        if buffer_event == BufferingEvent::RenderUnderrun {
            if let Some(ref mut dc) = self.delay_controller {
                dc.reset(false);
            }
        }

        let has_delay_estimator = !self.config.delay.use_external_delay_estimator;
        if has_delay_estimator {
            let dc = self
                .delay_controller
                .as_mut()
                .expect("delay controller must exist when not using external delay estimator");
            // Compute and apply the render delay.
            self.estimated_delay = dc.get_delay(
                self.render_buffer.get_downsampled_render_buffer(),
                self.render_buffer.delay(),
                capture_block,
            );

            if let Some(ref estimated_delay) = self.estimated_delay {
                let delay_change = self.render_buffer.align_from_delay(estimated_delay.delay);
                if delay_change {
                    echo_path_variability.delay_change = DelayAdjustment::NewDetectedDelay;
                }
            }

            echo_path_variability.clock_drift = dc.has_clockdrift();
        } else {
            self.render_buffer.align_from_external_delay();
        }

        // Remove the echo from the capture signal.
        if has_delay_estimator || self.render_buffer.has_received_buffer_delay() {
            let render_buffer = self.render_buffer.get_render_buffer();
            self.echo_remover.process_capture(
                echo_path_variability,
                capture_signal_saturation,
                &self.estimated_delay,
                &render_buffer,
                linear_output,
                capture_block,
            );
        }

        // Update the metrics.
        self.metrics.update_capture(false);
    }

    /// Buffers a block of render data.
    pub(crate) fn buffer_render(&mut self, block: &Block) {
        debug_assert_eq!(num_bands_for_rate(self.sample_rate_hz), block.num_bands());

        self.render_event = self.render_buffer.insert(block);

        self.metrics
            .update_render(self.render_event != BufferingEvent::None);

        self.render_properly_started = true;
        if let Some(ref dc) = self.delay_controller {
            dc.log_render_call();
        }
    }

    /// Reports whether echo leakage has been detected.
    pub(crate) fn update_echo_leakage_status(&mut self, leakage_detected: bool) {
        self.echo_remover
            .update_echo_leakage_status(leakage_detected);
    }

    /// Specifies whether the capture output will be used.
    pub(crate) fn set_capture_output_usage(&mut self, capture_output_used: bool) {
        self.echo_remover
            .set_capture_output_usage(capture_output_used);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::NUM_BLOCKS_PER_SECOND;

    #[test]
    fn basic_setup_and_api_calls() {
        for &rate in &[16000, 32000, 48000] {
            let config = EchoCanceller3Config::default();
            let mut block_processor = BlockProcessor::new(&config, rate, 1, 1);
            let block = Block::new_with_value(num_bands_for_rate(rate), 1, 1000.0);
            let mut capture = block.clone();
            for _ in 0..1 {
                block_processor.buffer_render(&block);
                block_processor.process_capture(false, false, None, &mut capture);
                block_processor.update_echo_leakage_status(false);
            }
        }
    }

    #[test]
    fn test_longer_call() {
        let config = EchoCanceller3Config::default();
        let mut block_processor = BlockProcessor::new(&config, 16000, 1, 1);
        let block = Block::new_with_value(1, 1, 1000.0);
        let mut capture = block.clone();
        for _ in 0..20 * NUM_BLOCKS_PER_SECOND {
            block_processor.buffer_render(&block);
            block_processor.process_capture(false, false, None, &mut capture);
            block_processor.update_echo_leakage_status(false);
        }
    }

    #[test]
    fn get_metrics_returns_values() {
        let config = EchoCanceller3Config::default();
        let block_processor = BlockProcessor::new(&config, 16000, 1, 1);
        let metrics = block_processor.get_metrics();
        // delay_ms should be non-negative and a multiple of block_size_ms (4).
        assert!(metrics.delay_ms >= 0);
        assert_eq!(metrics.delay_ms % 4, 0);
    }

    #[test]
    fn set_audio_buffer_delay() {
        let config = EchoCanceller3Config::default();
        let mut block_processor = BlockProcessor::new(&config, 16000, 1, 1);
        // Should not panic.
        block_processor.set_audio_buffer_delay(20);
    }

    #[test]
    fn set_capture_output_usage() {
        let config = EchoCanceller3Config::default();
        let mut block_processor = BlockProcessor::new(&config, 16000, 1, 1);
        block_processor.set_capture_output_usage(false);
        block_processor.set_capture_output_usage(true);
    }
}
