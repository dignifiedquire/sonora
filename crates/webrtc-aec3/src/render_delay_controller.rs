//! Render delay controller — aligns render and capture signals.
//!
//! The C++ version is an abstract class with a factory. We port the single
//! concrete implementation directly.
//!
//! Ported from `modules/audio_processing/aec3/render_delay_controller.h/cc`.

use crate::block::Block;
use crate::clockdrift_detector::ClockdriftLevel;
use crate::common::{BLOCK_SIZE_LOG2, NUM_BLOCKS_PER_SECOND};
use crate::config::EchoCanceller3Config;
use crate::delay_estimate::{DelayEstimate, DelayEstimateQuality};
use crate::downsampled_render_buffer::DownsampledRenderBuffer;
use crate::echo_path_delay_estimator::EchoPathDelayEstimator;
use crate::render_delay_controller_metrics::RenderDelayControllerMetrics;
use webrtc_simd::SimdBackend;

/// Computes the buffer delay from the estimated delay in samples.
fn compute_buffer_delay(
    current_delay: &Option<DelayEstimate>,
    hysteresis_limit_blocks: i32,
    estimated_delay: DelayEstimate,
) -> DelayEstimate {
    // Compute the buffer delay increase required to achieve the desired latency.
    let mut new_delay_blocks = estimated_delay.delay >> BLOCK_SIZE_LOG2;

    // Add hysteresis.
    if let Some(current) = current_delay {
        let current_delay_blocks = current.delay;
        if new_delay_blocks > current_delay_blocks
            && new_delay_blocks <= current_delay_blocks + hysteresis_limit_blocks as usize
        {
            new_delay_blocks = current_delay_blocks;
        }
    }

    DelayEstimate {
        delay: new_delay_blocks,
        ..estimated_delay
    }
}

/// Aligns render and capture signals using delay estimation.
pub(crate) struct RenderDelayController {
    hysteresis_limit_blocks: i32,
    delay: Option<DelayEstimate>,
    delay_estimator: EchoPathDelayEstimator,
    metrics: RenderDelayControllerMetrics,
    delay_samples: Option<DelayEstimate>,
    capture_call_counter: usize,
    delay_change_counter: i32,
    last_delay_estimate_quality: DelayEstimateQuality,
}

impl RenderDelayController {
    pub(crate) fn new(
        backend: SimdBackend,
        config: &EchoCanceller3Config,
        num_capture_channels: usize,
    ) -> Self {
        Self {
            hysteresis_limit_blocks: config.delay.hysteresis_limit_blocks as i32,
            delay: None,
            delay_estimator: EchoPathDelayEstimator::new(backend, config, num_capture_channels),
            metrics: RenderDelayControllerMetrics::new(),
            delay_samples: None,
            capture_call_counter: 0,
            delay_change_counter: 0,
            last_delay_estimate_quality: DelayEstimateQuality::Coarse,
        }
    }

    /// Resets the delay controller. If `reset_delay_confidence` is true, the
    /// reset behavior is as if the call is restarted.
    pub(crate) fn reset(&mut self, reset_delay_confidence: bool) {
        self.delay = None;
        self.delay_samples = None;
        self.delay_estimator.reset(reset_delay_confidence);
        self.delay_change_counter = 0;
        if reset_delay_confidence {
            self.last_delay_estimate_quality = DelayEstimateQuality::Coarse;
        }
    }

    /// Logs a render call (no-op in production).
    pub(crate) fn log_render_call(&self) {}

    /// Aligns the render buffer content with the capture signal.
    pub(crate) fn get_delay(
        &mut self,
        render_buffer: &DownsampledRenderBuffer,
        _render_delay_buffer_delay: usize,
        capture: &Block,
    ) -> Option<DelayEstimate> {
        self.capture_call_counter += 1;

        let delay_samples = self.delay_estimator.estimate_delay(render_buffer, capture);

        if let Some(new_est) = delay_samples {
            if self.delay_samples.is_none()
                || self
                    .delay_samples
                    .as_ref()
                    .is_some_and(|d| d.delay != new_est.delay)
            {
                self.delay_change_counter = 0;
            }
            if let Some(existing) = &mut self.delay_samples {
                existing.blocks_since_last_change = if existing.delay == new_est.delay {
                    existing.blocks_since_last_change + 1
                } else {
                    0
                };
                existing.blocks_since_last_update = 0;
                existing.delay = new_est.delay;
                existing.quality = new_est.quality;
            } else {
                self.delay_samples = Some(new_est);
            }
        } else if let Some(existing) = &mut self.delay_samples {
            existing.blocks_since_last_change += 1;
            existing.blocks_since_last_update += 1;
        }

        if self.delay_change_counter < 2 * NUM_BLOCKS_PER_SECOND as i32 {
            self.delay_change_counter += 1;
        }

        if let Some(ds) = &self.delay_samples {
            // Compute the render delay buffer delay.
            let use_hysteresis = self.last_delay_estimate_quality == DelayEstimateQuality::Refined
                && ds.quality == DelayEstimateQuality::Refined;
            self.delay = Some(compute_buffer_delay(
                &self.delay,
                if use_hysteresis {
                    self.hysteresis_limit_blocks
                } else {
                    0
                },
                *ds,
            ));
            self.last_delay_estimate_quality = ds.quality;
        }

        self.metrics.update(
            self.delay_samples.as_ref().map(|d| d.delay),
            self.delay.as_ref().map(|d| d.delay),
            self.delay_estimator.clockdrift(),
        );

        self.delay
    }

    /// Returns true if clock drift has been detected.
    pub(crate) fn has_clockdrift(&self) -> bool {
        self.delay_estimator.clockdrift() != ClockdriftLevel::None
    }
}
