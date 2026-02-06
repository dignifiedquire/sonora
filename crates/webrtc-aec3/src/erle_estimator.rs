//! ERLE (Echo Return Loss Enhancement) estimator coordinator.
//!
//! Combines fullband, subband, and optionally signal-dependent ERLE estimators.
//!
//! Ported from `modules/audio_processing/aec3/erle_estimator.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::config::EchoCanceller3Config;
use crate::fullband_erle_estimator::FullBandErleEstimator;
use crate::render_buffer::RenderBuffer;
use crate::signal_dependent_erle_estimator::SignalDependentErleEstimator;
use crate::subband_erle_estimator::SubbandErleEstimator;

/// Estimates the echo return loss enhancement. One estimate is done per subband
/// and another one is done using the aggregation of energy over all subbands.
pub(crate) struct ErleEstimator {
    startup_phase_length_blocks: usize,
    fullband_erle_estimator: FullBandErleEstimator,
    subband_erle_estimator: SubbandErleEstimator,
    signal_dependent_erle_estimator: Option<SignalDependentErleEstimator>,
    blocks_since_reset: usize,
}

impl ErleEstimator {
    pub(crate) fn new(
        startup_phase_length_blocks: usize,
        config: &EchoCanceller3Config,
        num_capture_channels: usize,
    ) -> Self {
        let signal_dependent = if config.erle.num_sections > 1 {
            Some(SignalDependentErleEstimator::new(
                config,
                num_capture_channels,
            ))
        } else {
            None
        };

        let mut s = Self {
            startup_phase_length_blocks,
            fullband_erle_estimator: FullBandErleEstimator::new(&config.erle, num_capture_channels),
            subband_erle_estimator: SubbandErleEstimator::new(config, num_capture_channels),
            signal_dependent_erle_estimator: signal_dependent,
            blocks_since_reset: 0,
        };
        s.reset(true);
        s
    }

    /// Resets the fullband ERLE estimator and the subband ERLE estimators.
    pub(crate) fn reset(&mut self, delay_change: bool) {
        self.fullband_erle_estimator.reset();
        self.subband_erle_estimator.reset();
        if let Some(ref mut sd) = self.signal_dependent_erle_estimator {
            sd.reset();
        }
        if delay_change {
            self.blocks_since_reset = 0;
        }
    }

    /// Updates the ERLE estimates.
    #[allow(clippy::too_many_arguments, reason = "mirrors C++ API")]
    pub(crate) fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        avg_render_spectrum_with_reverb: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        capture_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        subtractor_spectra: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        let x2_reverb = avg_render_spectrum_with_reverb;
        let y2 = capture_spectra;
        let e2 = subtractor_spectra;

        self.blocks_since_reset += 1;
        if self.blocks_since_reset < self.startup_phase_length_blocks {
            return;
        }

        self.subband_erle_estimator
            .update(x2_reverb, y2, e2, converged_filters);

        if let Some(ref mut sd) = self.signal_dependent_erle_estimator {
            sd.update(
                render_buffer,
                filter_frequency_responses,
                x2_reverb,
                y2,
                e2,
                self.subband_erle_estimator.erle(false),
                self.subband_erle_estimator.erle(true),
                converged_filters,
            );
        }

        self.fullband_erle_estimator
            .update(x2_reverb, y2, e2, converged_filters);
    }

    /// Returns the most recent subband ERLE estimates.
    pub(crate) fn erle(&self, onset_compensated: bool) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if let Some(ref sd) = self.signal_dependent_erle_estimator {
            sd.erle(onset_compensated)
        } else {
            self.subband_erle_estimator.erle(onset_compensated)
        }
    }

    /// Returns the non-capped subband ERLE.
    pub(crate) fn erle_unbounded(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if self.signal_dependent_erle_estimator.is_none() {
            self.subband_erle_estimator.erle_unbounded()
        } else {
            self.signal_dependent_erle_estimator
                .as_ref()
                .unwrap()
                .erle(false)
        }
    }

    /// Returns the subband ERLE estimated during onsets (only used for testing).
    pub(crate) fn erle_during_onsets(&self) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        self.subband_erle_estimator.erle_during_onsets()
    }

    /// Returns the fullband ERLE estimate in log2 units.
    pub(crate) fn fullband_erle_log2(&self) -> f32 {
        self.fullband_erle_estimator.fullband_erle_log2()
    }

    /// Returns an estimation of the current linear filter quality.
    pub(crate) fn get_inst_linear_quality_estimates(&self) -> &[Option<f32>] {
        self.fullband_erle_estimator
            .get_inst_linear_quality_estimates()
    }
}
