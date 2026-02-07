//! Subtractor output analyzer.
//!
//! Analyzes the properties of the subtractor output to determine filter
//! convergence and divergence status.
//!
//! Ported from
//! `modules/audio_processing/aec3/subtractor_output_analyzer.h/cc`.

use crate::common::BLOCK_SIZE;
use crate::subtractor_output::SubtractorOutput;

/// Analyzes the subtractor output for convergence/divergence.
pub(crate) struct SubtractorOutputAnalyzer {
    filters_converged: Vec<bool>,
}

impl SubtractorOutputAnalyzer {
    pub(crate) fn new(num_capture_channels: usize) -> Self {
        Self {
            filters_converged: vec![false; num_capture_channels],
        }
    }

    /// Analyzes the subtractor output and updates convergence flags.
    ///
    /// Returns (any_filter_converged, any_coarse_filter_converged,
    /// all_filters_diverged).
    pub(crate) fn update(&mut self, subtractor_output: &[SubtractorOutput]) -> (bool, bool, bool) {
        debug_assert_eq!(subtractor_output.len(), self.filters_converged.len());

        let mut any_filter_converged = false;
        let mut any_coarse_filter_converged = false;
        let mut all_filters_diverged = true;

        for (ch, output) in subtractor_output.iter().enumerate() {
            let y2 = output.y2;
            let e2_refined = output.e2_refined_sum;
            let e2_coarse = output.e2_coarse_sum;

            const CONVERGENCE_THRESHOLD: f32 = 50.0 * 50.0 * BLOCK_SIZE as f32;
            const CONVERGENCE_THRESHOLD_LOW_LEVEL: f32 = 20.0 * 20.0 * BLOCK_SIZE as f32;

            let refined_filter_converged = e2_refined < 0.5 * y2 && y2 > CONVERGENCE_THRESHOLD;
            let coarse_filter_converged_strict =
                e2_coarse < 0.05 * y2 && y2 > CONVERGENCE_THRESHOLD;
            let coarse_filter_converged_relaxed =
                e2_coarse < 0.3 * y2 && y2 > CONVERGENCE_THRESHOLD_LOW_LEVEL;
            let min_e2 = e2_refined.min(e2_coarse);
            let filter_diverged = min_e2 > 1.5 * y2 && y2 > 30.0 * 30.0 * BLOCK_SIZE as f32;

            self.filters_converged[ch] = refined_filter_converged || coarse_filter_converged_strict;

            any_filter_converged = any_filter_converged || self.filters_converged[ch];
            any_coarse_filter_converged =
                any_coarse_filter_converged || coarse_filter_converged_relaxed;
            all_filters_diverged = all_filters_diverged && filter_diverged;
        }

        (
            any_filter_converged,
            any_coarse_filter_converged,
            all_filters_diverged,
        )
    }

    pub(crate) fn converged_filters(&self) -> &[bool] {
        &self.filters_converged
    }

    /// Handle echo path change.
    pub(crate) fn handle_echo_path_change(&mut self) {
        self.filters_converged.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_not_converged() {
        let analyzer = SubtractorOutputAnalyzer::new(2);
        assert!(analyzer.converged_filters().iter().all(|&c| !c));
    }

    #[test]
    fn convergence_detection() {
        let mut analyzer = SubtractorOutputAnalyzer::new(1);
        let mut output = SubtractorOutput::default();

        // Set y2 high, e2_refined low → converged.
        output.y2 = 200_000.0;
        output.e2_refined_sum = 10_000.0; // < 0.5 * y2
        output.e2_coarse_sum = 200_000.0;

        let (any_converged, _, _) = analyzer.update(&[output]);
        assert!(any_converged);
        assert!(analyzer.converged_filters()[0]);
    }

    #[test]
    fn divergence_detection() {
        let mut analyzer = SubtractorOutputAnalyzer::new(1);
        let mut output = SubtractorOutput::default();

        // min(e2_refined, e2_coarse) > 1.5 * y2, and y2 > threshold.
        output.y2 = 100_000.0;
        output.e2_refined_sum = 200_000.0;
        output.e2_coarse_sum = 200_000.0;

        let (_, _, all_diverged) = analyzer.update(&[output]);
        assert!(all_diverged);
    }

    #[test]
    fn handle_echo_path_change_resets() {
        let mut analyzer = SubtractorOutputAnalyzer::new(2);
        let mut output = SubtractorOutput::default();
        output.y2 = 200_000.0;
        output.e2_refined_sum = 10_000.0;
        output.e2_coarse_sum = 200_000.0;

        analyzer.update(&[output, SubtractorOutput::default()]);
        assert!(analyzer.converged_filters()[0]);

        analyzer.handle_echo_path_change();
        assert!(analyzer.converged_filters().iter().all(|&c| !c));
    }
}
