//! Render signal analyzer — analyzes the render signal for narrowband content
//! and excitation levels.
//!
//! This is a forward declaration / stub for the full implementation in Step 16.
//! Only the methods needed by CoarseFilterUpdateGain and RefinedFilterUpdateGain
//! are provided here.
//!
//! Ported from
//! `modules/audio_processing/aec3/render_signal_analyzer.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;

/// Analyzes the render signal for narrowband content and excitation levels.
pub(crate) struct RenderSignalAnalyzer {
    poor_signal_excitation: bool,
    narrow_band_regions: Vec<(usize, usize)>,
}

impl Default for RenderSignalAnalyzer {
    fn default() -> Self {
        Self {
            poor_signal_excitation: false,
            narrow_band_regions: Vec::new(),
        }
    }
}

impl RenderSignalAnalyzer {
    /// Returns true if the render signal has poor excitation.
    pub(crate) fn poor_signal_excitation(&self) -> bool {
        self.poor_signal_excitation
    }

    /// Sets the poor signal excitation flag.
    pub(crate) fn set_poor_signal_excitation(&mut self, poor: bool) {
        self.poor_signal_excitation = poor;
    }

    /// Masks regions around detected narrow bands by zeroing `v`.
    pub(crate) fn mask_regions_around_narrow_bands(&self, v: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        for &(start, end) in &self.narrow_band_regions {
            for k in start..=end.min(FFT_LENGTH_BY_2_PLUS_1 - 1) {
                v[k] = 0.0;
            }
        }
    }
}
