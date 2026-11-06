//! Delay estimate data structure.
//!
//! Ported from `modules/audio_processing/aec3/delay_estimate.h`.

/// Quality level of a delay estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DelayEstimateQuality {
    Coarse,
    Refined,
}

/// Stores a delay estimate with associated metadata.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DelayEstimate {
    pub quality: DelayEstimateQuality,
    pub delay: usize,
}

impl DelayEstimate {
    pub(crate) fn new(quality: DelayEstimateQuality, delay: usize) -> Self {
        Self { quality, delay }
    }
}
