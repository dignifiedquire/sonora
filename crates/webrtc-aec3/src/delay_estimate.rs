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
#[derive(Debug, Clone)]
pub(crate) struct DelayEstimate {
    pub quality: DelayEstimateQuality,
    pub delay: usize,
    pub blocks_since_last_change: usize,
    pub blocks_since_last_update: usize,
}

impl DelayEstimate {
    pub(crate) fn new(quality: DelayEstimateQuality, delay: usize) -> Self {
        Self {
            quality,
            delay,
            blocks_since_last_change: 0,
            blocks_since_last_update: 0,
        }
    }
}
