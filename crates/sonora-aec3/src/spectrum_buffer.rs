//! Circular buffer of power spectra.
//!
//! Ported from `modules/audio_processing/aec3/spectrum_buffer.h/cc`.

use crate::circular_buffer::RingIndex;
use crate::common::FFT_LENGTH_BY_2_PLUS_1;

/// Circular buffer of per-channel power spectra with read/write indices.
///
/// Each slot holds `num_channels` spectra, each `FFT_LENGTH_BY_2_PLUS_1` bins.
pub(crate) struct SpectrumBuffer {
    pub index: RingIndex,
    /// `buffer[slot][channel]` — power spectrum array.
    pub buffer: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
}

impl SpectrumBuffer {
    pub(crate) fn new(size: usize, num_channels: usize) -> Self {
        Self {
            index: RingIndex::new(size),
            buffer: vec![vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_channels]; size],
        }
    }
}
