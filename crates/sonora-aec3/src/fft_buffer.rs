//! Circular buffer of `FftData` objects.
//!
//! Ported from `modules/audio_processing/aec3/fft_buffer.h/cc`.

use crate::circular_buffer::RingIndex;
use crate::fft_data::FftData;

/// Circular buffer of per-channel `FftData` with read/write indices.
///
/// Each slot holds `num_channels` `FftData` structs.
pub(crate) struct FftBuffer {
    pub index: RingIndex,
    /// `buffer[slot][channel]` — FFT data.
    pub buffer: Vec<Vec<FftData>>,
}

impl FftBuffer {
    pub(crate) fn new(size: usize, num_channels: usize) -> Self {
        Self {
            index: RingIndex::new(size),
            buffer: vec![vec![FftData::default(); num_channels]; size],
        }
    }
}
