//! Circular buffer of `Block` objects.
//!
//! Ported from `modules/audio_processing/aec3/block_buffer.h/cc`.

use crate::block::Block;
use crate::circular_buffer::RingIndex;

/// Circular buffer of `Block` objects with read/write indices.
pub(crate) struct BlockBuffer {
    pub index: RingIndex,
    pub buffer: Vec<Block>,
}

impl BlockBuffer {
    pub(crate) fn new(size: usize, num_bands: usize, num_channels: usize) -> Self {
        Self {
            index: RingIndex::new(size),
            buffer: (0..size)
                .map(|_| Block::new(num_bands, num_channels))
                .collect(),
        }
    }
}
