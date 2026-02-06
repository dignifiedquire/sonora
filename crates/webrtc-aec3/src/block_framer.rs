//! Block-to-frame conversion.
//!
//! Ported from `modules/audio_processing/aec3/block_framer.h/cc`.
//!
//! Converts 64-sample blocks back into 80-sample sub-frames. Works together
//! with `FrameBlocker` to produce output frames at the same rate as input
//! frames.

use crate::block::Block;
use crate::common::{BLOCK_SIZE, SUB_FRAME_LENGTH};

/// Produces 80-sample sub-frames from 64-sample blocks.
pub(crate) struct BlockFramer {
    num_bands: usize,
    num_channels: usize,
    /// `buffer[band][channel]` — residual samples from previous blocks.
    buffer: Vec<Vec<Vec<f32>>>,
}

impl BlockFramer {
    pub(crate) fn new(num_bands: usize, num_channels: usize) -> Self {
        debug_assert!(num_bands > 0);
        debug_assert!(num_channels > 0);
        Self {
            num_bands,
            num_channels,
            buffer: vec![vec![vec![0.0f32; BLOCK_SIZE]; num_channels]; num_bands],
        }
    }

    /// Adds a 64-sample block without extracting a sub-frame.
    pub(crate) fn insert_block(&mut self, block: &Block) {
        debug_assert_eq!(self.num_bands, block.num_bands());
        debug_assert_eq!(self.num_channels, block.num_channels());
        for band in 0..self.num_bands {
            for channel in 0..self.num_channels {
                debug_assert!(self.buffer[band][channel].is_empty());
                self.buffer[band][channel].clear();
                self.buffer[band][channel].extend_from_slice(block.view(band, channel));
            }
        }
    }

    /// Adds a 64-sample block and extracts an 80-sample sub-frame.
    ///
    /// `sub_frame` is indexed as `sub_frame[band][channel]`, each inner `Vec`
    /// has `SUB_FRAME_LENGTH` (80) samples.
    pub(crate) fn insert_block_and_extract_sub_frame(
        &mut self,
        block: &Block,
        sub_frame: &mut [Vec<Vec<f32>>],
    ) {
        debug_assert_eq!(self.num_bands, block.num_bands());
        debug_assert_eq!(self.num_channels, block.num_channels());
        debug_assert_eq!(self.num_bands, sub_frame.len());
        for band in 0..self.num_bands {
            debug_assert_eq!(self.num_channels, sub_frame[band].len());
            for channel in 0..self.num_channels {
                let buf_len = self.buffer[band][channel].len();
                debug_assert!(SUB_FRAME_LENGTH <= buf_len + BLOCK_SIZE);
                debug_assert!(buf_len <= BLOCK_SIZE);
                debug_assert_eq!(SUB_FRAME_LENGTH, sub_frame[band][channel].len());

                let samples_to_frame = SUB_FRAME_LENGTH - buf_len;
                let out = &mut sub_frame[band][channel];

                // Copy buffered samples first.
                out[..buf_len].copy_from_slice(&self.buffer[band][channel]);
                // Fill the rest from the block.
                out[buf_len..SUB_FRAME_LENGTH]
                    .copy_from_slice(&block.view(band, channel)[..samples_to_frame]);

                // Store remainder in buffer.
                self.buffer[band][channel].clear();
                self.buffer[band][channel]
                    .extend_from_slice(&block.view(band, channel)[samples_to_frame..]);
            }
        }
    }
}
