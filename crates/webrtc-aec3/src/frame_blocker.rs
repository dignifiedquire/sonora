//! Frame-to-block conversion.
//!
//! Ported from `modules/audio_processing/aec3/frame_blocker.h/cc`.
//!
//! Converts 80-sample sub-frames into 64-sample blocks. The internal buffer
//! accumulates residual samples across calls. Every 4th sub-frame insertion
//! produces an extra block that can be extracted with `extract_block`.

use crate::block::Block;
use crate::common::{BLOCK_SIZE, SUB_FRAME_LENGTH};

/// Produces 64-sample blocks from 80-sample sub-frames.
pub struct FrameBlocker {
    num_bands: usize,
    num_channels: usize,
    /// `buffer[band][channel]` — residual samples not yet output.
    buffer: Vec<Vec<Vec<f32>>>,
}

impl FrameBlocker {
    pub fn new(num_bands: usize, num_channels: usize) -> Self {
        debug_assert!(num_bands > 0);
        debug_assert!(num_channels > 0);
        Self {
            num_bands,
            num_channels,
            buffer: vec![vec![Vec::with_capacity(BLOCK_SIZE); num_channels]; num_bands],
        }
    }

    /// Inserts one 80-sample sub-frame and extracts one 64-sample block.
    ///
    /// `sub_frame` is indexed as `sub_frame[band][channel]`, where each inner
    /// slice has `SUB_FRAME_LENGTH` (80) samples.
    pub fn insert_sub_frame_and_extract_block(
        &mut self,
        sub_frame: &[Vec<&[f32]>],
        block: &mut Block,
    ) {
        debug_assert_eq!(self.num_bands, block.num_bands());
        debug_assert_eq!(self.num_bands, sub_frame.len());
        for band in 0..self.num_bands {
            debug_assert_eq!(self.num_channels, block.num_channels());
            debug_assert_eq!(self.num_channels, sub_frame[band].len());
            for channel in 0..self.num_channels {
                debug_assert!(self.buffer[band][channel].len() <= BLOCK_SIZE - 16);
                debug_assert_eq!(SUB_FRAME_LENGTH, sub_frame[band][channel].len());

                let buf = &self.buffer[band][channel];
                let buf_len = buf.len();
                let samples_to_block = BLOCK_SIZE - buf_len;
                let out = block.view_mut(band, channel);

                // Copy buffered samples first.
                out[..buf_len].copy_from_slice(buf);
                // Fill the rest from the sub-frame.
                out[buf_len..].copy_from_slice(&sub_frame[band][channel][..samples_to_block]);

                // Store remainder in buffer.
                self.buffer[band][channel].clear();
                self.buffer[band][channel]
                    .extend_from_slice(&sub_frame[band][channel][samples_to_block..]);
            }
        }
    }

    /// Returns `true` if a full 64-sample block is available for extraction.
    pub fn is_block_available(&self) -> bool {
        self.buffer[0][0].len() == BLOCK_SIZE
    }

    /// Extracts a buffered 64-sample block (only valid when `is_block_available`
    /// returns `true`).
    pub fn extract_block(&mut self, block: &mut Block) {
        debug_assert_eq!(self.num_bands, block.num_bands());
        debug_assert_eq!(self.num_channels, block.num_channels());
        debug_assert!(self.is_block_available());
        for band in 0..self.num_bands {
            for channel in 0..self.num_channels {
                debug_assert_eq!(BLOCK_SIZE, self.buffer[band][channel].len());
                block
                    .view_mut(band, channel)
                    .copy_from_slice(&self.buffer[band][channel]);
                self.buffer[band][channel].clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_framer::BlockFramer;
    use crate::common::num_bands_for_rate;

    fn compute_sample_value(
        chunk_counter: usize,
        chunk_size: usize,
        band: usize,
        channel: usize,
        sample_index: usize,
        offset: i32,
    ) -> f32 {
        let value = (chunk_counter * chunk_size + sample_index + channel) as i32 + offset;
        if value > 0 {
            5000.0 * band as f32 + value as f32
        } else {
            0.0
        }
    }

    fn fill_sub_frame(sub_frame_counter: usize, offset: i32, sub_frame: &mut [Vec<Vec<f32>>]) {
        for (band, band_data) in sub_frame.iter_mut().enumerate() {
            for (channel, channel_data) in band_data.iter_mut().enumerate() {
                for (sample, v) in channel_data.iter_mut().enumerate() {
                    *v = compute_sample_value(
                        sub_frame_counter,
                        SUB_FRAME_LENGTH,
                        band,
                        channel,
                        sample,
                        offset,
                    );
                }
            }
        }
    }

    fn make_sub_frame_view(sub_frame: &[Vec<Vec<f32>>]) -> Vec<Vec<&[f32]>> {
        sub_frame
            .iter()
            .map(|band| band.iter().map(|ch| ch.as_slice()).collect())
            .collect()
    }

    fn verify_block(block_counter: usize, offset: i32, block: &Block) -> bool {
        for band in 0..block.num_bands() {
            for channel in 0..block.num_channels() {
                let view = block.view(band, channel);
                for sample in 0..BLOCK_SIZE {
                    let expected = compute_sample_value(
                        block_counter,
                        BLOCK_SIZE,
                        band,
                        channel,
                        sample,
                        offset,
                    );
                    if expected != view[sample] {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn verify_sub_frame(
        sub_frame_counter: usize,
        offset: i32,
        sub_frame: &[Vec<Vec<f32>>],
    ) -> bool {
        let num_bands = sub_frame.len();
        let num_channels = sub_frame[0].len();
        let len = sub_frame[0][0].len();
        let mut reference = vec![vec![vec![0.0f32; len]; num_channels]; num_bands];
        fill_sub_frame(sub_frame_counter, offset, &mut reference);
        for band in 0..num_bands {
            for channel in 0..num_channels {
                for sample in 0..len {
                    if reference[band][channel][sample] != sub_frame[band][channel][sample] {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn run_blocker_test(sample_rate_hz: usize, num_channels: usize) {
        const NUM_SUB_FRAMES: usize = 20;
        let num_bands = num_bands_for_rate(sample_rate_hz);

        let mut block = Block::new(num_bands, num_channels);
        let mut input_sub_frame =
            vec![vec![vec![0.0f32; SUB_FRAME_LENGTH]; num_channels]; num_bands];
        let mut blocker = FrameBlocker::new(num_bands, num_channels);

        let mut block_counter = 0;
        for sub_frame_index in 0..NUM_SUB_FRAMES {
            fill_sub_frame(sub_frame_index, 0, &mut input_sub_frame);
            let view = make_sub_frame_view(&input_sub_frame);

            blocker.insert_sub_frame_and_extract_block(&view, &mut block);
            assert!(
                verify_block(block_counter, 0, &block),
                "block {block_counter} mismatch"
            );
            block_counter += 1;

            if (sub_frame_index + 1) % 4 == 0 {
                assert!(blocker.is_block_available());
            } else {
                assert!(!blocker.is_block_available());
            }
            if blocker.is_block_available() {
                blocker.extract_block(&mut block);
                assert!(
                    verify_block(block_counter, 0, &block),
                    "extra block {block_counter} mismatch"
                );
                block_counter += 1;
            }
        }
    }

    fn run_blocker_and_framer_test(sample_rate_hz: usize, num_channels: usize) {
        const NUM_SUB_FRAMES: usize = 20;
        let num_bands = num_bands_for_rate(sample_rate_hz);

        let mut block = Block::new(num_bands, num_channels);
        let mut input_sub_frame =
            vec![vec![vec![0.0f32; SUB_FRAME_LENGTH]; num_channels]; num_bands];
        let mut output_sub_frame =
            vec![vec![vec![0.0f32; SUB_FRAME_LENGTH]; num_channels]; num_bands];
        let mut blocker = FrameBlocker::new(num_bands, num_channels);
        let mut framer = BlockFramer::new(num_bands, num_channels);

        for sub_frame_index in 0..NUM_SUB_FRAMES {
            fill_sub_frame(sub_frame_index, 0, &mut input_sub_frame);
            let view = make_sub_frame_view(&input_sub_frame);

            blocker.insert_sub_frame_and_extract_block(&view, &mut block);
            framer.insert_block_and_extract_sub_frame(&block, &mut output_sub_frame);

            if (sub_frame_index + 1) % 4 == 0 {
                assert!(blocker.is_block_available());
            } else {
                assert!(!blocker.is_block_available());
            }
            if blocker.is_block_available() {
                blocker.extract_block(&mut block);
                framer.insert_block(&block);
            }
            if sub_frame_index > 1 {
                assert!(
                    verify_sub_frame(sub_frame_index, -64, &output_sub_frame),
                    "sub_frame {sub_frame_index} mismatch"
                );
            }
        }
    }

    #[test]
    fn block_bitexactness() {
        for rate in [16000, 32000, 48000] {
            for num_channels in [1, 2, 4, 8] {
                run_blocker_test(rate, num_channels);
            }
        }
    }

    #[test]
    fn blocker_and_framer() {
        for rate in [16000, 32000, 48000] {
            for num_channels in [1, 2, 4, 8] {
                run_blocker_and_framer_test(rate, num_channels);
            }
        }
    }
}
