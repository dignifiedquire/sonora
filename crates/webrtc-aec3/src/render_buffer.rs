//! Render buffer providing access to block, spectrum, and FFT data.
//!
//! Ported from `modules/audio_processing/aec3/render_buffer.h/cc`.

use crate::block::Block;
use crate::block_buffer::BlockBuffer;
use crate::common::FFT_LENGTH_BY_2_PLUS_1;
use crate::fft_buffer::FftBuffer;
use crate::fft_data::FftData;
use crate::spectrum_buffer::SpectrumBuffer;

/// Provides a view into the render data buffers for the echo remover.
///
/// The C++ version takes raw pointers to externally owned buffers. In Rust,
/// we use lifetime-annotated references.
pub(crate) struct RenderBuffer<'a> {
    block_buffer: &'a BlockBuffer,
    spectrum_buffer: &'a SpectrumBuffer,
    fft_buffer: &'a FftBuffer,
    render_activity: bool,
}

impl<'a> RenderBuffer<'a> {
    pub(crate) fn new(
        block_buffer: &'a BlockBuffer,
        spectrum_buffer: &'a SpectrumBuffer,
        fft_buffer: &'a FftBuffer,
    ) -> Self {
        debug_assert_eq!(block_buffer.buffer.len(), fft_buffer.buffer.len());
        debug_assert_eq!(spectrum_buffer.buffer.len(), fft_buffer.buffer.len());
        debug_assert_eq!(spectrum_buffer.index.read, fft_buffer.index.read);
        debug_assert_eq!(spectrum_buffer.index.write, fft_buffer.index.write);
        Self {
            block_buffer,
            spectrum_buffer,
            fft_buffer,
            render_activity: false,
        }
    }

    /// Get a block at the given offset from the read position.
    pub(crate) fn get_block(&self, buffer_offset_blocks: i32) -> &Block {
        let position = self
            .block_buffer
            .index
            .offset_index(self.block_buffer.index.read, buffer_offset_blocks);
        &self.block_buffer.buffer[position]
    }

    /// Get the per-channel spectra at the given offset from the read position.
    pub(crate) fn spectrum(&self, buffer_offset_ffts: i32) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        let position = self
            .spectrum_buffer
            .index
            .offset_index(self.spectrum_buffer.index.read, buffer_offset_ffts);
        &self.spectrum_buffer.buffer[position]
    }

    /// Returns the circular FFT buffer.
    pub(crate) fn get_fft_buffer(&self) -> &[Vec<FftData>] {
        &self.fft_buffer.buffer
    }

    /// Returns the current read position in the circular buffer.
    pub(crate) fn position(&self) -> usize {
        debug_assert_eq!(self.spectrum_buffer.index.read, self.fft_buffer.index.read);
        debug_assert_eq!(
            self.spectrum_buffer.index.write,
            self.fft_buffer.index.write
        );
        self.fft_buffer.index.read
    }

    /// Computes the sum of spectra for `num_spectra` consecutive FFTs
    /// starting from the read position.
    pub(crate) fn spectral_sum(&self, num_spectra: usize, x2: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        x2.fill(0.0);
        let mut position = self.spectrum_buffer.index.read;
        for _ in 0..num_spectra {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (out, &val) in x2.iter_mut().zip(channel_spectrum.iter()) {
                    *out += val;
                }
            }
            position = self.spectrum_buffer.index.inc_index(position);
        }
    }

    /// Computes the sums of spectra for two different FFT counts.
    ///
    /// `num_spectra_shorter` must be <= `num_spectra_longer`. The shorter sum
    /// is computed first, then the longer sum continues from where the shorter
    /// left off (avoiding redundant accumulation).
    pub(crate) fn spectral_sums(
        &self,
        num_spectra_shorter: usize,
        num_spectra_longer: usize,
        x2_shorter: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        x2_longer: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        debug_assert!(num_spectra_shorter <= num_spectra_longer);
        x2_shorter.fill(0.0);
        let mut position = self.spectrum_buffer.index.read;

        // Accumulate for the shorter range.
        for _ in 0..num_spectra_shorter {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (out, &val) in x2_shorter.iter_mut().zip(channel_spectrum.iter()) {
                    *out += val;
                }
            }
            position = self.spectrum_buffer.index.inc_index(position);
        }

        // Copy shorter into longer, then continue accumulating.
        x2_longer.copy_from_slice(x2_shorter);
        for _ in num_spectra_shorter..num_spectra_longer {
            for channel_spectrum in &self.spectrum_buffer.buffer[position] {
                for (out, &val) in x2_longer.iter_mut().zip(channel_spectrum.iter()) {
                    *out += val;
                }
            }
            position = self.spectrum_buffer.index.inc_index(position);
        }
    }

    /// Gets the recent activity seen in the render signal.
    pub(crate) fn get_render_activity(&self) -> bool {
        self.render_activity
    }

    /// Specifies the recent activity seen in the render signal.
    pub(crate) fn set_render_activity(&mut self, activity: bool) {
        self.render_activity = activity;
    }

    /// Returns the headroom between the write and read positions in the buffer.
    pub(crate) fn headroom(&self) -> usize {
        let size = self.fft_buffer.index.size;
        let write = self.fft_buffer.index.write;
        let read = self.fft_buffer.index.read;
        let headroom = if write < read {
            read - write
        } else {
            size - write + read
        };
        debug_assert!(headroom <= size);
        headroom
    }

    /// Returns a reference to the spectrum buffer.
    pub(crate) fn get_spectrum_buffer(&self) -> &SpectrumBuffer {
        self.spectrum_buffer
    }

    /// Returns a reference to the block buffer.
    pub(crate) fn get_block_buffer(&self) -> &BlockBuffer {
        self.block_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_sum_accumulates_all_channels() {
        let size = 4;
        let num_channels = 2;
        let mut spectrum_buffer = SpectrumBuffer::new(size, num_channels);
        let block_buffer = BlockBuffer::new(size, 1, num_channels);
        let fft_buffer = FftBuffer::new(size, num_channels);

        // Fill first two slots with known values.
        for ch in 0..num_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                spectrum_buffer.buffer[0][ch][k] = 1.0;
                spectrum_buffer.buffer[1][ch][k] = 2.0;
            }
        }

        let rb = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);
        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        rb.spectral_sum(2, &mut x2);

        // Each slot has 2 channels, each with value 1.0 or 2.0.
        // Sum = 2*1.0 + 2*2.0 = 6.0 per bin.
        for &v in &x2 {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn spectral_sums_shorter_is_prefix_of_longer() {
        let size = 4;
        let num_channels = 1;
        let mut spectrum_buffer = SpectrumBuffer::new(size, num_channels);
        let block_buffer = BlockBuffer::new(size, 1, num_channels);
        let fft_buffer = FftBuffer::new(size, num_channels);

        for slot in 0..3 {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                spectrum_buffer.buffer[slot][0][k] = (slot + 1) as f32;
            }
        }

        let rb = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);
        let mut x2_short = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut x2_long = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        rb.spectral_sums(1, 3, &mut x2_short, &mut x2_long);

        // Short: slot 0 only → 1.0.
        for &v in &x2_short {
            assert!((v - 1.0).abs() < 1e-6);
        }
        // Long: slots 0+1+2 → 1+2+3 = 6.0.
        for &v in &x2_long {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    /// Helper to set matching indices on all three buffers.
    fn set_indices(
        block_buffer: &mut BlockBuffer,
        spectrum_buffer: &mut SpectrumBuffer,
        fft_buffer: &mut FftBuffer,
        write: usize,
        read: usize,
    ) {
        block_buffer.index.write = write;
        block_buffer.index.read = read;
        spectrum_buffer.index.write = write;
        spectrum_buffer.index.read = read;
        fft_buffer.index.write = write;
        fft_buffer.index.read = read;
    }

    #[test]
    fn headroom_calculation() {
        let size = 10;
        let num_channels = 1;
        let mut block_buffer = BlockBuffer::new(size, 1, num_channels);
        let mut spectrum_buffer = SpectrumBuffer::new(size, num_channels);
        let mut fft_buffer = FftBuffer::new(size, num_channels);

        // write=0, read=0 → headroom = size - 0 + 0 = size
        set_indices(
            &mut block_buffer,
            &mut spectrum_buffer,
            &mut fft_buffer,
            0,
            0,
        );
        let rb = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);
        assert_eq!(rb.headroom(), size);

        // write=3, read=7 → headroom = 7 - 3 = 4
        set_indices(
            &mut block_buffer,
            &mut spectrum_buffer,
            &mut fft_buffer,
            3,
            7,
        );
        let rb = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);
        assert_eq!(rb.headroom(), 4);

        // write=7, read=3 → headroom = size - 7 + 3 = 6
        set_indices(
            &mut block_buffer,
            &mut spectrum_buffer,
            &mut fft_buffer,
            7,
            3,
        );
        let rb = RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);
        assert_eq!(rb.headroom(), 6);
    }
}
