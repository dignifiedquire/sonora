//! Circular buffer holding downsampled render data for delay estimation.
//!
//! Ported from `modules/audio_processing/aec3/downsampled_render_buffer.h/cc`.

/// Circular buffer of downsampled render data with read/write indices.
///
/// Unlike `BlockBuffer`/`SpectrumBuffer`/`FftBuffer`, this stores a flat
/// `Vec<f32>` rather than structured per-slot objects, so it has its own
/// index management instead of using `RingIndex`.
pub(crate) struct DownsampledRenderBuffer {
    pub size: usize,
    pub buffer: Vec<f32>,
    pub write: usize,
    pub read: usize,
}

impl DownsampledRenderBuffer {
    pub(crate) fn new(downsampled_buffer_size: usize) -> Self {
        Self {
            size: downsampled_buffer_size,
            buffer: vec![0.0f32; downsampled_buffer_size],
            write: 0,
            read: 0,
        }
    }

    pub(crate) fn inc_index(&self, index: usize) -> usize {
        if index < self.size - 1 { index + 1 } else { 0 }
    }

    pub(crate) fn dec_index(&self, index: usize) -> usize {
        if index > 0 { index - 1 } else { self.size - 1 }
    }

    pub(crate) fn offset_index(&self, index: usize, offset: i32) -> usize {
        ((self.size as i32 + index as i32 + offset) as usize) % self.size
    }

    pub(crate) fn update_write_index(&mut self, offset: i32) {
        self.write = self.offset_index(self.write, offset);
    }

    pub(crate) fn inc_write_index(&mut self) {
        self.write = self.inc_index(self.write);
    }

    pub(crate) fn dec_write_index(&mut self) {
        self.write = self.dec_index(self.write);
    }

    pub(crate) fn update_read_index(&mut self, offset: i32) {
        self.read = self.offset_index(self.read, offset);
    }

    pub(crate) fn inc_read_index(&mut self) {
        self.read = self.inc_index(self.read);
    }

    pub(crate) fn dec_read_index(&mut self) {
        self.read = self.dec_index(self.read);
    }
}
