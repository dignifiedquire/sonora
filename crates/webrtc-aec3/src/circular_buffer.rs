//! Circular buffer index management shared by BlockBuffer, SpectrumBuffer, and
//! FftBuffer.
//!
//! The C++ code duplicates identical IncIndex/DecIndex/OffsetIndex/read/write
//! logic in three separate structs. We extract the common index management here.

/// Circular buffer index state with `read` and `write` cursors.
#[derive(Debug, Clone)]
pub(crate) struct RingIndex {
    pub size: usize,
    pub write: usize,
    pub read: usize,
}

impl RingIndex {
    pub(crate) fn new(size: usize) -> Self {
        Self {
            size,
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

    pub(crate) fn inc_write(&mut self) {
        self.write = self.inc_index(self.write);
    }

    pub(crate) fn dec_write(&mut self) {
        self.write = self.dec_index(self.write);
    }

    pub(crate) fn update_write(&mut self, offset: i32) {
        self.write = self.offset_index(self.write, offset);
    }

    pub(crate) fn inc_read(&mut self) {
        self.read = self.inc_index(self.read);
    }

    pub(crate) fn dec_read(&mut self) {
        self.read = self.dec_index(self.read);
    }

    pub(crate) fn update_read(&mut self, offset: i32) {
        self.read = self.offset_index(self.read, offset);
    }
}
