//! Ring buffer for the saturation protector.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/saturation_protector_buffer.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::SATURATION_PROTECTOR_BUFFER_SIZE;

/// Ring buffer that supports push back and read oldest item.
#[derive(Debug, Clone)]
pub(crate) struct SaturationProtectorBuffer {
    buffer: [f32; SATURATION_PROTECTOR_BUFFER_SIZE],
    next: usize,
    size: usize,
}

impl Default for SaturationProtectorBuffer {
    fn default() -> Self {
        Self {
            buffer: [0.0; SATURATION_PROTECTOR_BUFFER_SIZE],
            next: 0,
            size: 0,
        }
    }
}

impl PartialEq for SaturationProtectorBuffer {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        let mut i0 = self.front_index();
        let mut i1 = other.front_index();
        for _ in 0..self.size {
            if self.buffer[i0 % SATURATION_PROTECTOR_BUFFER_SIZE]
                != other.buffer[i1 % SATURATION_PROTECTOR_BUFFER_SIZE]
            {
                return false;
            }
            i0 += 1;
            i1 += 1;
        }
        true
    }
}

impl Eq for SaturationProtectorBuffer {}

impl SaturationProtectorBuffer {
    /// Maximum number of values the buffer can contain.
    pub(crate) fn capacity(&self) -> usize {
        SATURATION_PROTECTOR_BUFFER_SIZE
    }

    /// Number of values currently in the buffer.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Resets the buffer to empty.
    pub(crate) fn reset(&mut self) {
        self.next = 0;
        self.size = 0;
    }

    /// Pushes `v` to the back. If full, the oldest value is replaced.
    pub(crate) fn push_back(&mut self, v: f32) {
        self.buffer[self.next] = v;
        self.next += 1;
        if self.next == SATURATION_PROTECTOR_BUFFER_SIZE {
            self.next = 0;
        }
        if self.size < SATURATION_PROTECTOR_BUFFER_SIZE {
            self.size += 1;
        }
    }

    /// Returns the oldest item, or `None` if empty.
    pub(crate) fn front(&self) -> Option<f32> {
        if self.size == 0 {
            return None;
        }
        Some(self.buffer[self.front_index()])
    }

    fn front_index(&self) -> usize {
        if self.size == SATURATION_PROTECTOR_BUFFER_SIZE {
            self.next
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct() {
        let buffer = SaturationProtectorBuffer::default();
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.capacity(), SATURATION_PROTECTOR_BUFFER_SIZE);
        assert_eq!(buffer.front(), None);
    }

    #[test]
    fn push_and_front() {
        let mut buffer = SaturationProtectorBuffer::default();
        buffer.push_back(1.0);
        assert_eq!(buffer.size(), 1);
        assert_eq!(buffer.front(), Some(1.0));

        buffer.push_back(2.0);
        assert_eq!(buffer.size(), 2);
        assert_eq!(buffer.front(), Some(1.0));
    }

    #[test]
    fn wrap_around() {
        let mut buffer = SaturationProtectorBuffer::default();
        for i in 0..SATURATION_PROTECTOR_BUFFER_SIZE {
            buffer.push_back(i as f32);
        }
        assert_eq!(buffer.size(), SATURATION_PROTECTOR_BUFFER_SIZE);
        assert_eq!(buffer.front(), Some(0.0));

        // Overflow: oldest (0.0) is replaced.
        buffer.push_back(100.0);
        assert_eq!(buffer.size(), SATURATION_PROTECTOR_BUFFER_SIZE);
        assert_eq!(buffer.front(), Some(1.0));
    }

    #[test]
    fn reset() {
        let mut buffer = SaturationProtectorBuffer::default();
        buffer.push_back(1.0);
        buffer.push_back(2.0);
        buffer.reset();
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.front(), None);
    }

    #[test]
    fn equality() {
        let mut a = SaturationProtectorBuffer::default();
        let mut b = SaturationProtectorBuffer::default();
        assert_eq!(a, b);

        a.push_back(1.0);
        assert_ne!(a, b);

        b.push_back(1.0);
        assert_eq!(a, b);

        // Same logical content but different internal state.
        a.push_back(2.0);
        a.push_back(3.0);
        b.push_back(2.0);
        b.push_back(3.0);
        assert_eq!(a, b);
    }
}
