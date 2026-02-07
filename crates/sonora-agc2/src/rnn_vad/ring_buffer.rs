//! Fixed-size ring buffer for arrays.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/ring_buffer.h`.

/// Ring buffer for `N` arrays each with size `S`.
#[derive(Debug)]
pub struct RingBuffer<const S: usize, const N: usize> {
    tail: usize,
    buffer: Vec<f32>,
}

impl<const S: usize, const N: usize> Default for RingBuffer<S, N> {
    fn default() -> Self {
        Self {
            tail: 0,
            buffer: vec![0.0; S * N],
        }
    }
}

impl<const S: usize, const N: usize> RingBuffer<S, N> {
    /// Set all buffer values to zero.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Replace the least recently pushed array with `new_values`.
    pub fn push(&mut self, new_values: &[f32; S]) {
        let start = S * self.tail;
        self.buffer[start..start + S].copy_from_slice(new_values);
        self.tail += 1;
        if self.tail == N {
            self.tail = 0;
        }
    }

    /// Return a reference to the array with the given delay.
    ///
    /// Delay 0 returns the most recently pushed array.
    /// Delay N-1 returns the least recently pushed array.
    pub fn get_array_view(&self, delay: usize) -> &[f32; S] {
        debug_assert!(delay < N);
        let mut offset = self.tail as isize - 1 - delay as isize;
        if offset < 0 {
            offset += N as isize;
        }
        let start = S * offset as usize;
        self.buffer[start..start + S].try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_basic() {
        let mut ring_buf = RingBuffer::<2, 3>::default();

        // Push [1, 1]
        ring_buf.push(&[1.0, 1.0]);
        assert_eq!(ring_buf.get_array_view(0), &[1.0, 1.0]);

        // Push [2, 2]
        ring_buf.push(&[2.0, 2.0]);
        assert_eq!(ring_buf.get_array_view(0), &[2.0, 2.0]);
        assert_eq!(ring_buf.get_array_view(1), &[1.0, 1.0]);

        // Push [3, 3]
        ring_buf.push(&[3.0, 3.0]);
        assert_eq!(ring_buf.get_array_view(0), &[3.0, 3.0]);
        assert_eq!(ring_buf.get_array_view(1), &[2.0, 2.0]);
        assert_eq!(ring_buf.get_array_view(2), &[1.0, 1.0]);

        // Push [4, 4] — wraps around, overwrites [1, 1]
        ring_buf.push(&[4.0, 4.0]);
        assert_eq!(ring_buf.get_array_view(0), &[4.0, 4.0]);
        assert_eq!(ring_buf.get_array_view(1), &[3.0, 3.0]);
        assert_eq!(ring_buf.get_array_view(2), &[2.0, 2.0]);
    }

    #[test]
    fn ring_buffer_array_views_differ() {
        let mut ring_buf = RingBuffer::<3, 4>::default();
        let pushed = [1.0; 3];
        for _ in 0..=4 {
            for i in 0..4 {
                let view_i = ring_buf.get_array_view(i);
                for j in (i + 1)..4 {
                    let view_j = ring_buf.get_array_view(j);
                    assert_ne!(view_i as *const _ as usize, view_j as *const _ as usize);
                }
            }
            ring_buf.push(&pushed);
        }
    }

    #[test]
    fn ring_buffer_single_element() {
        let mut ring_buf = RingBuffer::<1, 1>::default();
        ring_buf.push(&[42.0]);
        assert_eq!(ring_buf.get_array_view(0), &[42.0]);
        ring_buf.push(&[99.0]);
        assert_eq!(ring_buf.get_array_view(0), &[99.0]);
    }

    #[test]
    fn ring_buffer_full_cycle() {
        let mut ring_buf = RingBuffer::<5, 5>::default();
        for v in 0..5u32 {
            ring_buf.push(&[v as f32; 5]);
        }
        for delay in 0..5 {
            assert_eq!(ring_buf.get_array_view(delay), &[(4 - delay) as f32; 5]);
        }
    }

    #[test]
    fn ring_buffer_reset() {
        let mut ring_buf = RingBuffer::<3, 2>::default();
        ring_buf.push(&[1.0, 2.0, 3.0]);
        ring_buf.reset();
        assert_eq!(ring_buf.get_array_view(0), &[0.0, 0.0, 0.0]);
    }
}
