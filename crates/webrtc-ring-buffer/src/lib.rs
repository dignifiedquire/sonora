//! Ring buffer matching WebRTC semantics.
//!
//! Provides a fixed-capacity FIFO ring buffer with:
//! - Zero-copy reads when data is contiguous (no wrap-around)
//! - Bidirectional read cursor movement (flush forward or stuff backward)
//! - Generic element type
//!
//! This is a faithful port of `webrtc/common_audio/ring_buffer.c`.

#![deny(unsafe_code)]

use std::num::NonZero;

/// A fixed-capacity ring buffer with WebRTC-compatible semantics.
///
/// The buffer tracks separate read and write positions and supports:
/// - [`write`](Self::write): append elements, up to available space
/// - [`read`](Self::read): consume elements into a caller-provided slice
/// - [`read_zero_copy`](Self::read_zero_copy): consume elements, returning a
///   direct slice into the buffer when contiguous (avoiding a copy)
/// - [`move_read_ptr`](Self::move_read_ptr): advance (positive) or rewind
///   (negative) the read cursor
///
/// # Invariants
///
/// - `read_pos` and `write_pos` are in `0..=capacity` (may equal capacity,
///   matching the C implementation where positions wrap on the *next* operation)
/// - `available_read() + available_write() == capacity` always holds
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: Vec<T>,
    read_pos: usize,
    write_pos: usize,
    /// Tracks whether write has wrapped around but read hasn't yet.
    /// When `true`, readable data spans from `read_pos..capacity` then
    /// `0..write_pos`.
    wrapped: bool,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// Creates a new ring buffer that can hold `capacity` elements.
    pub fn new(capacity: NonZero<usize>) -> Self {
        Self {
            data: vec![T::default(); capacity.get()],
            read_pos: 0,
            write_pos: 0,
            wrapped: false,
        }
    }

    /// Resets the buffer to its initial empty state, zeroing all elements.
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.wrapped = false;
        self.data.fill(T::default());
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Returns the total capacity of the buffer in elements.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of elements available to read.
    pub fn available_read(&self) -> usize {
        if self.wrapped {
            self.capacity() - self.read_pos + self.write_pos
        } else {
            self.write_pos - self.read_pos
        }
    }

    /// Returns the number of elements that can be written.
    pub fn available_write(&self) -> usize {
        self.capacity() - self.available_read()
    }

    /// Writes elements from `data` into the buffer.
    ///
    /// Returns the number of elements actually written (limited by available
    /// space).
    pub fn write(&mut self, data: &[T]) -> usize {
        let free = self.available_write();
        let write_count = data.len().min(free);
        let margin = self.capacity() - self.write_pos;

        if write_count > margin {
            self.data[self.write_pos..].clone_from_slice(&data[..margin]);
            self.write_pos = 0;
            let remaining = write_count - margin;
            self.wrapped = true;
            self.data[..remaining].clone_from_slice(&data[margin..margin + remaining]);
            self.write_pos = remaining;
        } else {
            self.data[self.write_pos..self.write_pos + write_count]
                .clone_from_slice(&data[..write_count]);
            self.write_pos += write_count;
        }

        write_count
    }

    /// Reads up to `output.len()` elements, copying them into `output`.
    ///
    /// Returns the number of elements actually read (limited by available
    /// data). The read cursor advances past the consumed elements.
    pub fn read(&mut self, output: &mut [T]) -> usize {
        let readable = self.available_read();
        let read_count = output.len().min(readable);
        if read_count == 0 {
            return 0;
        }

        let margin = self.capacity() - self.read_pos;
        if read_count > margin {
            output[..margin].clone_from_slice(&self.data[self.read_pos..]);
            output[margin..read_count].clone_from_slice(&self.data[..read_count - margin]);
        } else {
            output[..read_count]
                .clone_from_slice(&self.data[self.read_pos..self.read_pos + read_count]);
        }

        self.advance_read_pos(read_count);
        read_count
    }

    /// Reads up to `count` elements with zero-copy when possible.
    ///
    /// If the requested data is contiguous in the internal buffer (no
    /// wrap-around), returns [`ZeroCopyResult::Contiguous`] with a direct
    /// slice. Otherwise, copies the data into `fallback` and returns
    /// [`ZeroCopyResult::Copied`].
    ///
    /// In both cases the read cursor advances past the consumed elements.
    pub fn read_zero_copy<'a>(
        &'a mut self,
        fallback: &'a mut [T],
        count: usize,
    ) -> ZeroCopyResult<'a, T> {
        let readable = self.available_read();
        let read_count = count.min(readable);
        if read_count == 0 {
            return ZeroCopyResult::Copied(&mut fallback[..0]);
        }

        let margin = self.capacity() - self.read_pos;
        if read_count > margin {
            fallback[..margin].clone_from_slice(&self.data[self.read_pos..]);
            fallback[margin..read_count].clone_from_slice(&self.data[..read_count - margin]);
            self.advance_read_pos(read_count);
            ZeroCopyResult::Copied(&mut fallback[..read_count])
        } else {
            let start = self.read_pos;
            self.advance_read_pos(read_count);
            ZeroCopyResult::Contiguous(&self.data[start..start + read_count])
        }
    }

    /// Moves the read pointer by `offset` elements.
    ///
    /// Positive values advance toward the write position (flushing data).
    /// Negative values move away from it (stuffing data back).
    ///
    /// The offset is clamped to `[-available_write, available_read]`.
    ///
    /// Returns the actual number of elements moved (may differ from `offset`
    /// due to clamping).
    pub fn move_read_ptr(&mut self, offset: isize) -> isize {
        let readable = self.available_read() as isize;
        let free = self.available_write() as isize;

        let clamped = offset.clamp(-free, readable);

        let mut pos = self.read_pos as isize + clamped;
        let cap = self.capacity() as isize;

        // `read_pos == capacity` is valid (matches C semantics where the
        // position wraps on the next operation, not eagerly).
        if pos > cap {
            pos -= cap;
            self.wrapped = false;
        } else if pos < 0 {
            pos += cap;
            self.wrapped = true;
        }

        self.read_pos = pos as usize;
        clamped
    }

    /// Advances the read position forward by `count` elements.
    ///
    /// This is the common case used by `read` and `read_zero_copy` where the
    /// count is already known to be valid (bounded by `available_read`).
    fn advance_read_pos(&mut self, count: usize) {
        let mut pos = self.read_pos + count;
        // Position can equal capacity (wraps lazily), but not exceed it.
        if pos > self.capacity() {
            pos -= self.capacity();
            self.wrapped = false;
        }
        self.read_pos = pos;
    }
}

/// Result of a [`RingBuffer::read_zero_copy`] operation.
#[derive(Debug)]
pub enum ZeroCopyResult<'a, T> {
    /// Data was contiguous; this slice points directly into the ring buffer.
    /// Valid until the next [`RingBuffer::write`].
    Contiguous(&'a [T]),
    /// Data wrapped around; it was copied into the caller's fallback buffer.
    Copied(&'a mut [T]),
}

impl<T> ZeroCopyResult<'_, T> {
    /// Returns a shared slice to the data regardless of variant.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Contiguous(s) => s,
            Self::Copied(s) => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use proptest::collection::vec as pvec;
    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::{RingBuffer, ZeroCopyResult};

    fn rb(capacity: usize) -> RingBuffer<i32> {
        RingBuffer::new(NonZero::new(capacity).unwrap())
    }

    // -- Helpers matching the C++ test patterns --

    fn set_incrementing_data(data: &mut [i32], start: i32) -> i32 {
        let mut val = start;
        for elem in data.iter_mut() {
            *elem = val;
            val += 1;
        }
        val
    }

    fn check_incrementing_data(data: &[i32], start: i32) -> i32 {
        let mut val = start;
        for (i, &elem) in data.iter().enumerate() {
            assert_eq!(elem, val, "mismatch at index {i}");
            val += 1;
        }
        val
    }

    /// Port of `RandomStressTest` from ring_buffer_unittest.cc.
    fn random_stress_test(seed: u64, use_zero_copy: bool) {
        let mut rng_state = seed;
        let mut next = || -> usize {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state as usize
        };

        let num_tests = 10;
        let num_ops = 1000;
        let max_buffer_size = 1000;

        for _ in 0..num_tests {
            let buffer_size = (next() % max_buffer_size).max(1);
            let mut write_data = vec![0i32; buffer_size];
            let mut read_data = vec![0i32; buffer_size];
            let mut buffer = rb(buffer_size);

            let mut buffer_consumed: usize = 0;
            let mut write_element: i32 = 0;
            let mut read_element: i32 = 0;

            for _ in 0..num_ops {
                let do_write = next() % 2 == 0;
                let num_elements = next() % buffer_size;

                if do_write {
                    let buffer_available = buffer_size - buffer_consumed;
                    assert_eq!(buffer_available, buffer.available_write());
                    let expected = num_elements.min(buffer_available);
                    write_element =
                        set_incrementing_data(&mut write_data[..expected], write_element);
                    let written = buffer.write(&write_data[..num_elements]);
                    assert_eq!(expected, written);
                    buffer_consumed = (buffer_consumed + expected).min(buffer_size);
                } else {
                    let expected = num_elements.min(buffer_consumed);
                    assert_eq!(buffer_consumed, buffer.available_read());

                    if use_zero_copy {
                        let result =
                            buffer.read_zero_copy(&mut read_data[..num_elements], num_elements);
                        let check = result.as_slice();
                        read_element = check_incrementing_data(&check[..expected], read_element);
                    } else {
                        let read_count = buffer.read(&mut read_data[..num_elements]);
                        assert_eq!(expected, read_count);
                        read_element =
                            check_incrementing_data(&read_data[..expected], read_element);
                    }

                    buffer_consumed = buffer_consumed.saturating_sub(expected);
                }
            }
        }
    }

    // -- Unit tests matching upstream C++ tests --

    #[test]
    fn stress_test_with_copy() {
        random_stress_test(12345, false);
    }

    #[test]
    fn stress_test_with_zero_copy() {
        random_stress_test(12345, true);
    }

    #[test]
    fn zero_copy_returns_contiguous_when_no_wrap() {
        let mut buf = rb(4);
        let data = [10, 20, 30, 40];
        assert_eq!(4, buf.write(&data));

        let mut fallback = [0i32; 4];
        let result = buf.read_zero_copy(&mut fallback, 4);

        assert!(matches!(result, ZeroCopyResult::Contiguous(_)));
        assert_eq!(result.as_slice(), &[10, 20, 30, 40]);
        assert_eq!(fallback, [0, 0, 0, 0]);
    }

    #[test]
    fn zero_copy_copies_on_wrap() {
        let mut buf = rb(4);

        assert_eq!(4, buf.write(&[1, 2, 3, 4]));
        let mut discard = [0i32; 2];
        buf.read(&mut discard);
        assert_eq!(2, buf.write(&[5, 6]));

        let mut fallback = [0i32; 4];
        let result = buf.read_zero_copy(&mut fallback, 4);

        assert!(matches!(result, ZeroCopyResult::Copied(_)));
        assert_eq!(result.as_slice(), &[3, 4, 5, 6]);
    }

    #[test]
    fn move_read_ptr_forward() {
        let mut buf = rb(8);
        buf.write(&[1, 2, 3, 4, 5]);
        assert_eq!(5, buf.available_read());

        let moved = buf.move_read_ptr(3);
        assert_eq!(3, moved);
        assert_eq!(2, buf.available_read());

        let mut out = [0i32; 2];
        buf.read(&mut out);
        assert_eq!(out, [4, 5]);
    }

    #[test]
    fn move_read_ptr_backward() {
        let mut buf = rb(8);
        buf.write(&[1, 2, 3, 4, 5]);

        let mut discard = [0i32; 3];
        buf.read(&mut discard);
        assert_eq!(2, buf.available_read());
        assert_eq!(6, buf.available_write());

        let moved = buf.move_read_ptr(-2);
        assert_eq!(-2, moved);
        assert_eq!(4, buf.available_read());

        let mut out = [0i32; 4];
        buf.read(&mut out);
        assert_eq!(out, [2, 3, 4, 5]);
    }

    #[test]
    fn move_read_ptr_backward_with_wrap() {
        let mut buf = rb(4);
        buf.write(&[1, 2, 3, 4]);

        let mut discard = [0i32; 3];
        buf.read(&mut discard);
        buf.write(&[5, 6, 7]);
        assert_eq!(4, buf.available_read());
        assert_eq!(0, buf.available_write());

        let mut out = [0i32; 2];
        buf.read(&mut out);
        assert_eq!(out, [4, 5]);

        let moved = buf.move_read_ptr(-1);
        assert_eq!(-1, moved);
        assert_eq!(3, buf.available_read());

        let mut out3 = [0i32; 3];
        buf.read(&mut out3);
        assert_eq!(out3, [5, 6, 7]);
    }

    #[test]
    fn move_read_ptr_clamped() {
        let mut buf = rb(4);
        buf.write(&[1, 2, 3]);

        let moved = buf.move_read_ptr(10);
        assert_eq!(3, moved);
        assert_eq!(0, buf.available_read());

        let moved = buf.move_read_ptr(-10);
        assert_eq!(-4, moved);
        assert_eq!(4, buf.available_read());
    }

    #[test]
    fn write_limited_by_capacity() {
        let mut buf = rb(4);
        let written = buf.write(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(4, written);
        assert_eq!(4, buf.available_read());
        assert_eq!(0, buf.available_write());
    }

    #[test]
    fn read_from_empty() {
        let mut buf = rb(4);
        let mut out = [0i32; 4];
        assert_eq!(0, buf.read(&mut out));
    }

    #[test]
    fn clear_resets_state() {
        let mut buf = rb(4);
        buf.write(&[1, 2, 3, 4]);
        assert_eq!(4, buf.available_read());

        buf.clear();
        assert_eq!(0, buf.available_read());
        assert_eq!(4, buf.available_write());
    }

    // -- Property tests --

    #[proptest]
    fn available_read_plus_write_equals_capacity(
        #[strategy(1..=500usize)] capacity: usize,
        #[strategy(pvec(any::<i32>(), 0..500))] data: Vec<i32>,
    ) {
        let mut buf = rb(capacity);
        buf.write(&data);
        prop_assert_eq!(buf.available_read() + buf.available_write(), capacity);
    }

    #[proptest]
    fn write_then_read_roundtrips(
        #[strategy(1..=500usize)] capacity: usize,
        #[strategy(pvec(any::<i32>(), 0..=#capacity))] data: Vec<i32>,
    ) {
        let mut buf = rb(capacity);
        let written = buf.write(&data);
        let mut out = vec![0i32; written];
        let read_count = buf.read(&mut out);
        prop_assert_eq!(read_count, written);
        prop_assert_eq!(&out[..read_count], &data[..written]);
    }

    #[proptest]
    fn move_read_ptr_preserves_invariants(
        #[strategy(1..=200usize)] capacity: usize,
        #[strategy(pvec(any::<i32>(), 0..200))] data: Vec<i32>,
        #[strategy(-200isize..=200)] offset: isize,
    ) {
        let mut buf = rb(capacity);
        buf.write(&data);
        buf.move_read_ptr(offset);
        prop_assert_eq!(buf.available_read() + buf.available_write(), capacity);
        prop_assert!(buf.available_read() <= capacity);
    }

    #[proptest]
    fn stress_random_seed(#[strategy(1..=100_000u64)] seed: u64) {
        random_stress_test(seed, false);
    }

    #[proptest]
    fn stress_random_seed_zero_copy(#[strategy(1..=100_000u64)] seed: u64) {
        random_stress_test(seed, true);
    }
}
