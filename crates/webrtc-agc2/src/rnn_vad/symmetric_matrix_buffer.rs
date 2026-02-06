//! Buffer for pair-wise comparison results in a symmetric matrix.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/symmetric_matrix_buffer.h`.

/// Buffer for results of pair-wise comparisons between items stored in a ring
/// buffer.
///
/// Every time the oldest item is replaced in a ring buffer, the new one is
/// compared to the remaining items. The results are buffered here and
/// automatically evicted when one of the corresponding items leaves the ring
/// buffer.
///
/// The comparison is assumed symmetric and comparing an item with itself is not
/// needed.
///
/// Internally stores the upper-right triangular matrix (excluding diagonal)
/// encoded as a `(S-1) x (S-1)` square array, allowing `Push` to shift data
/// with a single `copy_within`.
#[derive(Debug)]
pub struct SymmetricMatrixBuffer<const S: usize> {
    buf: Vec<f32>,
}

impl<const S: usize> Default for SymmetricMatrixBuffer<S> {
    fn default() -> Self {
        const { assert!(S > 2) };
        Self {
            buf: vec![0.0; (S - 1) * (S - 1)],
        }
    }
}

impl<const S: usize> SymmetricMatrixBuffer<S> {
    /// Sets all buffer values to zero.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
    }

    /// Pushes the results from the comparison between the most recent item
    /// and those still in the ring buffer.
    ///
    /// `values` must have length `S - 1`.
    /// `values[0]` is the comparison between the most recent item and the
    /// second most recent one; `values[S-2]` is the comparison with the
    /// oldest one.
    pub fn push(&mut self, values: &[f32]) {
        debug_assert_eq!(values.len(), S - 1);
        // Move the lower-right sub-matrix of size (S-2)x(S-2) one row up
        // and one column left.
        self.buf.copy_within(S.., 0);
        // Copy new values in the last column in the right order.
        for (i, &val) in values.iter().enumerate().take(S - 1) {
            let index = (S - 1 - i) * (S - 1) - 1;
            debug_assert!(index < self.buf.len());
            self.buf[index] = val;
        }
    }

    /// Reads the value corresponding to the comparison of two items in the
    /// ring buffer having delay `delay1` and `delay2`.
    ///
    /// The two arguments must not be equal and both must be in `0..S`.
    pub fn get_value(&self, delay1: usize, delay2: usize) -> f32 {
        use std::mem;
        debug_assert_ne!(delay1, delay2, "The diagonal cannot be accessed.");
        let mut row = S - 1 - delay1;
        let mut col = S - 1 - delay2;
        if row > col {
            mem::swap(&mut row, &mut col);
        }
        debug_assert!(row < S - 1);
        debug_assert!(col >= 1 && col < S);
        let index = row * (S - 1) + (col - 1);
        debug_assert!(index < self.buf.len());
        self.buf[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn_vad::ring_buffer::RingBuffer;

    fn check_symmetry<const S: usize>(sym_matrix_buf: &SymmetricMatrixBuffer<S>) {
        for row in 0..S - 1 {
            for col in row + 1..S {
                assert_eq!(
                    sym_matrix_buf.get_value(row, col),
                    sym_matrix_buf.get_value(col, row),
                    "Asymmetry at ({row}, {col})"
                );
            }
        }
    }

    /// Test that shows how to combine RingBuffer and SymmetricMatrixBuffer to
    /// efficiently compute pair-wise scores. Verifies that the evolution of a
    /// SymmetricMatrixBuffer instance follows that of RingBuffer.
    ///
    /// This is a simplified adaptation of the C++ test which uses `PairType`.
    /// Since our RingBuffer is f32-only, we store single float values and
    /// verify the matrix tracks pair-wise products correctly.
    #[test]
    fn symmetric_matrix_buffer_use_case() {
        const RING_BUF_SIZE: usize = 10;
        let mut ring_buf = RingBuffer::<1, RING_BUF_SIZE>::default();
        let mut sym_matrix_buf = SymmetricMatrixBuffer::<RING_BUF_SIZE>::default();

        for t in 1..=100u32 {
            let t_f = t as f32;
            ring_buf.push(&[t_f]);

            // The head of the ring buffer is `t`.
            assert_eq!(ring_buf.get_array_view(0), &[t_f]);

            // Create comparisons: encode pair (older, newer) as older * 1000 + newer.
            let mut new_comparisons = [0.0_f32; RING_BUF_SIZE - 1];
            for (i, cmp) in new_comparisons.iter_mut().enumerate() {
                let delay = i + 1;
                let t_prev = ring_buf.get_array_view(delay)[0];
                *cmp = t_prev * 1000.0 + t_f;
            }
            sym_matrix_buf.push(&new_comparisons);

            // Check symmetry.
            check_symmetry(&sym_matrix_buf);

            // Check that the pairs resulting from the ring buffer content are
            // in the right position.
            for delay1 in 0..RING_BUF_SIZE - 1 {
                for delay2 in delay1 + 1..RING_BUF_SIZE {
                    let t1 = ring_buf.get_array_view(delay1)[0];
                    let t2 = ring_buf.get_array_view(delay2)[0];
                    assert!(t2 <= t1);
                    let val = sym_matrix_buf.get_value(delay1, delay2);
                    let expected = t2 * 1000.0 + t1;
                    assert_eq!(
                        val, expected,
                        "Mismatch at t={t}, delay1={delay1}, delay2={delay2}"
                    );
                }
            }
        }
    }
}
