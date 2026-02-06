//! Linear sequence buffer with push semantics.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/sequence_buffer.h`.

/// Linear buffer to push fixed-size chunks of sequential data and view
/// contiguous parts of the buffer.
///
/// The buffer has size `S` and pushed chunks have size `N`. When pushed,
/// the buffer shifts left by `N` and the new chunk fills the rightmost
/// `N` positions. For instance, when `S = 2*N` the first half is replaced
/// with the second half, and new values are written at the end.
#[derive(Debug)]
pub struct SequenceBuffer<const S: usize, const N: usize> {
    buffer: Vec<f32>,
}

impl<const S: usize, const N: usize> Default for SequenceBuffer<S, N> {
    fn default() -> Self {
        const { assert!(N <= S, "Chunk size cannot be larger than buffer size") };
        Self {
            buffer: vec![0.0; S],
        }
    }
}

impl<const S: usize, const N: usize> SequenceBuffer<S, N> {
    /// Returns the buffer size.
    pub fn size(&self) -> usize {
        S
    }

    /// Returns the chunk size.
    pub fn chunks_size(&self) -> usize {
        N
    }

    /// Sets all buffer values to zero.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Returns a view on the whole buffer.
    pub fn get_buffer_view(&self) -> &[f32] {
        &self.buffer
    }

    /// Returns a view on the `M` most recent values of the buffer.
    pub fn get_most_recent_values_view<const M: usize>(&self) -> &[f32; M] {
        const { assert!(M <= S, "Most recent values cannot exceed buffer size") };
        self.buffer[S - M..].try_into().unwrap()
    }

    /// Shifts the buffer left by `N` items and adds new `N` items at the end.
    pub fn push(&mut self, new_values: &[f32; N]) {
        if S > N {
            self.buffer.copy_within(N.., 0);
        }
        self.buffer[S - N..].copy_from_slice(new_values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_push_op<const S: usize, const N: usize>() {
        let mut seq_buf = SequenceBuffer::<S, N>::default();
        let _buf_view_ptr = seq_buf.buffer.as_ptr();

        // Check that a chunk is fully gone after ceil(S / N) push ops.
        let ones = [1.0_f32; N];
        seq_buf.push(&ones);
        let zeros = [0.0_f32; N];
        let required_push_ops = if !S.is_multiple_of(N) {
            S / N + 1
        } else {
            S / N
        };
        for _ in 0..required_push_ops - 1 {
            seq_buf.push(&zeros);
            let max = seq_buf
                .buffer
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(max, 1.0, "Value should still be in buffer");
        }
        // Gone after another push.
        seq_buf.push(&zeros);
        let max = seq_buf
            .buffer
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(max, 0.0, "Value should be gone");

        // Check that the last item moves left by N positions after a push.
        if S > N {
            let mut chunk = [0.0_f32; N];
            for (i, val) in chunk.iter_mut().enumerate() {
                *val = (i + 1) as f32;
            }
            seq_buf.push(&chunk);
            let last = chunk[N - 1];
            let mut chunk2 = [0.0_f32; N];
            for (i, val) in chunk2.iter_mut().enumerate() {
                *val = last + (i + 1) as f32;
            }
            seq_buf.push(&chunk2);
            assert_eq!(seq_buf.buffer[S - N - 1], last);
        }
    }

    #[test]
    fn sequence_buffer_getters() {
        let mut seq_buf = SequenceBuffer::<8, 8>::default();
        assert_eq!(seq_buf.size(), 8);
        assert_eq!(seq_buf.chunks_size(), 8);

        let view = seq_buf.get_buffer_view();
        assert_eq!(view[0], 0.0);
        assert_eq!(view[view.len() - 1], 0.0);

        let chunk: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        seq_buf.push(&chunk);
        let view = seq_buf.get_buffer_view();
        assert_eq!(view[0], 10.0);
        assert_eq!(view[view.len() - 1], 80.0);
    }

    #[test]
    fn sequence_buffer_push_25_percent() {
        test_push_op::<32, 8>();
    }

    #[test]
    fn sequence_buffer_push_50_percent() {
        test_push_op::<32, 16>();
    }

    #[test]
    fn sequence_buffer_push_100_percent() {
        test_push_op::<32, 32>();
    }

    #[test]
    fn sequence_buffer_push_non_integer_ratio() {
        test_push_op::<23, 7>();
    }

    #[test]
    fn sequence_buffer_most_recent() {
        let mut seq_buf = SequenceBuffer::<8, 4>::default();
        seq_buf.push(&[1.0, 2.0, 3.0, 4.0]);
        let recent: &[f32; 4] = seq_buf.get_most_recent_values_view::<4>();
        assert_eq!(recent, &[1.0, 2.0, 3.0, 4.0]);

        seq_buf.push(&[5.0, 6.0, 7.0, 8.0]);
        let recent: &[f32; 4] = seq_buf.get_most_recent_values_view::<4>();
        assert_eq!(recent, &[5.0, 6.0, 7.0, 8.0]);

        let all: &[f32; 8] = seq_buf.get_most_recent_values_view::<8>();
        assert_eq!(all, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
