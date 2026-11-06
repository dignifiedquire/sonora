//! Circular moving average filter.
//!
//! Ported from `modules/audio_processing/aec3/moving_average.h/cc`.

/// Computes the running average over the last `mem_len` input vectors.
///
/// Until `mem_len` inputs have been seen, the average is taken over the
/// inputs seen so far rather than over a window padded with zeros.
#[derive(Debug)]
pub(crate) struct MovingAverage {
    num_elem: usize,
    mem_len: usize,
    memory: Vec<f32>,
    mem_index: usize,
    /// Number of stored inputs that contribute to the average (C++
    /// `number_updates_`), saturating at `mem_len`.
    number_updates: usize,
}

impl MovingAverage {
    /// Creates an instance that accepts inputs of length `num_elem` and
    /// averages over the last `mem_len` inputs.
    pub(crate) fn new(num_elem: usize, mem_len: usize) -> Self {
        debug_assert!(num_elem > 0);
        debug_assert!(mem_len > 0);
        let stored = mem_len - 1; // current input is not stored until after use
        Self {
            num_elem,
            mem_len: stored,
            memory: vec![0.0; num_elem * stored],
            mem_index: 0,
            number_updates: 0,
        }
    }

    /// Computes the average of `input` and up to `mem_len - 1` previous
    /// inputs, writing the result to `output`.
    pub(crate) fn average(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.num_elem);
        debug_assert_eq!(output.len(), self.num_elem);

        // Start with the current input.
        output.copy_from_slice(input);

        // Sum all stored contributions.
        for chunk in self.memory.chunks_exact(self.num_elem) {
            for (o, &m) in output.iter_mut().zip(chunk.iter()) {
                *o += m;
            }
        }

        // Divide by the number of points used to compute the average.
        let scaling = 1.0 / (self.number_updates + 1) as f32;
        for o in output.iter_mut() {
            *o *= scaling;
        }

        // Update memory ring buffer.
        if self.mem_len > 0 {
            let start = self.mem_index * self.num_elem;
            self.memory[start..start + self.num_elem].copy_from_slice(input);
            self.mem_index = (self.mem_index + 1) % self.mem_len;
        }
        self.number_updates = self.mem_len.min(self.number_updates + 1);
    }

    /// If `mem_len` differs from the current window length, resets the state
    /// and clears the memory to use the new window length (C++
    /// `UpdateMemoryLength`).
    pub(crate) fn update_memory_length(&mut self, mem_len: usize) {
        if self.mem_len + 1 == mem_len {
            return;
        }
        debug_assert!(mem_len > 0);
        self.mem_len = mem_len - 1;
        self.memory.resize(self.num_elem * self.mem_len, 0.0);
        self.memory.fill(0.0);
        self.mem_index = 0;
        self.number_updates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_with_memory() {
        let num_elem = 4;
        let mem_len = 3;
        let e = 1e-6;
        let mut ma = MovingAverage::new(num_elem, mem_len);

        let data1 = [1.0, 2.0, 3.0, 4.0];
        let data2 = [5.0, 1.0, 9.0, 7.0];
        let data3 = [3.0, 3.0, 5.0, 6.0];
        let data4 = [8.0, 4.0, 2.0, 1.0];
        let mut output = [0.0f32; 4];

        // First call: only data1 has been seen, so the average is data1
        // itself. The empty memory slots must not dilute it.
        ma.average(&data1, &mut output);
        for i in 0..num_elem {
            assert!((output[i] - data1[i] / 1.0).abs() < e, "step 1, elem {i}");
        }

        // Second call: average of the two inputs seen so far.
        ma.average(&data2, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data1[i] + data2[i]) / 2.0).abs() < e,
                "step 2, elem {i}"
            );
        }

        // Third call: data1 + data2 + data3.
        ma.average(&data3, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data1[i] + data2[i] + data3[i]) / 3.0).abs() < e,
                "step 3, elem {i}"
            );
        }

        // Fourth call: oldest (data1) dropped, now data2 + data3 + data4.
        ma.average(&data4, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data2[i] + data3[i] + data4[i]) / 3.0).abs() < e,
                "step 4, elem {i}"
            );
        }
    }

    #[test]
    fn update_memory_length() {
        let num_elem = 4;
        let e = 1e-6;
        let mut ma = MovingAverage::new(num_elem, 3);

        let data1 = [1.0, 2.0, 3.0, 4.0];
        let data2 = [5.0, 1.0, 9.0, 7.0];
        let mut output = [0.0f32; 4];

        ma.average(&data1, &mut output);
        assert!((output[0] - data1[0]).abs() < e);

        ma.update_memory_length(1);
        ma.average(&data2, &mut output);
        // After the update, it behaves as if it was just created with
        // mem_len = 1.
        for i in 0..num_elem {
            assert!((output[i] - data2[i]).abs() < e, "elem {i}");
        }
    }

    /// A suppressor config switch changes the averaging window at runtime.
    /// Inputs from the old window must not leak into the new one, and the
    /// new window must ramp up like a freshly created instance. Keeping the
    /// same window length must keep the history.
    #[test]
    fn update_memory_length_restarts_ramp_up() {
        let num_elem = 4;
        let e = 1e-6;
        let mut ma = MovingAverage::new(num_elem, 3);

        let data1 = [1.0, 2.0, 3.0, 4.0];
        let data2 = [5.0, 1.0, 9.0, 7.0];
        let data3 = [3.0, 3.0, 5.0, 6.0];
        let data4 = [8.0, 4.0, 2.0, 1.0];
        let mut output = [0.0f32; 4];

        ma.average(&data1, &mut output);
        ma.average(&data2, &mut output);

        // Same length: no reset, so data1 and data2 still count.
        ma.update_memory_length(3);
        ma.average(&data3, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data1[i] + data2[i] + data3[i]) / 3.0).abs() < e,
                "same length, elem {i}"
            );
        }

        // New length: history is dropped and the ramp-up starts over.
        ma.update_memory_length(2);
        ma.average(&data4, &mut output);
        for i in 0..num_elem {
            assert!((output[i] - data4[i]).abs() < e, "after resize, elem {i}");
        }
        ma.average(&data1, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data4[i] + data1[i]) / 2.0).abs() < e,
                "after resize, second call, elem {i}"
            );
        }
    }

    #[test]
    fn pass_through_with_mem_len_1() {
        let num_elem = 4;
        let mem_len = 1;
        let e = 1e-6;
        let mut ma = MovingAverage::new(num_elem, mem_len);

        let data1 = [1.0, 2.0, 3.0, 4.0];
        let data2 = [5.0, 1.0, 9.0, 7.0];
        let data3 = [3.0, 3.0, 5.0, 6.0];
        let data4 = [8.0, 4.0, 2.0, 1.0];
        let mut output = [0.0f32; 4];

        // With mem_len=1, output should equal input exactly.
        for (data, step) in [data1, data2, data3, data4].iter().zip(1..) {
            ma.average(data, &mut output);
            for i in 0..num_elem {
                assert!(
                    (output[i] - data[i]).abs() < e,
                    "step {step}, elem {i}: got {}, expected {}",
                    output[i],
                    data[i]
                );
            }
        }
    }
}
