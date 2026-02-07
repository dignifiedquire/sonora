//! Circular moving average filter.
//!
//! Ported from `modules/audio_processing/aec3/moving_average.h/cc`.

/// Computes the running average over the last `mem_len` input vectors.
pub(crate) struct MovingAverage {
    num_elem: usize,
    mem_len: usize,
    scaling: f32,
    memory: Vec<f32>,
    mem_index: usize,
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
            scaling: 1.0 / mem_len as f32,
            memory: vec![0.0; num_elem * stored],
            mem_index: 0,
        }
    }

    /// Computes the average of `input` and the `mem_len - 1` previous inputs,
    /// writing the result to `output`.
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

        // Divide by total window length.
        for o in output.iter_mut() {
            *o *= self.scaling;
        }

        // Update memory ring buffer.
        if self.mem_len > 0 {
            let start = self.mem_index * self.num_elem;
            self.memory[start..start + self.num_elem].copy_from_slice(input);
            self.mem_index = (self.mem_index + 1) % self.mem_len;
        }
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

        // First call: only data1, memory is zeros.
        ma.average(&data1, &mut output);
        for i in 0..num_elem {
            assert!((output[i] - data1[i] / 3.0).abs() < e, "step 1, elem {i}");
        }

        // Second call: data1 + data2 in memory.
        ma.average(&data2, &mut output);
        for i in 0..num_elem {
            assert!(
                (output[i] - (data1[i] + data2[i]) / 3.0).abs() < e,
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
