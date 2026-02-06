//! Circular buffer for speech probabilities with transient removal.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/speech_probability_buffer.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

const ACTIVITY_THRESHOLD: f32 = 0.9;
const NUM_ANALYSIS_FRAMES: usize = 100;
const TRANSIENT_WIDTH_THRESHOLD: i32 = 7;

/// Circular buffer that stores speech probabilities for a speech segment
/// and estimates speech activity for that segment.
pub(crate) struct SpeechProbabilityBuffer {
    low_probability_threshold: f32,
    sum_probabilities: f32,
    probabilities: Vec<f32>,
    buffer_index: usize,
    buffer_is_full: bool,
    num_high_probability_observations: i32,
}

impl SpeechProbabilityBuffer {
    /// Creates a new buffer. `low_probability_threshold` must be in `[0.0, 1.0]`.
    pub(crate) fn new(low_probability_threshold: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&low_probability_threshold));
        Self {
            low_probability_threshold,
            sum_probabilities: 0.0,
            probabilities: vec![0.0; NUM_ANALYSIS_FRAMES],
            buffer_index: 0,
            buffer_is_full: false,
            num_high_probability_observations: 0,
        }
    }

    /// Adds `probability` to the buffer and updates the running sum.
    /// `probability` must be in `[0.0, 1.0]`.
    pub(crate) fn update(&mut self, mut probability: f32) {
        // Remove the oldest entry if the circular buffer is full.
        if self.buffer_is_full {
            self.sum_probabilities -= self.probabilities[self.buffer_index];
        }

        // Check for transients.
        if probability <= self.low_probability_threshold {
            probability = 0.0;

            if self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD {
                self.remove_transient();
            }
            self.num_high_probability_observations = 0;
        } else if self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD {
            self.num_high_probability_observations += 1;
        }

        // Update the circular buffer and the current sum.
        self.probabilities[self.buffer_index] = probability;
        self.sum_probabilities += probability;

        // Increment the buffer index and check for wrap-around.
        self.buffer_index += 1;
        if self.buffer_index >= NUM_ANALYSIS_FRAMES {
            self.buffer_index = 0;
            self.buffer_is_full = true;
        }
    }

    /// Resets the buffer, forgetting the past.
    pub(crate) fn reset(&mut self) {
        self.sum_probabilities = 0.0;
        self.buffer_index = 0;
        self.buffer_is_full = false;
        self.num_high_probability_observations = 0;
    }

    /// Returns true if the segment is active (a long enough segment with
    /// average speech probability above `low_probability_threshold`).
    pub(crate) fn is_active_segment(&self) -> bool {
        self.buffer_is_full
            && self.sum_probabilities >= ACTIVITY_THRESHOLD * NUM_ANALYSIS_FRAMES as f32
    }

    /// Returns the current sum of probabilities (for testing).
    #[cfg(test)]
    fn get_sum_probabilities(&self) -> f32 {
        self.sum_probabilities
    }

    fn remove_transient(&mut self) {
        debug_assert!(self.num_high_probability_observations <= TRANSIENT_WIDTH_THRESHOLD);

        let mut index = if self.buffer_index > 0 {
            self.buffer_index - 1
        } else {
            NUM_ANALYSIS_FRAMES - 1
        };

        let mut count = self.num_high_probability_observations;
        while count > 0 {
            count -= 1;
            self.sum_probabilities -= self.probabilities[index];
            self.probabilities[index] = 0.0;

            index = if index > 0 {
                index - 1
            } else {
                NUM_ANALYSIS_FRAMES - 1
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOW_PROB_THRESHOLD: f32 = 0.5;

    #[test]
    fn check_sum_after_initialization() {
        let buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        assert_eq!(buffer.get_sum_probabilities(), 0.0);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_sum_after_update() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        buffer.update(0.8);
        assert!((buffer.get_sum_probabilities() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn check_sum_after_reset() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        for _ in 0..50 {
            buffer.update(0.8);
        }
        buffer.reset();
        assert_eq!(buffer.get_sum_probabilities(), 0.0);
        assert!(!buffer.is_active_segment());
    }

    #[test]
    fn check_sum_after_transient_not_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        // Add enough high-probability observations to exceed the transient threshold.
        for _ in 0..TRANSIENT_WIDTH_THRESHOLD + 1 {
            buffer.update(0.8);
        }
        let expected_sum = 0.8 * (TRANSIENT_WIDTH_THRESHOLD + 1) as f32;
        assert!(
            (buffer.get_sum_probabilities() - expected_sum).abs() < 1e-5,
            "sum {} != expected {}",
            buffer.get_sum_probabilities(),
            expected_sum
        );
    }

    #[test]
    fn check_sum_after_transient_removed() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        // Add a few high-probability observations (within transient threshold).
        for _ in 0..TRANSIENT_WIDTH_THRESHOLD {
            buffer.update(0.8);
        }
        // Then a low-probability observation triggers transient removal.
        buffer.update(0.1);
        // The transient should have been removed, sum should be ~0.
        assert!(
            buffer.get_sum_probabilities().abs() < 1e-5,
            "sum {} should be ~0 after transient removal",
            buffer.get_sum_probabilities()
        );
    }

    #[test]
    fn is_active_after_full_high_probability() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        // Fill buffer with high probability values.
        for _ in 0..NUM_ANALYSIS_FRAMES {
            buffer.update(1.0);
        }
        assert!(buffer.is_active_segment());
    }

    #[test]
    fn is_not_active_before_full() {
        let mut buffer = SpeechProbabilityBuffer::new(LOW_PROB_THRESHOLD);
        for _ in 0..NUM_ANALYSIS_FRAMES - 1 {
            buffer.update(1.0);
        }
        assert!(!buffer.is_active_segment());
    }
}
