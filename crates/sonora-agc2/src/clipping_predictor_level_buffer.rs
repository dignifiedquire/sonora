//! Circular buffer for frame-wise level metrics used by the clipping predictor.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/clipping_predictor_level_buffer.h/cc`.

/// Recommended maximum capacity. It is possible to create a buffer with a
/// larger capacity, but the implementation is not optimized for large values.
pub(crate) const MAX_CAPACITY: usize = 100;

/// Frame-wise level metrics: average (mean squared) and max (peak).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Level {
    pub average: f32,
    pub max: f32,
}

impl PartialEq for Level {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6;
        (self.average - other.average).abs() < EPSILON && (self.max - other.max).abs() < EPSILON
    }
}

/// A circular buffer to store frame-wise [`Level`] items for clipping prediction.
/// The current implementation is not optimized for large buffer lengths.
pub(crate) struct ClippingPredictorLevelBuffer {
    tail: isize,
    size: usize,
    data: Vec<Level>,
}

impl ClippingPredictorLevelBuffer {
    /// Creates a new buffer with the given capacity (minimum 1).
    /// Logs a warning if capacity exceeds [`MAX_CAPACITY`].
    pub(crate) fn new(capacity: i32) -> Self {
        let capacity = capacity.max(1) as usize;
        if capacity > MAX_CAPACITY {
            tracing::warn!(
                "[agc]: ClippingPredictorLevelBuffer exceeds the maximum allowed capacity. Capacity: {}",
                capacity
            );
        }
        Self {
            tail: -1,
            size: 0,
            data: vec![
                Level {
                    average: 0.0,
                    max: 0.0
                };
                capacity
            ],
        }
    }

    /// Resets the buffer, discarding all stored items.
    pub(crate) fn reset(&mut self) {
        self.tail = -1;
        self.size = 0;
    }

    /// Returns the current number of items stored in the buffer.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Returns the capacity of the buffer.
    pub(crate) fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Pushes a [`Level`] item into the circular buffer. If the buffer is full,
    /// the oldest item is replaced.
    pub(crate) fn push(&mut self, level: Level) {
        self.tail += 1;
        if self.tail as usize == self.capacity() {
            self.tail = 0;
        }
        if self.size < self.capacity() {
            self.size += 1;
        }
        self.data[self.tail as usize] = level;
    }

    /// If at least `num_items + delay` items have been pushed, returns the
    /// average and maximum value for the `num_items` most recently pushed items
    /// starting from `delay` positions back (delay=0 is the most recent item).
    pub(crate) fn compute_partial_metrics(&self, delay: usize, num_items: usize) -> Option<Level> {
        debug_assert!(delay < self.capacity());
        debug_assert!(num_items > 0);
        debug_assert!(num_items <= self.capacity());
        debug_assert!(delay + num_items <= self.capacity());

        if delay + num_items > self.size {
            return None;
        }

        let mut sum = 0.0_f32;
        let mut max = 0.0_f32;
        for i in 0..num_items.min(self.size) {
            let mut idx = self.tail - delay as isize - i as isize;
            if idx < 0 {
                idx += self.capacity() as isize;
            }
            sum += self.data[idx as usize].average;
            max = max.max(self.data[idx as usize].max);
        }
        Some(Level {
            average: sum / num_items as f32,
            max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Parametrized test helper: checks empty buffer size for various capacities.
    #[test]
    fn check_empty_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let buffer = ClippingPredictorLevelBuffer::new(capacity);
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), 0);
        }
    }

    #[test]
    fn check_half_empty_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..buffer.capacity() / 2 {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), capacity.max(1) as usize / 2);
        }
    }

    #[test]
    fn check_full_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..buffer.capacity() {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), capacity.max(1) as usize);
        }
    }

    #[test]
    fn check_large_buffer_size() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            for _ in 0..2 * buffer.capacity() {
                buffer.push(Level {
                    average: 2.0,
                    max: 4.0,
                });
            }
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), capacity.max(1) as usize);
        }
    }

    #[test]
    fn check_size_after_reset() {
        for &capacity in &[-1, 0, 1, 123] {
            let mut buffer = ClippingPredictorLevelBuffer::new(capacity);
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            buffer.reset();
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), 0);
            buffer.push(Level {
                average: 1.0,
                max: 1.0,
            });
            assert_eq!(buffer.capacity(), capacity.max(1) as usize);
            assert_eq!(buffer.size(), 1);
        }
    }

    #[test]
    fn check_metrics_after_full_buffer() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });

        let m = buffer.compute_partial_metrics(0, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 3.0,
                max: 6.0
            }
        );

        let m = buffer.compute_partial_metrics(1, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 1.0,
                max: 2.0
            }
        );

        let m = buffer.compute_partial_metrics(0, 2).unwrap();
        assert_eq!(
            m,
            Level {
                average: 2.0,
                max: 6.0
            }
        );
    }

    #[test]
    fn check_metrics_after_push_beyond_capacity() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 1.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });
        buffer.push(Level {
            average: 5.0,
            max: 10.0,
        });
        buffer.push(Level {
            average: 7.0,
            max: 14.0,
        });
        buffer.push(Level {
            average: 6.0,
            max: 12.0,
        });

        let m = buffer.compute_partial_metrics(0, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 6.0,
                max: 12.0
            }
        );

        let m = buffer.compute_partial_metrics(1, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 7.0,
                max: 14.0
            }
        );

        let m = buffer.compute_partial_metrics(0, 2).unwrap();
        assert_eq!(
            m,
            Level {
                average: 6.5,
                max: 14.0
            }
        );
    }

    #[test]
    fn check_metrics_after_too_few_items() {
        let mut buffer = ClippingPredictorLevelBuffer::new(4);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.push(Level {
            average: 3.0,
            max: 6.0,
        });

        assert!(buffer.compute_partial_metrics(0, 3).is_none());
        assert!(buffer.compute_partial_metrics(2, 1).is_none());
    }

    #[test]
    fn check_metrics_after_reset() {
        let mut buffer = ClippingPredictorLevelBuffer::new(2);
        buffer.push(Level {
            average: 1.0,
            max: 2.0,
        });
        buffer.reset();
        buffer.push(Level {
            average: 5.0,
            max: 10.0,
        });
        buffer.push(Level {
            average: 7.0,
            max: 14.0,
        });

        let m = buffer.compute_partial_metrics(0, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 7.0,
                max: 14.0
            }
        );

        let m = buffer.compute_partial_metrics(0, 2).unwrap();
        assert_eq!(
            m,
            Level {
                average: 6.0,
                max: 14.0
            }
        );

        let m = buffer.compute_partial_metrics(1, 1).unwrap();
        assert_eq!(
            m,
            Level {
                average: 5.0,
                max: 10.0
            }
        );
    }
}
