//! API call jitter metrics — tracks render/capture call ordering and timing.
//!
//! Ported from `modules/audio_processing/aec3/api_call_jitter_metrics.h/cc`.

/// Tracks min and max values for jitter reporting.
#[derive(Debug, Clone)]
pub(crate) struct Jitter {
    min: i32,
    max: i32,
}

impl Jitter {
    fn new() -> Self {
        Self {
            max: 0,
            min: i32::MAX,
        }
    }

    fn update(&mut self, num_api_calls_in_a_row: i32) {
        self.min = self.min.min(num_api_calls_in_a_row);
        self.max = self.max.max(num_api_calls_in_a_row);
    }

    fn reset(&mut self) {
        self.min = i32::MAX;
        self.max = 0;
    }

    pub(crate) fn min(&self) -> i32 {
        self.min
    }

    pub(crate) fn max(&self) -> i32 {
        self.max
    }
}

/// Returns true when it's time to report metrics.
fn time_to_report_metrics(frames_since_last_report: i32) -> bool {
    const NUM_FRAMES_PER_SECOND: i32 = 100;
    const REPORTING_INTERVAL_FRAMES: i32 = 10 * NUM_FRAMES_PER_SECOND;
    frames_since_last_report == REPORTING_INTERVAL_FRAMES
}

/// Stores data for reporting metrics on the API call jitter.
pub(crate) struct ApiCallJitterMetrics {
    render_jitter: Jitter,
    capture_jitter: Jitter,
    num_api_calls_in_a_row: i32,
    frames_since_last_report: i32,
    last_call_was_render: bool,
    proper_call_observed: bool,
}

impl ApiCallJitterMetrics {
    pub(crate) fn new() -> Self {
        let mut s = Self {
            render_jitter: Jitter::new(),
            capture_jitter: Jitter::new(),
            num_api_calls_in_a_row: 0,
            frames_since_last_report: 0,
            last_call_was_render: false,
            proper_call_observed: false,
        };
        s.reset();
        s
    }

    /// Updates metrics for a render API call.
    pub(crate) fn report_render_call(&mut self) {
        if !self.last_call_was_render {
            // If the previous call was a capture and a proper call has been
            // observed, store the last number of capture calls.
            if self.proper_call_observed {
                self.capture_jitter.update(self.num_api_calls_in_a_row);
            }
            // Reset the call counter to start counting render calls.
            self.num_api_calls_in_a_row = 0;
        }
        self.num_api_calls_in_a_row += 1;
        self.last_call_was_render = true;
    }

    /// Updates and periodically reports metrics for a capture API call.
    pub(crate) fn report_capture_call(&mut self) {
        if self.last_call_was_render {
            // If the previous call was a render and a proper call has been
            // observed, store the last number of render calls.
            if self.proper_call_observed {
                self.render_jitter.update(self.num_api_calls_in_a_row);
            }
            // Reset the call counter to start counting capture calls.
            self.num_api_calls_in_a_row = 0;

            // If this statement is reached, at least one render and one capture
            // call have been observed.
            self.proper_call_observed = true;
        }
        self.num_api_calls_in_a_row += 1;
        self.last_call_was_render = false;

        // Only report and update jitter metrics when a proper call, containing
        // both render and capture data, has been observed.
        if self.proper_call_observed {
            self.frames_since_last_report += 1;
            if time_to_report_metrics(self.frames_since_last_report) {
                // RTC_HISTOGRAM reporting skipped (no metrics system in Rust port).
                self.frames_since_last_report = 0;
                self.reset();
            }
        }
    }

    /// Returns a reference to the render jitter tracker.
    pub(crate) fn render_jitter(&self) -> &Jitter {
        &self.render_jitter
    }

    /// Returns a reference to the capture jitter tracker.
    pub(crate) fn capture_jitter(&self) -> &Jitter {
        &self.capture_jitter
    }

    /// Returns true if metrics will be reported at the next capture call.
    pub(crate) fn will_report_metrics_at_next_capture(&self) -> bool {
        time_to_report_metrics(self.frames_since_last_report + 1)
    }

    fn reset(&mut self) {
        self.render_jitter.reset();
        self.capture_jitter.reset();
        self.num_api_calls_in_a_row = 0;
        self.frames_since_last_report = 0;
        self.last_call_was_render = false;
        self.proper_call_observed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::NUM_BLOCKS_PER_SECOND;

    #[test]
    fn constant_jitter() {
        for jitter in 1..20 {
            let mut metrics = ApiCallJitterMetrics::new();
            for _ in 0..30 * NUM_BLOCKS_PER_SECOND {
                for _ in 0..jitter {
                    metrics.report_render_call();
                }
                for _ in 0..jitter {
                    metrics.report_capture_call();

                    if metrics.will_report_metrics_at_next_capture() {
                        assert_eq!(jitter as i32, metrics.render_jitter().min());
                        assert_eq!(jitter as i32, metrics.render_jitter().max());
                        assert_eq!(jitter as i32, metrics.capture_jitter().min());
                        assert_eq!(jitter as i32, metrics.capture_jitter().max());
                    }
                }
            }
        }
    }

    #[test]
    fn jitter_peak_render() {
        const MIN_JITTER: i32 = 2;
        const JITTER_PEAK: i32 = 10;
        const PEAK_INTERVAL: usize = 100;

        let mut metrics = ApiCallJitterMetrics::new();
        let mut render_surplus: i32 = 0;

        for k in 0..30 * NUM_BLOCKS_PER_SECOND {
            let num_render_calls = if k % PEAK_INTERVAL == 0 {
                JITTER_PEAK
            } else {
                MIN_JITTER
            };
            for _ in 0..num_render_calls {
                metrics.report_render_call();
                render_surplus += 1;
            }

            assert!(MIN_JITTER <= render_surplus);
            let num_capture_calls = if render_surplus == MIN_JITTER {
                MIN_JITTER
            } else {
                MIN_JITTER + 1
            };
            for _ in 0..num_capture_calls {
                metrics.report_capture_call();

                if metrics.will_report_metrics_at_next_capture() {
                    assert_eq!(MIN_JITTER, metrics.render_jitter().min());
                    assert_eq!(JITTER_PEAK, metrics.render_jitter().max());
                    assert_eq!(MIN_JITTER, metrics.capture_jitter().min());
                    assert_eq!(MIN_JITTER + 1, metrics.capture_jitter().max());
                }
                render_surplus -= 1;
            }
        }
    }

    #[test]
    fn jitter_peak_capture() {
        const MIN_JITTER: i32 = 2;
        const JITTER_PEAK: i32 = 10;
        const PEAK_INTERVAL: usize = 100;

        let mut metrics = ApiCallJitterMetrics::new();
        let mut capture_surplus: i32 = MIN_JITTER;

        for k in 0..30 * NUM_BLOCKS_PER_SECOND {
            assert!(MIN_JITTER <= capture_surplus);
            let num_render_calls = if capture_surplus == MIN_JITTER {
                MIN_JITTER
            } else {
                MIN_JITTER + 1
            };
            for _ in 0..num_render_calls {
                metrics.report_render_call();
                capture_surplus -= 1;
            }

            let num_capture_calls = if k % PEAK_INTERVAL == 0 {
                JITTER_PEAK
            } else {
                MIN_JITTER
            };
            for _ in 0..num_capture_calls {
                metrics.report_capture_call();

                if metrics.will_report_metrics_at_next_capture() {
                    assert_eq!(MIN_JITTER, metrics.render_jitter().min());
                    assert_eq!(MIN_JITTER + 1, metrics.render_jitter().max());
                    assert_eq!(MIN_JITTER, metrics.capture_jitter().min());
                    assert_eq!(JITTER_PEAK, metrics.capture_jitter().max());
                }
                capture_surplus += 1;
            }
        }
    }
}
