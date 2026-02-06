//! Clock drift detection by analyzing estimated delay changes.
//!
//! Ported from `modules/audio_processing/aec3/clockdrift_detector.h/cc`.

/// Level of detected clock drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClockdriftLevel {
    None,
    Probable,
    Verified,
}

/// Detects clock drift by analyzing the estimated delay.
pub(crate) struct ClockdriftDetector {
    delay_history: [i32; 3],
    level: ClockdriftLevel,
    stability_counter: usize,
}

impl ClockdriftDetector {
    pub(crate) fn new() -> Self {
        Self {
            delay_history: [0; 3],
            level: ClockdriftLevel::None,
            stability_counter: 0,
        }
    }

    pub(crate) fn update(&mut self, delay_estimate: i32) {
        if delay_estimate == self.delay_history[0] {
            // Reset clockdrift level if delay estimate is stable for 7500
            // blocks (30 seconds).
            self.stability_counter += 1;
            if self.stability_counter > 7500 {
                self.level = ClockdriftLevel::None;
            }
            return;
        }

        self.stability_counter = 0;
        let d1 = self.delay_history[0] - delay_estimate;
        let d2 = self.delay_history[1] - delay_estimate;
        let d3 = self.delay_history[2] - delay_estimate;

        // Patterns recognized as positive clockdrift:
        // [x-3], x-2, x-1, x.
        // [x-3], x-1, x-2, x.
        let probable_drift_up = (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
        let drift_up = probable_drift_up && d3 == -3;

        // Patterns recognized as negative clockdrift:
        // [x+3], x+2, x+1, x.
        // [x+3], x+1, x+2, x.
        let probable_drift_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
        let drift_down = probable_drift_down && d3 == 3;

        // Set clockdrift level.
        if drift_up || drift_down {
            self.level = ClockdriftLevel::Verified;
        } else if (probable_drift_up || probable_drift_down) && self.level == ClockdriftLevel::None
        {
            self.level = ClockdriftLevel::Probable;
        }

        // Shift delay history one step.
        self.delay_history[2] = self.delay_history[1];
        self.delay_history[1] = self.delay_history[0];
        self.delay_history[0] = delay_estimate;
    }

    pub(crate) fn clockdrift_level(&self) -> ClockdriftLevel {
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clockdrift_detector() {
        let mut c = ClockdriftDetector::new();
        // No clockdrift at start.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::None);

        // Monotonically increasing delay.
        for _ in 0..100 {
            c.update(1000);
        }
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::None);

        for _ in 0..100 {
            c.update(1001);
        }
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::None);

        for _ in 0..100 {
            c.update(1002);
        }
        // Probable clockdrift.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::Probable);

        for _ in 0..100 {
            c.update(1003);
        }
        // Verified clockdrift.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::Verified);

        // Stable delay.
        for _ in 0..10000 {
            c.update(1003);
        }
        // No clockdrift.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::None);

        // Decreasing delay.
        for _ in 0..100 {
            c.update(1001);
        }
        for _ in 0..100 {
            c.update(999);
        }
        // Probable clockdrift.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::Probable);

        for _ in 0..100 {
            c.update(1000);
        }
        for _ in 0..100 {
            c.update(998);
        }
        // Verified clockdrift.
        assert_eq!(c.clockdrift_level(), ClockdriftLevel::Verified);
    }
}
