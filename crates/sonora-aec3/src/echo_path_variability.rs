//! Echo path variability tracking.
//!
//! Ported from `modules/audio_processing/aec3/echo_path_variability.h/cc`.

/// Type of delay adjustment that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DelayAdjustment {
    None,
    BufferFlush,
    NewDetectedDelay,
}

/// Tracks changes in the echo path.
#[derive(Debug, Clone)]
pub(crate) struct EchoPathVariability {
    pub gain_change: bool,
    pub delay_change: DelayAdjustment,
    pub clock_drift: bool,
}

impl EchoPathVariability {
    pub(crate) fn new(gain_change: bool, delay_change: DelayAdjustment, clock_drift: bool) -> Self {
        Self {
            gain_change,
            delay_change,
            clock_drift,
        }
    }

    /// Returns whether the audio path has changed (gain or delay).
    pub(crate) fn audio_path_changed(&self) -> bool {
        self.gain_change || self.delay_change != DelayAdjustment::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_behavior() {
        // gain_change=true, delay=NewDetectedDelay
        let v = EchoPathVariability::new(true, DelayAdjustment::NewDetectedDelay, false);
        assert!(v.gain_change);
        assert_eq!(v.delay_change, DelayAdjustment::NewDetectedDelay);
        assert!(v.audio_path_changed());
        assert!(!v.clock_drift);

        // gain_change=true, delay=None
        let v = EchoPathVariability::new(true, DelayAdjustment::None, false);
        assert!(v.gain_change);
        assert_eq!(v.delay_change, DelayAdjustment::None);
        assert!(v.audio_path_changed());
        assert!(!v.clock_drift);

        // gain_change=false, delay=NewDetectedDelay
        let v = EchoPathVariability::new(false, DelayAdjustment::NewDetectedDelay, false);
        assert!(!v.gain_change);
        assert_eq!(v.delay_change, DelayAdjustment::NewDetectedDelay);
        assert!(v.audio_path_changed());
        assert!(!v.clock_drift);

        // gain_change=false, delay=None
        let v = EchoPathVariability::new(false, DelayAdjustment::None, false);
        assert!(!v.gain_change);
        assert_eq!(v.delay_change, DelayAdjustment::None);
        assert!(!v.audio_path_changed());
        assert!(!v.clock_drift);
    }
}
