//! Multi-channel content detection — determines whether audio is proper
//! multichannel or upmixed mono.
//!
//! Ported from
//! `modules/audio_processing/aec3/multi_channel_content_detector.h/cc`.

const NUM_FRAMES_PER_SECOND: i32 = 100;

/// Checks whether any sample pair across bands exceeds the detection threshold.
fn has_stereo_content(frame: &[Vec<Vec<f32>>], detection_threshold: f32) -> bool {
    if frame[0].len() < 2 {
        return false;
    }
    for band in frame {
        for k in 0..band[0].len() {
            if (band[0][k] - band[1][k]).abs() > detection_threshold {
                return true;
            }
        }
    }
    false
}

/// Analyzes audio content to determine whether it is proper multichannel
/// or only upmixed mono.
pub(crate) struct MultiChannelContentDetector {
    detect_stereo_content: bool,
    detection_threshold: f32,
    detection_timeout_threshold_frames: Option<i64>,
    stereo_detection_hysteresis_frames: i64,
    persistent_multichannel_content_detected: bool,
    temporary_multichannel_content_detected: bool,
    frames_since_stereo_detected_last: i64,
    consecutive_frames_with_stereo: i64,
}

impl MultiChannelContentDetector {
    pub(crate) fn new(
        detect_stereo_content: bool,
        num_render_input_channels: usize,
        detection_threshold: f32,
        stereo_detection_timeout_threshold_seconds: i32,
        stereo_detection_hysteresis_seconds: f32,
    ) -> Self {
        let detection_timeout_threshold_frames = if stereo_detection_timeout_threshold_seconds > 0 {
            Some(stereo_detection_timeout_threshold_seconds as i64 * NUM_FRAMES_PER_SECOND as i64)
        } else {
            None
        };

        Self {
            detect_stereo_content,
            detection_threshold,
            detection_timeout_threshold_frames,
            stereo_detection_hysteresis_frames: (stereo_detection_hysteresis_seconds
                * NUM_FRAMES_PER_SECOND as f32)
                as i64,
            persistent_multichannel_content_detected: !detect_stereo_content
                && num_render_input_channels > 1,
            temporary_multichannel_content_detected: false,
            frames_since_stereo_detected_last: 0,
            consecutive_frames_with_stereo: 0,
        }
    }

    /// Updates the detection with a new frame. Returns `true` if the
    /// persistent multichannel detection status changed.
    ///
    /// `frame` is indexed as `[band][channel][sample]`.
    pub(crate) fn update_detection(&mut self, frame: &[Vec<Vec<f32>>]) -> bool {
        if !self.detect_stereo_content {
            debug_assert_eq!(
                frame[0].len() > 1,
                self.persistent_multichannel_content_detected
            );
            return false;
        }

        let previous = self.persistent_multichannel_content_detected;
        let stereo_detected_in_frame = has_stereo_content(frame, self.detection_threshold);

        self.consecutive_frames_with_stereo = if stereo_detected_in_frame {
            self.consecutive_frames_with_stereo + 1
        } else {
            0
        };
        self.frames_since_stereo_detected_last = if stereo_detected_in_frame {
            0
        } else {
            self.frames_since_stereo_detected_last + 1
        };

        // Detect persistent multichannel content.
        if self.consecutive_frames_with_stereo > self.stereo_detection_hysteresis_frames {
            self.persistent_multichannel_content_detected = true;
        }
        if let Some(timeout) = self.detection_timeout_threshold_frames {
            if self.frames_since_stereo_detected_last >= timeout {
                self.persistent_multichannel_content_detected = false;
            }
        }

        // Detect temporary multichannel content.
        self.temporary_multichannel_content_detected =
            if self.persistent_multichannel_content_detected {
                false
            } else {
                stereo_detected_in_frame
            };

        previous != self.persistent_multichannel_content_detected
    }

    /// Returns whether persistent multichannel content has been detected.
    pub(crate) fn is_proper_multi_channel_content_detected(&self) -> bool {
        self.persistent_multichannel_content_detected
    }

    /// Returns whether temporary multichannel content has been detected.
    pub(crate) fn is_temporary_multi_channel_content_detected(&self) -> bool {
        self.temporary_multichannel_content_detected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mono_frame(num_bands: usize, samples_per_band: usize) -> Vec<Vec<Vec<f32>>> {
        // 2 channels, identical content.
        (0..num_bands)
            .map(|_| {
                let ch = vec![0.0f32; samples_per_band];
                vec![ch.clone(), ch]
            })
            .collect()
    }

    fn make_stereo_frame(
        num_bands: usize,
        samples_per_band: usize,
        diff: f32,
    ) -> Vec<Vec<Vec<f32>>> {
        (0..num_bands)
            .map(|_| {
                let ch0 = vec![0.0f32; samples_per_band];
                let ch1 = vec![diff; samples_per_band];
                vec![ch0, ch1]
            })
            .collect()
    }

    #[test]
    fn mono_input_not_detected_as_stereo() {
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, 300, 2.0);
        let frame = make_mono_frame(1, 160);
        for _ in 0..500 {
            detector.update_detection(&frame);
        }
        assert!(!detector.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn stereo_content_detected_after_hysteresis() {
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, 300, 0.5);
        let frame = make_stereo_frame(1, 160, 1.0);
        let hysteresis_frames = (0.5 * NUM_FRAMES_PER_SECOND as f32) as i32;
        // Should not be detected yet.
        for _ in 0..hysteresis_frames {
            detector.update_detection(&frame);
        }
        // One more frame should trigger persistent detection.
        detector.update_detection(&frame);
        assert!(detector.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn detection_disabled_with_flag() {
        let mut detector = MultiChannelContentDetector::new(false, 2, 0.0, 300, 2.0);
        let frame = make_stereo_frame(1, 160, 1.0);
        // When detection is disabled and num_channels > 1, persistent is always true.
        assert!(detector.is_proper_multi_channel_content_detected());
        let changed = detector.update_detection(&frame);
        assert!(!changed);
    }

    #[test]
    fn single_channel_not_detected() {
        let mut detector = MultiChannelContentDetector::new(true, 1, 0.0, 300, 2.0);
        let frame = vec![vec![vec![0.0f32; 160]]]; // 1 channel
        detector.update_detection(&frame);
        assert!(!detector.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn timeout_resets_detection() {
        let timeout_seconds = 1;
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, timeout_seconds, 0.0);
        let stereo_frame = make_stereo_frame(1, 160, 1.0);
        let mono_frame = make_mono_frame(1, 160);

        // First make it detect stereo (hysteresis = 0, so immediate).
        detector.update_detection(&stereo_frame);
        assert!(detector.is_proper_multi_channel_content_detected());

        // Now feed mono frames for timeout duration.
        let timeout_frames = timeout_seconds * NUM_FRAMES_PER_SECOND;
        for _ in 0..timeout_frames {
            detector.update_detection(&mono_frame);
        }
        assert!(!detector.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn temporary_detection_when_not_persistent() {
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, 300, 10.0);
        let stereo_frame = make_stereo_frame(1, 160, 1.0);
        // Hysteresis is very long (10s), so we won't get persistent.
        detector.update_detection(&stereo_frame);
        assert!(!detector.is_proper_multi_channel_content_detected());
        assert!(detector.is_temporary_multi_channel_content_detected());
    }

    #[test]
    fn threshold_filters_small_differences() {
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.5, 300, 0.0);
        // Difference of 0.1 < threshold of 0.5 → no detection.
        let frame = make_stereo_frame(1, 160, 0.1);
        for _ in 0..200 {
            detector.update_detection(&frame);
        }
        assert!(!detector.is_proper_multi_channel_content_detected());
    }

    #[test]
    fn no_timeout_keeps_persistent() {
        // timeout_seconds <= 0 means no timeout.
        let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, 0, 0.0);
        let stereo_frame = make_stereo_frame(1, 160, 1.0);
        let mono_frame = make_mono_frame(1, 160);

        detector.update_detection(&stereo_frame);
        assert!(detector.is_proper_multi_channel_content_detected());

        // Feed lots of mono — should stay persistent.
        for _ in 0..1000 {
            detector.update_detection(&mono_frame);
        }
        assert!(detector.is_proper_multi_channel_content_detected());
    }
}
