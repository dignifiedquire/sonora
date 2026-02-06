//! Speech level estimator based on framewise RMS and speech probability.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/speech_level_estimator_impl.h/.cc`.
//!
//! The C++ code has a factory pattern with a field trial for an experimental
//! implementation. We port only the default `SpeechLevelEstimatorImpl`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{
    FRAME_DURATION_MS, LEVEL_ESTIMATOR_LEAK_FACTOR, LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS,
    SATURATION_PROTECTOR_INITIAL_HEADROOM_DB, VAD_CONFIDENCE_THRESHOLD,
};

fn clamp_level_estimate_dbfs(level_estimate_dbfs: f32) -> f32 {
    level_estimate_dbfs.clamp(-90.0, 30.0)
}

/// Configuration for the adaptive digital gain controller that affects the
/// speech level estimator.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveDigitalConfig {
    pub(crate) headroom_db: f32,
    pub(crate) initial_gain_db: f32,
}

impl Default for AdaptiveDigitalConfig {
    fn default() -> Self {
        Self {
            headroom_db: 5.0,
            initial_gain_db: 15.0,
        }
    }
}

/// Returns the initial speech level estimate needed to apply the initial gain.
fn get_initial_speech_level_estimate_dbfs(config: &AdaptiveDigitalConfig) -> f32 {
    clamp_level_estimate_dbfs(
        -SATURATION_PROTECTOR_INITIAL_HEADROOM_DB - config.initial_gain_db - config.headroom_db,
    )
}

/// Part of the level estimator state used for check-pointing and restore ops.
#[derive(Clone, Copy)]
struct LevelEstimatorState {
    /// Time remaining until the estimator becomes confident.
    time_to_confidence_ms: i32,
    /// Weighted ratio for level estimation.
    level_dbfs: Ratio,
}

#[derive(Clone, Copy)]
struct Ratio {
    numerator: f32,
    denominator: f32,
}

impl Ratio {
    fn get_ratio(&self) -> f32 {
        debug_assert!(self.denominator != 0.0);
        self.numerator / self.denominator
    }
}

/// Active speech level estimator based on the analysis of the following
/// framewise properties: RMS level (dBFS), speech probability.
pub(crate) struct SpeechLevelEstimator {
    initial_speech_level_dbfs: f32,
    adjacent_speech_frames_threshold: i32,
    preliminary_state: LevelEstimatorState,
    reliable_state: LevelEstimatorState,
    level_dbfs: f32,
    is_confident: bool,
    num_adjacent_speech_frames: i32,
}

impl SpeechLevelEstimator {
    /// Creates a new speech level estimator.
    pub(crate) fn new(
        config: &AdaptiveDigitalConfig,
        adjacent_speech_frames_threshold: i32,
    ) -> Self {
        debug_assert!(adjacent_speech_frames_threshold >= 1);
        let initial_speech_level_dbfs = get_initial_speech_level_estimate_dbfs(config);
        let mut est = Self {
            initial_speech_level_dbfs,
            adjacent_speech_frames_threshold,
            preliminary_state: LevelEstimatorState {
                time_to_confidence_ms: 0,
                level_dbfs: Ratio {
                    numerator: 0.0,
                    denominator: 1.0,
                },
            },
            reliable_state: LevelEstimatorState {
                time_to_confidence_ms: 0,
                level_dbfs: Ratio {
                    numerator: 0.0,
                    denominator: 1.0,
                },
            },
            level_dbfs: initial_speech_level_dbfs,
            is_confident: false,
            num_adjacent_speech_frames: 0,
        };
        est.reset();
        est
    }

    /// Updates the level estimation.
    pub(crate) fn update(&mut self, rms_dbfs: f32, speech_probability: f32) {
        debug_assert!(rms_dbfs > -150.0);
        debug_assert!(rms_dbfs < 50.0);
        debug_assert!(speech_probability >= 0.0);
        debug_assert!(speech_probability <= 1.0);

        if speech_probability < VAD_CONFIDENCE_THRESHOLD {
            // Not a speech frame.
            if self.adjacent_speech_frames_threshold > 1 {
                // When two or more adjacent speech frames are required in order to
                // update the state, we need to decide whether to discard or confirm
                // the updates based on the speech sequence length.
                if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                    // First non-speech frame after a long enough sequence of speech
                    // frames. Update the reliable state.
                    self.reliable_state = self.preliminary_state;
                } else if self.num_adjacent_speech_frames > 0 {
                    // First non-speech frame after a too short sequence of speech
                    // frames. Reset to the last reliable state.
                    self.preliminary_state = self.reliable_state;
                }
            }
            self.num_adjacent_speech_frames = 0;
        } else {
            // Speech frame observed.
            self.num_adjacent_speech_frames += 1;

            // Update preliminary level estimate.
            debug_assert!(self.preliminary_state.time_to_confidence_ms >= 0);
            let buffer_is_full = self.preliminary_state.time_to_confidence_ms == 0;
            if !buffer_is_full {
                self.preliminary_state.time_to_confidence_ms -= FRAME_DURATION_MS;
            }
            // Weighted average of levels with speech probability as weight.
            debug_assert!(speech_probability > 0.0);
            let leak_factor = if buffer_is_full {
                LEVEL_ESTIMATOR_LEAK_FACTOR
            } else {
                1.0
            };
            self.preliminary_state.level_dbfs.numerator =
                self.preliminary_state.level_dbfs.numerator * leak_factor
                    + rms_dbfs * speech_probability;
            self.preliminary_state.level_dbfs.denominator =
                self.preliminary_state.level_dbfs.denominator * leak_factor + speech_probability;

            let level_dbfs = self.preliminary_state.level_dbfs.get_ratio();

            if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                // `preliminary_state` is now reliable. Update the last level estimation.
                self.level_dbfs = clamp_level_estimate_dbfs(level_dbfs);
            }
        }
        self.update_is_confident();
    }

    /// Returns the estimated speech plus noise level.
    pub(crate) fn level_dbfs(&self) -> f32 {
        self.level_dbfs
    }

    /// Returns true if the estimator is confident on its current estimate.
    pub(crate) fn is_confident(&self) -> bool {
        self.is_confident
    }

    /// Resets the estimator.
    pub(crate) fn reset(&mut self) {
        self.reset_level_estimator_state(&mut self.preliminary_state.clone());
        let preliminary = self.make_initial_state();
        self.preliminary_state = preliminary;
        let reliable = self.make_initial_state();
        self.reliable_state = reliable;
        self.level_dbfs = self.initial_speech_level_dbfs;
        self.num_adjacent_speech_frames = 0;
    }

    fn update_is_confident(&mut self) {
        if self.adjacent_speech_frames_threshold == 1 {
            // Ignore `reliable_state` when a single frame is enough to update the
            // level estimate (because it is not used).
            self.is_confident = self.preliminary_state.time_to_confidence_ms == 0;
            return;
        }
        // Once confident, it remains confident.
        // During the first long enough speech sequence, `reliable_state` must be
        // ignored since `preliminary_state` is used.
        self.is_confident = self.reliable_state.time_to_confidence_ms == 0
            || (self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold
                && self.preliminary_state.time_to_confidence_ms == 0);
    }

    fn make_initial_state(&self) -> LevelEstimatorState {
        LevelEstimatorState {
            time_to_confidence_ms: LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS as i32,
            level_dbfs: Ratio {
                numerator: self.initial_speech_level_dbfs,
                denominator: 1.0,
            },
        }
    }

    fn reset_level_estimator_state(&self, state: &mut LevelEstimatorState) {
        state.time_to_confidence_ms = LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS as i32;
        state.level_dbfs.numerator = self.initial_speech_level_dbfs;
        state.level_dbfs.denominator = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Number of speech frames that the level estimator must observe in order to
    /// become confident about the estimated level.
    const NUM_FRAMES_TO_CONFIDENCE: i32 =
        LEVEL_ESTIMATOR_TIME_TO_CONFIDENCE_MS as i32 / FRAME_DURATION_MS;

    const CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE: f32 = 0.5;

    const NO_SPEECH_PROBABILITY: f32 = 0.0;
    const LOW_SPEECH_PROBABILITY: f32 = VAD_CONFIDENCE_THRESHOLD / 2.0;
    const MAX_SPEECH_PROBABILITY: f32 = 1.0;

    /// Provides the level values `num_iterations` times to `level_estimator`.
    fn run_on_constant_level(
        num_iterations: i32,
        rms_dbfs: f32,
        speech_probability: f32,
        level_estimator: &mut SpeechLevelEstimator,
    ) {
        for _ in 0..num_iterations {
            level_estimator.update(rms_dbfs, speech_probability);
        }
    }

    /// Level estimator helper for tests.
    struct TestLevelEstimator {
        estimator: SpeechLevelEstimator,
        initial_speech_level_dbfs: f32,
        level_rms_dbfs: f32,
        #[allow(dead_code, reason = "matches C++ test struct")]
        level_peak_dbfs: f32,
    }

    impl TestLevelEstimator {
        fn new(adjacent_speech_frames_threshold: i32) -> Self {
            let config = AdaptiveDigitalConfig::default();
            let estimator = SpeechLevelEstimator::new(&config, adjacent_speech_frames_threshold);
            let initial_speech_level_dbfs = estimator.level_dbfs();
            let level_rms_dbfs = initial_speech_level_dbfs / 2.0;
            let level_peak_dbfs = initial_speech_level_dbfs / 3.0;
            debug_assert!(level_rms_dbfs < level_peak_dbfs);
            debug_assert!(initial_speech_level_dbfs < level_rms_dbfs);
            debug_assert!(
                level_rms_dbfs - initial_speech_level_dbfs > 5.0,
                "Adjust `level_rms_dbfs` so that the difference from the initial \
                 level is wide enough for the tests"
            );
            Self {
                estimator,
                initial_speech_level_dbfs,
                level_rms_dbfs,
                level_peak_dbfs,
            }
        }
    }

    #[test]
    fn level_stabilizes() {
        let mut t = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        let estimated_level_dbfs = t.estimator.level_dbfs();
        run_on_constant_level(
            1,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        assert!(
            (t.estimator.level_dbfs() - estimated_level_dbfs).abs() < 0.1,
            "level {} should be near {}",
            t.estimator.level_dbfs(),
            estimated_level_dbfs
        );
    }

    #[test]
    fn is_not_confident() {
        let mut t = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE / 2,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        assert!(!t.estimator.is_confident());
    }

    #[test]
    fn is_confident() {
        let mut t = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        assert!(t.estimator.is_confident());
    }

    #[test]
    fn estimator_ignores_non_speech_frames() {
        let mut t = TestLevelEstimator::new(1);
        // Simulate speech.
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        let estimated_level_dbfs = t.estimator.level_dbfs();
        // Simulate full-scale non-speech.
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            0.0,
            NO_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        // No estimated level change is expected.
        assert_eq!(t.estimator.level_dbfs(), estimated_level_dbfs);
    }

    #[test]
    fn convergence_speed_before_confidence() {
        let mut t = TestLevelEstimator::new(1);
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        assert!(
            (t.estimator.level_dbfs() - t.level_rms_dbfs).abs()
                <= CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE,
            "level {} should be near {}",
            t.estimator.level_dbfs(),
            t.level_rms_dbfs
        );
    }

    #[test]
    fn convergence_speed_after_confidence() {
        let mut t = TestLevelEstimator::new(1);
        // Reach confidence using the initial level estimate.
        run_on_constant_level(
            NUM_FRAMES_TO_CONFIDENCE,
            t.initial_speech_level_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        // No estimate change should occur, but confidence is achieved.
        assert_eq!(t.estimator.level_dbfs(), t.initial_speech_level_dbfs);
        assert!(t.estimator.is_confident());
        // After confidence.
        let convergence_time_after_confidence_num_frames = 700; // 7 seconds.
        assert!(convergence_time_after_confidence_num_frames > NUM_FRAMES_TO_CONFIDENCE);
        run_on_constant_level(
            convergence_time_after_confidence_num_frames,
            t.level_rms_dbfs,
            MAX_SPEECH_PROBABILITY,
            &mut t.estimator,
        );
        assert!(
            (t.estimator.level_dbfs() - t.level_rms_dbfs).abs()
                <= CONVERGENCE_SPEED_TESTS_LEVEL_TOLERANCE,
            "level {} should be near {}",
            t.estimator.level_dbfs(),
            t.level_rms_dbfs
        );
    }

    // Parametrized tests for adjacent_speech_frames_threshold = 1, 9, 17.

    #[test]
    fn do_not_adapt_to_short_speech_segments_threshold_1() {
        do_not_adapt_to_short_speech_segments(1);
    }

    #[test]
    fn do_not_adapt_to_short_speech_segments_threshold_9() {
        do_not_adapt_to_short_speech_segments(9);
    }

    #[test]
    fn do_not_adapt_to_short_speech_segments_threshold_17() {
        do_not_adapt_to_short_speech_segments(17);
    }

    fn do_not_adapt_to_short_speech_segments(threshold: i32) {
        let mut t = TestLevelEstimator::new(threshold);
        let initial_level = t.estimator.level_dbfs();
        assert!(initial_level < t.level_peak_dbfs);
        for _ in 0..threshold - 1 {
            t.estimator.update(t.level_rms_dbfs, MAX_SPEECH_PROBABILITY);
            assert_eq!(
                initial_level,
                t.estimator.level_dbfs(),
                "level should not change before threshold"
            );
        }
        t.estimator.update(t.level_rms_dbfs, LOW_SPEECH_PROBABILITY);
        assert_eq!(
            initial_level,
            t.estimator.level_dbfs(),
            "level should not change after low-probability frame"
        );
    }

    #[test]
    fn adapt_to_enough_speech_segments_threshold_1() {
        adapt_to_enough_speech_segments(1);
    }

    #[test]
    fn adapt_to_enough_speech_segments_threshold_9() {
        adapt_to_enough_speech_segments(9);
    }

    #[test]
    fn adapt_to_enough_speech_segments_threshold_17() {
        adapt_to_enough_speech_segments(17);
    }

    fn adapt_to_enough_speech_segments(threshold: i32) {
        let mut t = TestLevelEstimator::new(threshold);
        let initial_level = t.estimator.level_dbfs();
        assert!(initial_level < t.level_peak_dbfs);
        for _ in 0..threshold {
            t.estimator.update(t.level_rms_dbfs, MAX_SPEECH_PROBABILITY);
        }
        assert!(
            initial_level < t.estimator.level_dbfs(),
            "level should increase after enough speech frames"
        );
    }
}
