//! Adaptive digital gain controller.
//!
//! Selects the target digital gain, decides when and how quickly to adapt to
//! the target and applies the current gain to 10 ms frames.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/adaptive_digital_gain_controller.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{
    FRAME_DURATION_MS, LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS, VAD_CONFIDENCE_THRESHOLD, db_to_ratio,
};
use crate::gain_applier::GainApplier;
use crate::speech_level_estimator::AdaptiveDigitalConfig;

/// Information about a frame to process.
pub(crate) struct FrameInfo {
    /// Probability of speech in the [0, 1] range.
    pub(crate) speech_probability: f32,
    /// Estimated speech level (dBFS).
    pub(crate) speech_level_dbfs: f32,
    /// True with reliable speech level estimation.
    pub(crate) speech_level_reliable: bool,
    /// Estimated noise RMS level (dBFS).
    pub(crate) noise_rms_dbfs: f32,
    /// Headroom (dB).
    pub(crate) headroom_db: f32,
    /// Envelope level from the limiter (dBFS).
    pub(crate) limiter_envelope_dbfs: f32,
}

/// Computes the gain for `input_level_dbfs` to reach `-config.headroom_db`.
/// Clamps the gain in [0, `config.max_gain_db`].
fn compute_gain_db(input_level_dbfs: f32, config: &AdaptiveDigitalConfig) -> f32 {
    // If the level is very low, apply the maximum gain.
    if input_level_dbfs < -(config.headroom_db + config.max_gain_db) {
        return config.max_gain_db;
    }
    // We expect to end up here most of the time: the level is below
    // -headroom, but we can boost it to -headroom.
    if input_level_dbfs < -config.headroom_db {
        return -config.headroom_db - input_level_dbfs;
    }
    // The level is too high and we can't boost.
    0.0
}

/// Returns `target_gain_db` if applying such a gain to `input_noise_level_dbfs`
/// does not exceed `max_output_noise_level_dbfs`. Otherwise lowers the gain.
fn limit_gain_by_noise(
    target_gain_db: f32,
    input_noise_level_dbfs: f32,
    max_output_noise_level_dbfs: f32,
) -> f32 {
    let max_allowed_gain_db = max_output_noise_level_dbfs - input_noise_level_dbfs;
    target_gain_db.min(max_allowed_gain_db.max(0.0))
}

fn limit_gain_by_low_confidence(
    target_gain_db: f32,
    last_gain_db: f32,
    limiter_audio_level_dbfs: f32,
    estimate_is_confident: bool,
) -> f32 {
    if estimate_is_confident || limiter_audio_level_dbfs <= LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS {
        return target_gain_db;
    }
    let limiter_level_dbfs_before_gain = limiter_audio_level_dbfs - last_gain_db;

    // Compute a new gain so that `limiter_level_dbfs_before_gain` +
    // `new_target_gain_db` is not greater than the threshold.
    let new_target_gain_db =
        (LIMITER_THRESHOLD_FOR_AGC_GAIN_DBFS - limiter_level_dbfs_before_gain).max(0.0);
    new_target_gain_db.min(target_gain_db)
}

/// Computes how the gain should change during this frame.
fn compute_gain_change_this_frame_db(
    target_gain_db: f32,
    last_gain_db: f32,
    gain_increase_allowed: bool,
    max_gain_decrease_db: f32,
    max_gain_increase_db: f32,
) -> f32 {
    debug_assert!(max_gain_decrease_db > 0.0);
    debug_assert!(max_gain_increase_db > 0.0);
    let mut target_gain_difference_db = target_gain_db - last_gain_db;
    if !gain_increase_allowed {
        target_gain_difference_db = target_gain_difference_db.min(0.0);
    }
    target_gain_difference_db.clamp(-max_gain_decrease_db, max_gain_increase_db)
}

/// Adaptive digital gain controller.
pub(crate) struct AdaptiveDigitalGainController {
    gain_applier: GainApplier,
    config: AdaptiveDigitalConfig,
    adjacent_speech_frames_threshold: i32,
    max_gain_change_db_per_10ms: f32,
    frames_to_gain_increase_allowed: i32,
    last_gain_db: f32,
}

impl AdaptiveDigitalGainController {
    pub(crate) fn new(
        config: AdaptiveDigitalConfig,
        adjacent_speech_frames_threshold: i32,
    ) -> Self {
        let max_gain_change_db_per_10ms =
            config.max_gain_change_db_per_second * FRAME_DURATION_MS as f32 / 1000.0;
        debug_assert!(max_gain_change_db_per_10ms > 0.0);
        debug_assert!(adjacent_speech_frames_threshold >= 1);
        debug_assert!(config.max_output_noise_level_dbfs >= -90.0);
        debug_assert!(config.max_output_noise_level_dbfs <= 0.0);
        Self {
            gain_applier: GainApplier::new(false, db_to_ratio(config.initial_gain_db)),
            config,
            adjacent_speech_frames_threshold,
            max_gain_change_db_per_10ms,
            frames_to_gain_increase_allowed: adjacent_speech_frames_threshold,
            last_gain_db: config.initial_gain_db,
        }
    }

    /// Analyzes `info`, updates the digital gain and applies it to a 10 ms frame.
    pub(crate) fn process(&mut self, info: &FrameInfo, frame: &mut [&mut [f32]]) {
        debug_assert!(info.speech_level_dbfs >= -150.0);
        debug_assert!(!frame.is_empty());

        // Compute the input level used to select the desired gain.
        debug_assert!(info.headroom_db > 0.0);
        let input_level_dbfs = info.speech_level_dbfs + info.headroom_db;

        let target_gain_db = limit_gain_by_low_confidence(
            limit_gain_by_noise(
                compute_gain_db(input_level_dbfs, &self.config),
                info.noise_rms_dbfs,
                self.config.max_output_noise_level_dbfs,
            ),
            self.last_gain_db,
            info.limiter_envelope_dbfs,
            info.speech_level_reliable,
        );

        // Forbid increasing the gain until enough adjacent speech frames are
        // observed.
        let mut first_confident_speech_frame = false;
        if info.speech_probability < VAD_CONFIDENCE_THRESHOLD {
            self.frames_to_gain_increase_allowed = self.adjacent_speech_frames_threshold;
        } else if self.frames_to_gain_increase_allowed > 0 {
            self.frames_to_gain_increase_allowed -= 1;
            first_confident_speech_frame = self.frames_to_gain_increase_allowed == 0;
        }

        let gain_increase_allowed = self.frames_to_gain_increase_allowed == 0;

        let mut max_gain_increase_db = self.max_gain_change_db_per_10ms;
        if first_confident_speech_frame {
            // No gain increase happened while waiting for a long enough speech
            // sequence. Therefore, temporarily allow a faster gain increase.
            debug_assert!(gain_increase_allowed);
            max_gain_increase_db *= self.adjacent_speech_frames_threshold as f32;
        }

        let gain_change_this_frame_db = compute_gain_change_this_frame_db(
            target_gain_db,
            self.last_gain_db,
            gain_increase_allowed,
            self.max_gain_change_db_per_10ms,
            max_gain_increase_db,
        );

        // Optimization: avoid calling math functions if gain does not change.
        if gain_change_this_frame_db != 0.0 {
            self.gain_applier
                .set_gain_factor(db_to_ratio(self.last_gain_db + gain_change_this_frame_db));
        }

        self.gain_applier.apply_gain(frame);

        // Remember that the gain has changed for the next iteration.
        self.last_gain_db += gain_change_this_frame_db;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        ADJACENT_SPEECH_FRAMES_THRESHOLD, FRAME_DURATION_MS, MIN_LEVEL_DBFS, db_to_ratio,
    };

    const MONO: usize = 1;
    const STEREO: usize = 2;
    const FRAME_LEN_10MS_8KHZ: usize = 80;
    const FRAME_LEN_10MS_48KHZ: usize = 480;

    const MAX_SPEECH_PROBABILITY: f32 = 1.0;

    // Constants used in place of estimated noise levels.
    const NO_NOISE_DBFS: f32 = MIN_LEVEL_DBFS;
    const WITH_NOISE_DBFS: f32 = -20.0;

    // Number of additional frames to process to ensure convergence.
    const NUM_EXTRA_FRAMES: i32 = 10;

    const fn get_max_gain_change_per_frame_db(max_gain_change_db_per_second: f32) -> f32 {
        max_gain_change_db_per_second * FRAME_DURATION_MS as f32 / 1000.0
    }

    fn default_config() -> AdaptiveDigitalConfig {
        AdaptiveDigitalConfig {
            headroom_db: 5.0,
            max_gain_db: 50.0,
            initial_gain_db: 15.0,
            max_gain_change_db_per_second: 6.0,
            max_output_noise_level_dbfs: -50.0,
        }
    }

    /// Returns a `FrameInfo` that should cause no gain adaptation.
    fn get_frame_info_to_not_adapt(config: &AdaptiveDigitalConfig) -> FrameInfo {
        FrameInfo {
            speech_probability: MAX_SPEECH_PROBABILITY,
            speech_level_dbfs: -config.initial_gain_db - config.headroom_db,
            speech_level_reliable: true,
            noise_rms_dbfs: NO_NOISE_DBFS,
            headroom_db: config.headroom_db,
            limiter_envelope_dbfs: -2.0,
        }
    }

    fn make_frame(num_channels: usize, samples_per_channel: usize, value: f32) -> Vec<Vec<f32>> {
        vec![vec![value; samples_per_channel]; num_channels]
    }

    fn as_mut_slices(frame: &mut [Vec<f32>]) -> Vec<&mut [f32]> {
        frame.iter_mut().map(|ch| ch.as_mut_slice()).collect()
    }

    #[test]
    fn gain_applier_should_not_crash() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);
        let mut audio = make_frame(STEREO, FRAME_LEN_10MS_48KHZ, 10000.0);
        let mut slices = as_mut_slices(&mut audio);
        controller.process(&get_frame_info_to_not_adapt(&config), &mut slices);
    }

    #[test]
    fn max_gain_applied() {
        let config = default_config();
        let num_frames_to_adapt = (config.max_gain_db
            / get_max_gain_change_per_frame_db(config.max_gain_change_db_per_second))
            as i32
            + NUM_EXTRA_FRAMES;
        let high_noise_config = AdaptiveDigitalConfig {
            max_output_noise_level_dbfs: -40.0,
            ..config
        };
        let mut controller = AdaptiveDigitalGainController::new(
            high_noise_config,
            ADJACENT_SPEECH_FRAMES_THRESHOLD,
        );
        let mut info = get_frame_info_to_not_adapt(&high_noise_config);
        info.speech_level_dbfs = -60.0;
        let mut applied_gain = 0.0_f32;
        for _ in 0..num_frames_to_adapt {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            controller.process(&info, &mut slices);
            applied_gain = audio[0][0];
        }
        let applied_gain_db = 20.0 * applied_gain.log10();
        assert!(
            (applied_gain_db - config.max_gain_db).abs() < 0.1,
            "applied_gain_db={applied_gain_db}, expected ~{}",
            config.max_gain_db
        );
    }

    #[test]
    fn gain_does_not_change_fast() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);

        let initial_level_dbfs = -25.0_f32;
        let max_gain_change_db_per_frame =
            get_max_gain_change_per_frame_db(config.max_gain_change_db_per_second);
        let num_frames_to_adapt =
            (initial_level_dbfs / max_gain_change_db_per_frame) as i32 + NUM_EXTRA_FRAMES;

        let max_change_per_frame_linear = db_to_ratio(max_gain_change_db_per_frame);

        let mut last_gain_linear = 1.0_f32;
        for _ in 0..num_frames_to_adapt {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            let mut info = get_frame_info_to_not_adapt(&config);
            info.speech_level_dbfs = initial_level_dbfs;
            controller.process(&info, &mut slices);
            let current_gain_linear = audio[0][0];
            assert!(
                (current_gain_linear - last_gain_linear).abs() <= max_change_per_frame_linear,
                "gain change {} exceeds max {}",
                (current_gain_linear - last_gain_linear).abs(),
                max_change_per_frame_linear
            );
            last_gain_linear = current_gain_linear;
        }

        // Check that the same is true when gain decreases as well.
        for _ in 0..num_frames_to_adapt {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_8KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            let mut info = get_frame_info_to_not_adapt(&config);
            info.speech_level_dbfs = 0.0;
            controller.process(&info, &mut slices);
            let current_gain_linear = audio[0][0];
            assert!(
                (current_gain_linear - last_gain_linear).abs() <= max_change_per_frame_linear,
                "gain change {} exceeds max {}",
                (current_gain_linear - last_gain_linear).abs(),
                max_change_per_frame_linear
            );
            last_gain_linear = current_gain_linear;
        }
    }

    #[test]
    fn gain_is_ramped_in_a_frame() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);

        let initial_level_dbfs = -25.0_f32;

        let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
        let mut slices = as_mut_slices(&mut audio);
        let mut info = get_frame_info_to_not_adapt(&config);
        info.speech_level_dbfs = initial_level_dbfs;
        controller.process(&info, &mut slices);
        let mut maximal_difference = 0.0_f32;
        let mut current_value = 1.0 * db_to_ratio(config.initial_gain_db);
        for &x in &audio[0] {
            let difference = (x - current_value).abs();
            maximal_difference = maximal_difference.max(difference);
            current_value = x;
        }

        let max_change_per_frame_linear = db_to_ratio(get_max_gain_change_per_frame_db(
            config.max_gain_change_db_per_second,
        ));
        let max_change_per_sample = max_change_per_frame_linear / FRAME_LEN_10MS_48KHZ as f32;

        assert!(
            maximal_difference <= max_change_per_sample,
            "maximal_difference {maximal_difference} > max_change_per_sample {max_change_per_sample}"
        );
    }

    #[test]
    fn noise_limits_gain() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);

        let initial_level_dbfs = -25.0_f32;
        let num_initial_frames = (config.initial_gain_db
            / get_max_gain_change_per_frame_db(config.max_gain_change_db_per_second))
            as i32;
        let num_frames = 50;

        assert!(
            WITH_NOISE_DBFS > config.max_output_noise_level_dbfs,
            "WITH_NOISE_DBFS is too low"
        );

        for i in 0..num_initial_frames + num_frames {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            let mut info = get_frame_info_to_not_adapt(&config);
            info.speech_level_dbfs = initial_level_dbfs;
            info.noise_rms_dbfs = WITH_NOISE_DBFS;
            controller.process(&info, &mut slices);

            // Wait so that the adaptive gain applier has time to lower the gain.
            if i > num_initial_frames {
                let maximal_ratio = audio[0].iter().copied().reduce(f32::max).unwrap();
                assert!(
                    (maximal_ratio - 1.0).abs() < 0.001,
                    "frame {i}: maximal_ratio={maximal_ratio}, expected ~1.0"
                );
            }
        }
    }

    #[test]
    fn can_handle_positive_speech_levels() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);

        let mut audio = make_frame(STEREO, FRAME_LEN_10MS_48KHZ, 10000.0);
        let mut slices = as_mut_slices(&mut audio);
        let mut info = get_frame_info_to_not_adapt(&config);
        info.speech_level_dbfs = 5.0;
        controller.process(&info, &mut slices);
    }

    #[test]
    fn audio_level_limits_gain() {
        let config = default_config();
        let mut controller =
            AdaptiveDigitalGainController::new(config, ADJACENT_SPEECH_FRAMES_THRESHOLD);

        let initial_level_dbfs = -25.0_f32;
        let num_initial_frames = (config.initial_gain_db
            / get_max_gain_change_per_frame_db(config.max_gain_change_db_per_second))
            as i32;
        let num_frames = 50;

        assert!(
            WITH_NOISE_DBFS > config.max_output_noise_level_dbfs,
            "WITH_NOISE_DBFS is too low"
        );

        for i in 0..num_initial_frames + num_frames {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            let mut info = get_frame_info_to_not_adapt(&config);
            info.speech_level_dbfs = initial_level_dbfs;
            info.limiter_envelope_dbfs = 1.0;
            info.speech_level_reliable = false;
            controller.process(&info, &mut slices);

            // Wait so that the adaptive gain applier has time to lower the gain.
            if i > num_initial_frames {
                let maximal_ratio = audio[0].iter().copied().reduce(f32::max).unwrap();
                assert!(
                    (maximal_ratio - 1.0).abs() < 0.001,
                    "frame {i}: maximal_ratio={maximal_ratio}, expected ~1.0"
                );
            }
        }
    }

    // Parametrized tests for adjacent_speech_frames_threshold = 1, 7, 31, 12.

    #[test]
    fn do_not_increase_gain_with_too_few_speech_frames_1() {
        do_not_increase_gain_with_too_few_speech_frames(1);
    }

    #[test]
    fn do_not_increase_gain_with_too_few_speech_frames_7() {
        do_not_increase_gain_with_too_few_speech_frames(7);
    }

    #[test]
    fn do_not_increase_gain_with_too_few_speech_frames_31() {
        do_not_increase_gain_with_too_few_speech_frames(31);
    }

    #[test]
    fn do_not_increase_gain_with_too_few_speech_frames_default() {
        do_not_increase_gain_with_too_few_speech_frames(ADJACENT_SPEECH_FRAMES_THRESHOLD);
    }

    fn do_not_increase_gain_with_too_few_speech_frames(threshold: i32) {
        let config = default_config();
        let mut controller = AdaptiveDigitalGainController::new(config, threshold);

        // Lower the speech level so that the target gain will be increased.
        let mut info = get_frame_info_to_not_adapt(&config);
        info.speech_level_dbfs -= 12.0;

        let mut prev_gain = 0.0_f32;
        for i in 0..threshold {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            controller.process(&info, &mut slices);
            let gain = audio[0][0];
            if i > 0 {
                assert_eq!(prev_gain, gain, "no gain increase expected at frame {i}");
            }
            prev_gain = gain;
        }
    }

    #[test]
    fn increase_gain_with_enough_speech_frames_1() {
        increase_gain_with_enough_speech_frames(1);
    }

    #[test]
    fn increase_gain_with_enough_speech_frames_7() {
        increase_gain_with_enough_speech_frames(7);
    }

    #[test]
    fn increase_gain_with_enough_speech_frames_31() {
        increase_gain_with_enough_speech_frames(31);
    }

    #[test]
    fn increase_gain_with_enough_speech_frames_default() {
        increase_gain_with_enough_speech_frames(ADJACENT_SPEECH_FRAMES_THRESHOLD);
    }

    fn increase_gain_with_enough_speech_frames(threshold: i32) {
        let config = default_config();
        let mut controller = AdaptiveDigitalGainController::new(config, threshold);

        // Lower the speech level so that the target gain will be increased.
        let mut info = get_frame_info_to_not_adapt(&config);
        info.speech_level_dbfs -= 12.0;

        let mut prev_gain = 0.0_f32;
        for _ in 0..threshold {
            let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
            let mut slices = as_mut_slices(&mut audio);
            controller.process(&info, &mut slices);
            prev_gain = audio[0][0];
        }

        // Process one more speech frame.
        let mut audio = make_frame(MONO, FRAME_LEN_10MS_48KHZ, 1.0);
        let mut slices = as_mut_slices(&mut audio);
        controller.process(&info, &mut slices);

        // An increased gain has been applied.
        assert!(
            audio[0][0] > prev_gain,
            "gain {} should be > prev_gain {}",
            audio[0][0],
            prev_gain
        );
    }
}
