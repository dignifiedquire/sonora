//! Saturation protector that recommends headroom based on recent peaks.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/saturation_protector.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{FRAME_DURATION_MS, MIN_LEVEL_DBFS, VAD_CONFIDENCE_THRESHOLD};
use crate::saturation_protector_buffer::SaturationProtectorBuffer;

const PEAK_ENVELOPER_SUPER_FRAME_LENGTH_MS: i32 = 400;
const MIN_MARGIN_DB: f32 = 12.0;
const MAX_MARGIN_DB: f32 = 25.0;
const ATTACK: f32 = 0.998_849_4;
const DECAY: f32 = 0.999_769_75;

/// Saturation protector state. Used for check-pointing and restore ops.
#[derive(Clone, PartialEq)]
struct SaturationProtectorState {
    headroom_db: f32,
    peak_delay_buffer: SaturationProtectorBuffer,
    max_peaks_dbfs: f32,
    time_since_push_ms: i32,
}

/// Resets the saturation protector state.
fn reset_saturation_protector_state(
    initial_headroom_db: f32,
    state: &mut SaturationProtectorState,
) {
    state.headroom_db = initial_headroom_db;
    state.peak_delay_buffer.reset();
    state.max_peaks_dbfs = MIN_LEVEL_DBFS;
    state.time_since_push_ms = 0;
}

/// Updates `state` by analyzing the estimated speech level and the peak level.
fn update_saturation_protector_state(
    peak_dbfs: f32,
    speech_level_dbfs: f32,
    state: &mut SaturationProtectorState,
) {
    // Get the max peak over `PEAK_ENVELOPER_SUPER_FRAME_LENGTH_MS` ms.
    state.max_peaks_dbfs = state.max_peaks_dbfs.max(peak_dbfs);
    state.time_since_push_ms += FRAME_DURATION_MS;
    if state.time_since_push_ms > PEAK_ENVELOPER_SUPER_FRAME_LENGTH_MS {
        // Push `max_peaks_dbfs` back into the ring buffer.
        state.peak_delay_buffer.push_back(state.max_peaks_dbfs);
        // Reset.
        state.max_peaks_dbfs = MIN_LEVEL_DBFS;
        state.time_since_push_ms = 0;
    }

    // Update the headroom by comparing the estimated speech level and the delayed
    // max speech peak.
    let delayed_peak_dbfs = state
        .peak_delay_buffer
        .front()
        .unwrap_or(state.max_peaks_dbfs);
    let difference_db = delayed_peak_dbfs - speech_level_dbfs;
    if difference_db > state.headroom_db {
        // Attack.
        state.headroom_db = state.headroom_db * ATTACK + difference_db * (1.0 - ATTACK);
    } else {
        // Decay.
        state.headroom_db = state.headroom_db * DECAY + difference_db * (1.0 - DECAY);
    }

    state.headroom_db = state.headroom_db.clamp(MIN_MARGIN_DB, MAX_MARGIN_DB);
}

/// Saturation protector which recommends a headroom based on the recent peaks.
pub(crate) struct SaturationProtector {
    initial_headroom_db: f32,
    adjacent_speech_frames_threshold: i32,
    num_adjacent_speech_frames: i32,
    headroom_db: f32,
    preliminary_state: SaturationProtectorState,
    reliable_state: SaturationProtectorState,
}

impl SaturationProtector {
    /// Creates a new saturation protector that starts at `initial_headroom_db`.
    pub(crate) fn new(initial_headroom_db: f32, adjacent_speech_frames_threshold: i32) -> Self {
        let mut sp = Self {
            initial_headroom_db,
            adjacent_speech_frames_threshold,
            num_adjacent_speech_frames: 0,
            headroom_db: initial_headroom_db,
            preliminary_state: SaturationProtectorState {
                headroom_db: initial_headroom_db,
                peak_delay_buffer: SaturationProtectorBuffer::default(),
                max_peaks_dbfs: MIN_LEVEL_DBFS,
                time_since_push_ms: 0,
            },
            reliable_state: SaturationProtectorState {
                headroom_db: initial_headroom_db,
                peak_delay_buffer: SaturationProtectorBuffer::default(),
                max_peaks_dbfs: MIN_LEVEL_DBFS,
                time_since_push_ms: 0,
            },
        };
        sp.reset();
        sp
    }

    /// Returns the recommended headroom in dB.
    pub(crate) fn headroom_db(&self) -> f32 {
        self.headroom_db
    }

    /// Analyzes the peak level of a 10 ms frame along with its speech probability
    /// and the current speech level estimate to update the recommended headroom.
    pub(crate) fn analyze(
        &mut self,
        speech_probability: f32,
        peak_dbfs: f32,
        speech_level_dbfs: f32,
    ) {
        if speech_probability < VAD_CONFIDENCE_THRESHOLD {
            // Not a speech frame.
            if self.adjacent_speech_frames_threshold > 1 {
                if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                    // First non-speech frame after a long enough sequence of speech
                    // frames. Update the reliable state.
                    self.reliable_state = self.preliminary_state.clone();
                } else if self.num_adjacent_speech_frames > 0 {
                    // First non-speech frame after a too short sequence of speech
                    // frames. Reset to the last reliable state.
                    self.preliminary_state = self.reliable_state.clone();
                }
            }
            self.num_adjacent_speech_frames = 0;
        } else {
            // Speech frame observed.
            self.num_adjacent_speech_frames += 1;

            // Update preliminary level estimate.
            update_saturation_protector_state(
                peak_dbfs,
                speech_level_dbfs,
                &mut self.preliminary_state,
            );

            if self.num_adjacent_speech_frames >= self.adjacent_speech_frames_threshold {
                // `preliminary_state` is now reliable. Update the headroom.
                self.headroom_db = self.preliminary_state.headroom_db;
            }
        }
    }

    /// Resets the internal state.
    pub(crate) fn reset(&mut self) {
        self.num_adjacent_speech_frames = 0;
        self.headroom_db = self.initial_headroom_db;
        reset_saturation_protector_state(self.initial_headroom_db, &mut self.preliminary_state);
        reset_saturation_protector_state(self.initial_headroom_db, &mut self.reliable_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::FRAME_DURATION_MS;

    const INITIAL_HEADROOM_DB: f32 = 20.0;
    const NO_ADJACENT_SPEECH_FRAMES_REQUIRED: i32 = 1;
    const MAX_SPEECH_PROBABILITY: f32 = 1.0;

    /// Calls `analyze` `num_iterations` times and returns the largest
    /// headroom difference between two consecutive calls.
    fn run_on_constant_level(
        num_iterations: i32,
        speech_probability: f32,
        peak_dbfs: f32,
        speech_level_dbfs: f32,
        saturation_protector: &mut SaturationProtector,
    ) -> f32 {
        let mut last_headroom = saturation_protector.headroom_db();
        let mut max_difference = 0.0_f32;
        for _ in 0..num_iterations {
            saturation_protector.analyze(speech_probability, peak_dbfs, speech_level_dbfs);
            let new_headroom = saturation_protector.headroom_db();
            max_difference = max_difference.max((new_headroom - last_headroom).abs());
            last_headroom = new_headroom;
        }
        max_difference
    }

    #[test]
    fn reset() {
        let mut sp =
            SaturationProtector::new(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);
        let initial_headroom_db = sp.headroom_db();
        run_on_constant_level(10, MAX_SPEECH_PROBABILITY, 0.0, -10.0, &mut sp);
        // Make sure that there are side-effects.
        assert_ne!(initial_headroom_db, sp.headroom_db());
        sp.reset();
        assert_eq!(initial_headroom_db, sp.headroom_db());
    }

    #[test]
    fn estimates_crest_ratio() {
        let num_iterations = 2000;
        let peak_level_dbfs = -20.0;
        let crest_factor_db = INITIAL_HEADROOM_DB + 1.0;
        let speech_level_dbfs = peak_level_dbfs - crest_factor_db;
        let max_difference_db = 0.5 * (INITIAL_HEADROOM_DB - crest_factor_db).abs();

        let mut sp =
            SaturationProtector::new(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);
        run_on_constant_level(
            num_iterations,
            MAX_SPEECH_PROBABILITY,
            peak_level_dbfs,
            speech_level_dbfs,
            &mut sp,
        );
        assert!(
            (sp.headroom_db() - crest_factor_db).abs() <= max_difference_db,
            "headroom {} should be near crest_factor {}",
            sp.headroom_db(),
            crest_factor_db
        );
    }

    #[test]
    fn change_slowly() {
        let num_iterations = 1000;
        let peak_level_dbfs = -20.0;
        let crest_factor_db = INITIAL_HEADROOM_DB - 5.0;
        let other_crest_factor_db = INITIAL_HEADROOM_DB;
        let speech_level_dbfs = peak_level_dbfs - crest_factor_db;
        let other_speech_level_dbfs = peak_level_dbfs - other_crest_factor_db;

        let mut sp =
            SaturationProtector::new(INITIAL_HEADROOM_DB, NO_ADJACENT_SPEECH_FRAMES_REQUIRED);
        let mut max_difference_db = run_on_constant_level(
            num_iterations,
            MAX_SPEECH_PROBABILITY,
            peak_level_dbfs,
            speech_level_dbfs,
            &mut sp,
        );
        max_difference_db = max_difference_db.max(run_on_constant_level(
            num_iterations,
            MAX_SPEECH_PROBABILITY,
            peak_level_dbfs,
            other_speech_level_dbfs,
            &mut sp,
        ));
        let max_change_speed_db_per_second = 0.5; // 1 db / 2 seconds.
        assert!(
            max_difference_db <= max_change_speed_db_per_second / 1000.0 * FRAME_DURATION_MS as f32,
            "max_difference_db {max_difference_db} exceeds max change speed"
        );
    }

    // Parametrized tests for adjacent_speech_frames_threshold = 2, 9, 17.

    #[test]
    fn do_not_adapt_to_short_speech_segments_threshold_2() {
        do_not_adapt_to_short_speech_segments(2);
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
        let mut sp = SaturationProtector::new(INITIAL_HEADROOM_DB, threshold);
        let initial_headroom_db = sp.headroom_db();
        run_on_constant_level(threshold - 1, MAX_SPEECH_PROBABILITY, 0.0, -10.0, &mut sp);
        // No adaptation expected.
        assert_eq!(initial_headroom_db, sp.headroom_db());
    }

    #[test]
    fn adapt_to_enough_speech_segments_threshold_2() {
        adapt_to_enough_speech_segments(2);
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
        let mut sp = SaturationProtector::new(INITIAL_HEADROOM_DB, threshold);
        let initial_headroom_db = sp.headroom_db();
        run_on_constant_level(threshold + 1, MAX_SPEECH_PROBABILITY, 0.0, -10.0, &mut sp);
        // Adaptation expected.
        assert_ne!(initial_headroom_db, sp.headroom_db());
    }
}
