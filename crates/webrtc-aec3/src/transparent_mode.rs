//! Transparent mode detection for reducing echo suppression when headsets
//! are used.
//!
//! Ported from `modules/audio_processing/aec3/transparent_mode.h/cc`.
//!
//! The C++ code has a virtual base with factory creating one of three
//! implementations based on field trials:
//! - `TransparentModeImpl` (HMM-based)
//! - `LegacyTransparentModeImpl` (counter-based)
//! - null (disabled when `bounded_erl`)
//!
//! Since we skip field trials, we use LegacyTransparentModeImpl as default
//! and represent the choice as an enum (not trait objects).

use crate::common::NUM_BLOCKS_PER_SECOND;
use crate::config::EchoCanceller3Config;

const BLOCKS_SINCE_CONVERGED_FILTER_INIT: usize = 10000;
const BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT: usize = 10000;
const INITIAL_TRANSPARENT_STATE_PROBABILITY: f32 = 0.2;

/// Transparent mode detection — reduces echo suppression when there is
/// no echo (e.g. headset scenarios).
pub(crate) enum TransparentMode {
    /// HMM-based transparent mode classifier.
    Hmm(TransparentModeHmm),
    /// Legacy counter-based transparent mode classifier.
    Legacy(LegacyTransparentMode),
}

impl TransparentMode {
    /// Creates a transparent mode detector.
    ///
    /// Returns `None` when transparent mode is disabled (bounded ERL).
    /// Without field trials, defaults to the Legacy implementation.
    pub(crate) fn create(config: &EchoCanceller3Config) -> Option<Self> {
        if config.ep_strength.bounded_erl {
            None
        } else {
            // Without field trials, use Legacy (the default in C++).
            Some(TransparentMode::Legacy(LegacyTransparentMode::new(config)))
        }
    }

    /// Returns whether transparent mode is currently active.
    pub(crate) fn active(&self) -> bool {
        match self {
            TransparentMode::Hmm(hmm) => hmm.active(),
            TransparentMode::Legacy(legacy) => legacy.active(),
        }
    }

    /// Resets the detector state.
    pub(crate) fn reset(&mut self) {
        match self {
            TransparentMode::Hmm(hmm) => hmm.reset(),
            TransparentMode::Legacy(legacy) => legacy.reset(),
        }
    }

    /// Updates the detection decision based on new data.
    #[allow(
        clippy::too_many_arguments,
        reason = "matches C++ virtual method signature"
    )]
    pub(crate) fn update(
        &mut self,
        filter_delay_blocks: i32,
        any_filter_consistent: bool,
        any_filter_converged: bool,
        any_coarse_filter_converged: bool,
        all_filters_diverged: bool,
        active_render: bool,
        saturated_capture: bool,
    ) {
        match self {
            TransparentMode::Hmm(hmm) => hmm.update(any_coarse_filter_converged, active_render),
            TransparentMode::Legacy(legacy) => legacy.update(
                filter_delay_blocks,
                any_filter_consistent,
                any_filter_converged,
                all_filters_diverged,
                active_render,
                saturated_capture,
            ),
        }
    }
}

/// HMM-based transparent mode classifier.
pub(crate) struct TransparentModeHmm {
    transparency_activated: bool,
    prob_transparent_state: f32,
}

impl TransparentModeHmm {
    pub(crate) fn new() -> Self {
        Self {
            transparency_activated: false,
            prob_transparent_state: INITIAL_TRANSPARENT_STATE_PROBABILITY,
        }
    }

    fn active(&self) -> bool {
        self.transparency_activated
    }

    fn reset(&mut self) {
        self.transparency_activated = false;
        self.prob_transparent_state = INITIAL_TRANSPARENT_STATE_PROBABILITY;
    }

    fn update(&mut self, any_coarse_filter_converged: bool, active_render: bool) {
        if !active_render {
            return;
        }

        const K_SWITCH: f32 = 0.000_001;
        const K_CONVERGED_NORMAL: f32 = 0.01;
        const K_CONVERGED_TRANSPARENT: f32 = 0.001;

        // Transition probabilities.
        const K_A: [f32; 2] = [K_SWITCH, 1.0 - K_SWITCH];

        // Observation probabilities for [normal, transparent] x [not_converged, converged].
        const K_B: [[f32; 2]; 2] = [
            [1.0 - K_CONVERGED_NORMAL, K_CONVERGED_NORMAL],
            [1.0 - K_CONVERGED_TRANSPARENT, K_CONVERGED_TRANSPARENT],
        ];

        let prob_transparent = self.prob_transparent_state;
        let prob_normal = 1.0 - prob_transparent;

        let prob_transition_transparent = prob_normal * K_A[0] + prob_transparent * K_A[1];
        let prob_transition_normal = 1.0 - prob_transition_transparent;

        let out = any_coarse_filter_converged as usize;

        let prob_joint_normal = prob_transition_normal * K_B[0][out];
        let prob_joint_transparent = prob_transition_transparent * K_B[1][out];

        debug_assert!(prob_joint_normal + prob_joint_transparent > 0.0);
        self.prob_transparent_state =
            prob_joint_transparent / (prob_joint_normal + prob_joint_transparent);

        if self.prob_transparent_state > 0.95 {
            self.transparency_activated = true;
        } else if self.prob_transparent_state < 0.5 {
            self.transparency_activated = false;
        }
    }
}

/// Legacy counter-based transparent mode classifier.
pub(crate) struct LegacyTransparentMode {
    linear_and_stable_echo_path: bool,
    capture_block_counter: usize,
    transparency_activated: bool,
    active_blocks_since_sane_filter: usize,
    sane_filter_observed: bool,
    finite_erl_recently_detected: bool,
    non_converged_sequence_size: usize,
    diverged_sequence_size: usize,
    active_non_converged_sequence_size: usize,
    num_converged_blocks: usize,
    recent_convergence_during_activity: bool,
    strong_not_saturated_render_blocks: usize,
}

impl LegacyTransparentMode {
    pub(crate) fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            linear_and_stable_echo_path: config.echo_removal_control.linear_and_stable_echo_path,
            capture_block_counter: 0,
            transparency_activated: false,
            active_blocks_since_sane_filter: BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT,
            sane_filter_observed: false,
            finite_erl_recently_detected: false,
            non_converged_sequence_size: BLOCKS_SINCE_CONVERGED_FILTER_INIT,
            diverged_sequence_size: 0,
            active_non_converged_sequence_size: 0,
            num_converged_blocks: 0,
            recent_convergence_during_activity: false,
            strong_not_saturated_render_blocks: 0,
        }
    }

    fn active(&self) -> bool {
        self.transparency_activated
    }

    fn reset(&mut self) {
        self.non_converged_sequence_size = BLOCKS_SINCE_CONVERGED_FILTER_INIT;
        self.diverged_sequence_size = 0;
        self.strong_not_saturated_render_blocks = 0;
        if self.linear_and_stable_echo_path {
            self.recent_convergence_during_activity = false;
        }
    }

    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    fn update(
        &mut self,
        filter_delay_blocks: i32,
        any_filter_consistent: bool,
        any_filter_converged: bool,
        all_filters_diverged: bool,
        active_render: bool,
        saturated_capture: bool,
    ) {
        self.capture_block_counter += 1;
        self.strong_not_saturated_render_blocks += if active_render && !saturated_capture {
            1
        } else {
            0
        };

        if any_filter_consistent && filter_delay_blocks < 5 {
            self.sane_filter_observed = true;
            self.active_blocks_since_sane_filter = 0;
        } else if active_render {
            self.active_blocks_since_sane_filter += 1;
        }

        let sane_filter_recently_seen = if !self.sane_filter_observed {
            self.capture_block_counter <= 5 * NUM_BLOCKS_PER_SECOND
        } else {
            self.active_blocks_since_sane_filter <= 30 * NUM_BLOCKS_PER_SECOND
        };

        if any_filter_converged {
            self.recent_convergence_during_activity = true;
            self.active_non_converged_sequence_size = 0;
            self.non_converged_sequence_size = 0;
            self.num_converged_blocks += 1;
        } else {
            self.non_converged_sequence_size += 1;
            if self.non_converged_sequence_size > 20 * NUM_BLOCKS_PER_SECOND {
                self.num_converged_blocks = 0;
            }

            if active_render {
                self.active_non_converged_sequence_size += 1;
                if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
                    self.recent_convergence_during_activity = false;
                }
            }
        }

        if !all_filters_diverged {
            self.diverged_sequence_size = 0;
        } else {
            self.diverged_sequence_size += 1;
            if self.diverged_sequence_size >= 60 {
                self.non_converged_sequence_size = BLOCKS_SINCE_CONVERGED_FILTER_INIT;
            }
        }

        if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
            self.finite_erl_recently_detected = false;
        }
        if self.num_converged_blocks > 50 {
            self.finite_erl_recently_detected = true;
        }

        if self.finite_erl_recently_detected {
            self.transparency_activated = false;
        } else if sane_filter_recently_seen && self.recent_convergence_during_activity {
            self.transparency_activated = false;
        } else {
            let filter_should_have_converged =
                self.strong_not_saturated_render_blocks > 6 * NUM_BLOCKS_PER_SECOND;
            self.transparency_activated = filter_should_have_converged;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmm_initial_state_not_active() {
        let hmm = TransparentModeHmm::new();
        assert!(!hmm.active());
    }

    #[test]
    fn hmm_reset_clears_state() {
        let mut hmm = TransparentModeHmm::new();
        hmm.transparency_activated = true;
        hmm.reset();
        assert!(!hmm.active());
        assert!((hmm.prob_transparent_state - INITIAL_TRANSPARENT_STATE_PROBABILITY).abs() < 1e-6);
    }

    #[test]
    fn legacy_initial_state_not_active() {
        let config = EchoCanceller3Config::default();
        let legacy = LegacyTransparentMode::new(&config);
        assert!(!legacy.active());
    }

    #[test]
    fn disabled_when_bounded_erl() {
        let mut config = EchoCanceller3Config::default();
        config.ep_strength.bounded_erl = true;
        let mode = TransparentMode::create(&config);
        assert!(mode.is_none());
    }

    #[test]
    fn enabled_when_not_bounded_erl() {
        let mut config = EchoCanceller3Config::default();
        config.ep_strength.bounded_erl = false;
        let mode = TransparentMode::create(&config);
        assert!(mode.is_some());
        assert!(!mode.unwrap().active());
    }
}
