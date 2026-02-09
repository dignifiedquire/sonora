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
//! Both modes are available, selectable via
//! [`TransparentModeType`](crate::config::TransparentModeType) in the config.

use crate::common::NUM_BLOCKS_PER_SECOND;
use crate::config::{EchoCanceller3Config, TransparentModeType};

/// Current filter and render state used to update transparent mode.
pub(crate) struct TransparentModeState {
    pub filter_delay_blocks: i32,
    pub any_filter_consistent: bool,
    pub any_filter_converged: bool,
    pub any_coarse_filter_converged: bool,
    pub all_filters_diverged: bool,
    pub active_render: bool,
    pub saturated_capture: bool,
}

const BLOCKS_SINCE_CONVERGED_FILTER_INIT: usize = 10000;
const BLOCKS_SINCE_CONSISTENT_ESTIMATE_INIT: usize = 10000;
const INITIAL_TRANSPARENT_STATE_PROBABILITY: f32 = 0.2;

/// Transparent mode detection — reduces echo suppression when there is
/// no echo (e.g. headset scenarios).
#[derive(Debug)]
pub(crate) enum TransparentMode {
    /// Legacy counter-based transparent mode classifier.
    Legacy(LegacyTransparentMode),
    /// HMM-based transparent mode classifier.
    Hmm(HmmTransparentMode),
}

impl TransparentMode {
    /// Creates a transparent mode detector.
    ///
    /// Returns `None` when transparent mode is disabled (bounded ERL).
    pub(crate) fn create(config: &EchoCanceller3Config) -> Option<Self> {
        if config.ep_strength.bounded_erl {
            None
        } else {
            match config.echo_removal_control.transparent_mode {
                TransparentModeType::Legacy => {
                    Some(Self::Legacy(LegacyTransparentMode::new(config)))
                }
                TransparentModeType::Hmm => Some(Self::Hmm(HmmTransparentMode::new())),
            }
        }
    }

    /// Returns whether transparent mode is currently active.
    pub(crate) fn active(&self) -> bool {
        match self {
            Self::Legacy(legacy) => legacy.active(),
            Self::Hmm(hmm) => hmm.active(),
        }
    }

    /// Resets the detector state.
    pub(crate) fn reset(&mut self) {
        match self {
            Self::Legacy(legacy) => legacy.reset(),
            Self::Hmm(hmm) => hmm.reset(),
        }
    }

    /// Updates the detection decision based on new data.
    pub(crate) fn update(&mut self, state: &TransparentModeState) {
        match self {
            Self::Legacy(legacy) => legacy.update(state),
            Self::Hmm(hmm) => hmm.update(state.any_coarse_filter_converged, state.active_render),
        }
    }
}

/// Legacy counter-based transparent mode classifier.
#[derive(Debug)]
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

    fn update(&mut self, state: &TransparentModeState) {
        self.capture_block_counter += 1;
        self.strong_not_saturated_render_blocks +=
            if state.active_render && !state.saturated_capture {
                1
            } else {
                0
            };

        if state.any_filter_consistent && state.filter_delay_blocks < 5 {
            self.sane_filter_observed = true;
            self.active_blocks_since_sane_filter = 0;
        } else if state.active_render {
            self.active_blocks_since_sane_filter += 1;
        }

        let sane_filter_recently_seen = if !self.sane_filter_observed {
            self.capture_block_counter <= 5 * NUM_BLOCKS_PER_SECOND
        } else {
            self.active_blocks_since_sane_filter <= 30 * NUM_BLOCKS_PER_SECOND
        };

        if state.any_filter_converged {
            self.recent_convergence_during_activity = true;
            self.active_non_converged_sequence_size = 0;
            self.non_converged_sequence_size = 0;
            self.num_converged_blocks += 1;
        } else {
            self.non_converged_sequence_size += 1;
            if self.non_converged_sequence_size > 20 * NUM_BLOCKS_PER_SECOND {
                self.num_converged_blocks = 0;
            }

            if state.active_render {
                self.active_non_converged_sequence_size += 1;
                if self.active_non_converged_sequence_size > 60 * NUM_BLOCKS_PER_SECOND {
                    self.recent_convergence_during_activity = false;
                }
            }
        }

        if !state.all_filters_diverged {
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

        if self.finite_erl_recently_detected
            || (sane_filter_recently_seen && self.recent_convergence_during_activity)
        {
            self.transparency_activated = false;
        } else {
            let filter_should_have_converged =
                self.strong_not_saturated_render_blocks > 6 * NUM_BLOCKS_PER_SECOND;
            self.transparency_activated = filter_should_have_converged;
        }
    }
}

/// HMM-based transparent mode classifier.
///
/// Uses a two-state Hidden Markov Model (normal vs. transparent) that
/// observes whether the adaptive filter has converged. The posterior
/// probability of the transparent state is updated via Bayes' theorem,
/// with hysteresis thresholds to avoid oscillation.
///
/// Ported from `TransparentModeImpl` in C++ `transparent_mode.cc`.
#[derive(Debug)]
pub(crate) struct HmmTransparentMode {
    transparency_activated: bool,
    prob_transparent_state: f32,
}

impl HmmTransparentMode {
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

        // HMM transition probability (probability of switching states).
        const SWITCH: f32 = 0.000001;
        // Observation probability: P(filter converged | normal state).
        const CONVERGED_NORMAL: f32 = 0.01;
        // Observation probability: P(filter converged | transparent state).
        const CONVERGED_TRANSPARENT: f32 = 0.001;

        // Transition matrix: [P(switch), P(stay)].
        const A: [f32; 2] = [SWITCH, 1.0 - SWITCH];
        // Observation matrix: B[state][observation].
        const B: [[f32; 2]; 2] = [
            [1.0 - CONVERGED_NORMAL, CONVERGED_NORMAL],
            [1.0 - CONVERGED_TRANSPARENT, CONVERGED_TRANSPARENT],
        ];

        let prob_transparent = self.prob_transparent_state;
        let prob_normal = 1.0 - prob_transparent;

        // Predict step: apply transition model.
        let prob_transition_transparent = prob_normal * A[0] + prob_transparent * A[1];
        let prob_transition_normal = 1.0 - prob_transition_transparent;

        // Observation index: 1 if filter converged, 0 otherwise.
        let obs = usize::from(any_coarse_filter_converged);

        // Update step: Bayes' theorem.
        let prob_joint_normal = prob_transition_normal * B[0][obs];
        let prob_joint_transparent = prob_transition_transparent * B[1][obs];
        self.prob_transparent_state =
            prob_joint_transparent / (prob_joint_normal + prob_joint_transparent);

        // Hysteresis to avoid oscillation.
        if self.prob_transparent_state > 0.95 {
            self.transparency_activated = true;
        } else if self.prob_transparent_state < 0.5 {
            self.transparency_activated = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn hmm_initial_state_not_active() {
        let hmm = HmmTransparentMode::new();
        assert!(!hmm.active());
    }

    #[test]
    fn hmm_creates_via_config() {
        let mut config = EchoCanceller3Config::default();
        config.echo_removal_control.transparent_mode = TransparentModeType::Hmm;
        let mode = TransparentMode::create(&config);
        assert!(matches!(mode, Some(TransparentMode::Hmm(_))));
    }

    #[test]
    fn hmm_activates_without_convergence() {
        let mut hmm = HmmTransparentMode::new();
        // Feed many blocks with active render but no filter convergence.
        // The HMM should eventually activate transparent mode.
        for _ in 0..10_000 {
            hmm.update(false, true);
        }
        assert!(hmm.active());
    }

    #[test]
    fn hmm_deactivates_with_convergence() {
        let mut hmm = HmmTransparentMode::new();
        // First, drive into transparent state.
        for _ in 0..10_000 {
            hmm.update(false, true);
        }
        assert!(hmm.active());
        // Now feed convergence observations — should deactivate.
        for _ in 0..100 {
            hmm.update(true, true);
        }
        assert!(!hmm.active());
    }

    #[test]
    fn hmm_no_update_without_active_render() {
        let mut hmm = HmmTransparentMode::new();
        let initial_prob = hmm.prob_transparent_state;
        hmm.update(false, false);
        assert_eq!(hmm.prob_transparent_state, initial_prob);
    }

    #[test]
    fn hmm_reset_restores_initial_state() {
        let mut hmm = HmmTransparentMode::new();
        for _ in 0..10_000 {
            hmm.update(false, true);
        }
        assert!(hmm.active());
        hmm.reset();
        assert!(!hmm.active());
        assert_eq!(
            hmm.prob_transparent_state,
            INITIAL_TRANSPARENT_STATE_PROBABILITY
        );
    }
}
