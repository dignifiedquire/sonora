//! Multi-channel to mono alignment mixer for delay estimation.
//!
//! Performs channel conversion to mono for providing a decent mono input
//! for the delay estimation.
//!
//! Ported from `modules/audio_processing/aec3/alignment_mixer.h/cc`.

use crate::block::Block;
use crate::common::{BLOCK_SIZE, NUM_BLOCKS_PER_SECOND};
use crate::config::AlignmentMixing;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixingVariant {
    Downmix,
    Adaptive,
    Fixed,
}

fn choose_mixing_variant(
    downmix: bool,
    adaptive_selection: bool,
    num_channels: usize,
) -> MixingVariant {
    debug_assert!(!(adaptive_selection && downmix));
    debug_assert!(num_channels > 0);

    if num_channels == 1 {
        return MixingVariant::Fixed;
    }
    if downmix {
        return MixingVariant::Downmix;
    }
    if adaptive_selection {
        return MixingVariant::Adaptive;
    }
    MixingVariant::Fixed
}

/// Mixes multi-channel audio down to mono for delay estimation.
pub(crate) struct AlignmentMixer {
    num_channels: usize,
    one_by_num_channels: f32,
    excitation_energy_threshold: f32,
    prefer_first_two_channels: bool,
    selection_variant: MixingVariant,
    strong_block_counters: [usize; 2],
    cumulative_energies: Vec<f32>,
    selected_channel: usize,
    block_counter: usize,
}

impl AlignmentMixer {
    pub(crate) fn new(num_channels: usize, config: &AlignmentMixing) -> Self {
        Self::new_with_params(
            num_channels,
            config.downmix,
            config.adaptive_selection,
            config.activity_power_threshold,
            config.prefer_first_two_channels,
        )
    }

    pub(crate) fn new_with_params(
        num_channels: usize,
        downmix: bool,
        adaptive_selection: bool,
        activity_power_threshold: f32,
        prefer_first_two_channels: bool,
    ) -> Self {
        let selection_variant = choose_mixing_variant(downmix, adaptive_selection, num_channels);
        let cumulative_energies = if selection_variant == MixingVariant::Adaptive {
            vec![0.0f32; num_channels]
        } else {
            Vec::new()
        };

        Self {
            num_channels,
            one_by_num_channels: 1.0 / num_channels as f32,
            excitation_energy_threshold: BLOCK_SIZE as f32 * activity_power_threshold,
            prefer_first_two_channels,
            selection_variant,
            strong_block_counters: [0; 2],
            cumulative_energies,
            selected_channel: 0,
            block_counter: 0,
        }
    }

    /// Produces mono output from multi-channel input.
    pub(crate) fn produce_output(&mut self, x: &Block, y: &mut [f32; BLOCK_SIZE]) {
        debug_assert_eq!(x.num_channels(), self.num_channels);

        if self.selection_variant == MixingVariant::Downmix {
            self.downmix(x, y);
            return;
        }

        let ch = if self.selection_variant == MixingVariant::Fixed {
            0
        } else {
            self.select_channel(x)
        };

        debug_assert!(x.num_channels() > ch);
        y.copy_from_slice(x.view(0, ch));
    }

    fn downmix(&self, x: &Block, y: &mut [f32; BLOCK_SIZE]) {
        debug_assert_eq!(x.num_channels(), self.num_channels);
        debug_assert!(self.num_channels >= 2);

        y.copy_from_slice(x.view(0, 0));
        for ch in 1..self.num_channels {
            let x_ch = x.view(0, ch);
            for (out, &inp) in y.iter_mut().zip(x_ch.iter()) {
                *out += inp;
            }
        }
        for v in y.iter_mut() {
            *v *= self.one_by_num_channels;
        }
    }

    fn select_channel(&mut self, x: &Block) -> usize {
        debug_assert_eq!(x.num_channels(), self.num_channels);
        debug_assert!(self.num_channels >= 2);
        debug_assert_eq!(self.cumulative_energies.len(), self.num_channels);

        let blocks_to_choose_left_or_right = NUM_BLOCKS_PER_SECOND / 2;
        let good_signal_in_left_or_right = self.prefer_first_two_channels
            && (self.strong_block_counters[0] > blocks_to_choose_left_or_right
                || self.strong_block_counters[1] > blocks_to_choose_left_or_right);

        let num_ch_to_analyze = if good_signal_in_left_or_right {
            2
        } else {
            self.num_channels
        };

        let num_blocks_before_energy_smoothing = 60 * NUM_BLOCKS_PER_SECOND;
        self.block_counter += 1;

        for ch in 0..num_ch_to_analyze {
            let x_ch = x.view(0, ch);
            let x2_sum: f32 = x_ch.iter().map(|&v| v * v).sum();

            if ch < 2 && x2_sum > self.excitation_energy_threshold {
                self.strong_block_counters[ch] += 1;
            }

            if self.block_counter <= num_blocks_before_energy_smoothing {
                self.cumulative_energies[ch] += x2_sum;
            } else {
                let smoothing = 1.0 / (10 * NUM_BLOCKS_PER_SECOND) as f32;
                self.cumulative_energies[ch] += smoothing * (x2_sum - self.cumulative_energies[ch]);
            }
        }

        // Normalize energies to allow smoothing-based computation.
        if self.block_counter == num_blocks_before_energy_smoothing {
            let scale = 1.0 / num_blocks_before_energy_smoothing as f32;
            for ch in 0..num_ch_to_analyze {
                self.cumulative_energies[ch] *= scale;
            }
        }

        let mut strongest_ch = 0;
        for ch in 0..num_ch_to_analyze {
            if self.cumulative_energies[ch] > self.cumulative_energies[strongest_ch] {
                strongest_ch = ch;
            }
        }

        if (good_signal_in_left_or_right && self.selected_channel > 1)
            || self.cumulative_energies[strongest_ch]
                > 2.0 * self.cumulative_energies[self.selected_channel]
        {
            self.selected_channel = strongest_ch;
        }

        self.selected_channel
    }
}
