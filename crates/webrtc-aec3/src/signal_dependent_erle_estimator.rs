//! Signal-dependent ERLE estimator.
//!
//! Refines ERLE estimation by analyzing which filter sections contribute
//! most to the echo estimate, enabling signal-dependent correction factors.
//!
//! Ported from `modules/audio_processing/aec3/signal_dependent_erle_estimator.h/cc`.

use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::EchoCanceller3Config;
use crate::render_buffer::RenderBuffer;

pub(crate) const SUBBANDS: usize = 6;

const BAND_BOUNDARIES: [usize; SUBBANDS + 1] = [1, 8, 16, 24, 32, 48, FFT_LENGTH_BY_2_PLUS_1];

fn form_subband_map() -> [usize; FFT_LENGTH_BY_2_PLUS_1] {
    let mut map = [0usize; FFT_LENGTH_BY_2_PLUS_1];
    let mut subband = 1;
    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
        debug_assert!(subband < BAND_BOUNDARIES.len());
        if k >= BAND_BOUNDARIES[subband] {
            subband += 1;
            debug_assert!(k < BAND_BOUNDARIES[subband]);
        }
        map[k] = subband - 1;
    }
    map
}

fn define_filter_section_sizes(
    delay_headroom_blocks: usize,
    num_blocks: usize,
    num_sections: usize,
) -> Vec<usize> {
    let filter_length_blocks = num_blocks - delay_headroom_blocks;
    let mut section_sizes = vec![0usize; num_sections];
    let mut remaining_blocks = filter_length_blocks;
    let mut remaining_sections = num_sections;
    let mut estimator_size = 2;
    let mut idx = 0;
    while remaining_sections > 1 && remaining_blocks > estimator_size * remaining_sections {
        debug_assert!(idx < section_sizes.len());
        section_sizes[idx] = estimator_size;
        remaining_blocks -= estimator_size;
        remaining_sections -= 1;
        estimator_size *= 2;
        idx += 1;
    }

    let last_groups_size = remaining_blocks / remaining_sections;
    for i in idx..num_sections {
        section_sizes[i] = last_groups_size;
    }
    section_sizes[num_sections - 1] += remaining_blocks - last_groups_size * remaining_sections;
    section_sizes
}

fn set_sections_boundaries(
    delay_headroom_blocks: usize,
    num_blocks: usize,
    num_sections: usize,
) -> Vec<usize> {
    let mut boundaries = vec![0usize; num_sections + 1];
    if boundaries.len() == 2 {
        boundaries[0] = 0;
        boundaries[1] = num_blocks;
        return boundaries;
    }
    debug_assert!(boundaries.len() > 2);
    let section_sizes =
        define_filter_section_sizes(delay_headroom_blocks, num_blocks, boundaries.len() - 1);

    let mut idx = 0;
    let mut current_size_block = 0;
    debug_assert_eq!(section_sizes.len() + 1, boundaries.len());
    boundaries[0] = delay_headroom_blocks;
    for k in delay_headroom_blocks..num_blocks {
        current_size_block += 1;
        if current_size_block >= section_sizes[idx] {
            idx += 1;
            if idx == section_sizes.len() {
                break;
            }
            boundaries[idx] = k + 1;
            current_size_block = 0;
        }
    }
    boundaries[section_sizes.len()] = num_blocks;
    boundaries
}

fn set_max_erle_subbands(
    max_erle_l: f32,
    max_erle_h: f32,
    limit_subband_l: usize,
) -> [f32; SUBBANDS] {
    let mut max_erle = [0.0f32; SUBBANDS];
    for i in 0..limit_subband_l {
        max_erle[i] = max_erle_l;
    }
    for i in limit_subband_l..SUBBANDS {
        max_erle[i] = max_erle_h;
    }
    max_erle
}

/// Estimates the dependency of ERLE on the input signal by analyzing
/// which filter sections contribute most to the echo estimate.
pub(crate) struct SignalDependentErleEstimator {
    min_erle: f32,
    num_sections: usize,
    #[allow(dead_code, reason = "stored for consistency with C++")]
    num_blocks: usize,
    #[allow(dead_code, reason = "stored for consistency with C++")]
    delay_headroom_blocks: usize,
    band_to_subband: [usize; FFT_LENGTH_BY_2_PLUS_1],
    max_erle: [f32; SUBBANDS],
    section_boundaries_blocks: Vec<usize>,
    use_onset_detection: bool,
    erle: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    erle_onset_compensated: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    s2_section_accum: Vec<Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>>,
    erle_estimators: Vec<Vec<[f32; SUBBANDS]>>,
    erle_ref: Vec<[f32; SUBBANDS]>,
    correction_factors: Vec<Vec<[f32; SUBBANDS]>>,
    num_updates: Vec<[i32; SUBBANDS]>,
    n_active_sections: Vec<[usize; FFT_LENGTH_BY_2_PLUS_1]>,
}

impl SignalDependentErleEstimator {
    pub(crate) fn new(config: &EchoCanceller3Config, num_capture_channels: usize) -> Self {
        let num_sections = config.erle.num_sections;
        let num_blocks = config.filter.refined.length_blocks;
        let delay_headroom_blocks = config.delay.delay_headroom_samples / BLOCK_SIZE;
        let band_to_subband = form_subband_map();

        debug_assert!(num_sections <= num_blocks);
        debug_assert!(num_sections >= 1);

        let mut s = Self {
            min_erle: config.erle.min,
            num_sections,
            num_blocks,
            delay_headroom_blocks,
            band_to_subband,
            max_erle: set_max_erle_subbands(
                config.erle.max_l,
                config.erle.max_h,
                band_to_subband[FFT_LENGTH_BY_2 / 2],
            ),
            section_boundaries_blocks: set_sections_boundaries(
                delay_headroom_blocks,
                num_blocks,
                num_sections,
            ),
            use_onset_detection: config.erle.onset_detection,
            erle: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            erle_onset_compensated: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            s2_section_accum: vec![
                vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_sections];
                num_capture_channels
            ],
            erle_estimators: vec![vec![[0.0; SUBBANDS]; num_sections]; num_capture_channels],
            erle_ref: vec![[0.0; SUBBANDS]; num_capture_channels],
            correction_factors: vec![vec![[0.0; SUBBANDS]; num_sections]; num_capture_channels],
            num_updates: vec![[0; SUBBANDS]; num_capture_channels],
            n_active_sections: vec![[0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
        };
        s.reset();
        s
    }

    pub(crate) fn reset(&mut self) {
        for ch in 0..self.erle.len() {
            self.erle[ch].fill(self.min_erle);
            self.erle_onset_compensated[ch].fill(self.min_erle);
            for est in &mut self.erle_estimators[ch] {
                est.fill(self.min_erle);
            }
            self.erle_ref[ch].fill(self.min_erle);
            for factor in &mut self.correction_factors[ch] {
                factor.fill(1.0);
            }
            self.num_updates[ch].fill(0);
            self.n_active_sections[ch].fill(0);
        }
    }

    /// Returns the ERLE per frequency subband.
    pub(crate) fn erle(&self, onset_compensated: bool) -> &[[f32; FFT_LENGTH_BY_2_PLUS_1]] {
        if onset_compensated && self.use_onset_detection {
            &self.erle_onset_compensated
        } else {
            &self.erle
        }
    }

    /// Updates the ERLE estimate.
    #[allow(clippy::too_many_arguments, reason = "mirrors C++ API")]
    pub(crate) fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        average_erle: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        average_erle_onset_compensated: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        debug_assert!(self.num_sections > 1);

        self.compute_number_of_active_filter_sections(render_buffer, filter_frequency_responses);
        self.update_correction_factors(x2, y2, e2, converged_filters);

        for ch in 0..self.erle.len() {
            for k in 0..FFT_LENGTH_BY_2 {
                debug_assert!(self.n_active_sections[ch][k] < self.correction_factors[ch].len());
                let correction_factor = self.correction_factors[ch][self.n_active_sections[ch][k]]
                    [self.band_to_subband[k]];
                self.erle[ch][k] = (average_erle[ch][k] * correction_factor)
                    .clamp(self.min_erle, self.max_erle[self.band_to_subband[k]]);
                if self.use_onset_detection {
                    self.erle_onset_compensated[ch][k] = (average_erle_onset_compensated[ch][k]
                        * correction_factor)
                        .clamp(self.min_erle, self.max_erle[self.band_to_subband[k]]);
                }
            }
        }
    }

    fn compute_number_of_active_filter_sections(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
    ) {
        debug_assert!(self.num_sections > 1);
        self.compute_echo_estimate_per_filter_section(render_buffer, filter_frequency_responses);
        self.compute_active_filter_sections();
    }

    fn update_correction_factors(
        &mut self,
        x2: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        y2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        e2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        converged_filters: &[bool],
    ) {
        const X2_THRESHOLD: f32 = 44015068.0;
        const SMTH_CONSTANT_DECREASES: f32 = 0.1;
        const SMTH_CONSTANT_INCREASES: f32 = SMTH_CONSTANT_DECREASES / 2.0;

        for ch in 0..converged_filters.len() {
            if !converged_filters[ch] {
                continue;
            }

            let mut x2_subbands = [0.0f32; SUBBANDS];
            let mut e2_subbands = [0.0f32; SUBBANDS];
            let mut y2_subbands = [0.0f32; SUBBANDS];
            for subband in 0..SUBBANDS {
                let start = BAND_BOUNDARIES[subband];
                let end = BAND_BOUNDARIES[subband + 1];
                x2_subbands[subband] = x2[start..end].iter().sum();
                e2_subbands[subband] = e2[ch][start..end].iter().sum();
                y2_subbands[subband] = y2[ch][start..end].iter().sum();
            }

            let mut idx_subbands = [0usize; SUBBANDS];
            for subband in 0..SUBBANDS {
                let start = BAND_BOUNDARIES[subband];
                let end = BAND_BOUNDARIES[subband + 1].min(self.n_active_sections[ch].len());
                idx_subbands[subband] = self.n_active_sections[ch][start..end]
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(0);
            }

            let mut new_erle = [0.0f32; SUBBANDS];
            let mut is_erle_updated = [false; SUBBANDS];
            for subband in 0..SUBBANDS {
                if x2_subbands[subband] > X2_THRESHOLD && e2_subbands[subband] > 0.0 {
                    new_erle[subband] = y2_subbands[subband] / e2_subbands[subband];
                    debug_assert!(new_erle[subband] > 0.0);
                    is_erle_updated[subband] = true;
                    self.num_updates[ch][subband] += 1;
                }
            }

            for subband in 0..SUBBANDS {
                let idx = idx_subbands[subband];
                debug_assert!(idx < self.erle_estimators[ch].len());
                let alpha = if new_erle[subband] > self.erle_estimators[ch][idx][subband] {
                    SMTH_CONSTANT_INCREASES
                } else {
                    SMTH_CONSTANT_DECREASES
                };
                let alpha = if is_erle_updated[subband] { alpha } else { 0.0 };
                self.erle_estimators[ch][idx][subband] +=
                    alpha * (new_erle[subband] - self.erle_estimators[ch][idx][subband]);
                self.erle_estimators[ch][idx][subband] = self.erle_estimators[ch][idx][subband]
                    .clamp(self.min_erle, self.max_erle[subband]);
            }

            for subband in 0..SUBBANDS {
                let alpha = if new_erle[subband] > self.erle_ref[ch][subband] {
                    SMTH_CONSTANT_INCREASES
                } else {
                    SMTH_CONSTANT_DECREASES
                };
                let alpha = if is_erle_updated[subband] { alpha } else { 0.0 };
                self.erle_ref[ch][subband] +=
                    alpha * (new_erle[subband] - self.erle_ref[ch][subband]);
                self.erle_ref[ch][subband] =
                    self.erle_ref[ch][subband].clamp(self.min_erle, self.max_erle[subband]);
            }

            for subband in 0..SUBBANDS {
                const NUM_UPDATE_THR: i32 = 50;
                if is_erle_updated[subband] && self.num_updates[ch][subband] > NUM_UPDATE_THR {
                    let idx = idx_subbands[subband];
                    debug_assert!(self.erle_ref[ch][subband] > 0.0);
                    let new_correction_factor =
                        self.erle_estimators[ch][idx][subband] / self.erle_ref[ch][subband];
                    self.correction_factors[ch][idx][subband] +=
                        0.1 * (new_correction_factor - self.correction_factors[ch][idx][subband]);
                }
            }
        }
    }

    fn compute_echo_estimate_per_filter_section(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        filter_frequency_responses: &[Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>],
    ) {
        let spectrum_buffer = render_buffer.get_spectrum_buffer();
        let num_render_channels = spectrum_buffer.buffer[0].len();
        let num_capture_channels = self.s2_section_accum.len();
        let one_by_num_render_channels = 1.0 / num_render_channels as f32;

        debug_assert_eq!(
            self.s2_section_accum.len(),
            filter_frequency_responses.len()
        );

        for capture_ch in 0..num_capture_channels {
            debug_assert_eq!(
                self.s2_section_accum[capture_ch].len() + 1,
                self.section_boundaries_blocks.len()
            );
            let mut idx_render = render_buffer.position();
            idx_render = spectrum_buffer
                .index
                .offset_index(idx_render, self.section_boundaries_blocks[0] as i32);

            for section in 0..self.num_sections {
                let mut x2_section = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
                let mut h2_section = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

                let block_limit = self.section_boundaries_blocks[section + 1]
                    .min(filter_frequency_responses[capture_ch].len());
                for block in self.section_boundaries_blocks[section]..block_limit {
                    for render_ch in 0..spectrum_buffer.buffer[idx_render].len() {
                        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                            x2_section[k] += spectrum_buffer.buffer[idx_render][render_ch][k]
                                * one_by_num_render_channels;
                        }
                    }
                    for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                        h2_section[k] += filter_frequency_responses[capture_ch][block][k];
                    }
                    idx_render = spectrum_buffer.index.inc_index(idx_render);
                }

                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    self.s2_section_accum[capture_ch][section][k] = x2_section[k] * h2_section[k];
                }
            }

            // Accumulate sections.
            for section in 1..self.num_sections {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    let prev = self.s2_section_accum[capture_ch][section - 1][k];
                    self.s2_section_accum[capture_ch][section][k] += prev;
                }
            }
        }
    }

    fn compute_active_filter_sections(&mut self) {
        for ch in 0..self.n_active_sections.len() {
            self.n_active_sections[ch].fill(0);
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                let mut section = self.num_sections;
                let target = 0.9 * self.s2_section_accum[ch][self.num_sections - 1][k];
                while section > 0 && self.s2_section_accum[ch][section - 1][k] >= target {
                    section -= 1;
                    self.n_active_sections[ch][k] = section;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_buffer::BlockBuffer;
    use crate::fft_buffer::FftBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    /// Sweep different filter lengths, delay headrooms, and num_sections.
    /// Matches C++ SweepSettings test — just verifies no panics.
    #[test]
    fn sweep_settings() {
        let num_capture_channels = 1;
        let max_length_blocks = 50;
        let mut blocks = 1;
        while blocks < max_length_blocks {
            for delay_headroom in 0..5 {
                for num_sections in 2..max_length_blocks {
                    let mut cfg = EchoCanceller3Config::default();
                    cfg.filter.refined.length_blocks = blocks;
                    cfg.filter.refined_initial.length_blocks =
                        cfg.filter.refined_initial.length_blocks.min(blocks);
                    cfg.delay.delay_headroom_samples = delay_headroom * BLOCK_SIZE;
                    cfg.erle.num_sections = num_sections;
                    if cfg.validate() {
                        let s = SignalDependentErleEstimator::new(&cfg, num_capture_channels);
                        assert_eq!(s.erle(false).len(), num_capture_channels);
                    }
                }
            }
            blocks += 10;
        }
    }

    /// Longer run with a specific simple configuration.
    #[test]
    fn longer_run() {
        let num_capture_channels = 1;
        let num_render_channels = 1;
        let mut cfg = EchoCanceller3Config::default();
        cfg.filter.refined.length_blocks = 2;
        cfg.filter.refined_initial.length_blocks = 1;
        cfg.delay.delay_headroom_samples = 0;
        cfg.delay.hysteresis_limit_blocks = 0;
        cfg.erle.num_sections = 2;
        assert!(cfg.validate());

        let mut s = SignalDependentErleEstimator::new(&cfg, num_capture_channels);
        let mut average_erle = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        for e in &mut average_erle {
            e.fill(cfg.erle.max_l);
        }

        let buf_size = cfg.filter.refined.length_blocks + 10;
        let block_buffer = BlockBuffer::new(buf_size, 1, num_render_channels);
        let mut spectrum_buffer = SpectrumBuffer::new(buf_size, num_render_channels);
        let fft_buffer = FftBuffer::new(buf_size, num_render_channels);

        for slot in 0..buf_size {
            for ch in 0..num_render_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    spectrum_buffer.buffer[slot][ch][k] = 500.0 * 1000.0 * 1000.0;
                }
            }
        }

        let render_buffer =
            crate::render_buffer::RenderBuffer::new(&block_buffer, &spectrum_buffer, &fft_buffer);

        let filter_freq_resp =
            vec![
                vec![[1.0f32; FFT_LENGTH_BY_2_PLUS_1]; cfg.filter.refined.length_blocks];
                num_capture_channels
            ];
        let mut x2 = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x2.fill(500.0 * 1000.0 * 1000.0);
        let mut y2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        let mut e2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels];
        for ch in 0..num_capture_channels {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                y2[ch][k] = x2[k] * 9.0;
                e2[ch][k] = y2[ch][k] / 10.0;
            }
        }
        let converged = vec![true; num_capture_channels];

        for _ in 0..200 {
            s.update(
                &render_buffer,
                &filter_freq_resp,
                &x2,
                &y2,
                &e2,
                &average_erle,
                &average_erle,
                &converged,
            );
        }
        for &v in s.erle(false)[0].iter() {
            assert!(v.is_finite());
        }
    }
}
