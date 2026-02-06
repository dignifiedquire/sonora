//! Frequency response estimation for the reverb.
//!
//! Ported from `modules/audio_processing/aec3/reverb_frequency_response.h/cc`.

use crate::common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};

/// Computes the ratio of energies between the direct path and the tail,
/// skipping the DC bin.
fn average_decay_within_filter(
    freq_resp_direct_path: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    freq_resp_tail: &[f32; FFT_LENGTH_BY_2_PLUS_1],
) -> f32 {
    const SKIP_BINS: usize = 1;
    let direct_path_energy: f32 = freq_resp_direct_path[SKIP_BINS..].iter().sum();
    if direct_path_energy == 0.0 {
        return 0.0;
    }
    let tail_energy: f32 = freq_resp_tail[SKIP_BINS..].iter().sum();
    tail_energy / direct_path_energy
}

/// Estimates the frequency response of the reverb tail.
pub(crate) struct ReverbFrequencyResponse {
    use_conservative_tail_frequency_response: bool,
    average_decay: f32,
    tail_response: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl ReverbFrequencyResponse {
    pub(crate) fn new(use_conservative_tail_frequency_response: bool) -> Self {
        Self {
            use_conservative_tail_frequency_response,
            average_decay: 0.0,
            tail_response: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }

    /// Returns the estimated frequency response for the reverb.
    pub(crate) fn frequency_response(&self) -> &[f32; FFT_LENGTH_BY_2_PLUS_1] {
        &self.tail_response
    }

    /// Updates the frequency response estimate.
    pub(crate) fn update(
        &mut self,
        frequency_response: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        filter_delay_blocks: usize,
        linear_filter_quality: Option<f32>,
        stationary_block: bool,
    ) {
        if stationary_block {
            return;
        }
        if let Some(quality) = linear_filter_quality {
            self.update_inner(frequency_response, filter_delay_blocks, quality);
        }
    }

    fn update_inner(
        &mut self,
        frequency_response: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        filter_delay_blocks: usize,
        linear_filter_quality: f32,
    ) {
        let freq_resp_tail = &frequency_response[frequency_response.len() - 1];
        let freq_resp_direct_path = &frequency_response[filter_delay_blocks];

        let average_decay = average_decay_within_filter(freq_resp_direct_path, freq_resp_tail);

        let smoothing = 0.2 * linear_filter_quality;
        self.average_decay += smoothing * (average_decay - self.average_decay);

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            self.tail_response[k] = freq_resp_direct_path[k] * self.average_decay;
        }

        if self.use_conservative_tail_frequency_response {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                self.tail_response[k] = self.tail_response[k].max(freq_resp_tail[k]);
            }
        }

        // Neighbor averaging (skip DC and Nyquist).
        for k in 1..FFT_LENGTH_BY_2 {
            let avg_neighbour = 0.5 * (self.tail_response[k - 1] + self.tail_response[k + 1]);
            self.tail_response[k] = self.tail_response[k].max(avg_neighbour);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_response_is_zero() {
        let rfr = ReverbFrequencyResponse::new(false);
        for &v in rfr.frequency_response() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn stationary_block_skips_update() {
        let mut rfr = ReverbFrequencyResponse::new(false);
        let mut freq_resp = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 13];
        freq_resp[0].fill(1.0);
        rfr.update(&freq_resp, 0, Some(1.0), true);
        for &v in rfr.frequency_response() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn update_produces_nonzero_response() {
        let mut rfr = ReverbFrequencyResponse::new(false);
        let mut freq_resp = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 13];
        // Direct path has energy.
        freq_resp[2].fill(10.0);
        // Tail has some energy.
        freq_resp[12].fill(1.0);
        rfr.update(&freq_resp, 2, Some(1.0), false);
        // The tail response should be non-zero now.
        let has_nonzero = rfr.frequency_response().iter().any(|&v| v > 0.0);
        assert!(has_nonzero);
    }

    #[test]
    fn conservative_mode_uses_max() {
        let mut rfr_conservative = ReverbFrequencyResponse::new(true);
        let mut rfr_normal = ReverbFrequencyResponse::new(false);
        let mut freq_resp = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 13];
        freq_resp[2].fill(10.0);
        // Make tail quite large.
        freq_resp[12].fill(5.0);
        rfr_conservative.update(&freq_resp, 2, Some(1.0), false);
        rfr_normal.update(&freq_resp, 2, Some(1.0), false);
        // Conservative mode should produce >= normal mode values.
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert!(rfr_conservative.frequency_response()[k] >= rfr_normal.frequency_response()[k]);
        }
    }
}
