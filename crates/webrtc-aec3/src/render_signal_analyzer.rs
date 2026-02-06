//! Render signal analyzer — analyzes the render signal for narrowband content
//! and excitation levels.
//!
//! Ported from `modules/audio_processing/aec3/render_signal_analyzer.h/cc`.

use crate::common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::EchoCanceller3Config;
use crate::render_buffer::RenderBuffer;

const COUNTER_THRESHOLD: usize = 5;

/// Identifies local bands with narrow characteristics.
fn identify_small_narrow_band_regions(
    render_buffer: &RenderBuffer<'_>,
    delay_partitions: Option<usize>,
    narrow_band_counters: &mut [usize; FFT_LENGTH_BY_2 - 1],
) {
    let Some(delay) = delay_partitions else {
        narrow_band_counters.fill(0);
        return;
    };

    let mut channel_counters = [0usize; FFT_LENGTH_BY_2 - 1];
    let x2 = render_buffer.spectrum(delay as i32);
    for ch_spectrum in x2 {
        for k in 1..FFT_LENGTH_BY_2 {
            if ch_spectrum[k] > 3.0 * ch_spectrum[k - 1].max(ch_spectrum[k + 1]) {
                channel_counters[k - 1] += 1;
            }
        }
    }
    for k in 0..FFT_LENGTH_BY_2 - 1 {
        narrow_band_counters[k] = if channel_counters[k] > 0 {
            narrow_band_counters[k] + 1
        } else {
            0
        };
    }
}

/// Identifies whether the signal has a single strong narrow-band component.
fn identify_strong_narrow_band_component(
    render_buffer: &RenderBuffer<'_>,
    strong_peak_freeze_duration: usize,
    narrow_peak_band: &mut Option<usize>,
    narrow_peak_counter: &mut usize,
) {
    if let Some(_) = narrow_peak_band {
        *narrow_peak_counter += 1;
        if *narrow_peak_counter > strong_peak_freeze_duration {
            *narrow_peak_band = None;
        }
    }

    let x_latest = render_buffer.get_block(0);
    let mut max_peak_level = 0.0f32;

    for channel in 0..x_latest.num_channels() {
        let x2_latest = &render_buffer.spectrum(0)[channel];

        // Identify the spectral peak.
        let peak_bin = x2_latest
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Compute the level around the peak.
        let mut non_peak_power = 0.0f32;
        let start_low = peak_bin.saturating_sub(14);
        let end_low = peak_bin.saturating_sub(4);
        for k in start_low..end_low {
            non_peak_power = non_peak_power.max(x2_latest[k]);
        }
        let start_high = peak_bin + 5;
        let end_high = (peak_bin + 15).min(FFT_LENGTH_BY_2_PLUS_1);
        for k in start_high..end_high {
            non_peak_power = non_peak_power.max(x2_latest[k]);
        }

        // Assess the render signal strength.
        let band0 = x_latest.view(0, channel);
        let mut max_abs = band0.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

        if x_latest.num_bands() > 1 {
            let band1 = x_latest.view(1, channel);
            let max_abs1 = band1.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            max_abs = max_abs.max(max_abs1);
        }

        // Detect whether the spectral peak has a strong narrowband nature.
        let peak_level = x2_latest[peak_bin];
        if peak_bin > 0 && max_abs > 100.0 && peak_level > 100.0 * non_peak_power {
            if peak_level > max_peak_level {
                max_peak_level = peak_level;
                *narrow_peak_band = Some(peak_bin);
                *narrow_peak_counter = 0;
            }
        }
    }
}

/// Analyzes the properties of the render signal.
pub(crate) struct RenderSignalAnalyzer {
    strong_peak_freeze_duration: usize,
    narrow_band_counters: [usize; FFT_LENGTH_BY_2 - 1],
    narrow_peak_band: Option<usize>,
    narrow_peak_counter: usize,
}

impl Default for RenderSignalAnalyzer {
    fn default() -> Self {
        Self::new(&EchoCanceller3Config::default())
    }
}

impl RenderSignalAnalyzer {
    pub(crate) fn new(config: &EchoCanceller3Config) -> Self {
        Self {
            strong_peak_freeze_duration: config.filter.refined.length_blocks,
            narrow_band_counters: [0; FFT_LENGTH_BY_2 - 1],
            narrow_peak_band: None,
            narrow_peak_counter: 0,
        }
    }

    /// Updates the render signal analysis.
    pub(crate) fn update(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        delay_partitions: Option<usize>,
    ) {
        identify_small_narrow_band_regions(
            render_buffer,
            delay_partitions,
            &mut self.narrow_band_counters,
        );
        identify_strong_narrow_band_component(
            render_buffer,
            self.strong_peak_freeze_duration,
            &mut self.narrow_peak_band,
            &mut self.narrow_peak_counter,
        );
    }

    /// Returns true if the render signal is poorly exciting.
    pub(crate) fn poor_signal_excitation(&self) -> bool {
        debug_assert!(self.narrow_band_counters.len() > 2);
        self.narrow_band_counters.iter().any(|&a| a > 10)
    }

    /// Zeros the array around regions with narrow band signal characteristics.
    pub(crate) fn mask_regions_around_narrow_bands(&self, v: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        if self.narrow_band_counters[0] > COUNTER_THRESHOLD {
            v[1] = 0.0;
            v[0] = 0.0;
        }
        for k in 2..FFT_LENGTH_BY_2 - 1 {
            if self.narrow_band_counters[k - 1] > COUNTER_THRESHOLD {
                v[k - 2] = 0.0;
                v[k - 1] = 0.0;
                v[k] = 0.0;
                v[k + 1] = 0.0;
                v[k + 2] = 0.0;
            }
        }
        if self.narrow_band_counters[FFT_LENGTH_BY_2 - 2] > COUNTER_THRESHOLD {
            v[FFT_LENGTH_BY_2] = 0.0;
            v[FFT_LENGTH_BY_2 - 1] = 0.0;
        }
    }

    /// Returns the narrow peak band, if detected.
    pub(crate) fn narrow_peak_band(&self) -> Option<usize> {
        self.narrow_peak_band
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::block_buffer::BlockBuffer;
    use crate::fft_buffer::FftBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    fn make_render_buffer(
        size: usize,
        num_channels: usize,
    ) -> (BlockBuffer, SpectrumBuffer, FftBuffer) {
        (
            BlockBuffer::new(size, 1, num_channels),
            SpectrumBuffer::new(size, num_channels),
            FftBuffer::new(size, num_channels),
        )
    }

    #[test]
    fn no_narrow_bands_initially() {
        let config = EchoCanceller3Config::default();
        let analyzer = RenderSignalAnalyzer::new(&config);
        assert!(!analyzer.poor_signal_excitation());
        assert!(analyzer.narrow_peak_band().is_none());
    }

    #[test]
    fn mask_regions_with_no_narrow_bands() {
        let config = EchoCanceller3Config::default();
        let analyzer = RenderSignalAnalyzer::new(&config);
        let mut v = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        analyzer.mask_regions_around_narrow_bands(&mut v);
        // No masking should occur — all values stay 1.0.
        for &val in &v {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn update_without_delay_resets_counters() {
        let config = EchoCanceller3Config::default();
        let mut analyzer = RenderSignalAnalyzer::new(&config);
        let size = 20;
        let (bb, sb, fb) = make_render_buffer(size, 1);
        let rb = RenderBuffer::new(&bb, &sb, &fb);
        analyzer.update(&rb, None);
        assert!(!analyzer.poor_signal_excitation());
    }
}
