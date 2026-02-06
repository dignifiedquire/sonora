//! Suppression gain — computes the frequency-domain gain to suppress echo.
//!
//! Ported from `modules/audio_processing/aec3/suppression_gain.h/cc`.

use crate::aec_state::AecState;
use crate::block::Block;
use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::EchoCanceller3Config;
use crate::moving_average::MovingAverage;
use crate::nearend_detector::{DominantNearendDetector, NearendDetector, SubbandNearendDetector};
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::vector_math::VectorMath;

/// Limits the low frequency gains to avoid the impact of the high-pass filter
/// on the lower-frequency gain influencing the overall achieved gain.
fn limit_low_frequency_gains(gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    gain[0] = gain[1].min(gain[2]);
    gain[1] = gain[0];
}

/// Limits the high frequency gains to avoid echo leakage due to an imperfect
/// filter.
fn limit_high_frequency_gains(
    config: &crate::config::Suppressor,
    gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let limiting_gain_band = config.high_frequency_suppression.limiting_gain_band as usize;
    let bands_in_limiting_gain = config.high_frequency_suppression.bands_in_limiting_gain as usize;
    if bands_in_limiting_gain > 0 {
        debug_assert!(limiting_gain_band + bands_in_limiting_gain <= gain.len());
        let mut min_upper_gain = 1.0f32;
        for band in limiting_gain_band..limiting_gain_band + bands_in_limiting_gain {
            min_upper_gain = min_upper_gain.min(gain[band]);
        }
        for g in &mut gain[limiting_gain_band + 1..] {
            *g = (*g).min(min_upper_gain);
        }
    }
    gain[FFT_LENGTH_BY_2] = gain[FFT_LENGTH_BY_2 - 1];

    if config.conservative_hf_suppression {
        // Limits the gain in the frequencies for which the adaptive filter has
        // not converged.
        const K_UPPER_ACCURATE_BAND_PLUS_1: usize = 29;

        let one_by_bands_in_sum = 1.0 / (K_UPPER_ACCURATE_BAND_PLUS_1 - 20) as f32;
        let hf_gain_bound: f32 =
            gain[20..K_UPPER_ACCURATE_BAND_PLUS_1].iter().sum::<f32>() * one_by_bands_in_sum;

        for g in &mut gain[K_UPPER_ACCURATE_BAND_PLUS_1..] {
            *g = (*g).min(hf_gain_bound);
        }
    }
}

/// Scales the echo according to assessed audibility at the other end.
fn weight_echo_for_audibility(
    config: &EchoCanceller3Config,
    echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    weighted_echo: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let weigh = |threshold: f32,
                 normalizer: f32,
                 begin: usize,
                 end: usize,
                 echo: &[f32],
                 weighted_echo: &mut [f32]| {
        for k in begin..end {
            if echo[k] < threshold {
                let tmp = (threshold - echo[k]) * normalizer;
                weighted_echo[k] = echo[k] * (1.0 - tmp * tmp).max(0.0);
            } else {
                weighted_echo[k] = echo[k];
            }
        }
    };

    let mut threshold =
        config.echo_audibility.floor_power * config.echo_audibility.audibility_threshold_lf;
    let mut normalizer = 1.0 / (threshold - config.echo_audibility.floor_power);
    weigh(threshold, normalizer, 0, 3, echo, weighted_echo);

    threshold = config.echo_audibility.floor_power * config.echo_audibility.audibility_threshold_mf;
    normalizer = 1.0 / (threshold - config.echo_audibility.floor_power);
    weigh(threshold, normalizer, 3, 7, echo, weighted_echo);

    threshold = config.echo_audibility.floor_power * config.echo_audibility.audibility_threshold_hf;
    normalizer = 1.0 / (threshold - config.echo_audibility.floor_power);
    weigh(
        threshold,
        normalizer,
        7,
        FFT_LENGTH_BY_2_PLUS_1,
        echo,
        weighted_echo,
    );
}

/// Per-band masking thresholds computed from the tuning config.
struct GainParameters {
    max_inc_factor: f32,
    max_dec_factor_lf: f32,
    enr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
    enr_suppress: [f32; FFT_LENGTH_BY_2_PLUS_1],
    emr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl GainParameters {
    fn new(last_lf_band: i32, first_hf_band: i32, tuning: &crate::config::Tuning) -> Self {
        let mut enr_transparent = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut enr_suppress = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut emr_transparent = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        debug_assert!(last_lf_band < first_hf_band);

        let lf = &tuning.mask_lf;
        let hf = &tuning.mask_hf;

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let a = if k as i32 <= last_lf_band {
                0.0f32
            } else if (k as i32) < first_hf_band {
                (k as i32 - last_lf_band) as f32 / (first_hf_band - last_lf_band) as f32
            } else {
                1.0f32
            };
            enr_transparent[k] = (1.0 - a) * lf.enr_transparent + a * hf.enr_transparent;
            enr_suppress[k] = (1.0 - a) * lf.enr_suppress + a * hf.enr_suppress;
            emr_transparent[k] = (1.0 - a) * lf.emr_transparent + a * hf.emr_transparent;
        }

        Self {
            max_inc_factor: tuning.max_inc_factor,
            max_dec_factor_lf: tuning.max_dec_factor_lf,
            enr_transparent,
            enr_suppress,
            emr_transparent,
        }
    }
}

/// Detects when the render signal can be considered to have low power and
/// consist of stationary noise.
struct LowNoiseRenderDetector {
    average_power: f32,
}

impl LowNoiseRenderDetector {
    fn new() -> Self {
        Self {
            average_power: 32768.0 * 32768.0,
        }
    }

    fn detect(&mut self, render: &Block) -> bool {
        let mut x2_sum = 0.0f32;
        let mut x2_max = 0.0f32;
        for ch in 0..render.num_channels() {
            for &x_k in render.view(0, ch) {
                let x2 = x_k * x_k;
                x2_sum += x2;
                x2_max = x2_max.max(x2);
            }
        }
        x2_sum /= render.num_channels() as f32;

        const K_THRESHOLD: f32 = 50.0 * 50.0 * 64.0;
        let low_noise_render =
            self.average_power < K_THRESHOLD && x2_max < 3.0 * self.average_power;
        self.average_power = self.average_power * 0.9 + x2_sum * 0.1;
        low_noise_render
    }
}

/// Computes the frequency-domain suppression gain.
pub(crate) struct SuppressionGain {
    vector_math: VectorMath,
    config: EchoCanceller3Config,
    num_capture_channels: usize,
    state_change_duration_blocks: i32,
    last_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    last_nearend: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    last_echo: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    low_render_detector: LowNoiseRenderDetector,
    initial_state: bool,
    initial_state_change_counter: i32,
    nearend_smoothers: Vec<MovingAverage>,
    nearend_params: GainParameters,
    normal_params: GainParameters,
    use_unbounded_echo_spectrum: bool,
    dominant_nearend_detector: NearendDetector,
}

impl SuppressionGain {
    pub(crate) fn new(
        config: &EchoCanceller3Config,
        sample_rate_hz: usize,
        num_capture_channels: usize,
    ) -> Self {
        let _ = sample_rate_hz; // unused in C++ too
        let state_change_duration_blocks = config.filter.config_change_duration_blocks as i32;
        debug_assert!(state_change_duration_blocks > 0);

        let dominant_nearend_detector = if config.suppressor.use_subband_nearend_detection {
            NearendDetector::Subband(SubbandNearendDetector::new(
                &config.suppressor.subband_nearend_detection,
                num_capture_channels,
            ))
        } else {
            NearendDetector::Dominant(DominantNearendDetector::new(
                &config.suppressor.dominant_nearend_detection,
                num_capture_channels,
            ))
        };

        let backend = webrtc_simd::detect_backend();
        Self {
            vector_math: VectorMath::new(backend),
            config: config.clone(),
            num_capture_channels,
            state_change_duration_blocks,
            last_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            last_nearend: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            last_echo: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            low_render_detector: LowNoiseRenderDetector::new(),
            initial_state: true,
            initial_state_change_counter: 0,
            nearend_smoothers: (0..num_capture_channels)
                .map(|_| {
                    MovingAverage::new(
                        FFT_LENGTH_BY_2_PLUS_1,
                        config.suppressor.nearend_average_blocks,
                    )
                })
                .collect(),
            nearend_params: GainParameters::new(
                config.suppressor.last_lf_band,
                config.suppressor.first_hf_band,
                &config.suppressor.nearend_tuning,
            ),
            normal_params: GainParameters::new(
                config.suppressor.last_lf_band,
                config.suppressor.first_hf_band,
                &config.suppressor.normal_tuning,
            ),
            use_unbounded_echo_spectrum: config
                .suppressor
                .dominant_nearend_detection
                .use_unbounded_echo_spectrum,
            dominant_nearend_detector,
        }
    }

    /// Computes the suppression gains.
    #[allow(clippy::too_many_arguments, reason = "matches C++ API")]
    pub(crate) fn get_gain(
        &mut self,
        nearend_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo_spectrum_unbounded: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        render_signal_analyzer: &RenderSignalAnalyzer,
        aec_state: &AecState,
        render: &Block,
        clock_drift: bool,
        high_bands_gain: &mut f32,
        low_band_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        // Choose residual echo spectrum for dominant nearend detection.
        let echo = if self.use_unbounded_echo_spectrum {
            residual_echo_spectrum_unbounded
        } else {
            residual_echo_spectrum
        };

        // Update the nearend state selection.
        self.dominant_nearend_detector.update(
            nearend_spectrum,
            echo,
            comfort_noise_spectrum,
            self.initial_state,
        );

        // Compute gain for the lower band.
        let low_noise_render = self.low_render_detector.detect(render);
        self.lower_band_gain(
            low_noise_render,
            aec_state,
            nearend_spectrum,
            residual_echo_spectrum,
            comfort_noise_spectrum,
            clock_drift,
            low_band_gain,
        );

        // Compute the gain for the upper bands.
        let narrow_peak_band = render_signal_analyzer.narrow_peak_band();

        *high_bands_gain = self.upper_bands_gain(
            echo_spectrum,
            comfort_noise_spectrum,
            narrow_peak_band,
            aec_state.saturated_echo(),
            render,
            low_band_gain,
        );
    }

    /// Returns true if the dominant nearend detector is in nearend state.
    pub(crate) fn is_dominant_nearend(&self) -> bool {
        self.dominant_nearend_detector.is_nearend_state()
    }

    /// Toggles the usage of the initial state.
    pub(crate) fn set_initial_state(&mut self, state: bool) {
        self.initial_state = state;
        if state {
            self.initial_state_change_counter = self.state_change_duration_blocks;
        } else {
            self.initial_state_change_counter = 0;
        }
    }

    /// Computes the gain to apply for the bands beyond the first band.
    fn upper_bands_gain(
        &self,
        echo_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise_spectrum: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        narrow_peak_band: Option<usize>,
        saturated_echo: bool,
        render: &Block,
        low_band_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    ) -> f32 {
        debug_assert!(render.num_bands() > 0);
        if render.num_bands() == 1 {
            return 1.0;
        }
        let num_render_channels = render.num_channels();

        if let Some(peak_band) = narrow_peak_band {
            if peak_band > FFT_LENGTH_BY_2_PLUS_1 - 10 {
                return 0.001;
            }
        }

        const K_LOW_BAND_GAIN_LIMIT: usize = FFT_LENGTH_BY_2 / 2;
        let gain_below_8_khz = low_band_gain[K_LOW_BAND_GAIN_LIMIT..]
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(1.0);

        // Always attenuate the upper bands when there is saturated echo.
        if saturated_echo {
            return 0.001f32.min(gain_below_8_khz);
        }

        // Compute the upper and lower band energies.
        let mut low_band_energy = 0.0f32;
        for ch in 0..num_render_channels {
            let channel_energy: f32 = render.view(0, ch).iter().map(|x| x * x).sum();
            low_band_energy = low_band_energy.max(channel_energy);
        }
        let mut high_band_energy = 0.0f32;
        for k in 1..render.num_bands() {
            for ch in 0..num_render_channels {
                let energy: f32 = render.view(k, ch).iter().map(|x| x * x).sum();
                high_band_energy = high_band_energy.max(energy);
            }
        }

        // If there is more power in the lower frequencies than the upper
        // frequencies, or if the power in upper frequencies is low, do not
        // bound the gain in the upper bands.
        let activation_threshold = BLOCK_SIZE as f32
            * self
                .config
                .suppressor
                .high_bands_suppression
                .anti_howling_activation_threshold;
        let anti_howling_gain = if high_band_energy < low_band_energy.max(activation_threshold) {
            1.0
        } else {
            debug_assert!(high_band_energy > 0.0);
            self.config
                .suppressor
                .high_bands_suppression
                .anti_howling_gain
                * (low_band_energy / high_band_energy).sqrt()
        };

        let mut gain_bound = 1.0f32;
        if !self.dominant_nearend_detector.is_nearend_state() {
            // Bound the upper gain during significant echo activity.
            let cfg = &self.config.suppressor.high_bands_suppression;
            let low_frequency_energy =
                |spectrum: &[f32; FFT_LENGTH_BY_2_PLUS_1]| -> f32 { spectrum[1..16].iter().sum() };
            for ch in 0..self.num_capture_channels {
                let echo_sum = low_frequency_energy(&echo_spectrum[ch]);
                let noise_sum = low_frequency_energy(&comfort_noise_spectrum[ch]);
                if echo_sum > cfg.enr_threshold * noise_sum {
                    gain_bound = cfg.max_gain_during_echo;
                    break;
                }
            }
        }

        // Choose the gain as the minimum of the lower and upper gains.
        gain_below_8_khz.min(anti_howling_gain).min(gain_bound)
    }

    /// Computes the gain to reduce the echo to a non audible level.
    fn gain_to_no_audible_echo(
        &self,
        nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        masker: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        let p = if self.dominant_nearend_detector.is_nearend_state() {
            &self.nearend_params
        } else {
            &self.normal_params
        };
        for k in 0..gain.len() {
            let enr = echo[k] / (nearend[k] + 1.0); // Echo-to-nearend ratio
            let emr = echo[k] / (masker[k] + 1.0); // Echo-to-masker (noise) ratio
            let mut g = 1.0f32;
            if enr > p.enr_transparent[k] && emr > p.emr_transparent[k] {
                g = (p.enr_suppress[k] - enr) / (p.enr_suppress[k] - p.enr_transparent[k]);
                g = g.max(p.emr_transparent[k] / emr);
            }
            gain[k] = g;
        }
    }

    /// Compute the minimum gain as the attenuating gain to put the signal just
    /// above the zero sample values.
    fn get_min_gain(
        &self,
        weighted_residual_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        low_noise_render: bool,
        saturated_echo: bool,
        min_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        if !saturated_echo {
            let min_echo_power = if low_noise_render {
                self.config.echo_audibility.low_render_limit
            } else {
                self.config.echo_audibility.normal_render_limit
            };

            for k in 0..min_gain.len() {
                min_gain[k] = if weighted_residual_echo[k] > 0.0 {
                    (min_echo_power / weighted_residual_echo[k]).min(1.0)
                } else {
                    1.0
                };
            }

            if !self.initial_state || self.config.suppressor.lf_smoothing_during_initial_phase {
                let dec = if self.dominant_nearend_detector.is_nearend_state() {
                    self.nearend_params.max_dec_factor_lf
                } else {
                    self.normal_params.max_dec_factor_lf
                };

                for k in 0..=self.config.suppressor.last_lf_smoothing_band as usize {
                    // Make sure the gains of the low frequencies do not decrease
                    // too quickly after strong nearend.
                    if last_nearend[k] > last_echo[k]
                        || k <= self.config.suppressor.last_permanent_lf_smoothing_band as usize
                    {
                        min_gain[k] = min_gain[k].max(self.last_gain[k] * dec);
                        min_gain[k] = min_gain[k].min(1.0);
                    }
                }
            }
        } else {
            min_gain.fill(0.0);
        }
    }

    /// Compute the maximum gain by limiting the gain increase from the previous
    /// gain.
    fn get_max_gain(&self, max_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        let inc = if self.dominant_nearend_detector.is_nearend_state() {
            self.nearend_params.max_inc_factor
        } else {
            self.normal_params.max_inc_factor
        };
        let floor = self.config.suppressor.floor_first_increase;
        for k in 0..max_gain.len() {
            max_gain[k] = (self.last_gain[k] * inc).max(floor).min(1.0);
        }
    }

    #[allow(clippy::too_many_arguments, reason = "matches C++ method signature")]
    fn lower_band_gain(
        &mut self,
        low_noise_render: bool,
        aec_state: &AecState,
        suppressor_input: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        residual_echo: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        comfort_noise: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
        clock_drift: bool,
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        gain.fill(1.0);
        let saturated_echo = aec_state.saturated_echo();
        let mut max_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        self.get_max_gain(&mut max_gain);

        for ch in 0..self.num_capture_channels {
            let mut g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut nearend = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.nearend_smoothers[ch].average(&suppressor_input[ch], &mut nearend);

            // Weight echo power in terms of audibility.
            let mut weighted_residual_echo = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            weight_echo_for_audibility(
                &self.config,
                &residual_echo[ch],
                &mut weighted_residual_echo,
            );

            let mut min_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.get_min_gain(
                &weighted_residual_echo,
                &self.last_nearend[ch],
                &self.last_echo[ch],
                low_noise_render,
                saturated_echo,
                &mut min_gain,
            );

            self.gain_to_no_audible_echo(
                &nearend,
                &weighted_residual_echo,
                &comfort_noise[0],
                &mut g,
            );

            // Clamp gains.
            for k in 0..gain.len() {
                g[k] = g[k].min(max_gain[k]).max(min_gain[k]);
                gain[k] = gain[k].min(g[k]);
            }

            // Store data required for the gain computation of the next block.
            self.last_nearend[ch] = nearend;
            self.last_echo[ch] = weighted_residual_echo;
        }

        limit_low_frequency_gains(gain);
        // Use conservative high-frequency gains during clock-drift or when not
        // in dominant nearend.
        if !self.dominant_nearend_detector.is_nearend_state()
            || clock_drift
            || self.config.suppressor.conservative_hf_suppression
        {
            limit_high_frequency_gains(&self.config.suppressor, gain);
        }

        // Store computed gains.
        self.last_gain = *gain;

        // Transform gains to amplitude domain.
        self.vector_math.sqrt(gain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_gain_is_transparent() {
        let config = EchoCanceller3Config::default();
        let gain = SuppressionGain::new(&config, 16000, 1);
        // All last_gain should be 1.0 initially.
        for &g in &gain.last_gain {
            assert_eq!(g, 1.0);
        }
    }

    #[test]
    fn low_noise_detector_high_power_not_low() {
        let mut det = LowNoiseRenderDetector::new();
        let render = Block::new_with_value(1, 1, 1000.0);
        assert!(!det.detect(&render));
    }
}
