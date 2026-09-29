//! Suppression gain — computes the frequency-domain gain to suppress echo.
//!
//! Ported from `modules/audio_processing/aec3/suppression_gain.h/cc`.

use crate::aec_state::AecState;
use crate::block::Block;
use crate::common::{BLOCK_SIZE, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use crate::config::{
    EchoAudibility, EchoCanceller3Config, HighFrequencySuppression, Suppressor, Tuning,
};
use crate::moving_average::MovingAverage;
use crate::nearend_detector::{DominantNearendDetector, NearendDetector, SubbandNearendDetector};
use crate::render_signal_analyzer::RenderSignalAnalyzer;
use crate::vector_math::VectorMath;

/// Input spectra and state for computing suppression gains.
pub(crate) struct SuppressionInput<'a> {
    pub nearend_spectrum: &'a [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    pub echo_spectrum: &'a [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    pub residual_echo_spectrum: &'a [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    pub residual_echo_spectrum_unbounded: &'a [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    pub comfort_noise_spectrum: &'a [[f32; FFT_LENGTH_BY_2_PLUS_1]],
    pub render_signal_analyzer: &'a RenderSignalAnalyzer,
    pub aec_state: &'a AecState,
    pub render: &'a Block,
    pub clock_drift: bool,
}

/// Limits the low frequency gains to avoid the impact of the high-pass filter
/// on the lower-frequency gain influencing the overall achieved gain.
fn limit_low_frequency_gains(gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
    gain[0] = gain[1].min(gain[2]);
    gain[1] = gain[0];
}

/// Limits the high frequency gains to avoid echo leakage due to an imperfect
/// filter.
fn limit_high_frequency_gains(
    high_frequency_suppression: &HighFrequencySuppression,
    conservative_hf_suppression: bool,
    gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let limiting_gain_band = high_frequency_suppression.limiting_gain_band as usize;
    let bands_in_limiting_gain = high_frequency_suppression.bands_in_limiting_gain as usize;
    if bands_in_limiting_gain > 0 {
        debug_assert!(limiting_gain_band + bands_in_limiting_gain <= gain.len());
        let mut min_upper_gain = 1.0f32;
        for &g in &gain[limiting_gain_band..limiting_gain_band + bands_in_limiting_gain] {
            min_upper_gain = min_upper_gain.min(g);
        }
        for g in &mut gain[limiting_gain_band + 1..] {
            *g = (*g).min(min_upper_gain);
        }
    }
    gain[FFT_LENGTH_BY_2] = gain[FFT_LENGTH_BY_2 - 1];

    if conservative_hf_suppression {
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
    echo_audibility: &EchoAudibility,
    echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
    weighted_echo: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    let weigh = |threshold: f32,
                 normalizer: f32,
                 begin: usize,
                 end: usize,
                 echo: &[f32],
                 weighted_echo: &mut [f32]| {
        for (we_k, &e_k) in weighted_echo[begin..end]
            .iter_mut()
            .zip(echo[begin..end].iter())
        {
            if e_k < threshold {
                let tmp = (threshold - e_k) * normalizer;
                *we_k = e_k * (1.0 - tmp * tmp).max(0.0);
            } else {
                *we_k = e_k;
            }
        }
    };

    let mut threshold = echo_audibility.floor_power * echo_audibility.audibility_threshold_lf;
    let mut normalizer = 1.0 / (threshold - echo_audibility.floor_power);
    weigh(threshold, normalizer, 0, 3, echo, weighted_echo);

    threshold = echo_audibility.floor_power * echo_audibility.audibility_threshold_mf;
    normalizer = 1.0 / (threshold - echo_audibility.floor_power);
    weigh(threshold, normalizer, 3, 7, echo, weighted_echo);

    threshold = echo_audibility.floor_power * echo_audibility.audibility_threshold_hf;
    normalizer = 1.0 / (threshold - echo_audibility.floor_power);
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
#[derive(Debug)]
struct GainParameters {
    max_inc_factor: f32,
    max_dec_factor_lf: f32,
    enr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
    enr_suppress: [f32; FFT_LENGTH_BY_2_PLUS_1],
    emr_transparent: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl GainParameters {
    fn new(last_lf_band: i32, first_hf_band: i32, tuning: &Tuning) -> Self {
        let mut params = Self {
            max_inc_factor: 0.0,
            max_dec_factor_lf: 0.0,
            enr_transparent: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            enr_suppress: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            emr_transparent: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        };
        params.set_config(last_lf_band, first_hf_band, tuning);
        params
    }

    /// Recomputes the parameters from the tuning config (C++
    /// `GainParameters::SetConfig`).
    fn set_config(&mut self, last_lf_band: i32, first_hf_band: i32, tuning: &Tuning) {
        self.max_inc_factor = tuning.max_inc_factor;
        self.max_dec_factor_lf = tuning.max_dec_factor_lf;
        // Compute per-band masking thresholds.
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
            self.enr_transparent[k] = (1.0 - a) * lf.enr_transparent + a * hf.enr_transparent;
            self.enr_suppress[k] = (1.0 - a) * lf.enr_suppress + a * hf.enr_suppress;
            self.emr_transparent[k] = (1.0 - a) * lf.emr_transparent + a * hf.emr_transparent;
        }
    }
}

/// Detects when the render signal can be considered to have low power and
/// consist of stationary noise.
#[derive(Debug)]
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
#[derive(Debug)]
pub(crate) struct SuppressionGain {
    vector_math: VectorMath,
    num_capture_channels: usize,
    echo_audibility_config: EchoAudibility,
    use_subband_nearend_detection: bool,
    last_gain: [f32; FFT_LENGTH_BY_2_PLUS_1],
    last_nearend: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    last_echo: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
    low_render_detector: LowNoiseRenderDetector,
    initial_state: bool,
    nearend_smoothers: Vec<MovingAverage>,
    nearend_params: GainParameters,
    normal_params: GainParameters,
    dominant_nearend_detector: NearendDetector,
}

impl SuppressionGain {
    pub(crate) fn new(
        config: &EchoCanceller3Config,
        sample_rate_hz: usize,
        num_capture_channels: usize,
    ) -> Self {
        let _ = sample_rate_hz; // unused in C++ too

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

        let backend = sonora_simd::detect_backend();
        Self {
            vector_math: VectorMath::new(backend),
            num_capture_channels,
            echo_audibility_config: config.echo_audibility.clone(),
            use_subband_nearend_detection: config.suppressor.use_subband_nearend_detection,
            last_gain: [1.0; FFT_LENGTH_BY_2_PLUS_1],
            last_nearend: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            last_echo: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; num_capture_channels],
            low_render_detector: LowNoiseRenderDetector::new(),
            initial_state: true,
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
            dominant_nearend_detector,
        }
    }

    /// Computes the suppression gains using `suppressor_config`. Set
    /// `config_changed` when `suppressor_config` differs from the one used on
    /// the previous call, so that the config-dependent state is updated.
    pub(crate) fn get_gain(
        &mut self,
        suppressor_config: &Suppressor,
        config_changed: bool,
        input: &SuppressionInput<'_>,
        high_bands_gain: &mut f32,
        low_band_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        if config_changed {
            self.update_state_depending_on_config(suppressor_config);
        }

        // Choose residual echo spectrum for dominant nearend detection.
        let echo = if suppressor_config
            .dominant_nearend_detection
            .use_unbounded_echo_spectrum
        {
            input.residual_echo_spectrum_unbounded
        } else {
            input.residual_echo_spectrum
        };

        // Update the nearend state selection.
        self.dominant_nearend_detector.update(
            input.nearend_spectrum,
            echo,
            input.comfort_noise_spectrum,
            self.initial_state,
        );

        // Compute gain for the lower band.
        let low_noise_render = self.low_render_detector.detect(input.render);
        self.lower_band_gain(suppressor_config, low_noise_render, input, low_band_gain);

        // Compute the gain for the upper bands.
        let narrow_peak_band = input.render_signal_analyzer.narrow_peak_band();

        *high_bands_gain = self.upper_bands_gain(
            suppressor_config,
            input.echo_spectrum,
            input.comfort_noise_spectrum,
            narrow_peak_band,
            input.aec_state.saturated_echo(),
            input.render,
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
    }

    /// Updates the internal state, e.g. sizes and parameters, if the config
    /// changes (C++ `UpdateStateDependingOnConfig`).
    fn update_state_depending_on_config(&mut self, suppressor_config: &Suppressor) {
        debug_assert_eq!(
            suppressor_config.use_subband_nearend_detection,
            self.use_subband_nearend_detection
        );
        // Update nearend average blocks.
        for smoother in &mut self.nearend_smoothers {
            smoother.update_memory_length(suppressor_config.nearend_average_blocks);
        }
        self.dominant_nearend_detector.set_config(suppressor_config);
        self.nearend_params.set_config(
            suppressor_config.last_lf_band,
            suppressor_config.first_hf_band,
            &suppressor_config.nearend_tuning,
        );

        self.normal_params.set_config(
            suppressor_config.last_lf_band,
            suppressor_config.first_hf_band,
            &suppressor_config.normal_tuning,
        );
    }

    /// Computes the gain to apply for the bands beyond the first band.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors C++ UpperBandsGain, which takes the active suppressor config"
    )]
    fn upper_bands_gain(
        &self,
        suppressor_config: &Suppressor,
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

        if let Some(peak_band) = narrow_peak_band
            && peak_band > FFT_LENGTH_BY_2_PLUS_1 - 10
        {
            return 0.001;
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
            * suppressor_config
                .high_bands_suppression
                .anti_howling_activation_threshold;
        let anti_howling_gain = if high_band_energy < low_band_energy.max(activation_threshold) {
            1.0
        } else {
            debug_assert!(high_band_energy > 0.0);
            suppressor_config.high_bands_suppression.anti_howling_gain
                * (low_band_energy / high_band_energy).sqrt()
        };

        let mut gain_bound = 1.0f32;
        if !self.dominant_nearend_detector.is_nearend_state() {
            // Bound the upper gain during significant echo activity.
            let cfg = &suppressor_config.high_bands_suppression;
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
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors C++ GetMinGain, which takes the active suppressor config"
    )]
    fn get_min_gain(
        &self,
        suppressor_config: &Suppressor,
        weighted_residual_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_nearend: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        last_echo: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        low_noise_render: bool,
        saturated_echo: bool,
        min_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        if !saturated_echo {
            let min_echo_power = if low_noise_render {
                self.echo_audibility_config.low_render_limit
            } else {
                self.echo_audibility_config.normal_render_limit
            };

            for k in 0..min_gain.len() {
                min_gain[k] = if weighted_residual_echo[k] > 0.0 {
                    (min_echo_power / weighted_residual_echo[k]).min(1.0)
                } else {
                    1.0
                };
            }

            if !self.initial_state || suppressor_config.lf_smoothing_during_initial_phase {
                let dec = if self.dominant_nearend_detector.is_nearend_state() {
                    self.nearend_params.max_dec_factor_lf
                } else {
                    self.normal_params.max_dec_factor_lf
                };

                for k in 0..=suppressor_config.last_lf_smoothing_band as usize {
                    // Make sure the gains of the low frequencies do not decrease
                    // too quickly after strong nearend.
                    if last_nearend[k] > last_echo[k]
                        || k <= suppressor_config.last_permanent_lf_smoothing_band as usize
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
    fn get_max_gain(
        &self,
        floor_first_increase: f32,
        max_gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        let inc = if self.dominant_nearend_detector.is_nearend_state() {
            self.nearend_params.max_inc_factor
        } else {
            self.normal_params.max_inc_factor
        };
        for (mg_k, &lg_k) in max_gain.iter_mut().zip(self.last_gain.iter()) {
            *mg_k = (lg_k * inc).max(floor_first_increase).min(1.0);
        }
    }

    fn lower_band_gain(
        &mut self,
        suppressor_config: &Suppressor,
        low_noise_render: bool,
        input: &SuppressionInput<'_>,
        gain: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
    ) {
        gain.fill(1.0);
        let saturated_echo = input.aec_state.saturated_echo();
        let mut max_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        self.get_max_gain(suppressor_config.floor_first_increase, &mut max_gain);

        for ch in 0..self.num_capture_channels {
            let mut g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            let mut nearend = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.nearend_smoothers[ch].average(&input.nearend_spectrum[ch], &mut nearend);

            // Weight echo power in terms of audibility.
            let mut weighted_residual_echo = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            weight_echo_for_audibility(
                &self.echo_audibility_config,
                &input.residual_echo_spectrum[ch],
                &mut weighted_residual_echo,
            );

            let mut min_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
            self.get_min_gain(
                suppressor_config,
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
                &input.comfort_noise_spectrum[0],
                &mut g,
            );

            // Clamp gains.
            for ((g_k, gain_k), (&max_k, &min_k)) in g
                .iter_mut()
                .zip(gain.iter_mut())
                .zip(max_gain.iter().zip(min_gain.iter()))
            {
                *g_k = g_k.min(max_k).max(min_k);
                *gain_k = gain_k.min(*g_k);
            }

            // Store data required for the gain computation of the next block.
            self.last_nearend[ch] = nearend;
            self.last_echo[ch] = weighted_residual_echo;
        }

        limit_low_frequency_gains(gain);
        // Use conservative high-frequency gains during clock-drift or when not
        // in dominant nearend.
        if !self.dominant_nearend_detector.is_nearend_state()
            || input.clock_drift
            || suppressor_config.conservative_hf_suppression
        {
            limit_high_frequency_gains(
                &suppressor_config.high_frequency_suppression,
                suppressor_config.conservative_hf_suppression,
                gain,
            );
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
    use crate::aec_state::AecStateUpdate;
    use crate::common::{NUM_BLOCKS_PER_SECOND, num_bands_for_rate};
    use crate::render_delay_buffer::RenderDelayBuffer;
    use crate::subtractor::Subtractor;
    use crate::subtractor_output::SubtractorOutput;

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

    #[test]
    fn update_state_depending_on_config() {
        let mut config = EchoCanceller3Config::default();
        config.suppressor.nearend_tuning.max_inc_factor = 2.0;
        config.suppressor.normal_tuning.max_dec_factor_lf = 0.2;

        let mut suppression_gain = SuppressionGain::new(&config, 16000, 1);

        // Initial call to set up the state.
        suppression_gain.update_state_depending_on_config(&config.suppressor);

        assert_eq!(suppression_gain.nearend_params.max_inc_factor, 2.0);
        assert_eq!(suppression_gain.normal_params.max_dec_factor_lf, 0.2);

        // Change config and verify state is updated.
        let mut new_config = config.clone();
        new_config.suppressor.nearend_tuning.max_inc_factor = 3.0;
        new_config.suppressor.normal_tuning.max_dec_factor_lf = 0.3;

        suppression_gain.update_state_depending_on_config(&new_config.suppressor);

        assert_eq!(suppression_gain.nearend_params.max_inc_factor, 3.0);
        assert_eq!(suppression_gain.normal_params.max_dec_factor_lf, 0.3);
    }

    /// Spectra and state that `get_gain` reads, mirroring the setup of the C++
    /// `SuppressionGainTest.BasicGainComputation`.
    struct GainTestSetup {
        config: EchoCanceller3Config,
        e2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        s2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        y2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        r2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        r2_unbounded: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        n2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
        output: Vec<SubtractorOutput>,
        x: Block,
        aec_state: AecState,
        subtractor: Subtractor,
        render_delay_buffer: RenderDelayBuffer,
        analyzer: RenderSignalAnalyzer,
    }

    impl GainTestSetup {
        const NUM_RENDER_CHANNELS: usize = 1;
        const NUM_CAPTURE_CHANNELS: usize = 2;
        const SAMPLE_RATE_HZ: usize = 16000;

        fn new() -> Self {
            let config = EchoCanceller3Config::default();
            let n = Self::NUM_CAPTURE_CHANNELS;
            Self {
                e2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                s2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                y2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                r2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                r2_unbounded: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                n2: vec![[0.0; FFT_LENGTH_BY_2_PLUS_1]; n],
                output: (0..n).map(|_| SubtractorOutput::default()).collect(),
                x: Block::new(
                    num_bands_for_rate(Self::SAMPLE_RATE_HZ),
                    Self::NUM_RENDER_CHANNELS,
                ),
                aec_state: AecState::new(&config, n),
                subtractor: Subtractor::new(
                    sonora_simd::detect_backend(),
                    &config,
                    Self::NUM_RENDER_CHANNELS,
                    n,
                ),
                render_delay_buffer: RenderDelayBuffer::new(
                    &config,
                    Self::SAMPLE_RATE_HZ,
                    Self::NUM_RENDER_CHANNELS,
                ),
                analyzer: RenderSignalAnalyzer::new(&config),
                config,
            }
        }

        fn update_aec_state(&mut self) {
            let render_buffer = self.render_delay_buffer.get_render_buffer();
            self.aec_state.update(&AecStateUpdate {
                external_delay: &None,
                adaptive_filter_frequency_responses: self.subtractor.filter_frequency_responses(),
                adaptive_filter_impulse_responses: self.subtractor.filter_impulse_responses(),
                render_buffer: &render_buffer,
                e2_refined: &self.e2,
                y2: &self.y2,
                subtractor_output: &self.output,
            });
        }

        fn get_gain(
            &self,
            suppression_gain: &mut SuppressionGain,
            suppressor_config: &Suppressor,
            config_changed: bool,
            g: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
        ) {
            let mut high_bands_gain = 0.0f32;
            suppression_gain.get_gain(
                suppressor_config,
                config_changed,
                &SuppressionInput {
                    nearend_spectrum: &self.e2,
                    echo_spectrum: &self.s2,
                    residual_echo_spectrum: &self.r2,
                    residual_echo_spectrum_unbounded: &self.r2_unbounded,
                    comfort_noise_spectrum: &self.n2,
                    render_signal_analyzer: &self.analyzer,
                    aec_state: &self.aec_state,
                    render: &self.x,
                    clock_drift: false,
                },
                &mut high_bands_gain,
                g,
            );
        }
    }

    /// Port of C++ `SuppressionGainTest.BasicGainComputation`: strong noise or
    /// nearend masks weak echo (gain 1), and strong echo on one of two capture
    /// channels is suppressed on the shared gain (gain 0).
    #[test]
    fn basic_gain_computation() {
        let mut t = GainTestSetup::new();
        let mut suppression_gain = SuppressionGain::new(
            &t.config,
            GainTestSetup::SAMPLE_RATE_HZ,
            GainTestSetup::NUM_CAPTURE_CHANNELS,
        );
        let suppressor = t.config.suppressor.clone();
        let mut g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        // Ensure that a strong noise is detected to mask any echoes.
        for ch in 0..GainTestSetup::NUM_CAPTURE_CHANNELS {
            t.e2[ch].fill(10.0);
            t.y2[ch].fill(10.0);
            t.r2[ch].fill(0.1);
            t.r2_unbounded[ch].fill(0.1);
            t.n2[ch].fill(100.0);
        }

        // Ensure that the gain is no longer forced to zero.
        for _ in 0..=NUM_BLOCKS_PER_SECOND / 5 + 1 {
            t.update_aec_state();
        }

        for _ in 0..100 {
            t.update_aec_state();
            t.get_gain(&mut suppression_gain, &suppressor, false, &mut g);
        }
        for &a in &g {
            assert!((a - 1.0).abs() <= 0.001, "gain {a} not near 1");
        }

        // Ensure that a strong nearend is detected to mask any echoes.
        for ch in 0..GainTestSetup::NUM_CAPTURE_CHANNELS {
            t.e2[ch].fill(100.0);
            t.y2[ch].fill(100.0);
            t.r2[ch].fill(0.1);
            t.r2_unbounded[ch].fill(0.1);
            t.s2[ch].fill(0.1);
            t.n2[ch].fill(0.0);
        }

        for _ in 0..100 {
            t.update_aec_state();
            t.get_gain(&mut suppression_gain, &suppressor, false, &mut g);
        }
        for &a in &g {
            assert!((a - 1.0).abs() <= 0.001, "gain {a} not near 1");
        }

        // Add a strong echo to one of the channels and ensure that it is
        // suppressed.
        t.e2[1].fill(1_000_000_000.0);
        t.r2[1].fill(10_000_000_000_000.0);
        t.r2_unbounded[1].fill(10_000_000_000_000.0);

        for _ in 0..10 {
            t.get_gain(&mut suppression_gain, &suppressor, false, &mut g);
        }
        for &a in &g {
            assert!(a.abs() <= 0.001, "gain {a} not near 0");
        }
    }

    /// `get_gain` must apply the config it is given when `config_changed` is
    /// set, and only then. This is how the echo remover switches suppressor
    /// tunings on the fly (upstream e10cd19640) without re-creating the
    /// suppressor; a dropped update would keep the old tuning in effect.
    #[test]
    fn get_gain_applies_config_only_when_changed() {
        let mut t = GainTestSetup::new();
        let mut suppression_gain = SuppressionGain::new(
            &t.config,
            GainTestSetup::SAMPLE_RATE_HZ,
            GainTestSetup::NUM_CAPTURE_CHANNELS,
        );
        let mut g = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let old_max_inc_factor = t.config.suppressor.nearend_tuning.max_inc_factor;
        let old_max_dec_factor_lf = t.config.suppressor.normal_tuning.max_dec_factor_lf;

        let mut new_suppressor = t.config.suppressor.clone();
        new_suppressor.nearend_tuning.max_inc_factor = old_max_inc_factor + 1.0;
        new_suppressor.normal_tuning.max_dec_factor_lf = old_max_dec_factor_lf + 0.1;
        new_suppressor.nearend_average_blocks = 1;

        t.update_aec_state();

        // Without the flag, the stored state stays as constructed.
        t.get_gain(&mut suppression_gain, &new_suppressor, false, &mut g);
        assert_eq!(
            suppression_gain.nearend_params.max_inc_factor,
            old_max_inc_factor
        );
        assert_eq!(
            suppression_gain.normal_params.max_dec_factor_lf,
            old_max_dec_factor_lf
        );

        // With the flag, the new tuning and averaging window take effect.
        t.get_gain(&mut suppression_gain, &new_suppressor, true, &mut g);
        assert_eq!(
            suppression_gain.nearend_params.max_inc_factor,
            new_suppressor.nearend_tuning.max_inc_factor
        );
        assert_eq!(
            suppression_gain.normal_params.max_dec_factor_lf,
            new_suppressor.normal_tuning.max_dec_factor_lf
        );
        let new = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut out = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        suppression_gain.nearend_smoothers[0].average(&new, &mut out);
        assert_eq!(out, new);
    }

    /// A config switch without re-creating the suppressor must also resize
    /// the nearend averaging window. Otherwise the gains keep smoothing over
    /// the old number of blocks.
    #[test]
    fn config_change_applies_new_nearend_average_blocks() {
        let config = EchoCanceller3Config::default();
        assert!(config.suppressor.nearend_average_blocks > 1);
        let mut suppression_gain = SuppressionGain::new(&config, 16000, 1);

        let old = [100.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut out = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        suppression_gain.nearend_smoothers[0].average(&old, &mut out);
        suppression_gain.nearend_smoothers[0].average(&old, &mut out);

        let mut new_config = config.clone();
        new_config.suppressor.nearend_average_blocks = 1;
        suppression_gain.update_state_depending_on_config(&new_config.suppressor);

        // With a one-block window, the output is the input alone.
        let new = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        suppression_gain.nearend_smoothers[0].average(&new, &mut out);
        assert_eq!(out, new);
    }
}
