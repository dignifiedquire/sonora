//! Clipping prediction and clipped level step estimation.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/clipping_predictor.h/cc`
//! and `webrtc/modules/audio_processing/agc2/gain_map_internal.h`.

use crate::clipping_predictor_level_buffer::{ClippingPredictorLevelBuffer, Level};
use crate::common::float_s16_to_dbfs;

/// Maximum gain change for the clipping predictor (dB).
const MAX_GAIN_CHANGE: i32 = 15;

/// Maps input volumes (0..=255) to gains in dB.
///
/// Generated with numpy:
/// ```text
/// SI = 2                        # Initial slope.
/// SF = 0.25                     # Final slope.
/// D = 8/256                     # Quantization factor.
/// x = np.linspace(0, 255, 256)  # Input volumes.
/// y = (SF * x + (SI - SF) * (1 - np.exp(-D*x)) / D - 56).round()
/// ```
const GAIN_MAP: [i32; 256] = [
    -56, -54, -52, -50, -48, -47, -45, -43, -42, -40, -38, -37, -35, -34, -33, -31, -30, -29, -27,
    -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14, -14, -13, -12, -11, -10, -9, -8,
    -8, -7, -6, -5, -5, -4, -3, -2, -2, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
    9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 19,
    19, 19, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27,
    27, 27, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33,
    34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40,
    40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46,
    46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52,
    52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58,
    58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64,
];

/// Clipping predictor mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClippingPredictorMode {
    /// Crest factor-based clipping event prediction.
    ClippingEvent,
    /// Crest factor-based clipping peak prediction with adaptive step estimation.
    AdaptiveStepClippingPeak,
    /// Crest factor-based clipping peak prediction with fixed step.
    FixedStepClippingPeak,
}

/// Configuration for the clipping predictor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClippingPredictorConfig {
    pub enabled: bool,
    pub mode: ClippingPredictorMode,
    pub window_length: i32,
    pub reference_window_length: i32,
    pub reference_window_delay: i32,
    pub clipping_threshold: f32,
    pub crest_factor_margin: f32,
}

impl Default for ClippingPredictorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ClippingPredictorMode::ClippingEvent,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        }
    }
}

/// Returns an input volume in the [`min_input_volume`, `max_input_volume`] range
/// that reduces `gain_error_db`, according to a fixed gain map.
fn compute_volume_update(
    gain_error_db: i32,
    input_volume: i32,
    min_input_volume: i32,
    max_input_volume: i32,
) -> i32 {
    debug_assert!(input_volume >= 0);
    debug_assert!(input_volume <= max_input_volume);
    if gain_error_db == 0 {
        return input_volume;
    }
    let mut new_volume = input_volume;
    if gain_error_db > 0 {
        while GAIN_MAP[new_volume as usize] - GAIN_MAP[input_volume as usize] < gain_error_db
            && new_volume < max_input_volume
        {
            new_volume += 1;
        }
    } else {
        while GAIN_MAP[new_volume as usize] - GAIN_MAP[input_volume as usize] > gain_error_db
            && new_volume > min_input_volume
        {
            new_volume -= 1;
        }
    }
    new_volume
}

/// Computes the crest factor (peak dBFS - RMS dBFS) from a level metric.
fn compute_crest_factor(level: &Level) -> f32 {
    float_s16_to_dbfs(level.max) - float_s16_to_dbfs(level.average.sqrt())
}

/// Frame-wise clipping prediction and clipped level step estimation.
///
/// Analyzes 10 ms multi-channel frames and estimates an analog mic level
/// decrease step to possibly avoid clipping when predicted.
pub(crate) enum ClippingPredictor {
    /// Crest factor-based clipping event prediction.
    Event(ClippingEventPredictor),
    /// Crest factor-based clipping peak prediction.
    Peak(ClippingPeakPredictor),
}

impl ClippingPredictor {
    /// Resets the internal state.
    pub(crate) fn reset(&mut self) {
        match self {
            Self::Event(p) => p.reset(),
            Self::Peak(p) => p.reset(),
        }
    }

    /// Analyzes a 10 ms multi-channel audio frame.
    /// `frame` is a slice of channel slices (each channel has `samples_per_channel` samples).
    pub(crate) fn analyze(&mut self, frame: &[&[f32]]) {
        match self {
            Self::Event(p) => p.analyze(frame),
            Self::Peak(p) => p.analyze(frame),
        }
    }

    /// Predicts if clipping is going to occur for the specified `channel` and,
    /// if so, returns a recommended analog mic level decrease step.
    /// Returns `None` if clipping is not predicted.
    pub(crate) fn estimate_clipped_level_step(
        &self,
        channel: usize,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32> {
        match self {
            Self::Event(p) => p.estimate_clipped_level_step(
                channel,
                level,
                default_step,
                min_mic_level,
                max_mic_level,
            ),
            Self::Peak(p) => p.estimate_clipped_level_step(
                channel,
                level,
                default_step,
                min_mic_level,
                max_mic_level,
            ),
        }
    }
}

/// Creates a [`ClippingPredictor`] based on the provided config.
/// Returns `None` if the config is disabled.
pub(crate) fn create_clipping_predictor(
    num_channels: usize,
    config: &ClippingPredictorConfig,
) -> Option<ClippingPredictor> {
    if !config.enabled {
        return None;
    }
    match config.mode {
        ClippingPredictorMode::ClippingEvent => {
            Some(ClippingPredictor::Event(ClippingEventPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold,
                config.crest_factor_margin,
            )))
        }
        ClippingPredictorMode::AdaptiveStepClippingPeak => {
            Some(ClippingPredictor::Peak(ClippingPeakPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold as i32,
                true,
            )))
        }
        ClippingPredictorMode::FixedStepClippingPeak => {
            Some(ClippingPredictor::Peak(ClippingPeakPredictor::new(
                num_channels,
                config.window_length,
                config.reference_window_length,
                config.reference_window_delay,
                config.clipping_threshold as i32,
                false,
            )))
        }
    }
}

/// Crest factor-based clipping event prediction.
pub(crate) struct ClippingEventPredictor {
    ch_buffers: Vec<ClippingPredictorLevelBuffer>,
    window_length: i32,
    reference_window_length: i32,
    reference_window_delay: i32,
    clipping_threshold: f32,
    crest_factor_margin: f32,
}

impl ClippingEventPredictor {
    fn new(
        num_channels: usize,
        window_length: i32,
        reference_window_length: i32,
        reference_window_delay: i32,
        clipping_threshold: f32,
        crest_factor_margin: f32,
    ) -> Self {
        debug_assert!(num_channels > 0);
        debug_assert!(window_length > 0);
        debug_assert!(reference_window_length > 0);
        debug_assert!(reference_window_delay >= 0);
        debug_assert!(reference_window_length + reference_window_delay > window_length);

        let buffer_length = reference_window_delay + reference_window_length;
        let ch_buffers = (0..num_channels)
            .map(|_| ClippingPredictorLevelBuffer::new(buffer_length))
            .collect();

        Self {
            ch_buffers,
            window_length,
            reference_window_length,
            reference_window_delay,
            clipping_threshold,
            crest_factor_margin,
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.ch_buffers {
            buf.reset();
        }
    }

    fn analyze(&mut self, frame: &[&[f32]]) {
        let num_channels = frame.len();
        debug_assert_eq!(num_channels, self.ch_buffers.len());

        for (channel, samples) in frame.iter().enumerate() {
            let samples_per_channel = samples.len();
            debug_assert!(samples_per_channel > 0);

            let mut sum_squares = 0.0_f32;
            let mut peak = 0.0_f32;
            for &sample in *samples {
                sum_squares += sample * sample;
                peak = peak.max(sample.abs());
            }
            self.ch_buffers[channel].push(Level {
                average: sum_squares / samples_per_channel as f32,
                max: peak,
            });
        }
    }

    fn estimate_clipped_level_step(
        &self,
        channel: usize,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32> {
        debug_assert!(channel < self.ch_buffers.len());
        if level <= min_mic_level {
            return None;
        }
        if self.predict_clipping_event(channel) {
            let new_level = (level - default_step).clamp(min_mic_level, max_mic_level);
            let step = level - new_level;
            if step > 0 {
                return Some(step);
            }
        }
        None
    }

    fn predict_clipping_event(&self, channel: usize) -> bool {
        let metrics =
            self.ch_buffers[channel].compute_partial_metrics(0, self.window_length as usize);
        let metrics = match metrics {
            Some(m) if float_s16_to_dbfs(m.max) > self.clipping_threshold => m,
            _ => return false,
        };

        let Some(reference_metrics) = self.ch_buffers[channel].compute_partial_metrics(
            self.reference_window_delay as usize,
            self.reference_window_length as usize,
        ) else {
            return false;
        };

        let crest_factor = compute_crest_factor(&metrics);
        let reference_crest_factor = compute_crest_factor(&reference_metrics);
        crest_factor < reference_crest_factor - self.crest_factor_margin
    }
}

/// Crest factor-based clipping peak prediction.
pub(crate) struct ClippingPeakPredictor {
    ch_buffers: Vec<ClippingPredictorLevelBuffer>,
    window_length: i32,
    reference_window_length: i32,
    reference_window_delay: i32,
    clipping_threshold: i32,
    adaptive_step_estimation: bool,
}

impl ClippingPeakPredictor {
    fn new(
        num_channels: usize,
        window_length: i32,
        reference_window_length: i32,
        reference_window_delay: i32,
        clipping_threshold: i32,
        adaptive_step_estimation: bool,
    ) -> Self {
        debug_assert!(num_channels > 0);
        debug_assert!(window_length > 0);
        debug_assert!(reference_window_length > 0);
        debug_assert!(reference_window_delay >= 0);
        debug_assert!(reference_window_length + reference_window_delay > window_length);

        let buffer_length = reference_window_delay + reference_window_length;
        let ch_buffers = (0..num_channels)
            .map(|_| ClippingPredictorLevelBuffer::new(buffer_length))
            .collect();

        Self {
            ch_buffers,
            window_length,
            reference_window_length,
            reference_window_delay,
            clipping_threshold,
            adaptive_step_estimation,
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.ch_buffers {
            buf.reset();
        }
    }

    fn analyze(&mut self, frame: &[&[f32]]) {
        let num_channels = frame.len();
        debug_assert_eq!(num_channels, self.ch_buffers.len());

        for (channel, samples) in frame.iter().enumerate() {
            let samples_per_channel = samples.len();
            debug_assert!(samples_per_channel > 0);

            let mut sum_squares = 0.0_f32;
            let mut peak = 0.0_f32;
            for &sample in *samples {
                sum_squares += sample * sample;
                peak = peak.max(sample.abs());
            }
            self.ch_buffers[channel].push(Level {
                average: sum_squares / samples_per_channel as f32,
                max: peak,
            });
        }
    }

    fn estimate_clipped_level_step(
        &self,
        channel: usize,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
    ) -> Option<i32> {
        debug_assert!(channel < self.ch_buffers.len());
        if level <= min_mic_level {
            return None;
        }
        let estimate_db = self.estimate_peak_value(channel)?;
        if estimate_db > self.clipping_threshold as f32 {
            let step = if !self.adaptive_step_estimation {
                default_step
            } else {
                let estimated_gain_change =
                    (-(estimate_db.ceil() as i32)).clamp(-MAX_GAIN_CHANGE, 0);
                let volume_update = compute_volume_update(
                    estimated_gain_change,
                    level,
                    min_mic_level,
                    max_mic_level,
                );
                (level - volume_update).max(default_step)
            };
            let new_level = (level - step).clamp(min_mic_level, max_mic_level);
            if level > new_level {
                return Some(level - new_level);
            }
        }
        None
    }

    fn estimate_peak_value(&self, channel: usize) -> Option<f32> {
        let reference_metrics = self.ch_buffers[channel].compute_partial_metrics(
            self.reference_window_delay as usize,
            self.reference_window_length as usize,
        )?;

        let metrics =
            self.ch_buffers[channel].compute_partial_metrics(0, self.window_length as usize);
        let metrics = match metrics {
            Some(m) if float_s16_to_dbfs(m.max) > self.clipping_threshold as f32 => m,
            _ => return None,
        };

        let reference_crest_factor = compute_crest_factor(&reference_metrics);
        let mean_squares = metrics.average;
        let projected_peak = reference_crest_factor + float_s16_to_dbfs(mean_squares.sqrt());
        Some(projected_peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: usize = 32000;
    const NUM_CHANNELS: usize = 1;
    const SAMPLES_PER_CHANNEL: usize = SAMPLE_RATE_HZ / 100;
    const MAX_MIC_LEVEL: i32 = 255;
    const MIN_MIC_LEVEL: i32 = 12;
    const DEFAULT_CLIPPED_LEVEL_STEP: i32 = 15;
    const MAX_SAMPLE_S16: f32 = i16::MAX as f32;

    /// Threshold in dB corresponding to a signal with an amplitude equal to 99%
    /// of the dynamic range - i.e., computed as `20*log10(0.99)`.
    const CLIPPING_THRESHOLD_DB: f32 = -0.087_296_11;

    fn call_analyze(num_calls: usize, frame: &[&[f32]], predictor: &mut ClippingPredictor) {
        for _ in 0..num_calls {
            predictor.analyze(frame);
        }
    }

    /// Creates and analyzes an audio frame with a non-zero (~4.15 dB) crest factor.
    fn analyze_non_zero_crest_factor_audio(
        num_calls: usize,
        num_channels: usize,
        peak_ratio: f32,
        predictor: &mut ClippingPredictor,
    ) {
        let mut audio_data = vec![0.0_f32; num_channels * SAMPLES_PER_CHANNEL];
        for channel in 0..num_channels {
            let offset = channel * SAMPLES_PER_CHANNEL;
            for sample in (0..SAMPLES_PER_CHANNEL).step_by(10) {
                audio_data[offset + sample] = 0.1 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 1] = 0.2 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 2] = 0.3 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 3] = 0.4 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 4] = 0.5 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 5] = 0.6 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 6] = 0.7 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 7] = 0.8 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 8] = 0.9 * peak_ratio * MAX_SAMPLE_S16;
                audio_data[offset + sample + 9] = 1.0 * peak_ratio * MAX_SAMPLE_S16;
            }
        }
        let channel_ptrs: Vec<&[f32]> = (0..num_channels)
            .map(|ch| {
                let offset = ch * SAMPLES_PER_CHANNEL;
                &audio_data[offset..offset + SAMPLES_PER_CHANNEL]
            })
            .collect();
        call_analyze(num_calls, &channel_ptrs, predictor);
    }

    /// Creates and analyzes an audio frame with a zero crest factor.
    fn analyze_zero_crest_factor_audio(
        num_calls: usize,
        num_channels: usize,
        peak_ratio: f32,
        predictor: &mut ClippingPredictor,
    ) {
        let mut audio_data = vec![0.0_f32; num_channels * SAMPLES_PER_CHANNEL];
        for channel in 0..num_channels {
            let offset = channel * SAMPLES_PER_CHANNEL;
            for sample in 0..SAMPLES_PER_CHANNEL {
                audio_data[offset + sample] = peak_ratio * MAX_SAMPLE_S16;
            }
        }
        let channel_ptrs: Vec<&[f32]> = (0..num_channels)
            .map(|ch| {
                let offset = ch * SAMPLES_PER_CHANNEL;
                &audio_data[offset..offset + SAMPLES_PER_CHANNEL]
            })
            .collect();
        call_analyze(num_calls, &channel_ptrs, predictor);
    }

    fn check_channel_estimates_with_value(
        num_channels: usize,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
        predictor: &ClippingPredictor,
        expected: i32,
    ) {
        for i in 0..num_channels {
            assert_eq!(
                predictor.estimate_clipped_level_step(
                    i,
                    level,
                    default_step,
                    min_mic_level,
                    max_mic_level
                ),
                Some(expected),
                "channel {i}"
            );
        }
    }

    fn check_channel_estimates_without_value(
        num_channels: usize,
        level: i32,
        default_step: i32,
        min_mic_level: i32,
        max_mic_level: i32,
        predictor: &ClippingPredictor,
    ) {
        for i in 0..num_channels {
            assert_eq!(
                predictor.estimate_clipped_level_step(
                    i,
                    level,
                    default_step,
                    min_mic_level,
                    max_mic_level
                ),
                None,
                "channel {i}"
            );
        }
    }

    #[test]
    fn no_predictor_created() {
        let result = create_clipping_predictor(
            NUM_CHANNELS,
            &ClippingPredictorConfig {
                enabled: false,
                ..Default::default()
            },
        );
        assert!(result.is_none());
    }

    #[test]
    fn clipping_event_prediction_created() {
        let result = create_clipping_predictor(
            NUM_CHANNELS,
            &ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::ClippingEvent,
                ..Default::default()
            },
        );
        assert!(result.is_some());
    }

    #[test]
    fn adaptive_step_clipping_peak_prediction_created() {
        let result = create_clipping_predictor(
            NUM_CHANNELS,
            &ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::AdaptiveStepClippingPeak,
                ..Default::default()
            },
        );
        assert!(result.is_some());
    }

    #[test]
    fn fixed_step_clipping_peak_prediction_created() {
        let result = create_clipping_predictor(
            NUM_CHANNELS,
            &ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::FixedStepClippingPeak,
                ..Default::default()
            },
        );
        assert!(result.is_some());
    }

    // Parametrized: ClippingPredictorParameterization
    // num_channels: [1, 5], window_length: [1, 5, 10], ref_window_length: [1, 5], ref_window_delay: [0, 1, 5]

    fn clipping_predictor_params() -> Vec<(usize, i32, i32, i32)> {
        let mut params = Vec::new();
        for &num_ch in &[1, 5] {
            for &wl in &[1, 5, 10] {
                for &rwl in &[1, 5] {
                    for &rwd in &[0, 1, 5] {
                        params.push((num_ch, wl, rwl, rwd));
                    }
                }
            }
        }
        params
    }

    #[test]
    fn check_clipping_event_predictor_estimate_after_crest_factor_drop() {
        for (num_ch, wl, rwl, rwd) in clipping_predictor_params() {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::ClippingEvent,
                window_length: wl,
                reference_window_length: rwl,
                reference_window_delay: rwd,
                clipping_threshold: -1.0,
                crest_factor_margin: 0.5,
            };
            if rwl + rwd <= wl {
                continue;
            }
            let mut predictor = create_clipping_predictor(num_ch, &config).unwrap();
            analyze_non_zero_crest_factor_audio(
                (rwl + rwd - wl) as usize,
                num_ch,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_zero_crest_factor_audio(wl as usize, num_ch, 0.99, &mut predictor);
            check_channel_estimates_with_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
                DEFAULT_CLIPPED_LEVEL_STEP,
            );
        }
    }

    #[test]
    fn check_clipping_event_predictor_no_estimate_after_constant_crest_factor() {
        for (num_ch, wl, rwl, rwd) in clipping_predictor_params() {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::ClippingEvent,
                window_length: wl,
                reference_window_length: rwl,
                reference_window_delay: rwd,
                clipping_threshold: -1.0,
                crest_factor_margin: 0.5,
            };
            if rwl + rwd <= wl {
                continue;
            }
            let mut predictor = create_clipping_predictor(num_ch, &config).unwrap();
            analyze_non_zero_crest_factor_audio(
                (rwl + rwd - wl) as usize,
                num_ch,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_non_zero_crest_factor_audio(wl as usize, num_ch, 0.99, &mut predictor);
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
        }
    }

    #[test]
    fn check_clipping_peak_predictor_estimate_after_high_crest_factor() {
        for (num_ch, wl, rwl, rwd) in clipping_predictor_params() {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::AdaptiveStepClippingPeak,
                window_length: wl,
                reference_window_length: rwl,
                reference_window_delay: rwd,
                clipping_threshold: -1.0,
                crest_factor_margin: 0.5,
            };
            if rwl + rwd <= wl {
                continue;
            }
            let mut predictor = create_clipping_predictor(num_ch, &config).unwrap();
            analyze_non_zero_crest_factor_audio(
                (rwl + rwd - wl) as usize,
                num_ch,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_non_zero_crest_factor_audio(wl as usize, num_ch, 0.99, &mut predictor);
            check_channel_estimates_with_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
                DEFAULT_CLIPPED_LEVEL_STEP,
            );
        }
    }

    #[test]
    fn check_clipping_peak_predictor_no_estimate_after_low_crest_factor() {
        for (num_ch, wl, rwl, rwd) in clipping_predictor_params() {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode: ClippingPredictorMode::AdaptiveStepClippingPeak,
                window_length: wl,
                reference_window_length: rwl,
                reference_window_delay: rwd,
                clipping_threshold: -1.0,
                crest_factor_margin: 0.5,
            };
            if rwl + rwd <= wl {
                continue;
            }
            let mut predictor = create_clipping_predictor(num_ch, &config).unwrap();
            analyze_zero_crest_factor_audio(
                (rwl + rwd - wl) as usize,
                num_ch,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_non_zero_crest_factor_audio(wl as usize, num_ch, 0.99, &mut predictor);
            check_channel_estimates_without_value(
                num_ch,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
        }
    }

    // ClippingEventPredictorParameterization:
    // clipping_threshold: [-1.0, 0.0], crest_factor_margin: [3.0, 4.16]
    #[test]
    fn check_event_estimate_after_crest_factor_drop_parametrized() {
        for &clipping_threshold in &[-1.0_f32, 0.0] {
            for &crest_factor_margin in &[3.0_f32, 4.16] {
                let config = ClippingPredictorConfig {
                    enabled: true,
                    mode: ClippingPredictorMode::ClippingEvent,
                    window_length: 5,
                    reference_window_length: 5,
                    reference_window_delay: 5,
                    clipping_threshold,
                    crest_factor_margin,
                };
                let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
                analyze_non_zero_crest_factor_audio(
                    config.reference_window_length as usize,
                    NUM_CHANNELS,
                    0.99,
                    &mut predictor,
                );
                check_channel_estimates_without_value(
                    NUM_CHANNELS,
                    255,
                    DEFAULT_CLIPPED_LEVEL_STEP,
                    MIN_MIC_LEVEL,
                    MAX_MIC_LEVEL,
                    &predictor,
                );
                analyze_zero_crest_factor_audio(
                    config.window_length as usize,
                    NUM_CHANNELS,
                    0.99,
                    &mut predictor,
                );
                if clipping_threshold < CLIPPING_THRESHOLD_DB && crest_factor_margin < 4.15 {
                    check_channel_estimates_with_value(
                        NUM_CHANNELS,
                        255,
                        DEFAULT_CLIPPED_LEVEL_STEP,
                        MIN_MIC_LEVEL,
                        MAX_MIC_LEVEL,
                        &predictor,
                        DEFAULT_CLIPPED_LEVEL_STEP,
                    );
                } else {
                    check_channel_estimates_without_value(
                        NUM_CHANNELS,
                        255,
                        DEFAULT_CLIPPED_LEVEL_STEP,
                        MIN_MIC_LEVEL,
                        MAX_MIC_LEVEL,
                        &predictor,
                    );
                }
            }
        }
    }

    // ClippingPredictorModeParameterization:
    // modes: [AdaptiveStep, FixedStep]
    #[test]
    fn check_estimate_after_high_crest_factor_with_no_clipping_margin() {
        for &mode in &[
            ClippingPredictorMode::AdaptiveStepClippingPeak,
            ClippingPredictorMode::FixedStepClippingPeak,
        ] {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode,
                window_length: 5,
                reference_window_length: 5,
                reference_window_delay: 5,
                clipping_threshold: 0.0,
                crest_factor_margin: 3.0,
            };
            let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
            analyze_non_zero_crest_factor_audio(
                config.reference_window_length as usize,
                NUM_CHANNELS,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_zero_crest_factor_audio(
                config.window_length as usize,
                NUM_CHANNELS,
                0.99,
                &mut predictor,
            );
            // Since the clipping threshold is set to 0 dBFS, expect no estimate.
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
        }
    }

    #[test]
    fn check_estimate_after_high_crest_factor_with_clipping_margin() {
        for &mode in &[
            ClippingPredictorMode::AdaptiveStepClippingPeak,
            ClippingPredictorMode::FixedStepClippingPeak,
        ] {
            let config = ClippingPredictorConfig {
                enabled: true,
                mode,
                window_length: 5,
                reference_window_length: 5,
                reference_window_delay: 5,
                clipping_threshold: -1.0,
                crest_factor_margin: 3.0,
            };
            let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
            analyze_non_zero_crest_factor_audio(
                config.reference_window_length as usize,
                NUM_CHANNELS,
                0.99,
                &mut predictor,
            );
            check_channel_estimates_without_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
            );
            analyze_zero_crest_factor_audio(
                config.window_length as usize,
                NUM_CHANNELS,
                0.99,
                &mut predictor,
            );
            let expected_step = if mode == ClippingPredictorMode::AdaptiveStepClippingPeak {
                17
            } else {
                DEFAULT_CLIPPED_LEVEL_STEP
            };
            check_channel_estimates_with_value(
                NUM_CHANNELS,
                255,
                DEFAULT_CLIPPED_LEVEL_STEP,
                MIN_MIC_LEVEL,
                MAX_MIC_LEVEL,
                &predictor,
                expected_step,
            );
        }
    }

    #[test]
    fn check_event_estimate_after_reset() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::ClippingEvent,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
        predictor.reset();
        analyze_zero_crest_factor_audio(
            config.window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
    }

    #[test]
    fn check_peak_no_estimate_after_reset() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::AdaptiveStepClippingPeak,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
        predictor.reset();
        analyze_zero_crest_factor_audio(
            config.window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
    }

    #[test]
    fn check_adaptive_step_estimate() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::AdaptiveStepClippingPeak,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
        analyze_zero_crest_factor_audio(
            config.window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_with_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
            17,
        );
    }

    #[test]
    fn check_fixed_step_estimate() {
        let config = ClippingPredictorConfig {
            enabled: true,
            mode: ClippingPredictorMode::FixedStepClippingPeak,
            window_length: 5,
            reference_window_length: 5,
            reference_window_delay: 5,
            clipping_threshold: -1.0,
            crest_factor_margin: 3.0,
        };
        let mut predictor = create_clipping_predictor(NUM_CHANNELS, &config).unwrap();
        analyze_non_zero_crest_factor_audio(
            config.reference_window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_without_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
        );
        analyze_zero_crest_factor_audio(
            config.window_length as usize,
            NUM_CHANNELS,
            0.99,
            &mut predictor,
        );
        check_channel_estimates_with_value(
            NUM_CHANNELS,
            255,
            DEFAULT_CLIPPED_LEVEL_STEP,
            MIN_MIC_LEVEL,
            MAX_MIC_LEVEL,
            &predictor,
            DEFAULT_CLIPPED_LEVEL_STEP,
        );
    }
}
