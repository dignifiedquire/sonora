//! Audio processing statistics.
//!
//! Ported from `AudioProcessingStats` in
//! `api/audio/audio_processing_statistics.h`.

/// Statistics from the audio processing pipeline.
#[derive(Debug, Clone, Default)]
pub struct AudioProcessingStats {
    /// ERL = 10 log10(P_far / P_echo).
    pub echo_return_loss: Option<f64>,
    /// ERLE = 10 log10(P_echo / P_out).
    pub echo_return_loss_enhancement: Option<f64>,
    /// Fraction of time that the AEC linear filter is divergent (1-second
    /// non-overlapped aggregation window).
    pub divergent_filter_fraction: Option<f64>,
    /// Delay median in milliseconds.
    pub delay_median_ms: Option<i32>,
    /// Delay standard deviation in milliseconds.
    pub delay_standard_deviation_ms: Option<i32>,
    /// Residual echo detector likelihood.
    pub residual_echo_likelihood: Option<f64>,
    /// Maximum residual echo likelihood from the last time period.
    pub residual_echo_likelihood_recent_max: Option<f64>,
    /// Instantaneous delay estimate from the AEC in milliseconds.
    pub delay_ms: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stats_has_no_values() {
        let stats = AudioProcessingStats::default();
        assert!(stats.echo_return_loss.is_none());
        assert!(stats.echo_return_loss_enhancement.is_none());
        assert!(stats.divergent_filter_fraction.is_none());
        assert!(stats.delay_median_ms.is_none());
        assert!(stats.delay_standard_deviation_ms.is_none());
        assert!(stats.residual_echo_likelihood.is_none());
        assert!(stats.residual_echo_likelihood_recent_max.is_none());
        assert!(stats.delay_ms.is_none());
    }
}
