//! Top-level noise suppressor pipeline.
//!
//! Combines FFT analysis, noise estimation, speech probability estimation,
//! Wiener filtering, and overlap-add synthesis into a complete noise
//! suppression pipeline.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/noise_suppressor.cc`

use crate::config::{
    FFT_SIZE, FFT_SIZE_BY_2_PLUS_1, NS_FRAME_SIZE, NsConfig, OVERLAP_SIZE, SuppressionLevel,
};
use crate::fast_math::sqrt_fast_approximation;
use crate::noise_estimator::NoiseEstimator;
use crate::ns_fft::NsFft;
use crate::speech_probability_estimator::SpeechProbabilityEstimator;
use crate::suppression_params::SuppressionParams;
use crate::wiener_filter::WienerFilter;

/// Hybrid Hanning + flat window for the filterbank (first half, 96 samples).
///
/// Applied to both ends of the 256-sample extended frame. The middle 64
/// samples (indices 96..160) are left unwindowed (gain = 1.0).
#[allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "match C++ source table exactly"
)]
const BLOCKS_160W256_FIRST_HALF: [f32; 96] = [
    0.00000000, 0.01636173, 0.03271908, 0.04906767, 0.06540313, 0.08172107, 0.09801714, 0.11428696,
    0.13052619, 0.14673047, 0.16289547, 0.17901686, 0.19509032, 0.21111155, 0.22707626, 0.24298018,
    0.25881905, 0.27458862, 0.29028468, 0.30590302, 0.32143947, 0.33688985, 0.35225005, 0.36751594,
    0.38268343, 0.39774847, 0.41270703, 0.42755509, 0.44228869, 0.45690388, 0.47139674, 0.48576339,
    0.50000000, 0.51410274, 0.52806785, 0.54189158, 0.55557023, 0.56910015, 0.58247770, 0.59569930,
    0.60876143, 0.62166057, 0.63439328, 0.64695615, 0.65934582, 0.67155895, 0.68359230, 0.69544264,
    0.70710678, 0.71858162, 0.72986407, 0.74095113, 0.75183981, 0.76252720, 0.77301045, 0.78328675,
    0.79335334, 0.80320753, 0.81284668, 0.82226822, 0.83146961, 0.84044840, 0.84920218, 0.85772861,
    0.86602540, 0.87409034, 0.88192126, 0.88951608, 0.89687274, 0.90398929, 0.91086382, 0.91749450,
    0.92387953, 0.93001722, 0.93590593, 0.94154407, 0.94693013, 0.95206268, 0.95694034, 0.96156180,
    0.96592583, 0.97003125, 0.97387698, 0.97746197, 0.98078528, 0.98384601, 0.98664333, 0.98917651,
    0.99144486, 0.99344778, 0.99518473, 0.99665524, 0.99785892, 0.99879546, 0.99946459, 0.99986614,
];

/// Apply the analysis/synthesis window to an extended frame.
fn apply_filterbank_window(x: &mut [f32; FFT_SIZE]) {
    for (x_i, &w) in x[..OVERLAP_SIZE]
        .iter_mut()
        .zip(BLOCKS_160W256_FIRST_HALF.iter())
    {
        *x_i *= w;
    }
    // x[96..160] are left as-is (window = 1.0).
    for (x_i, &w) in x[NS_FRAME_SIZE + 1..]
        .iter_mut()
        .zip(BLOCKS_160W256_FIRST_HALF.iter().rev())
    {
        *x_i *= w;
    }
}

/// Form an extended frame by prepending old data.
fn form_extended_frame(
    frame: &[f32; NS_FRAME_SIZE],
    old_data: &mut [f32; FFT_SIZE - NS_FRAME_SIZE],
    extended_frame: &mut [f32; FFT_SIZE],
) {
    extended_frame[..old_data.len()].copy_from_slice(old_data);
    extended_frame[old_data.len()..].copy_from_slice(frame);
    old_data.copy_from_slice(&extended_frame[NS_FRAME_SIZE..]);
}

/// Overlap-and-add to produce an output frame.
fn overlap_and_add(
    extended_frame: &[f32; FFT_SIZE],
    overlap_memory: &mut [f32; OVERLAP_SIZE],
    output_frame: &mut [f32; NS_FRAME_SIZE],
) {
    for i in 0..OVERLAP_SIZE {
        output_frame[i] = overlap_memory[i] + extended_frame[i];
    }
    output_frame[OVERLAP_SIZE..].copy_from_slice(&extended_frame[OVERLAP_SIZE..NS_FRAME_SIZE]);
    overlap_memory.copy_from_slice(&extended_frame[NS_FRAME_SIZE..]);
}

/// Compute magnitude spectrum from FFT output.
fn compute_magnitude_spectrum(
    real: &[f32; FFT_SIZE],
    imag: &[f32; FFT_SIZE],
    signal_spectrum: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
) {
    signal_spectrum[0] = real[0].abs() + 1.0;
    signal_spectrum[FFT_SIZE_BY_2_PLUS_1 - 1] = real[FFT_SIZE_BY_2_PLUS_1 - 1].abs() + 1.0;

    for i in 1..FFT_SIZE_BY_2_PLUS_1 - 1 {
        signal_spectrum[i] = sqrt_fast_approximation(real[i] * real[i] + imag[i] * imag[i]) + 1.0;
    }
}

/// Compute prior and post SNR.
fn compute_snr(
    filter: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prev_signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    signal_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prev_noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    noise_spectrum: &[f32; FFT_SIZE_BY_2_PLUS_1],
    prior_snr: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
    post_snr: &mut [f32; FFT_SIZE_BY_2_PLUS_1],
) {
    for i in 0..FFT_SIZE_BY_2_PLUS_1 {
        // Previous estimate: based on previous frame with gain filter.
        let prev_estimate = prev_signal_spectrum[i] / (prev_noise_spectrum[i] + 0.0001) * filter[i];
        // Post SNR.
        if signal_spectrum[i] > noise_spectrum[i] {
            post_snr[i] = signal_spectrum[i] / (noise_spectrum[i] + 0.0001) - 1.0;
        } else {
            post_snr[i] = 0.0;
        }
        // Directed decision estimate of the prior SNR.
        prior_snr[i] = 0.98 * prev_estimate + (1.0 - 0.98) * post_snr[i];
    }
}

/// Compute energy of an extended frame.
fn compute_energy(x: &[f32; FFT_SIZE]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}

/// Per-channel processing state.
#[derive(Debug)]
struct ChannelState {
    speech_probability_estimator: SpeechProbabilityEstimator,
    wiener_filter: WienerFilter,
    noise_estimator: NoiseEstimator,
    prev_analysis_signal_spectrum: [f32; FFT_SIZE_BY_2_PLUS_1],
    analyze_analysis_memory: [f32; FFT_SIZE - NS_FRAME_SIZE],
    process_analysis_memory: [f32; OVERLAP_SIZE],
    process_synthesis_memory: [f32; OVERLAP_SIZE],
}

impl ChannelState {
    fn new(suppression_params: &'static SuppressionParams) -> Self {
        Self {
            speech_probability_estimator: SpeechProbabilityEstimator::default(),
            wiener_filter: WienerFilter::new(suppression_params),
            noise_estimator: NoiseEstimator::new(suppression_params),
            prev_analysis_signal_spectrum: [1.0; FFT_SIZE_BY_2_PLUS_1],
            analyze_analysis_memory: [0.0; FFT_SIZE - NS_FRAME_SIZE],
            process_analysis_memory: [0.0; OVERLAP_SIZE],
            process_synthesis_memory: [0.0; OVERLAP_SIZE],
        }
    }
}

/// Single-channel noise suppressor.
///
/// Processes 10ms frames (160 samples at 16kHz) using overlap-add
/// with a 256-point FFT. Call [`analyze`] before [`process`] for each frame.
///
/// # Example
///
/// ```
/// use webrtc_ns::config::NsConfig;
/// use webrtc_ns::noise_suppressor::NoiseSuppressor;
///
/// let mut ns = NoiseSuppressor::new(NsConfig::default());
/// let mut frame = [0.0f32; 160];
/// // ... fill frame with audio ...
/// ns.analyze(&frame);
/// ns.process(&mut frame);
/// ```
#[derive(Debug)]
pub struct NoiseSuppressor {
    num_analyzed_frames: i32,
    fft: NsFft,
    channel: ChannelState,
}

impl NoiseSuppressor {
    /// Create a new noise suppressor with the given configuration.
    pub fn new(config: NsConfig) -> Self {
        let suppression_params = SuppressionParams::for_level(config.target_level);
        Self {
            num_analyzed_frames: -1,
            fft: NsFft::default(),
            channel: ChannelState::new(suppression_params),
        }
    }

    /// Create a noise suppressor with the given suppression level.
    pub fn with_level(level: SuppressionLevel) -> Self {
        Self::new(NsConfig {
            target_level: level,
        })
    }

    /// Analyze a frame for noise estimation.
    ///
    /// This should be called before any echo cancellation or other processing,
    /// so the noise estimator sees the raw signal (not comfort noise).
    /// `frame` must have exactly [`NS_FRAME_SIZE`] (160) samples.
    pub fn analyze(&mut self, frame: &[f32; NS_FRAME_SIZE]) {
        let ch = &mut self.channel;

        // Prepare the noise estimator.
        ch.noise_estimator.prepare_analysis();

        // Check for zero frame.
        let energy = {
            let mut e = 0.0f32;
            for &v in ch.analyze_analysis_memory.iter() {
                e += v * v;
            }
            for &v in frame.iter() {
                e += v * v;
            }
            e
        };
        if energy == 0.0 {
            return;
        }

        // Increment analysis counter.
        self.num_analyzed_frames += 1;
        if self.num_analyzed_frames < 0 {
            self.num_analyzed_frames = 0;
        }

        // Form extended frame and apply analysis window.
        let mut extended_frame = [0.0f32; FFT_SIZE];
        form_extended_frame(frame, &mut ch.analyze_analysis_memory, &mut extended_frame);
        apply_filterbank_window(&mut extended_frame);

        // Compute FFT and magnitude spectrum.
        let mut real = [0.0f32; FFT_SIZE];
        let mut imag = [0.0f32; FFT_SIZE];
        self.fft.fft(&mut extended_frame, &mut real, &mut imag);

        let mut signal_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        compute_magnitude_spectrum(&real, &imag, &mut signal_spectrum);

        // Compute energies.
        let mut signal_energy = 0.0f32;
        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            signal_energy += real[i] * real[i] + imag[i] * imag[i];
        }
        signal_energy /= FFT_SIZE_BY_2_PLUS_1 as f32;

        let signal_spectral_sum: f32 = signal_spectrum.iter().sum();

        // Estimate noise spectra.
        ch.noise_estimator.pre_update(
            self.num_analyzed_frames,
            &signal_spectrum,
            signal_spectral_sum,
        );

        // Compute SNR.
        let mut post_snr = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        let mut prior_snr = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        compute_snr(
            ch.wiener_filter.filter(),
            &ch.prev_analysis_signal_spectrum,
            &signal_spectrum,
            ch.noise_estimator.prev_noise_spectrum(),
            ch.noise_estimator.noise_spectrum(),
            &mut prior_snr,
            &mut post_snr,
        );

        // Update speech probability.
        ch.speech_probability_estimator.update(
            self.num_analyzed_frames,
            &prior_snr,
            &post_snr,
            ch.noise_estimator.conservative_noise_spectrum(),
            &signal_spectrum,
            signal_spectral_sum,
            signal_energy,
        );

        // Post-update noise estimator with speech probability.
        ch.noise_estimator.post_update(
            ch.speech_probability_estimator.probability(),
            &signal_spectrum,
        );

        // Store magnitude spectrum for the process step.
        ch.prev_analysis_signal_spectrum = signal_spectrum;
    }

    /// Apply noise suppression to the frame.
    ///
    /// `frame` must have exactly [`NS_FRAME_SIZE`] (160) samples.
    /// The frame is modified in-place with the suppressed output.
    pub fn process(&mut self, frame: &mut [f32; NS_FRAME_SIZE]) {
        let ch = &mut self.channel;

        // Form extended frame and apply analysis window.
        let mut extended_frame = [0.0f32; FFT_SIZE];
        form_extended_frame(frame, &mut ch.process_analysis_memory, &mut extended_frame);
        apply_filterbank_window(&mut extended_frame);

        let energy_before_filtering = compute_energy(&extended_frame);

        // FFT and magnitude spectrum.
        let mut real = [0.0f32; FFT_SIZE];
        let mut imag = [0.0f32; FFT_SIZE];
        self.fft.fft(&mut extended_frame, &mut real, &mut imag);

        let mut signal_spectrum = [0.0f32; FFT_SIZE_BY_2_PLUS_1];
        compute_magnitude_spectrum(&real, &imag, &mut signal_spectrum);

        // Update the Wiener filter.
        ch.wiener_filter.update(
            self.num_analyzed_frames,
            ch.noise_estimator.noise_spectrum(),
            ch.noise_estimator.prev_noise_spectrum(),
            ch.noise_estimator.parametric_noise_spectrum(),
            &signal_spectrum,
        );

        // Apply the filter to the frequency domain.
        let filter = ch.wiener_filter.filter();
        for i in 0..FFT_SIZE_BY_2_PLUS_1 {
            real[i] *= filter[i];
            imag[i] *= filter[i];
        }

        // Inverse FFT.
        self.fft.ifft(&real, &imag, &mut extended_frame);

        let energy_after_filtering = compute_energy(&extended_frame);

        // Apply synthesis window.
        apply_filterbank_window(&mut extended_frame);

        // Compute overall gain adjustment.
        let gain_adjustment = ch.wiener_filter.compute_overall_scaling_factor(
            self.num_analyzed_frames,
            ch.speech_probability_estimator.prior_probability(),
            energy_before_filtering,
            energy_after_filtering,
        );

        // Apply gain adjustment.
        for v in extended_frame.iter_mut() {
            *v *= gain_adjustment;
        }

        // Overlap-and-add to produce the output frame.
        overlap_and_add(&extended_frame, &mut ch.process_synthesis_memory, frame);

        // Clamp output to valid range.
        for v in frame.iter_mut() {
            *v = v.clamp(-32768.0, 32767.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_valid_instance() {
        let ns = NoiseSuppressor::new(NsConfig::default());
        assert_eq!(ns.num_analyzed_frames, -1);
    }

    #[test]
    fn with_level_creates_valid_instance() {
        let ns = NoiseSuppressor::with_level(SuppressionLevel::K18dB);
        assert_eq!(ns.num_analyzed_frames, -1);
    }

    #[test]
    fn analyze_zero_frame_skips_processing() {
        let mut ns = NoiseSuppressor::new(NsConfig::default());
        let frame = [0.0f32; NS_FRAME_SIZE];
        ns.analyze(&frame);
        // Frame counter should not increment for zero frames.
        assert_eq!(ns.num_analyzed_frames, -1);
    }

    #[test]
    fn analyze_nonzero_frame_increments_counter() {
        let mut ns = NoiseSuppressor::new(NsConfig::default());
        let mut frame = [0.0f32; NS_FRAME_SIZE];
        frame[0] = 1.0;
        ns.analyze(&frame);
        assert_eq!(ns.num_analyzed_frames, 0);
        ns.analyze(&frame);
        assert_eq!(ns.num_analyzed_frames, 1);
    }

    #[test]
    fn process_zero_frame_produces_near_zero() {
        let mut ns = NoiseSuppressor::new(NsConfig::default());
        let zero_frame = [0.0f32; NS_FRAME_SIZE];
        ns.analyze(&zero_frame);

        let mut output = [0.0f32; NS_FRAME_SIZE];
        ns.process(&mut output);

        let energy: f32 = output.iter().map(|&v| v * v).sum();
        assert!(
            energy < 1e-6,
            "zero input should produce near-zero output, got energy {energy}"
        );
    }

    #[test]
    fn process_produces_bounded_output() {
        let mut ns = NoiseSuppressor::new(NsConfig::default());

        // Feed several frames of noise.
        for _ in 0..100 {
            let mut frame = [0.0f32; NS_FRAME_SIZE];
            for (i, v) in frame.iter_mut().enumerate() {
                *v = (i as f32 * 0.1).sin() * 10000.0;
            }
            ns.analyze(&frame);
            ns.process(&mut frame);

            // All output samples should be in [-32768, 32767].
            for &v in &frame {
                assert!(
                    (-32768.0..=32767.0).contains(&v),
                    "output {v} out of bounds"
                );
            }
        }
    }

    #[test]
    fn noise_is_suppressed() {
        let mut ns = NoiseSuppressor::with_level(SuppressionLevel::K21dB);

        // Feed constant-level "noise" for many frames to let estimator converge.
        let mut total_input_energy = 0.0f32;
        let mut total_output_energy = 0.0f32;

        for i in 0..500 {
            let mut frame = [0.0f32; NS_FRAME_SIZE];
            // Pseudo-random noise pattern.
            for (j, v) in frame.iter_mut().enumerate() {
                let t = (i * NS_FRAME_SIZE + j) as f32;
                *v =
                    (t * 0.073).sin() * 100.0 + (t * 0.137).sin() * 50.0 + (t * 0.291).sin() * 25.0;
            }

            let input_energy: f32 = frame.iter().map(|&v| v * v).sum();
            total_input_energy += input_energy;

            ns.analyze(&frame);
            ns.process(&mut frame);

            let output_energy: f32 = frame.iter().map(|&v| v * v).sum();
            total_output_energy += output_energy;
        }

        // After convergence, output energy should be significantly less than input.
        let ratio = total_output_energy / total_input_energy;
        assert!(
            ratio < 0.8,
            "noise should be suppressed: output/input energy ratio = {ratio}"
        );
    }

    #[test]
    fn filterbank_window_shape() {
        let mut x = [1.0f32; FFT_SIZE];
        apply_filterbank_window(&mut x);

        // First 96 samples are the rising taper: x[i] == w[i].
        for i in 0..OVERLAP_SIZE {
            assert_eq!(x[i], BLOCKS_160W256_FIRST_HALF[i], "rising taper at {i}");
        }

        // Middle samples (96..161) should be 1.0 (unwindowed).
        for (i, &v) in x
            .iter()
            .enumerate()
            .take(NS_FRAME_SIZE + 1)
            .skip(OVERLAP_SIZE)
        {
            assert_eq!(v, 1.0, "middle sample {i} should be 1.0");
        }

        // Last 95 samples (161..256) are the falling taper: x[i] == w[k], k=95..1.
        // C++ loop: for (i = 161, k = 95; i < 256; ++i, --k)
        for (&x_i, &w) in x[NS_FRAME_SIZE + 1..]
            .iter()
            .zip(BLOCKS_160W256_FIRST_HALF.iter().rev())
        {
            assert!(
                (x_i - w).abs() < 1e-6,
                "falling taper mismatch: got {x_i}, expected {w}"
            );
        }
    }

    #[test]
    fn overlap_add_roundtrip() {
        let mut memory = [0.0f32; OVERLAP_SIZE];
        let extended = [1.0f32; FFT_SIZE];
        let mut output = [0.0f32; NS_FRAME_SIZE];

        overlap_and_add(&extended, &mut memory, &mut output);

        // First overlap region: 0 + 1 = 1.
        for &v in &output[..OVERLAP_SIZE] {
            assert_eq!(v, 1.0);
        }
        // Remaining: direct copy.
        for &v in &output[OVERLAP_SIZE..] {
            assert_eq!(v, 1.0);
        }
        // Memory should hold the tail.
        for &v in &memory {
            assert_eq!(v, 1.0);
        }
    }
}
