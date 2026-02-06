//! Voice Activity Detector wrapper with resampling and periodic reset.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/vad_wrapper.cc`.

// Used by adaptive_digital_gain_controller (Step 8).
#![allow(dead_code, reason = "consumed by later AGC2 modules")]

use crate::common::{FRAME_DURATION_MS, VAD_RESET_PERIOD_MS};
use crate::rnn_vad::common::{FRAME_SIZE_10MS_24K_HZ, FeatureVector, SAMPLE_RATE_24K_HZ};
use crate::rnn_vad::features_extraction::FeaturesExtractor;
use crate::rnn_vad::rnn::RnnVad;
use webrtc_common_audio::push_resampler::PushResampler;
use webrtc_simd::SimdBackend;

const NUM_FRAMES_PER_SECOND: i32 = 100;

/// Single-channel VAD interface.
///
/// The default implementation uses the RNN VAD. A mock can be injected for
/// testing.
pub(crate) trait MonoVad {
    /// Returns the sample rate (Hz) required for input frames.
    fn sample_rate_hz(&self) -> i32;
    /// Resets the internal state.
    fn reset(&mut self);
    /// Analyzes an audio frame and returns the speech probability.
    fn analyze(&mut self, frame: &[f32]) -> f32;
}

/// Default RNN-based mono VAD implementation.
struct RnnMonoVad {
    features_extractor: FeaturesExtractor,
    rnn_vad: RnnVad,
    feature_vector: FeatureVector,
}

impl RnnMonoVad {
    fn new(backend: SimdBackend) -> Self {
        Self {
            features_extractor: FeaturesExtractor::new(backend),
            rnn_vad: RnnVad::new(backend),
            feature_vector: bytemuck::Zeroable::zeroed(),
        }
    }
}

impl MonoVad for RnnMonoVad {
    fn sample_rate_hz(&self) -> i32 {
        SAMPLE_RATE_24K_HZ
    }

    fn reset(&mut self) {
        self.rnn_vad.reset();
    }

    fn analyze(&mut self, frame: &[f32]) -> f32 {
        debug_assert_eq!(frame.len(), FRAME_SIZE_10MS_24K_HZ);
        let is_silence = self
            .features_extractor
            .check_silence_compute_features(frame, &mut self.feature_vector);
        self.rnn_vad
            .compute_vad_probability(&self.feature_vector, is_silence)
    }
}

/// Wraps a single-channel VAD with resampling and periodic reset.
///
/// Analyzes the first channel of input audio frames, resampling to the
/// VAD's expected sample rate if necessary.
#[derive(derive_more::Debug)]
pub(crate) struct VoiceActivityDetectorWrapper {
    vad_reset_period_frames: i32,
    frame_size: usize,
    time_to_vad_reset: i32,
    #[debug(skip)]
    vad: Box<dyn MonoVad>,
    #[debug(skip)]
    resampled_buffer: Vec<f32>,
    #[debug(skip)]
    resampler: PushResampler<f32>,
}

impl VoiceActivityDetectorWrapper {
    /// Creates a new wrapper using the default RNN VAD and default reset
    /// period.
    pub(crate) fn new(backend: SimdBackend, sample_rate_hz: i32) -> Self {
        Self::with_reset_period(VAD_RESET_PERIOD_MS, backend, sample_rate_hz)
    }

    /// Creates a new wrapper using the default RNN VAD with a custom reset
    /// period.
    pub(crate) fn with_reset_period(
        vad_reset_period_ms: i32,
        backend: SimdBackend,
        sample_rate_hz: i32,
    ) -> Self {
        Self::with_vad(
            vad_reset_period_ms,
            Box::new(RnnMonoVad::new(backend)),
            sample_rate_hz,
        )
    }

    /// Creates a new wrapper with a custom VAD implementation.
    pub(crate) fn with_vad(
        vad_reset_period_ms: i32,
        mut vad: Box<dyn MonoVad>,
        sample_rate_hz: i32,
    ) -> Self {
        let vad_reset_period_frames = vad_reset_period_ms / FRAME_DURATION_MS;
        debug_assert!(vad_reset_period_frames > 1);
        let frame_size = (sample_rate_hz / NUM_FRAMES_PER_SECOND) as usize;
        let resampled_size = (vad.sample_rate_hz() / NUM_FRAMES_PER_SECOND) as usize;

        let resampler = PushResampler::new(frame_size, resampled_size, 1);

        vad.reset();

        Self {
            vad_reset_period_frames,
            frame_size,
            time_to_vad_reset: vad_reset_period_frames,
            vad,
            resampled_buffer: vec![0.0; resampled_size],
            resampler,
        }
    }

    /// Analyzes the first channel of `frame` and returns the speech
    /// probability.
    ///
    /// `frame` must contain at least `frame_size` samples (the first channel
    /// of a 10 ms frame at the configured sample rate).
    pub(crate) fn analyze(&mut self, frame: &[f32]) -> f32 {
        // Periodically reset the VAD.
        self.time_to_vad_reset -= 1;
        if self.time_to_vad_reset <= 0 {
            self.vad.reset();
            self.time_to_vad_reset = self.vad_reset_period_frames;
        }

        // Resample the first channel.
        debug_assert!(frame.len() >= self.frame_size);
        let src = &frame[..self.frame_size];
        self.resampler
            .resample_mono(src, &mut self.resampled_buffer);

        self.vad.analyze(&self.resampled_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    const SAMPLE_RATE_8K_HZ: i32 = 8000;
    const NO_VAD_PERIODIC_RESET: i32 = FRAME_DURATION_MS * (i32::MAX / FRAME_DURATION_MS);

    /// Mock VAD that records calls and returns pre-configured probabilities.
    struct MockVad {
        sample_rate: i32,
        probabilities: Vec<f32>,
        prob_index: usize,
        state: Rc<RefCell<MockVadState>>,
    }

    #[derive(Default)]
    struct MockVadState {
        reset_count: usize,
        analyze_frames: Vec<usize>,
    }

    impl MockVad {
        fn new(
            sample_rate: i32,
            probabilities: Vec<f32>,
            state: Rc<RefCell<MockVadState>>,
        ) -> Self {
            Self {
                sample_rate,
                probabilities,
                prob_index: 0,
                state,
            }
        }
    }

    impl MonoVad for MockVad {
        fn sample_rate_hz(&self) -> i32 {
            self.sample_rate
        }

        fn reset(&mut self) {
            self.state.borrow_mut().reset_count += 1;
        }

        fn analyze(&mut self, frame: &[f32]) -> f32 {
            self.state.borrow_mut().analyze_frames.push(frame.len());
            let p = self.probabilities[self.prob_index % self.probabilities.len()];
            self.prob_index += 1;
            p
        }
    }

    fn create_mock_vad_wrapper(
        vad_reset_period_ms: i32,
        sample_rate_hz: i32,
        probabilities: Vec<f32>,
        state: Rc<RefCell<MockVadState>>,
    ) -> VoiceActivityDetectorWrapper {
        let vad = MockVad::new(sample_rate_hz, probabilities, state);
        VoiceActivityDetectorWrapper::with_vad(vad_reset_period_ms, Box::new(vad), sample_rate_hz)
    }

    fn make_frame(sample_rate_hz: i32) -> Vec<f32> {
        vec![0.0_f32; (sample_rate_hz / NUM_FRAMES_PER_SECOND) as usize]
    }

    #[test]
    fn check_speech_probabilities() {
        let probabilities = vec![
            0.709, 0.484, 0.882, 0.167, 0.44, 0.525, 0.858, 0.314, 0.653, 0.965, 0.413, 0.0,
        ];
        let state = Rc::new(RefCell::new(MockVadState::default()));
        let mut wrapper = create_mock_vad_wrapper(
            NO_VAD_PERIODIC_RESET,
            SAMPLE_RATE_8K_HZ,
            probabilities.clone(),
            Rc::clone(&state),
        );
        let frame = make_frame(SAMPLE_RATE_8K_HZ);
        for (i, &expected) in probabilities.iter().enumerate() {
            let actual = wrapper.analyze(&frame);
            assert_eq!(
                expected, actual,
                "mismatch at frame {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn vad_no_periodic_reset() {
        let num_frames = 19;
        let state = Rc::new(RefCell::new(MockVadState::default()));
        let mut wrapper = create_mock_vad_wrapper(
            NO_VAD_PERIODIC_RESET,
            SAMPLE_RATE_8K_HZ,
            vec![1.0],
            Rc::clone(&state),
        );
        let frame = make_frame(SAMPLE_RATE_8K_HZ);
        for _ in 0..num_frames {
            wrapper.analyze(&frame);
        }
        // Only the initial reset from the constructor.
        assert_eq!(state.borrow().reset_count, 1);
    }

    #[test]
    fn vad_periodic_reset() {
        let test_cases: Vec<(i32, i32)> = vec![
            (1, 2),
            (1, 5),
            (1, 20),
            (1, 53),
            (19, 2),
            (19, 5),
            (19, 20),
            (19, 53),
            (123, 2),
            (123, 5),
            (123, 20),
            (123, 53),
        ];

        for (num_frames, vad_reset_period_frames) in test_cases {
            let vad_reset_period_ms = vad_reset_period_frames * FRAME_DURATION_MS;
            let state = Rc::new(RefCell::new(MockVadState::default()));
            let mut wrapper = create_mock_vad_wrapper(
                vad_reset_period_ms,
                SAMPLE_RATE_8K_HZ,
                vec![1.0],
                Rc::clone(&state),
            );
            let frame = make_frame(SAMPLE_RATE_8K_HZ);
            for _ in 0..num_frames {
                wrapper.analyze(&frame);
            }
            let expected_resets = 1 + num_frames / vad_reset_period_frames;
            assert_eq!(
                state.borrow().reset_count,
                expected_resets as usize,
                "num_frames={num_frames}, period={vad_reset_period_frames}"
            );
        }
    }

    #[test]
    fn check_resampled_frame_size() {
        let input_rates = [8000, 16000, 44100, 48000];
        let vad_rates = [6000, 8000, 12000, 16000, 24000];

        for &input_rate in &input_rates {
            for &vad_rate in &vad_rates {
                let state = Rc::new(RefCell::new(MockVadState::default()));
                let vad = MockVad::new(vad_rate, vec![1.0], Rc::clone(&state));
                let mut wrapper = VoiceActivityDetectorWrapper::with_vad(
                    NO_VAD_PERIODIC_RESET,
                    Box::new(vad),
                    input_rate,
                );
                let frame = make_frame(input_rate);
                wrapper.analyze(&frame);

                let expected_frame_size = (vad_rate / NUM_FRAMES_PER_SECOND) as usize;
                assert_eq!(
                    state.borrow().analyze_frames[0],
                    expected_frame_size,
                    "input_rate={input_rate}, vad_rate={vad_rate}"
                );
            }
        }
    }
}
