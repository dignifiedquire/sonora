//! Audio buffer generators for property-based testing.

use proptest::prelude::*;

/// Generate a mono audio frame at a given sample rate (~10ms frame).
pub fn audio_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate interleaved stereo audio (~10ms frame).
pub fn stereo_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize * 2;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate a mono i16 audio frame at a given sample rate (~10ms frame).
pub fn audio_frame_i16(sample_rate: u32) -> impl Strategy<Value = Vec<i16>> {
    let frame_size = (sample_rate / 100) as usize;
    proptest::collection::vec(i16::MIN..=i16::MAX, frame_size..=frame_size)
}

/// Generate multi-channel audio frames (~10ms).
pub fn audio_frame_multichannel_f32(
    sample_rate: u32,
    channels: usize,
) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize * channels;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate FIR filter coefficients.
pub fn fir_coefficients(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..=1.0f32, 1..=max_len)
}

/// Generate a valid WebRTC sample rate.
pub fn sample_rate() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(8000u32),
        Just(16000u32),
        Just(32000u32),
        Just(48000u32),
    ]
}

/// Generate a valid channel count (1 or 2).
pub fn channel_count() -> impl Strategy<Value = usize> {
    prop_oneof![Just(1usize), Just(2usize),]
}

/// Generate a pair of sample rate and matching mono f32 frame.
pub fn sample_rate_and_frame_f32() -> impl Strategy<Value = (u32, Vec<f32>)> {
    sample_rate().prop_flat_map(|sr| (Just(sr), audio_frame_f32(sr)))
}

/// Generate a pair of sample rate and matching mono i16 frame.
pub fn sample_rate_and_frame_i16() -> impl Strategy<Value = (u32, Vec<i16>)> {
    sample_rate().prop_flat_map(|sr| (Just(sr), audio_frame_i16(sr)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::test_runner::{Config, TestRunner};

    #[test]
    fn frame_f32_correct_length() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&audio_frame_f32(16000), |frame| {
                prop_assert_eq!(frame.len(), 160); // 16000 / 100
                for &s in &frame {
                    prop_assert!((-1.0..=1.0).contains(&s));
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn frame_f32_48k_length() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&audio_frame_f32(48000), |frame| {
                prop_assert_eq!(frame.len(), 480); // 48000 / 100
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn frame_i16_correct_length() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&audio_frame_i16(8000), |frame| {
                prop_assert_eq!(frame.len(), 80); // 8000 / 100
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn stereo_frame_correct_length() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&stereo_frame_f32(16000), |frame| {
                prop_assert_eq!(frame.len(), 320); // 160 * 2
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn multichannel_frame_correct_length() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&audio_frame_multichannel_f32(48000, 3), |frame| {
                prop_assert_eq!(frame.len(), 1440); // 480 * 3
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn sample_rate_valid_values() {
        let mut runner = TestRunner::new(Config::with_cases(50));
        runner
            .run(&sample_rate(), |sr| {
                prop_assert!([8000, 16000, 32000, 48000].contains(&sr));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn sample_rate_and_frame_consistent() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&sample_rate_and_frame_f32(), |(sr, frame)| {
                prop_assert_eq!(frame.len(), (sr / 100) as usize);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn fir_coefficients_valid_range() {
        let mut runner = TestRunner::new(Config::with_cases(20));
        runner
            .run(&fir_coefficients(64), |coeffs| {
                prop_assert!(!coeffs.is_empty());
                prop_assert!(coeffs.len() <= 64);
                for &c in &coeffs {
                    prop_assert!((-1.0..=1.0).contains(&c));
                }
                Ok(())
            })
            .unwrap();
    }
}
