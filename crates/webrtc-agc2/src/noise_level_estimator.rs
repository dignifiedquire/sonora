//! Noise floor estimator based on minimum statistics.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/noise_level_estimator.h/.cc`.

#![allow(dead_code, reason = "consumed by later AGC2 modules")]

const FRAMES_PER_SECOND: i32 = 100;

/// Update the noise floor every 5 seconds.
const UPDATE_PERIOD_NUM_FRAMES: i32 = 500;

fn frame_energy(audio: &[&[f32]]) -> f32 {
    let mut energy = 0.0_f32;
    for channel in audio {
        let channel_energy: f32 = channel.iter().map(|&s| s * s).sum();
        energy = energy.max(channel_energy);
    }
    energy
}

fn energy_to_dbfs(signal_energy: f32, num_samples: i32) -> f32 {
    debug_assert!(signal_energy >= 0.0);
    let rms_square = signal_energy / num_samples as f32;
    const MIN_DBFS: f32 = -90.309;
    if rms_square <= 1.0 {
        return MIN_DBFS;
    }
    10.0 * rms_square.log10() + MIN_DBFS
}

/// Updates the noise floor with instant decay and slow attack. This tuning is
/// specific for AGC2, so that (i) it can promptly increase the gain if the noise
/// floor drops (instant decay) and (ii) in case of music or fast speech, due to
/// which the noise floor can be overestimated, the gain reduction is slowed
/// down.
fn smooth_noise_floor_estimate(current_estimate: f32, new_estimate: f32) -> f32 {
    const ATTACK: f32 = 0.5;
    if current_estimate < new_estimate {
        // Attack phase.
        ATTACK * new_estimate + (1.0 - ATTACK) * current_estimate
    } else {
        // Instant decay.
        new_estimate
    }
}

/// Noise level estimator based on noise floor detection.
pub(crate) struct NoiseLevelEstimator {
    sample_rate_hz: i32,
    min_noise_energy: f32,
    first_period: bool,
    preliminary_noise_energy_set: bool,
    preliminary_noise_energy: f32,
    noise_energy: f32,
    counter: i32,
}

impl Default for NoiseLevelEstimator {
    fn default() -> Self {
        let mut est = Self {
            sample_rate_hz: 0,
            min_noise_energy: 0.0,
            first_period: true,
            preliminary_noise_energy_set: false,
            preliminary_noise_energy: 0.0,
            noise_energy: 0.0,
            counter: 0,
        };
        // Initially assume that 48 kHz will be used. `analyze()` will detect the
        // used sample rate and call `initialize()` again if needed.
        est.initialize(48000);
        est
    }
}

impl NoiseLevelEstimator {
    /// Analyzes a 10 ms frame, updates the noise level estimation and returns
    /// the value for the latter in dBFS.
    pub(crate) fn analyze(&mut self, frame: &[&[f32]]) -> f32 {
        debug_assert!(!frame.is_empty());
        let samples_per_channel = frame[0].len();

        // Detect sample rate changes.
        let sample_rate_hz = samples_per_channel as i32 * FRAMES_PER_SECOND;
        if sample_rate_hz != self.sample_rate_hz {
            self.initialize(sample_rate_hz);
        }

        let frame_energy = frame_energy(frame);
        if frame_energy <= self.min_noise_energy {
            // Ignore frames when muted or below the minimum measurable energy.
            return energy_to_dbfs(self.noise_energy, samples_per_channel as i32);
        }

        if self.preliminary_noise_energy_set {
            self.preliminary_noise_energy = self.preliminary_noise_energy.min(frame_energy);
        } else {
            self.preliminary_noise_energy = frame_energy;
            self.preliminary_noise_energy_set = true;
        }

        if self.counter == 0 {
            // Full period observed.
            self.first_period = false;
            // Update the estimated noise floor energy with the preliminary
            // estimation.
            self.noise_energy =
                smooth_noise_floor_estimate(self.noise_energy, self.preliminary_noise_energy);
            // Reset for a new observation period.
            self.counter = UPDATE_PERIOD_NUM_FRAMES;
            self.preliminary_noise_energy_set = false;
        } else if self.first_period {
            // While analyzing the signal during the initial period, continuously
            // update the estimated noise energy, which is monotonic.
            self.noise_energy = self.preliminary_noise_energy;
            self.counter -= 1;
        } else {
            // During the observation period it's only allowed to lower the energy.
            self.noise_energy = self.noise_energy.min(self.preliminary_noise_energy);
            self.counter -= 1;
        }

        energy_to_dbfs(self.noise_energy, samples_per_channel as i32)
    }

    fn initialize(&mut self, sample_rate_hz: i32) {
        self.sample_rate_hz = sample_rate_hz;
        self.first_period = true;
        self.preliminary_noise_energy_set = false;
        // Initialize the minimum noise energy to -84 dBFS.
        self.min_noise_energy = sample_rate_hz as f32 * 2.0 * 2.0 / FRAMES_PER_SECOND as f32;
        self.preliminary_noise_energy = self.min_noise_energy;
        self.noise_energy = self.min_noise_energy;
        self.counter = UPDATE_PERIOD_NUM_FRAMES;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const NUM_ITERATIONS: i32 = 200;

    const MIN_S16: f32 = -32768.0;
    const MAX_S16: f32 = 32767.0;

    /// PRNG matching WebRTC's `Random` class (xorshift128+).
    struct WebRtcRandom {
        x: u64,
        y: u64,
    }

    impl WebRtcRandom {
        fn new(seed: u64) -> Self {
            Self { x: seed, y: seed }
        }

        fn next_u64(&mut self) -> u64 {
            let mut s1 = self.x;
            let s0 = self.y;
            self.x = s0;
            s1 ^= s1 << 23;
            s1 ^= s1 >> 17;
            s1 ^= s0;
            s1 ^= s0 >> 26;
            self.y = s1;
            self.x.wrapping_add(self.y)
        }

        fn rand_int(&mut self, low: i32, high: i32) -> i32 {
            let range = (high as i64 - low as i64 + 1) as u64;
            let val = self.next_u64() % range;
            low + val as i32
        }
    }

    /// Generates white noise.
    struct WhiteNoiseGenerator {
        rng: WebRtcRandom,
        min_amplitude: i32,
        max_amplitude: i32,
    }

    impl WhiteNoiseGenerator {
        fn new(min_amplitude: i32, max_amplitude: i32) -> Self {
            Self {
                rng: WebRtcRandom::new(42),
                min_amplitude,
                max_amplitude,
            }
        }

        fn generate(&mut self) -> f32 {
            self.rng.rand_int(self.min_amplitude, self.max_amplitude) as f32
        }
    }

    /// Generates a sine function.
    struct SineGenerator {
        amplitude: f32,
        frequency_hz: f32,
        sample_rate_hz: i32,
        x_radians: f32,
    }

    impl SineGenerator {
        fn new(amplitude: f32, frequency_hz: f32, sample_rate_hz: i32) -> Self {
            Self {
                amplitude,
                frequency_hz,
                sample_rate_hz,
                x_radians: 0.0,
            }
        }

        fn generate(&mut self) -> f32 {
            self.x_radians += self.frequency_hz / self.sample_rate_hz as f32 * 2.0 * PI;
            if self.x_radians >= 2.0 * PI {
                self.x_radians -= 2.0 * PI;
            }
            self.amplitude * self.x_radians.sin()
        }
    }

    /// Generates periodic pulses.
    struct PulseGenerator {
        pulse_amplitude: f32,
        no_pulse_amplitude: f32,
        samples_period: i32,
        sample_counter: i32,
    }

    impl PulseGenerator {
        fn new(
            pulse_amplitude: f32,
            no_pulse_amplitude: f32,
            frequency_hz: f32,
            sample_rate_hz: i32,
        ) -> Self {
            Self {
                pulse_amplitude,
                no_pulse_amplitude,
                samples_period: (sample_rate_hz as f32 / frequency_hz) as i32,
                sample_counter: 0,
            }
        }

        fn generate(&mut self) -> f32 {
            self.sample_counter += 1;
            if self.sample_counter >= self.samples_period {
                self.sample_counter -= self.samples_period;
            }
            if self.sample_counter == 0 {
                self.pulse_amplitude
            } else {
                self.no_pulse_amplitude
            }
        }
    }

    /// Runs the noise estimator on audio generated by `gen_fn`
    /// for NUM_ITERATIONS. Returns the last noise level estimate.
    fn run_estimator(
        gen_fn: &mut dyn FnMut() -> f32,
        estimator: &mut NoiseLevelEstimator,
        sample_rate_hz: i32,
    ) -> f32 {
        let samples_per_channel = sample_rate_hz / FRAMES_PER_SECOND;
        let mut signal = vec![0.0_f32; samples_per_channel as usize];

        for _ in 0..NUM_ITERATIONS {
            for s in &mut signal {
                *s = gen_fn();
            }
            let channels: [&[f32]; 1] = [&signal];
            estimator.analyze(&channels);
        }

        // Final analysis.
        for s in &mut signal {
            *s = gen_fn();
        }
        let channels: [&[f32]; 1] = [&signal];
        estimator.analyze(&channels)
    }

    #[test]
    fn noise_floor_estimator_with_random_noise_8000() {
        noise_floor_estimator_with_random_noise(8000);
    }

    #[test]
    fn noise_floor_estimator_with_random_noise_16000() {
        noise_floor_estimator_with_random_noise(16000);
    }

    #[test]
    fn noise_floor_estimator_with_random_noise_32000() {
        noise_floor_estimator_with_random_noise(32000);
    }

    #[test]
    fn noise_floor_estimator_with_random_noise_48000() {
        noise_floor_estimator_with_random_noise(48000);
    }

    fn noise_floor_estimator_with_random_noise(sample_rate_hz: i32) {
        let mut estimator = NoiseLevelEstimator::default();
        let mut noise_gen = WhiteNoiseGenerator::new(MIN_S16 as i32, MAX_S16 as i32);
        let noise_level_dbfs =
            run_estimator(&mut || noise_gen.generate(), &mut estimator, sample_rate_hz);
        assert!(
            (noise_level_dbfs - (-5.5)).abs() < 0.5,
            "noise_level_dbfs={noise_level_dbfs} at {sample_rate_hz}Hz, expected ~-5.5"
        );
    }

    #[test]
    fn noise_floor_estimator_with_sine_tone_8000() {
        noise_floor_estimator_with_sine_tone(8000);
    }

    #[test]
    fn noise_floor_estimator_with_sine_tone_16000() {
        noise_floor_estimator_with_sine_tone(16000);
    }

    #[test]
    fn noise_floor_estimator_with_sine_tone_32000() {
        noise_floor_estimator_with_sine_tone(32000);
    }

    #[test]
    fn noise_floor_estimator_with_sine_tone_48000() {
        noise_floor_estimator_with_sine_tone(48000);
    }

    fn noise_floor_estimator_with_sine_tone(sample_rate_hz: i32) {
        let mut estimator = NoiseLevelEstimator::default();
        let mut sine_gen = SineGenerator::new(MAX_S16, 600.0, sample_rate_hz);
        let noise_level_dbfs =
            run_estimator(&mut || sine_gen.generate(), &mut estimator, sample_rate_hz);
        assert!(
            (noise_level_dbfs - (-3.0)).abs() < 0.1,
            "noise_level_dbfs={noise_level_dbfs} at {sample_rate_hz}Hz, expected ~-3.0"
        );
    }

    #[test]
    fn noise_floor_estimator_with_pulse_tone_8000() {
        noise_floor_estimator_with_pulse_tone(8000);
    }

    #[test]
    fn noise_floor_estimator_with_pulse_tone_16000() {
        noise_floor_estimator_with_pulse_tone(16000);
    }

    #[test]
    fn noise_floor_estimator_with_pulse_tone_32000() {
        noise_floor_estimator_with_pulse_tone(32000);
    }

    #[test]
    fn noise_floor_estimator_with_pulse_tone_48000() {
        noise_floor_estimator_with_pulse_tone(48000);
    }

    fn noise_floor_estimator_with_pulse_tone(sample_rate_hz: i32) {
        let mut estimator = NoiseLevelEstimator::default();
        let no_pulse_amplitude = 10.0_f32;
        let mut pulse_gen = PulseGenerator::new(MAX_S16, no_pulse_amplitude, 20.0, sample_rate_hz);
        let noise_level_dbfs =
            run_estimator(&mut || pulse_gen.generate(), &mut estimator, sample_rate_hz);
        let expected_noise_floor_dbfs = 20.0 * (no_pulse_amplitude / MAX_S16).log10();
        assert!(
            (noise_level_dbfs - expected_noise_floor_dbfs).abs() < 0.5,
            "noise_level_dbfs={noise_level_dbfs} at {sample_rate_hz}Hz, expected ~{expected_noise_floor_dbfs}"
        );
    }
}
