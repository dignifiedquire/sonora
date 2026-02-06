//! Internal pitch search helpers.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/pitch_search_internal.cc`.

use super::common::{
    BUF_SIZE_24K_HZ, FRAME_SIZE_20MS_12K_HZ, FRAME_SIZE_20MS_24K_HZ, INITIAL_NUM_LAGS_24K_HZ,
    MAX_PITCH_24K_HZ, MAX_PITCH_48K_HZ, MIN_PITCH_24K_HZ, MIN_PITCH_48K_HZ, NUM_LAGS_12K_HZ,
    REFINE_NUM_LAGS_24K_HZ,
};
use webrtc_simd::SimdBackend;

/// Performs 2x decimation without any anti-aliasing filter.
pub fn decimate_2x(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BUF_SIZE_24K_HZ);
    debug_assert_eq!(dst.len(), BUF_SIZE_24K_HZ / 2);
    for (i, d) in dst.iter_mut().enumerate() {
        *d = src[2 * i];
    }
}

/// Top-2 pitch period candidates (inverted lags).
#[derive(Debug, Clone, Copy, Default)]
pub struct CandidatePitchPeriods {
    pub best: i32,
    pub second_best: i32,
}

/// Pitch period and strength.
#[derive(Debug, Clone, Copy, Default)]
pub struct PitchInfo {
    pub period: i32,
    pub strength: f32,
}

fn compute_auto_correlation_single(
    inverted_lag: usize,
    pitch_buffer: &[f32],
    backend: SimdBackend,
) -> f32 {
    debug_assert!(inverted_lag < BUF_SIZE_24K_HZ);
    debug_assert!(inverted_lag < REFINE_NUM_LAGS_24K_HZ);
    let x = &pitch_buffer[MAX_PITCH_24K_HZ..MAX_PITCH_24K_HZ + FRAME_SIZE_20MS_24K_HZ];
    let y = &pitch_buffer[inverted_lag..inverted_lag + FRAME_SIZE_20MS_24K_HZ];
    backend.dot_product(x, y)
}

/// Pseudo-interpolation offset: returns -1, 0, or +1.
fn get_pitch_pseudo_interpolation_offset(
    prev_auto_correlation: f32,
    curr_auto_correlation: f32,
    next_auto_correlation: f32,
) -> i32 {
    if (next_auto_correlation - prev_auto_correlation)
        > 0.7 * (curr_auto_correlation - prev_auto_correlation)
    {
        1
    } else if (prev_auto_correlation - next_auto_correlation)
        > 0.7 * (curr_auto_correlation - next_auto_correlation)
    {
        -1
    } else {
        0
    }
}

/// Refines a pitch period `lag` with pseudo-interpolation. Output rate is 2x.
fn pitch_pseudo_interpolation_lag(lag: i32, pitch_buffer: &[f32], backend: SimdBackend) -> i32 {
    let mut offset = 0;
    if lag > 0 && lag < MAX_PITCH_24K_HZ as i32 {
        let inverted_lag = MAX_PITCH_24K_HZ as i32 - lag;
        offset = get_pitch_pseudo_interpolation_offset(
            compute_auto_correlation_single((inverted_lag + 1) as usize, pitch_buffer, backend),
            compute_auto_correlation_single(inverted_lag as usize, pitch_buffer, backend),
            compute_auto_correlation_single((inverted_lag - 1) as usize, pitch_buffer, backend),
        );
    }
    2 * lag + offset
}

/// Integer multipliers used when looking for sub-harmonics.
const SUB_HARMONIC_MULTIPLIERS: [i32; 14] = [3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Number of analyzed pitches to the left/right of a pitch candidate.
const PITCH_NEIGHBORHOOD_RADIUS: i32 = 2;

/// Maximum number of analyzed pitch periods.
const NUM_PITCH_CANDIDATES: usize = 2;
const MAX_PITCH_PERIODS_24K_HZ: usize =
    NUM_PITCH_CANDIDATES * (2 * PITCH_NEIGHBORHOOD_RADIUS as usize + 1);

#[derive(Debug, Clone, Copy)]
struct Range {
    min: i32,
    max: i32,
}

fn create_inverted_lag_range(inverted_lag: i32) -> Range {
    Range {
        min: (inverted_lag - PITCH_NEIGHBORHOOD_RADIUS).max(0),
        max: (inverted_lag + PITCH_NEIGHBORHOOD_RADIUS).min(INITIAL_NUM_LAGS_24K_HZ as i32 - 1),
    }
}

/// Collection of inverted lags.
struct InvertedLagsIndex {
    inverted_lags: [i32; MAX_PITCH_PERIODS_24K_HZ],
    num_entries: usize,
}

impl InvertedLagsIndex {
    fn new() -> Self {
        Self {
            inverted_lags: [0; MAX_PITCH_PERIODS_24K_HZ],
            num_entries: 0,
        }
    }

    fn append(&mut self, inverted_lag: i32) {
        debug_assert!(self.num_entries < MAX_PITCH_PERIODS_24K_HZ);
        self.inverted_lags[self.num_entries] = inverted_lag;
        self.num_entries += 1;
    }

    fn as_slice(&self) -> &[i32] {
        &self.inverted_lags[..self.num_entries]
    }
}

/// Computes auto-correlation for inverted lags in a range.
fn compute_auto_correlation_range(
    inverted_lags: Range,
    pitch_buffer: &[f32],
    auto_correlation: &mut [f32],
    inverted_lags_index: &mut InvertedLagsIndex,
    backend: SimdBackend,
) {
    debug_assert!(inverted_lags.min <= inverted_lags.max);
    // Trick to avoid zero initialization — needed by pseudo-interpolation.
    if inverted_lags.min > 0 {
        auto_correlation[inverted_lags.min as usize - 1] = 0.0;
    }
    if (inverted_lags.max as usize) < INITIAL_NUM_LAGS_24K_HZ - 1 {
        auto_correlation[inverted_lags.max as usize + 1] = 0.0;
    }
    debug_assert!(inverted_lags.min >= 0);
    debug_assert!((inverted_lags.max as usize) < INITIAL_NUM_LAGS_24K_HZ);
    for inverted_lag in inverted_lags.min..=inverted_lags.max {
        auto_correlation[inverted_lag as usize] =
            compute_auto_correlation_single(inverted_lag as usize, pitch_buffer, backend);
        inverted_lags_index.append(inverted_lag);
    }
}

/// Searches the strongest pitch period at 24 kHz and returns its inverted lag
/// at 48 kHz.
fn compute_pitch_period_48k_hz_from_lags(
    inverted_lags: &[i32],
    auto_correlation: &[f32],
    y_energy: &[f32],
) -> i32 {
    let mut best_inverted_lag = 0_i32;
    let mut best_numerator = -1.0_f32;
    let mut best_denominator = 0.0_f32;
    for &inverted_lag in inverted_lags {
        let il = inverted_lag as usize;
        if auto_correlation[il] > 0.0 {
            let numerator = auto_correlation[il] * auto_correlation[il];
            let denominator = y_energy[il];
            if numerator * best_denominator > best_numerator * denominator {
                best_inverted_lag = inverted_lag;
                best_numerator = numerator;
                best_denominator = denominator;
            }
        }
    }
    // Pseudo-interpolation to transform to 48 kHz.
    if best_inverted_lag == 0 || best_inverted_lag >= INITIAL_NUM_LAGS_24K_HZ as i32 - 1 {
        return best_inverted_lag * 2;
    }
    let il = best_inverted_lag as usize;
    let offset = get_pitch_pseudo_interpolation_offset(
        auto_correlation[il + 1],
        auto_correlation[il],
        auto_correlation[il - 1],
    );
    2 * best_inverted_lag + offset
}

/// Returns an alternative pitch period for `pitch_period` given a `multiplier`
/// and a `divisor`.
const fn get_alternative_pitch_period(pitch_period: i32, multiplier: i32, divisor: i32) -> i32 {
    (2 * multiplier * pitch_period + divisor) / (2 * divisor)
}

/// Returns true if the alternative pitch period is stronger than the initial one.
fn is_alternative_pitch_stronger_than_initial(
    last: PitchInfo,
    initial: PitchInfo,
    alternative: PitchInfo,
    period_divisor: i32,
) -> bool {
    // Computed as [5*k*k for k in range(16)].
    const INITIAL_PITCH_PERIOD_THRESHOLDS: [i32; 14] = [
        20, 45, 80, 125, 180, 245, 320, 405, 500, 605, 720, 845, 980, 1125,
    ];

    debug_assert!(last.period >= 0);
    debug_assert!(initial.period >= 0);
    debug_assert!(alternative.period >= 0);
    debug_assert!(period_divisor >= 2);

    // Pitch tracking term.
    let mut lower_threshold_term = 0.0_f32;
    if (alternative.period - last.period).abs() <= 1 {
        lower_threshold_term = last.strength;
    } else if (alternative.period - last.period).abs() == 2
        && initial.period > INITIAL_PITCH_PERIOD_THRESHOLDS[(period_divisor - 2) as usize]
    {
        lower_threshold_term = 0.5 * last.strength;
    }

    let mut threshold = (0.7 * initial.strength - lower_threshold_term).max(0.3);
    if alternative.period < 3 * MIN_PITCH_24K_HZ as i32 {
        threshold = (0.85 * initial.strength - lower_threshold_term).max(0.4);
    } else if alternative.period < 2 * MIN_PITCH_24K_HZ as i32 {
        threshold = (0.9 * initial.strength - lower_threshold_term).max(0.5);
    }
    alternative.strength > threshold
}

/// Computes the sum of squared samples for every sliding frame `y` in the
/// pitch buffer.
pub fn compute_sliding_frame_square_energies_24k_hz(
    pitch_buffer: &[f32],
    y_energy: &mut [f32],
    backend: SimdBackend,
) {
    debug_assert_eq!(pitch_buffer.len(), BUF_SIZE_24K_HZ);
    debug_assert_eq!(y_energy.len(), REFINE_NUM_LAGS_24K_HZ);

    let frame_view = &pitch_buffer[..FRAME_SIZE_20MS_24K_HZ];
    let mut yy = backend.dot_product(frame_view, frame_view);
    y_energy[0] = yy;
    for inverted_lag in 0..MAX_PITCH_24K_HZ {
        yy -= pitch_buffer[inverted_lag] * pitch_buffer[inverted_lag];
        yy += pitch_buffer[inverted_lag + FRAME_SIZE_20MS_24K_HZ]
            * pitch_buffer[inverted_lag + FRAME_SIZE_20MS_24K_HZ];
        yy = yy.max(1.0);
        y_energy[inverted_lag + 1] = yy;
    }
}

/// Computes the candidate pitch periods at 12 kHz.
pub fn compute_pitch_period_12k_hz(
    pitch_buffer: &[f32],
    auto_correlation: &[f32],
    backend: SimdBackend,
) -> CandidatePitchPeriods {
    debug_assert_eq!(pitch_buffer.len(), BUF_SIZE_24K_HZ / 2);
    debug_assert_eq!(auto_correlation.len(), NUM_LAGS_12K_HZ);

    #[derive(Clone, Copy)]
    struct PitchCandidate {
        period_inverted_lag: i32,
        strength_numerator: f32,
        strength_denominator: f32,
    }

    impl PitchCandidate {
        fn has_stronger_pitch_than(&self, b: &Self) -> bool {
            self.strength_numerator * b.strength_denominator
                > b.strength_numerator * self.strength_denominator
        }
    }

    let frame_view = &pitch_buffer[..FRAME_SIZE_20MS_12K_HZ + 1];
    let mut denominator = 1.0 + backend.dot_product(frame_view, frame_view);

    let mut best = PitchCandidate {
        period_inverted_lag: 0,
        strength_numerator: -1.0,
        strength_denominator: 0.0,
    };
    let mut second_best = PitchCandidate {
        period_inverted_lag: 1,
        strength_numerator: -1.0,
        strength_denominator: 0.0,
    };

    for inverted_lag in 0..NUM_LAGS_12K_HZ {
        if auto_correlation[inverted_lag] > 0.0 {
            let candidate = PitchCandidate {
                period_inverted_lag: inverted_lag as i32,
                strength_numerator: auto_correlation[inverted_lag] * auto_correlation[inverted_lag],
                strength_denominator: denominator,
            };
            if candidate.has_stronger_pitch_than(&second_best) {
                if candidate.has_stronger_pitch_than(&best) {
                    second_best = best;
                    best = candidate;
                } else {
                    second_best = candidate;
                }
            }
        }
        // Update energy for the next inverted lag.
        let y_old = pitch_buffer[inverted_lag];
        let y_new = pitch_buffer[inverted_lag + FRAME_SIZE_20MS_12K_HZ];
        denominator -= y_old * y_old;
        denominator += y_new * y_new;
        denominator = denominator.max(0.0);
    }

    CandidatePitchPeriods {
        best: best.period_inverted_lag,
        second_best: second_best.period_inverted_lag,
    }
}

/// Computes the pitch period at 48 kHz given the 24 kHz pitch buffer,
/// sliding frame energies and pitch period candidates at 24 kHz.
pub fn compute_pitch_period_48k_hz(
    pitch_buffer: &[f32],
    y_energy: &[f32],
    pitch_candidates: CandidatePitchPeriods,
    backend: SimdBackend,
) -> i32 {
    debug_assert_eq!(pitch_buffer.len(), BUF_SIZE_24K_HZ);
    debug_assert_eq!(y_energy.len(), REFINE_NUM_LAGS_24K_HZ);

    let mut auto_correlation = [0.0_f32; INITIAL_NUM_LAGS_24K_HZ];
    let mut inverted_lags_index = InvertedLagsIndex::new();

    // Create two inverted lag ranges so that r1 precedes r2.
    let swap = pitch_candidates.best > pitch_candidates.second_best;
    let r1 = create_inverted_lag_range(if swap {
        pitch_candidates.second_best
    } else {
        pitch_candidates.best
    });
    let r2 = create_inverted_lag_range(if swap {
        pitch_candidates.best
    } else {
        pitch_candidates.second_best
    });

    debug_assert!(r1.min <= r1.max);
    debug_assert!(r2.min <= r2.max);
    debug_assert!(r1.min <= r2.min);
    debug_assert!(r1.max <= r2.max);

    if r1.max + 1 >= r2.min {
        // Overlapping or adjacent ranges.
        compute_auto_correlation_range(
            Range {
                min: r1.min,
                max: r2.max,
            },
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            backend,
        );
    } else {
        // Disjoint ranges.
        compute_auto_correlation_range(
            r1,
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            backend,
        );
        compute_auto_correlation_range(
            r2,
            pitch_buffer,
            &mut auto_correlation,
            &mut inverted_lags_index,
            backend,
        );
    }

    compute_pitch_period_48k_hz_from_lags(
        inverted_lags_index.as_slice(),
        &auto_correlation,
        y_energy,
    )
}

/// Computes the pitch period at 48 kHz searching in an extended pitch range.
pub fn compute_extended_pitch_period_48k_hz(
    pitch_buffer: &[f32],
    y_energy: &[f32],
    initial_pitch_period_48k_hz: i32,
    last_pitch_48k_hz: PitchInfo,
    backend: SimdBackend,
) -> PitchInfo {
    debug_assert_eq!(pitch_buffer.len(), BUF_SIZE_24K_HZ);
    debug_assert_eq!(y_energy.len(), REFINE_NUM_LAGS_24K_HZ);
    debug_assert!(MIN_PITCH_48K_HZ as i32 <= initial_pitch_period_48k_hz);
    debug_assert!(initial_pitch_period_48k_hz <= MAX_PITCH_48K_HZ as i32);

    #[derive(Clone, Copy)]
    struct RefinedPitchCandidate {
        period: i32,
        strength: f32,
        xy: f32,
        y_energy: f32,
    }

    let x_energy = y_energy[MAX_PITCH_24K_HZ];
    let pitch_strength = |xy: f32, ye: f32| -> f32 {
        debug_assert!(x_energy * ye >= 0.0);
        xy / (1.0 + x_energy * ye).sqrt()
    };

    // Initialize the best pitch candidate.
    let initial_period = (initial_pitch_period_48k_hz / 2).min(MAX_PITCH_24K_HZ as i32 - 1);
    let initial_xy = compute_auto_correlation_single(
        (MAX_PITCH_24K_HZ as i32 - initial_period) as usize,
        pitch_buffer,
        backend,
    );
    let initial_ye = y_energy[(MAX_PITCH_24K_HZ as i32 - initial_period) as usize];
    let initial_str = pitch_strength(initial_xy, initial_ye);

    let mut best_pitch = RefinedPitchCandidate {
        period: initial_period,
        strength: initial_str,
        xy: initial_xy,
        y_energy: initial_ye,
    };

    let initial_pitch = PitchInfo {
        period: best_pitch.period,
        strength: best_pitch.strength,
    };
    let last_pitch = PitchInfo {
        period: last_pitch_48k_hz.period / 2,
        strength: last_pitch_48k_hz.strength,
    };

    // Find max_period_divisor.
    let max_period_divisor = (2 * initial_pitch.period) / (2 * MIN_PITCH_24K_HZ as i32 - 1);
    for period_divisor in 2..=max_period_divisor {
        let alt_period = get_alternative_pitch_period(initial_pitch.period, 1, period_divisor);
        debug_assert!(alt_period >= MIN_PITCH_24K_HZ as i32);

        let mut dual_alt_period = get_alternative_pitch_period(
            initial_pitch.period,
            SUB_HARMONIC_MULTIPLIERS[(period_divisor - 2) as usize],
            period_divisor,
        );
        debug_assert!(dual_alt_period > 0);
        if period_divisor == 2 && dual_alt_period > MAX_PITCH_24K_HZ as i32 {
            dual_alt_period = initial_pitch.period;
        }
        debug_assert_ne!(alt_period, dual_alt_period);

        let xy_primary = compute_auto_correlation_single(
            (MAX_PITCH_24K_HZ as i32 - alt_period) as usize,
            pitch_buffer,
            backend,
        );
        let xy_secondary = compute_auto_correlation_single(
            (MAX_PITCH_24K_HZ as i32 - dual_alt_period) as usize,
            pitch_buffer,
            backend,
        );
        let xy = 0.5 * (xy_primary + xy_secondary);
        let yy = 0.5
            * (y_energy[(MAX_PITCH_24K_HZ as i32 - alt_period) as usize]
                + y_energy[(MAX_PITCH_24K_HZ as i32 - dual_alt_period) as usize]);

        let alternative_pitch = PitchInfo {
            period: alt_period,
            strength: pitch_strength(xy, yy),
        };

        if is_alternative_pitch_stronger_than_initial(
            last_pitch,
            initial_pitch,
            alternative_pitch,
            period_divisor,
        ) {
            best_pitch = RefinedPitchCandidate {
                period: alternative_pitch.period,
                strength: alternative_pitch.strength,
                xy,
                y_energy: yy,
            };
        }
    }

    // Final pitch strength and period.
    best_pitch.xy = best_pitch.xy.max(0.0);
    debug_assert!(best_pitch.y_energy >= 0.0);
    let mut final_pitch_strength = if best_pitch.y_energy <= best_pitch.xy {
        1.0
    } else {
        best_pitch.xy / (best_pitch.y_energy + 1.0)
    };
    final_pitch_strength = best_pitch.strength.min(final_pitch_strength);

    let final_pitch_period_48k_hz = (MIN_PITCH_48K_HZ as i32).max(pitch_pseudo_interpolation_lag(
        best_pitch.period,
        pitch_buffer,
        backend,
    ));

    PitchInfo {
        period: final_pitch_period_48k_hz,
        strength: final_pitch_strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnn_vad::common::BUF_SIZE_12K_HZ;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;

    fn test_resources_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/resources/audio_processing/agc2/rnn_vad")
    }

    /// Loads PitchTestData from `pitch_search_int.dat`.
    struct PitchTestData {
        pitch_buffer_24k: Vec<f32>,
        square_energies_24k: Vec<f32>,
        auto_correlation_12k: Vec<f32>,
    }

    impl PitchTestData {
        fn load() -> Self {
            let path = test_resources_dir().join("pitch_search_int.dat");
            let mut file = fs::File::open(&path)
                .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            let mut offset = 0;
            let pitch_buffer_24k = floats[offset..offset + BUF_SIZE_24K_HZ].to_vec();
            offset += BUF_SIZE_24K_HZ;
            let mut square_energies_24k = floats[offset..offset + REFINE_NUM_LAGS_24K_HZ].to_vec();
            offset += REFINE_NUM_LAGS_24K_HZ;
            let auto_correlation_12k = floats[offset..offset + NUM_LAGS_12K_HZ].to_vec();
            // Reverse (required after WebRTC CL 191703).
            square_energies_24k.reverse();

            Self {
                pitch_buffer_24k,
                square_energies_24k,
                auto_correlation_12k,
            }
        }
    }

    fn expect_near_absolute(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() <= tolerance,
                "Mismatch at index {i}: expected {e}, got {a}, diff {}",
                (e - a).abs()
            );
        }
    }

    #[test]
    fn compute_sliding_frame_square_energies_24k_hz_within_tolerance() {
        let test_data = PitchTestData::load();
        let mut computed = vec![0.0_f32; REFINE_NUM_LAGS_24K_HZ];
        let backend = webrtc_simd::detect_backend();
        compute_sliding_frame_square_energies_24k_hz(
            &test_data.pitch_buffer_24k,
            &mut computed,
            backend,
        );
        expect_near_absolute(&test_data.square_energies_24k, &computed, 1e-3);
    }

    #[test]
    fn compute_pitch_period_12k_hz_bit_exactness() {
        let test_data = PitchTestData::load();
        let mut pitch_buf_decimated = vec![0.0_f32; BUF_SIZE_12K_HZ];
        decimate_2x(&test_data.pitch_buffer_24k, &mut pitch_buf_decimated);
        let backend = webrtc_simd::detect_backend();
        let candidates = compute_pitch_period_12k_hz(
            &pitch_buf_decimated,
            &test_data.auto_correlation_12k,
            backend,
        );
        assert_eq!(candidates.best, 140);
        assert_eq!(candidates.second_best, 142);
    }

    #[test]
    fn compute_pitch_period_48k_hz_bit_exactness() {
        let test_data = PitchTestData::load();
        let mut y_energy = vec![0.0_f32; REFINE_NUM_LAGS_24K_HZ];
        let backend = webrtc_simd::detect_backend();
        compute_sliding_frame_square_energies_24k_hz(
            &test_data.pitch_buffer_24k,
            &mut y_energy,
            backend,
        );
        assert_eq!(
            compute_pitch_period_48k_hz(
                &test_data.pitch_buffer_24k,
                &y_energy,
                CandidatePitchPeriods {
                    best: 280,
                    second_best: 284,
                },
                backend,
            ),
            560
        );
        assert_eq!(
            compute_pitch_period_48k_hz(
                &test_data.pitch_buffer_24k,
                &y_energy,
                CandidatePitchPeriods {
                    best: 260,
                    second_best: 284,
                },
                backend,
            ),
            568
        );
    }

    #[test]
    fn compute_pitch_period_48k_hz_order_does_not_matter() {
        let test_data = PitchTestData::load();
        let mut y_energy = vec![0.0_f32; REFINE_NUM_LAGS_24K_HZ];
        let backend = webrtc_simd::detect_backend();
        compute_sliding_frame_square_energies_24k_hz(
            &test_data.pitch_buffer_24k,
            &mut y_energy,
            backend,
        );

        let test_cases: &[(i32, i32)] = &[
            (0, 2),
            (260, 284),
            (280, 284),
            (
                INITIAL_NUM_LAGS_24K_HZ as i32 - 2,
                INITIAL_NUM_LAGS_24K_HZ as i32 - 1,
            ),
        ];

        for &(best, second_best) in test_cases {
            let result1 = compute_pitch_period_48k_hz(
                &test_data.pitch_buffer_24k,
                &y_energy,
                CandidatePitchPeriods { best, second_best },
                backend,
            );
            let result2 = compute_pitch_period_48k_hz(
                &test_data.pitch_buffer_24k,
                &y_energy,
                CandidatePitchPeriods {
                    best: second_best,
                    second_best: best,
                },
                backend,
            );
            assert_eq!(
                result1, result2,
                "Order matters for candidates ({best}, {second_best})"
            );
        }
    }

    const TEST_PITCH_PERIODS_LOW: i32 = 3 * MIN_PITCH_48K_HZ as i32 / 2;
    const TEST_PITCH_PERIODS_HIGH: i32 =
        (3 * MIN_PITCH_48K_HZ as i32 + MAX_PITCH_48K_HZ as i32) / 2;
    const TEST_PITCH_STRENGTH_LOW: f32 = 0.35;
    const TEST_PITCH_STRENGTH_HIGH: f32 = 0.75;

    #[test]
    fn extended_pitch_period_search() {
        let test_data = PitchTestData::load();
        let mut y_energy = vec![0.0_f32; REFINE_NUM_LAGS_24K_HZ];
        let backend = webrtc_simd::detect_backend();
        compute_sliding_frame_square_energies_24k_hz(
            &test_data.pitch_buffer_24k,
            &mut y_energy,
            backend,
        );

        for &last_pitch_period in &[TEST_PITCH_PERIODS_LOW, TEST_PITCH_PERIODS_HIGH] {
            for &last_pitch_strength in &[TEST_PITCH_STRENGTH_LOW, TEST_PITCH_STRENGTH_HIGH] {
                let last_pitch = PitchInfo {
                    period: last_pitch_period,
                    strength: last_pitch_strength,
                };

                // Test with low initial pitch period.
                let result = compute_extended_pitch_period_48k_hz(
                    &test_data.pitch_buffer_24k,
                    &y_energy,
                    TEST_PITCH_PERIODS_LOW,
                    last_pitch,
                    backend,
                );
                assert_eq!(result.period, 91);
                assert!(
                    (result.strength - (-0.0188608_f32)).abs() < 1e-6,
                    "Strength mismatch: expected -0.0188608, got {}",
                    result.strength
                );

                // Test with high initial pitch period.
                let result = compute_extended_pitch_period_48k_hz(
                    &test_data.pitch_buffer_24k,
                    &y_energy,
                    TEST_PITCH_PERIODS_HIGH,
                    last_pitch,
                    backend,
                );
                assert_eq!(result.period, 475);
                assert!(
                    (result.strength - (-0.0904344_f32)).abs() < 1e-6,
                    "Strength mismatch: expected -0.0904344, got {}",
                    result.strength
                );
            }
        }
    }
}
