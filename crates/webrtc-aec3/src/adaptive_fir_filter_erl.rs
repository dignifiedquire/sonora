//! Echo Return Loss (ERL) computation for the adaptive FIR filter.
//!
//! The ERL is the sum of the partition frequency responses (H2) across all
//! partitions.
//!
//! Ported from
//! `modules/audio_processing/aec3/adaptive_fir_filter_erl.h/cc`.

use crate::common::{FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};
use webrtc_simd::SimdBackend;

/// Computes the Echo Return Loss (ERL) from the partition frequency responses.
///
/// The ERL is the element-wise sum across all partitions: erl[k] = sum_p H2[p][k].
pub(crate) fn compute_erl(
    backend: SimdBackend,
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    erl.fill(0.0);
    for h2_j in h2 {
        // Vectorized: elementwise accumulate for bins [0..64]
        backend.elementwise_accumulate(&h2_j[..FFT_LENGTH_BY_2], &mut erl[..FFT_LENGTH_BY_2]);
        // Scalar tail: bin 64 (Nyquist)
        erl[FFT_LENGTH_BY_2] += h2_j[FFT_LENGTH_BY_2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erl_sums_partitions() {
        let num_partitions = 3;
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_partitions];

        // Set known values.
        for (p, h2_p) in h2.iter_mut().enumerate() {
            h2_p.fill((p + 1) as f32);
        }

        let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(SimdBackend::Scalar, &h2, &mut erl);

        // Expected: 1 + 2 + 3 = 6 for each bin.
        for &v in &erl {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn erl_empty_partitions() {
        let h2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]> = Vec::new();
        let mut erl = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(SimdBackend::Scalar, &h2, &mut erl);
        for &v in &erl {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn erl_single_partition() {
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; 1];
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            h2[0][k] = k as f32;
        }
        let mut erl = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(SimdBackend::Scalar, &h2, &mut erl);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert!((erl[k] - k as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn compute_erl_simd_matches_scalar() {
        let num_partitions = 6;
        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_partitions];
        for (p, h2_p) in h2.iter_mut().enumerate() {
            for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                h2_p[k] = ((p * 7 + k * 13) as f32 * 0.0037).sin().abs();
            }
        }

        let mut erl_scalar = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut erl_simd = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];

        compute_erl(SimdBackend::Scalar, &h2, &mut erl_scalar);
        compute_erl(webrtc_simd::detect_backend(), &h2, &mut erl_simd);

        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let diff = (erl_scalar[k] - erl_simd[k]).abs();
            let scale = erl_scalar[k].abs().max(1e-10);
            assert!(
                diff / scale < 1e-5,
                "erl mismatch at k={k}: scalar={}, simd={}",
                erl_scalar[k],
                erl_simd[k]
            );
        }
    }
}
