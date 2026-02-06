//! Echo Return Loss (ERL) computation for the adaptive FIR filter.
//!
//! The ERL is the sum of the partition frequency responses (H2) across all
//! partitions.
//!
//! Ported from
//! `modules/audio_processing/aec3/adaptive_fir_filter_erl.h/cc`.

use crate::common::FFT_LENGTH_BY_2_PLUS_1;

/// Computes the Echo Return Loss (ERL) from the partition frequency responses.
///
/// The ERL is the element-wise sum across all partitions: erl[k] = sum_p H2[p][k].
pub(crate) fn compute_erl(
    h2: &[[f32; FFT_LENGTH_BY_2_PLUS_1]],
    erl: &mut [f32; FFT_LENGTH_BY_2_PLUS_1],
) {
    erl.fill(0.0);
    for h2_j in h2 {
        for (erl_k, &h2_jk) in erl.iter_mut().zip(h2_j.iter()) {
            *erl_k += h2_jk;
        }
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
        compute_erl(&h2, &mut erl);

        // Expected: 1 + 2 + 3 = 6 for each bin.
        for &v in &erl {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn erl_empty_partitions() {
        let h2: Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]> = Vec::new();
        let mut erl = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        compute_erl(&h2, &mut erl);
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
        compute_erl(&h2, &mut erl);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert!((erl[k] - k as f32).abs() < 1e-6);
        }
    }
}
