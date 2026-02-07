//! AVX2+FMA implementation of matched filter core.
//!
//! Ported from `matched_filter_avx2.cc` (`MatchedFilterCore_AVX2` and
//! `MatchedFilterCore_AccumulatedError_AVX2`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute horizontal sums of two 256-bit registers,
/// returning [hsum_a, hsum_b, hsum_a, hsum_b].
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn hsum_ab(a: __m256, b: __m256) -> __m128 {
    unsafe {
        let s_256 = _mm256_hadd_ps(a, b);
        let mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        let s_256 = _mm256_permutevar8x32_ps(s_256, mask);
        let lo = _mm256_extractf128_ps::<0>(s_256);
        let hi = _mm256_extractf128_ps::<1>(s_256);
        let s = _mm_hadd_ps(lo, hi);
        _mm_hadd_ps(s, s)
    }
}

/// Extract a single f32 from a __m128 by index.
///
/// # Safety
/// Requires SSE2. `idx` must be 0..3.
#[target_feature(enable = "sse2")]
unsafe fn extract_f32_128(v: __m128, idx: i32) -> f32 {
    unsafe {
        #[allow(clippy::cast_ptr_alignment, reason = "__m128 is 16-byte aligned")]
        *(&v as *const __m128).cast::<f32>().offset(idx as isize)
    }
}

/// Extract a single f32 from a __m256 by index.
///
/// # Safety
/// Requires AVX. `idx` must be 0..7.
#[target_feature(enable = "avx")]
unsafe fn extract_f32_256(v: __m256, idx: i32) -> f32 {
    unsafe {
        #[allow(clippy::cast_ptr_alignment, reason = "__m256 is 32-byte aligned")]
        *(&v as *const __m256).cast::<f32>().offset(idx as isize)
    }
}

/// AVX2+FMA matched filter core without accumulated error.
///
/// # Safety
/// Requires AVX2+FMA support. `h.len()` must be divisible by 8.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn matched_filter_core(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
) {
    unsafe {
        let h_size = h.len() as i32;
        let x_size = x.len() as i32;
        debug_assert_eq!(0, h_size % 8);

        for i in 0..y.len() {
            debug_assert!((x_start_index as i32) < x_size);
            let mut x_p = x.as_ptr().add(x_start_index);
            let mut h_p = h.as_ptr();

            let mut s_256 = _mm256_setzero_ps();
            let mut s_256_8 = _mm256_setzero_ps();
            let mut x2_sum_256 = _mm256_setzero_ps();
            let mut x2_sum_256_8 = _mm256_setzero_ps();
            let mut x2_sum = 0.0f32;
            let mut s = 0.0f32;

            let chunk1 = h_size.min(x_size - x_start_index as i32);
            let chunk2 = h_size - chunk1;

            for limit in [chunk1, chunk2] {
                let limit_by_16 = limit >> 4;
                for _ in 0..limit_by_16 {
                    let x_k = _mm256_loadu_ps(x_p);
                    let h_k = _mm256_loadu_ps(h_p);
                    let x_k_8 = _mm256_loadu_ps(x_p.add(8));
                    let h_k_8 = _mm256_loadu_ps(h_p.add(8));
                    x2_sum_256 = _mm256_fmadd_ps(x_k, x_k, x2_sum_256);
                    x2_sum_256_8 = _mm256_fmadd_ps(x_k_8, x_k_8, x2_sum_256_8);
                    s_256 = _mm256_fmadd_ps(h_k, x_k, s_256);
                    s_256_8 = _mm256_fmadd_ps(h_k_8, x_k_8, s_256_8);
                    h_p = h_p.add(16);
                    x_p = x_p.add(16);
                }
                for _ in 0..(limit - limit_by_16 * 16) {
                    let x_k = *x_p;
                    x2_sum += x_k * x_k;
                    s += *h_p * x_k;
                    h_p = h_p.add(1);
                    x_p = x_p.add(1);
                }
                x_p = x.as_ptr();
            }

            x2_sum_256 = _mm256_add_ps(x2_sum_256, x2_sum_256_8);
            s_256 = _mm256_add_ps(s_256, s_256_8);
            let sum = hsum_ab(x2_sum_256, s_256);
            x2_sum += extract_f32_128(sum, 0);
            s += extract_f32_128(sum, 1);

            let e = y[i] - s;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_256 = _mm256_set1_ps(alpha);

                let mut h_p2 = h.as_mut_ptr();
                x_p = x.as_ptr().add(x_start_index);

                for limit in [chunk1, chunk2] {
                    let limit_by_8 = limit >> 3;
                    for _ in 0..limit_by_8 {
                        let mut h_k = _mm256_loadu_ps(h_p2);
                        let x_k = _mm256_loadu_ps(x_p);
                        h_k = _mm256_fmadd_ps(x_k, alpha_256, h_k);
                        _mm256_storeu_ps(h_p2, h_k);
                        h_p2 = h_p2.add(8);
                        x_p = x_p.add(8);
                    }
                    for _ in 0..(limit - limit_by_8 * 8) {
                        *h_p2 += alpha * *x_p;
                        h_p2 = h_p2.add(1);
                        x_p = x_p.add(1);
                    }
                    x_p = x.as_ptr();
                }

                *filters_updated = true;
            }

            x_start_index = if x_start_index > 0 {
                x_start_index - 1
            } else {
                x_size as usize - 1
            };
        }
    }
}

/// AVX2+FMA matched filter core with accumulated error computation.
///
/// # Safety
/// Requires AVX2+FMA support. `h.len()` must be divisible by 16.
/// `scratch_memory.len()` must be >= `h.len()`.
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn matched_filter_core_accumulated_error(
    mut x_start_index: usize,
    x2_sum_threshold: f32,
    smoothing: f32,
    x: &[f32],
    y: &[f32],
    h: &mut [f32],
    filters_updated: &mut bool,
    error_sum: &mut f32,
    accumulated_error: &mut [f32],
    scratch_memory: &mut [f32],
) {
    unsafe {
        let h_size = h.len() as i32;
        let x_size = x.len() as i32;
        debug_assert_eq!(0, h_size % 16);

        accumulated_error.iter_mut().for_each(|v| *v = 0.0);

        for i in 0..y.len() {
            debug_assert!((x_start_index as i32) < x_size);

            let chunk1 = h_size.min(x_size - x_start_index as i32);
            if chunk1 != h_size {
                let chunk2 = (h_size - chunk1) as usize;
                let chunk1_usize = chunk1 as usize;
                scratch_memory[..chunk1_usize]
                    .copy_from_slice(&x[x_start_index..x_start_index + chunk1_usize]);
                scratch_memory[chunk1_usize..chunk1_usize + chunk2].copy_from_slice(&x[..chunk2]);
            }

            let x_p_base = if chunk1 != h_size {
                scratch_memory.as_ptr()
            } else {
                x.as_ptr().add(x_start_index)
            };
            let mut x_p = x_p_base;
            let mut h_p = h.as_ptr();
            let mut a_p = accumulated_error.as_mut_ptr();

            let mut x2_sum_256 = _mm256_setzero_ps();
            let mut x2_sum_256_8 = _mm256_setzero_ps();
            let mut x2_sum = 0.0f32;
            let mut s_acum = 0.0f32;

            let limit_by_16 = h_size >> 4;
            for _ in 0..limit_by_16 {
                let x_k = _mm256_loadu_ps(x_p);
                let h_k = _mm256_loadu_ps(h_p);
                let x_k_8 = _mm256_loadu_ps(x_p.add(8));
                let h_k_8 = _mm256_loadu_ps(h_p.add(8));

                x2_sum_256 = _mm256_fmadd_ps(x_k, x_k, x2_sum_256);
                x2_sum_256_8 = _mm256_fmadd_ps(x_k_8, x_k_8, x2_sum_256_8);

                let s_inst_256 = _mm256_mul_ps(h_k, x_k);
                let s_inst_256_8 = _mm256_mul_ps(h_k_8, x_k_8);

                let s_inst_hadd = _mm256_hadd_ps(s_inst_256, s_inst_256_8);
                let s_inst_hadd = _mm256_hadd_ps(s_inst_hadd, s_inst_hadd);

                // AVX hadd layout across lanes:
                // [0]=first4_lo, [4]=first4_hi, [1]=second4_lo, [5]=second4_hi
                s_acum += extract_f32_256(s_inst_hadd, 0);
                let e0 = s_acum - y[i];
                s_acum += extract_f32_256(s_inst_hadd, 4);
                let e1 = s_acum - y[i];
                s_acum += extract_f32_256(s_inst_hadd, 1);
                let e2 = s_acum - y[i];
                s_acum += extract_f32_256(s_inst_hadd, 5);
                let e3 = s_acum - y[i];

                let acum_error = _mm_loadu_ps(a_p);
                let e_128 = _mm_set_ps(e3, e2, e1, e0);
                let acum_error = _mm_add_ps(acum_error, _mm_mul_ps(e_128, e_128));
                _mm_storeu_ps(a_p, acum_error);

                h_p = h_p.add(16);
                x_p = x_p.add(16);
                a_p = a_p.add(4);
            }

            x2_sum_256 = _mm256_add_ps(x2_sum_256, x2_sum_256_8);
            let x2_lo = _mm256_extractf128_ps::<0>(x2_sum_256);
            let x2_hi = _mm256_extractf128_ps::<1>(x2_sum_256);
            let x2_sum_128 = _mm_add_ps(x2_lo, x2_hi);
            x2_sum += extract_f32_128(x2_sum_128, 0)
                + extract_f32_128(x2_sum_128, 1)
                + extract_f32_128(x2_sum_128, 2)
                + extract_f32_128(x2_sum_128, 3);

            let e = y[i] - s_acum;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_256 = _mm256_set1_ps(alpha);

                let mut h_p2 = h.as_mut_ptr();
                let mut x_p2 = x_p_base;

                let limit_by_8 = h_size >> 3;
                for _ in 0..limit_by_8 {
                    let mut h_k = _mm256_loadu_ps(h_p2);
                    let x_k = _mm256_loadu_ps(x_p2);
                    h_k = _mm256_fmadd_ps(x_k, alpha_256, h_k);
                    _mm256_storeu_ps(h_p2, h_k);
                    h_p2 = h_p2.add(8);
                    x_p2 = x_p2.add(8);
                }

                *filters_updated = true;
            }

            x_start_index = if x_start_index > 0 {
                x_start_index - 1
            } else {
                x_size as usize - 1
            };
        }
    }
}
