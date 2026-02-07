//! SSE2 implementation of matched filter core.
//!
//! Ported from `matched_filter.cc` (`MatchedFilterCore_SSE2` and
//! `MatchedFilterCore_AccumulatedError_SSE2`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 horizontal sum of a 128-bit register.
///
/// # Safety
/// Requires SSE2 support.
#[target_feature(enable = "sse2")]
unsafe fn hsum_ps(v: __m128) -> f32 {
    unsafe {
        let shuf = _mm_shuffle_ps::<0b01_00_11_10>(v, v);
        let sum1 = _mm_add_ps(v, shuf);
        let shuf2 = _mm_shuffle_ps::<0b00_01_00_01>(sum1, sum1);
        let sum2 = _mm_add_ps(sum1, shuf2);
        _mm_cvtss_f32(sum2)
    }
}

/// SSE2 matched filter core without accumulated error.
///
/// # Safety
/// Requires SSE2 support. `h.len()` must be divisible by 4.
#[target_feature(enable = "sse2")]
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
        debug_assert_eq!(0, h_size % 4);

        for i in 0..y.len() {
            debug_assert!((x_start_index as i32) < x_size);
            let mut x_p = x.as_ptr().add(x_start_index);
            let mut h_p = h.as_ptr();

            let mut s_128 = _mm_setzero_ps();
            let mut s_128_4 = _mm_setzero_ps();
            let mut x2_sum_128 = _mm_setzero_ps();
            let mut x2_sum_128_4 = _mm_setzero_ps();
            let mut x2_sum = 0.0f32;
            let mut s = 0.0f32;

            let chunk1 = h_size.min(x_size - x_start_index as i32);
            let chunk2 = h_size - chunk1;

            for limit in [chunk1, chunk2] {
                let limit_by_8 = limit >> 3;
                for _ in 0..limit_by_8 {
                    let x_k = _mm_loadu_ps(x_p);
                    let h_k = _mm_loadu_ps(h_p);
                    let x_k_4 = _mm_loadu_ps(x_p.add(4));
                    let h_k_4 = _mm_loadu_ps(h_p.add(4));
                    let xx = _mm_mul_ps(x_k, x_k);
                    let xx_4 = _mm_mul_ps(x_k_4, x_k_4);
                    x2_sum_128 = _mm_add_ps(x2_sum_128, xx);
                    x2_sum_128_4 = _mm_add_ps(x2_sum_128_4, xx_4);
                    let hx = _mm_mul_ps(h_k, x_k);
                    let hx_4 = _mm_mul_ps(h_k_4, x_k_4);
                    s_128 = _mm_add_ps(s_128, hx);
                    s_128_4 = _mm_add_ps(s_128_4, hx_4);
                    h_p = h_p.add(8);
                    x_p = x_p.add(8);
                }
                for _ in 0..(limit - limit_by_8 * 8) {
                    let x_k = *x_p;
                    x2_sum += x_k * x_k;
                    s += *h_p * x_k;
                    h_p = h_p.add(1);
                    x_p = x_p.add(1);
                }
                x_p = x.as_ptr();
            }

            x2_sum_128 = _mm_add_ps(x2_sum_128, x2_sum_128_4);
            x2_sum += hsum_ps(x2_sum_128);
            s_128 = _mm_add_ps(s_128, s_128_4);
            s += hsum_ps(s_128);

            let e = y[i] - s;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_128 = _mm_set1_ps(alpha);

                let mut h_p2 = h.as_mut_ptr();
                x_p = x.as_ptr().add(x_start_index);

                for limit in [chunk1, chunk2] {
                    let limit_by_4 = limit >> 2;
                    for _ in 0..limit_by_4 {
                        let mut h_k = _mm_loadu_ps(h_p2);
                        let x_k = _mm_loadu_ps(x_p);
                        let alpha_x = _mm_mul_ps(alpha_128, x_k);
                        h_k = _mm_add_ps(h_k, alpha_x);
                        _mm_storeu_ps(h_p2, h_k);
                        h_p2 = h_p2.add(4);
                        x_p = x_p.add(4);
                    }
                    for _ in 0..(limit - limit_by_4 * 4) {
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

/// SSE2 matched filter core with accumulated error computation.
///
/// # Safety
/// Requires SSE2 support. `h.len()` must be divisible by 8.
/// `scratch_memory.len()` must be >= `h.len()`.
#[target_feature(enable = "sse2")]
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
        debug_assert_eq!(0, h_size % 8);

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

            let mut x2_sum_128 = _mm_setzero_ps();
            let mut x2_sum_128_4 = _mm_setzero_ps();
            let mut x2_sum = 0.0f32;
            let mut s_acum = 0.0f32;

            let limit_by_8 = h_size >> 3;
            for _ in 0..limit_by_8 {
                let x_k = _mm_loadu_ps(x_p);
                let h_k = _mm_loadu_ps(h_p);
                let x_k_4 = _mm_loadu_ps(x_p.add(4));
                let h_k_4 = _mm_loadu_ps(h_p.add(4));
                let xx = _mm_mul_ps(x_k, x_k);
                let xx_4 = _mm_mul_ps(x_k_4, x_k_4);
                x2_sum_128 = _mm_add_ps(x2_sum_128, xx);
                x2_sum_128_4 = _mm_add_ps(x2_sum_128_4, xx_4);

                let s_inst = _mm_mul_ps(h_k, x_k);
                s_acum += hsum_ps(s_inst);
                let e0 = s_acum - y[i];
                let s_inst_4 = _mm_mul_ps(h_k_4, x_k_4);
                s_acum += hsum_ps(s_inst_4);
                let e1 = s_acum - y[i];

                *a_p += e0 * e0;
                *a_p.add(1) += e1 * e1;

                h_p = h_p.add(8);
                x_p = x_p.add(8);
                a_p = a_p.add(2);
            }

            x2_sum_128 = _mm_add_ps(x2_sum_128, x2_sum_128_4);
            x2_sum += hsum_ps(x2_sum_128);

            let e = y[i] - s_acum;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_128 = _mm_set1_ps(alpha);

                let mut h_p2 = h.as_mut_ptr();
                let mut x_p2 = x_p_base;

                let limit_by_4 = h_size >> 2;
                for _ in 0..limit_by_4 {
                    let mut h_k = _mm_loadu_ps(h_p2);
                    let x_k = _mm_loadu_ps(x_p2);
                    let alpha_x = _mm_mul_ps(alpha_128, x_k);
                    h_k = _mm_add_ps(h_k, alpha_x);
                    _mm_storeu_ps(h_p2, h_k);
                    h_p2 = h_p2.add(4);
                    x_p2 = x_p2.add(4);
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
