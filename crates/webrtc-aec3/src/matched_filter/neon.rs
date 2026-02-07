//! NEON implementation of matched filter core.
//!
//! Ported from `matched_filter.cc` (`MatchedFilterCore_NEON` and
//! `MatchedFilterCoreWithAccumulatedError_NEON`).

use std::arch::aarch64::*;

/// NEON horizontal sum of a 128-bit register (4 floats → scalar).
///
/// # Safety
/// Requires NEON support (always available on aarch64).
#[target_feature(enable = "neon")]
unsafe fn sum_all_elements(elements: float32x4_t) -> f32 {
    unsafe {
        let sum = vpadd_f32(vget_low_f32(elements), vget_high_f32(elements));
        let sum = vpadd_f32(sum, sum);
        vget_lane_f32::<0>(sum)
    }
}

/// NEON matched filter core without accumulated error.
///
/// # Safety
/// Requires NEON support. `h.len()` must be divisible by 4.
#[target_feature(enable = "neon")]
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

            let mut s_128 = vdupq_n_f32(0.0);
            let mut x2_sum_128 = vdupq_n_f32(0.0);
            let mut x2_sum = 0.0f32;
            let mut s = 0.0f32;

            let chunk1 = h_size.min(x_size - x_start_index as i32);
            let chunk2 = h_size - chunk1;

            for limit in [chunk1, chunk2] {
                let limit_by_4 = limit >> 2;
                for _ in 0..limit_by_4 {
                    let x_k = vld1q_f32(x_p);
                    let h_k = vld1q_f32(h_p);
                    x2_sum_128 = vmlaq_f32(x2_sum_128, x_k, x_k);
                    s_128 = vmlaq_f32(s_128, h_k, x_k);
                    h_p = h_p.add(4);
                    x_p = x_p.add(4);
                }
                for _ in 0..(limit - limit_by_4 * 4) {
                    let x_k = *x_p;
                    x2_sum += x_k * x_k;
                    s += *h_p * x_k;
                    h_p = h_p.add(1);
                    x_p = x_p.add(1);
                }
                x_p = x.as_ptr();
            }

            s += sum_all_elements(s_128);
            x2_sum += sum_all_elements(x2_sum_128);

            let e = y[i] - s;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_128 = vmovq_n_f32(alpha);

                let mut h_p2 = h.as_mut_ptr();
                x_p = x.as_ptr().add(x_start_index);

                for limit in [chunk1, chunk2] {
                    let limit_by_4 = limit >> 2;
                    for _ in 0..limit_by_4 {
                        let mut h_k = vld1q_f32(h_p2);
                        let x_k = vld1q_f32(x_p);
                        h_k = vmlaq_f32(h_k, alpha_128, x_k);
                        vst1q_f32(h_p2, h_k);
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

/// NEON matched filter core with accumulated error computation.
///
/// # Safety
/// Requires NEON support. `h.len()` must be divisible by 4.
/// `scratch_memory.len()` must be >= `h.len()`.
#[target_feature(enable = "neon")]
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
        debug_assert_eq!(0, h_size % 4);

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
            let mut h_cp = h.as_ptr();
            let mut a_p = accumulated_error.as_mut_ptr();

            let mut x2_sum_128 = vdupq_n_f32(0.0);
            let mut x2_sum = 0.0f32;
            let mut s = 0.0f32;

            let limit_by_4 = h_size >> 2;
            for _ in 0..limit_by_4 {
                let x_k = vld1q_f32(x_p);
                let h_k = vld1q_f32(h_cp);
                x2_sum_128 = vmlaq_f32(x2_sum_128, x_k, x_k);

                let hk_xk = vmulq_f32(h_k, x_k);
                s += sum_all_elements(hk_xk);
                let e = s - y[i];
                *a_p += e * e;

                h_cp = h_cp.add(4);
                x_p = x_p.add(4);
                a_p = a_p.add(1);
            }

            x2_sum += sum_all_elements(x2_sum_128);

            let e = y[i] - s;
            let saturation = y[i] >= 32000.0 || y[i] <= -32000.0;
            *error_sum += e * e;

            if x2_sum > x2_sum_threshold && !saturation {
                debug_assert!(x2_sum > 0.0);
                let alpha = smoothing * e / x2_sum;
                let alpha_128 = vmovq_n_f32(alpha);

                let mut h_p = h.as_mut_ptr();
                let mut x_p2 = x_p_base;

                for _ in 0..limit_by_4 {
                    let mut h_k = vld1q_f32(h_p);
                    let x_k = vld1q_f32(x_p2);
                    h_k = vmlaq_f32(h_k, alpha_128, x_k);
                    vst1q_f32(h_p, h_k);
                    h_p = h_p.add(4);
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
