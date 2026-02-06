//! NEON implementations of SIMD operations (aarch64).

use crate::SimdOps;
use std::arch::aarch64::*;

pub struct NeonOps;

impl SimdOps for NeonOps {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // Safety: we're on aarch64, NEON is always available
        unsafe { dot_product_neon(a, b) }
    }

    fn dual_dot_product(&self, input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
        debug_assert_eq!(input.len(), k1.len());
        debug_assert_eq!(input.len(), k2.len());
        unsafe { dual_dot_product_neon(input, k1, k2) }
    }

    fn multiply_accumulate(&self, acc: &mut [f32], a: &[f32], b: &[f32]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        unsafe { multiply_accumulate_neon(acc, a, b) }
    }

    fn sum(&self, x: &[f32]) -> f32 {
        unsafe { sum_neon(x) }
    }
}

/// NEON dot product: processes 4 floats at a time with vmlaq_f32.
///
/// Mirrors the pattern in fir_filter_neon.cc and sinc_resampler_neon.cc.
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        acc = vmlaq_f32(acc, va, vb);
    }

    // Horizontal sum: reduce 4 lanes to scalar
    let mut result = horizontal_sum_f32(acc);

    // Handle remainder
    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

/// NEON dual dot product for sinc resampler convolution.
///
/// Computes dot(input, k1) and dot(input, k2) simultaneously.
#[inline]
unsafe fn dual_dot_product_neon(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
    let len = input.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);

    let input_ptr = input.as_ptr();
    let k1_ptr = k1.as_ptr();
    let k2_ptr = k2.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vi = vld1q_f32(input_ptr.add(offset));
        let vk1 = vld1q_f32(k1_ptr.add(offset));
        let vk2 = vld1q_f32(k2_ptr.add(offset));
        acc1 = vmlaq_f32(acc1, vi, vk1);
        acc2 = vmlaq_f32(acc2, vi, vk2);
    }

    let mut sum1 = horizontal_sum_f32(acc1);
    let mut sum2 = horizontal_sum_f32(acc2);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum1 += input[idx] * k1[idx];
        sum2 += input[idx] * k2[idx];
    }

    (sum1, sum2)
}

/// NEON multiply-accumulate: acc[i] += a[i] * b[i]
#[inline]
unsafe fn multiply_accumulate_neon(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let len = acc.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let acc_ptr = acc.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vacc = vld1q_f32(acc_ptr.add(offset));
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let result = vmlaq_f32(vacc, va, vb);
        vst1q_f32(acc_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc[idx] += a[idx] * b[idx];
    }
}

/// NEON sum of all elements.
#[inline]
unsafe fn sum_neon(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    let ptr = x.as_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vaddq_f32(acc, v);
    }

    let mut result = horizontal_sum_f32(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += x[tail_start + i];
    }

    result
}

/// Reduce a float32x4_t to a scalar sum.
///
/// Mirrors the NEON reduction pattern used in the C++ code:
/// vadd_f32(vget_high_f32(v), vget_low_f32(v)) -> vpadd_f32 -> vget_lane_f32
#[inline(always)]
unsafe fn horizontal_sum_f32(v: float32x4_t) -> f32 {
    let sum_pair = vadd_f32(vget_high_f32(v), vget_low_f32(v));
    let sum = vpadd_f32(sum_pair, sum_pair);
    vget_lane_f32::<0>(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_dot_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = NeonOps.dot_product(&a, &b);
        // 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
        assert!((result - 120.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_dot_product_non_aligned() {
        // 5 elements: 4 NEON + 1 scalar remainder
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let result = NeonOps.dot_product(&a, &b);
        // 2 + 6 + 12 + 20 + 30 = 70
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_horizontal_sum() {
        unsafe {
            let v = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            assert!((horizontal_sum_f32(v) - 10.0).abs() < 1e-6);
        }
    }
}
