//! NEON implementations of SIMD operations (aarch64).

use std::arch::aarch64::*;

/// NEON dot product: processes 4 floats at a time with vmlaq_f32.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[inline]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            acc = vmlaq_f32(acc, va, vb);
        }
    }

    let mut result = unsafe { horizontal_sum(acc) };

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

/// NEON dual dot product for sinc resampler convolution.
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub unsafe fn dual_dot_product(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
    let len = input.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let input_ptr = input.as_ptr();
    let k1_ptr = k1.as_ptr();
    let k2_ptr = k2.as_ptr();

    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vi = vld1q_f32(input_ptr.add(offset));
            let vk1 = vld1q_f32(k1_ptr.add(offset));
            let vk2 = vld1q_f32(k2_ptr.add(offset));
            acc1 = vmlaq_f32(acc1, vi, vk1);
            acc2 = vmlaq_f32(acc2, vi, vk2);
        }
    }

    let mut sum1 = unsafe { horizontal_sum(acc1) };
    let mut sum2 = unsafe { horizontal_sum(acc2) };

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum1 += input[idx] * k1[idx];
        sum2 += input[idx] * k2[idx];
    }

    (sum1, sum2)
}

/// NEON multiply-accumulate: acc[i] += a[i] * b[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub unsafe fn multiply_accumulate(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let len = acc.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let acc_ptr = acc.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vacc = vld1q_f32(acc_ptr.add(offset));
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let result = vmlaq_f32(vacc, va, vb);
            vst1q_f32(acc_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc[idx] += a[idx] * b[idx];
    }
}

/// NEON sum of all elements.
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub unsafe fn sum(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let ptr = x.as_ptr();

    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }
    }

    let mut result = unsafe { horizontal_sum(acc) };

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += x[tail_start + i];
    }

    result
}

/// Reduce a float32x4_t to a scalar sum.
#[inline(always)]
unsafe fn horizontal_sum(v: float32x4_t) -> f32 {
    unsafe {
        let sum_pair = vadd_f32(vget_high_f32(v), vget_low_f32(v));
        let sum = vpadd_f32(sum_pair, sum_pair);
        vget_lane_f32::<0>(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_dot_product() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = unsafe { dot_product(&a, &b) };
        assert!((result - 120.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_dot_product_non_aligned() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let result = unsafe { dot_product(&a, &b) };
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_horizontal_sum() {
        unsafe {
            let v = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            assert!((horizontal_sum(v) - 10.0).abs() < 1e-6);
        }
    }
}
