//! SSE2 implementations of SIMD operations (x86/x86_64).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 dot product: processes 4 floats at a time.
///
/// # Safety
/// Caller must ensure SSE2 is available (via `is_x86_feature_detected!`).
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    let mut result = horizontal_sum(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

/// SSE2 dual dot product for sinc resampler convolution.
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn dual_dot_product(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
    let len = input.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();

    let input_ptr = input.as_ptr();
    let k1_ptr = k1.as_ptr();
    let k2_ptr = k2.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vi = _mm_loadu_ps(input_ptr.add(offset));
        let vk1 = _mm_loadu_ps(k1_ptr.add(offset));
        let vk2 = _mm_loadu_ps(k2_ptr.add(offset));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(vi, vk1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(vi, vk2));
    }

    let mut sum1 = horizontal_sum(acc1);
    let mut sum2 = horizontal_sum(acc2);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum1 += input[idx] * k1[idx];
        sum2 += input[idx] * k2[idx];
    }

    (sum1, sum2)
}

/// SSE2 multiply-accumulate: acc[i] += a[i] * b[i]
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn multiply_accumulate(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let len = acc.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let acc_ptr = acc.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vacc = _mm_loadu_ps(acc_ptr.add(offset));
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let result = _mm_add_ps(vacc, _mm_mul_ps(va, vb));
        _mm_storeu_ps(acc_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc[idx] += a[idx] * b[idx];
    }
}

/// SSE2 sum of all elements.
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn sum(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = _mm_setzero_ps();
    let ptr = x.as_ptr();

    for i in 0..chunks {
        let v = _mm_loadu_ps(ptr.add(i * 4));
        acc = _mm_add_ps(acc, v);
    }

    let mut result = horizontal_sum(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += x[tail_start + i];
    }

    result
}

/// Reduce an __m128 to a scalar sum.
#[inline(always)]
#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum(v: __m128) -> f32 {
    let hi = _mm_movehl_ps(v, v);
    let sum = _mm_add_ps(v, hi);
    let shuf = _mm_shuffle_ps(sum, sum, 1);
    let result = _mm_add_ss(sum, shuf);
    _mm_cvtss_f32(result)
}
