//! AVX2+FMA implementations of SIMD operations (x86/x86_64).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::SimdOps;

pub struct Avx2Ops;

impl SimdOps for Avx2Ops {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // Safety: caller guarantees AVX2+FMA is available via runtime detection
        unsafe { dot_product_avx2(a, b) }
    }

    fn dual_dot_product(&self, input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
        debug_assert_eq!(input.len(), k1.len());
        debug_assert_eq!(input.len(), k2.len());
        unsafe { dual_dot_product_avx2(input, k1, k2) }
    }

    fn multiply_accumulate(&self, acc: &mut [f32], a: &[f32], b: &[f32]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        unsafe { multiply_accumulate_avx2(acc, a, b) }
    }

    fn sum(&self, x: &[f32]) -> f32 {
        unsafe { sum_avx2(x) }
    }
}

/// AVX2+FMA dot product: processes 8 floats at a time with FMA.
///
/// Mirrors the pattern in fir_filter_avx2.cc.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut result = horizontal_sum_avx2(acc);

    // Handle remainder with scalar
    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

/// AVX2+FMA dual dot product for sinc resampler convolution.
#[target_feature(enable = "avx2,fma")]
unsafe fn dual_dot_product_avx2(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
    let len = input.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();

    let input_ptr = input.as_ptr();
    let k1_ptr = k1.as_ptr();
    let k2_ptr = k2.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vi = _mm256_loadu_ps(input_ptr.add(offset));
        let vk1 = _mm256_loadu_ps(k1_ptr.add(offset));
        let vk2 = _mm256_loadu_ps(k2_ptr.add(offset));
        acc1 = _mm256_fmadd_ps(vi, vk1, acc1);
        acc2 = _mm256_fmadd_ps(vi, vk2, acc2);
    }

    let mut sum1 = horizontal_sum_avx2(acc1);
    let mut sum2 = horizontal_sum_avx2(acc2);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum1 += input[idx] * k1[idx];
        sum2 += input[idx] * k2[idx];
    }

    (sum1, sum2)
}

/// AVX2+FMA multiply-accumulate: acc[i] += a[i] * b[i]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply_accumulate_avx2(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let len = acc.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let acc_ptr = acc.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vacc = _mm256_loadu_ps(acc_ptr.add(offset));
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let result = _mm256_fmadd_ps(va, vb, vacc);
        _mm256_storeu_ps(acc_ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc[idx] += a[idx] * b[idx];
    }
}

/// AVX2 sum of all elements.
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    let ptr = x.as_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }

    let mut result = horizontal_sum_avx2(acc);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += x[tail_start + i];
    }

    result
}

/// Reduce an __m256 to a scalar sum.
///
/// Mirrors the AVX2 reduction pattern in the C++ code:
/// extract 128-bit lanes, add, then SSE horizontal reduction.
#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);

    // SSE horizontal reduction
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let shuf = _mm_shuffle_ps(sum64, sum64, 1);
    let result = _mm_add_ss(sum64, shuf);
    _mm_cvtss_f32(result)
}
