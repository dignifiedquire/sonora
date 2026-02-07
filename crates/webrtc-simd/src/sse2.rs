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

/// SSE2 elementwise square root: x[i] = sqrt(x[i])
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn elementwise_sqrt(x: &mut [f32]) {
    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let ptr = x.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let v = _mm_loadu_ps(ptr.add(offset));
        let result = _mm_sqrt_ps(v);
        _mm_storeu_ps(ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        x[tail_start + i] = x[tail_start + i].sqrt();
    }
}

/// SSE2 elementwise multiply: z[i] = x[i] * y[i]
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn elementwise_multiply(x: &[f32], y: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vx = _mm_loadu_ps(x_ptr.add(offset));
        let vy = _mm_loadu_ps(y_ptr.add(offset));
        let result = _mm_mul_ps(vx, vy);
        _mm_storeu_ps(z_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] = x[idx] * y[idx];
    }
}

/// SSE2 elementwise accumulate: z[i] += x[i]
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn elementwise_accumulate(x: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let x_ptr = x.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vx = _mm_loadu_ps(x_ptr.add(offset));
        let vz = _mm_loadu_ps(z_ptr.add(offset));
        let result = _mm_add_ps(vz, vx);
        _mm_storeu_ps(z_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] += x[idx];
    }
}

/// SSE2 power spectrum: out[i] = re[i]^2 + im[i]^2
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn power_spectrum(re: &[f32], im: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let re_ptr = re.as_ptr();
    let im_ptr = im.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vr = _mm_loadu_ps(re_ptr.add(offset));
        let vi = _mm_loadu_ps(im_ptr.add(offset));
        let rr = _mm_mul_ps(vr, vr);
        let ii = _mm_mul_ps(vi, vi);
        let result = _mm_add_ps(rr, ii);
        _mm_storeu_ps(out_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = re[idx] * re[idx] + im[idx] * im[idx];
    }
}

/// SSE2 elementwise min: out[i] = min(a[i], b[i])
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn elementwise_min(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let result = _mm_min_ps(va, vb);
        _mm_storeu_ps(out_ptr.add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = a[idx].min(b[idx]);
    }
}

/// SSE2 complex multiply-accumulate (AEC3 conjugate convention):
///   acc_re[i] += x_re[i]*h_re[i] + x_im[i]*h_im[i]
///   acc_im[i] += x_re[i]*h_im[i] - x_im[i]*h_re[i]
///
/// # Safety
/// Caller must ensure SSE2 is available.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn complex_multiply_accumulate(
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
    acc_re: &mut [f32],
    acc_im: &mut [f32],
) {
    let len = acc_re.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let xr_ptr = x_re.as_ptr();
    let xi_ptr = x_im.as_ptr();
    let hr_ptr = h_re.as_ptr();
    let hi_ptr = h_im.as_ptr();
    let ar_ptr = acc_re.as_mut_ptr();
    let ai_ptr = acc_im.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let vxr = _mm_loadu_ps(xr_ptr.add(offset));
        let vxi = _mm_loadu_ps(xi_ptr.add(offset));
        let vhr = _mm_loadu_ps(hr_ptr.add(offset));
        let vhi = _mm_loadu_ps(hi_ptr.add(offset));

        // acc_re += x_re*h_re + x_im*h_im
        let var = _mm_loadu_ps(ar_ptr.add(offset));
        let re_part = _mm_add_ps(_mm_mul_ps(vxr, vhr), _mm_mul_ps(vxi, vhi));
        _mm_storeu_ps(ar_ptr.add(offset), _mm_add_ps(var, re_part));

        // acc_im += x_re*h_im - x_im*h_re
        let vai = _mm_loadu_ps(ai_ptr.add(offset));
        let im_part = _mm_sub_ps(_mm_mul_ps(vxr, vhi), _mm_mul_ps(vxi, vhr));
        _mm_storeu_ps(ai_ptr.add(offset), _mm_add_ps(vai, im_part));
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc_re[idx] += x_re[idx] * h_re[idx] + x_im[idx] * h_im[idx];
        acc_im[idx] += x_re[idx] * h_im[idx] - x_im[idx] * h_re[idx];
    }
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
