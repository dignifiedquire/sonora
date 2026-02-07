//! AVX2+FMA implementations of SIMD operations (x86/x86_64).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2+FMA dot product: processes 8 floats at a time with FMA.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (via `is_x86_feature_detected!`).
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
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

    let mut result = horizontal_sum(acc);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

/// AVX2+FMA dual dot product for sinc resampler convolution.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dual_dot_product(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
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

    let mut sum1 = horizontal_sum(acc1);
    let mut sum2 = horizontal_sum(acc2);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum1 += input[idx] * k1[idx];
        sum2 += input[idx] * k2[idx];
    }

    (sum1, sum2)
}

/// AVX2+FMA multiply-accumulate: acc[i] += a[i] * b[i]
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn multiply_accumulate(acc: &mut [f32], a: &[f32], b: &[f32]) {
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
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sum(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();
    let ptr = x.as_ptr();

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }

    let mut result = horizontal_sum(acc);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += x[tail_start + i];
    }

    result
}

/// AVX2 elementwise square root: x[i] = sqrt(x[i])
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn elementwise_sqrt(x: &mut [f32]) {
    let len = x.len();
    let chunks = len / 8;
    let remainder = len % 8;
    let ptr = x.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(ptr.add(offset));
        let result = _mm256_sqrt_ps(v);
        _mm256_storeu_ps(ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        x[tail_start + i] = x[tail_start + i].sqrt();
    }
}

/// AVX2 elementwise multiply: z[i] = x[i] * y[i]
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn elementwise_multiply(x: &[f32], y: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x_ptr.add(offset));
        let vy = _mm256_loadu_ps(y_ptr.add(offset));
        let result = _mm256_mul_ps(vx, vy);
        _mm256_storeu_ps(z_ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] = x[idx] * y[idx];
    }
}

/// AVX2 elementwise accumulate: z[i] += x[i]
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn elementwise_accumulate(x: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let x_ptr = x.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x_ptr.add(offset));
        let vz = _mm256_loadu_ps(z_ptr.add(offset));
        let result = _mm256_add_ps(vz, vx);
        _mm256_storeu_ps(z_ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] += x[idx];
    }
}

/// AVX2 power spectrum: out[i] = re[i]^2 + im[i]^2
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn power_spectrum(re: &[f32], im: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let re_ptr = re.as_ptr();
    let im_ptr = im.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vr = _mm256_loadu_ps(re_ptr.add(offset));
        let vi = _mm256_loadu_ps(im_ptr.add(offset));
        let rr = _mm256_mul_ps(vr, vr);
        let ii = _mm256_mul_ps(vi, vi);
        let result = _mm256_add_ps(rr, ii);
        _mm256_storeu_ps(out_ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = re[idx] * re[idx] + im[idx] * im[idx];
    }
}

/// AVX2 elementwise min: out[i] = min(a[i], b[i])
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn elementwise_min(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let result = _mm256_min_ps(va, vb);
        _mm256_storeu_ps(out_ptr.add(offset), result);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = a[idx].min(b[idx]);
    }
}

/// AVX2+FMA complex multiply-accumulate (AEC3 conjugate convention):
///   acc_re[i] += x_re[i]*h_re[i] + x_im[i]*h_im[i]
///   acc_im[i] += x_re[i]*h_im[i] - x_im[i]*h_re[i]
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn complex_multiply_accumulate(
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
    acc_re: &mut [f32],
    acc_im: &mut [f32],
) {
    let len = acc_re.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let xr_ptr = x_re.as_ptr();
    let xi_ptr = x_im.as_ptr();
    let hr_ptr = h_re.as_ptr();
    let hi_ptr = h_im.as_ptr();
    let ar_ptr = acc_re.as_mut_ptr();
    let ai_ptr = acc_im.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vxr = _mm256_loadu_ps(xr_ptr.add(offset));
        let vxi = _mm256_loadu_ps(xi_ptr.add(offset));
        let vhr = _mm256_loadu_ps(hr_ptr.add(offset));
        let vhi = _mm256_loadu_ps(hi_ptr.add(offset));

        // acc_re += x_re*h_re + x_im*h_im (two FMAs)
        let mut var = _mm256_loadu_ps(ar_ptr.add(offset));
        var = _mm256_fmadd_ps(vxr, vhr, var);
        var = _mm256_fmadd_ps(vxi, vhi, var);
        _mm256_storeu_ps(ar_ptr.add(offset), var);

        // acc_im += x_re*h_im - x_im*h_re (FMA + FNMA)
        let mut vai = _mm256_loadu_ps(ai_ptr.add(offset));
        vai = _mm256_fmadd_ps(vxr, vhi, vai);
        vai = _mm256_fnmadd_ps(vxi, vhr, vai);
        _mm256_storeu_ps(ai_ptr.add(offset), vai);
    }

    let tail_start = chunks * 8;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc_re[idx] += x_re[idx] * h_re[idx] + x_im[idx] * h_im[idx];
        acc_im[idx] += x_re[idx] * h_im[idx] - x_im[idx] * h_re[idx];
    }
}

/// Reduce an __m256 to a scalar sum.
#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);

    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let shuf = _mm_shuffle_ps(sum64, sum64, 1);
    let result = _mm_add_ss(sum64, shuf);
    _mm_cvtss_f32(result)
}
