//! NEON implementations of SIMD operations (aarch64).

use std::arch::aarch64::*;

/// NEON dot product: processes 4 floats at a time with vmlaq_f32.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[inline]
pub(crate) unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
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
pub(crate) unsafe fn dual_dot_product(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
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

/// NEON sinc resampler convolution: dual dot product with vector interpolation.
///
/// Matches C++ `Convolve_NEON`: interpolation happens on float32x4_t vectors
/// using `vmlaq_f32` *before* horizontal reduction.
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn convolve_sinc(
    input: &[f32],
    k1: &[f32],
    k2: &[f32],
    kernel_interpolation_factor: f64,
) -> f32 {
    let len = input.len();
    let chunks = len / 4;

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

    // Linearly interpolate on vectors before horizontal reduction.
    // Matches C++ Convolve_NEON:
    //   m_sums1 = vmlaq_f32(
    //       vmulq_f32(m_sums1, vmovq_n_f32(1.0 - kernel_interpolation_factor)),
    //       m_sums2, vmovq_n_f32(kernel_interpolation_factor));
    let factor = kernel_interpolation_factor as f32;
    unsafe {
        acc1 = vmlaq_f32(
            vmulq_f32(acc1, vdupq_n_f32(1.0 - factor)),
            acc2,
            vdupq_n_f32(factor),
        );
    }

    let mut result = unsafe { horizontal_sum(acc1) };

    // Scalar tail (KERNEL_SIZE=32 is divisible by 4, so never reached).
    let tail_start = chunks * 4;
    let remainder = len % 4;
    if remainder > 0 {
        for i in 0..remainder {
            let idx = tail_start + i;
            result += (1.0 - factor) * input[idx] * k1[idx] + factor * input[idx] * k2[idx];
        }
    }

    result
}

/// NEON multiply-accumulate: acc[i] += a[i] * b[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn multiply_accumulate(acc: &mut [f32], a: &[f32], b: &[f32]) {
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
pub(crate) unsafe fn sum(x: &[f32]) -> f32 {
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

/// NEON elementwise square root: x[i] = sqrt(x[i])
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn elementwise_sqrt(x: &mut [f32]) {
    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let ptr = x.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let v = vld1q_f32(ptr.add(offset));
            let result = vsqrtq_f32(v);
            vst1q_f32(ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        x[tail_start + i] = x[tail_start + i].sqrt();
    }
}

/// NEON elementwise multiply: z[i] = x[i] * y[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn elementwise_multiply(x: &[f32], y: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vx = vld1q_f32(x_ptr.add(offset));
            let vy = vld1q_f32(y_ptr.add(offset));
            let result = vmulq_f32(vx, vy);
            vst1q_f32(z_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] = x[idx] * y[idx];
    }
}

/// NEON elementwise accumulate: z[i] += x[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn elementwise_accumulate(x: &[f32], z: &mut [f32]) {
    let len = z.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let x_ptr = x.as_ptr();
    let z_ptr = z.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vx = vld1q_f32(x_ptr.add(offset));
            let vz = vld1q_f32(z_ptr.add(offset));
            let result = vaddq_f32(vz, vx);
            vst1q_f32(z_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        z[idx] += x[idx];
    }
}

/// NEON power spectrum: out[i] = re[i]^2 + im[i]^2
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn power_spectrum(re: &[f32], im: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let re_ptr = re.as_ptr();
    let im_ptr = im.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vr = vld1q_f32(re_ptr.add(offset));
            let vi = vld1q_f32(im_ptr.add(offset));
            let rr = vmulq_f32(vr, vr);
            let result = vmlaq_f32(rr, vi, vi);
            vst1q_f32(out_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = re[idx] * re[idx] + im[idx] * im[idx];
    }
}

/// NEON elementwise min: out[i] = min(a[i], b[i])
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn elementwise_min(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let result = vminq_f32(va, vb);
            vst1q_f32(out_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = a[idx].min(b[idx]);
    }
}

/// NEON elementwise max: out[i] = max(a[i], b[i])
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn elementwise_max(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = out.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let result = vmaxq_f32(va, vb);
            vst1q_f32(out_ptr.add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        out[idx] = a[idx].max(b[idx]);
    }
}

/// NEON complex multiply-accumulate (AEC3 conjugate convention):
///   acc_re[i] += x_re[i]*h_re[i] + x_im[i]*h_im[i]
///   acc_im[i] += x_re[i]*h_im[i] - x_im[i]*h_re[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
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
        unsafe {
            let vxr = vld1q_f32(xr_ptr.add(offset));
            let vxi = vld1q_f32(xi_ptr.add(offset));
            let vhr = vld1q_f32(hr_ptr.add(offset));
            let vhi = vld1q_f32(hi_ptr.add(offset));

            // acc_re += x_re*h_re + x_im*h_im
            let mut var = vld1q_f32(ar_ptr.add(offset));
            var = vmlaq_f32(var, vxr, vhr);
            var = vmlaq_f32(var, vxi, vhi);
            vst1q_f32(ar_ptr.add(offset), var);

            // acc_im += x_re*h_im - x_im*h_re
            let mut vai = vld1q_f32(ai_ptr.add(offset));
            vai = vmlaq_f32(vai, vxr, vhi);
            vai = vmlsq_f32(vai, vxi, vhr);
            vst1q_f32(ai_ptr.add(offset), vai);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc_re[idx] += x_re[idx] * h_re[idx] + x_im[idx] * h_im[idx];
        acc_im[idx] += x_re[idx] * h_im[idx] - x_im[idx] * h_re[idx];
    }
}

/// NEON standard complex multiply-accumulate:
///   acc_re[i] += x_re[i]*h_re[i] - x_im[i]*h_im[i]
///   acc_im[i] += x_re[i]*h_im[i] + x_im[i]*h_re[i]
///
/// # Safety
/// Caller must ensure NEON is available.
#[inline]
pub(crate) unsafe fn complex_multiply_accumulate_standard(
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
        unsafe {
            let vxr = vld1q_f32(xr_ptr.add(offset));
            let vxi = vld1q_f32(xi_ptr.add(offset));
            let vhr = vld1q_f32(hr_ptr.add(offset));
            let vhi = vld1q_f32(hi_ptr.add(offset));

            // acc_re += x_re*h_re - x_im*h_im
            let mut var = vld1q_f32(ar_ptr.add(offset));
            var = vmlaq_f32(var, vxr, vhr);
            var = vmlsq_f32(var, vxi, vhi);
            vst1q_f32(ar_ptr.add(offset), var);

            // acc_im += x_re*h_im + x_im*h_re
            let mut vai = vld1q_f32(ai_ptr.add(offset));
            vai = vmlaq_f32(vai, vxr, vhi);
            vai = vmlaq_f32(vai, vxi, vhr);
            vst1q_f32(ai_ptr.add(offset), vai);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        acc_re[idx] += x_re[idx] * h_re[idx] - x_im[idx] * h_im[idx];
        acc_im[idx] += x_re[idx] * h_im[idx] + x_im[idx] * h_re[idx];
    }
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
