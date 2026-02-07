//! Scalar fallback implementations of SIMD operations.

pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(crate) fn dual_dot_product(input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    for i in 0..input.len() {
        sum1 += input[i] * k1[i];
        sum2 += input[i] * k2[i];
    }
    (sum1, sum2)
}

pub(crate) fn multiply_accumulate(acc: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..acc.len() {
        acc[i] += a[i] * b[i];
    }
}

pub(crate) fn sum(x: &[f32]) -> f32 {
    x.iter().sum()
}

pub(crate) fn elementwise_sqrt(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.sqrt();
    }
}

pub(crate) fn elementwise_multiply(x: &[f32], y: &[f32], z: &mut [f32]) {
    for i in 0..z.len() {
        z[i] = x[i] * y[i];
    }
}

pub(crate) fn elementwise_accumulate(x: &[f32], z: &mut [f32]) {
    for i in 0..z.len() {
        z[i] += x[i];
    }
}

pub(crate) fn power_spectrum(re: &[f32], im: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = re[i] * re[i] + im[i] * im[i];
    }
}

pub(crate) fn elementwise_min(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i].min(b[i]);
    }
}

pub(crate) fn elementwise_max(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i].max(b[i]);
    }
}

pub(crate) fn complex_multiply_accumulate(
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
    acc_re: &mut [f32],
    acc_im: &mut [f32],
) {
    for i in 0..acc_re.len() {
        // (x_re + j*x_im) * (h_re + j*h_im) =
        //   (x_re*h_re - x_im*h_im) + j*(x_re*h_im + x_im*h_re)
        // Note: AEC3 uses conjugate multiply convention (+ for real, - for imag):
        //   real = x_re*h_re + x_im*h_im
        //   imag = x_re*h_im - x_im*h_re
        acc_re[i] += x_re[i] * h_re[i] + x_im[i] * h_im[i];
        acc_im[i] += x_re[i] * h_im[i] - x_im[i] * h_re[i];
    }
}

pub(crate) fn complex_multiply_accumulate_standard(
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
    acc_re: &mut [f32],
    acc_im: &mut [f32],
) {
    for i in 0..acc_re.len() {
        // Standard complex multiply: (x_re + j*x_im) * (h_re + j*h_im)
        //   real = x_re*h_re - x_im*h_im
        //   imag = x_re*h_im + x_im*h_re
        acc_re[i] += x_re[i] * h_re[i] - x_im[i] * h_im[i];
        acc_im[i] += x_re[i] * h_im[i] + x_im[i] * h_re[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_empty() {
        assert_eq!(dot_product(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        assert_eq!(dot_product(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn test_sum_empty() {
        assert_eq!(sum(&[]), 0.0);
    }
}
