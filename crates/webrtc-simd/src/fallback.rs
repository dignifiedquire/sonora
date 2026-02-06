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
