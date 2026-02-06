//! Scalar fallback implementations of SIMD operations.

use crate::SimdOps;

pub struct ScalarOps;

impl SimdOps for ScalarOps {
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn dual_dot_product(&self, input: &[f32], k1: &[f32], k2: &[f32]) -> (f32, f32) {
        debug_assert_eq!(input.len(), k1.len());
        debug_assert_eq!(input.len(), k2.len());
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        for i in 0..input.len() {
            sum1 += input[i] * k1[i];
            sum2 += input[i] * k2[i];
        }
        (sum1, sum2)
    }

    fn multiply_accumulate(&self, acc: &mut [f32], a: &[f32], b: &[f32]) {
        debug_assert_eq!(acc.len(), a.len());
        debug_assert_eq!(acc.len(), b.len());
        for i in 0..acc.len() {
            acc[i] += a[i] * b[i];
        }
    }

    fn sum(&self, x: &[f32]) -> f32 {
        x.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_empty() {
        assert_eq!(ScalarOps.dot_product(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        assert_eq!(ScalarOps.dot_product(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    fn test_sum_empty() {
        assert_eq!(ScalarOps.sum(&[]), 0.0);
    }
}
