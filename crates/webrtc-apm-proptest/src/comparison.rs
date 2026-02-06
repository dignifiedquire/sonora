//! Float comparison utilities for verifying Rust vs C++ output.

use std::fmt;

/// Result of comparing two audio buffers.
#[derive(Debug)]
pub struct ComparisonResult {
    pub max_abs_diff: f32,
    pub max_abs_diff_index: usize,
    pub mean_abs_diff: f32,
    pub mismatches: usize,
    pub total: usize,
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_abs_diff={} (at index {}), mean_abs_diff={}, mismatches={}/{}",
            self.max_abs_diff,
            self.max_abs_diff_index,
            self.mean_abs_diff,
            self.mismatches,
            self.total,
        )
    }
}

/// Compare two f32 slices, returning detailed statistics.
pub fn compare_f32(actual: &[f32], expected: &[f32], tolerance: f32) -> ComparisonResult {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    let total = actual.len();
    let mut max_abs_diff = 0.0f32;
    let mut max_abs_diff_index = 0;
    let mut sum_abs_diff = 0.0f64;
    let mut mismatches = 0;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        sum_abs_diff += diff as f64;
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_abs_diff_index = i;
        }
        if diff > tolerance {
            mismatches += 1;
        }
    }

    ComparisonResult {
        max_abs_diff,
        max_abs_diff_index,
        mean_abs_diff: if total > 0 {
            (sum_abs_diff / total as f64) as f32
        } else {
            0.0
        },
        mismatches,
        total,
    }
}

/// Assert two f32 slices are equal within absolute tolerance.
pub fn assert_f32_near(actual: &[f32], expected: &[f32], tolerance: f32) {
    let result = compare_f32(actual, expected, tolerance);
    assert!(
        result.mismatches == 0,
        "f32 comparison failed: {result}\n  actual[{}]={}, expected[{}]={}",
        result.max_abs_diff_index,
        actual[result.max_abs_diff_index],
        result.max_abs_diff_index,
        expected[result.max_abs_diff_index],
    );
}

/// Assert two f32 slices are equal within relative + absolute tolerance.
///
/// For each element pair, the tolerance is `max(abs_tol, |expected| * rel_tol)`.
pub fn assert_f32_relative(actual: &[f32], expected: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = abs_tol.max(e.abs() * rel_tol);
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "Mismatch at index {i}: actual={a}, expected={e}, diff={diff}, tol={tol}",
        );
    }
}

/// Assert two i16 slices are bit-exact.
pub fn assert_i16_exact(actual: &[i16], expected: &[i16]) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "Mismatch at index {i}: actual={a}, expected={e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_identical_passes() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_f32_near(&a, &b, 1e-6);
    }

    #[test]
    fn near_within_tolerance_passes() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.00001, 2.00001, 3.00001];
        assert_f32_near(&a, &b, 1e-4);
    }

    #[test]
    #[should_panic(expected = "f32 comparison failed")]
    fn near_beyond_tolerance_fails() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0, 2.1, 3.0];
        assert_f32_near(&a, &b, 1e-4);
    }

    #[test]
    fn compare_f32_statistics() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.1, 2.0, 3.05, 4.0];
        let result = compare_f32(&a, &b, 0.06);
        assert!((result.max_abs_diff - 0.1).abs() < 1e-6);
        assert_eq!(result.max_abs_diff_index, 0);
        assert_eq!(result.mismatches, 1); // only index 0 exceeds 0.06
        assert_eq!(result.total, 4);
    }

    #[test]
    fn relative_tolerance_scales() {
        // Large values get more tolerance
        let a = [0.001, 100.0];
        let b = [0.001, 100.05];
        // rel_tol=0.001 gives tol=0.1 for index 1, which covers 0.05
        assert_f32_relative(&a, &b, 0.001, 1e-6);
    }

    #[test]
    #[should_panic(expected = "Mismatch at index 0")]
    fn relative_small_value_tight() {
        let a = [0.001f32];
        let b = [0.01];
        // rel_tol=0.001 gives tol=max(1e-6, 0.01*0.001)=1e-5, diff=0.009
        assert_f32_relative(&a, &b, 0.001, 1e-6);
    }

    #[test]
    fn i16_exact_passes() {
        let a = [0i16, 1000, -1000, i16::MAX, i16::MIN];
        let b = [0i16, 1000, -1000, i16::MAX, i16::MIN];
        assert_i16_exact(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Mismatch at index 1")]
    fn i16_exact_detects_diff() {
        let a = [0i16, 1000];
        let b = [0i16, 1001];
        assert_i16_exact(&a, &b);
    }

    #[test]
    fn empty_slices_pass() {
        assert_f32_near(&[], &[], 1e-6);
        assert_i16_exact(&[], &[]);
    }
}
