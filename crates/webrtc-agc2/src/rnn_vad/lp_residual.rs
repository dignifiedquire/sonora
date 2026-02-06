//! Linear prediction residual computation.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/lp_residual.cc`.

/// Number of LPC coefficients.
pub const NUM_LPC_COEFFICIENTS: usize = 5;

/// Computes and post-processes LPC coefficients tailored for pitch estimation.
pub fn compute_and_post_process_lpc_coefficients(
    x: &[f32],
    lpc_coeffs: &mut [f32; NUM_LPC_COEFFICIENTS],
) {
    let mut auto_corr = [0.0_f32; NUM_LPC_COEFFICIENTS];
    compute_auto_correlation(x, &mut auto_corr);

    if auto_corr[0] == 0.0 {
        // Empty frame.
        lpc_coeffs.fill(0.0);
        return;
    }

    denoise_auto_correlation(&mut auto_corr);
    let mut lpc_coeffs_pre = [0.0_f32; NUM_LPC_COEFFICIENTS - 1];
    compute_initial_inverse_filter_coefficients(&auto_corr, &mut lpc_coeffs_pre);

    // LPC coefficients post-processing.
    lpc_coeffs_pre[0] *= 0.9;
    lpc_coeffs_pre[1] *= 0.9 * 0.9;
    lpc_coeffs_pre[2] *= 0.9 * 0.9 * 0.9;
    lpc_coeffs_pre[3] *= 0.9 * 0.9 * 0.9 * 0.9;

    const C: f32 = 0.8;
    lpc_coeffs[0] = lpc_coeffs_pre[0] + C;
    lpc_coeffs[1] = lpc_coeffs_pre[1] + C * lpc_coeffs_pre[0];
    lpc_coeffs[2] = lpc_coeffs_pre[2] + C * lpc_coeffs_pre[1];
    lpc_coeffs[3] = lpc_coeffs_pre[3] + C * lpc_coeffs_pre[2];
    lpc_coeffs[4] = C * lpc_coeffs_pre[3];
}

/// Computes the LP residual for the input frame `x` and the LPC coefficients.
///
/// `y` and `x` can point to the same slice for in-place computation.
pub fn compute_lp_residual(lpc_coeffs: &[f32; NUM_LPC_COEFFICIENTS], x: &[f32], y: &mut [f32]) {
    debug_assert!(x.len() > NUM_LPC_COEFFICIENTS);
    debug_assert_eq!(x.len(), y.len());

    // y[i] = x[i] + sum(lpc_coeffs[k] * x[i-1-k] for k in 0..NUM_LPC_COEFFICIENTS)
    // Edge case: i < NUM_LPC_COEFFICIENTS.
    y[0] = x[0];
    for i in 1..NUM_LPC_COEFFICIENTS {
        let mut sum = x[i];
        for k in 0..i {
            sum += lpc_coeffs[k] * x[i - 1 - k];
        }
        y[i] = sum;
    }
    // Regular case.
    for i in NUM_LPC_COEFFICIENTS..x.len() {
        let mut sum = x[i];
        for k in 0..NUM_LPC_COEFFICIENTS {
            sum += lpc_coeffs[k] * x[i - 1 - k];
        }
        y[i] = sum;
    }
}

/// Computes auto-correlation coefficients for `x`.
fn compute_auto_correlation(x: &[f32], auto_corr: &mut [f32; NUM_LPC_COEFFICIENTS]) {
    debug_assert!(x.len() > NUM_LPC_COEFFICIENTS);
    for lag in 0..NUM_LPC_COEFFICIENTS {
        auto_corr[lag] = x[..x.len() - lag]
            .iter()
            .zip(x[lag..].iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }
}

/// Applies denoising to the auto-correlation coefficients.
fn denoise_auto_correlation(auto_corr: &mut [f32; NUM_LPC_COEFFICIENTS]) {
    // Assume -40 dB white noise floor.
    auto_corr[0] *= 1.0001;
    // Hard-coded values: [np.float32((0.008*0.008*i*i)) for i in range(1,5)]
    auto_corr[1] -= auto_corr[1] * 0.000064;
    auto_corr[2] -= auto_corr[2] * 0.000256;
    auto_corr[3] -= auto_corr[3] * 0.000576;
    auto_corr[4] -= auto_corr[4] * 0.001024;
}

/// Computes the initial inverse filter coefficients given auto-correlation coefficients.
/// Uses the Levinson-Durbin algorithm.
fn compute_initial_inverse_filter_coefficients(
    auto_corr: &[f32; NUM_LPC_COEFFICIENTS],
    lpc_coeffs: &mut [f32; NUM_LPC_COEFFICIENTS - 1],
) {
    let mut error = auto_corr[0];
    for i in 0..NUM_LPC_COEFFICIENTS - 1 {
        let mut reflection_coeff = 0.0_f32;
        for j in 0..i {
            reflection_coeff += lpc_coeffs[j] * auto_corr[i - j];
        }
        reflection_coeff += auto_corr[i + 1];

        // Avoid division by numbers close to zero.
        const MIN_ERROR_MAGNITUDE: f32 = 1e-6;
        if error.abs() < MIN_ERROR_MAGNITUDE {
            error = error.signum() * MIN_ERROR_MAGNITUDE;
        }

        reflection_coeff /= -error;
        // Update LPC coefficients and total error.
        lpc_coeffs[i] = reflection_coeff;
        for j in 0..((i + 1) >> 1) {
            let tmp1 = lpc_coeffs[j];
            let tmp2 = lpc_coeffs[i - 1 - j];
            lpc_coeffs[j] = tmp1 + reflection_coeff * tmp2;
            lpc_coeffs[i - 1 - j] = tmp2 + reflection_coeff * tmp1;
        }
        error -= reflection_coeff * reflection_coeff * error;
        if error < 0.001 * auto_corr[0] {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_residual_of_empty_frame() {
        let frame = [0.0_f32; 320];
        let mut lpc_coeffs = [0.0_f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(&frame, &mut lpc_coeffs);
        assert_eq!(lpc_coeffs, [0.0; NUM_LPC_COEFFICIENTS]);

        let mut residual = vec![0.0_f32; 320];
        compute_lp_residual(&lpc_coeffs, &frame, &mut residual);
        for &r in &residual {
            assert_eq!(r, 0.0);
        }
    }

    #[test]
    fn lp_residual_of_unit_impulse() {
        // A frame with an impulse at the start should produce non-trivial LPC.
        let mut frame = [0.0_f32; 320];
        frame[0] = 1.0;
        let mut lpc_coeffs = [0.0_f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(&frame, &mut lpc_coeffs);
        // LPC coeffs should not all be zero (the frame has energy).
        assert!(lpc_coeffs.iter().any(|&c| c != 0.0));

        let mut residual = vec![0.0_f32; 320];
        compute_lp_residual(&lpc_coeffs, &frame, &mut residual);
        // First element should be the impulse value.
        assert_eq!(residual[0], 1.0);
    }

    #[test]
    fn lp_residual_preserves_length() {
        let frame = vec![1.0_f32; 480];
        let mut lpc_coeffs = [0.0_f32; NUM_LPC_COEFFICIENTS];
        compute_and_post_process_lpc_coefficients(&frame, &mut lpc_coeffs);

        let mut residual = vec![0.0_f32; 480];
        compute_lp_residual(&lpc_coeffs, &frame, &mut residual);
        assert_eq!(residual.len(), 480);
    }
}
