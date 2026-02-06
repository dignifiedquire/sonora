//! FFT-based auto-correlation computation on the pitch buffer.
//!
//! Ported from `webrtc/modules/audio_processing/agc2/rnn_vad/auto_correlation.cc`.

use std::fmt;

use super::common::{BUF_SIZE_12K_HZ, MAX_PITCH_12K_HZ, NUM_LAGS_12K_HZ};
use webrtc_fft::pffft::{FftType, Pffft, PffftBuffer};

/// FFT order for auto-correlation (length-512 FFT).
const AUTO_CORRELATION_FFT_ORDER: u32 = 9;

/// Class to compute the auto-correlation on the pitch buffer for a target
/// pitch interval.
pub struct AutoCorrelationCalculator {
    fft: Pffft,
    tmp: PffftBuffer,
    tmp2: PffftBuffer,
    x_buf: PffftBuffer,
    h_buf: PffftBuffer,
}

impl fmt::Debug for AutoCorrelationCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutoCorrelationCalculator").finish()
    }
}

impl Default for AutoCorrelationCalculator {
    fn default() -> Self {
        let fft_size = 1 << AUTO_CORRELATION_FFT_ORDER;
        let fft = Pffft::new(fft_size, FftType::Real);
        let tmp = fft.create_buffer();
        let tmp2 = fft.create_buffer();
        let x_buf = fft.create_buffer();
        let h_buf = fft.create_buffer();
        Self {
            fft,
            tmp,
            tmp2,
            x_buf,
            h_buf,
        }
    }
}

impl AutoCorrelationCalculator {
    /// Computes the auto-correlation coefficients for a target pitch interval.
    ///
    /// `pitch_buf` must have size `BUF_SIZE_12K_HZ`.
    /// `auto_corr` must have size `NUM_LAGS_12K_HZ`. Indexes are inverted lags.
    pub fn compute_on_pitch_buffer(&mut self, pitch_buf: &[f32], auto_corr: &mut [f32]) {
        debug_assert_eq!(pitch_buf.len(), BUF_SIZE_12K_HZ);
        debug_assert_eq!(auto_corr.len(), NUM_LAGS_12K_HZ);

        let fft_frame_size = 1 << AUTO_CORRELATION_FFT_ORDER;
        let convolution_length = BUF_SIZE_12K_HZ - MAX_PITCH_12K_HZ;

        // Compute the FFT for the reversed reference frame.
        // pitch_buf[-convolution_length:]
        let tmp = self.tmp.as_mut_slice();
        for i in 0..convolution_length {
            tmp[i] = pitch_buf[BUF_SIZE_12K_HZ - 1 - i];
        }
        tmp[convolution_length..fft_frame_size].fill(0.0);
        self.fft.forward(&self.tmp, &mut self.h_buf, false);

        // Compute the FFT for the sliding frames chunk.
        // pitch_buf[:convolution_length + NUM_LAGS_12K_HZ]
        let tmp = self.tmp.as_mut_slice();
        let copy_len = convolution_length + NUM_LAGS_12K_HZ;
        tmp[..copy_len].copy_from_slice(&pitch_buf[..copy_len]);
        tmp[copy_len..fft_frame_size].fill(0.0);
        self.fft.forward(&self.tmp, &mut self.x_buf, false);

        // Convolve in the frequency domain.
        let scaling = 1.0 / fft_frame_size as f32;
        self.tmp.as_mut_slice().fill(0.0);
        self.fft
            .convolve_accumulate(&self.x_buf, &self.h_buf, &mut self.tmp, scaling);
        self.fft.backward(&self.tmp, &mut self.tmp2, false);

        // Extract the auto-correlation coefficients.
        let tmp = self.tmp2.as_slice();
        auto_corr.copy_from_slice(
            &tmp[convolution_length - 1..convolution_length - 1 + NUM_LAGS_12K_HZ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_correlation_on_constant_pitch_buffer() {
        let pitch_buf = [1.0_f32; BUF_SIZE_12K_HZ];
        let mut auto_corr = [0.0_f32; NUM_LAGS_12K_HZ];
        let mut calc = AutoCorrelationCalculator::default();
        calc.compute_on_pitch_buffer(&pitch_buf, &mut auto_corr);

        // For a constant signal, auto-correlation at all lags should be
        // approximately equal to the convolution_length.
        let convolution_length = (BUF_SIZE_12K_HZ - MAX_PITCH_12K_HZ) as f32;
        for (i, &ac) in auto_corr.iter().enumerate() {
            assert!(
                (ac - convolution_length).abs() < 1.0,
                "auto_corr[{i}] = {ac}, expected ~{convolution_length}"
            );
        }
    }

    #[test]
    fn auto_correlation_on_zero_buffer() {
        let pitch_buf = [0.0_f32; BUF_SIZE_12K_HZ];
        let mut auto_corr = [0.0_f32; NUM_LAGS_12K_HZ];
        let mut calc = AutoCorrelationCalculator::default();
        calc.compute_on_pitch_buffer(&pitch_buf, &mut auto_corr);

        for (i, &ac) in auto_corr.iter().enumerate() {
            assert!(ac.abs() < 1e-6, "auto_corr[{i}] = {ac}, expected ~0");
        }
    }
}
