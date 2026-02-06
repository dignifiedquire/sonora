//! FFT data type for AEC3.
//!
//! Ported from `modules/audio_processing/aec3/fft_data.h` and
//! `fft_data_avx2.cc`.

use crate::common::{FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1};

/// Holds the real and imaginary parts produced from a 128-point real-valued FFT.
///
/// The FFT of a real 128-sample signal produces 65 complex bins (DC through
/// Nyquist). The DC and Nyquist bins are always real-valued, so `im[0]` and
/// `im[64]` are kept at zero.
#[derive(Clone)]
pub(crate) struct FftData {
    pub re: [f32; FFT_LENGTH_BY_2_PLUS_1],
    pub im: [f32; FFT_LENGTH_BY_2_PLUS_1],
}

impl Default for FftData {
    fn default() -> Self {
        Self {
            re: [0.0; FFT_LENGTH_BY_2_PLUS_1],
            im: [0.0; FFT_LENGTH_BY_2_PLUS_1],
        }
    }
}

impl FftData {
    /// Copies data from `src`, forcing `im[0]` and `im[N/2]` to zero.
    pub(crate) fn assign(&mut self, src: &FftData) {
        self.re = src.re;
        self.im = src.im;
        self.im[0] = 0.0;
        self.im[FFT_LENGTH_BY_2] = 0.0;
    }

    /// Sets all bins to zero.
    pub(crate) fn clear(&mut self) {
        self.re.fill(0.0);
        self.im.fill(0.0);
    }

    /// Computes the power spectrum: `out[k] = re[k]^2 + im[k]^2`.
    ///
    /// Delegates to `webrtc_simd::power_spectrum` for SIMD acceleration.
    pub(crate) fn spectrum(&self, power_spectrum: &mut [f32; FFT_LENGTH_BY_2_PLUS_1]) {
        webrtc_simd::detect_backend().power_spectrum(&self.re, &self.im, power_spectrum);
    }

    /// Unpacks from Ooura's interleaved format into separate re/im arrays.
    ///
    /// Ooura packs a 128-point real FFT result as:
    /// ```text
    /// v[0] = DC,  v[1] = Nyquist,
    /// v[2] = re[1], v[3] = im[1], v[4] = re[2], v[5] = im[2], ...
    /// ```
    pub(crate) fn copy_from_packed_array(&mut self, v: &[f32; FFT_LENGTH]) {
        self.re[0] = v[0];
        self.re[FFT_LENGTH_BY_2] = v[1];
        self.im[0] = 0.0;
        self.im[FFT_LENGTH_BY_2] = 0.0;
        for k in 1..FFT_LENGTH_BY_2 {
            self.re[k] = v[2 * k];
            self.im[k] = v[2 * k + 1];
        }
    }

    /// Packs re/im arrays back into Ooura's interleaved format.
    pub(crate) fn copy_to_packed_array(&self, v: &mut [f32; FFT_LENGTH]) {
        v[0] = self.re[0];
        v[1] = self.re[FFT_LENGTH_BY_2];
        for k in 1..FFT_LENGTH_BY_2 {
            v[2 * k] = self.re[k];
            v[2 * k + 1] = self.im[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_fft_data() -> FftData {
        let mut x = FftData::default();
        for k in 0..x.re.len() {
            x.re[k] = (k + 1) as f32;
        }
        x.im[0] = 0.0;
        x.im[FFT_LENGTH_BY_2] = 0.0;
        for k in 1..x.im.len() - 1 {
            x.im[k] = 2.0 * (k + 1) as f32;
        }
        x
    }

    #[test]
    fn assign_copies_and_zeros_dc_nyquist_im() {
        let x = make_test_fft_data();
        let mut y = FftData::default();
        // Deliberately set im values that should be zeroed.
        let mut src = x.clone();
        src.im[0] = 999.0;
        src.im[FFT_LENGTH_BY_2] = 888.0;

        y.assign(&src);
        assert_eq!(y.re, src.re);
        assert_eq!(y.im[0], 0.0);
        assert_eq!(y.im[FFT_LENGTH_BY_2], 0.0);
        for k in 1..FFT_LENGTH_BY_2 {
            assert_eq!(y.im[k], src.im[k]);
        }
    }

    #[test]
    fn clear_zeros_everything() {
        let mut x = make_test_fft_data();
        x.clear();
        assert!(x.re.iter().all(|&v| v == 0.0));
        assert!(x.im.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn spectrum_scalar() {
        let x = make_test_fft_data();
        let mut spectrum = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        x.spectrum(&mut spectrum);

        // DC: im[0] = 0, so spectrum[0] = re[0]^2
        assert_eq!(spectrum[0], x.re[0] * x.re[0]);
        // Nyquist: im[64] = 0, so spectrum[64] = re[64]^2
        assert_eq!(
            spectrum[FFT_LENGTH_BY_2],
            x.re[FFT_LENGTH_BY_2] * x.re[FFT_LENGTH_BY_2]
        );
        // Middle bins
        for k in 1..FFT_LENGTH_BY_2 {
            assert_eq!(spectrum[k], x.re[k] * x.re[k] + x.im[k] * x.im[k]);
        }
    }

    #[test]
    fn copy_to_packed_array() {
        let x = make_test_fft_data();
        let mut packed = [0.0f32; FFT_LENGTH];
        x.copy_to_packed_array(&mut packed);

        assert_eq!(packed[0], x.re[0]);
        assert_eq!(packed[1], x.re[FFT_LENGTH_BY_2]);
        for k in 1..FFT_LENGTH_BY_2 {
            assert_eq!(packed[2 * k], x.re[k]);
            assert_eq!(packed[2 * k + 1], x.im[k]);
        }
    }

    #[test]
    fn copy_from_packed_array() {
        let x_ref = make_test_fft_data();
        let mut packed = [0.0f32; FFT_LENGTH];
        x_ref.copy_to_packed_array(&mut packed);

        let mut x = FftData::default();
        x.copy_from_packed_array(&packed);

        assert_eq!(x.re, x_ref.re);
        assert_eq!(x.im, x_ref.im);
    }

    #[test]
    fn roundtrip_packed() {
        // Verify pack → unpack roundtrip preserves data exactly.
        let original = make_test_fft_data();
        let mut packed = [0.0f32; FFT_LENGTH];
        original.copy_to_packed_array(&mut packed);

        let mut restored = FftData::default();
        restored.copy_from_packed_array(&packed);

        assert_eq!(original.re, restored.re);
        assert_eq!(original.im, restored.im);
    }
}
