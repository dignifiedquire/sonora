//! 256-point FFT wrapper for the noise suppressor.
//!
//! Wraps the Ooura fft4g algorithm to convert between time-domain
//! frames and split real/imaginary frequency-domain arrays.
//!
//! C++ source: `webrtc/modules/audio_processing/ns/ns_fft.cc`

use webrtc_fft::fft4g::Fft4g;

use crate::config::{FFT_SIZE, FFT_SIZE_BY_2_PLUS_1};

/// 256-point real FFT for noise suppression.
///
/// Maintains pre-initialized twiddle tables for the Ooura fft4g algorithm.
#[derive(Debug)]
pub struct NsFft {
    fft: Fft4g,
}

impl Default for NsFft {
    fn default() -> Self {
        Self::new()
    }
}

impl NsFft {
    /// Create a new FFT instance with pre-computed tables.
    pub fn new() -> Self {
        Self {
            fft: Fft4g::new(FFT_SIZE),
        }
    }

    /// Forward FFT: time domain -> split real/imaginary arrays.
    ///
    /// `time_data` is transformed in-place (used as scratch), then the
    /// packed output is split into separate `real` and `imag` arrays
    /// of length `FFT_SIZE_BY_2_PLUS_1` (129).
    pub fn fft(
        &mut self,
        time_data: &mut [f32; FFT_SIZE],
        real: &mut [f32; FFT_SIZE],
        imag: &mut [f32; FFT_SIZE],
    ) {
        self.fft.rdft(time_data);

        // Ooura packing: time_data[0] = DC, time_data[1] = Nyquist,
        // time_data[2k], time_data[2k+1] = real/imag of bin k.
        imag[0] = 0.0;
        real[0] = time_data[0];

        imag[FFT_SIZE_BY_2_PLUS_1 - 1] = 0.0;
        real[FFT_SIZE_BY_2_PLUS_1 - 1] = time_data[1];

        for i in 1..FFT_SIZE_BY_2_PLUS_1 - 1 {
            real[i] = time_data[2 * i];
            imag[i] = time_data[2 * i + 1];
        }
    }

    /// Inverse FFT: split real/imaginary arrays -> time domain.
    ///
    /// Re-packs `real` and `imag` into the Ooura format, performs the
    /// inverse FFT, and scales the output by `2/N`.
    pub fn ifft(&mut self, real: &[f32], imag: &[f32], time_data: &mut [f32; FFT_SIZE]) {
        // Pack into Ooura format.
        time_data[0] = real[0];
        time_data[1] = real[FFT_SIZE_BY_2_PLUS_1 - 1];
        for i in 1..FFT_SIZE_BY_2_PLUS_1 - 1 {
            time_data[2 * i] = real[i];
            time_data[2 * i + 1] = imag[i];
        }

        self.fft.irdft(time_data);

        // Scale the output (Ooura convention).
        let scaling = 2.0 / FFT_SIZE as f32;
        for d in time_data.iter_mut() {
            *d *= scaling;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_ifft_roundtrip() {
        let mut fft = NsFft::new();
        let mut time_data = [0.0_f32; FFT_SIZE];
        for (i, v) in time_data.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).sin();
        }
        let original = time_data;

        let mut real = [0.0_f32; FFT_SIZE];
        let mut imag = [0.0_f32; FFT_SIZE];
        fft.fft(&mut time_data, &mut real, &mut imag);

        // DC and Nyquist should have zero imaginary.
        assert_eq!(imag[0], 0.0);
        assert_eq!(imag[FFT_SIZE_BY_2_PLUS_1 - 1], 0.0);

        let mut recovered = [0.0_f32; FFT_SIZE];
        fft.ifft(&real, &imag, &mut recovered);

        for (i, (&o, &r)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (o - r).abs() < 1e-4,
                "mismatch at {i}: original={o}, recovered={r}"
            );
        }
    }

    #[test]
    fn fft_dc_signal() {
        let mut fft = NsFft::new();
        let mut time_data = [1.0_f32; FFT_SIZE];
        let mut real = [0.0_f32; FFT_SIZE];
        let mut imag = [0.0_f32; FFT_SIZE];

        fft.fft(&mut time_data, &mut real, &mut imag);

        // DC bin should equal sum of samples = N.
        assert!(
            (real[0] - FFT_SIZE as f32).abs() < 1e-3,
            "DC = {}, expected {}",
            real[0],
            FFT_SIZE
        );
        // All other bins should be near zero.
        for k in 1..FFT_SIZE_BY_2_PLUS_1 {
            assert!(
                real[k].abs() < 1e-3 && imag[k].abs() < 1e-3,
                "bin {k}: real={}, imag={}",
                real[k],
                imag[k]
            );
        }
    }

    #[test]
    fn fft_impulse() {
        let mut fft = NsFft::new();
        let mut time_data = [0.0_f32; FFT_SIZE];
        time_data[0] = 1.0;
        let mut real = [0.0_f32; FFT_SIZE];
        let mut imag = [0.0_f32; FFT_SIZE];

        fft.fft(&mut time_data, &mut real, &mut imag);

        // All real bins should be 1.0, all imag should be 0.0.
        for k in 0..FFT_SIZE_BY_2_PLUS_1 {
            assert!(
                (real[k] - 1.0).abs() < 1e-4,
                "bin {k} real: expected 1.0, got {}",
                real[k]
            );
            assert!(
                imag[k].abs() < 1e-4,
                "bin {k} imag: expected 0.0, got {}",
                imag[k]
            );
        }
    }
}
