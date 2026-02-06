//! AEC3 FFT wrapper.
//!
//! Ported from `modules/audio_processing/aec3/aec3_fft.h/cc`.

use crate::common::{FFT_LENGTH, FFT_LENGTH_BY_2};
use crate::fft_data::FftData;

/// Window type for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Window {
    Rectangular,
    Hanning,
    SqrtHanning,
}

/// Hanning window coefficients for 64 samples (half-length).
/// Generated from `0.5 * (1 - cos(2*pi*n/127))` for n=0..63.
const HANNING_64: [f32; FFT_LENGTH_BY_2] = [
    0.0,
    0.002_484_61,
    0.009_913_76,
    0.022_213_6,
    0.039_261_89,
    0.060_889_21,
    0.086_880_61,
    0.116_977_78,
    0.150_881_59,
    0.188_255_1,
    0.228_726_87,
    0.271_894_67,
    0.317_329_49,
    0.364_579_77,
    0.413_175_91,
    0.462_634_95,
    0.512_465_35,
    0.562_171_85,
    0.611_260_47,
    0.659_243_33,
    0.705_643_55,
    0.75,
    0.791_871_84,
    0.830_842_92,
    0.866_525_94,
    0.898_566_25,
    0.926_645_44,
    0.950_484_43,
    0.969_846_31,
    0.984_538_64,
    0.994_415_41,
    0.999_378_46,
    0.999_378_46,
    0.994_415_41,
    0.984_538_64,
    0.969_846_31,
    0.950_484_43,
    0.926_645_44,
    0.898_566_25,
    0.866_525_94,
    0.830_842_92,
    0.791_871_84,
    0.75,
    0.705_643_55,
    0.659_243_33,
    0.611_260_47,
    0.562_171_85,
    0.512_465_35,
    0.462_634_95,
    0.413_175_91,
    0.364_579_77,
    0.317_329_49,
    0.271_894_67,
    0.228_726_87,
    0.188_255_1,
    0.150_881_59,
    0.116_977_78,
    0.086_880_61,
    0.060_889_21,
    0.039_261_89,
    0.022_213_6,
    0.009_913_76,
    0.002_484_61,
    0.0,
];

/// Sqrt-Hanning window coefficients for 128 samples.
/// From Matlab `sqrt(hanning(128))`.
#[allow(clippy::approx_constant, reason = "exact upstream table values")]
const SQRT_HANNING_128: [f32; FFT_LENGTH] = [
    0.000_000_000_000_00,
    0.024_541_228_522_91,
    0.049_067_674_327_42,
    0.073_564_563_599_67,
    0.098_017_140_329_56,
    0.122_410_675_199_22,
    0.146_730_474_455_36,
    0.170_961_888_760_30,
    0.195_090_322_016_13,
    0.219_101_240_156_87,
    0.242_980_179_903_26,
    0.266_712_757_474_90,
    0.290_284_677_254_46,
    0.313_681_740_398_89,
    0.336_889_853_392_22,
    0.359_895_036_534_99,
    0.382_683_432_365_09,
    0.405_241_314_004_99,
    0.427_555_093_430_28,
    0.449_611_329_654_61,
    0.471_396_736_826_00,
    0.492_898_192_229_78,
    0.514_102_744_193_22,
    0.534_997_619_887_10,
    0.555_570_233_019_60,
    0.575_808_191_417_85,
    0.595_699_304_492_43,
    0.615_231_590_580_63,
    0.634_393_284_163_65,
    0.653_172_842_953_78,
    0.671_558_954_847_02,
    0.689_540_544_737_07,
    0.707_106_781_186_55,
    0.724_247_082_951_47,
    0.740_951_125_354_96,
    0.757_208_846_506_48,
    0.773_010_453_362_74,
    0.788_346_427_626_61,
    0.803_207_531_480_64,
    0.817_584_813_151_58,
    0.831_469_612_302_55,
    0.844_853_565_249_71,
    0.857_728_610_000_27,
    0.870_086_991_108_71,
    0.881_921_264_348_35,
    0.893_224_301_195_52,
    0.903_989_293_123_44,
    0.914_209_755_703_53,
    0.923_879_532_511_29,
    0.932_992_798_834_74,
    0.941_544_065_183_02,
    0.949_528_180_593_04,
    0.956_940_335_732_21,
    0.963_776_065_795_44,
    0.970_031_253_194_54,
    0.975_702_130_038_53,
    0.980_785_280_403_23,
    0.985_277_642_388_94,
    0.989_176_509_964_78,
    0.992_479_534_598_71,
    0.995_184_726_672_20,
    0.997_290_456_678_69,
    0.998_795_456_205_17,
    0.999_698_818_696_20,
    1.000_000_000_000_00,
    0.999_698_818_696_20,
    0.998_795_456_205_17,
    0.997_290_456_678_69,
    0.995_184_726_672_20,
    0.992_479_534_598_71,
    0.989_176_509_964_78,
    0.985_277_642_388_94,
    0.980_785_280_403_23,
    0.975_702_130_038_53,
    0.970_031_253_194_54,
    0.963_776_065_795_44,
    0.956_940_335_732_21,
    0.949_528_180_593_04,
    0.941_544_065_183_02,
    0.932_992_798_834_74,
    0.923_879_532_511_29,
    0.914_209_755_703_53,
    0.903_989_293_123_44,
    0.893_224_301_195_52,
    0.881_921_264_348_35,
    0.870_086_991_108_71,
    0.857_728_610_000_27,
    0.844_853_565_249_71,
    0.831_469_612_302_55,
    0.817_584_813_151_58,
    0.803_207_531_480_64,
    0.788_346_427_626_61,
    0.773_010_453_362_74,
    0.757_208_846_506_48,
    0.740_951_125_354_96,
    0.724_247_082_951_47,
    0.707_106_781_186_55,
    0.689_540_544_737_07,
    0.671_558_954_847_02,
    0.653_172_842_953_78,
    0.634_393_284_163_65,
    0.615_231_590_580_63,
    0.595_699_304_492_43,
    0.575_808_191_417_85,
    0.555_570_233_019_60,
    0.534_997_619_887_10,
    0.514_102_744_193_22,
    0.492_898_192_229_78,
    0.471_396_736_826_00,
    0.449_611_329_654_61,
    0.427_555_093_430_28,
    0.405_241_314_004_99,
    0.382_683_432_365_09,
    0.359_895_036_534_99,
    0.336_889_853_392_22,
    0.313_681_740_398_89,
    0.290_284_677_254_46,
    0.266_712_757_474_90,
    0.242_980_179_903_26,
    0.219_101_240_156_87,
    0.195_090_322_016_13,
    0.170_961_888_760_30,
    0.146_730_474_455_36,
    0.122_410_675_199_22,
    0.098_017_140_329_56,
    0.073_564_563_599_67,
    0.049_067_674_327_42,
    0.024_541_228_522_91,
];

/// Wrapper providing 128-point real-valued FFT for AEC3.
///
/// Uses `webrtc_fft::ooura_fft` (128-point Ooura FFT) under the hood.
pub(crate) struct Aec3Fft;

impl Aec3Fft {
    pub(crate) fn new() -> Self {
        Self
    }

    /// Computes the forward FFT.
    ///
    /// `x` is used as scratch space and is modified in place by the Ooura
    /// routine. The result is unpacked into `x_out`.
    pub(crate) fn fft(&self, x: &mut [f32; FFT_LENGTH], x_out: &mut FftData) {
        webrtc_fft::ooura_fft::forward(x);
        x_out.copy_from_packed_array(x);
    }

    /// Computes the inverse FFT.
    ///
    /// Packs `x_in` into Ooura format, then runs the inverse transform.
    pub(crate) fn ifft(&self, x_in: &FftData, x: &mut [f32; FFT_LENGTH]) {
        x_in.copy_to_packed_array(x);
        webrtc_fft::ooura_fft::inverse(x);
    }

    /// Windows the input with the specified window, zero-pads the first half,
    /// then computes the FFT.
    ///
    /// Input `x` must be `FFT_LENGTH_BY_2` (64) samples long. The first 64
    /// samples of the internal buffer are zeros; the last 64 are `x` (optionally
    /// windowed).
    pub(crate) fn zero_padded_fft(&self, x: &[f32], window: Window, x_out: &mut FftData) {
        debug_assert_eq!(FFT_LENGTH_BY_2, x.len());
        let mut fft_buf = [0.0f32; FFT_LENGTH];
        // First half is zeros (already initialized).
        match window {
            Window::Rectangular => {
                fft_buf[FFT_LENGTH_BY_2..].copy_from_slice(x);
            }
            Window::Hanning => {
                for (dst, (src, w)) in fft_buf[FFT_LENGTH_BY_2..]
                    .iter_mut()
                    .zip(x.iter().zip(HANNING_64.iter()))
                {
                    *dst = src * w;
                }
            }
            Window::SqrtHanning => {
                unreachable!("SqrtHanning not supported for ZeroPaddedFft");
            }
        }
        self.fft(&mut fft_buf, x_out);
    }

    /// Concatenates `x_old` and `x` (each 64 samples), then computes the FFT.
    ///
    /// Uses rectangular window (no windowing).
    pub(crate) fn padded_fft_rect(&self, x: &[f32], x_old: &[f32], x_out: &mut FftData) {
        self.padded_fft(x, x_old, Window::Rectangular, x_out);
    }

    /// Concatenates `x_old` and `x` (each 64 samples), optionally applies a
    /// window, then computes the FFT.
    pub(crate) fn padded_fft(&self, x: &[f32], x_old: &[f32], window: Window, x_out: &mut FftData) {
        debug_assert_eq!(FFT_LENGTH_BY_2, x.len());
        debug_assert_eq!(FFT_LENGTH_BY_2, x_old.len());
        let mut fft_buf = [0.0f32; FFT_LENGTH];

        match window {
            Window::Rectangular => {
                fft_buf[..FFT_LENGTH_BY_2].copy_from_slice(x_old);
                fft_buf[FFT_LENGTH_BY_2..].copy_from_slice(x);
            }
            Window::Hanning => {
                unreachable!("Hanning not supported for PaddedFft");
            }
            Window::SqrtHanning => {
                for (dst, (src, w)) in fft_buf[..FFT_LENGTH_BY_2]
                    .iter_mut()
                    .zip(x_old.iter().zip(SQRT_HANNING_128[..FFT_LENGTH_BY_2].iter()))
                {
                    *dst = src * w;
                }
                for (dst, (src, w)) in fft_buf[FFT_LENGTH_BY_2..]
                    .iter_mut()
                    .zip(x.iter().zip(SQRT_HANNING_128[FFT_LENGTH_BY_2..].iter()))
                {
                    *dst = src * w;
                }
            }
        }
        self.fft(&mut fft_buf, x_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_all_zeros() {
        let fft = Aec3Fft::new();
        let mut x = [0.0f32; FFT_LENGTH];
        let mut x_out = FftData::default();
        fft.fft(&mut x, &mut x_out);
        assert!(x_out.re.iter().all(|&v| v == 0.0));
        assert!(x_out.im.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fft_impulse() {
        let fft = Aec3Fft::new();
        let mut x = [0.0f32; FFT_LENGTH];
        x[0] = 1.0;
        let mut x_out = FftData::default();
        fft.fft(&mut x, &mut x_out);
        // FFT of unit impulse at t=0 → all re = 1.0, all im = 0.0
        for &v in &x_out.re {
            assert_eq!(v, 1.0);
        }
        for &v in &x_out.im {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn fft_dc() {
        let fft = Aec3Fft::new();
        let mut x = [1.0f32; FFT_LENGTH];
        let mut x_out = FftData::default();
        fft.fft(&mut x, &mut x_out);
        // FFT of all-ones: re[0] = 128, rest = 0
        assert_eq!(x_out.re[0], 128.0);
        for &v in &x_out.re[1..] {
            assert_eq!(v, 0.0);
        }
        for &v in &x_out.im {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn ifft_all_zeros() {
        let fft = Aec3Fft::new();
        let x_in = FftData::default();
        let mut x = [0.0f32; FFT_LENGTH];
        fft.ifft(&x_in, &mut x);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ifft_all_ones_re() {
        let fft = Aec3Fft::new();
        let mut x_in = FftData::default();
        x_in.re.fill(1.0);
        let mut x = [0.0f32; FFT_LENGTH];
        fft.ifft(&x_in, &mut x);
        // IFFT of (1,0,0,...) in re → impulse at x[0] = 64
        assert_eq!(x[0], 64.0);
        for &v in &x[1..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn ifft_dc_only() {
        let fft = Aec3Fft::new();
        let mut x_in = FftData::default();
        x_in.re[0] = 128.0;
        let mut x = [0.0f32; FFT_LENGTH];
        fft.ifft(&x_in, &mut x);
        for &v in &x {
            assert_eq!(v, 64.0);
        }
    }

    #[test]
    fn fft_and_ifft_roundtrip() {
        let fft = Aec3Fft::new();
        let mut v = 0;
        for _ in 0..20 {
            let mut x = [0.0f32; FFT_LENGTH];
            let mut x_ref = [0.0f32; FFT_LENGTH];
            for j in 0..FFT_LENGTH {
                x[j] = v as f32;
                // Ooura's IFFT(FFT(x)) = 64*x (unnormalized)
                x_ref[j] = v as f32 * 64.0;
                v += 1;
            }
            let mut x_out = FftData::default();
            fft.fft(&mut x, &mut x_out);
            fft.ifft(&x_out, &mut x);
            for j in 0..FFT_LENGTH {
                assert!(
                    (x_ref[j] - x[j]).abs() < 0.001,
                    "mismatch at {j}: expected {}, got {}",
                    x_ref[j],
                    x[j]
                );
            }
        }
    }

    #[test]
    fn zero_padded_fft_rectangular() {
        let fft = Aec3Fft::new();
        let mut v = 0;
        for _ in 0..20 {
            let mut x_in = [0.0f32; FFT_LENGTH_BY_2];
            let mut x_ref = [0.0f32; FFT_LENGTH];
            for j in 0..FFT_LENGTH_BY_2 {
                x_in[j] = v as f32;
                x_ref[j + FFT_LENGTH_BY_2] = v as f32 * 64.0;
                v += 1;
            }
            let mut x_out = FftData::default();
            fft.zero_padded_fft(&x_in, Window::Rectangular, &mut x_out);
            let mut x_result = [0.0f32; FFT_LENGTH];
            fft.ifft(&x_out, &mut x_result);
            for j in 0..FFT_LENGTH {
                assert!(
                    (x_ref[j] - x_result[j]).abs() < 0.1,
                    "mismatch at {j}: expected {}, got {}",
                    x_ref[j],
                    x_result[j]
                );
            }
        }
    }

    #[test]
    fn padded_fft_rectangular() {
        let fft = Aec3Fft::new();
        let mut v = 0;
        let mut x_old = [0.0f32; FFT_LENGTH_BY_2];
        for _ in 0..20 {
            let mut x_in = [0.0f32; FFT_LENGTH_BY_2];
            for j in 0..FFT_LENGTH_BY_2 {
                x_in[j] = v as f32;
                v += 1;
            }

            let mut x_ref = [0.0f32; FFT_LENGTH];
            x_ref[..FFT_LENGTH_BY_2].copy_from_slice(&x_old);
            x_ref[FFT_LENGTH_BY_2..].copy_from_slice(&x_in);
            for val in &mut x_ref {
                *val *= 64.0;
            }

            let x_old_ref: [f32; FFT_LENGTH_BY_2] = x_in;

            let mut x_out = FftData::default();
            fft.padded_fft_rect(&x_in, &x_old, &mut x_out);
            x_old = x_in;

            let mut x_result = [0.0f32; FFT_LENGTH];
            fft.ifft(&x_out, &mut x_result);
            for j in 0..FFT_LENGTH {
                assert!(
                    (x_ref[j] - x_result[j]).abs() < 0.1,
                    "mismatch at {j}: expected {}, got {}",
                    x_ref[j],
                    x_result[j]
                );
            }

            assert_eq!(x_old_ref, x_old);
        }
    }
}
