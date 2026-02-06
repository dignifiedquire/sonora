//! Suppression filter — applies the suppression gain in the frequency domain
//! and synthesizes the output using overlap-add with a sqrt-Hanning window.
//!
//! Ported from `modules/audio_processing/aec3/suppression_filter.h/cc`.

use crate::aec3_fft::Aec3Fft;
use crate::block::Block;
use crate::common::{
    FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, num_bands_for_rate, valid_full_band_rate,
};
use crate::fft_data::FftData;
use crate::vector_math::VectorMath;

/// Sqrt-Hanning window from Matlab command `win = sqrt(hanning(128))`.
#[allow(clippy::approx_constant, reason = "exact upstream table values")]
const K_SQRT_HANNING: [f32; FFT_LENGTH] = [
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

/// Applies frequency-domain suppression gain and comfort noise, then
/// synthesizes the output using overlap-add with a sqrt-Hanning window.
pub(crate) struct SuppressionFilter {
    sample_rate_hz: usize,
    num_capture_channels: usize,
    fft: Aec3Fft,
    vector_math: VectorMath,
    e_output_old: Vec<Vec<[f32; FFT_LENGTH_BY_2]>>,
}

impl SuppressionFilter {
    pub(crate) fn new(sample_rate_hz: usize, num_capture_channels: usize) -> Self {
        debug_assert!(valid_full_band_rate(sample_rate_hz));
        let num_bands = num_bands_for_rate(sample_rate_hz);
        let backend = webrtc_simd::detect_backend();
        Self {
            sample_rate_hz,
            num_capture_channels,
            fft: Aec3Fft::new(),
            vector_math: VectorMath::new(backend),
            e_output_old: vec![vec![[0.0; FFT_LENGTH_BY_2]; num_capture_channels]; num_bands],
        }
    }

    /// Applies the suppression gain and comfort noise to the error signal.
    pub(crate) fn apply_gain(
        &mut self,
        comfort_noise: &[FftData],
        comfort_noise_high_band: &[FftData],
        suppression_gain: &[f32; FFT_LENGTH_BY_2_PLUS_1],
        high_bands_gain: f32,
        e_lowest_band: &[FftData],
        e: &mut Block,
    ) {
        debug_assert_eq!(e.num_bands(), num_bands_for_rate(self.sample_rate_hz));

        // Comfort noise gain is sqrt(1-g^2), where g is the suppression gain.
        let mut noise_gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        for i in 0..FFT_LENGTH_BY_2_PLUS_1 {
            noise_gain[i] = 1.0 - suppression_gain[i] * suppression_gain[i];
        }
        self.vector_math.sqrt(&mut noise_gain);

        let high_bands_noise_scaling = 0.4 * (1.0 - high_bands_gain * high_bands_gain).sqrt();

        for ch in 0..self.num_capture_channels {
            let mut e_freq = FftData::default();

            // Analysis filterbank.
            e_freq.assign(&e_lowest_band[ch]);

            for i in 0..FFT_LENGTH_BY_2_PLUS_1 {
                // Apply suppression gains.
                let e_real = e_freq.re[i] * suppression_gain[i];
                let e_imag = e_freq.im[i] * suppression_gain[i];

                // Scale and add the comfort noise.
                e_freq.re[i] = e_real + noise_gain[i] * comfort_noise[ch].re[i];
                e_freq.im[i] = e_imag + noise_gain[i] * comfort_noise[ch].im[i];
            }

            // Synthesis filterbank.
            let mut e_extended = [0.0f32; FFT_LENGTH];
            const K_IFFT_NORMALIZATION: f32 = 2.0 / FFT_LENGTH as f32;
            self.fft.ifft(&e_freq, &mut e_extended);

            let e0 = e.view_mut(0, ch);

            // Window and add the first half of e_extended with the second half
            // of e_extended from the previous block.
            for i in 0..FFT_LENGTH_BY_2 {
                let e0_i = self.e_output_old[0][ch][i] * K_SQRT_HANNING[FFT_LENGTH_BY_2 + i]
                    + e_extended[i] * K_SQRT_HANNING[i];
                e0[i] = e0_i * K_IFFT_NORMALIZATION;
            }

            // The second half of e_extended is stored for the succeeding frame.
            self.e_output_old[0][ch].copy_from_slice(&e_extended[FFT_LENGTH_BY_2..FFT_LENGTH]);

            // Apply suppression gain to upper bands.
            for b in 1..e.num_bands() {
                let e_band = e.view_mut(b, ch);
                for i in 0..FFT_LENGTH_BY_2 {
                    e_band[i] *= high_bands_gain;
                }
            }

            // Add comfort noise to band 1.
            if e.num_bands() > 1 {
                let mut cn_high = FftData::default();
                cn_high.assign(&comfort_noise_high_band[ch]);
                let mut time_domain_high_band_noise = [0.0f32; FFT_LENGTH];
                self.fft.ifft(&cn_high, &mut time_domain_high_band_noise);

                let e1 = e.view_mut(1, ch);
                let gain = high_bands_noise_scaling * K_IFFT_NORMALIZATION;
                for i in 0..FFT_LENGTH_BY_2 {
                    e1[i] += time_domain_high_band_noise[i] * gain;
                }
            }

            // Delay upper bands to match the delay of the filter bank.
            for b in 1..e.num_bands() {
                let e_band = e.view_mut(b, ch);
                for i in 0..FFT_LENGTH_BY_2 {
                    std::mem::swap(&mut e_band[i], &mut self.e_output_old[b][ch][i]);
                }
            }

            // Clamp output of all bands.
            for b in 0..e.num_bands() {
                let e_band = e.view_mut(b, ch);
                for i in 0..FFT_LENGTH_BY_2 {
                    e_band[i] = e_band[i].clamp(-32768.0, 32767.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aec3_fft::{Aec3Fft, Window};
    use crate::common::BLOCK_SIZE;
    use std::f32::consts::PI;

    fn produce_sinusoid(
        sample_rate_hz: usize,
        sinusoidal_frequency_hz: f32,
        sample_counter: &mut usize,
        x: &mut Block,
    ) {
        for j in 0..BLOCK_SIZE {
            let k = *sample_counter + j;
            for channel in 0..x.num_channels() {
                x.view_mut(0, channel)[j] = 32767.0
                    * (2.0 * PI * sinusoidal_frequency_hz * k as f32 / sample_rate_hz as f32).sin();
            }
        }
        *sample_counter += BLOCK_SIZE;

        for band in 1..x.num_bands() {
            for channel in 0..x.num_channels() {
                x.view_mut(band, channel).fill(0.0);
            }
        }
    }

    #[test]
    fn comfort_noise_in_unity_gain() {
        let mut filter = SuppressionFilter::new(48000, 1);
        let mut cn = vec![FftData::default(); 1];
        let mut cn_high_bands = vec![FftData::default(); 1];
        let mut gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let fft = Aec3Fft::new();

        gain.fill(1.0);
        cn[0].re.fill(1.0);
        cn[0].im.fill(1.0);
        cn_high_bands[0].re.fill(1.0);
        cn_high_bands[0].im.fill(1.0);

        let mut e = Block::new(3, 1);
        let e_ref = e.clone();

        let mut e_fft = vec![FftData::default(); 1];
        fft.padded_fft(e.view(0, 0), &e_old, Window::SqrtHanning, &mut e_fft[0]);
        e_old.copy_from_slice(e.view(0, 0));

        filter.apply_gain(&cn, &cn_high_bands, &gain, 1.0, &e_fft, &mut e);

        for band in 0..e.num_bands() {
            for channel in 0..e.num_channels() {
                for sample in 0..BLOCK_SIZE {
                    assert_eq!(
                        e_ref.view(band, channel)[sample],
                        e.view(band, channel)[sample],
                        "band {band}, ch {channel}, sample {sample}"
                    );
                }
            }
        }
    }

    #[test]
    fn signal_suppression() {
        const SAMPLE_RATE_HZ: usize = 48000;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let num_channels = 1;

        let mut filter = SuppressionFilter::new(SAMPLE_RATE_HZ, 1);
        let cn = vec![FftData::default(); 1];
        let cn_high_bands = vec![FftData::default(); 1];
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let fft = Aec3Fft::new();
        let mut gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut e = Block::new(num_bands, num_channels);

        gain.fill(1.0);
        for g in &mut gain[10..] {
            *g = 0.0;
        }

        let mut sample_counter = 0;

        let mut e0_input = 0.0f32;
        let mut e0_output = 0.0f32;
        for _ in 0..100 {
            produce_sinusoid(
                16000,
                16000.0 * 40.0 / FFT_LENGTH_BY_2 as f32 / 2.0,
                &mut sample_counter,
                &mut e,
            );
            e0_input += e.view(0, 0).iter().map(|x| x * x).sum::<f32>();

            let mut e_fft = vec![FftData::default(); 1];
            fft.padded_fft(e.view(0, 0), &e_old, Window::SqrtHanning, &mut e_fft[0]);
            e_old.copy_from_slice(e.view(0, 0));

            filter.apply_gain(&cn, &cn_high_bands, &gain, 1.0, &e_fft, &mut e);
            e0_output += e.view(0, 0).iter().map(|x| x * x).sum::<f32>();
        }

        assert!(
            e0_output < e0_input / 1000.0,
            "e0_output={e0_output}, e0_input={e0_input}"
        );
    }

    #[test]
    fn signal_transparency() {
        const SAMPLE_RATE_HZ: usize = 48000;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let num_channels = 1;

        let mut filter = SuppressionFilter::new(SAMPLE_RATE_HZ, 1);
        let cn = vec![FftData::default(); 1];
        let cn_high_bands = vec![FftData::default(); 1];
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let fft = Aec3Fft::new();
        let mut gain = [0.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut e = Block::new(num_bands, num_channels);

        gain.fill(1.0);
        for g in &mut gain[30..] {
            *g = 0.0;
        }

        let mut sample_counter = 0;

        let mut e0_input = 0.0f32;
        let mut e0_output = 0.0f32;
        for _ in 0..100 {
            produce_sinusoid(
                16000,
                16000.0 * 10.0 / FFT_LENGTH_BY_2 as f32 / 2.0,
                &mut sample_counter,
                &mut e,
            );
            e0_input += e.view(0, 0).iter().map(|x| x * x).sum::<f32>();

            let mut e_fft = vec![FftData::default(); 1];
            fft.padded_fft(e.view(0, 0), &e_old, Window::SqrtHanning, &mut e_fft[0]);
            e_old.copy_from_slice(e.view(0, 0));

            filter.apply_gain(&cn, &cn_high_bands, &gain, 1.0, &e_fft, &mut e);
            e0_output += e.view(0, 0).iter().map(|x| x * x).sum::<f32>();
        }

        assert!(
            0.9 * e0_input < e0_output,
            "e0_output={e0_output}, 0.9*e0_input={}",
            0.9 * e0_input
        );
    }

    #[test]
    fn delay() {
        const SAMPLE_RATE_HZ: usize = 48000;
        let num_bands = num_bands_for_rate(SAMPLE_RATE_HZ);
        let num_channels = 1;

        let mut filter = SuppressionFilter::new(SAMPLE_RATE_HZ, 1);
        let cn = vec![FftData::default(); 1];
        let cn_high_bands = vec![FftData::default(); 1];
        let mut e_old = [0.0f32; FFT_LENGTH_BY_2];
        let fft = Aec3Fft::new();
        let mut gain = [1.0f32; FFT_LENGTH_BY_2_PLUS_1];
        let mut e = Block::new(num_bands, num_channels);

        for k in 0..100usize {
            for band in 0..num_bands {
                for channel in 0..num_channels {
                    let e_view = e.view_mut(band, channel);
                    for sample in 0..BLOCK_SIZE {
                        e_view[sample] = (k * BLOCK_SIZE + sample + channel) as f32;
                    }
                }
            }

            let mut e_fft = vec![FftData::default(); 1];
            fft.padded_fft(e.view(0, 0), &e_old, Window::SqrtHanning, &mut e_fft[0]);
            e_old.copy_from_slice(e.view(0, 0));

            filter.apply_gain(&cn, &cn_high_bands, &gain, 1.0, &e_fft, &mut e);
            if k > 2 {
                for band in 0..num_bands {
                    for channel in 0..num_channels {
                        let e_view = e.view(band, channel);
                        for sample in 0..BLOCK_SIZE {
                            let expected = (k * BLOCK_SIZE + sample - BLOCK_SIZE + channel) as f32;
                            assert!(
                                (expected - e_view[sample]).abs() < 0.01,
                                "k={k}, band={band}, ch={channel}, sample={sample}: expected {expected}, got {}",
                                e_view[sample]
                            );
                        }
                    }
                }
            }
        }
    }
}
