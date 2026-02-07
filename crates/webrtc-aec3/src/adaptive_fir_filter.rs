//! Frequency-domain adaptive FIR filter.
//!
//! Implements a partitioned frequency-domain adaptive filter used for echo
//! subtraction. The three critical inner functions (`compute_frequency_response`,
//! `adapt_partitions`, `apply_filter`) are implemented in scalar form.
//!
//! Ported from `modules/audio_processing/aec3/adaptive_fir_filter.h/cc`.

use crate::aec3_fft::Aec3Fft;
use crate::common::{FFT_LENGTH, FFT_LENGTH_BY_2, FFT_LENGTH_BY_2_PLUS_1, get_time_domain_length};
use crate::fft_data::FftData;
use crate::render_buffer::RenderBuffer;
use webrtc_simd::SimdBackend;

// ---------------------------------------------------------------------------
// Free functions — core NLMS operations on partitioned FftData
// ---------------------------------------------------------------------------

/// Computes and stores the frequency response (max power across channels)
/// for each partition.
pub(crate) fn compute_frequency_response(
    backend: SimdBackend,
    num_partitions: usize,
    h: &[Vec<FftData>],
    h2: &mut Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>,
) {
    for h2_p in h2.iter_mut() {
        h2_p.fill(0.0);
    }

    let num_render_channels = h[0].len();
    let mut power = [0.0f32; FFT_LENGTH_BY_2];
    let mut maxed = [0.0f32; FFT_LENGTH_BY_2];
    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            // Vectorized: power spectrum + elementwise max for bins [0..64]
            backend.power_spectrum(
                &h[p][ch].re[..FFT_LENGTH_BY_2],
                &h[p][ch].im[..FFT_LENGTH_BY_2],
                &mut power,
            );
            backend.elementwise_max(&h2[p][..FFT_LENGTH_BY_2], &power, &mut maxed);
            h2[p][..FFT_LENGTH_BY_2].copy_from_slice(&maxed);
            // Scalar tail: bin 64 (Nyquist)
            let t = h[p][ch].re[FFT_LENGTH_BY_2] * h[p][ch].re[FFT_LENGTH_BY_2]
                + h[p][ch].im[FFT_LENGTH_BY_2] * h[p][ch].im[FFT_LENGTH_BY_2];
            h2[p][FFT_LENGTH_BY_2] = h2[p][FFT_LENGTH_BY_2].max(t);
        }
    }
}

/// Adapts the filter partitions: H(t+1) = H(t) + G(t) * conj(X(t)).
pub(crate) fn adapt_partitions(
    backend: SimdBackend,
    render_buffer: &RenderBuffer<'_>,
    g: &FftData,
    num_partitions: usize,
    h: &mut [Vec<FftData>],
) {
    let render_buffer_data = render_buffer.get_fft_buffer();
    let mut index = render_buffer.position();
    let num_render_channels = render_buffer_data[index].len();
    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            let x_p_ch = &render_buffer_data[index][ch];
            let h_p_ch = &mut h[p][ch];
            // Vectorized: conjugate complex multiply-accumulate for bins [0..64]
            backend.complex_multiply_accumulate(
                &x_p_ch.re[..FFT_LENGTH_BY_2],
                &x_p_ch.im[..FFT_LENGTH_BY_2],
                &g.re[..FFT_LENGTH_BY_2],
                &g.im[..FFT_LENGTH_BY_2],
                &mut h_p_ch.re[..FFT_LENGTH_BY_2],
                &mut h_p_ch.im[..FFT_LENGTH_BY_2],
            );
            // Scalar tail: bin 64 (Nyquist)
            let k = FFT_LENGTH_BY_2;
            h_p_ch.re[k] += x_p_ch.re[k] * g.re[k] + x_p_ch.im[k] * g.im[k];
            h_p_ch.im[k] += x_p_ch.re[k] * g.im[k] - x_p_ch.im[k] * g.re[k];
        }
        index = if index < render_buffer_data.len() - 1 {
            index + 1
        } else {
            0
        };
    }
}

/// Produces the filter output: S = sum_p H[p] * X[p].
pub(crate) fn apply_filter(
    backend: SimdBackend,
    render_buffer: &RenderBuffer<'_>,
    num_partitions: usize,
    h: &[Vec<FftData>],
    s: &mut FftData,
) {
    s.clear();

    let render_buffer_data = render_buffer.get_fft_buffer();
    let mut index = render_buffer.position();
    let num_render_channels = render_buffer_data[index].len();
    for p in 0..num_partitions {
        for ch in 0..num_render_channels {
            let x_p_ch = &render_buffer_data[index][ch];
            let h_p_ch = &h[p][ch];
            // Vectorized: standard complex multiply-accumulate for bins [0..64]
            backend.complex_multiply_accumulate_standard(
                &x_p_ch.re[..FFT_LENGTH_BY_2],
                &x_p_ch.im[..FFT_LENGTH_BY_2],
                &h_p_ch.re[..FFT_LENGTH_BY_2],
                &h_p_ch.im[..FFT_LENGTH_BY_2],
                &mut s.re[..FFT_LENGTH_BY_2],
                &mut s.im[..FFT_LENGTH_BY_2],
            );
            // Scalar tail: bin 64 (Nyquist)
            let k = FFT_LENGTH_BY_2;
            s.re[k] += x_p_ch.re[k] * h_p_ch.re[k] - x_p_ch.im[k] * h_p_ch.im[k];
            s.im[k] += x_p_ch.re[k] * h_p_ch.im[k] + x_p_ch.im[k] * h_p_ch.re[k];
        }
        index = if index < render_buffer_data.len() - 1 {
            index + 1
        } else {
            0
        };
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Clears filter partitions in range [old_size, new_size).
fn zero_filter(old_size: usize, new_size: usize, h: &mut [Vec<FftData>]) {
    for p in old_size..new_size {
        for ch_data in &mut h[p] {
            ch_data.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptiveFirFilter
// ---------------------------------------------------------------------------

/// Frequency-domain adaptive FIR filter with partitioned convolution.
pub(crate) struct AdaptiveFirFilter {
    fft: Aec3Fft,
    backend: SimdBackend,
    num_render_channels: usize,
    max_size_partitions: usize,
    size_change_duration_blocks: i32,
    one_by_size_change_duration_blocks: f32,
    current_size_partitions: usize,
    target_size_partitions: usize,
    old_target_size_partitions: usize,
    size_change_counter: i32,
    h: Vec<Vec<FftData>>,
    partition_to_constrain: usize,
}

impl AdaptiveFirFilter {
    pub(crate) fn new(
        backend: SimdBackend,
        max_size_partitions: usize,
        initial_size_partitions: usize,
        size_change_duration_blocks: usize,
        num_render_channels: usize,
    ) -> Self {
        debug_assert!(max_size_partitions >= initial_size_partitions);
        debug_assert!(size_change_duration_blocks > 0);

        let mut h: Vec<Vec<FftData>> = (0..max_size_partitions)
            .map(|_| {
                (0..num_render_channels)
                    .map(|_| FftData::default())
                    .collect()
            })
            .collect();

        zero_filter(0, max_size_partitions, &mut h);

        let mut filter = Self {
            fft: Aec3Fft::new(),
            backend,
            num_render_channels,
            max_size_partitions,
            size_change_duration_blocks: size_change_duration_blocks as i32,
            one_by_size_change_duration_blocks: 1.0 / size_change_duration_blocks as f32,
            current_size_partitions: initial_size_partitions,
            target_size_partitions: initial_size_partitions,
            old_target_size_partitions: initial_size_partitions,
            size_change_counter: 0,
            h,
            partition_to_constrain: 0,
        };

        filter.set_size_partitions(initial_size_partitions, true);
        filter
    }

    /// Produces the output of the filter.
    pub(crate) fn filter(&self, render_buffer: &RenderBuffer<'_>, s: &mut FftData) {
        apply_filter(
            self.backend,
            render_buffer,
            self.current_size_partitions,
            &self.h,
            s,
        );
    }

    /// Adapts the filter.
    pub(crate) fn adapt(&mut self, render_buffer: &RenderBuffer<'_>, g: &FftData) {
        self.adapt_and_update_size(render_buffer, g);
        self.constrain();
    }

    /// Adapts the filter and updates an externally stored impulse response
    /// estimate.
    pub(crate) fn adapt_with_impulse_response(
        &mut self,
        render_buffer: &RenderBuffer<'_>,
        g: &FftData,
        impulse_response: &mut Vec<f32>,
    ) {
        self.adapt_and_update_size(render_buffer, g);
        self.constrain_and_update_impulse_response(impulse_response);
    }

    /// Receives reports that known echo path changes have occurred.
    pub(crate) fn handle_echo_path_change(&mut self) {
        zero_filter(
            self.current_size_partitions,
            self.max_size_partitions,
            &mut self.h,
        );
    }

    /// Returns the filter size in partitions.
    pub(crate) fn size_partitions(&self) -> usize {
        self.current_size_partitions
    }

    /// Sets the filter size.
    pub(crate) fn set_size_partitions(&mut self, size: usize, immediate_effect: bool) {
        self.target_size_partitions = size.min(self.max_size_partitions);
        if immediate_effect {
            let old_size = self.current_size_partitions;
            self.current_size_partitions = self.target_size_partitions;
            self.old_target_size_partitions = self.target_size_partitions;
            zero_filter(old_size, self.current_size_partitions, &mut self.h);
            self.partition_to_constrain = self
                .partition_to_constrain
                .min(self.current_size_partitions.saturating_sub(1));
            self.size_change_counter = 0;
        } else {
            self.size_change_counter = self.size_change_duration_blocks;
        }
    }

    /// Computes the frequency responses for the filter partitions.
    pub(crate) fn compute_frequency_response(&self, h2: &mut Vec<[f32; FFT_LENGTH_BY_2_PLUS_1]>) {
        h2.resize(self.current_size_partitions, [0.0; FFT_LENGTH_BY_2_PLUS_1]);
        compute_frequency_response(self.backend, self.current_size_partitions, &self.h, h2);
    }

    /// Returns the maximum number of partitions for the filter.
    pub(crate) fn max_filter_size_partitions(&self) -> usize {
        self.max_size_partitions
    }

    /// Scales the filter impulse response and spectrum by a factor.
    pub(crate) fn scale_filter(&mut self, factor: f32) {
        for h_p in &mut self.h {
            for h_p_ch in h_p {
                for re in &mut h_p_ch.re {
                    *re *= factor;
                }
                for im in &mut h_p_ch.im {
                    *im *= factor;
                }
            }
        }
    }

    /// Sets the filter coefficients.
    pub(crate) fn set_filter(&mut self, num_partitions: usize, h: &[Vec<FftData>]) {
        let min_num_partitions = self.current_size_partitions.min(num_partitions);
        for p in 0..min_num_partitions {
            debug_assert_eq!(self.h[p].len(), h[p].len());
            for ch in 0..self.num_render_channels {
                self.h[p][ch].assign(&h[p][ch]);
            }
        }
    }

    /// Gets a reference to the filter coefficients.
    pub(crate) fn get_filter(&self) -> &[Vec<FftData>] {
        &self.h
    }

    // --- Private methods ---

    fn adapt_and_update_size(&mut self, render_buffer: &RenderBuffer<'_>, g: &FftData) {
        self.update_size();
        adapt_partitions(
            self.backend,
            render_buffer,
            g,
            self.current_size_partitions,
            &mut self.h,
        );
    }

    fn update_size(&mut self) {
        let old_size = self.current_size_partitions;
        if self.size_change_counter > 0 {
            self.size_change_counter -= 1;

            let change_factor =
                self.size_change_counter as f32 * self.one_by_size_change_duration_blocks;

            self.current_size_partitions = (self.old_target_size_partitions as f32 * change_factor
                + self.target_size_partitions as f32 * (1.0 - change_factor))
                as usize;

            self.partition_to_constrain = self
                .partition_to_constrain
                .min(self.current_size_partitions.saturating_sub(1));
        } else {
            self.current_size_partitions = self.target_size_partitions;
            self.old_target_size_partitions = self.target_size_partitions;
        }
        zero_filter(old_size, self.current_size_partitions, &mut self.h);
    }

    fn constrain(&mut self) {
        let mut h_td = [0.0f32; FFT_LENGTH];
        for ch in 0..self.num_render_channels {
            self.fft
                .ifft(&self.h[self.partition_to_constrain][ch], &mut h_td);

            const SCALE: f32 = 1.0 / FFT_LENGTH_BY_2 as f32;
            for v in &mut h_td[..FFT_LENGTH_BY_2] {
                *v *= SCALE;
            }
            h_td[FFT_LENGTH_BY_2..].fill(0.0);

            self.fft
                .fft(&mut h_td, &mut self.h[self.partition_to_constrain][ch]);
        }

        self.partition_to_constrain =
            if self.partition_to_constrain < self.current_size_partitions - 1 {
                self.partition_to_constrain + 1
            } else {
                0
            };
    }

    fn constrain_and_update_impulse_response(&mut self, impulse_response: &mut Vec<f32>) {
        impulse_response.resize(get_time_domain_length(self.current_size_partitions), 0.0);

        let ir_start = self.partition_to_constrain * FFT_LENGTH_BY_2;
        let ir_end = ir_start + FFT_LENGTH_BY_2;
        impulse_response[ir_start..ir_end].fill(0.0);

        let mut h_td = [0.0f32; FFT_LENGTH];
        for ch in 0..self.num_render_channels {
            self.fft
                .ifft(&self.h[self.partition_to_constrain][ch], &mut h_td);

            const SCALE: f32 = 1.0 / FFT_LENGTH_BY_2 as f32;
            for v in &mut h_td[..FFT_LENGTH_BY_2] {
                *v *= SCALE;
            }
            h_td[FFT_LENGTH_BY_2..].fill(0.0);

            if ch == 0 {
                impulse_response[ir_start..ir_end].copy_from_slice(&h_td[..FFT_LENGTH_BY_2]);
            } else {
                for (ir_val, &h_val) in impulse_response[ir_start..ir_end]
                    .iter_mut()
                    .zip(&h_td[..FFT_LENGTH_BY_2])
                {
                    if ir_val.abs() < h_val.abs() {
                        *ir_val = h_val;
                    }
                }
            }

            self.fft
                .fft(&mut h_td, &mut self.h[self.partition_to_constrain][ch]);
        }

        self.partition_to_constrain =
            if self.partition_to_constrain < self.current_size_partitions - 1 {
                self.partition_to_constrain + 1
            } else {
                0
            };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_buffer::BlockBuffer;
    use crate::fft_buffer::FftBuffer;
    use crate::spectrum_buffer::SpectrumBuffer;

    /// Helper to create a RenderBuffer with known FFT data.
    fn make_render_buffer_with_data(
        num_partitions: usize,
        num_channels: usize,
    ) -> (BlockBuffer, SpectrumBuffer, FftBuffer) {
        let bb = BlockBuffer::new(num_partitions, 1, num_channels);
        let sb = SpectrumBuffer::new(num_partitions, num_channels);
        let mut fb = FftBuffer::new(num_partitions, num_channels);

        // Fill FFT buffer with simple test data.
        for p in 0..num_partitions {
            for ch in 0..num_channels {
                for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
                    fb.buffer[p][ch].re[k] = (p * num_channels + ch + k) as f32 * 0.01;
                    fb.buffer[p][ch].im[k] = (p * num_channels + ch + k) as f32 * 0.005;
                }
            }
        }
        (bb, sb, fb)
    }

    #[test]
    fn filter_size() {
        let filter = AdaptiveFirFilter::new(SimdBackend::Scalar, 10, 5, 2, 1);
        assert_eq!(filter.size_partitions(), 5);
        assert_eq!(filter.max_filter_size_partitions(), 10);
    }

    #[test]
    fn filter_size_change_immediate() {
        let mut filter = AdaptiveFirFilter::new(SimdBackend::Scalar, 10, 5, 2, 1);
        filter.set_size_partitions(8, true);
        assert_eq!(filter.size_partitions(), 8);
    }

    #[test]
    fn compute_frequency_response_basic() {
        let num_partitions = 4;
        let num_channels = 1;
        let mut h: Vec<Vec<FftData>> = (0..num_partitions)
            .map(|_| (0..num_channels).map(|_| FftData::default()).collect())
            .collect();

        // Set known values.
        h[0][0].re[0] = 3.0;
        h[0][0].im[0] = 4.0;
        // Expected: 3^2 + 4^2 = 25.0

        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_partitions];
        compute_frequency_response(SimdBackend::Scalar, num_partitions, &h, &mut h2);
        assert!((h2[0][0] - 25.0).abs() < 1e-6);
        // Other bins should be 0.
        assert!((h2[1][0]).abs() < 1e-6);
    }

    #[test]
    fn compute_frequency_response_multichannel_takes_max() {
        let num_partitions = 2;
        let num_channels = 2;
        let mut h: Vec<Vec<FftData>> = (0..num_partitions)
            .map(|_| (0..num_channels).map(|_| FftData::default()).collect())
            .collect();

        // Ch 0: re=1, im=0 → power=1
        h[0][0].re[0] = 1.0;
        // Ch 1: re=2, im=0 → power=4
        h[0][1].re[0] = 2.0;

        let mut h2 = vec![[0.0f32; FFT_LENGTH_BY_2_PLUS_1]; num_partitions];
        compute_frequency_response(SimdBackend::Scalar, num_partitions, &h, &mut h2);
        // Should take max(1, 4) = 4.
        assert!((h2[0][0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn apply_filter_accumulates() {
        let num_partitions = 4;
        let num_channels = 1;
        let (bb, sb, fb) = make_render_buffer_with_data(num_partitions, num_channels);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        // Create filter coefficients — all ones for real, zeros for imag.
        let h: Vec<Vec<FftData>> = (0..num_partitions)
            .map(|_| {
                let mut fft = FftData::default();
                fft.re.fill(1.0);
                vec![fft]
            })
            .collect();

        let mut s = FftData::default();
        apply_filter(
            SimdBackend::Scalar,
            &render_buffer,
            num_partitions,
            &h,
            &mut s,
        );

        // With H=1+0j, S should equal sum of all X across partitions.
        // S.re[k] = sum_p X[p].re[k], S.im[k] = sum_p X[p].im[k]
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            let expected_re: f32 = (0..num_partitions).map(|p| fb.buffer[p][0].re[k]).sum();
            let expected_im: f32 = (0..num_partitions).map(|p| fb.buffer[p][0].im[k]).sum();
            assert!(
                (s.re[k] - expected_re).abs() < 1e-4,
                "re mismatch at k={k}: {} vs {expected_re}",
                s.re[k]
            );
            assert!(
                (s.im[k] - expected_im).abs() < 1e-4,
                "im mismatch at k={k}: {} vs {expected_im}",
                s.im[k]
            );
        }
    }

    #[test]
    fn adapt_partitions_updates_h() {
        let num_partitions = 4;
        let num_channels = 1;
        let (bb, sb, fb) = make_render_buffer_with_data(num_partitions, num_channels);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        let mut h: Vec<Vec<FftData>> = (0..num_partitions)
            .map(|_| vec![FftData::default()])
            .collect();

        // Create gradient with known values.
        let mut g = FftData::default();
        g.re.fill(1.0);

        // After adaptation, H should be non-zero.
        adapt_partitions(
            SimdBackend::Scalar,
            &render_buffer,
            &g,
            num_partitions,
            &mut h,
        );

        let all_zero = h
            .iter()
            .all(|h_p| h_p[0].re.iter().all(|&v| v == 0.0) && h_p[0].im.iter().all(|&v| v == 0.0));
        assert!(!all_zero, "H should be non-zero after adaptation");
    }

    #[test]
    fn adapt_partitions_complex_multiply() {
        // Verify: H += X.re * G.re + X.im * G.im (real part)
        //         H += X.re * G.im - X.im * G.re (imag part)
        let num_partitions = 1;
        let num_channels = 1;
        let (bb, sb, mut fb) = make_render_buffer_with_data(num_partitions, num_channels);

        // Set known X.
        fb.buffer[0][0].re[0] = 3.0;
        fb.buffer[0][0].im[0] = 4.0;

        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        let mut h: Vec<Vec<FftData>> = vec![vec![FftData::default()]];

        let mut g = FftData::default();
        g.re[0] = 2.0;
        g.im[0] = 1.0;

        adapt_partitions(
            SimdBackend::Scalar,
            &render_buffer,
            &g,
            num_partitions,
            &mut h,
        );

        // H.re[0] = X.re * G.re + X.im * G.im = 3*2 + 4*1 = 10
        // H.im[0] = X.re * G.im - X.im * G.re = 3*1 - 4*2 = -5
        assert!(
            (h[0][0].re[0] - 10.0).abs() < 1e-6,
            "re: {} vs 10",
            h[0][0].re[0]
        );
        assert!(
            (h[0][0].im[0] - (-5.0)).abs() < 1e-6,
            "im: {} vs -5",
            h[0][0].im[0]
        );
    }

    #[test]
    fn scale_filter_works() {
        let mut filter = AdaptiveFirFilter::new(SimdBackend::Scalar, 4, 4, 2, 1);
        // Manually set some coefficients.
        filter.h[0][0].re[0] = 2.0;
        filter.h[0][0].im[0] = 3.0;
        filter.scale_filter(0.5);
        assert!((filter.h[0][0].re[0] - 1.0).abs() < 1e-6);
        assert!((filter.h[0][0].im[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn constrain_does_not_panic() {
        let num_partitions = 4;
        let num_channels = 1;
        let (bb, sb, fb) = make_render_buffer_with_data(num_partitions, num_channels);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        let mut filter = AdaptiveFirFilter::new(
            SimdBackend::Scalar,
            num_partitions,
            num_partitions,
            2,
            num_channels,
        );

        let g = FftData::default();
        // Should not panic.
        filter.adapt(&render_buffer, &g);
    }

    #[test]
    fn filter_and_adapt_basic() {
        let num_partitions = 4;
        let num_channels = 1;
        let (bb, sb, fb) = make_render_buffer_with_data(num_partitions, num_channels);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        let mut filter = AdaptiveFirFilter::new(
            SimdBackend::Scalar,
            num_partitions,
            num_partitions,
            2,
            num_channels,
        );

        // Filter output should be zero initially (all H are zero).
        let mut s = FftData::default();
        filter.filter(&render_buffer, &mut s);
        for k in 0..FFT_LENGTH_BY_2_PLUS_1 {
            assert!(s.re[k].abs() < 1e-10);
            assert!(s.im[k].abs() < 1e-10);
        }

        // After adaptation, output should be non-zero.
        let mut g = FftData::default();
        g.re.fill(0.1);
        filter.adapt(&render_buffer, &g);
        filter.filter(&render_buffer, &mut s);

        let has_nonzero =
            s.re.iter().any(|&v| v.abs() > 1e-10) || s.im.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "Output should be non-zero after adaptation");
    }

    #[test]
    fn impulse_response_update() {
        let num_partitions = 4;
        let num_channels = 1;
        let (bb, sb, fb) = make_render_buffer_with_data(num_partitions, num_channels);
        let render_buffer = RenderBuffer::new(&bb, &sb, &fb);

        let mut filter = AdaptiveFirFilter::new(
            SimdBackend::Scalar,
            num_partitions,
            num_partitions,
            2,
            num_channels,
        );
        let mut ir = Vec::with_capacity(get_time_domain_length(num_partitions));

        let mut g = FftData::default();
        g.re.fill(0.1);
        filter.adapt_with_impulse_response(&render_buffer, &g, &mut ir);

        assert_eq!(ir.len(), get_time_domain_length(num_partitions));
    }
}
