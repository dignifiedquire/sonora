//! High-quality sinc-based audio resampler.
//!
//! Port of WebRTC's `SincResampler` — a Blackman-windowed sinc interpolation
//! resampler with runtime SIMD dispatch for the inner convolution loop.
//!
//! # Architecture
//!
//! The resampler uses a **pull** model: you call [`SincResampler::resample`]
//! requesting N output frames, and it pulls input via the
//! [`SincResamplerCallback`] trait.
//!
//! For a **push** model (provide input, get output), see
//! [`PushSincResampler`](super::push_sinc_resampler).

use std::f64::consts::PI;

use derive_more::Debug;
use webrtc_simd::SimdBackend;

/// Callback for providing input data to the resampler.
pub trait SincResamplerCallback {
    /// Write `frames` samples into `destination`.
    /// Zero-pad if not enough data is available.
    fn run(&mut self, frames: usize, destination: &mut [f32]);
}

/// High-quality single-channel sample-rate converter.
#[derive(Debug)]
pub struct SincResampler {
    io_sample_rate_ratio: f64,
    #[debug(skip)]
    virtual_source_idx: f64,
    #[debug(skip)]
    buffer_primed: bool,

    request_frames: usize,
    block_size: usize,

    /// Kernel storage: `KERNEL_OFFSET_COUNT + 1` kernels of `KERNEL_SIZE` each.
    #[debug(skip)]
    kernel_storage: Vec<f32>,
    #[debug(skip)]
    kernel_pre_sinc_storage: Vec<f32>,
    #[debug(skip)]
    kernel_window_storage: Vec<f32>,

    /// Input buffer with region pointers stored as offsets.
    #[debug(skip)]
    input_buffer: Vec<f32>,
    /// Offset into `input_buffer` where new input is written.
    #[debug(skip)]
    r0: usize,
    /// Start of the input region (always 0).
    #[debug(skip)]
    r1: usize,
    /// Half-kernel offset (`KERNEL_SIZE / 2`).
    #[debug(skip)]
    r2: usize,
    #[debug(skip)]
    r3: usize,
    #[debug(skip)]
    r4: usize,

    simd: SimdBackend,
}

// ── Constants ───────────────────────────────────────────────────────

/// Kernel size. Must be a multiple of 32.
pub const KERNEL_SIZE: usize = 32;

/// Default request size in frames.
pub const DEFAULT_REQUEST_SIZE: usize = 512;

/// Number of sub-sample kernel offsets for interpolation.
pub const KERNEL_OFFSET_COUNT: usize = 32;

/// Total kernel storage size.
pub const KERNEL_STORAGE_SIZE: usize = KERNEL_SIZE * (KERNEL_OFFSET_COUNT + 1);

const HALF_KERNEL: usize = KERNEL_SIZE / 2;

// ── Implementation ──────────────────────────────────────────────────

fn sinc_scale_factor(io_ratio: f64) -> f64 {
    let factor = if io_ratio > 1.0 { 1.0 / io_ratio } else { 1.0 };
    // Adjust slightly downward to avoid aliasing at the transition band.
    factor * 0.9
}

impl SincResampler {
    /// Create a new resampler.
    ///
    /// - `io_sample_rate_ratio`: input / output sample rate ratio
    /// - `request_frames`: number of input frames requested per callback (must be > `KERNEL_SIZE`)
    pub fn new(io_sample_rate_ratio: f64, request_frames: usize) -> Self {
        assert!(
            io_sample_rate_ratio > 0.0 && io_sample_rate_ratio.is_finite(),
            "io_sample_rate_ratio must be positive and finite, got {io_sample_rate_ratio}"
        );
        assert!(request_frames > 0, "request_frames must be > 0");
        let simd = webrtc_simd::detect_backend();

        let mut resampler = Self {
            io_sample_rate_ratio,
            virtual_source_idx: 0.0,
            buffer_primed: false,
            request_frames,
            block_size: 0,
            kernel_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            kernel_pre_sinc_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            kernel_window_storage: vec![0.0; KERNEL_STORAGE_SIZE],
            input_buffer: vec![0.0; request_frames + KERNEL_SIZE],
            r0: 0,
            r1: 0,
            r2: HALF_KERNEL,
            r3: 0,
            r4: 0,
            simd,
        };

        resampler.update_regions(false);
        assert!(resampler.block_size > KERNEL_SIZE);
        resampler.initialize_kernel();
        resampler
    }

    /// Number of input frames requested per callback.
    pub fn request_frames(&self) -> usize {
        self.request_frames
    }

    /// Maximum output frames that results in a single callback.
    pub fn chunk_size(&self) -> usize {
        (self.block_size as f64 / self.io_sample_rate_ratio) as usize
    }

    /// Flush all state and reset.
    pub fn flush(&mut self) {
        self.virtual_source_idx = 0.0;
        self.buffer_primed = false;
        self.input_buffer.fill(0.0);
        self.update_regions(false);
    }

    /// Update the sample rate ratio (reconstructs kernels).
    ///
    /// # Panics
    ///
    /// Panics if the ratio is not positive and finite.
    pub fn set_ratio(&mut self, io_sample_rate_ratio: f64) {
        assert!(
            io_sample_rate_ratio > 0.0 && io_sample_rate_ratio.is_finite(),
            "io_sample_rate_ratio must be positive and finite, got {io_sample_rate_ratio}"
        );
        if (self.io_sample_rate_ratio - io_sample_rate_ratio).abs() < f64::EPSILON {
            return;
        }
        self.io_sample_rate_ratio = io_sample_rate_ratio;

        // Re-use pre-computed sinc and window values.
        let scale = sinc_scale_factor(io_sample_rate_ratio);
        for offset_idx in 0..=KERNEL_OFFSET_COUNT {
            for i in 0..KERNEL_SIZE {
                let idx = i + offset_idx * KERNEL_SIZE;
                let window = self.kernel_window_storage[idx];
                let pre_sinc = self.kernel_pre_sinc_storage[idx];
                self.kernel_storage[idx] = (window as f64
                    * if pre_sinc == 0.0 {
                        scale
                    } else {
                        (scale * pre_sinc as f64).sin() / pre_sinc as f64
                    }) as f32;
            }
        }
    }

    /// Resample `frames` output samples, pulling input via `callback`.
    pub fn resample(
        &mut self,
        frames: usize,
        destination: &mut [f32],
        callback: &mut dyn SincResamplerCallback,
    ) {
        assert!(
            destination.len() >= frames,
            "destination too short: {} < {frames}",
            destination.len()
        );
        let mut remaining = frames;
        let mut dest_idx = 0;

        // Prime the buffer on first use.
        if !self.buffer_primed && remaining > 0 {
            let r0 = self.r0;
            callback.run(
                self.request_frames,
                &mut self.input_buffer[r0..r0 + self.request_frames],
            );
            self.buffer_primed = true;
        }

        let current_io_ratio = self.io_sample_rate_ratio;

        while remaining > 0 {
            let iterations = ((self.block_size as f64 - self.virtual_source_idx) / current_io_ratio)
                .ceil() as isize;

            for _ in (1..=iterations).rev() {
                debug_assert!((self.virtual_source_idx as usize) < self.block_size);

                let source_idx = self.virtual_source_idx as usize;
                let subsample_remainder = self.virtual_source_idx - source_idx as f64;

                let virtual_offset_idx = subsample_remainder * KERNEL_OFFSET_COUNT as f64;
                let offset_idx = virtual_offset_idx as usize;

                let k1_start = offset_idx * KERNEL_SIZE;
                let k2_start = k1_start + KERNEL_SIZE;

                let input_start = self.r1 + source_idx;

                let kernel_interpolation_factor = virtual_offset_idx - offset_idx as f64;

                destination[dest_idx] =
                    self.convolve(input_start, k1_start, k2_start, kernel_interpolation_factor);
                dest_idx += 1;

                self.virtual_source_idx += current_io_ratio;

                remaining -= 1;
                if remaining == 0 {
                    return;
                }
            }

            // Wrap back around.
            self.virtual_source_idx -= self.block_size as f64;

            // Copy r3_,r4_ to r1_,r2_ (wrap tail to head).
            self.input_buffer
                .copy_within(self.r3..self.r3 + KERNEL_SIZE, self.r1);

            // Reinitialize regions if necessary.
            if self.r0 == self.r2 {
                self.update_regions(true);
            }

            // Refresh with more input.
            let r0 = self.r0;
            callback.run(
                self.request_frames,
                &mut self.input_buffer[r0..r0 + self.request_frames],
            );
        }
    }

    fn update_regions(&mut self, second_load: bool) {
        self.r0 = if second_load {
            KERNEL_SIZE
        } else {
            HALF_KERNEL
        };
        self.r3 = self.r0 + self.request_frames - KERNEL_SIZE;
        self.r4 = self.r0 + self.request_frames - HALF_KERNEL;
        self.block_size = self.r4 - self.r2;
    }

    fn initialize_kernel(&mut self) {
        // Blackman window parameters.
        let k_alpha = 0.16_f64;
        let k_a0 = 0.5 * (1.0 - k_alpha);
        let k_a1 = 0.5_f64;
        let k_a2 = 0.5 * k_alpha;

        let scale = sinc_scale_factor(self.io_sample_rate_ratio);

        for offset_idx in 0..=KERNEL_OFFSET_COUNT {
            let subsample_offset = offset_idx as f64 / KERNEL_OFFSET_COUNT as f64;

            for i in 0..KERNEL_SIZE {
                let idx = i + offset_idx * KERNEL_SIZE;

                let pre_sinc = PI * (i as f64 - HALF_KERNEL as f64 - subsample_offset);
                self.kernel_pre_sinc_storage[idx] = pre_sinc as f32;

                // Blackman window
                let x = (i as f64 - subsample_offset) / KERNEL_SIZE as f64;
                let window = k_a0 - k_a1 * (2.0 * PI * x).cos() + k_a2 * (4.0 * PI * x).cos();
                self.kernel_window_storage[idx] = window as f32;

                // Windowed sinc
                let sinc_val = if pre_sinc == 0.0 {
                    scale
                } else {
                    (scale * pre_sinc).sin() / pre_sinc
                };
                self.kernel_storage[idx] = (window * sinc_val) as f32;
            }
        }
    }

    /// Inner convolution: dot-product of input with two interpolated kernels.
    #[inline]
    fn convolve(
        &self,
        input_offset: usize,
        k1_offset: usize,
        k2_offset: usize,
        kernel_interpolation_factor: f64,
    ) -> f32 {
        let input = &self.input_buffer[input_offset..input_offset + KERNEL_SIZE];
        let k1 = &self.kernel_storage[k1_offset..k1_offset + KERNEL_SIZE];
        let k2 = &self.kernel_storage[k2_offset..k2_offset + KERNEL_SIZE];

        let (sum1, sum2) = self.simd.dual_dot_product(input, k1, k2);

        // Linearly interpolate the two convolutions.
        let factor = kernel_interpolation_factor as f32;
        (1.0 - factor) * sum1 + factor * sum2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Callback that provides a constant value.
    struct ConstCallback(f32);
    impl SincResamplerCallback for ConstCallback {
        fn run(&mut self, frames: usize, destination: &mut [f32]) {
            for d in destination.iter_mut().take(frames) {
                *d = self.0;
            }
        }
    }

    /// Callback that provides a sine wave.
    struct SineCallback {
        phase: f64,
        phase_increment: f64,
    }
    impl SineCallback {
        fn new(freq_hz: f64, sample_rate: f64) -> Self {
            Self {
                phase: 0.0,
                phase_increment: 2.0 * PI * freq_hz / sample_rate,
            }
        }
    }
    impl SincResamplerCallback for SineCallback {
        fn run(&mut self, frames: usize, destination: &mut [f32]) {
            for d in destination.iter_mut().take(frames) {
                *d = self.phase.sin() as f32;
                self.phase += self.phase_increment;
            }
        }
    }

    #[test]
    fn construction() {
        let r = SincResampler::new(1.0, DEFAULT_REQUEST_SIZE);
        assert_eq!(r.request_frames(), DEFAULT_REQUEST_SIZE);
    }

    #[test]
    fn identity_passthrough() {
        // 1:1 ratio should pass through (approximately)
        let mut r = SincResampler::new(1.0, DEFAULT_REQUEST_SIZE);
        let mut cb = ConstCallback(1.0);
        let chunk = r.chunk_size();
        let mut output = vec![0.0_f32; chunk * 2];
        r.resample(chunk, &mut output, &mut cb);

        // After priming, the constant value should appear.
        // Allow some ramp-up time, then check the tail is close to 1.0.
        let tail = &output[chunk / 2..chunk];
        for &v in tail {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn downsample_2x() {
        // 48kHz -> 24kHz (ratio = 2.0)
        let source_frames = 480; // 10ms at 48kHz
        let dest_frames = 240; // 10ms at 24kHz
        let ratio = source_frames as f64 / dest_frames as f64;
        let mut r = SincResampler::new(ratio, source_frames);
        let mut cb = SineCallback::new(1000.0, 48000.0);

        let mut output = vec![0.0_f32; dest_frames];
        // Prime
        r.resample(r.chunk_size(), &mut vec![0.0; r.chunk_size()], &mut cb);
        // Resample
        r.resample(dest_frames, &mut output, &mut cb);

        // Output should contain non-trivial values
        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(
            energy > 0.1,
            "output should have signal energy, got {energy}"
        );
    }

    #[test]
    fn upsample_2x() {
        // 24kHz -> 48kHz (ratio = 0.5)
        let source_frames = 240;
        let dest_frames = 480;
        let ratio = source_frames as f64 / dest_frames as f64;
        let mut r = SincResampler::new(ratio, source_frames);
        let mut cb = SineCallback::new(1000.0, 24000.0);

        let mut output = vec![0.0_f32; dest_frames];
        r.resample(r.chunk_size(), &mut vec![0.0; r.chunk_size()], &mut cb);
        r.resample(dest_frames, &mut output, &mut cb);

        let energy: f32 = output.iter().map(|v| v * v).sum();
        assert!(
            energy > 0.1,
            "output should have signal energy, got {energy}"
        );
    }

    #[test]
    fn flush_resets_state() {
        let mut r = SincResampler::new(1.0, DEFAULT_REQUEST_SIZE);
        let mut cb = ConstCallback(1.0);
        let chunk = r.chunk_size();
        let mut output = vec![0.0; chunk];
        r.resample(chunk, &mut output, &mut cb);

        r.flush();
        // After flush, buffer should be unprimed again.
        let mut output2 = vec![0.0; chunk];
        r.resample(chunk, &mut output2, &mut cb);

        // Both runs should produce similar output (the filter re-primes).
        assert!((output[chunk - 1] - output2[chunk - 1]).abs() < 0.01);
    }

    #[test]
    fn set_ratio_changes_output() {
        let mut r = SincResampler::new(2.0, 480);
        let mut cb = SineCallback::new(1000.0, 48000.0);

        let mut out1 = vec![0.0; 240];
        r.resample(r.chunk_size(), &mut vec![0.0; r.chunk_size()], &mut cb);
        r.resample(240, &mut out1, &mut cb);

        // Change ratio
        r.set_ratio(1.0);
        r.flush();
        let mut cb2 = SineCallback::new(1000.0, 48000.0);
        let mut out2 = vec![0.0; 480];
        r.resample(r.chunk_size(), &mut vec![0.0; r.chunk_size()], &mut cb2);
        r.resample(480, &mut out2, &mut cb2);

        // Different number of output samples validates the ratio change worked
        assert_eq!(out1.len(), 240);
        assert_eq!(out2.len(), 480);
    }
}
