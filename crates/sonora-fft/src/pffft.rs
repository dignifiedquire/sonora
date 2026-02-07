//! Variable-size FFT supporting composite sizes, based on FFTPACK.
//!
//! Scalar-only port of the PFFFT library. Supports FFT sizes of the form
//! `N = 2^a * 3^b * 5^c` with minimum size constraints depending on the
//! transform type.
//!
//! # Data format
//!
//! **Ordered mode** (`ordered = true`):
//! - Real forward: `[DC, f1_re, f1_im, f2_re, f2_im, ..., Nyquist]`
//! - Complex forward: standard interleaved `[re0, im0, re1, im1, ...]`
//!
//! **Unordered mode** (`ordered = false`):
//! - Internal FFTPACK layout used for efficient convolution.
//! - Use `convolve_accumulate` on unordered data.

use std::f32::consts::SQRT_2;
use std::f64::consts::PI;

// Trigonometric constants for FFTPACK radix-3 and radix-5 butterflies.
// Named after the C FFTPACK variables they replace.

/// cos(2π/5) ≈ 0.309017
const TR11: f32 = 0.309_017_f32;
/// sin(2π/5) ≈ 0.951057
const TI11: f32 = 0.951_056_5_f32;
/// cos(4π/5) ≈ −0.809017
const TR12: f32 = -0.809_017_f32;
/// sin(4π/5) ≈ 0.587785
const TI12: f32 = 0.587_785_25_f32;
/// cos(2π/3) = −1/2
const TAUR: f32 = -0.5_f32;
/// sin(π/3) = √3/2 ≈ 0.866025
const TAUI: f32 = 0.866_025_4_f32;
/// −1/√2 ≈ −0.707107
const MINUS_HSQT2: f32 = -0.707_106_77_f32;

/// FFT transform type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftType {
    /// Real-valued transform. Input is N reals, output is N/2+1 complex bins.
    Real,
    /// Complex-valued transform. Input/output are N complex values (2N floats).
    Complex,
}

/// Variable-size FFT supporting composite sizes.
///
/// Scalar-only implementation based on FFTPACK. Supports sizes of the form
/// `N = 2^a * 3^b * 5^c` where `a >= 5` for real transforms and `a >= 4`
/// for complex transforms (when using scalar mode with `SIMD_SZ = 1`, the
/// minimums are `a >= 1` for real and `a >= 0` for complex, but we keep
/// the original PFFFT minimum sizes for API compatibility).
///
/// C sources: `webrtc/third_party/pffft/src/pffft.c` (algorithm),
/// `webrtc/modules/audio_processing/utility/pffft_wrapper.cc` (C++ wrapper).
#[derive(Debug, Clone)]
pub struct Pffft {
    n: usize,
    ncvec: usize,
    fft_type: FftType,
    ifac: [i32; 15],
    twiddle: Vec<f32>,
    /// Pre-allocated scratch buffer, reused across transforms
    /// (matches C++ `PffftWrapper::scratch_buffer_`).
    scratch: Vec<f32>,
}

/// Buffer for PFFFT I/O data.
#[derive(Debug, Clone)]
pub struct PffftBuffer {
    data: Vec<f32>,
}

impl PffftBuffer {
    /// View the buffer contents as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// View the buffer contents as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl Pffft {
    /// Create a new FFT setup for the given size and type
    /// (C: `pffft_new_setup`, C++: `PffftWrapper` constructor).
    ///
    /// # Panics
    ///
    /// Panics if `fft_size` is not valid (see [`is_valid_fft_size`]).
    pub fn new(fft_size: usize, fft_type: FftType) -> Self {
        assert!(
            Self::is_valid_fft_size(fft_size, fft_type),
            "invalid FFT size {fft_size} for {fft_type:?}"
        );

        let n = fft_size;
        // With SIMD_SZ=1: Ncvec = N/2 for real, N for complex
        let ncvec = match fft_type {
            FftType::Real => n / 2,
            FftType::Complex => n,
        };

        let mut ifac = [0_i32; 15];

        // Compute twiddle factors.
        // The C code allocates 2*Ncvec floats for twiddle storage.
        let mut twiddle = vec![0.0_f32; 2 * ncvec];
        match fft_type {
            FftType::Real => {
                rffti1(ncvec * 2, &mut twiddle, &mut ifac);
            }
            FftType::Complex => {
                cffti1(ncvec, &mut twiddle, &mut ifac);
            }
        }

        // Verify decomposition is complete.
        let mut m = 1_i32;
        for k in 0..ifac[1] as usize {
            m *= ifac[2 + k];
        }
        let expected = match fft_type {
            FftType::Real => ncvec * 2,
            FftType::Complex => ncvec,
        };
        assert_eq!(
            m as usize, expected,
            "FFT size {n} cannot be decomposed into factors of 2, 3, 5"
        );

        // Pre-allocate scratch buffer (C++ wrapper does this in constructor).
        let buf_len = match fft_type {
            FftType::Real => n,
            FftType::Complex => n * 2,
        };
        let scratch = vec![0.0_f32; buf_len];

        Self {
            n,
            ncvec,
            fft_type,
            ifac,
            twiddle,
            scratch,
        }
    }

    /// Check if the given FFT size is valid for the given type
    /// (C: `pffft_is_valid_size`).
    ///
    /// Valid sizes are of the form `N = 2^a * 3^b * 5^c` where:
    /// - Real: minimum N = 32 (effectively `a >= 1` with scalar, but kept >= 32 for compat)
    /// - Complex: minimum N = 16
    ///
    /// For scalar mode (`SIMD_SZ = 1`):
    /// - Real: `N % 2 == 0`
    /// - Complex: `N % 1 == 0` (any valid decomposition)
    pub fn is_valid_fft_size(fft_size: usize, fft_type: FftType) -> bool {
        if fft_size == 0 {
            return false;
        }

        // PFFFT original constraints (with SIMD_SZ=1):
        // Real: N % (2 * SIMD_SZ * SIMD_SZ) == 0 → N % 2 == 0
        // Complex: N % (SIMD_SZ * SIMD_SZ) == 0 → N % 1 == 0
        match fft_type {
            FftType::Real => {
                if !fft_size.is_multiple_of(2) {
                    return false;
                }
            }
            FftType::Complex => {}
        }

        // Check that N decomposes into only factors of 2, 3, 5.
        let mut n = fft_size;
        while n.is_multiple_of(2) {
            n /= 2;
        }
        while n.is_multiple_of(3) {
            n /= 3;
        }
        while n.is_multiple_of(5) {
            n /= 5;
        }
        n == 1
    }

    /// Create a buffer suitable for this FFT's I/O
    /// (C++: `CreateBuffer`, C: `pffft_aligned_malloc`).
    pub fn create_buffer(&self) -> PffftBuffer {
        let size = match self.fft_type {
            FftType::Real => self.n,
            FftType::Complex => self.n * 2,
        };
        PffftBuffer {
            data: vec![0.0; size],
        }
    }

    /// Forward FFT (C++: `ForwardTransform`, C: `pffft_transform`/`pffft_transform_ordered`).
    ///
    /// If `ordered` is true, the output is in standard frequency-domain order.
    /// If false, the output is in FFTPACK internal order (suitable for
    /// `convolve_accumulate`).
    pub fn forward(&mut self, input: &PffftBuffer, output: &mut PffftBuffer, ordered: bool) {
        self.transform_internal(input, output, Direction::Forward, ordered);
    }

    /// Backward (inverse) FFT (C++: `BackwardTransform`, C: `pffft_transform`/`pffft_transform_ordered`).
    ///
    /// The output is NOT scaled by `1/N`. To recover the original signal,
    /// divide each output element by `N`.
    pub fn backward(&mut self, input: &PffftBuffer, output: &mut PffftBuffer, ordered: bool) {
        self.transform_internal(input, output, Direction::Backward, ordered);
    }

    /// Frequency-domain convolution with accumulation
    /// (C++: `FrequencyDomainConvolution`, C: `pffft_zconvolve_accumulate`).
    ///
    /// Computes `out += scaling * (x * y)` where `x` and `y` are in
    /// unordered FFTPACK layout (from `forward(..., ordered=false)`).
    pub fn convolve_accumulate(
        &self,
        x: &PffftBuffer,
        y: &PffftBuffer,
        out: &mut PffftBuffer,
        scaling: f32,
    ) {
        let a = &x.data;
        let b = &y.data;
        let ab = &mut out.data;
        let mut ncvec = self.ncvec;

        let mut offset = 0;
        if self.fft_type == FftType::Real {
            // Take care of fftpack ordering.
            ab[0] += a[0] * b[0] * scaling;
            ab[2 * ncvec - 1] += a[2 * ncvec - 1] * b[2 * ncvec - 1] * scaling;
            offset = 1;
            ncvec -= 1;
        }

        let a = &a[offset..];
        let b = &b[offset..];
        let ab = &mut ab[offset..];

        for i in 0..ncvec {
            let mut ar = a[2 * i];
            let mut ai = a[2 * i + 1];
            let br = b[2 * i];
            let bi = b[2 * i + 1];
            // VCPLXMUL
            let tmp = ar * bi;
            ar = ar * br - ai * bi;
            ai = ai * br + tmp;
            ab[2 * i] += ar * scaling;
            ab[2 * i + 1] += ai * scaling;
        }
    }

    fn transform_internal(
        &mut self,
        input: &PffftBuffer,
        output: &mut PffftBuffer,
        direction: Direction,
        ordered: bool,
    ) {
        let ncvec = self.ncvec;
        let nf_odd = (self.ifac[1] & 1) != 0;
        let buf_len = output.data.len();

        let ordered_flag = if self.fft_type == FftType::Complex {
            false // Complex transforms are always ordered in scalar mode.
        } else {
            ordered
        };

        // C: ib = (nf_odd ^ ordered ? 1 : 0)
        let ib: bool = nf_odd ^ ordered_flag;

        self.scratch[..buf_len].fill(0.0);

        match direction {
            Direction::Forward => {
                let result = if self.fft_type == FftType::Real {
                    rfftf1(
                        ncvec * 2,
                        &input.data,
                        &mut output.data,
                        &mut self.scratch,
                        &self.twiddle,
                        &self.ifac,
                        ib,
                    )
                } else {
                    cfftf1(
                        ncvec,
                        &input.data,
                        &mut output.data,
                        &mut self.scratch,
                        &self.twiddle,
                        &self.ifac,
                        -1,
                        ib,
                    )
                };

                if result == BufferIndex::Scratch {
                    output.data.copy_from_slice(&self.scratch[..buf_len]);
                }

                if ordered_flag {
                    // Reorder in-place: copy output to scratch, then reorder back.
                    self.scratch[..buf_len].copy_from_slice(&output.data[..buf_len]);
                    zreorder(
                        self.n,
                        self.fft_type,
                        &self.scratch,
                        &mut output.data,
                        Direction::Forward,
                    );
                }
            }
            Direction::Backward => {
                // For ordered backward, reorder from standard to FFTPACK order.
                // We allocate a temporary buffer here because rfftb1/cfftf1 use
                // both output.data and self.scratch as work buffers, leaving no
                // pre-allocated buffer free to hold the reordered input.
                let reordered_buf;
                let src: &[f32] = if ordered_flag {
                    let mut tmp = vec![0.0_f32; buf_len];
                    zreorder(
                        self.n,
                        self.fft_type,
                        &input.data,
                        &mut tmp,
                        Direction::Backward,
                    );
                    reordered_buf = tmp;
                    &reordered_buf
                } else {
                    &input.data
                };

                let result = if self.fft_type == FftType::Real {
                    rfftb1(
                        ncvec * 2,
                        src,
                        &mut output.data,
                        &mut self.scratch,
                        &self.twiddle,
                        &self.ifac,
                        ib,
                    )
                } else {
                    cfftf1(
                        ncvec,
                        src,
                        &mut output.data,
                        &mut self.scratch,
                        &self.twiddle,
                        &self.ifac,
                        1,
                        ib,
                    )
                };

                if result == BufferIndex::Scratch {
                    output.data.copy_from_slice(&self.scratch[..buf_len]);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferIndex {
    Output,
    Scratch,
}

/// Reorder between FFTPACK internal layout and standard frequency-domain order (scalar path).
fn zreorder(n: usize, fft_type: FftType, input: &[f32], output: &mut [f32], direction: Direction) {
    if fft_type == FftType::Complex {
        output[..2 * n].copy_from_slice(&input[..2 * n]);
        return;
    }

    match direction {
        Direction::Forward => {
            let x_n = input[n - 1];
            for k in (2..n).rev() {
                output[k] = input[k - 1];
            }
            output[0] = input[0];
            output[1] = x_n;
        }
        Direction::Backward => {
            let x_n = input[1];
            output[1..n - 1].copy_from_slice(&input[2..n]);
            output[0] = input[0];
            output[n - 1] = x_n;
        }
    }
}

/// Decompose `n` into factors from `ntryh`, storing results in `ifac`.
fn decompose(n: i32, ifac: &mut [i32], ntryh: &[i32]) -> i32 {
    let mut nl = n;
    let mut nf = 0_i32;
    for &ntry in ntryh {
        if ntry == 0 {
            break;
        }
        while nl != 1 {
            let nq = nl / ntry;
            let nr = nl - ntry * nq;
            if nr == 0 {
                ifac[2 + nf as usize] = ntry;
                nf += 1;
                nl = nq;
                if ntry == 2 && nf != 1 {
                    for i in (2..=nf).rev() {
                        ifac[i as usize + 1] = ifac[i as usize];
                    }
                    ifac[2] = 2;
                }
            } else {
                break;
            }
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    nf
}

/// Initialize twiddle factors for real FFT of size `n`.
fn rffti1(n: usize, wa: &mut [f32], ifac: &mut [i32]) {
    let ntryh = [4, 2, 3, 5, 0];
    let nf = decompose(n as i32, ifac, &ntryh);
    let argh = (2.0 * PI) / n as f64;
    let mut is = 0_usize;
    let nfm1 = nf - 1;
    let mut l1 = 1_usize;
    for k1 in 1..=nfm1 as usize {
        let ip = ifac[k1 + 1] as usize;
        let mut ld = 0_usize;
        let l2 = l1 * ip;
        let ido = n / l2;
        let ipm = ip - 1;
        for _j in 1..=ipm {
            let mut fi = 0_usize;
            ld += l1;
            let argld = ld as f64 * argh;
            let mut i = is;
            for _ii in (3..=ido).step_by(2) {
                i += 2;
                fi += 1;
                wa[i - 2] = (fi as f64 * argld).cos() as f32;
                wa[i - 1] = (fi as f64 * argld).sin() as f32;
            }
            is += ido;
        }
        l1 = l2;
    }
}

/// Initialize twiddle factors for complex FFT of size `n`.
fn cffti1(n: usize, wa: &mut [f32], ifac: &mut [i32]) {
    let ntryh = [5, 3, 4, 2, 0];
    let nf = decompose(n as i32, ifac, &ntryh);
    let argh = (2.0 * PI) / n as f64;
    let mut i = 1_usize;
    let mut l1 = 1_usize;
    for k1 in 1..=nf as usize {
        let ip = ifac[k1 + 1] as usize;
        let mut ld = 0_usize;
        let l2 = l1 * ip;
        let ido = n / l2;
        let idot = ido + ido + 2;
        let ipm = ip - 1;
        for _j in 1..=ipm {
            let i1 = i;
            let mut fi = 0_usize;
            wa[i - 1] = 1.0;
            wa[i] = 0.0;
            ld += l1;
            let argld = ld as f64 * argh;
            for _ii in (4..=idot).step_by(2) {
                i += 2;
                fi += 1;
                wa[i - 1] = (fi as f64 * argld).cos() as f32;
                wa[i] = (fi as f64 * argld).sin() as f32;
            }
            if ip > 5 {
                wa[i1 - 1] = wa[i - 1];
                wa[i1] = wa[i];
            }
        }
        l1 = l2;
    }
}

/// Complex radix-2 butterfly pass.
fn passf2(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], fsign: f32) {
    let l1ido = l1 * ido;
    if ido <= 2 {
        let mut ch_off = 0;
        let mut cc_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            ch[ch_off] = cc[cc_off] + cc[cc_off + ido];
            ch[ch_off + l1ido] = cc[cc_off] - cc[cc_off + ido];
            ch[ch_off + 1] = cc[cc_off + 1] + cc[cc_off + ido + 1];
            ch[ch_off + l1ido + 1] = cc[cc_off + 1] - cc[cc_off + ido + 1];
            ch_off += ido;
            cc_off += 2 * ido;
        }
    } else {
        let mut ch_off = 0;
        let mut cc_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            for i in (0..ido - 1).step_by(2) {
                let mut tr2 = cc[cc_off + i] - cc[cc_off + i + ido];
                let mut ti2 = cc[cc_off + i + 1] - cc[cc_off + i + ido + 1];
                let wr = wa1[i];
                let wi = fsign * wa1[i + 1];
                ch[ch_off + i] = cc[cc_off + i] + cc[cc_off + i + ido];
                ch[ch_off + i + 1] = cc[cc_off + i + 1] + cc[cc_off + i + ido + 1];
                vcplxmul(&mut tr2, &mut ti2, wr, wi);
                ch[ch_off + i + l1ido] = tr2;
                ch[ch_off + i + l1ido + 1] = ti2;
            }
            ch_off += ido;
            cc_off += 2 * ido;
        }
    }
}

/// Complex radix-3 butterfly pass.
fn passf3(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], wa2: &[f32], fsign: f32) {
    let taui = TAUI * fsign;
    let l1ido = l1 * ido;

    assert!(ido > 2);
    let mut ch_off = 0;
    let mut cc_off = 0;
    for _k in (0..l1ido).step_by(ido) {
        for i in (0..ido - 1).step_by(2) {
            let tr2 = cc[cc_off + i + ido] + cc[cc_off + i + 2 * ido];
            let cr2 = cc[cc_off + i] + TAUR * tr2;
            ch[ch_off + i] = cc[cc_off + i] + tr2;
            let ti2 = cc[cc_off + i + ido + 1] + cc[cc_off + i + 2 * ido + 1];
            let ci2 = cc[cc_off + i + 1] + TAUR * ti2;
            ch[ch_off + i + 1] = cc[cc_off + i + 1] + ti2;
            let cr3 = taui * (cc[cc_off + i + ido] - cc[cc_off + i + 2 * ido]);
            let ci3 = taui * (cc[cc_off + i + ido + 1] - cc[cc_off + i + 2 * ido + 1]);
            let mut dr2 = cr2 - ci3;
            let mut dr3 = cr2 + ci3;
            let mut di2 = ci2 + cr3;
            let mut di3 = ci2 - cr3;
            let wr1 = wa1[i];
            let wi1 = fsign * wa1[i + 1];
            let wr2 = wa2[i];
            let wi2 = fsign * wa2[i + 1];
            vcplxmul(&mut dr2, &mut di2, wr1, wi1);
            ch[ch_off + i + l1ido] = dr2;
            ch[ch_off + i + l1ido + 1] = di2;
            vcplxmul(&mut dr3, &mut di3, wr2, wi2);
            ch[ch_off + i + 2 * l1ido] = dr3;
            ch[ch_off + i + 2 * l1ido + 1] = di3;
        }
        ch_off += ido;
        cc_off += 3 * ido;
    }
}

/// Complex radix-4 butterfly pass.
#[allow(
    clippy::too_many_arguments,
    reason = "matches FFTPACK radix-4 signature"
)]
fn passf4(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
    fsign: f32,
) {
    let l1ido = l1 * ido;
    if ido == 2 {
        let mut ch_off = 0;
        let mut cc_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            let tr1 = cc[cc_off] - cc[cc_off + 2 * ido];
            let tr2 = cc[cc_off] + cc[cc_off + 2 * ido];
            let ti1 = cc[cc_off + 1] - cc[cc_off + 2 * ido + 1];
            let ti2 = cc[cc_off + 1] + cc[cc_off + 2 * ido + 1];
            let ti4 = (cc[cc_off + ido] - cc[cc_off + 3 * ido]) * fsign;
            let tr4 = (cc[cc_off + 3 * ido + 1] - cc[cc_off + ido + 1]) * fsign;
            let tr3 = cc[cc_off + ido] + cc[cc_off + 3 * ido];
            let ti3 = cc[cc_off + ido + 1] + cc[cc_off + 3 * ido + 1];

            ch[ch_off] = tr2 + tr3;
            ch[ch_off + 1] = ti2 + ti3;
            ch[ch_off + l1ido] = tr1 + tr4;
            ch[ch_off + l1ido + 1] = ti1 + ti4;
            ch[ch_off + 2 * l1ido] = tr2 - tr3;
            ch[ch_off + 2 * l1ido + 1] = ti2 - ti3;
            ch[ch_off + 3 * l1ido] = tr1 - tr4;
            ch[ch_off + 3 * l1ido + 1] = ti1 - ti4;

            ch_off += ido;
            cc_off += 4 * ido;
        }
    } else {
        let mut ch_off = 0;
        let mut cc_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            for i in (0..ido - 1).step_by(2) {
                let tr1 = cc[cc_off + i] - cc[cc_off + i + 2 * ido];
                let tr2 = cc[cc_off + i] + cc[cc_off + i + 2 * ido];
                let ti1 = cc[cc_off + i + 1] - cc[cc_off + i + 2 * ido + 1];
                let ti2 = cc[cc_off + i + 1] + cc[cc_off + i + 2 * ido + 1];
                let tr4 = (cc[cc_off + i + 3 * ido + 1] - cc[cc_off + i + ido + 1]) * fsign;
                let ti4 = (cc[cc_off + i + ido] - cc[cc_off + i + 3 * ido]) * fsign;
                let tr3 = cc[cc_off + i + ido] + cc[cc_off + i + 3 * ido];
                let ti3 = cc[cc_off + i + ido + 1] + cc[cc_off + i + 3 * ido + 1];

                ch[ch_off + i] = tr2 + tr3;
                let mut cr3 = tr2 - tr3;
                ch[ch_off + i + 1] = ti2 + ti3;
                let mut ci3 = ti2 - ti3;

                let mut cr2 = tr1 + tr4;
                let mut cr4 = tr1 - tr4;
                let mut ci2 = ti1 + ti4;
                let mut ci4 = ti1 - ti4;

                let wr1 = wa1[i];
                let wi1 = fsign * wa1[i + 1];
                vcplxmul(&mut cr2, &mut ci2, wr1, wi1);
                let wr2 = wa2[i];
                let wi2 = fsign * wa2[i + 1];
                ch[ch_off + i + l1ido] = cr2;
                ch[ch_off + i + l1ido + 1] = ci2;

                vcplxmul(&mut cr3, &mut ci3, wr2, wi2);
                let wr3 = wa3[i];
                let wi3 = fsign * wa3[i + 1];
                ch[ch_off + i + 2 * l1ido] = cr3;
                ch[ch_off + i + 2 * l1ido + 1] = ci3;

                vcplxmul(&mut cr4, &mut ci4, wr3, wi3);
                ch[ch_off + i + 3 * l1ido] = cr4;
                ch[ch_off + i + 3 * l1ido + 1] = ci4;
            }
            ch_off += ido;
            cc_off += 4 * ido;
        }
    }
}

/// Complex radix-5 butterfly pass.
#[allow(
    clippy::too_many_arguments,
    reason = "matches FFTPACK radix-5 signature"
)]
fn passf5(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
    wa4: &[f32],
    fsign: f32,
) {
    let ti11 = TI11 * fsign;
    let ti12 = TI12 * fsign;

    // C macros (with +1 offset baked in):
    //   cc_ref(a_1, a_2) = cc[(a_2-1)*ido + a_1 + 1]
    //   ch_ref(a_1, a_3) = ch[(a_3-1)*l1*ido + a_1 + 1]
    // C loop: for (i = 0; i < ido-1; i += 2)
    //   cc_ref(i-1, a_2) = cc[(a_2-1)*ido + i]       (real part)
    //   cc_ref(i,   a_2) = cc[(a_2-1)*ido + i + 1]   (imag part)
    // In Rust with cc_off advancing by 5*ido per k:
    //   real: cc[cc_off + (a_2-1)*ido + i]
    //   imag: cc[cc_off + (a_2-1)*ido + i + 1]
    assert!(ido > 2);
    let mut ch_off = 0;
    let mut cc_off = 0;
    let l1ido = l1 * ido;
    for _k in 0..l1 {
        for i in (0..ido - 1).step_by(2) {
            let ti5 = cc[cc_off + ido + i + 1] - cc[cc_off + 4 * ido + i + 1];
            let ti2 = cc[cc_off + ido + i + 1] + cc[cc_off + 4 * ido + i + 1];
            let ti4 = cc[cc_off + 2 * ido + i + 1] - cc[cc_off + 3 * ido + i + 1];
            let ti3 = cc[cc_off + 2 * ido + i + 1] + cc[cc_off + 3 * ido + i + 1];
            let tr5 = cc[cc_off + ido + i] - cc[cc_off + 4 * ido + i];
            let tr2 = cc[cc_off + ido + i] + cc[cc_off + 4 * ido + i];
            let tr4 = cc[cc_off + 2 * ido + i] - cc[cc_off + 3 * ido + i];
            let tr3 = cc[cc_off + 2 * ido + i] + cc[cc_off + 3 * ido + i];

            ch[ch_off + i] = cc[cc_off + i] + tr2 + tr3;
            ch[ch_off + i + 1] = cc[cc_off + i + 1] + ti2 + ti3;
            let cr2 = cc[cc_off + i] + TR11 * tr2 + TR12 * tr3;
            let ci2 = cc[cc_off + i + 1] + TR11 * ti2 + TR12 * ti3;
            let cr3 = cc[cc_off + i] + TR12 * tr2 + TR11 * tr3;
            let ci3 = cc[cc_off + i + 1] + TR12 * ti2 + TR11 * ti3;
            let cr5 = ti11 * tr5 + ti12 * tr4;
            let ci5 = ti11 * ti5 + ti12 * ti4;
            let cr4 = ti12 * tr5 - ti11 * tr4;
            let ci4 = ti12 * ti5 - ti11 * ti4;
            let mut dr3 = cr3 - ci4;
            let mut dr4 = cr3 + ci4;
            let mut di3 = ci3 + cr4;
            let mut di4 = ci3 - cr4;
            let mut dr5 = cr2 + ci5;
            let mut dr2 = cr2 - ci5;
            let mut di5 = ci2 - cr5;
            let mut di2 = ci2 + cr5;

            let wr1 = wa1[i];
            let wi1 = fsign * wa1[i + 1];
            let wr2 = wa2[i];
            let wi2 = fsign * wa2[i + 1];
            let wr3 = wa3[i];
            let wi3 = fsign * wa3[i + 1];
            let wr4 = wa4[i];
            let wi4 = fsign * wa4[i + 1];
            vcplxmul(&mut dr2, &mut di2, wr1, wi1);
            ch[ch_off + i + l1ido] = dr2;
            ch[ch_off + i + 1 + l1ido] = di2;
            vcplxmul(&mut dr3, &mut di3, wr2, wi2);
            ch[ch_off + i + 2 * l1ido] = dr3;
            ch[ch_off + i + 1 + 2 * l1ido] = di3;
            vcplxmul(&mut dr4, &mut di4, wr3, wi3);
            ch[ch_off + i + 3 * l1ido] = dr4;
            ch[ch_off + i + 1 + 3 * l1ido] = di4;
            vcplxmul(&mut dr5, &mut di5, wr4, wi4);
            ch[ch_off + i + 4 * l1ido] = dr5;
            ch[ch_off + i + 1 + 4 * l1ido] = di5;
        }
        ch_off += ido;
        cc_off += 5 * ido;
    }
}

/// Real forward radix-2 pass.
fn radf2(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32]) {
    let l1ido = l1 * ido;
    for k in (0..l1ido).step_by(ido) {
        let a = cc[k];
        let b = cc[k + l1ido];
        ch[2 * k] = a + b;
        ch[2 * (k + ido) - 1] = a - b;
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        for k in (0..l1ido).step_by(ido) {
            for i in (2..ido).step_by(2) {
                let mut tr2 = cc[i - 1 + k + l1ido];
                let mut ti2 = cc[i + k + l1ido];
                let br = cc[i - 1 + k];
                let bi = cc[i + k];
                vcplxmulconj(&mut tr2, &mut ti2, wa1[i - 2], wa1[i - 1]);
                ch[i + 2 * k] = bi + ti2;
                ch[2 * (k + ido) - i] = ti2 - bi;
                ch[i - 1 + 2 * k] = br + tr2;
                ch[2 * (k + ido) - i - 1] = br - tr2;
            }
        }
        if ido % 2 == 1 {
            return;
        }
    }
    for k in (0..l1ido).step_by(ido) {
        ch[2 * k + ido] = -cc[ido - 1 + k + l1ido];
        ch[2 * k + ido - 1] = cc[k + ido - 1];
    }
}

/// Real backward radix-2 pass.
fn radb2(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32]) {
    let l1ido = l1 * ido;
    for k in (0..l1ido).step_by(ido) {
        let a = cc[2 * k];
        let b = cc[2 * (k + ido) - 1];
        ch[k] = a + b;
        ch[k + l1ido] = a - b;
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        for k in (0..l1ido).step_by(ido) {
            for i in (2..ido).step_by(2) {
                let a = cc[i - 1 + 2 * k];
                let b = cc[2 * (k + ido) - i - 1];
                let c = cc[i + 2 * k];
                let d = cc[2 * (k + ido) - i];
                ch[i - 1 + k] = a + b;
                let mut tr2 = a - b;
                ch[i + k] = c - d;
                let mut ti2 = c + d;
                vcplxmul(&mut tr2, &mut ti2, wa1[i - 2], wa1[i - 1]);
                ch[i - 1 + k + l1ido] = tr2;
                ch[i + k + l1ido] = ti2;
            }
        }
        if ido % 2 == 1 {
            return;
        }
    }
    for k in (0..l1ido).step_by(ido) {
        let a = cc[2 * k + ido - 1];
        let b = cc[2 * k + ido];
        ch[k + ido - 1] = a + a;
        ch[k + ido - 1 + l1ido] = -2.0 * b;
    }
}

/// Real forward radix-3 pass.
fn radf3(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], wa2: &[f32]) {
    for k in 0..l1 {
        let cr2 = cc[(k + l1) * ido] + cc[(k + 2 * l1) * ido];
        ch[3 * k * ido] = cc[k * ido] + cr2;
        ch[(3 * k + 2) * ido] = TAUI * (cc[(k + 2 * l1) * ido] - cc[(k + l1) * ido]);
        ch[ido - 1 + (3 * k + 1) * ido] = cc[k * ido] + TAUR * cr2;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let mut dr2 = cc[i - 1 + (k + l1) * ido];
            let mut di2 = cc[i + (k + l1) * ido];
            vcplxmulconj(&mut dr2, &mut di2, wa1[i - 2], wa1[i - 1]);

            let mut dr3 = cc[i - 1 + (k + 2 * l1) * ido];
            let mut di3 = cc[i + (k + 2 * l1) * ido];
            vcplxmulconj(&mut dr3, &mut di3, wa2[i - 2], wa2[i - 1]);

            let cr2 = dr2 + dr3;
            let ci2 = di2 + di3;
            ch[i - 1 + 3 * k * ido] = cc[i - 1 + k * ido] + cr2;
            ch[i + 3 * k * ido] = cc[i + k * ido] + ci2;
            let tr2 = cc[i - 1 + k * ido] + TAUR * cr2;
            let ti2 = cc[i + k * ido] + TAUR * ci2;
            let tr3 = TAUI * (di2 - di3);
            let ti3 = TAUI * (dr3 - dr2);
            ch[i - 1 + (3 * k + 2) * ido] = tr2 + tr3;
            ch[ic - 1 + (3 * k + 1) * ido] = tr2 - tr3;
            ch[i + (3 * k + 2) * ido] = ti2 + ti3;
            ch[ic + (3 * k + 1) * ido] = ti3 - ti2;
        }
    }
}

/// Real backward radix-3 pass.
fn radb3(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], wa2: &[f32]) {
    let taui_2 = TAUI * 2.0;
    for k in 0..l1 {
        let tr2 = cc[ido - 1 + (3 * k + 1) * ido];
        let tr2 = tr2 + tr2;
        let cr2 = cc[3 * k * ido] + TAUR * tr2;
        ch[k * ido] = cc[3 * k * ido] + tr2;
        let ci3 = taui_2 * cc[(3 * k + 2) * ido];
        ch[(k + l1) * ido] = cr2 - ci3;
        ch[(k + 2 * l1) * ido] = cr2 + ci3;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2 = cc[i - 1 + (3 * k + 2) * ido] + cc[ic - 1 + (3 * k + 1) * ido];
            let cr2 = cc[i - 1 + 3 * k * ido] + TAUR * tr2;
            ch[i - 1 + k * ido] = cc[i - 1 + 3 * k * ido] + tr2;
            let ti2 = cc[i + (3 * k + 2) * ido] - cc[ic + (3 * k + 1) * ido];
            let ci2 = cc[i + 3 * k * ido] + TAUR * ti2;
            ch[i + k * ido] = cc[i + 3 * k * ido] + ti2;
            let cr3 = TAUI * (cc[i - 1 + (3 * k + 2) * ido] - cc[ic - 1 + (3 * k + 1) * ido]);
            let ci3 = TAUI * (cc[i + (3 * k + 2) * ido] + cc[ic + (3 * k + 1) * ido]);
            let mut dr2 = cr2 - ci3;
            let mut dr3 = cr2 + ci3;
            let mut di2 = ci2 + cr3;
            let mut di3 = ci2 - cr3;
            vcplxmul(&mut dr2, &mut di2, wa1[i - 2], wa1[i - 1]);
            ch[i - 1 + (k + l1) * ido] = dr2;
            ch[i + (k + l1) * ido] = di2;
            vcplxmul(&mut dr3, &mut di3, wa2[i - 2], wa2[i - 1]);
            ch[i - 1 + (k + 2 * l1) * ido] = dr3;
            ch[i + (k + 2 * l1) * ido] = di3;
        }
    }
}

/// Real forward radix-4 pass.
fn radf4(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], wa2: &[f32], wa3: &[f32]) {
    let l1ido = l1 * ido;
    {
        let mut cc_off = 0;
        let mut ch_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            let a0 = cc[cc_off];
            let a1 = cc[cc_off + l1ido];
            let a2 = cc[cc_off + 2 * l1ido];
            let a3 = cc[cc_off + 3 * l1ido];
            let tr1 = a1 + a3;
            let tr2 = a0 + a2;
            ch[ch_off + 2 * ido - 1] = a0 - a2;
            ch[ch_off + 2 * ido] = a3 - a1;
            ch[ch_off] = tr1 + tr2;
            ch[ch_off + 4 * ido - 1] = tr2 - tr1;
            cc_off += ido;
            ch_off += 4 * ido;
        }
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        for k in (0..l1ido).step_by(ido) {
            for i in (2..ido).step_by(2) {
                let ic = ido - i;
                let mut cr2 = cc[i - 1 + k + l1ido];
                let mut ci2 = cc[i + k + l1ido];
                vcplxmulconj(&mut cr2, &mut ci2, wa1[i - 2], wa1[i - 1]);

                let mut cr3 = cc[i - 1 + k + 2 * l1ido];
                let mut ci3 = cc[i + k + 2 * l1ido];
                vcplxmulconj(&mut cr3, &mut ci3, wa2[i - 2], wa2[i - 1]);

                let mut cr4 = cc[i - 1 + k + 3 * l1ido];
                let mut ci4 = cc[i + k + 3 * l1ido];
                vcplxmulconj(&mut cr4, &mut ci4, wa3[i - 2], wa3[i - 1]);

                let tr1 = cr2 + cr4;
                let tr4 = cr4 - cr2;
                let tr2 = cc[i - 1 + k] + cr3;
                let tr3 = cc[i - 1 + k] - cr3;
                ch[i - 1 + 4 * k] = tr1 + tr2;
                ch[ic - 1 + 4 * k + 3 * ido] = tr2 - tr1;
                let ti1 = ci2 + ci4;
                let ti4 = ci2 - ci4;
                ch[i - 1 + 4 * k + 2 * ido] = ti4 + tr3;
                ch[ic - 1 + 4 * k + ido] = tr3 - ti4;
                let ti2 = cc[i + k] + ci3;
                let ti3 = cc[i + k] - ci3;
                ch[i + 4 * k] = ti1 + ti2;
                ch[ic + 4 * k + 3 * ido] = ti1 - ti2;
                ch[i + 4 * k + 2 * ido] = tr4 + ti3;
                ch[ic + 4 * k + ido] = tr4 - ti3;
            }
        }
        if ido % 2 == 1 {
            return;
        }
    }
    for k in (0..l1ido).step_by(ido) {
        let a = cc[ido - 1 + k + l1ido];
        let b = cc[ido - 1 + k + 3 * l1ido];
        let c = cc[ido - 1 + k];
        let d = cc[ido - 1 + k + 2 * l1ido];
        let ti1 = MINUS_HSQT2 * (a + b);
        let tr1 = MINUS_HSQT2 * (b - a);
        ch[ido - 1 + 4 * k] = tr1 + c;
        ch[ido - 1 + 4 * k + 2 * ido] = c - tr1;
        ch[4 * k + ido] = ti1 - d;
        ch[4 * k + 3 * ido] = ti1 + d;
    }
}

/// Real backward radix-4 pass.
fn radb4(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32], wa2: &[f32], wa3: &[f32]) {
    let minus_sqrt2 = -SQRT_2;
    let l1ido = l1 * ido;
    {
        let mut cc_off = 0;
        let mut ch_off = 0;
        for _k in (0..l1ido).step_by(ido) {
            let a = cc[cc_off];
            let b = cc[cc_off + 4 * ido - 1];
            let c = cc[cc_off + 2 * ido];
            let d = cc[cc_off + 2 * ido - 1];
            let tr3 = 2.0 * d;
            let tr2 = a + b;
            let tr1 = a - b;
            let tr4 = 2.0 * c;
            ch[ch_off] = tr2 + tr3;
            ch[ch_off + 2 * l1ido] = tr2 - tr3;
            ch[ch_off + l1ido] = tr1 - tr4;
            ch[ch_off + 3 * l1ido] = tr1 + tr4;
            cc_off += 4 * ido;
            ch_off += ido;
        }
    }
    if ido < 2 {
        return;
    }
    if ido != 2 {
        for k in (0..l1ido).step_by(ido) {
            let pc = 4 * k; // C uses: pc = cc - 1 + 4*k, so all accesses need -1
            for i in (2..ido).step_by(2) {
                let tr1 = cc[pc + i - 1] - cc[pc + 4 * ido - i - 1];
                let tr2 = cc[pc + i - 1] + cc[pc + 4 * ido - i - 1];
                let ti4 = cc[pc + 2 * ido + i - 1] - cc[pc + 2 * ido - i - 1];
                let tr3 = cc[pc + 2 * ido + i - 1] + cc[pc + 2 * ido - i - 1];
                ch[i - 1 + k] = tr2 + tr3;
                let mut cr3 = tr2 - tr3;

                let ti3 = cc[pc + 2 * ido + i] - cc[pc + 2 * ido - i];
                let tr4 = cc[pc + 2 * ido + i] + cc[pc + 2 * ido - i];
                let mut cr2 = tr1 - tr4;
                let mut cr4 = tr1 + tr4;

                let ti1 = cc[pc + i] + cc[pc + 4 * ido - i];
                let ti2 = cc[pc + i] - cc[pc + 4 * ido - i];

                ch[i + k] = ti2 + ti3;
                let mut ci3 = ti2 - ti3;
                let mut ci2 = ti1 + ti4;
                let mut ci4 = ti1 - ti4;

                vcplxmul(&mut cr2, &mut ci2, wa1[i - 2], wa1[i - 1]);
                ch[i - 1 + k + l1ido] = cr2;
                ch[i + k + l1ido] = ci2;
                vcplxmul(&mut cr3, &mut ci3, wa2[i - 2], wa2[i - 1]);
                ch[i - 1 + k + 2 * l1ido] = cr3;
                ch[i + k + 2 * l1ido] = ci3;
                vcplxmul(&mut cr4, &mut ci4, wa3[i - 2], wa3[i - 1]);
                ch[i - 1 + k + 3 * l1ido] = cr4;
                ch[i + k + 3 * l1ido] = ci4;
            }
        }
        if ido % 2 == 1 {
            return;
        }
    }
    for k in (0..l1ido).step_by(ido) {
        let i0 = 4 * k + ido;
        let c = cc[i0 - 1];
        let d = cc[i0 + 2 * ido - 1];
        let a = cc[i0];
        let b = cc[i0 + 2 * ido];
        let tr1 = c - d;
        let tr2 = c + d;
        let ti1 = b + a;
        let ti2 = b - a;
        ch[ido - 1 + k] = tr2 + tr2;
        ch[ido - 1 + k + l1ido] = minus_sqrt2 * (ti1 - tr1);
        ch[ido - 1 + k + 2 * l1ido] = ti2 + ti2;
        ch[ido - 1 + k + 3 * l1ido] = minus_sqrt2 * (ti1 + tr1);
    }
}

/// Real forward radix-5 pass.
#[allow(
    clippy::too_many_arguments,
    reason = "matches FFTPACK radix-5 signature"
)]
fn radf5(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
    wa4: &[f32],
) {
    // 1-based indexing closures matching C macros, with -1 for 0-based arrays.
    let cc_ref =
        |a1: usize, a2: usize, a3: usize| -> usize { ((a3 - 1) * l1 + (a2 - 1)) * ido + (a1 - 1) };
    let ch_ref =
        |a1: usize, a2: usize, a3: usize| -> usize { ((a3 - 1) * 5 + (a2 - 1)) * ido + (a1 - 1) };

    for k in 1..=l1 {
        let cr2 = cc[cc_ref(1, k, 5)] + cc[cc_ref(1, k, 2)];
        let ci5 = cc[cc_ref(1, k, 5)] - cc[cc_ref(1, k, 2)];
        let cr3 = cc[cc_ref(1, k, 4)] + cc[cc_ref(1, k, 3)];
        let ci4 = cc[cc_ref(1, k, 4)] - cc[cc_ref(1, k, 3)];
        ch[ch_ref(1, 1, k)] = cc[cc_ref(1, k, 1)] + cr2 + cr3;
        ch[ch_ref(ido, 2, k)] = cc[cc_ref(1, k, 1)] + TR11 * cr2 + TR12 * cr3;
        ch[ch_ref(1, 3, k)] = TI11 * ci5 + TI12 * ci4;
        ch[ch_ref(ido, 4, k)] = cc[cc_ref(1, k, 1)] + TR12 * cr2 + TR11 * cr3;
        ch[ch_ref(1, 5, k)] = TI12 * ci5 - TI11 * ci4;
    }
    if ido == 1 {
        return;
    }
    let idp2 = ido + 2;
    for k in 1..=l1 {
        for i in (3..=ido).step_by(2) {
            let ic = idp2 - i;
            let mut dr2 = wa1[i - 3];
            let mut di2 = wa1[i - 2];
            let mut dr3 = wa2[i - 3];
            let mut di3 = wa2[i - 2];
            let mut dr4 = wa3[i - 3];
            let mut di4 = wa3[i - 2];
            let mut dr5 = wa4[i - 3];
            let mut di5 = wa4[i - 2];
            vcplxmulconj(
                &mut dr2,
                &mut di2,
                cc[cc_ref(i - 1, k, 2)],
                cc[cc_ref(i, k, 2)],
            );
            vcplxmulconj(
                &mut dr3,
                &mut di3,
                cc[cc_ref(i - 1, k, 3)],
                cc[cc_ref(i, k, 3)],
            );
            vcplxmulconj(
                &mut dr4,
                &mut di4,
                cc[cc_ref(i - 1, k, 4)],
                cc[cc_ref(i, k, 4)],
            );
            vcplxmulconj(
                &mut dr5,
                &mut di5,
                cc[cc_ref(i - 1, k, 5)],
                cc[cc_ref(i, k, 5)],
            );
            let cr2 = dr2 + dr5;
            let ci5 = dr5 - dr2;
            let cr5 = di2 - di5;
            let ci2 = di2 + di5;
            let cr3 = dr3 + dr4;
            let ci4 = dr4 - dr3;
            let cr4 = di3 - di4;
            let ci3 = di3 + di4;
            ch[ch_ref(i - 1, 1, k)] = cc[cc_ref(i - 1, k, 1)] + cr2 + cr3;
            ch[ch_ref(i, 1, k)] = cc[cc_ref(i, k, 1)] - (ci2 + ci3);
            let tr2 = cc[cc_ref(i - 1, k, 1)] + TR11 * cr2 + TR12 * cr3;
            let ti2 = cc[cc_ref(i, k, 1)] - (TR11 * ci2 + TR12 * ci3);
            let tr3 = cc[cc_ref(i - 1, k, 1)] + TR12 * cr2 + TR11 * cr3;
            let ti3 = cc[cc_ref(i, k, 1)] - (TR12 * ci2 + TR11 * ci3);
            let tr5 = TI11 * cr5 + TI12 * cr4;
            let ti5 = TI11 * ci5 + TI12 * ci4;
            let tr4 = TI12 * cr5 - TI11 * cr4;
            let ti4 = TI12 * ci5 - TI11 * ci4;
            ch[ch_ref(i - 1, 3, k)] = tr2 - tr5;
            ch[ch_ref(ic - 1, 2, k)] = tr2 + tr5;
            ch[ch_ref(i, 3, k)] = ti2 + ti5;
            ch[ch_ref(ic, 2, k)] = ti5 - ti2;
            ch[ch_ref(i - 1, 5, k)] = tr3 - tr4;
            ch[ch_ref(ic - 1, 4, k)] = tr3 + tr4;
            ch[ch_ref(i, 5, k)] = ti3 + ti4;
            ch[ch_ref(ic, 4, k)] = ti4 - ti3;
        }
    }
}

/// Real backward radix-5 pass.
#[allow(
    clippy::too_many_arguments,
    reason = "matches FFTPACK radix-5 signature"
)]
fn radb5(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
    wa4: &[f32],
) {
    let cc_ref =
        |a1: usize, a2: usize, a3: usize| -> usize { ((a3 - 1) * 5 + (a2 - 1)) * ido + (a1 - 1) };
    let ch_ref =
        |a1: usize, a2: usize, a3: usize| -> usize { ((a3 - 1) * l1 + (a2 - 1)) * ido + (a1 - 1) };

    for k in 1..=l1 {
        let ti5 = cc[cc_ref(1, 3, k)] + cc[cc_ref(1, 3, k)];
        let ti4 = cc[cc_ref(1, 5, k)] + cc[cc_ref(1, 5, k)];
        let tr2 = cc[cc_ref(ido, 2, k)] + cc[cc_ref(ido, 2, k)];
        let tr3 = cc[cc_ref(ido, 4, k)] + cc[cc_ref(ido, 4, k)];
        ch[ch_ref(1, k, 1)] = cc[cc_ref(1, 1, k)] + tr2 + tr3;
        let cr2 = cc[cc_ref(1, 1, k)] + TR11 * tr2 + TR12 * tr3;
        let cr3 = cc[cc_ref(1, 1, k)] + TR12 * tr2 + TR11 * tr3;
        let ci5 = TI11 * ti5 + TI12 * ti4;
        let ci4 = TI12 * ti5 - TI11 * ti4;
        ch[ch_ref(1, k, 2)] = cr2 - ci5;
        ch[ch_ref(1, k, 3)] = cr3 - ci4;
        ch[ch_ref(1, k, 4)] = cr3 + ci4;
        ch[ch_ref(1, k, 5)] = cr2 + ci5;
    }
    if ido == 1 {
        return;
    }
    let idp2 = ido + 2;
    for k in 1..=l1 {
        for i in (3..=ido).step_by(2) {
            let ic = idp2 - i;
            let ti5 = cc[cc_ref(i, 3, k)] + cc[cc_ref(ic, 2, k)];
            let ti2 = cc[cc_ref(i, 3, k)] - cc[cc_ref(ic, 2, k)];
            let ti4 = cc[cc_ref(i, 5, k)] + cc[cc_ref(ic, 4, k)];
            let ti3 = cc[cc_ref(i, 5, k)] - cc[cc_ref(ic, 4, k)];
            let tr5 = cc[cc_ref(i - 1, 3, k)] - cc[cc_ref(ic - 1, 2, k)];
            let tr2 = cc[cc_ref(i - 1, 3, k)] + cc[cc_ref(ic - 1, 2, k)];
            let tr4 = cc[cc_ref(i - 1, 5, k)] - cc[cc_ref(ic - 1, 4, k)];
            let tr3 = cc[cc_ref(i - 1, 5, k)] + cc[cc_ref(ic - 1, 4, k)];
            ch[ch_ref(i - 1, k, 1)] = cc[cc_ref(i - 1, 1, k)] + tr2 + tr3;
            ch[ch_ref(i, k, 1)] = cc[cc_ref(i, 1, k)] + ti2 + ti3;
            let cr2 = cc[cc_ref(i - 1, 1, k)] + TR11 * tr2 + TR12 * tr3;
            let ci2 = cc[cc_ref(i, 1, k)] + TR11 * ti2 + TR12 * ti3;
            let cr3 = cc[cc_ref(i - 1, 1, k)] + TR12 * tr2 + TR11 * tr3;
            let ci3 = cc[cc_ref(i, 1, k)] + TR12 * ti2 + TR11 * ti3;
            let cr5 = TI11 * tr5 + TI12 * tr4;
            let ci5 = TI11 * ti5 + TI12 * ti4;
            let cr4 = TI12 * tr5 - TI11 * tr4;
            let ci4 = TI12 * ti5 - TI11 * ti4;
            let mut dr3 = cr3 - ci4;
            let mut dr4 = cr3 + ci4;
            let mut di3 = ci3 + cr4;
            let mut di4 = ci3 - cr4;
            let mut dr5 = cr2 + ci5;
            let mut dr2 = cr2 - ci5;
            let mut di5 = ci2 - cr5;
            let mut di2 = ci2 + cr5;
            vcplxmul(&mut dr2, &mut di2, wa1[i - 3], wa1[i - 2]);
            vcplxmul(&mut dr3, &mut di3, wa2[i - 3], wa2[i - 2]);
            vcplxmul(&mut dr4, &mut di4, wa3[i - 3], wa3[i - 2]);
            vcplxmul(&mut dr5, &mut di5, wa4[i - 3], wa4[i - 2]);
            ch[ch_ref(i - 1, k, 2)] = dr2;
            ch[ch_ref(i, k, 2)] = di2;
            ch[ch_ref(i - 1, k, 3)] = dr3;
            ch[ch_ref(i, k, 3)] = di3;
            ch[ch_ref(i - 1, k, 4)] = dr4;
            ch[ch_ref(i, k, 4)] = di4;
            ch[ch_ref(i - 1, k, 5)] = dr5;
            ch[ch_ref(i, k, 5)] = di5;
        }
    }
}

/// Complex multiply: `(ar, ai) *= (br, bi)`.
#[inline(always)]
fn vcplxmul(ar: &mut f32, ai: &mut f32, br: f32, bi: f32) {
    let tmp = *ar * bi;
    *ar = *ar * br - *ai * bi;
    *ai = *ai * br + tmp;
}

/// Complex multiply by conjugate: `(ar, ai) *= conj(br, bi)`.
#[inline(always)]
fn vcplxmulconj(ar: &mut f32, ai: &mut f32, br: f32, bi: f32) {
    let tmp = *ar * bi;
    *ar = *ar * br + *ai * bi;
    *ai = *ai * br - tmp;
}

/// Real FFT forward: dispatches through radix passes.
/// Returns which buffer holds the result.
fn rfftf1(
    n: usize,
    input: &[f32],
    work1: &mut [f32],
    work2: &mut [f32],
    wa: &[f32],
    ifac: &[i32],
    start_in_work2: bool,
) -> BufferIndex {
    // Copy input into the starting buffer.
    if start_in_work2 {
        work2[..n].copy_from_slice(&input[..n]);
    } else {
        work1[..n].copy_from_slice(&input[..n]);
    }

    let nf = ifac[1] as usize;
    let mut l2 = n;
    let mut iw = n - 1;
    let mut in_is_work2 = start_in_work2;

    for k1 in 1..=nf {
        let kh = nf - k1;
        let ip = ifac[kh + 2] as usize;
        let l1 = l2 / ip;
        let ido = n / l2;
        iw -= (ip - 1) * ido;

        let (inp, out) = if in_is_work2 {
            (work2 as &[f32], work1 as &mut [f32])
        } else {
            (work1 as &[f32], work2 as &mut [f32])
        };

        match ip {
            5 => {
                let (ix2, ix3, ix4) = (iw + ido, iw + 2 * ido, iw + 3 * ido);
                radf5(
                    ido,
                    l1,
                    inp,
                    out,
                    &wa[iw..],
                    &wa[ix2..],
                    &wa[ix3..],
                    &wa[ix4..],
                );
            }
            4 => {
                let (ix2, ix3) = (iw + ido, iw + 2 * ido);
                radf4(ido, l1, inp, out, &wa[iw..], &wa[ix2..], &wa[ix3..]);
            }
            3 => {
                let ix2 = iw + ido;
                radf3(ido, l1, inp, out, &wa[iw..], &wa[ix2..]);
            }
            2 => {
                radf2(ido, l1, inp, out, &wa[iw..]);
            }
            _ => panic!("unsupported radix {ip}"),
        }
        l2 = l1;
        in_is_work2 = !in_is_work2;
    }

    // The result is in the buffer that was last written to (i.e., out).
    // After the loop, in_is_work2 was flipped, so the result is in the opposite.
    if in_is_work2 {
        BufferIndex::Scratch
    } else {
        BufferIndex::Output
    }
}

/// Real FFT backward: dispatches through radix passes.
fn rfftb1(
    n: usize,
    input: &[f32],
    work1: &mut [f32],
    work2: &mut [f32],
    wa: &[f32],
    ifac: &[i32],
    start_in_work2: bool,
) -> BufferIndex {
    if start_in_work2 {
        work2[..n].copy_from_slice(&input[..n]);
    } else {
        work1[..n].copy_from_slice(&input[..n]);
    }

    let nf = ifac[1] as usize;
    let mut l1 = 1;
    let mut iw = 0;
    let mut in_is_work2 = start_in_work2;

    for k1 in 1..=nf {
        let ip = ifac[k1 + 1] as usize;
        let l2 = ip * l1;
        let ido = n / l2;

        let (inp, out) = if in_is_work2 {
            (work2 as &[f32], work1 as &mut [f32])
        } else {
            (work1 as &[f32], work2 as &mut [f32])
        };

        match ip {
            5 => {
                let (ix2, ix3, ix4) = (iw + ido, iw + 2 * ido, iw + 3 * ido);
                radb5(
                    ido,
                    l1,
                    inp,
                    out,
                    &wa[iw..],
                    &wa[ix2..],
                    &wa[ix3..],
                    &wa[ix4..],
                );
            }
            4 => {
                let (ix2, ix3) = (iw + ido, iw + 2 * ido);
                radb4(ido, l1, inp, out, &wa[iw..], &wa[ix2..], &wa[ix3..]);
            }
            3 => {
                let ix2 = iw + ido;
                radb3(ido, l1, inp, out, &wa[iw..], &wa[ix2..]);
            }
            2 => {
                radb2(ido, l1, inp, out, &wa[iw..]);
            }
            _ => panic!("unsupported radix {ip}"),
        }
        l1 = l2;
        iw += (ip - 1) * ido;
        in_is_work2 = !in_is_work2;
    }

    if in_is_work2 {
        BufferIndex::Scratch
    } else {
        BufferIndex::Output
    }
}

/// Complex FFT: dispatches through radix passes. `isign = -1` for forward, `+1` for backward.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "matches FFTPACK dispatch signature and C indexing pattern"
)]
fn cfftf1(
    n: usize,
    input: &[f32],
    work1: &mut [f32],
    work2: &mut [f32],
    wa: &[f32],
    ifac: &[i32],
    isign: i32,
    start_in_work2: bool,
) -> BufferIndex {
    let len = n * 2; // complex = 2*N floats
    if start_in_work2 {
        work2[..len].copy_from_slice(&input[..len]);
    } else {
        work1[..len].copy_from_slice(&input[..len]);
    }

    let nf = ifac[1] as usize;
    let mut l1 = 1;
    let mut iw = 0;
    let fsign = isign as f32;
    let mut in_is_work2 = start_in_work2;

    for k1 in 2..=nf + 1 {
        let ip = ifac[k1] as usize;
        let l2 = ip * l1;
        let ido = n / l2;
        let idot = ido + ido;

        let (inp, out) = if in_is_work2 {
            (work2 as &[f32], work1 as &mut [f32])
        } else {
            (work1 as &[f32], work2 as &mut [f32])
        };

        match ip {
            5 => {
                let (ix2, ix3, ix4) = (iw + idot, iw + 2 * idot, iw + 3 * idot);
                passf5(
                    idot,
                    l1,
                    inp,
                    out,
                    &wa[iw..],
                    &wa[ix2..],
                    &wa[ix3..],
                    &wa[ix4..],
                    fsign,
                );
            }
            4 => {
                let (ix2, ix3) = (iw + idot, iw + 2 * idot);
                passf4(idot, l1, inp, out, &wa[iw..], &wa[ix2..], &wa[ix3..], fsign);
            }
            2 => {
                passf2(idot, l1, inp, out, &wa[iw..], fsign);
            }
            3 => {
                let ix2 = iw + idot;
                passf3(idot, l1, inp, out, &wa[iw..], &wa[ix2..], fsign);
            }
            _ => panic!("unsupported radix {ip}"),
        }
        l1 = l2;
        iw += (ip - 1) * idot;
        in_is_work2 = !in_is_work2;
    }

    if in_is_work2 {
        BufferIndex::Scratch
    } else {
        BufferIndex::Output
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;

    #[test]
    fn is_valid_fft_size_basic() {
        // Powers of 2.
        assert!(Pffft::is_valid_fft_size(32, FftType::Real));
        assert!(Pffft::is_valid_fft_size(64, FftType::Real));
        assert!(Pffft::is_valid_fft_size(128, FftType::Real));
        assert!(Pffft::is_valid_fft_size(256, FftType::Real));
        assert!(Pffft::is_valid_fft_size(512, FftType::Real));
        assert!(Pffft::is_valid_fft_size(1024, FftType::Real));

        // Composite sizes.
        assert!(Pffft::is_valid_fft_size(48, FftType::Real));
        assert!(Pffft::is_valid_fft_size(96, FftType::Real));
        assert!(Pffft::is_valid_fft_size(160, FftType::Real));
        assert!(Pffft::is_valid_fft_size(480, FftType::Real));

        // Complex.
        assert!(Pffft::is_valid_fft_size(16, FftType::Complex));
        assert!(Pffft::is_valid_fft_size(32, FftType::Complex));

        // Invalid.
        assert!(!Pffft::is_valid_fft_size(0, FftType::Real));
        assert!(!Pffft::is_valid_fft_size(7, FftType::Real));
        assert!(!Pffft::is_valid_fft_size(14, FftType::Real));
        assert!(!Pffft::is_valid_fft_size(1, FftType::Real));
        assert!(!Pffft::is_valid_fft_size(3, FftType::Real)); // odd
    }

    #[test]
    fn real_roundtrip_power_of_two() {
        for &n in &[32, 64, 128, 256, 512] {
            let mut pffft = Pffft::new(n, FftType::Real);
            let mut input = pffft.create_buffer();
            let mut freq = pffft.create_buffer();
            let mut output = pffft.create_buffer();

            // Fill with test signal.
            for (i, v) in input.as_mut_slice().iter_mut().enumerate() {
                *v = (i as f32 * 0.1).sin();
            }
            let original: Vec<f32> = input.as_slice().to_vec();

            pffft.forward(&input, &mut freq, true);
            pffft.backward(&freq, &mut output, true);

            let scale = 1.0 / n as f32;
            for (i, (&o, &r)) in original.iter().zip(output.as_slice().iter()).enumerate() {
                let recovered = r * scale;
                assert!(
                    (o - recovered).abs() < 1e-4,
                    "size {n}, index {i}: original={o}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn real_roundtrip_composite() {
        for &n in &[48, 96, 160, 480] {
            let mut pffft = Pffft::new(n, FftType::Real);
            let mut input = pffft.create_buffer();
            let mut freq = pffft.create_buffer();
            let mut output = pffft.create_buffer();

            for (i, v) in input.as_mut_slice().iter_mut().enumerate() {
                *v = (i as f32 * 0.05).cos();
            }
            let original: Vec<f32> = input.as_slice().to_vec();

            pffft.forward(&input, &mut freq, true);
            pffft.backward(&freq, &mut output, true);

            let scale = 1.0 / n as f32;
            for (i, (&o, &r)) in original.iter().zip(output.as_slice().iter()).enumerate() {
                let recovered = r * scale;
                assert!(
                    (o - recovered).abs() < 1e-3,
                    "size {n}, index {i}: original={o}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn complex_roundtrip() {
        for &n in &[16, 32, 64, 128] {
            let mut pffft = Pffft::new(n, FftType::Complex);
            let mut input = pffft.create_buffer();
            let mut freq = pffft.create_buffer();
            let mut output = pffft.create_buffer();

            for (i, v) in input.as_mut_slice().iter_mut().enumerate() {
                *v = ((i as f32) * 0.07).sin();
            }
            let original: Vec<f32> = input.as_slice().to_vec();

            pffft.forward(&input, &mut freq, true);
            pffft.backward(&freq, &mut output, true);

            let scale = 1.0 / n as f32;
            for (i, (&o, &r)) in original.iter().zip(output.as_slice().iter()).enumerate() {
                let recovered = r * scale;
                assert!(
                    (o - recovered).abs() < 1e-4,
                    "size {n}, index {i}: original={o}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn convolve_accumulate_basic() {
        let n = 512;
        let mut pffft = Pffft::new(n, FftType::Real);
        let mut input = pffft.create_buffer();
        let mut freq_x = pffft.create_buffer();
        let mut freq_y = pffft.create_buffer();
        let mut conv = pffft.create_buffer();
        let mut output = pffft.create_buffer();

        // Create impulse.
        input.as_mut_slice()[0] = 1.0;
        pffft.forward(&input, &mut freq_x, false);

        // Create delayed impulse.
        for v in input.as_mut_slice().iter_mut() {
            *v = 0.0;
        }
        input.as_mut_slice()[1] = 1.0;
        pffft.forward(&input, &mut freq_y, false);

        // Convolve.
        for v in conv.as_mut_slice().iter_mut() {
            *v = 0.0;
        }
        pffft.convolve_accumulate(&freq_x, &freq_y, &mut conv, 1.0);

        // Backward.
        pffft.backward(&conv, &mut output, false);

        // Result should be a delayed impulse at index 1.
        let scale = 1.0 / n as f32;
        let out = output.as_slice();
        assert!(
            (out[0] * scale).abs() < 1e-5,
            "expected ~0 at 0, got {}",
            out[0] * scale
        );
        assert!(
            (out[1] * scale - 1.0).abs() < 1e-4,
            "expected ~1 at 1, got {}",
            out[1] * scale
        );
        for (i, &v) in out.iter().enumerate().skip(2).take(n - 2) {
            assert!(
                (v * scale).abs() < 1e-4,
                "expected ~0 at {i}, got {}",
                v * scale
            );
        }
    }

    #[test]
    fn zero_input() {
        let mut pffft = Pffft::new(64, FftType::Real);
        let input = pffft.create_buffer();
        let mut output = pffft.create_buffer();
        pffft.forward(&input, &mut output, true);
        for (i, &v) in output.as_slice().iter().enumerate() {
            assert_eq!(v, 0.0, "expected zero at {i}, got {v}");
        }
    }

    #[test]
    #[should_panic(expected = "invalid FFT size")]
    fn rejects_invalid_size() {
        let _ = Pffft::new(7, FftType::Real);
    }

    // -- Property tests --

    /// Valid real FFT sizes for property tests (composite 2^a * 3^b * 5^c, >= 32).
    const REAL_SIZES: [usize; 10] = [32, 48, 64, 96, 128, 160, 256, 480, 512, 1024];

    /// Valid complex FFT sizes for property tests.
    const COMPLEX_SIZES: [usize; 6] = [16, 32, 48, 64, 128, 256];

    #[proptest]
    fn real_roundtrip_recovers_signal(
        #[strategy(prop::sample::select(&REAL_SIZES[..]))] n: usize,
        #[strategy(prop::collection::vec(-1.0f32..1.0, #n))] signal: Vec<f32>,
    ) {
        let mut pffft = Pffft::new(n, FftType::Real);
        let mut input = pffft.create_buffer();
        let mut freq = pffft.create_buffer();
        let mut output = pffft.create_buffer();

        input.as_mut_slice().copy_from_slice(&signal);
        pffft.forward(&input, &mut freq, true);
        pffft.backward(&freq, &mut output, true);

        let scale = 1.0 / n as f32;
        for (i, (&o, &r)) in signal.iter().zip(output.as_slice().iter()).enumerate() {
            prop_assert!(
                (o - r * scale).abs() < 1e-3,
                "size {n}, index {i}: original={o}, recovered={}",
                r * scale
            );
        }
    }

    #[proptest]
    fn complex_roundtrip_recovers_signal(
        #[strategy(prop::sample::select(&COMPLEX_SIZES[..]))] n: usize,
        #[strategy(prop::collection::vec(-1.0f32..1.0, #n * 2))] signal: Vec<f32>,
    ) {
        let mut pffft = Pffft::new(n, FftType::Complex);
        let mut input = pffft.create_buffer();
        let mut freq = pffft.create_buffer();
        let mut output = pffft.create_buffer();

        input.as_mut_slice().copy_from_slice(&signal);
        pffft.forward(&input, &mut freq, true);
        pffft.backward(&freq, &mut output, true);

        let scale = 1.0 / n as f32;
        for (i, (&o, &r)) in signal.iter().zip(output.as_slice().iter()).enumerate() {
            prop_assert!(
                (o - r * scale).abs() < 1e-3,
                "size {n}, index {i}: original={o}, recovered={}",
                r * scale
            );
        }
    }

    #[proptest]
    fn unordered_roundtrip_recovers_signal(
        #[strategy(prop::sample::select(&REAL_SIZES[..]))] n: usize,
        #[strategy(prop::collection::vec(-1.0f32..1.0, #n))] signal: Vec<f32>,
    ) {
        let mut pffft = Pffft::new(n, FftType::Real);
        let mut input = pffft.create_buffer();
        let mut freq = pffft.create_buffer();
        let mut output = pffft.create_buffer();

        input.as_mut_slice().copy_from_slice(&signal);
        pffft.forward(&input, &mut freq, false);
        pffft.backward(&freq, &mut output, false);

        let scale = 1.0 / n as f32;
        for (i, (&o, &r)) in signal.iter().zip(output.as_slice().iter()).enumerate() {
            prop_assert!(
                (o - r * scale).abs() < 1e-3,
                "size {n}, index {i}: original={o}, recovered={}",
                r * scale
            );
        }
    }

    #[proptest]
    fn real_linearity_holds(
        #[strategy(prop::sample::select(&REAL_SIZES[..]))] n: usize,
        #[strategy(prop::collection::vec(-1.0f32..1.0, #n))] sig_a: Vec<f32>,
        #[strategy(prop::collection::vec(-1.0f32..1.0, #n))] sig_b: Vec<f32>,
    ) {
        let mut pffft = Pffft::new(n, FftType::Real);
        let mut in_a = pffft.create_buffer();
        let mut in_b = pffft.create_buffer();
        let mut in_sum = pffft.create_buffer();
        let mut freq_a = pffft.create_buffer();
        let mut freq_b = pffft.create_buffer();
        let mut freq_sum = pffft.create_buffer();

        in_a.as_mut_slice().copy_from_slice(&sig_a);
        in_b.as_mut_slice().copy_from_slice(&sig_b);
        for (i, v) in in_sum.as_mut_slice().iter_mut().enumerate() {
            *v = sig_a[i] + sig_b[i];
        }

        pffft.forward(&in_a, &mut freq_a, true);
        pffft.forward(&in_b, &mut freq_b, true);
        pffft.forward(&in_sum, &mut freq_sum, true);

        for (i, ((&fa, &fb), &fs)) in freq_a
            .as_slice()
            .iter()
            .zip(freq_b.as_slice().iter())
            .zip(freq_sum.as_slice().iter())
            .enumerate()
        {
            let expected = fa + fb;
            prop_assert!(
                (fs - expected).abs() < 1e-2,
                "size {n}, index {i}: FFT(a+b)={fs}, FFT(a)+FFT(b)={expected}"
            );
        }
    }
}
