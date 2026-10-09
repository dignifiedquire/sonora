//! Variable-size real FFT using Ooura's fft4g algorithm.
//!
//! Port of WebRTC's `WebRtc_rdft` — a radix-4/2 decimation-in-frequency FFT
//! supporting any power-of-2 size. Twiddle tables are computed once at
//! construction time.
//!
//! # Data format
//!
//! Forward transform (`rdft`):
//! - Input: `n` real-valued time-domain samples
//! - Output: `a[2*k] = R[k]`, `a[2*k+1] = I[k]` for `0 <= k < n/2`,
//!   with `a[1] = R[n/2]` (Nyquist packed into index 1)
//!
//! Inverse transform (`irdft`):
//! - Input: Packed frequency-domain data (as produced by `rdft`)
//! - Output: `n` real-valued time-domain samples
//! - To recover the original signal, multiply each element by `2/n`
//!
//! # Safety: unchecked indexing
//!
//! The internal FFT routines (`apply_bitrv2`, `cft1st`, `cftmdl`, etc.) use
//! unchecked array accesses in their inner loops. This is necessary because
//! the Ooura algorithm computes array indices through arithmetic on loop
//! variables and twiddle offsets, and LLVM cannot prove these are in-bounds.
//! The resulting bounds checks add significant overhead (measured at 2x for
//! bit-reversal) compared to the C++ original which has no such checks.
//!
//! All indices are bounded by `n` (the FFT size), which equals `a.len()`.
//! This is guaranteed by the algorithm's structure — see the C reference in
//! `fft4g.cc` — and validated by the extensive test suite (roundtrip,
//! Parseval, linearity, property tests across sizes 4–512).

use std::f32::consts::FRAC_PI_4;
use std::ptr;

use sonora_simd::NATIVE_FMA;

/// Variable-size real FFT using Ooura's fft4g algorithm.
///
/// Supports power-of-2 sizes (`n >= 2`). Twiddle tables and bit-reversal
/// indices are eagerly computed during construction.
///
/// C source: `webrtc/common_audio/third_party/ooura/fft_size_256/fft4g.cc`
/// (`rdft` function).
#[derive(Debug, Clone)]
pub struct Fft4g {
    n: usize,
    nw: usize,
    nc: usize,
    /// Precomputed bit-reversal permutation indices.
    bitrv_ip: Vec<usize>,
    /// Precomputed m value for bit-reversal (determines loop bounds).
    bitrv_m: usize,
    /// Whether to use the 4-group swap pattern (true) or 2-group (false).
    bitrv_long: bool,
    /// Twiddle factor table: `w[0..nw-1]` = cos/sin for complex FFT,
    /// `w[nw..nw+nc-1]` = cos/sin for real FFT post-processing.
    w: Vec<f32>,
}

impl Fft4g {
    /// Create a new FFT for size `n` (must be a power of 2, `n >= 2`).
    ///
    /// # Panics
    ///
    /// Panics if `n < 2` or `n` is not a power of 2.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "FFT size must be >= 2, got {n}");
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of 2, got {n}"
        );

        // Allocate work arrays with sizes matching the C implementation.
        let nw = n >> 2;
        let ip_len = 2 + (1 << ((n / 2).ilog2() as usize / 2));
        let mut ip = vec![0_usize; ip_len.max(4)];
        let mut w = vec![0.0_f32; n / 2];

        // Initialize twiddle tables eagerly (C++ does this lazily on first call).
        if nw > 0 {
            makewt(nw, &mut ip, &mut w);
        }
        let nc = n >> 2;
        if nc > 0 {
            makect(nc, &mut ip, &mut w[nw..]);
        }

        // Precompute the bit-reversal index table (C rebuilds this every call).
        let (bitrv_ip, bitrv_m, bitrv_long) = build_bitrv_table(n);

        Self {
            n,
            nw,
            nc,
            bitrv_ip,
            bitrv_m,
            bitrv_long,
            w,
        }
    }

    /// Forward real DFT in-place (C: `rdft` with `isgn=1`).
    ///
    /// `a` must have length `n`. After the call:
    /// - `a[0]` = DC component
    /// - `a[1]` = Nyquist component
    /// - `a[2*k], a[2*k+1]` = real/imaginary of frequency bin k
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != n`.
    pub fn rdft(&self, a: &mut [f32]) {
        self.rdft_with::<NATIVE_FMA>(a);
    }

    /// [`Self::rdft`], with the twiddle multiplications fused by
    /// [`f32::mul_add`] if `FMA`, else in the plain expressions of the C
    /// reference and their operation order. The public methods pass
    /// [`NATIVE_FMA`]; the tests run both forms. Always inlined, so that
    /// LLVM optimizes this body as part of `rdft`.
    #[inline(always)]
    fn rdft_with<const FMA: bool>(&self, a: &mut [f32]) {
        assert_eq!(a.len(), self.n, "input length must be {}", self.n);
        let n = self.n;

        if n > 4 {
            apply_bitrv2(&self.bitrv_ip, self.bitrv_m, self.bitrv_long, a);
            cftfsub::<FMA>(n, a, &self.w);
            rftfsub::<FMA>(n, a, self.nc, &self.w[self.nw..]);
        } else if n == 4 {
            cftfsub::<FMA>(n, a, &self.w);
        }
        let xi = a[0] - a[1];
        a[0] += a[1];
        a[1] = xi;
    }

    /// Inverse real DFT in-place (C: `rdft` with `isgn=-1`).
    ///
    /// `a` must have length `n`. To recover the original signal,
    /// multiply each output element by `2/n`.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != n`.
    pub fn irdft(&self, a: &mut [f32]) {
        self.irdft_with::<NATIVE_FMA>(a);
    }

    /// [`Self::irdft`], in the form `FMA` selects; see [`Self::rdft_with`].
    #[inline(always)]
    fn irdft_with<const FMA: bool>(&self, a: &mut [f32]) {
        assert_eq!(a.len(), self.n, "input length must be {}", self.n);
        let n = self.n;

        a[1] = 0.5 * (a[0] - a[1]);
        a[0] -= a[1];
        if n > 4 {
            rftbsub::<FMA>(n, a, self.nc, &self.w[self.nw..]);
            apply_bitrv2(&self.bitrv_ip, self.bitrv_m, self.bitrv_long, a);
            cftbsub::<FMA>(n, a, &self.w);
        } else if n == 4 {
            cftfsub::<FMA>(n, a, &self.w);
        }
    }
}

/// Initialize cos/sin twiddle tables for the complex sub-transforms.
fn makewt(nw: usize, ip: &mut [usize], w: &mut [f32]) {
    ip[0] = nw;
    ip[1] = 1;
    if nw > 2 {
        let nwh = nw >> 1;
        let delta = FRAC_PI_4 / nwh as f32;
        w[0] = 1.0;
        w[1] = 0.0;
        w[nwh] = (delta * nwh as f32).cos();
        w[nwh + 1] = w[nwh];
        if nwh > 2 {
            for j in (2..nwh).step_by(2) {
                let x = (delta * j as f32).cos();
                let y = (delta * j as f32).sin();
                w[j] = x;
                w[j + 1] = y;
                w[nw - j] = y;
                w[nw - j + 1] = x;
            }
            bitrv2(nw, &mut ip[2..], w);
        }
    }
}

/// Initialize cos table for the real FFT post/pre-processing.
fn makect(nc: usize, ip: &mut [usize], c: &mut [f32]) {
    ip[1] = nc;
    if nc > 1 {
        let nch = nc >> 1;
        let delta = FRAC_PI_4 / nch as f32;
        c[0] = (delta * nch as f32).cos();
        c[nch] = 0.5 * c[0];
        for j in 1..nch {
            c[j] = 0.5 * (delta * j as f32).cos();
            c[nc - j] = 0.5 * (delta * j as f32).sin();
        }
    }
}

/// Build the bit-reversal index table for size `n`.
///
/// Returns `(ip, m, use_long_swap)` where `ip` is the index table, `m`
/// determines the loop bounds, and `use_long_swap` selects between the
/// 4-group and 2-group swap patterns. Called once during construction.
fn build_bitrv_table(n: usize) -> (Vec<usize>, usize, bool) {
    let mut ip = vec![0_usize; n];
    ip[0] = 0;
    let mut l = n;
    let mut m = 1_usize;
    while (m << 3) < l {
        l >>= 1;
        for j in 0..m {
            ip[m + j] = ip[j] + l;
        }
        m <<= 1;
    }
    let use_long_swap = (m << 3) == l;
    ip.truncate(2 * m); // only need up to 2*m entries
    (ip, m, use_long_swap)
}

/// Swap `a[i]` and `a[j]` without bounds checking.
///
/// # Safety
/// Both `i` and `j` must be in bounds for `a`.
#[inline(always)]
unsafe fn swap_unchecked(a: &mut [f32], i: usize, j: usize) {
    debug_assert!(i < a.len() && j < a.len());
    unsafe {
        ptr::swap(a.as_mut_ptr().add(i), a.as_mut_ptr().add(j));
    }
}

/// Read `a[i]` without bounds checking.
#[inline(always)]
unsafe fn get(a: &[f32], i: usize) -> f32 {
    debug_assert!(i < a.len());
    unsafe { *a.get_unchecked(i) }
}

/// Write `a[i] = v` without bounds checking.
#[inline(always)]
unsafe fn set(a: &mut [f32], i: usize, v: f32) {
    debug_assert!(i < a.len());
    unsafe {
        *a.get_unchecked_mut(i) = v;
    }
}

/// Apply precomputed bit-reversal permutation to `a`.
///
/// All indices are bounded by `a.len()` (the FFT size `n`), which is
/// guaranteed by the precomputed `ip` table and the structure of the
/// bit-reversal algorithm.
fn apply_bitrv2(ip: &[usize], m: usize, use_long_swap: bool, a: &mut [f32]) {
    let m2 = 2 * m;
    if use_long_swap {
        for k in 0..m {
            for j in 0..k {
                let j1 = 2 * j + ip[k];
                let k1 = 2 * k + ip[j];
                // SAFETY: All indices are < n (the FFT size = a.len()).
                // The bit-reversal table `ip` contains values < n/2, and
                // the maximum index is 3*m2 + 2*(m-1) + ip[m-1] + 1 < 4*m2 = n.
                unsafe {
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                    let j1 = j1 + m2;
                    let k1 = k1 + 2 * m2;
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                    let j1 = j1 + m2;
                    let k1 = k1 - m2;
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                    let j1 = j1 + m2;
                    let k1 = k1 + 2 * m2;
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                }
            }
            let j1 = 2 * k + m2 + ip[k];
            let k1 = j1 + m2;
            // SAFETY: j1 + m2 + 1 < 4*m2 = n.
            unsafe {
                swap_unchecked(a, j1, k1);
                swap_unchecked(a, j1 + 1, k1 + 1);
            }
        }
    } else {
        for k in 1..m {
            for j in 0..k {
                let j1 = 2 * j + ip[k];
                let k1 = 2 * k + ip[j];
                // SAFETY: j1 + m2 + 1 < 2*m2 = n, k1 + m2 + 1 < 2*m2 = n.
                unsafe {
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                    let j1 = j1 + m2;
                    let k1 = k1 + m2;
                    swap_unchecked(a, j1, k1);
                    swap_unchecked(a, j1 + 1, k1 + 1);
                }
            }
        }
    }
}

/// Build bit-reversal table and apply permutation in one pass (used during init).
fn bitrv2(n: usize, ip: &mut [usize], a: &mut [f32]) {
    ip[0] = 0;
    let mut l = n;
    let mut m = 1_usize;
    while (m << 3) < l {
        l >>= 1;
        for j in 0..m {
            ip[m + j] = ip[j] + l;
        }
        m <<= 1;
    }
    let m2 = 2 * m;
    if (m << 3) == l {
        for k in 0..m {
            for j in 0..k {
                let j1 = 2 * j + ip[k];
                let k1 = 2 * k + ip[j];
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
                let j1 = j1 + m2;
                let k1 = k1 + 2 * m2;
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
                let j1 = j1 + m2;
                let k1 = k1 - m2;
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
                let j1 = j1 + m2;
                let k1 = k1 + 2 * m2;
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
            }
            let j1 = 2 * k + m2 + ip[k];
            let k1 = j1 + m2;
            a.swap(j1, k1);
            a.swap(j1 + 1, k1 + 1);
        }
    } else {
        for k in 1..m {
            for j in 0..k {
                let j1 = 2 * j + ip[k];
                let k1 = 2 * k + ip[j];
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
                let j1 = j1 + m2;
                let k1 = k1 + m2;
                a.swap(j1, k1);
                a.swap(j1 + 1, k1 + 1);
            }
        }
    }
}

/// Forward complex sub-transform (radix-4 decomposition).
fn cftfsub<const FMA: bool>(n: usize, a: &mut [f32], w: &[f32]) {
    let mut l = 2;
    if n > 8 {
        cft1st::<FMA>(n, a, w);
        l = 8;
        while (l << 2) < n {
            cftmdl::<FMA>(n, l, a, w);
            l <<= 2;
        }
    }
    // SAFETY: All indices are bounded by `n` (see module-level doc).
    unsafe {
        if (l << 2) == n {
            for j in (0..l).step_by(2) {
                let j1 = j + l;
                let j2 = j1 + l;
                let j3 = j2 + l;
                let x0r = get(a, j) + get(a, j1);
                let x0i = get(a, j + 1) + get(a, j1 + 1);
                let x1r = get(a, j) - get(a, j1);
                let x1i = get(a, j + 1) - get(a, j1 + 1);
                let x2r = get(a, j2) + get(a, j3);
                let x2i = get(a, j2 + 1) + get(a, j3 + 1);
                let x3r = get(a, j2) - get(a, j3);
                let x3i = get(a, j2 + 1) - get(a, j3 + 1);
                set(a, j, x0r + x2r);
                set(a, j + 1, x0i + x2i);
                set(a, j2, x0r - x2r);
                set(a, j2 + 1, x0i - x2i);
                set(a, j1, x1r - x3i);
                set(a, j1 + 1, x1i + x3r);
                set(a, j3, x1r + x3i);
                set(a, j3 + 1, x1i - x3r);
            }
        } else {
            for j in (0..l).step_by(2) {
                let j1 = j + l;
                let x0r = get(a, j) - get(a, j1);
                let x0i = get(a, j + 1) - get(a, j1 + 1);
                set(a, j, get(a, j) + get(a, j1));
                set(a, j + 1, get(a, j + 1) + get(a, j1 + 1));
                set(a, j1, x0r);
                set(a, j1 + 1, x0i);
            }
        }
    }
}

/// Backward complex sub-transform (radix-4 decomposition).
fn cftbsub<const FMA: bool>(n: usize, a: &mut [f32], w: &[f32]) {
    let mut l = 2;
    if n > 8 {
        cft1st::<FMA>(n, a, w);
        l = 8;
        while (l << 2) < n {
            cftmdl::<FMA>(n, l, a, w);
            l <<= 2;
        }
    }
    // SAFETY: All indices are bounded by `n` (see module-level doc).
    unsafe {
        if (l << 2) == n {
            for j in (0..l).step_by(2) {
                let j1 = j + l;
                let j2 = j1 + l;
                let j3 = j2 + l;
                let x0r = get(a, j) + get(a, j1);
                let x0i = -get(a, j + 1) - get(a, j1 + 1);
                let x1r = get(a, j) - get(a, j1);
                let x1i = -get(a, j + 1) + get(a, j1 + 1);
                let x2r = get(a, j2) + get(a, j3);
                let x2i = get(a, j2 + 1) + get(a, j3 + 1);
                let x3r = get(a, j2) - get(a, j3);
                let x3i = get(a, j2 + 1) - get(a, j3 + 1);
                set(a, j, x0r + x2r);
                set(a, j + 1, x0i - x2i);
                set(a, j2, x0r - x2r);
                set(a, j2 + 1, x0i + x2i);
                set(a, j1, x1r - x3i);
                set(a, j1 + 1, x1i - x3r);
                set(a, j3, x1r + x3i);
                set(a, j3 + 1, x1i + x3r);
            }
        } else {
            for j in (0..l).step_by(2) {
                let j1 = j + l;
                let x0r = get(a, j) - get(a, j1);
                let x0i = -get(a, j + 1) + get(a, j1 + 1);
                set(a, j, get(a, j) + get(a, j1));
                set(a, j + 1, -get(a, j + 1) - get(a, j1 + 1));
                set(a, j1, x0r);
                set(a, j1 + 1, x0i);
            }
        }
    }
}

/// First-stage complex FFT butterfly.
///
/// # Safety contract
/// `a.len() >= n >= 16` and `w.len() >= n/4`. All indices stay within bounds.
fn cft1st<const FMA: bool>(n: usize, a: &mut [f32], w: &[f32]) {
    // SAFETY: All indices into `a` are in 0..n and all indices into `w` are
    // in 0..n/4. This is guaranteed by the Ooura algorithm structure and
    // validated by the test suite.
    unsafe {
        let x0r = get(a, 0) + get(a, 2);
        let x0i = get(a, 1) + get(a, 3);
        let x1r = get(a, 0) - get(a, 2);
        let x1i = get(a, 1) - get(a, 3);
        let x2r = get(a, 4) + get(a, 6);
        let x2i = get(a, 5) + get(a, 7);
        let x3r = get(a, 4) - get(a, 6);
        let x3i = get(a, 5) - get(a, 7);
        set(a, 0, x0r + x2r);
        set(a, 1, x0i + x2i);
        set(a, 4, x0r - x2r);
        set(a, 5, x0i - x2i);
        set(a, 2, x1r - x3i);
        set(a, 3, x1i + x3r);
        set(a, 6, x1r + x3i);
        set(a, 7, x1i - x3r);

        let wk1r = get(w, 2);
        let x0r = get(a, 8) + get(a, 10);
        let x0i = get(a, 9) + get(a, 11);
        let x1r = get(a, 8) - get(a, 10);
        let x1i = get(a, 9) - get(a, 11);
        let x2r = get(a, 12) + get(a, 14);
        let x2i = get(a, 13) + get(a, 15);
        let x3r = get(a, 12) - get(a, 14);
        let x3i = get(a, 13) - get(a, 15);
        set(a, 8, x0r + x2r);
        set(a, 9, x0i + x2i);
        set(a, 12, x2i - x0i);
        set(a, 13, x0r - x2r);
        let x0r = x1r - x3i;
        let x0i = x1i + x3r;
        set(a, 10, wk1r * (x0r - x0i));
        set(a, 11, wk1r * (x0r + x0i));
        let x0r = x3i + x1r;
        let x0i = x3r - x1i;
        set(a, 14, wk1r * (x0i - x0r));
        set(a, 15, wk1r * (x0i + x0r));

        let mut k1 = 0_usize;
        let mut j = 16;
        while j < n {
            k1 += 2;
            let k2 = 2 * k1;
            let wk2r = get(w, k1);
            let wk2i = get(w, k1 + 1);
            let wk1r = get(w, k2);
            let wk1i = get(w, k2 + 1);
            let (wk3r, wk3i) = if FMA {
                (
                    (-2.0 * wk2i).mul_add(wk1i, wk1r),
                    (2.0 * wk2i).mul_add(wk1r, -wk1i),
                )
            } else {
                (wk1r - 2.0 * wk2i * wk1i, 2.0 * wk2i * wk1r - wk1i)
            };

            let x0r = get(a, j) + get(a, j + 2);
            let x0i = get(a, j + 1) + get(a, j + 3);
            let x1r = get(a, j) - get(a, j + 2);
            let x1i = get(a, j + 1) - get(a, j + 3);
            let x2r = get(a, j + 4) + get(a, j + 6);
            let x2i = get(a, j + 5) + get(a, j + 7);
            let x3r = get(a, j + 4) - get(a, j + 6);
            let x3i = get(a, j + 5) - get(a, j + 7);
            set(a, j, x0r + x2r);
            set(a, j + 1, x0i + x2i);
            let x0r = x0r - x2r;
            let x0i = x0i - x2i;
            if FMA {
                set(a, j + 4, wk2r.mul_add(x0r, -wk2i * x0i));
                set(a, j + 5, wk2r.mul_add(x0i, wk2i * x0r));
            } else {
                set(a, j + 4, wk2r * x0r - wk2i * x0i);
                set(a, j + 5, wk2r * x0i + wk2i * x0r);
            }
            let x0r = x1r - x3i;
            let x0i = x1i + x3r;
            if FMA {
                set(a, j + 2, wk1r.mul_add(x0r, -wk1i * x0i));
                set(a, j + 3, wk1r.mul_add(x0i, wk1i * x0r));
            } else {
                set(a, j + 2, wk1r * x0r - wk1i * x0i);
                set(a, j + 3, wk1r * x0i + wk1i * x0r);
            }
            let x0r = x1r + x3i;
            let x0i = x1i - x3r;
            if FMA {
                set(a, j + 6, wk3r.mul_add(x0r, -wk3i * x0i));
                set(a, j + 7, wk3r.mul_add(x0i, wk3i * x0r));
            } else {
                set(a, j + 6, wk3r * x0r - wk3i * x0i);
                set(a, j + 7, wk3r * x0i + wk3i * x0r);
            }

            let wk1r = get(w, k2 + 2);
            let wk1i = get(w, k2 + 3);
            let (wk3r, wk3i) = if FMA {
                (
                    (-2.0 * wk2r).mul_add(wk1i, wk1r),
                    (2.0 * wk2r).mul_add(wk1r, -wk1i),
                )
            } else {
                (wk1r - 2.0 * wk2r * wk1i, 2.0 * wk2r * wk1r - wk1i)
            };

            let x0r = get(a, j + 8) + get(a, j + 10);
            let x0i = get(a, j + 9) + get(a, j + 11);
            let x1r = get(a, j + 8) - get(a, j + 10);
            let x1i = get(a, j + 9) - get(a, j + 11);
            let x2r = get(a, j + 12) + get(a, j + 14);
            let x2i = get(a, j + 13) + get(a, j + 15);
            let x3r = get(a, j + 12) - get(a, j + 14);
            let x3i = get(a, j + 13) - get(a, j + 15);
            set(a, j + 8, x0r + x2r);
            set(a, j + 9, x0i + x2i);
            let x0r = x0r - x2r;
            let x0i = x0i - x2i;
            if FMA {
                set(a, j + 12, (-wk2i).mul_add(x0r, -wk2r * x0i));
                set(a, j + 13, (-wk2i).mul_add(x0i, wk2r * x0r));
            } else {
                set(a, j + 12, -wk2i * x0r - wk2r * x0i);
                set(a, j + 13, -wk2i * x0i + wk2r * x0r);
            }
            let x0r = x1r - x3i;
            let x0i = x1i + x3r;
            if FMA {
                set(a, j + 10, wk1r.mul_add(x0r, -wk1i * x0i));
                set(a, j + 11, wk1r.mul_add(x0i, wk1i * x0r));
            } else {
                set(a, j + 10, wk1r * x0r - wk1i * x0i);
                set(a, j + 11, wk1r * x0i + wk1i * x0r);
            }
            let x0r = x1r + x3i;
            let x0i = x1i - x3r;
            if FMA {
                set(a, j + 14, wk3r.mul_add(x0r, -wk3i * x0i));
                set(a, j + 15, wk3r.mul_add(x0i, wk3i * x0r));
            } else {
                set(a, j + 14, wk3r * x0r - wk3i * x0i);
                set(a, j + 15, wk3r * x0i + wk3i * x0r);
            }

            j += 16;
        }
    }
}

/// Modular complex FFT butterfly stage.
///
/// # Safety contract
/// All indices into `a` are in `0..n` and into `w` are in `0..n/4`.
fn cftmdl<const FMA: bool>(n: usize, l: usize, a: &mut [f32], w: &[f32]) {
    let m = l << 2;

    // SAFETY: All indices are bounded by `n` (see module-level doc).
    unsafe {
        for j in (0..l).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = get(a, j) + get(a, j1);
            let x0i = get(a, j + 1) + get(a, j1 + 1);
            let x1r = get(a, j) - get(a, j1);
            let x1i = get(a, j + 1) - get(a, j1 + 1);
            let x2r = get(a, j2) + get(a, j3);
            let x2i = get(a, j2 + 1) + get(a, j3 + 1);
            let x3r = get(a, j2) - get(a, j3);
            let x3i = get(a, j2 + 1) - get(a, j3 + 1);
            set(a, j, x0r + x2r);
            set(a, j + 1, x0i + x2i);
            set(a, j2, x0r - x2r);
            set(a, j2 + 1, x0i - x2i);
            set(a, j1, x1r - x3i);
            set(a, j1 + 1, x1i + x3r);
            set(a, j3, x1r + x3i);
            set(a, j3 + 1, x1i - x3r);
        }

        let wk1r = get(w, 2);
        for j in (m..l + m).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = get(a, j) + get(a, j1);
            let x0i = get(a, j + 1) + get(a, j1 + 1);
            let x1r = get(a, j) - get(a, j1);
            let x1i = get(a, j + 1) - get(a, j1 + 1);
            let x2r = get(a, j2) + get(a, j3);
            let x2i = get(a, j2 + 1) + get(a, j3 + 1);
            let x3r = get(a, j2) - get(a, j3);
            let x3i = get(a, j2 + 1) - get(a, j3 + 1);
            set(a, j, x0r + x2r);
            set(a, j + 1, x0i + x2i);
            set(a, j2, x2i - x0i);
            set(a, j2 + 1, x0r - x2r);
            let x0r = x1r - x3i;
            let x0i = x1i + x3r;
            set(a, j1, wk1r * (x0r - x0i));
            set(a, j1 + 1, wk1r * (x0r + x0i));
            let x0r = x3i + x1r;
            let x0i = x3r - x1i;
            set(a, j3, wk1r * (x0i - x0r));
            set(a, j3 + 1, wk1r * (x0i + x0r));
        }

        let mut k1 = 0_usize;
        let m2 = 2 * m;
        let mut k = m2;
        while k < n {
            k1 += 2;
            let k2 = 2 * k1;
            let wk2r = get(w, k1);
            let wk2i = get(w, k1 + 1);
            let wk1r = get(w, k2);
            let wk1i = get(w, k2 + 1);
            let (wk3r, wk3i) = if FMA {
                (
                    (-2.0 * wk2i).mul_add(wk1i, wk1r),
                    (2.0 * wk2i).mul_add(wk1r, -wk1i),
                )
            } else {
                (wk1r - 2.0 * wk2i * wk1i, 2.0 * wk2i * wk1r - wk1i)
            };

            for j in (k..l + k).step_by(2) {
                let j1 = j + l;
                let j2 = j1 + l;
                let j3 = j2 + l;
                let x0r = get(a, j) + get(a, j1);
                let x0i = get(a, j + 1) + get(a, j1 + 1);
                let x1r = get(a, j) - get(a, j1);
                let x1i = get(a, j + 1) - get(a, j1 + 1);
                let x2r = get(a, j2) + get(a, j3);
                let x2i = get(a, j2 + 1) + get(a, j3 + 1);
                let x3r = get(a, j2) - get(a, j3);
                let x3i = get(a, j2 + 1) - get(a, j3 + 1);
                set(a, j, x0r + x2r);
                set(a, j + 1, x0i + x2i);
                let x0r = x0r - x2r;
                let x0i = x0i - x2i;
                if FMA {
                    set(a, j2, wk2r.mul_add(x0r, -wk2i * x0i));
                    set(a, j2 + 1, wk2r.mul_add(x0i, wk2i * x0r));
                } else {
                    set(a, j2, wk2r * x0r - wk2i * x0i);
                    set(a, j2 + 1, wk2r * x0i + wk2i * x0r);
                }
                let x0r = x1r - x3i;
                let x0i = x1i + x3r;
                if FMA {
                    set(a, j1, wk1r.mul_add(x0r, -wk1i * x0i));
                    set(a, j1 + 1, wk1r.mul_add(x0i, wk1i * x0r));
                } else {
                    set(a, j1, wk1r * x0r - wk1i * x0i);
                    set(a, j1 + 1, wk1r * x0i + wk1i * x0r);
                }
                let x0r = x1r + x3i;
                let x0i = x1i - x3r;
                if FMA {
                    set(a, j3, wk3r.mul_add(x0r, -wk3i * x0i));
                    set(a, j3 + 1, wk3r.mul_add(x0i, wk3i * x0r));
                } else {
                    set(a, j3, wk3r * x0r - wk3i * x0i);
                    set(a, j3 + 1, wk3r * x0i + wk3i * x0r);
                }
            }

            let wk1r = get(w, k2 + 2);
            let wk1i = get(w, k2 + 3);
            let (wk3r, wk3i) = if FMA {
                (
                    (-2.0 * wk2r).mul_add(wk1i, wk1r),
                    (2.0 * wk2r).mul_add(wk1r, -wk1i),
                )
            } else {
                (wk1r - 2.0 * wk2r * wk1i, 2.0 * wk2r * wk1r - wk1i)
            };

            for j in (k + m..l + (k + m)).step_by(2) {
                let j1 = j + l;
                let j2 = j1 + l;
                let j3 = j2 + l;
                let x0r = get(a, j) + get(a, j1);
                let x0i = get(a, j + 1) + get(a, j1 + 1);
                let x1r = get(a, j) - get(a, j1);
                let x1i = get(a, j + 1) - get(a, j1 + 1);
                let x2r = get(a, j2) + get(a, j3);
                let x2i = get(a, j2 + 1) + get(a, j3 + 1);
                let x3r = get(a, j2) - get(a, j3);
                let x3i = get(a, j2 + 1) - get(a, j3 + 1);
                set(a, j, x0r + x2r);
                set(a, j + 1, x0i + x2i);
                let x0r = x0r - x2r;
                let x0i = x0i - x2i;
                if FMA {
                    set(a, j2, (-wk2i).mul_add(x0r, -wk2r * x0i));
                    set(a, j2 + 1, (-wk2i).mul_add(x0i, wk2r * x0r));
                } else {
                    set(a, j2, -wk2i * x0r - wk2r * x0i);
                    set(a, j2 + 1, -wk2i * x0i + wk2r * x0r);
                }
                let x0r = x1r - x3i;
                let x0i = x1i + x3r;
                if FMA {
                    set(a, j1, wk1r.mul_add(x0r, -wk1i * x0i));
                    set(a, j1 + 1, wk1r.mul_add(x0i, wk1i * x0r));
                } else {
                    set(a, j1, wk1r * x0r - wk1i * x0i);
                    set(a, j1 + 1, wk1r * x0i + wk1i * x0r);
                }
                let x0r = x1r + x3i;
                let x0i = x1i - x3r;
                if FMA {
                    set(a, j3, wk3r.mul_add(x0r, -wk3i * x0i));
                    set(a, j3 + 1, wk3r.mul_add(x0i, wk3i * x0r));
                } else {
                    set(a, j3, wk3r * x0r - wk3i * x0i);
                    set(a, j3 + 1, wk3r * x0i + wk3i * x0r);
                }
            }

            k += m2;
        }
    }
}

/// Real FFT forward post-processing (split-radix real/imaginary separation).
fn rftfsub<const FMA: bool>(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    let mut j = 2;
    // SAFETY: j in 2..m by 2, k = n-j, all indices in 0..n. c indices in 0..nc.
    unsafe {
        while j < m {
            let k = n - j;
            kk += ks;
            let wkr = 0.5 - get(c, nc - kk);
            let wki = get(c, kk);
            let xr = get(a, j) - get(a, k);
            let xi = get(a, j + 1) + get(a, k + 1);
            let (yr, yi) = if FMA {
                (wkr.mul_add(xr, -wki * xi), wkr.mul_add(xi, wki * xr))
            } else {
                (wkr * xr - wki * xi, wkr * xi + wki * xr)
            };
            set(a, j, get(a, j) - yr);
            set(a, j + 1, get(a, j + 1) - yi);
            set(a, k, get(a, k) + yr);
            set(a, k + 1, get(a, k + 1) - yi);
            j += 2;
        }
    }
}

/// Real FFT backward pre-processing (split-radix real/imaginary recombination).
fn rftbsub<const FMA: bool>(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    // SAFETY: All indices in 0..n. c indices in 0..nc.
    unsafe {
        set(a, 1, -get(a, 1));
        let mut j = 2;
        while j < m {
            let k = n - j;
            kk += ks;
            let wkr = 0.5 - get(c, nc - kk);
            let wki = get(c, kk);
            let xr = get(a, j) - get(a, k);
            let xi = get(a, j + 1) + get(a, k + 1);
            let (yr, yi) = if FMA {
                (wkr.mul_add(xr, wki * xi), wkr.mul_add(xi, -wki * xr))
            } else {
                (wkr * xr + wki * xi, wkr * xi - wki * xr)
            };
            set(a, j, get(a, j) - yr);
            set(a, j + 1, yi - get(a, j + 1));
            set(a, k, get(a, k) + yr);
            set(a, k + 1, yi - get(a, k + 1));
            j += 2;
        }
        set(a, m + 1, -get(a, m + 1));
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;

    #[test]
    fn roundtrip_256() {
        let fft = Fft4g::new(256);
        let mut a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let original = a.clone();

        fft.rdft(&mut a);
        fft.irdft(&mut a);

        let scale = 2.0 / 256.0;
        for (i, (&o, &r)) in original.iter().zip(a.iter()).enumerate() {
            let recovered = r * scale;
            assert!(
                (o - recovered).abs() < 1e-4,
                "mismatch at {i}: original={o}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn roundtrip_multiple_sizes() {
        for &n in &[4, 8, 16, 32, 64, 128, 256, 512] {
            let fft = Fft4g::new(n);
            let mut a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
            let original = a.clone();

            fft.rdft(&mut a);
            fft.irdft(&mut a);

            let scale = 2.0 / n as f32;
            for (i, (&o, &r)) in original.iter().zip(a.iter()).enumerate() {
                let recovered = r * scale;
                assert!(
                    (o - recovered).abs() < 1e-3,
                    "size {n}, index {i}: original={o}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn impulse_256() {
        let fft = Fft4g::new(256);
        let mut a = vec![0.0_f32; 256];
        a[0] = 1.0;

        fft.rdft(&mut a);

        // DC = 1.0.
        assert!((a[0] - 1.0).abs() < 1e-6, "DC = {}", a[0]);
        // Nyquist = 1.0.
        assert!((a[1] - 1.0).abs() < 1e-6, "Nyquist = {}", a[1]);
        // All other bins: real=1.0, imag=0.0.
        for k in 1..128 {
            assert!((a[2 * k] - 1.0).abs() < 1e-5, "bin {k} real = {}", a[2 * k]);
            assert!(a[2 * k + 1].abs() < 1e-5, "bin {k} imag = {}", a[2 * k + 1]);
        }
    }

    #[test]
    fn parseval_energy() {
        let n = 256;
        let fft = Fft4g::new(n);
        let mut a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let time_energy: f32 = a.iter().map(|x| x * x).sum();

        fft.rdft(&mut a);

        // Parseval: sum |X[k]|^2 = N * sum |x[n]|^2
        // With Ooura's convention (no 1/N on forward):
        // DC^2 + Nyquist^2 + 2 * sum_{k=1}^{N/2-1} (Re^2 + Im^2) = N * time_energy
        let dc_sq = a[0] * a[0];
        let nyq_sq = a[1] * a[1];
        let mut freq_energy = dc_sq + nyq_sq;
        for k in 1..n / 2 {
            freq_energy += 2.0 * (a[2 * k] * a[2 * k] + a[2 * k + 1] * a[2 * k + 1]);
        }

        let expected = n as f32 * time_energy;
        assert!(
            (freq_energy - expected).abs() / expected < 1e-4,
            "Parseval: freq_energy={freq_energy}, expected={expected}"
        );
    }

    #[test]
    fn zero_input() {
        let fft = Fft4g::new(64);
        let mut a = vec![0.0_f32; 64];
        fft.rdft(&mut a);
        for (i, &v) in a.iter().enumerate() {
            assert_eq!(v, 0.0, "expected zero at {i}, got {v}");
        }
    }

    /// One twiddled radix-4 butterfly of `cft1st` (`l = 2`) and `cftmdl`, on
    /// the points `j`, `j + l`, `j + 2l` and `j + 3l`, with the twiddles the
    /// C code loads at `k1` for the first or the `second` group.
    fn reference_butterfly(
        a: &mut [f32],
        j: usize,
        l: usize,
        w: &[f32],
        k1: usize,
        second: bool,
        fused: bool,
    ) {
        let cmul = |(wr, wi): (f32, f32), (xr, xi): (f32, f32)| {
            if fused {
                (wr.mul_add(xr, -wi * xi), wr.mul_add(xi, wi * xr))
            } else {
                (wr * xr - wi * xi, wr * xi + wi * xr)
            }
        };
        let (k2, wk2r, wk2i) = (2 * k1, w[k1], w[k1 + 1]);
        let (w2, s, (wk1r, wk1i)) = if second {
            ((-wk2i, wk2r), wk2r, (w[k2 + 2], w[k2 + 3]))
        } else {
            ((wk2r, wk2i), wk2i, (w[k2], w[k2 + 1]))
        };
        let w3 = if fused {
            (
                (-2.0 * s).mul_add(wk1i, wk1r),
                (2.0 * s).mul_add(wk1r, -wk1i),
            )
        } else {
            (wk1r - 2.0 * s * wk1i, 2.0 * s * wk1r - wk1i)
        };
        let (j1, j2, j3) = (j + l, j + 2 * l, j + 3 * l);
        let x0 = (a[j] + a[j1], a[j + 1] + a[j1 + 1]);
        let x1 = (a[j] - a[j1], a[j + 1] - a[j1 + 1]);
        let x2 = (a[j2] + a[j3], a[j2 + 1] + a[j3 + 1]);
        let x3 = (a[j2] - a[j3], a[j2 + 1] - a[j3 + 1]);
        let out = [
            (j, (x0.0 + x2.0, x0.1 + x2.1)),
            (j2, cmul(w2, (x0.0 - x2.0, x0.1 - x2.1))),
            (j1, cmul((wk1r, wk1i), (x1.0 - x3.1, x1.1 + x3.0))),
            (j3, cmul(w3, (x1.0 + x3.1, x1.1 - x3.0))),
        ];
        for (i, (re, im)) in out {
            a[i] = re;
            a[i + 1] = im;
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Asserts that `fused` and `plain`, the outputs of a pass of butterflies
    /// with span `l` (`cft1st` is `l = 2`), differ in every twiddled output:
    /// at `j + l`, `j + 2l` and `j + 3l`, real and imaginary, in both groups,
    /// for at least one butterfly. A statement that computes one of these in
    /// the wrong form then fails the bit-exact comparison on its own.
    fn assert_each_twiddled_output_differs(fused: &[f32], plain: &[f32], l: usize) {
        let m = 4 * l;
        // differs[group][quarter][part]: quarter q holds a[j + q * l], and
        // quarter 0, a[j] = x0 + x2, has no twiddle.
        let mut differs = [[[false; 2]; 4]; 2];
        // The first block of 2m values has no `mul_add`.
        for p in 2 * m..fused.len() {
            if fused[p].to_bits() != plain[p].to_bits() {
                let (group, q) = (p % (2 * m) / m, p % m);
                differs[group][q / l][q % 2] = true;
            }
        }
        for (group, quarters) in differs.iter().enumerate() {
            for (quarter, parts) in quarters.iter().enumerate().skip(1) {
                for (part, &differs) in ["real", "imaginary"].iter().zip(parts) {
                    assert!(
                        differs,
                        "l {l}, group {group}: {part} a[j + {quarter}l] same in both forms"
                    );
                }
            }
        }
    }

    /// `n` pseudo-random values in `-0.5..0.5`, the same on every target.
    fn noise(n: usize) -> Vec<f32> {
        let mut x = 1_u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 8) as f32 / (1 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Both forms of the twiddled butterflies of `cft1st` and `cftmdl`, on
    /// every target: the fused form must match the `mul_add` chains, and the
    /// plain form the C expressions, bit for bit. The input is pseudo-random:
    /// in the golden-ratio sequence of the other tests, values `l` apart
    /// differ by almost the same amount, and some outputs then matched
    /// between the forms in every butterfly. With this input and these
    /// sizes, each twiddled output differs in dozens of butterflies.
    #[test]
    fn butterflies_match_fused_and_plain_references() {
        // cft1st, n = 4096: iteration t runs at j = 16t with k1 = 2t, as two
        // groups at l = 2. The untwiddled block a[0..16] is not checked.
        let fft = Fft4g::new(4096);
        let w = &fft.w;
        let input = noise(4096);
        let [fused, plain] = [true, false].map(|fused| {
            let mut e = input.clone();
            for t in 1..256 {
                reference_butterfly(&mut e, 16 * t, 2, w, 2 * t, false, fused);
                reference_butterfly(&mut e, 16 * t + 8, 2, w, 2 * t, true, fused);
            }
            e
        });
        assert_each_twiddled_output_differs(&fused, &plain, 2);
        let mut a = input.clone();
        cft1st::<true>(4096, &mut a, w);
        assert_eq!(bits(&a[16..]), bits(&fused[16..]));
        let mut a = input;
        cft1st::<false>(4096, &mut a, w);
        assert_eq!(bits(&a[16..]), bits(&plain[16..]));

        // cftmdl, n = 8192, l = 8: iteration t runs at k = 64t with k1 = 2t.
        // The untwiddled block a[0..64] is not checked.
        let fft = Fft4g::new(8192);
        let w = &fft.w;
        let input = noise(8192);
        let [fused, plain] = [true, false].map(|fused| {
            let mut e = input.clone();
            for t in 1..128 {
                for j in (64 * t..64 * t + 8).step_by(2) {
                    reference_butterfly(&mut e, j, 8, w, 2 * t, false, fused);
                    reference_butterfly(&mut e, j + 32, 8, w, 2 * t, true, fused);
                }
            }
            e
        });
        assert_each_twiddled_output_differs(&fused, &plain, 8);
        let mut a = input.clone();
        cftmdl::<true>(8192, 8, &mut a, w);
        assert_eq!(bits(&a[64..]), bits(&fused[64..]));
        let mut a = input;
        cftmdl::<false>(8192, 8, &mut a, w);
        assert_eq!(bits(&a[64..]), bits(&plain[64..]));
    }

    /// Both forms of `rftfsub` and `rftbsub`, on every target: the fused
    /// form must match the `mul_add` result, and the plain form the C
    /// expressions, bit for bit.
    #[test]
    fn rft_sub_matches_fused_and_plain_references() {
        // With n = 8 each routine runs one iteration: j = 2, k = 6,
        // wkr = 0.5 - c[1], wki = c[1]. a[2] = a[3] = 0 makes a[2] and a[3]
        // carry yr and yi exactly.
        let c = [0.0_f32, 0.3];
        let input = [0.4_f32, -0.8, 0.0, 0.0, 0.6, 0.9, 0.09, 0.65];
        let (wkr, wki) = (0.5 - c[1], c[1]);
        let (xr, xi) = (input[2] - input[6], input[3] + input[7]);

        // rftfsub: C `yr = wkr * xr - wki * xi; yi = wkr * xi + wki * xr;`
        let fused = (wkr.mul_add(xr, -wki * xi), wkr.mul_add(xi, wki * xr));
        let plain = (wkr * xr - wki * xi, wkr * xi + wki * xr);
        assert_ne!(fused.0.to_bits(), plain.0.to_bits());
        assert_ne!(fused.1.to_bits(), plain.1.to_bits());
        let expected = |(yr, yi): (f32, f32)| {
            let mut e = input;
            e[2] -= yr;
            e[3] -= yi;
            e[6] += yr;
            e[7] -= yi;
            e.map(f32::to_bits)
        };
        let mut a = input;
        rftfsub::<true>(8, &mut a, 2, &c);
        assert_eq!(a.map(f32::to_bits), expected(fused));
        let mut a = input;
        rftfsub::<false>(8, &mut a, 2, &c);
        assert_eq!(a.map(f32::to_bits), expected(plain));

        // rftbsub: C `yr = wkr * xr + wki * xi; yi = wkr * xi - wki * xr;`
        let fused = (wkr.mul_add(xr, wki * xi), wkr.mul_add(xi, -wki * xr));
        let plain = (wkr * xr + wki * xi, wkr * xi - wki * xr);
        assert_ne!(fused.0.to_bits(), plain.0.to_bits());
        assert_ne!(fused.1.to_bits(), plain.1.to_bits());
        let expected = |(yr, yi): (f32, f32)| {
            let mut e = input;
            e[1] = -e[1];
            e[2] -= yr;
            e[3] = yi - e[3];
            e[6] += yr;
            e[7] = yi - e[7];
            e[5] = -e[5];
            e.map(f32::to_bits)
        };
        let mut a = input;
        rftbsub::<true>(8, &mut a, 2, &c);
        assert_eq!(a.map(f32::to_bits), expected(fused));
        let mut a = input;
        rftbsub::<false>(8, &mut a, 2, &c);
        assert_eq!(a.map(f32::to_bits), expected(plain));
    }

    /// Output bits of `rdft_with::<FMA>` and `irdft_with::<FMA>` on `x`.
    fn run_forms<const FMA: bool>(fft: &Fft4g, x: &[f32]) -> [Vec<u32>; 2] {
        let mut forward = x.to_vec();
        fft.rdft_with::<FMA>(&mut forward);
        let mut inverse = x.to_vec();
        fft.irdft_with::<FMA>(&mut inverse);
        [bits(&forward), bits(&inverse)]
    }

    /// `rdft` and `irdft` run the [`NATIVE_FMA`] form.
    #[test]
    fn rdft_and_irdft_use_native_fma() {
        // n = 512 runs cft1st, cftmdl, and rftfsub or rftbsub.
        let fft = Fft4g::new(512);
        let input: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.618_034).fract() - 0.5)
            .collect();
        // The forms differ on this input, so the comparison below can fail.
        let [fused, plain] = [
            run_forms::<true>(&fft, &input),
            run_forms::<false>(&fft, &input),
        ];
        assert!(fused[0] != plain[0] && fused[1] != plain[1]);

        let mut forward = input.clone();
        fft.rdft(&mut forward);
        let mut inverse = input.clone();
        fft.irdft(&mut inverse);
        assert_eq!(
            [bits(&forward), bits(&inverse)],
            run_forms::<NATIVE_FMA>(&fft, &input)
        );
    }

    /// The last pass of `cftfsub`, or of `cftbsub` if `backward`, for
    /// `n = 4 * l`, copied from them. It has no `mul_add`, so both forms
    /// share it.
    fn last_radix4_pass(a: &mut [f32], l: usize, backward: bool) {
        for j in (0..l).step_by(2) {
            let (j1, j2, j3) = (j + l, j + 2 * l, j + 3 * l);
            let x0r = a[j] + a[j1];
            let x1r = a[j] - a[j1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            if backward {
                let x0i = -a[j + 1] - a[j1 + 1];
                let x1i = -a[j + 1] + a[j1 + 1];
                a[j] = x0r + x2r;
                a[j + 1] = x0i - x2i;
                a[j2] = x0r - x2r;
                a[j2 + 1] = x0i + x2i;
                a[j1] = x1r - x3i;
                a[j1 + 1] = x1i - x3r;
                a[j3] = x1r + x3i;
                a[j3 + 1] = x1i + x3r;
            } else {
                let x0i = a[j + 1] + a[j1 + 1];
                let x1i = a[j + 1] - a[j1 + 1];
                a[j] = x0r + x2r;
                a[j + 1] = x0i + x2i;
                a[j2] = x0r - x2r;
                a[j2 + 1] = x0i - x2i;
                a[j1] = x1r - x3i;
                a[j1 + 1] = x1i + x3r;
                a[j3] = x1r + x3i;
                a[j3 + 1] = x1i - x3r;
            }
        }
    }

    /// Output bits of `rdft_with` on `x` at n = 512, or of `irdft_with` if
    /// `inverse`, built from the kernels without the call chain in between:
    /// `cft1st`, both `cftmdl` passes, and `rftfsub` or `rftbsub`, each in
    /// the form `forms` gives for it.
    fn composed(fft: &Fft4g, x: &[f32], inverse: bool, forms: [bool; 3]) -> Vec<u32> {
        assert_eq!(fft.n, 512);
        let (n, nc, w, c) = (fft.n, fft.nc, &fft.w[..], &fft.w[fft.nw..]);
        let rft_sub = |a: &mut [f32]| match (inverse, forms[2]) {
            (false, true) => rftfsub::<true>(n, a, nc, c),
            (false, false) => rftfsub::<false>(n, a, nc, c),
            (true, true) => rftbsub::<true>(n, a, nc, c),
            (true, false) => rftbsub::<false>(n, a, nc, c),
        };
        let mut a = x.to_vec();
        if inverse {
            a[1] = 0.5 * (a[0] - a[1]);
            a[0] -= a[1];
            rft_sub(&mut a);
        }
        apply_bitrv2(&fft.bitrv_ip, fft.bitrv_m, fft.bitrv_long, &mut a);
        if forms[0] {
            cft1st::<true>(n, &mut a, w);
        } else {
            cft1st::<false>(n, &mut a, w);
        }
        for l in [8, 32] {
            if forms[1] {
                cftmdl::<true>(n, l, &mut a, w);
            } else {
                cftmdl::<false>(n, l, &mut a, w);
            }
        }
        last_radix4_pass(&mut a, 128, inverse);
        if !inverse {
            rft_sub(&mut a);
            let xi = a[0] - a[1];
            a[0] += a[1];
            a[1] = xi;
        }
        bits(&a)
    }

    /// `rdft_with` and `irdft_with` pass their form to every kernel: on every
    /// target, each form matches the kernels composed in that form, bit for
    /// bit.
    #[test]
    fn transforms_pass_the_form_to_every_kernel() {
        let fft = Fft4g::new(512);
        let input: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.618_034).fract() - 0.5)
            .collect();
        for fma in [true, false] {
            let expected = [false, true].map(|inverse| composed(&fft, &input, inverse, [fma; 3]));
            // Each `::<FMA>` in the call chain sets the form of one of these:
            // `cft1st`; both `cftmdl` passes; all three (the call of `cftfsub`
            // or `cftbsub`); `rftfsub` or `rftbsub`. Each of them, alone in
            // the other form, changes both transforms, so a wrong `::<FMA>`
            // fails the comparison below.
            for flipped in [
                [true, false, false],
                [false, true, false],
                [true, true, false],
                [false, false, true],
            ] {
                let forms = flipped.map(|flip| flip != fma);
                for (inverse, expected) in [false, true].into_iter().zip(&expected) {
                    assert_ne!(
                        &composed(&fft, &input, inverse, forms),
                        expected,
                        "fma {fma}, inverse {inverse}, forms {forms:?}"
                    );
                }
            }
            let actual = if fma {
                run_forms::<true>(&fft, &input)
            } else {
                run_forms::<false>(&fft, &input)
            };
            assert_eq!(actual, expected, "fma {fma}");
        }
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn rejects_non_power_of_two() {
        let _ = Fft4g::new(100);
    }

    #[test]
    #[should_panic(expected = ">= 2")]
    fn rejects_size_one() {
        let _ = Fft4g::new(1);
    }

    // -- Property tests --

    /// Map a small integer exponent to a power-of-2 FFT size (4..=512).
    fn fft_size_from_exp(exp: u32) -> usize {
        1 << exp
    }

    #[proptest]
    fn roundtrip_recovers_signal(
        #[strategy(2..=9u32)] exp: u32,
        #[strategy(prop::collection::vec(-1.0f32..1.0, 1 << #exp as usize))] signal: Vec<f32>,
    ) {
        let n = fft_size_from_exp(exp);
        let fft = Fft4g::new(n);
        let mut a = signal.clone();

        fft.rdft(&mut a);
        fft.irdft(&mut a);

        let scale = 2.0 / n as f32;
        for (i, (&o, &r)) in signal.iter().zip(a.iter()).enumerate() {
            prop_assert!(
                (o - r * scale).abs() < 1e-3,
                "size {n}, index {i}: original={o}, recovered={}",
                r * scale
            );
        }
    }

    #[proptest]
    fn linearity_holds(
        #[strategy(2..=9u32)] exp: u32,
        #[strategy(prop::collection::vec(-1.0f32..1.0, 1 << #exp as usize))] sig_a: Vec<f32>,
        #[strategy(prop::collection::vec(-1.0f32..1.0, 1 << #exp as usize))] sig_b: Vec<f32>,
    ) {
        let n = fft_size_from_exp(exp);
        let fft = Fft4g::new(n);

        let mut a = sig_a.clone();
        let mut b = sig_b.clone();
        let mut sum: Vec<f32> = sig_a.iter().zip(sig_b.iter()).map(|(x, y)| x + y).collect();

        fft.rdft(&mut a);
        fft.rdft(&mut b);
        fft.rdft(&mut sum);

        for (i, ((&fa, &fb), &fs)) in a.iter().zip(b.iter()).zip(sum.iter()).enumerate() {
            let expected = fa + fb;
            prop_assert!(
                (fs - expected).abs() < 1e-3,
                "size {n}, index {i}: FFT(a+b)={fs}, FFT(a)+FFT(b)={expected}"
            );
        }
    }

    #[proptest]
    fn parseval_energy_conservation(
        #[strategy(2..=9u32)] exp: u32,
        #[strategy(prop::collection::vec(-1.0f32..1.0, 1 << #exp as usize))] signal: Vec<f32>,
    ) {
        let n = fft_size_from_exp(exp);
        let fft = Fft4g::new(n);
        let time_energy: f32 = signal.iter().map(|x| x * x).sum();

        let mut a = signal;
        fft.rdft(&mut a);

        let dc_sq = a[0] * a[0];
        let nyq_sq = a[1] * a[1];
        let mut freq_energy = dc_sq + nyq_sq;
        for k in 1..n / 2 {
            freq_energy += 2.0 * (a[2 * k] * a[2 * k] + a[2 * k + 1] * a[2 * k + 1]);
        }

        let expected = n as f32 * time_energy;
        if expected > 1e-6 {
            prop_assert!(
                (freq_energy - expected).abs() / expected < 1e-3,
                "Parseval: freq_energy={freq_energy}, expected={expected}"
            );
        }
    }
}
