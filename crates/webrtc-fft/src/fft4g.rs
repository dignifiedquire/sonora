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

use std::f32::consts::FRAC_PI_4;

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
        assert_eq!(a.len(), self.n, "input length must be {}", self.n);
        let n = self.n;

        if n > 4 {
            apply_bitrv2(&self.bitrv_ip, self.bitrv_m, self.bitrv_long, a);
            cftfsub(n, a, &self.w);
            rftfsub(n, a, self.nc, &self.w[self.nw..]);
        } else if n == 4 {
            cftfsub(n, a, &self.w);
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
        assert_eq!(a.len(), self.n, "input length must be {}", self.n);
        let n = self.n;

        a[1] = 0.5 * (a[0] - a[1]);
        a[0] -= a[1];
        if n > 4 {
            rftbsub(n, a, self.nc, &self.w[self.nw..]);
            apply_bitrv2(&self.bitrv_ip, self.bitrv_m, self.bitrv_long, a);
            cftbsub(n, a, &self.w);
        } else if n == 4 {
            cftfsub(n, a, &self.w);
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

/// Apply precomputed bit-reversal permutation to `a`.
fn apply_bitrv2(ip: &[usize], m: usize, use_long_swap: bool, a: &mut [f32]) {
    let m2 = 2 * m;
    if use_long_swap {
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
fn cftfsub(n: usize, a: &mut [f32], w: &[f32]) {
    let mut l = 2;
    if n > 8 {
        cft1st(n, a, w);
        l = 8;
        while (l << 2) < n {
            cftmdl(n, l, a, w);
            l <<= 2;
        }
    }
    if (l << 2) == n {
        for j in (0..l).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = a[j] + a[j1];
            let x0i = a[j + 1] + a[j1 + 1];
            let x1r = a[j] - a[j1];
            let x1i = a[j + 1] - a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j] = x0r + x2r;
            a[j + 1] = x0i + x2i;
            a[j2] = x0r - x2r;
            a[j2 + 1] = x0i - x2i;
            a[j1] = x1r - x3i;
            a[j1 + 1] = x1i + x3r;
            a[j3] = x1r + x3i;
            a[j3 + 1] = x1i - x3r;
        }
    } else {
        for j in (0..l).step_by(2) {
            let j1 = j + l;
            let x0r = a[j] - a[j1];
            let x0i = a[j + 1] - a[j1 + 1];
            a[j] += a[j1];
            a[j + 1] += a[j1 + 1];
            a[j1] = x0r;
            a[j1 + 1] = x0i;
        }
    }
}

/// Backward complex sub-transform (radix-4 decomposition).
fn cftbsub(n: usize, a: &mut [f32], w: &[f32]) {
    let mut l = 2;
    if n > 8 {
        cft1st(n, a, w);
        l = 8;
        while (l << 2) < n {
            cftmdl(n, l, a, w);
            l <<= 2;
        }
    }
    if (l << 2) == n {
        for j in (0..l).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = a[j] + a[j1];
            let x0i = -a[j + 1] - a[j1 + 1];
            let x1r = a[j] - a[j1];
            let x1i = -a[j + 1] + a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j] = x0r + x2r;
            a[j + 1] = x0i - x2i;
            a[j2] = x0r - x2r;
            a[j2 + 1] = x0i + x2i;
            a[j1] = x1r - x3i;
            a[j1 + 1] = x1i - x3r;
            a[j3] = x1r + x3i;
            a[j3 + 1] = x1i + x3r;
        }
    } else {
        for j in (0..l).step_by(2) {
            let j1 = j + l;
            let x0r = a[j] - a[j1];
            let x0i = -a[j + 1] + a[j1 + 1];
            a[j] += a[j1];
            a[j + 1] = -a[j + 1] - a[j1 + 1];
            a[j1] = x0r;
            a[j1 + 1] = x0i;
        }
    }
}

/// First-stage complex FFT butterfly.
fn cft1st(n: usize, a: &mut [f32], w: &[f32]) {
    let x0r = a[0] + a[2];
    let x0i = a[1] + a[3];
    let x1r = a[0] - a[2];
    let x1i = a[1] - a[3];
    let x2r = a[4] + a[6];
    let x2i = a[5] + a[7];
    let x3r = a[4] - a[6];
    let x3i = a[5] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[2] = x1r - x3i;
    a[3] = x1i + x3r;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;

    let wk1r = w[2];
    let x0r = a[8] + a[10];
    let x0i = a[9] + a[11];
    let x1r = a[8] - a[10];
    let x1i = a[9] - a[11];
    let x2r = a[12] + a[14];
    let x2i = a[13] + a[15];
    let x3r = a[12] - a[14];
    let x3i = a[13] - a[15];
    a[8] = x0r + x2r;
    a[9] = x0i + x2i;
    a[12] = x2i - x0i;
    a[13] = x0r - x2r;
    let x0r = x1r - x3i;
    let x0i = x1i + x3r;
    a[10] = wk1r * (x0r - x0i);
    a[11] = wk1r * (x0r + x0i);
    let x0r = x3i + x1r;
    let x0i = x3r - x1i;
    a[14] = wk1r * (x0i - x0r);
    a[15] = wk1r * (x0i + x0r);

    let mut k1 = 0_usize;
    let mut j = 16;
    while j < n {
        k1 += 2;
        let k2 = 2 * k1;
        let wk2r = w[k1];
        let wk2i = w[k1 + 1];
        let wk1r = w[k2];
        let wk1i = w[k2 + 1];
        let wk3r = wk1r - 2.0 * wk2i * wk1i;
        let wk3i = 2.0 * wk2i * wk1r - wk1i;

        let x0r = a[j] + a[j + 2];
        let x0i = a[j + 1] + a[j + 3];
        let x1r = a[j] - a[j + 2];
        let x1i = a[j + 1] - a[j + 3];
        let x2r = a[j + 4] + a[j + 6];
        let x2i = a[j + 5] + a[j + 7];
        let x3r = a[j + 4] - a[j + 6];
        let x3i = a[j + 5] - a[j + 7];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        let x0r = x0r - x2r;
        let x0i = x0i - x2i;
        a[j + 4] = wk2r * x0r - wk2i * x0i;
        a[j + 5] = wk2r * x0i + wk2i * x0r;
        let x0r = x1r - x3i;
        let x0i = x1i + x3r;
        a[j + 2] = wk1r * x0r - wk1i * x0i;
        a[j + 3] = wk1r * x0i + wk1i * x0r;
        let x0r = x1r + x3i;
        let x0i = x1i - x3r;
        a[j + 6] = wk3r * x0r - wk3i * x0i;
        a[j + 7] = wk3r * x0i + wk3i * x0r;

        let wk1r = w[k2 + 2];
        let wk1i = w[k2 + 3];
        let wk3r = wk1r - 2.0 * wk2r * wk1i;
        let wk3i = 2.0 * wk2r * wk1r - wk1i;

        let x0r = a[j + 8] + a[j + 10];
        let x0i = a[j + 9] + a[j + 11];
        let x1r = a[j + 8] - a[j + 10];
        let x1i = a[j + 9] - a[j + 11];
        let x2r = a[j + 12] + a[j + 14];
        let x2i = a[j + 13] + a[j + 15];
        let x3r = a[j + 12] - a[j + 14];
        let x3i = a[j + 13] - a[j + 15];
        a[j + 8] = x0r + x2r;
        a[j + 9] = x0i + x2i;
        let x0r = x0r - x2r;
        let x0i = x0i - x2i;
        a[j + 12] = -wk2i * x0r - wk2r * x0i;
        a[j + 13] = -wk2i * x0i + wk2r * x0r;
        let x0r = x1r - x3i;
        let x0i = x1i + x3r;
        a[j + 10] = wk1r * x0r - wk1i * x0i;
        a[j + 11] = wk1r * x0i + wk1i * x0r;
        let x0r = x1r + x3i;
        let x0i = x1i - x3r;
        a[j + 14] = wk3r * x0r - wk3i * x0i;
        a[j + 15] = wk3r * x0i + wk3i * x0r;

        j += 16;
    }
}

/// Modular complex FFT butterfly stage.
fn cftmdl(n: usize, l: usize, a: &mut [f32], w: &[f32]) {
    let m = l << 2;

    for j in (0..l).step_by(2) {
        let j1 = j + l;
        let j2 = j1 + l;
        let j3 = j2 + l;
        let x0r = a[j] + a[j1];
        let x0i = a[j + 1] + a[j1 + 1];
        let x1r = a[j] - a[j1];
        let x1i = a[j + 1] - a[j1 + 1];
        let x2r = a[j2] + a[j3];
        let x2i = a[j2 + 1] + a[j3 + 1];
        let x3r = a[j2] - a[j3];
        let x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j2] = x0r - x2r;
        a[j2 + 1] = x0i - x2i;
        a[j1] = x1r - x3i;
        a[j1 + 1] = x1i + x3r;
        a[j3] = x1r + x3i;
        a[j3 + 1] = x1i - x3r;
    }

    let wk1r = w[2];
    for j in (m..l + m).step_by(2) {
        let j1 = j + l;
        let j2 = j1 + l;
        let j3 = j2 + l;
        let x0r = a[j] + a[j1];
        let x0i = a[j + 1] + a[j1 + 1];
        let x1r = a[j] - a[j1];
        let x1i = a[j + 1] - a[j1 + 1];
        let x2r = a[j2] + a[j3];
        let x2i = a[j2 + 1] + a[j3 + 1];
        let x3r = a[j2] - a[j3];
        let x3i = a[j2 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j2] = x2i - x0i;
        a[j2 + 1] = x0r - x2r;
        let x0r = x1r - x3i;
        let x0i = x1i + x3r;
        a[j1] = wk1r * (x0r - x0i);
        a[j1 + 1] = wk1r * (x0r + x0i);
        let x0r = x3i + x1r;
        let x0i = x3r - x1i;
        a[j3] = wk1r * (x0i - x0r);
        a[j3 + 1] = wk1r * (x0i + x0r);
    }

    let mut k1 = 0_usize;
    let m2 = 2 * m;
    let mut k = m2;
    while k < n {
        k1 += 2;
        let k2 = 2 * k1;
        let wk2r = w[k1];
        let wk2i = w[k1 + 1];
        let wk1r = w[k2];
        let wk1i = w[k2 + 1];
        let wk3r = wk1r - 2.0 * wk2i * wk1i;
        let wk3i = 2.0 * wk2i * wk1r - wk1i;

        for j in (k..l + k).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = a[j] + a[j1];
            let x0i = a[j + 1] + a[j1 + 1];
            let x1r = a[j] - a[j1];
            let x1i = a[j + 1] - a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j] = x0r + x2r;
            a[j + 1] = x0i + x2i;
            let x0r = x0r - x2r;
            let x0i = x0i - x2i;
            a[j2] = wk2r * x0r - wk2i * x0i;
            a[j2 + 1] = wk2r * x0i + wk2i * x0r;
            let x0r = x1r - x3i;
            let x0i = x1i + x3r;
            a[j1] = wk1r * x0r - wk1i * x0i;
            a[j1 + 1] = wk1r * x0i + wk1i * x0r;
            let x0r = x1r + x3i;
            let x0i = x1i - x3r;
            a[j3] = wk3r * x0r - wk3i * x0i;
            a[j3 + 1] = wk3r * x0i + wk3i * x0r;
        }

        let wk1r = w[k2 + 2];
        let wk1i = w[k2 + 3];
        let wk3r = wk1r - 2.0 * wk2r * wk1i;
        let wk3i = 2.0 * wk2r * wk1r - wk1i;

        for j in (k + m..l + (k + m)).step_by(2) {
            let j1 = j + l;
            let j2 = j1 + l;
            let j3 = j2 + l;
            let x0r = a[j] + a[j1];
            let x0i = a[j + 1] + a[j1 + 1];
            let x1r = a[j] - a[j1];
            let x1i = a[j + 1] - a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j] = x0r + x2r;
            a[j + 1] = x0i + x2i;
            let x0r = x0r - x2r;
            let x0i = x0i - x2i;
            a[j2] = -wk2i * x0r - wk2r * x0i;
            a[j2 + 1] = -wk2i * x0i + wk2r * x0r;
            let x0r = x1r - x3i;
            let x0i = x1i + x3r;
            a[j1] = wk1r * x0r - wk1i * x0i;
            a[j1 + 1] = wk1r * x0i + wk1i * x0r;
            let x0r = x1r + x3i;
            let x0i = x1i - x3r;
            a[j3] = wk3r * x0r - wk3i * x0i;
            a[j3 + 1] = wk3r * x0i + wk3i * x0r;
        }

        k += m2;
    }
}

/// Real FFT forward post-processing (split-radix real/imaginary separation).
fn rftfsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    let mut j = 2;
    while j < m {
        let k = n - j;
        kk += ks;
        let wkr = 0.5 - c[nc - kk];
        let wki = c[kk];
        let xr = a[j] - a[k];
        let xi = a[j + 1] + a[k + 1];
        let yr = wkr * xr - wki * xi;
        let yi = wkr * xi + wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
        j += 2;
    }
}

/// Real FFT backward pre-processing (split-radix real/imaginary recombination).
fn rftbsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let m = n >> 1;
    let ks = 2 * nc / m;
    let mut kk = 0;
    a[1] = -a[1];
    let mut j = 2;
    while j < m {
        let k = n - j;
        kk += ks;
        let wkr = 0.5 - c[nc - kk];
        let wki = c[kk];
        let xr = a[j] - a[k];
        let xi = a[j + 1] + a[k + 1];
        let yr = wkr * xr + wki * xi;
        let yi = wkr * xi - wki * xr;
        a[j] -= yr;
        a[j + 1] = yi - a[j + 1];
        a[k] += yr;
        a[k + 1] = yi - a[k + 1];
        j += 2;
    }
    a[m + 1] = -a[m + 1];
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
