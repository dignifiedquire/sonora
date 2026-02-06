//! 128-point real FFT using the Ooura algorithm.
//!
//! Port of WebRTC's `OouraFft` — a fixed-size, in-place, real-valued FFT
//! optimized for 128 samples. Originally by Takuya Ooura (1996-2001),
//! adapted by the WebRTC project with precomputed twiddle tables.
//!
//! C source: `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc`
//! (scalar path). Tables from `ooura_fft_tables_common.h`.
//!
//! # Data format
//!
//! Forward transform ([`forward`]):
//! - Input: 128 real-valued time-domain samples in `a[0..128]`
//! - Output: Packed frequency-domain data. `a[0]` = DC component,
//!   `a[1]` = Nyquist component, `a[2k], a[2k+1]` = real/imag of bin k.
//!
//! Inverse transform ([`inverse`]):
//! - Input: Packed frequency-domain data (as produced by [`forward`])
//! - Output: 128 real-valued time-domain samples (scaled by 2/N)

/// Fixed 128-point real FFT size.
pub const FFT_SIZE: usize = 128;

// Precomputed cos/sin values. These were originally computed at runtime
// but WebRTC hardcoded them for determinism and startup speed.

/// Common twiddle factors shared by all paths (C, SSE2, NEON).
const RDFT_W: [f32; 64] = [
    1.000_000_0,
    0.000_000_0,
    0.707_106_77,
    0.707_106_77,
    0.923_879_56,
    0.382_683_46,
    0.382_683_46,
    0.923_879_56,
    0.980_785_25,
    0.195_090_32,
    0.555_570_24,
    0.831_469_6,
    0.831_469_6,
    0.555_570_24,
    0.195_090_32,
    0.980_785_25,
    0.995_184_7,
    0.098_017_14,
    0.634_393_33,
    0.773_010_43,
    0.881_921_3,
    0.471_396_74,
    0.290_284_66,
    0.956_940_35,
    0.956_940_35,
    0.290_284_66,
    0.471_396_74,
    0.881_921_3,
    0.773_010_43,
    0.634_393_33,
    0.098_017_14,
    0.995_184_7,
    0.707_106_77,
    0.499_397_72,
    0.497_592_36,
    0.494_588_26,
    0.490_392_63,
    0.485_015_63,
    0.478_470_18,
    0.470_772_03,
    0.461_939_78,
    0.451_994_63,
    0.440_960_65,
    0.428_864_3,
    0.415_734_8,
    0.401_603_76,
    0.386_505_22,
    0.370_475_6,
    0.353_553_38,
    0.335_779_5,
    0.317_196_67,
    0.297_849_66,
    0.277_785_12,
    0.257_051_38,
    0.235_698_37,
    0.213_777_54,
    0.191_341_73,
    0.168_444_93,
    0.145_142_33,
    0.121_490_1,
    0.097_545_16,
    0.073_365_234,
    0.049_008_57,
    0.024_533_838,
];

/// Twiddle factors for the C/MIPS path (first half of k=3 factors).
const RDFT_WK3RI_FIRST: [f32; 16] = [
    1.000_000_0,
    0.000_000_0,
    0.382_683_46,
    0.923_879_56,
    0.831_469_54,
    0.555_570_24,
    -0.195_090_35,
    0.980_785_25,
    0.956_940_35,
    0.290_284_7,
    0.098_017_156,
    0.995_184_7,
    0.634_393_33,
    0.773_010_5,
    -0.471_396_86,
    0.881_921_2,
];

/// Twiddle factors for the C/MIPS path (second half of k=3 factors).
const RDFT_WK3RI_SECOND: [f32; 16] = [
    -0.707_106_77,
    0.707_106_77,
    -0.923_879_56,
    -0.382_683_46,
    -0.980_785_25,
    0.195_090_35,
    -0.555_570_24,
    -0.831_469_54,
    -0.881_921_2,
    0.471_396_86,
    -0.773_010_5,
    -0.634_393_33,
    -0.995_184_7,
    -0.098_017_156,
    -0.290_284_7,
    -0.956_940_35,
];

/// Forward 128-point real FFT (time domain → frequency domain)
/// (C: `WebRtc_rdft(128, 1, ...)`).
///
/// Transforms `a` in-place. After the call:
/// - `a[0]` = DC component
/// - `a[1]` = Nyquist component
/// - `a[2k], a[2k+1]` = real/imaginary of frequency bin k
///
/// Uses SSE2-accelerated butterflies on x86/x86_64, scalar otherwise.
pub fn forward(a: &mut [f32; FFT_SIZE]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse2") {
        return forward_sse2(a);
    }

    forward_scalar(a);
}

/// Inverse 128-point real FFT (frequency domain → time domain)
/// (C: `WebRtc_rdft(128, -1, ...)`).
///
/// Transforms `a` in-place. To recover the original signal,
/// multiply each output element by `2/N` (i.e., `2/128`).
///
/// Uses SSE2-accelerated butterflies on x86/x86_64, scalar otherwise.
pub fn inverse(a: &mut [f32; FFT_SIZE]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse2") {
        return inverse_sse2(a);
    }

    inverse_scalar(a);
}

/// Scalar forward FFT implementation.
pub(crate) fn forward_scalar(a: &mut [f32; FFT_SIZE]) {
    bitrv2_128(a);
    cftfsub_128(a);
    rftfsub_128(a);
    let xi = a[0] - a[1];
    a[0] += a[1];
    a[1] = xi;
}

/// Scalar inverse FFT implementation.
pub(crate) fn inverse_scalar(a: &mut [f32; FFT_SIZE]) {
    a[1] = 0.5 * (a[0] - a[1]);
    a[0] -= a[1];
    rftbsub_128(a);
    bitrv2_128(a);
    cftbsub_128(a);
}

/// SSE2-accelerated forward FFT implementation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn forward_sse2(a: &mut [f32; FFT_SIZE]) {
    bitrv2_128(a);
    // SAFETY: caller guarantees SSE2 is available (checked in `forward`).
    unsafe {
        crate::ooura_fft_sse2::cft1st_128_sse2(a);
        crate::ooura_fft_sse2::cftmdl_128_sse2(a);
    }
    cftfsub_128_final(a);
    // SAFETY: SSE2 available.
    unsafe {
        crate::ooura_fft_sse2::rftfsub_128_sse2(a);
    }
    let xi = a[0] - a[1];
    a[0] += a[1];
    a[1] = xi;
}

/// SSE2-accelerated inverse FFT implementation.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn inverse_sse2(a: &mut [f32; FFT_SIZE]) {
    a[1] = 0.5 * (a[0] - a[1]);
    a[0] -= a[1];
    // SAFETY: caller guarantees SSE2 is available (checked in `inverse`).
    unsafe {
        crate::ooura_fft_sse2::rftbsub_128_sse2(a);
    }
    bitrv2_128(a);
    // SAFETY: SSE2 available.
    unsafe {
        crate::ooura_fft_sse2::cft1st_128_sse2(a);
        crate::ooura_fft_sse2::cftmdl_128_sse2(a);
    }
    cftbsub_128_final(a);
}

/// Bit-reversal permutation for 128-point FFT.
fn bitrv2_128(a: &mut [f32; FFT_SIZE]) {
    let ip: [usize; 4] = [0, 64, 32, 96];
    for k in 0..4_usize {
        for j in 0..k {
            let j1 = 2 * j + ip[k];
            let k1 = 2 * k + ip[j];

            a.swap(j1, k1);
            a.swap(j1 + 1, k1 + 1);

            let j1 = j1 + 8;
            let k1 = k1 + 16;
            a.swap(j1, k1);
            a.swap(j1 + 1, k1 + 1);

            let j1 = j1 + 8;
            let k1 = k1 - 8;
            a.swap(j1, k1);
            a.swap(j1 + 1, k1 + 1);

            let j1 = j1 + 8;
            let k1 = k1 + 16;
            a.swap(j1, k1);
            a.swap(j1 + 1, k1 + 1);
        }
        let j1 = 2 * k + 8 + ip[k];
        let k1 = j1 + 8;
        a.swap(j1, k1);
        a.swap(j1 + 1, k1 + 1);
    }
}

/// First-stage complex FFT butterfly.
fn cft1st_128(a: &mut [f32; FFT_SIZE]) {
    let n = FFT_SIZE;

    // First 8 elements — simplified (no twiddle multiply).
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

    // Elements 8..16 — wk1r only.
    let wk1r = RDFT_W[2];
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

    // Remaining elements 16..128 with full twiddle factors.
    let mut k1 = 0_usize;
    let mut j = 16;
    while j < n {
        k1 += 2;
        let k2 = 2 * k1;
        let wk2r = RDFT_W[k1];
        let wk2i = RDFT_W[k1 + 1];
        let wk1r = RDFT_W[k2];
        let wk1i = RDFT_W[k2 + 1];
        let wk3r = RDFT_WK3RI_FIRST[k1];
        let wk3i = RDFT_WK3RI_FIRST[k1 + 1];

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

        let wk1r = RDFT_W[k2 + 2];
        let wk1i = RDFT_W[k2 + 3];
        let wk3r = RDFT_WK3RI_SECOND[k1];
        let wk3i = RDFT_WK3RI_SECOND[k1 + 1];

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
fn cftmdl_128(a: &mut [f32; FFT_SIZE]) {
    let l = 8_usize;
    let n = FFT_SIZE;
    let m = 32_usize;

    // First block: no twiddle factors.
    for j0 in (0..l).step_by(2) {
        let j1 = j0 + 8;
        let j2 = j0 + 16;
        let j3 = j0 + 24;
        let x0r = a[j0] + a[j1];
        let x0i = a[j0 + 1] + a[j1 + 1];
        let x1r = a[j0] - a[j1];
        let x1i = a[j0 + 1] - a[j1 + 1];
        let x2r = a[j2] + a[j3];
        let x2i = a[j2 + 1] + a[j3 + 1];
        let x3r = a[j2] - a[j3];
        let x3i = a[j2 + 1] - a[j3 + 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j2] = x0r - x2r;
        a[j2 + 1] = x0i - x2i;
        a[j1] = x1r - x3i;
        a[j1 + 1] = x1i + x3r;
        a[j3] = x1r + x3i;
        a[j3 + 1] = x1i - x3r;
    }

    // Second block: wk1r only.
    let wk1r = RDFT_W[2];
    for j0 in (m..l + m).step_by(2) {
        let j1 = j0 + 8;
        let j2 = j0 + 16;
        let j3 = j0 + 24;
        let x0r = a[j0] + a[j1];
        let x0i = a[j0 + 1] + a[j1 + 1];
        let x1r = a[j0] - a[j1];
        let x1i = a[j0 + 1] - a[j1 + 1];
        let x2r = a[j2] + a[j3];
        let x2i = a[j2 + 1] + a[j3 + 1];
        let x3r = a[j2] - a[j3];
        let x3i = a[j2 + 1] - a[j3 + 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
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

    // Remaining blocks with full twiddle factors.
    let mut k1 = 0_usize;
    let m2 = 2 * m;
    let mut k = m2;
    while k < n {
        k1 += 2;
        let k2 = 2 * k1;
        let wk2r = RDFT_W[k1];
        let wk2i = RDFT_W[k1 + 1];
        let wk1r = RDFT_W[k2];
        let wk1i = RDFT_W[k2 + 1];
        let wk3r = RDFT_WK3RI_FIRST[k1];
        let wk3i = RDFT_WK3RI_FIRST[k1 + 1];

        for j0 in (k..l + k).step_by(2) {
            let j1 = j0 + 8;
            let j2 = j0 + 16;
            let j3 = j0 + 24;
            let x0r = a[j0] + a[j1];
            let x0i = a[j0 + 1] + a[j1 + 1];
            let x1r = a[j0] - a[j1];
            let x1i = a[j0 + 1] - a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j0] = x0r + x2r;
            a[j0 + 1] = x0i + x2i;
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

        let wk1r = RDFT_W[k2 + 2];
        let wk1i = RDFT_W[k2 + 3];
        let wk3r = RDFT_WK3RI_SECOND[k1];
        let wk3i = RDFT_WK3RI_SECOND[k1 + 1];

        for j0 in (k + m..l + (k + m)).step_by(2) {
            let j1 = j0 + 8;
            let j2 = j0 + 16;
            let j3 = j0 + 24;
            let x0r = a[j0] + a[j1];
            let x0i = a[j0 + 1] + a[j1 + 1];
            let x1r = a[j0] - a[j1];
            let x1i = a[j0 + 1] - a[j1 + 1];
            let x2r = a[j2] + a[j3];
            let x2i = a[j2 + 1] + a[j3 + 1];
            let x3r = a[j2] - a[j3];
            let x3i = a[j2 + 1] - a[j3 + 1];
            a[j0] = x0r + x2r;
            a[j0 + 1] = x0i + x2i;
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

/// Forward complex sub-transform (radix-4 decomposition).
fn cftfsub_128(a: &mut [f32; FFT_SIZE]) {
    cft1st_128(a);
    cftmdl_128(a);
    cftfsub_128_final(a);
}

/// Final radix-4 pass of the forward complex sub-transform.
/// Shared between scalar and SIMD paths (only `cft1st_128` and `cftmdl_128`
/// are replaced by SIMD — this final pass is always scalar).
fn cftfsub_128_final(a: &mut [f32; FFT_SIZE]) {
    let l = 32_usize;
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
}

/// Backward complex sub-transform (radix-4 decomposition).
fn cftbsub_128(a: &mut [f32; FFT_SIZE]) {
    cft1st_128(a);
    cftmdl_128(a);
    cftbsub_128_final(a);
}

/// Final radix-4 pass of the backward complex sub-transform.
/// Shared between scalar and SIMD paths.
fn cftbsub_128_final(a: &mut [f32; FFT_SIZE]) {
    let l = 32_usize;
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
}

/// Real FFT forward post-processing (split-radix real/imaginary separation).
fn rftfsub_128(a: &mut [f32; FFT_SIZE]) {
    let c = &RDFT_W[32..];
    for j1 in 1..32_usize {
        let j2 = 2 * j1;
        let k2 = 128 - j2;
        let k1 = 32 - j1;
        let wkr = 0.5 - c[k1];
        let wki = c[j1];
        let xr = a[j2] - a[k2];
        let xi = a[j2 + 1] + a[k2 + 1];
        let yr = wkr * xr - wki * xi;
        let yi = wkr * xi + wki * xr;
        a[j2] -= yr;
        a[j2 + 1] -= yi;
        a[k2] += yr;
        a[k2 + 1] -= yi;
    }
}

/// Real FFT backward pre-processing (split-radix real/imaginary recombination).
fn rftbsub_128(a: &mut [f32; FFT_SIZE]) {
    let c = &RDFT_W[32..];
    a[1] = -a[1];
    for j1 in 1..32_usize {
        let j2 = 2 * j1;
        let k2 = 128 - j2;
        let k1 = 32 - j1;
        let wkr = 0.5 - c[k1];
        let wki = c[j1];
        let xr = a[j2] - a[k2];
        let xi = a[j2 + 1] + a[k2 + 1];
        let yr = wkr * xr + wki * xi;
        let yi = wkr * xi - wki * xr;
        a[j2] -= yr;
        a[j2 + 1] = yi - a[j2 + 1];
        a[k2] += yr;
        a[k2 + 1] = yi - a[k2 + 1];
    }
    a[65] = -a[65];
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;

    #[test]
    fn forward_inverse_roundtrip() {
        let mut a = [0.0_f32; FFT_SIZE];
        // Fill with a known signal.
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 + (-0.5_f32).powi(i as i32);
        }
        let original = a;

        forward(&mut a);
        inverse(&mut a);

        // Ooura convention: roundtrip requires scaling by 2/N.
        let scale = 2.0 / FFT_SIZE as f32;
        for (i, (&o, &r)) in original.iter().zip(a.iter()).enumerate() {
            let recovered = r * scale;
            assert!(
                (o - recovered).abs() < 1e-4,
                "mismatch at index {i}: original={o}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn impulse_response() {
        // Delta function at index 0.
        let mut a = [0.0_f32; FFT_SIZE];
        a[0] = 1.0;

        forward(&mut a);

        // DC component should be 1.0.
        assert!((a[0] - 1.0).abs() < 1e-6, "DC = {}", a[0]);
        // Nyquist should be 1.0 (all samples are +1 at even indices).
        assert!((a[1] - 1.0).abs() < 1e-6, "Nyquist = {}", a[1]);
        // All other real parts should be 1.0, imaginary 0.0.
        for k in 1..64 {
            let re = a[2 * k];
            let im = a[2 * k + 1];
            assert!(
                (re - 1.0).abs() < 1e-5,
                "bin {k} real: expected 1.0, got {re}"
            );
            assert!(im.abs() < 1e-5, "bin {k} imag: expected 0.0, got {im}");
        }
    }

    #[test]
    fn linearity() {
        let mut signal_a = [0.0_f32; FFT_SIZE];
        let mut signal_b = [0.0_f32; FFT_SIZE];
        let mut signal_sum = [0.0_f32; FFT_SIZE];

        for i in 0..FFT_SIZE {
            signal_a[i] = (i as f32 * 0.1).sin();
            signal_b[i] = (i as f32 * 0.3).cos();
            signal_sum[i] = signal_a[i] + signal_b[i];
        }

        forward(&mut signal_a);
        forward(&mut signal_b);
        forward(&mut signal_sum);

        // FFT(a+b) should equal FFT(a) + FFT(b).
        for i in 0..FFT_SIZE {
            let expected = signal_a[i] + signal_b[i];
            assert!(
                (signal_sum[i] - expected).abs() < 1e-4,
                "linearity failed at index {i}: FFT(a+b)={}, FFT(a)+FFT(b)={expected}",
                signal_sum[i]
            );
        }
    }

    #[test]
    fn zero_input() {
        let mut a = [0.0_f32; FFT_SIZE];
        forward(&mut a);
        for (i, &v) in a.iter().enumerate() {
            assert_eq!(v, 0.0, "expected zero at index {i}, got {v}");
        }
    }

    #[test]
    fn dc_signal() {
        let mut a = [1.0_f32; FFT_SIZE];
        forward(&mut a);

        // DC = sum of all samples = N.
        assert!(
            (a[0] - FFT_SIZE as f32).abs() < 1e-4,
            "DC = {}, expected {}",
            a[0],
            FFT_SIZE
        );
        // Nyquist = sum((-1)^n * x[n]) = 0 for constant signal.
        assert!(a[1].abs() < 1e-4, "Nyquist = {}, expected ~0", a[1]);
        // All other bins should be zero.
        for k in 1..64 {
            assert!(a[2 * k].abs() < 1e-4, "bin {k} real = {}", a[2 * k]);
            assert!(a[2 * k + 1].abs() < 1e-4, "bin {k} imag = {}", a[2 * k + 1]);
        }
    }

    // -- Property tests --

    #[proptest]
    fn roundtrip_recovers_signal(
        #[strategy(prop::collection::vec(-1.0f32..1.0, 128))] signal: Vec<f32>,
    ) {
        let mut a = [0.0f32; FFT_SIZE];
        a.copy_from_slice(&signal);
        let original = a;

        forward(&mut a);
        inverse(&mut a);

        let scale = 2.0 / FFT_SIZE as f32;
        for (i, (&o, &r)) in original.iter().zip(a.iter()).enumerate() {
            prop_assert!(
                (o - r * scale).abs() < 1e-4,
                "mismatch at {i}: original={o}, recovered={}",
                r * scale
            );
        }
    }

    #[proptest]
    fn linearity_holds(
        #[strategy(prop::collection::vec(-1.0f32..1.0, 128))] sig_a: Vec<f32>,
        #[strategy(prop::collection::vec(-1.0f32..1.0, 128))] sig_b: Vec<f32>,
    ) {
        let mut a = [0.0f32; FFT_SIZE];
        let mut b = [0.0f32; FFT_SIZE];
        let mut sum = [0.0f32; FFT_SIZE];
        a.copy_from_slice(&sig_a);
        b.copy_from_slice(&sig_b);
        for i in 0..FFT_SIZE {
            sum[i] = a[i] + b[i];
        }

        forward(&mut a);
        forward(&mut b);
        forward(&mut sum);

        for i in 0..FFT_SIZE {
            let expected = a[i] + b[i];
            prop_assert!(
                (sum[i] - expected).abs() < 1e-3,
                "linearity failed at {i}: FFT(a+b)={}, FFT(a)+FFT(b)={expected}",
                sum[i]
            );
        }
    }

    #[proptest]
    fn parseval_energy_conservation(
        #[strategy(prop::collection::vec(-1.0f32..1.0, 128))] signal: Vec<f32>,
    ) {
        let mut a = [0.0f32; FFT_SIZE];
        a.copy_from_slice(&signal);
        let time_energy: f32 = a.iter().map(|x| x * x).sum();

        forward(&mut a);

        let dc_sq = a[0] * a[0];
        let nyq_sq = a[1] * a[1];
        let mut freq_energy = dc_sq + nyq_sq;
        for k in 1..FFT_SIZE / 2 {
            freq_energy += 2.0 * (a[2 * k] * a[2 * k] + a[2 * k + 1] * a[2 * k + 1]);
        }

        let expected = FFT_SIZE as f32 * time_energy;
        // Use relative tolerance; skip near-zero energy to avoid division issues.
        if expected > 1e-6 {
            prop_assert!(
                (freq_energy - expected).abs() / expected < 1e-3,
                "Parseval: freq_energy={freq_energy}, expected={expected}"
            );
        }
    }

    /// Verify SSE2 forward/inverse FFT produce the same output as scalar.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let signals: [[f32; FFT_SIZE]; 5] = [
            std::array::from_fn(|i| i as f32 * 0.01),
            std::array::from_fn(|i| (i as f32 * 0.1).sin()),
            std::array::from_fn(|i| if i % 2 == 0 { 0.5 } else { -0.3 }),
            [0.7; FFT_SIZE],
            {
                let mut s = [0.0f32; FFT_SIZE];
                s[0] = 1.0;
                s
            },
        ];

        for signal in &signals {
            let mut scalar = *signal;
            forward_scalar(&mut scalar);

            let mut simd = *signal;
            forward_sse2(&mut simd);

            for i in 0..FFT_SIZE {
                assert!(
                    (scalar[i] - simd[i]).abs() < 1e-6,
                    "forward mismatch at {i}: scalar={}, sse2={}",
                    scalar[i],
                    simd[i]
                );
            }

            let mut scalar_inv = scalar;
            inverse_scalar(&mut scalar_inv);

            let mut simd_inv = scalar;
            inverse_sse2(&mut simd_inv);

            for i in 0..FFT_SIZE {
                assert!(
                    (scalar_inv[i] - simd_inv[i]).abs() < 1e-5,
                    "inverse mismatch at {i}: scalar={}, sse2={}",
                    scalar_inv[i],
                    simd_inv[i]
                );
            }
        }
    }
}
