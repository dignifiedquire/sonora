//! NEON-accelerated inner functions for the 128-point Ooura FFT.
//!
//! C source: `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_neon.cc`
//! Tables from `ooura_fft_tables_neon_sse2.h`.

#![allow(
    clippy::excessive_precision,
    reason = "twiddle tables are exact copies from upstream C++ for bit-exact reproducibility"
)]

use core::arch::aarch64::*;

use crate::ooura_fft::{FFT_SIZE, RDFT_W};

// ── SIMD twiddle tables ──────────────────────────────────────────────────────
// Shared with SSE2 — duplicated/rearranged for 4-wide SIMD loads.

const K_SWAP_SIGN: [f32; 4] = [-1.0, 1.0, -1.0, 1.0];

const RDFT_WK1R: [f32; 32] = [
    1.000_000_000,
    1.000_000_000,
    0.707_106_769,
    0.707_106_769,
    0.923_879_564,
    0.923_879_564,
    0.382_683_456,
    0.382_683_456,
    0.980_785_251,
    0.980_785_251,
    0.555_570_245,
    0.555_570_245,
    0.831_469_595,
    0.831_469_595,
    0.195_090_324,
    0.195_090_324,
    0.995_184_720,
    0.995_184_720,
    0.634_393_334,
    0.634_393_334,
    0.881_921_291,
    0.881_921_291,
    0.290_284_663,
    0.290_284_663,
    0.956_940_353,
    0.956_940_353,
    0.471_396_744,
    0.471_396_744,
    0.773_010_433,
    0.773_010_433,
    0.098_017_141,
    0.098_017_141,
];

const RDFT_WK2R: [f32; 32] = [
    1.000_000_000,
    1.000_000_000,
    -0.000_000_000,
    -0.000_000_000,
    0.707_106_769,
    0.707_106_769,
    -0.707_106_769,
    -0.707_106_769,
    0.923_879_564,
    0.923_879_564,
    -0.382_683_456,
    -0.382_683_456,
    0.382_683_456,
    0.382_683_456,
    -0.923_879_564,
    -0.923_879_564,
    0.980_785_251,
    0.980_785_251,
    -0.195_090_324,
    -0.195_090_324,
    0.555_570_245,
    0.555_570_245,
    -0.831_469_595,
    -0.831_469_595,
    0.831_469_595,
    0.831_469_595,
    -0.555_570_245,
    -0.555_570_245,
    0.195_090_324,
    0.195_090_324,
    -0.980_785_251,
    -0.980_785_251,
];

const RDFT_WK3R: [f32; 32] = [
    1.000_000_000,
    1.000_000_000,
    -0.707_106_769,
    -0.707_106_769,
    0.382_683_456,
    0.382_683_456,
    -0.923_879_564,
    -0.923_879_564,
    0.831_469_536,
    0.831_469_536,
    -0.980_785_251,
    -0.980_785_251,
    -0.195_090_353,
    -0.195_090_353,
    -0.555_570_245,
    -0.555_570_245,
    0.956_940_353,
    0.956_940_353,
    -0.881_921_172,
    -0.881_921_172,
    0.098_017_156,
    0.098_017_156,
    -0.773_010_492,
    -0.773_010_492,
    0.634_393_334,
    0.634_393_334,
    -0.995_184_720,
    -0.995_184_720,
    -0.471_396_863,
    -0.471_396_863,
    -0.290_284_693,
    -0.290_284_693,
];

const RDFT_WK1I: [f32; 32] = [
    -0.000_000_000,
    0.000_000_000,
    -0.707_106_769,
    0.707_106_769,
    -0.382_683_456,
    0.382_683_456,
    -0.923_879_564,
    0.923_879_564,
    -0.195_090_324,
    0.195_090_324,
    -0.831_469_595,
    0.831_469_595,
    -0.555_570_245,
    0.555_570_245,
    -0.980_785_251,
    0.980_785_251,
    -0.098_017_141,
    0.098_017_141,
    -0.773_010_433,
    0.773_010_433,
    -0.471_396_744,
    0.471_396_744,
    -0.956_940_353,
    0.956_940_353,
    -0.290_284_663,
    0.290_284_663,
    -0.881_921_291,
    0.881_921_291,
    -0.634_393_334,
    0.634_393_334,
    -0.995_184_720,
    0.995_184_720,
];

const RDFT_WK2I: [f32; 32] = [
    -0.000_000_000,
    0.000_000_000,
    -1.000_000_000,
    1.000_000_000,
    -0.707_106_769,
    0.707_106_769,
    -0.707_106_769,
    0.707_106_769,
    -0.382_683_456,
    0.382_683_456,
    -0.923_879_564,
    0.923_879_564,
    -0.923_879_564,
    0.923_879_564,
    -0.382_683_456,
    0.382_683_456,
    -0.195_090_324,
    0.195_090_324,
    -0.980_785_251,
    0.980_785_251,
    -0.831_469_595,
    0.831_469_595,
    -0.555_570_245,
    0.555_570_245,
    -0.555_570_245,
    0.555_570_245,
    -0.831_469_595,
    0.831_469_595,
    -0.980_785_251,
    0.980_785_251,
    -0.195_090_324,
    0.195_090_324,
];

const RDFT_WK3I: [f32; 32] = [
    -0.000_000_000,
    0.000_000_000,
    -0.707_106_769,
    0.707_106_769,
    -0.923_879_564,
    0.923_879_564,
    0.382_683_456,
    -0.382_683_456,
    -0.555_570_245,
    0.555_570_245,
    -0.195_090_353,
    0.195_090_353,
    -0.980_785_251,
    0.980_785_251,
    0.831_469_536,
    -0.831_469_536,
    -0.290_284_693,
    0.290_284_693,
    -0.471_396_863,
    0.471_396_863,
    -0.995_184_720,
    0.995_184_720,
    0.634_393_334,
    -0.634_393_334,
    -0.773_010_492,
    0.773_010_492,
    0.098_017_156,
    -0.098_017_156,
    -0.881_921_172,
    0.881_921_172,
    0.956_940_353,
    -0.956_940_353,
];

const CFTMDL_WK1R: [f32; 4] = [0.707_106_769, 0.707_106_769, 0.707_106_769, -0.707_106_769];

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Reverse element order: [A, B, C, D] -> [D, C, B, A]
/// C: `reverse_order_f32x4`
#[inline(always)]
unsafe fn reverse_order_f32x4(v: float32x4_t) -> float32x4_t {
    unsafe {
        // A B C D -> C D A B
        let rev = vcombine_f32(vget_high_f32(v), vget_low_f32(v));
        // C D A B -> D C B A
        vrev64q_f32(rev)
    }
}

// ── NEON inner functions ─────────────────────────────────────────────────────

/// First-stage complex FFT butterfly (NEON).
/// C: `cft1st_128_neon`
#[target_feature(enable = "neon")]
pub(crate) unsafe fn cft1st_128_neon(a: &mut [f32; FFT_SIZE]) {
    unsafe {
        let vec_swap_sign = vld1q_f32(K_SWAP_SIGN.as_ptr());

        let mut k2 = 0usize;
        let mut j = 0usize;
        while j < 128 {
            let a00v = vld1q_f32(a.as_ptr().add(j));
            let a04v = vld1q_f32(a.as_ptr().add(j + 4));
            let a08v = vld1q_f32(a.as_ptr().add(j + 8));
            let a12v = vld1q_f32(a.as_ptr().add(j + 12));

            let a01v = vcombine_f32(vget_low_f32(a00v), vget_low_f32(a08v));
            let a23v = vcombine_f32(vget_high_f32(a00v), vget_high_f32(a08v));
            let a45v = vcombine_f32(vget_low_f32(a04v), vget_low_f32(a12v));
            let a67v = vcombine_f32(vget_high_f32(a04v), vget_high_f32(a12v));

            let wk1rv = vld1q_f32(RDFT_WK1R.as_ptr().add(k2));
            let wk1iv = vld1q_f32(RDFT_WK1I.as_ptr().add(k2));
            let wk2rv = vld1q_f32(RDFT_WK2R.as_ptr().add(k2));
            let wk2iv = vld1q_f32(RDFT_WK2I.as_ptr().add(k2));
            let wk3rv = vld1q_f32(RDFT_WK3R.as_ptr().add(k2));
            let wk3iv = vld1q_f32(RDFT_WK3I.as_ptr().add(k2));

            let mut x0v = vaddq_f32(a01v, a23v);
            let x1v = vsubq_f32(a01v, a23v);
            let x2v = vaddq_f32(a45v, a67v);
            let x3v = vsubq_f32(a45v, a67v);
            let x3w = vrev64q_f32(x3v);

            let a01v = vaddq_f32(x0v, x2v);
            x0v = vsubq_f32(x0v, x2v);
            let x0w = vrev64q_f32(x0v);
            let mut a45v = vmulq_f32(wk2rv, x0v);
            a45v = vmlaq_f32(a45v, wk2iv, x0w);

            let x0v = vmlaq_f32(x1v, x3w, vec_swap_sign);
            let x0w = vrev64q_f32(x0v);
            let mut a23v = vmulq_f32(wk1rv, x0v);
            a23v = vmlaq_f32(a23v, wk1iv, x0w);

            let x0v = vmlsq_f32(x1v, x3w, vec_swap_sign);
            let x0w = vrev64q_f32(x0v);
            let mut a67v = vmulq_f32(wk3rv, x0v);
            a67v = vmlaq_f32(a67v, wk3iv, x0w);

            let a00v = vcombine_f32(vget_low_f32(a01v), vget_low_f32(a23v));
            let a04v = vcombine_f32(vget_low_f32(a45v), vget_low_f32(a67v));
            let a08v = vcombine_f32(vget_high_f32(a01v), vget_high_f32(a23v));
            let a12v = vcombine_f32(vget_high_f32(a45v), vget_high_f32(a67v));

            vst1q_f32(a.as_mut_ptr().add(j), a00v);
            vst1q_f32(a.as_mut_ptr().add(j + 4), a04v);
            vst1q_f32(a.as_mut_ptr().add(j + 8), a08v);
            vst1q_f32(a.as_mut_ptr().add(j + 12), a12v);

            j += 16;
            k2 += 4;
        }
    }
}

/// Modular complex FFT butterfly stage (NEON).
/// C: `cftmdl_128_neon`
#[target_feature(enable = "neon")]
pub(crate) unsafe fn cftmdl_128_neon(a: &mut [f32; FFT_SIZE]) {
    unsafe {
        let l = 8usize;
        let vec_swap_sign = vld1q_f32(K_SWAP_SIGN.as_ptr());
        let mut wk1rv = vld1q_f32(CFTMDL_WK1R.as_ptr());

        // First block (j = 0..8 step 2)
        for j in (0..l).step_by(2) {
            let a_00 = vld1_f32(a.as_ptr().add(j));
            let a_08 = vld1_f32(a.as_ptr().add(j + 8));
            let a_32 = vld1_f32(a.as_ptr().add(j + 32));
            let a_40 = vld1_f32(a.as_ptr().add(j + 40));
            let a_00_32 = vcombine_f32(a_00, a_32);
            let a_08_40 = vcombine_f32(a_08, a_40);
            let x0r0_0i0_0r1_x0i1 = vaddq_f32(a_00_32, a_08_40);
            let x1r0_1i0_1r1_x1i1 = vsubq_f32(a_00_32, a_08_40);

            let a_16 = vld1_f32(a.as_ptr().add(j + 16));
            let a_24 = vld1_f32(a.as_ptr().add(j + 24));
            let a_48 = vld1_f32(a.as_ptr().add(j + 48));
            let a_56 = vld1_f32(a.as_ptr().add(j + 56));
            let a_16_48 = vcombine_f32(a_16, a_48);
            let a_24_56 = vcombine_f32(a_24, a_56);
            let x2r0_2i0_2r1_x2i1 = vaddq_f32(a_16_48, a_24_56);
            let x3r0_3i0_3r1_x3i1 = vsubq_f32(a_16_48, a_24_56);

            let xx0 = vaddq_f32(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
            let xx1 = vsubq_f32(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);

            let x3i0_3r0_3i1_x3r1 = vrev64q_f32(x3r0_3i0_3r1_x3i1);
            let x1_x3_add = vmlaq_f32(x1r0_1i0_1r1_x1i1, vec_swap_sign, x3i0_3r0_3i1_x3r1);
            let x1_x3_sub = vmlsq_f32(x1r0_1i0_1r1_x1i1, vec_swap_sign, x3i0_3r0_3i1_x3r1);

            let yy0_a = vdup_lane_f32(vget_high_f32(x1_x3_add), 0);
            let yy0_s = vdup_lane_f32(vget_high_f32(x1_x3_sub), 0);
            let yy0_as = vcombine_f32(yy0_a, yy0_s);
            let yy1_a = vdup_lane_f32(vget_high_f32(x1_x3_add), 1);
            let yy1_s = vdup_lane_f32(vget_high_f32(x1_x3_sub), 1);
            let yy1_as = vcombine_f32(yy1_a, yy1_s);
            let yy0 = vmlaq_f32(yy0_as, vec_swap_sign, yy1_as);
            let yy4 = vmulq_f32(wk1rv, yy0);

            let xx1_rev = vrev64q_f32(xx1);
            let yy4_rev = vrev64q_f32(yy4);

            vst1_f32(a.as_mut_ptr().add(j), vget_low_f32(xx0));
            vst1_f32(a.as_mut_ptr().add(j + 32), vget_high_f32(xx0));
            vst1_f32(a.as_mut_ptr().add(j + 16), vget_low_f32(xx1));
            vst1_f32(a.as_mut_ptr().add(j + 48), vget_high_f32(xx1_rev));

            a[j + 48] = -a[j + 48];

            vst1_f32(a.as_mut_ptr().add(j + 8), vget_low_f32(x1_x3_add));
            vst1_f32(a.as_mut_ptr().add(j + 24), vget_low_f32(x1_x3_sub));
            vst1_f32(a.as_mut_ptr().add(j + 40), vget_low_f32(yy4));
            vst1_f32(a.as_mut_ptr().add(j + 56), vget_high_f32(yy4_rev));
        }

        // Second block (k=64)
        {
            let k = 64usize;
            let k2 = 4usize; // 2 * k1 where k1 = 2
            let wk2rv = vld1q_f32(RDFT_WK2R.as_ptr().add(k2));
            let wk2iv = vld1q_f32(RDFT_WK2I.as_ptr().add(k2));
            let wk1iv = vld1q_f32(RDFT_WK1I.as_ptr().add(k2));
            let wk3rv = vld1q_f32(RDFT_WK3R.as_ptr().add(k2));
            let wk3iv = vld1q_f32(RDFT_WK3I.as_ptr().add(k2));
            wk1rv = vld1q_f32(RDFT_WK1R.as_ptr().add(k2));

            for j in (k..l + k).step_by(2) {
                let a_00 = vld1_f32(a.as_ptr().add(j));
                let a_08 = vld1_f32(a.as_ptr().add(j + 8));
                let a_32 = vld1_f32(a.as_ptr().add(j + 32));
                let a_40 = vld1_f32(a.as_ptr().add(j + 40));
                let a_00_32 = vcombine_f32(a_00, a_32);
                let a_08_40 = vcombine_f32(a_08, a_40);
                let x0r0_0i0_0r1_x0i1 = vaddq_f32(a_00_32, a_08_40);
                let x1r0_1i0_1r1_x1i1 = vsubq_f32(a_00_32, a_08_40);

                let a_16 = vld1_f32(a.as_ptr().add(j + 16));
                let a_24 = vld1_f32(a.as_ptr().add(j + 24));
                let a_48 = vld1_f32(a.as_ptr().add(j + 48));
                let a_56 = vld1_f32(a.as_ptr().add(j + 56));
                let a_16_48 = vcombine_f32(a_16, a_48);
                let a_24_56 = vcombine_f32(a_24, a_56);
                let x2r0_2i0_2r1_x2i1 = vaddq_f32(a_16_48, a_24_56);
                let x3r0_3i0_3r1_x3i1 = vsubq_f32(a_16_48, a_24_56);

                let xx = vaddq_f32(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
                let xx1 = vsubq_f32(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
                let x3i0_3r0_3i1_x3r1 = vrev64q_f32(x3r0_3i0_3r1_x3i1);
                let x1_x3_add = vmlaq_f32(x1r0_1i0_1r1_x1i1, vec_swap_sign, x3i0_3r0_3i1_x3r1);
                let x1_x3_sub = vmlsq_f32(x1r0_1i0_1r1_x1i1, vec_swap_sign, x3i0_3r0_3i1_x3r1);

                let mut xx4 = vmulq_f32(wk2rv, xx1);
                let mut xx12 = vmulq_f32(wk1rv, x1_x3_add);
                let mut xx22 = vmulq_f32(wk3rv, x1_x3_sub);
                xx4 = vmlaq_f32(xx4, wk2iv, vrev64q_f32(xx1));
                xx12 = vmlaq_f32(xx12, wk1iv, vrev64q_f32(x1_x3_add));
                xx22 = vmlaq_f32(xx22, wk3iv, vrev64q_f32(x1_x3_sub));

                vst1_f32(a.as_mut_ptr().add(j), vget_low_f32(xx));
                vst1_f32(a.as_mut_ptr().add(j + 32), vget_high_f32(xx));
                vst1_f32(a.as_mut_ptr().add(j + 16), vget_low_f32(xx4));
                vst1_f32(a.as_mut_ptr().add(j + 48), vget_high_f32(xx4));
                vst1_f32(a.as_mut_ptr().add(j + 8), vget_low_f32(xx12));
                vst1_f32(a.as_mut_ptr().add(j + 40), vget_high_f32(xx12));
                vst1_f32(a.as_mut_ptr().add(j + 24), vget_low_f32(xx22));
                vst1_f32(a.as_mut_ptr().add(j + 56), vget_high_f32(xx22));
            }
        }
    }
}

/// Forward real FFT post-processing (NEON).
/// C: `rftfsub_128_neon`
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rftfsub_128_neon(a: &mut [f32; FFT_SIZE]) {
    unsafe {
        let c = &RDFT_W[32..];
        let mm_half = vdupq_n_f32(0.5);

        let mut j1 = 1usize;
        let mut j2 = 2usize;
        while j2 + 7 < 64 {
            let c_j1 = vld1q_f32(c.as_ptr().add(j1));
            let c_k1 = vld1q_f32(c.as_ptr().add(29 - j1));
            let wkrt = vsubq_f32(mm_half, c_k1);
            let wkr = reverse_order_f32x4(wkrt);
            let wki = c_j1;

            // Deinterleaved load: .0 = evens, .1 = odds
            let a_j2_p = vld2q_f32(a.as_ptr().add(j2));
            let k2_0_4 = vld2q_f32(a.as_ptr().add(122 - j2));
            let a_k2_p0 = reverse_order_f32x4(k2_0_4.0);
            let a_k2_p1 = reverse_order_f32x4(k2_0_4.1);

            let xr = vsubq_f32(a_j2_p.0, a_k2_p0);
            let xi = vaddq_f32(a_j2_p.1, a_k2_p1);

            // yr = wkr*xr - wki*xi; yi = wkr*xi + wki*xr
            let yr = vsubq_f32(vmulq_f32(wkr, xr), vmulq_f32(wki, xi));
            let yi = vaddq_f32(vmulq_f32(wkr, xi), vmulq_f32(wki, xr));

            let a_k2_p0n = vaddq_f32(a_k2_p0, yr);
            let a_k2_p1n = vsubq_f32(a_k2_p1, yi);

            // Re-interleave reversed k2 data
            let a_k2_p0nr = vrev64q_f32(a_k2_p0n);
            let a_k2_p1nr = vrev64q_f32(a_k2_p1n);
            let a_k2_n = vzipq_f32(a_k2_p0nr, a_k2_p1nr);

            let a_j2_p_out = float32x4x2_t(vsubq_f32(a_j2_p.0, yr), vsubq_f32(a_j2_p.1, yi));
            vst2q_f32(a.as_mut_ptr().add(j2), a_j2_p_out);

            vst1q_f32(a.as_mut_ptr().add(122 - j2), a_k2_n.1);
            vst1q_f32(a.as_mut_ptr().add(126 - j2), a_k2_n.0);

            j1 += 4;
            j2 += 8;
        }

        // Scalar tail.
        while j2 < 64 {
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
            j1 += 1;
            j2 += 2;
        }
    }
}

/// Backward real FFT pre-processing (NEON).
/// C: `rftbsub_128_neon`
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rftbsub_128_neon(a: &mut [f32; FFT_SIZE]) {
    unsafe {
        let c = &RDFT_W[32..];
        let mm_half = vdupq_n_f32(0.5);

        a[1] = -a[1];

        let mut j1 = 1usize;
        let mut j2 = 2usize;
        while j2 + 7 < 64 {
            let c_j1 = vld1q_f32(c.as_ptr().add(j1));
            let c_k1 = vld1q_f32(c.as_ptr().add(29 - j1));
            let wkrt = vsubq_f32(mm_half, c_k1);
            let wkr = reverse_order_f32x4(wkrt);
            let wki = c_j1;

            let a_j2_p = vld2q_f32(a.as_ptr().add(j2));
            let k2_0_4 = vld2q_f32(a.as_ptr().add(122 - j2));
            let a_k2_p0 = reverse_order_f32x4(k2_0_4.0);
            let a_k2_p1 = reverse_order_f32x4(k2_0_4.1);

            let xr = vsubq_f32(a_j2_p.0, a_k2_p0);
            let xi = vaddq_f32(a_j2_p.1, a_k2_p1);

            // yr = wkr*xr + wki*xi; yi = wkr*xi - wki*xr (note: signs differ from forward)
            let yr = vaddq_f32(vmulq_f32(wkr, xr), vmulq_f32(wki, xi));
            let yi = vsubq_f32(vmulq_f32(wkr, xi), vmulq_f32(wki, xr));

            let a_k2_p0n = vaddq_f32(a_k2_p0, yr);
            let a_k2_p1n = vsubq_f32(yi, a_k2_p1);

            let a_k2_p0nr = vrev64q_f32(a_k2_p0n);
            let a_k2_p1nr = vrev64q_f32(a_k2_p1n);
            let a_k2_n = vzipq_f32(a_k2_p0nr, a_k2_p1nr);

            let a_j2_p_out = float32x4x2_t(vsubq_f32(a_j2_p.0, yr), vsubq_f32(yi, a_j2_p.1));
            vst2q_f32(a.as_mut_ptr().add(j2), a_j2_p_out);

            vst1q_f32(a.as_mut_ptr().add(122 - j2), a_k2_n.1);
            vst1q_f32(a.as_mut_ptr().add(126 - j2), a_k2_n.0);

            j1 += 4;
            j2 += 8;
        }

        // Scalar tail.
        while j2 < 64 {
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
            j1 += 1;
            j2 += 2;
        }

        a[65] = -a[65];
    }
}
