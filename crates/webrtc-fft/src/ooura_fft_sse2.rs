//! SSE2-accelerated inner functions for the 128-point Ooura FFT.
//!
//! C source: `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc`
//! Tables from `ooura_fft_tables_neon_sse2.h`.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::ooura_fft::{FFT_SIZE, RDFT_W};

// ── SIMD twiddle tables ──────────────────────────────────────────────────────
// These are duplicated/rearranged versions of the scalar tables, laid out for
// 4-wide SIMD loads. Each pair of values is repeated so a single `_mm_load_ps`
// gives [v0, v0, v1, v1].

#[repr(align(16))]
struct Aligned16([f32; 4]);

const K_SWAP_SIGN: Aligned16 = Aligned16([-1.0, 1.0, -1.0, 1.0]);

#[repr(align(16))]
struct AlignedTable32([f32; 32]);

const RDFT_WK1R: AlignedTable32 = AlignedTable32([
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
]);

const RDFT_WK2R: AlignedTable32 = AlignedTable32([
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
]);

const RDFT_WK3R: AlignedTable32 = AlignedTable32([
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
]);

const RDFT_WK1I: AlignedTable32 = AlignedTable32([
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
]);

const RDFT_WK2I: AlignedTable32 = AlignedTable32([
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
]);

const RDFT_WK3I: AlignedTable32 = AlignedTable32([
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
]);

#[repr(align(16))]
struct Aligned16x4([f32; 4]);

const CFTMDL_WK1R: Aligned16x4 =
    Aligned16x4([0.707_106_769, 0.707_106_769, 0.707_106_769, -0.707_106_769]);

// ── SSE2 inner functions ─────────────────────────────────────────────────────

/// First-stage complex FFT butterfly (SSE2).
/// C: `cft1st_128_SSE2`
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn cft1st_128_sse2(a: &mut [f32; FFT_SIZE]) {
    let mm_swap_sign = _mm_load_ps(K_SWAP_SIGN.0.as_ptr());

    let mut k2 = 0usize;
    let mut j = 0usize;
    while j < 128 {
        let mut a00v = _mm_loadu_ps(a.as_ptr().add(j));
        let mut a04v = _mm_loadu_ps(a.as_ptr().add(j + 4));
        let mut a08v = _mm_loadu_ps(a.as_ptr().add(j + 8));
        let mut a12v = _mm_loadu_ps(a.as_ptr().add(j + 12));

        let a01v = _mm_shuffle_ps(a00v, a08v, 0b01_00_01_00); // _MM_SHUFFLE(1,0,1,0)
        let a23v = _mm_shuffle_ps(a00v, a08v, 0b11_10_11_10); // _MM_SHUFFLE(3,2,3,2)
        let a45v = _mm_shuffle_ps(a04v, a12v, 0b01_00_01_00);
        let a67v = _mm_shuffle_ps(a04v, a12v, 0b11_10_11_10);

        let wk1rv = _mm_load_ps(RDFT_WK1R.0.as_ptr().add(k2));
        let wk1iv = _mm_load_ps(RDFT_WK1I.0.as_ptr().add(k2));
        let wk2rv = _mm_load_ps(RDFT_WK2R.0.as_ptr().add(k2));
        let wk2iv = _mm_load_ps(RDFT_WK2I.0.as_ptr().add(k2));
        let wk3rv = _mm_load_ps(RDFT_WK3R.0.as_ptr().add(k2));
        let wk3iv = _mm_load_ps(RDFT_WK3I.0.as_ptr().add(k2));

        let mut x0v = _mm_add_ps(a01v, a23v);
        let x1v = _mm_sub_ps(a01v, a23v);
        let x2v = _mm_add_ps(a45v, a67v);
        let x3v = _mm_sub_ps(a45v, a67v);

        let a01v_new = _mm_add_ps(x0v, x2v);
        x0v = _mm_sub_ps(x0v, x2v);
        let mut x0w = _mm_shuffle_ps(x0v, x0v, 0b10_11_00_01); // _MM_SHUFFLE(2,3,0,1)

        // a45v = wk2r*x0v + wk2i*x0w
        let a45v_new = _mm_add_ps(_mm_mul_ps(wk2rv, x0v), _mm_mul_ps(wk2iv, x0w));

        // x3 swap + sign
        let x3w = _mm_shuffle_ps(x3v, x3v, 0b10_11_00_01);
        let x3s = _mm_mul_ps(mm_swap_sign, x3w);

        x0v = _mm_add_ps(x1v, x3s);
        x0w = _mm_shuffle_ps(x0v, x0v, 0b10_11_00_01);
        let a23v_new = _mm_add_ps(_mm_mul_ps(wk1rv, x0v), _mm_mul_ps(wk1iv, x0w));

        x0v = _mm_sub_ps(x1v, x3s);
        x0w = _mm_shuffle_ps(x0v, x0v, 0b10_11_00_01);
        let a67v_new = _mm_add_ps(_mm_mul_ps(wk3rv, x0v), _mm_mul_ps(wk3iv, x0w));

        a00v = _mm_shuffle_ps(a01v_new, a23v_new, 0b01_00_01_00);
        a04v = _mm_shuffle_ps(a45v_new, a67v_new, 0b01_00_01_00);
        a08v = _mm_shuffle_ps(a01v_new, a23v_new, 0b11_10_11_10);
        a12v = _mm_shuffle_ps(a45v_new, a67v_new, 0b11_10_11_10);

        _mm_storeu_ps(a.as_mut_ptr().add(j), a00v);
        _mm_storeu_ps(a.as_mut_ptr().add(j + 4), a04v);
        _mm_storeu_ps(a.as_mut_ptr().add(j + 8), a08v);
        _mm_storeu_ps(a.as_mut_ptr().add(j + 12), a12v);

        j += 16;
        k2 += 4;
    }
}

/// Modular complex FFT butterfly stage (SSE2).
/// C: `cftmdl_128_SSE2`
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn cftmdl_128_sse2(a: &mut [f32; FFT_SIZE]) {
    let l = 8usize;
    let mm_swap_sign = _mm_load_ps(K_SWAP_SIGN.0.as_ptr());

    let mut wk1rv = _mm_load_ps(CFTMDL_WK1R.0.as_ptr());

    // First block (j0 = 0..8 step 2)
    for j0 in (0..l).step_by(2) {
        let a_00 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0) as *const __m128i));
        let a_08 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 8) as *const __m128i));
        let a_32 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 32) as *const __m128i));
        let a_40 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 40) as *const __m128i));

        let a_00_32 = _mm_shuffle_ps(a_00, a_32, 0b01_00_01_00);
        let a_08_40 = _mm_shuffle_ps(a_08, a_40, 0b01_00_01_00);
        let x0r0_0i0_0r1_x0i1 = _mm_add_ps(a_00_32, a_08_40);
        let x1r0_1i0_1r1_x1i1 = _mm_sub_ps(a_00_32, a_08_40);

        let a_16 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 16) as *const __m128i));
        let a_24 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 24) as *const __m128i));
        let a_48 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 48) as *const __m128i));
        let a_56 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 56) as *const __m128i));

        let a_16_48 = _mm_shuffle_ps(a_16, a_48, 0b01_00_01_00);
        let a_24_56 = _mm_shuffle_ps(a_24, a_56, 0b01_00_01_00);
        let x2r0_2i0_2r1_x2i1 = _mm_add_ps(a_16_48, a_24_56);
        let x3r0_3i0_3r1_x3i1 = _mm_sub_ps(a_16_48, a_24_56);

        let xx0 = _mm_add_ps(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
        let xx1 = _mm_sub_ps(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);

        let x3_swapped = _mm_castsi128_ps(_mm_shuffle_epi32(
            _mm_castps_si128(x3r0_3i0_3r1_x3i1),
            0b10_11_00_01,
        ));
        let x3_signed = _mm_mul_ps(mm_swap_sign, x3_swapped);
        let x1_x3_add = _mm_add_ps(x1r0_1i0_1r1_x1i1, x3_signed);
        let x1_x3_sub = _mm_sub_ps(x1r0_1i0_1r1_x1i1, x3_signed);

        let yy0 = _mm_shuffle_ps(x1_x3_add, x1_x3_sub, 0b10_10_10_10); // _MM_SHUFFLE(2,2,2,2)
        let yy1 = _mm_shuffle_ps(x1_x3_add, x1_x3_sub, 0b11_11_11_11); // _MM_SHUFFLE(3,3,3,3)
        let yy2 = _mm_mul_ps(mm_swap_sign, yy1);
        let yy3 = _mm_add_ps(yy0, yy2);
        let yy4 = _mm_mul_ps(wk1rv, yy3);

        _mm_storel_epi64(
            a.as_mut_ptr().add(j0) as *mut __m128i,
            _mm_castps_si128(xx0),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 32) as *mut __m128i,
            _mm_shuffle_epi32(_mm_castps_si128(xx0), 0b11_10_11_10),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 16) as *mut __m128i,
            _mm_castps_si128(xx1),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 48) as *mut __m128i,
            _mm_shuffle_epi32(_mm_castps_si128(xx1), 0b10_11_10_11),
        );
        a[j0 + 48] = -a[j0 + 48];

        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 8) as *mut __m128i,
            _mm_castps_si128(x1_x3_add),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 24) as *mut __m128i,
            _mm_castps_si128(x1_x3_sub),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 40) as *mut __m128i,
            _mm_castps_si128(yy4),
        );
        _mm_storel_epi64(
            a.as_mut_ptr().add(j0 + 56) as *mut __m128i,
            _mm_shuffle_epi32(_mm_castps_si128(yy4), 0b10_11_10_11),
        );
    }

    // Second block (k=64)
    {
        let k = 64usize;
        let k1 = 2usize;
        let k2 = 2 * k1;
        let wk2rv = _mm_load_ps(RDFT_WK2R.0.as_ptr().add(k2));
        let wk2iv = _mm_load_ps(RDFT_WK2I.0.as_ptr().add(k2));
        let wk1iv = _mm_load_ps(RDFT_WK1I.0.as_ptr().add(k2));
        let wk3rv = _mm_load_ps(RDFT_WK3R.0.as_ptr().add(k2));
        let wk3iv = _mm_load_ps(RDFT_WK3I.0.as_ptr().add(k2));
        wk1rv = _mm_load_ps(RDFT_WK1R.0.as_ptr().add(k2));

        for j0 in (k..l + k).step_by(2) {
            let a_00 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0) as *const __m128i));
            let a_08 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 8) as *const __m128i));
            let a_32 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 32) as *const __m128i));
            let a_40 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 40) as *const __m128i));

            let a_00_32 = _mm_shuffle_ps(a_00, a_32, 0b01_00_01_00);
            let a_08_40 = _mm_shuffle_ps(a_08, a_40, 0b01_00_01_00);
            let x0r0_0i0_0r1_x0i1 = _mm_add_ps(a_00_32, a_08_40);
            let x1r0_1i0_1r1_x1i1 = _mm_sub_ps(a_00_32, a_08_40);

            let a_16 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 16) as *const __m128i));
            let a_24 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 24) as *const __m128i));
            let a_48 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 48) as *const __m128i));
            let a_56 = _mm_castsi128_ps(_mm_loadl_epi64(a.as_ptr().add(j0 + 56) as *const __m128i));

            let a_16_48 = _mm_shuffle_ps(a_16, a_48, 0b01_00_01_00);
            let a_24_56 = _mm_shuffle_ps(a_24, a_56, 0b01_00_01_00);
            let x2r0_2i0_2r1_x2i1 = _mm_add_ps(a_16_48, a_24_56);
            let x3r0_3i0_3r1_x3i1 = _mm_sub_ps(a_16_48, a_24_56);

            let xx = _mm_add_ps(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
            let xx1 = _mm_sub_ps(x0r0_0i0_0r1_x0i1, x2r0_2i0_2r1_x2i1);
            let xx2 = _mm_mul_ps(xx1, wk2rv);
            let xx3 = _mm_mul_ps(
                wk2iv,
                _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xx1), 0b10_11_00_01)),
            );
            let xx4 = _mm_add_ps(xx2, xx3);

            let x3_swapped = _mm_castsi128_ps(_mm_shuffle_epi32(
                _mm_castps_si128(x3r0_3i0_3r1_x3i1),
                0b10_11_00_01,
            ));
            let x3_signed = _mm_mul_ps(mm_swap_sign, x3_swapped);
            let x1_x3_add = _mm_add_ps(x1r0_1i0_1r1_x1i1, x3_signed);
            let x1_x3_sub = _mm_sub_ps(x1r0_1i0_1r1_x1i1, x3_signed);

            let xx10 = _mm_mul_ps(x1_x3_add, wk1rv);
            let xx11 = _mm_mul_ps(
                wk1iv,
                _mm_castsi128_ps(_mm_shuffle_epi32(
                    _mm_castps_si128(x1_x3_add),
                    0b10_11_00_01,
                )),
            );
            let xx12 = _mm_add_ps(xx10, xx11);

            let xx20 = _mm_mul_ps(x1_x3_sub, wk3rv);
            let xx21 = _mm_mul_ps(
                wk3iv,
                _mm_castsi128_ps(_mm_shuffle_epi32(
                    _mm_castps_si128(x1_x3_sub),
                    0b10_11_00_01,
                )),
            );
            let xx22 = _mm_add_ps(xx20, xx21);

            _mm_storel_epi64(a.as_mut_ptr().add(j0) as *mut __m128i, _mm_castps_si128(xx));
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 32) as *mut __m128i,
                _mm_shuffle_epi32(_mm_castps_si128(xx), 0b11_10_11_10),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 16) as *mut __m128i,
                _mm_castps_si128(xx4),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 48) as *mut __m128i,
                _mm_shuffle_epi32(_mm_castps_si128(xx4), 0b11_10_11_10),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 8) as *mut __m128i,
                _mm_castps_si128(xx12),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 40) as *mut __m128i,
                _mm_shuffle_epi32(_mm_castps_si128(xx12), 0b11_10_11_10),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 24) as *mut __m128i,
                _mm_castps_si128(xx22),
            );
            _mm_storel_epi64(
                a.as_mut_ptr().add(j0 + 56) as *mut __m128i,
                _mm_shuffle_epi32(_mm_castps_si128(xx22), 0b11_10_11_10),
            );
        }
    }
}

/// Forward real FFT post-processing (SSE2).
/// C: `rftfsub_128_SSE2`
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn rftfsub_128_sse2(a: &mut [f32; FFT_SIZE]) {
    let c = &RDFT_W[32..];
    let mm_half = _mm_set1_ps(0.5);

    // Vectorized loop (four at a time).
    let mut j1 = 1usize;
    let mut j2 = 2usize;
    while j2 + 7 < 64 {
        // Load twiddle factors.
        let c_j1 = _mm_loadu_ps(c.as_ptr().add(j1));
        let c_k1 = _mm_loadu_ps(c.as_ptr().add(29 - j1));
        let wkrt = _mm_sub_ps(mm_half, c_k1);
        let wkr = _mm_shuffle_ps(wkrt, wkrt, 0b00_01_10_11); // reverse
        let wki = c_j1;

        // Load and deinterleave a[j2..] and a[k2..].
        let a_j2_0 = _mm_loadu_ps(a.as_ptr().add(j2));
        let a_j2_4 = _mm_loadu_ps(a.as_ptr().add(j2 + 4));
        let a_k2_0 = _mm_loadu_ps(a.as_ptr().add(122 - j2));
        let a_k2_4 = _mm_loadu_ps(a.as_ptr().add(126 - j2));

        let a_j2_p0 = _mm_shuffle_ps(a_j2_0, a_j2_4, 0b10_00_10_00); // evens
        let a_j2_p1 = _mm_shuffle_ps(a_j2_0, a_j2_4, 0b11_01_11_01); // odds
        let a_k2_p0 = _mm_shuffle_ps(a_k2_4, a_k2_0, 0b00_10_00_10); // reversed evens
        let a_k2_p1 = _mm_shuffle_ps(a_k2_4, a_k2_0, 0b01_11_01_11); // reversed odds

        let xr = _mm_sub_ps(a_j2_p0, a_k2_p0);
        let xi = _mm_add_ps(a_j2_p1, a_k2_p1);

        // yr = wkr*xr - wki*xi; yi = wkr*xi + wki*xr
        let yr = _mm_sub_ps(_mm_mul_ps(wkr, xr), _mm_mul_ps(wki, xi));
        let yi = _mm_add_ps(_mm_mul_ps(wkr, xi), _mm_mul_ps(wki, xr));

        // Update.
        let a_j2_p0n = _mm_sub_ps(a_j2_p0, yr);
        let a_j2_p1n = _mm_sub_ps(a_j2_p1, yi);
        let a_k2_p0n = _mm_add_ps(a_k2_p0, yr);
        let a_k2_p1n = _mm_sub_ps(a_k2_p1, yi);

        // Re-interleave and store.
        let a_j2_0n = _mm_unpacklo_ps(a_j2_p0n, a_j2_p1n);
        let a_j2_4n = _mm_unpackhi_ps(a_j2_p0n, a_j2_p1n);
        let a_k2_0nt = _mm_unpackhi_ps(a_k2_p0n, a_k2_p1n);
        let a_k2_4nt = _mm_unpacklo_ps(a_k2_p0n, a_k2_p1n);
        let a_k2_0n = _mm_shuffle_ps(a_k2_0nt, a_k2_0nt, 0b01_00_11_10);
        let a_k2_4n = _mm_shuffle_ps(a_k2_4nt, a_k2_4nt, 0b01_00_11_10);

        _mm_storeu_ps(a.as_mut_ptr().add(j2), a_j2_0n);
        _mm_storeu_ps(a.as_mut_ptr().add(j2 + 4), a_j2_4n);
        _mm_storeu_ps(a.as_mut_ptr().add(122 - j2), a_k2_0n);
        _mm_storeu_ps(a.as_mut_ptr().add(126 - j2), a_k2_4n);

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

/// Backward real FFT pre-processing (SSE2).
/// C: `rftbsub_128_SSE2`
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn rftbsub_128_sse2(a: &mut [f32; FFT_SIZE]) {
    let c = &RDFT_W[32..];
    let mm_half = _mm_set1_ps(0.5);

    a[1] = -a[1];

    // Vectorized loop (four at a time).
    let mut j1 = 1usize;
    let mut j2 = 2usize;
    while j2 + 7 < 64 {
        // Load twiddle factors.
        let c_j1 = _mm_loadu_ps(c.as_ptr().add(j1));
        let c_k1 = _mm_loadu_ps(c.as_ptr().add(29 - j1));
        let wkrt = _mm_sub_ps(mm_half, c_k1);
        let wkr = _mm_shuffle_ps(wkrt, wkrt, 0b00_01_10_11); // reverse
        let wki = c_j1;

        // Load and deinterleave.
        let a_j2_0 = _mm_loadu_ps(a.as_ptr().add(j2));
        let a_j2_4 = _mm_loadu_ps(a.as_ptr().add(j2 + 4));
        let a_k2_0 = _mm_loadu_ps(a.as_ptr().add(122 - j2));
        let a_k2_4 = _mm_loadu_ps(a.as_ptr().add(126 - j2));

        let a_j2_p0 = _mm_shuffle_ps(a_j2_0, a_j2_4, 0b10_00_10_00);
        let a_j2_p1 = _mm_shuffle_ps(a_j2_0, a_j2_4, 0b11_01_11_01);
        let a_k2_p0 = _mm_shuffle_ps(a_k2_4, a_k2_0, 0b00_10_00_10);
        let a_k2_p1 = _mm_shuffle_ps(a_k2_4, a_k2_0, 0b01_11_01_11);

        let xr = _mm_sub_ps(a_j2_p0, a_k2_p0);
        let xi = _mm_add_ps(a_j2_p1, a_k2_p1);

        // yr = wkr*xr + wki*xi; yi = wkr*xi - wki*xr (note: signs differ from forward)
        let yr = _mm_add_ps(_mm_mul_ps(wkr, xr), _mm_mul_ps(wki, xi));
        let yi = _mm_sub_ps(_mm_mul_ps(wkr, xi), _mm_mul_ps(wki, xr));

        // Update (note: a[j2+1] = yi - a[j2+1], a[k2+1] = yi - a[k2+1])
        let a_j2_p0n = _mm_sub_ps(a_j2_p0, yr);
        let a_j2_p1n = _mm_sub_ps(yi, a_j2_p1);
        let a_k2_p0n = _mm_add_ps(a_k2_p0, yr);
        let a_k2_p1n = _mm_sub_ps(yi, a_k2_p1);

        // Re-interleave and store.
        let a_j2_0n = _mm_unpacklo_ps(a_j2_p0n, a_j2_p1n);
        let a_j2_4n = _mm_unpackhi_ps(a_j2_p0n, a_j2_p1n);
        let a_k2_0nt = _mm_unpackhi_ps(a_k2_p0n, a_k2_p1n);
        let a_k2_4nt = _mm_unpacklo_ps(a_k2_p0n, a_k2_p1n);
        let a_k2_0n = _mm_shuffle_ps(a_k2_0nt, a_k2_0nt, 0b01_00_11_10);
        let a_k2_4n = _mm_shuffle_ps(a_k2_4nt, a_k2_4nt, 0b01_00_11_10);

        _mm_storeu_ps(a.as_mut_ptr().add(j2), a_j2_0n);
        _mm_storeu_ps(a.as_mut_ptr().add(j2 + 4), a_j2_4n);
        _mm_storeu_ps(a.as_mut_ptr().add(122 - j2), a_k2_0n);
        _mm_storeu_ps(a.as_mut_ptr().add(126 - j2), a_k2_4n);

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
