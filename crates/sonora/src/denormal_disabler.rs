//! Scoped hardware flush-to-zero for the processing entry points.
//!
//! Mirrors WebRTC's `DenormalDisabler` (`rtc_base/denormal_disabler.{h,cc}`),
//! which `AudioProcessingImpl` places at every stream entry point. Several
//! recursive filters in the pipeline (the two-band QMF all-pass states, the
//! AEC3 decimator biquads, the reverb models) never decay to zero on silent
//! input under IEEE gradual underflow: they settle in a limit cycle a few
//! ulps above zero, in the subnormal range. On x86 every arithmetic operation
//! on a subnormal operand takes a microcode assist, which made AEC3 10-50x
//! slower while the render reference was silent (issue #34). Upstream hides
//! this with FTZ/DAZ; so does this guard.

use core::marker::PhantomData;

/// Sets hardware flush-to-zero for the current thread while alive.
///
/// - x86 / x86_64 with SSE: MXCSR.FTZ (bit 15) and MXCSR.DAZ (bit 6).
/// - aarch64 with NEON: FPCR.FZ (bit 24), which flushes inputs and outputs.
/// - Everything else is a no-op, including all 32-bit ARM targets
///   (`armv7-linux-androideabi`), `arm64ec`, wasm32, i586 and Miri. Upstream
///   also flushes on 32-bit ARM (FPSCR.FZ); this port does not yet.
///
/// Like upstream, the guard changes the register only if the bits are not
/// already all set, and restores the saved word only if it changed it, so
/// nested guards are harmless and a caller that already runs with FTZ keeps
/// it.
#[derive(Debug)]
#[must_use = "denormals are only disabled while the guard is alive"]
pub(crate) struct DenormalDisabler {
    /// Control word read by `new`; `Some` only if `new` changed the register.
    saved: Option<imp::Word>,
    /// FP control registers are per thread, so the guard is `!Send + !Sync`.
    _not_send: PhantomData<*const ()>,
}

impl DenormalDisabler {
    /// Sets the flush-to-zero bits for the current thread.
    ///
    /// # Safety
    ///
    /// Rust assumes the default floating-point environment: RFC 3514 makes
    /// any change to it undefined behavior, and the `core::arch` MXCSR
    /// documentation calls a changed DAZ bit "immediate Undefined Behavior".
    /// Until the guard drops, everything that runs on this thread sees
    /// subnormal inputs and results as zero. That includes code sonora does
    /// not control: `tracing` subscribers, the global allocator and the panic
    /// hook (which runs before unwinding drops the guard). On Linux and other
    /// platforms where a new thread inherits the floating-point environment,
    /// a thread that such code starts before the guard drops keeps
    /// flush-to-zero for its whole life; `Drop` restores only this thread. A
    /// comparison with a subnormal operand can change its result (`x > 0.0`
    /// is false for `x = f32::from_bits(1)`). LLVM may also move
    /// register-only float operations across the guard boundary, and
    /// compile-time float results can differ from run-time ones in the
    /// subnormal range. The caller must run only code that tolerates all of
    /// this. The audio pipeline has no subnormal-sensitive logic, and
    /// upstream runs the same algorithms with the same bits set.
    #[inline]
    pub(crate) unsafe fn new() -> Self {
        #[cfg(test)]
        if tests::BYPASS.get() {
            return Self {
                saved: None,
                _not_send: PhantomData,
            };
        }
        let before = imp::read();
        let saved = if imp::denormals_enabled(before) {
            // SAFETY: the caller accepts the non-default FP environment (see
            // the function contract). The written word differs from `before`
            // only in the defined FTZ/DAZ bits.
            unsafe { imp::write(imp::with_ftz(before)) };
            Some(before)
        } else {
            None
        };
        Self {
            saved,
            _not_send: PhantomData,
        }
    }

    /// Returns true if this target has a hardware flush-to-zero path.
    /// Mirrors upstream `DenormalDisabler::IsSupported()`.
    pub(crate) const fn is_supported() -> bool {
        imp::SUPPORTED
    }
}

impl Drop for DenormalDisabler {
    #[inline]
    fn drop(&mut self) {
        if let Some(word) = self.saved {
            // SAFETY: restores the exact word read in `new`, on the same
            // thread (the guard is `!Send`), which returns the caller to its
            // own environment. Runs during unwinding too.
            unsafe { imp::write(word) };
        }
    }
}

#[cfg(all(
    not(miri),
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse"
))]
mod imp {
    use core::arch::asm;

    pub(super) type Word = u32;
    pub(super) const SUPPORTED: bool = true;
    /// MXCSR.FTZ (bit 15) | MXCSR.DAZ (bit 6); upstream `kDenormalBitMask`.
    const MASK: Word = 0x8040;

    pub(super) const fn denormals_enabled(word: Word) -> bool {
        word & MASK != MASK
    }

    pub(super) const fn with_ftz(word: Word) -> Word {
        word | MASK
    }

    #[inline(always)]
    pub(super) fn read() -> Word {
        let mut csr: Word = 0;
        // SAFETY: `stmxcsr` stores MXCSR into the 4 bytes at `p`, a live
        // local. It changes no register and no flag. Never `pure`: the value
        // depends on hidden state.
        unsafe {
            asm!(
                "stmxcsr [{p}]",
                p = in(reg) &raw mut csr,
                options(nostack, preserves_flags),
            );
        }
        csr
    }

    /// # Safety
    ///
    /// `csr` must be a word read by `read`, optionally with `MASK` set, and
    /// the caller must accept the resulting FP environment.
    #[inline(always)]
    pub(super) unsafe fn write(csr: Word) {
        // SAFETY: `ldmxcsr` reads the 4 bytes at `p`, a live local. It
        // rewrites the MXCSR exception flags, so `preserves_flags` is not
        // claimed. No `nomem`/`readonly`: the block also orders memory the
        // compiler cannot prove private (not data behind a `&mut`).
        unsafe {
            asm!("ldmxcsr [{p}]", p = in(reg) &raw const csr, options(nostack));
        }
    }
}

#[cfg(all(not(miri), target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use core::arch::asm;

    pub(super) type Word = u64;
    pub(super) const SUPPORTED: bool = true;
    /// FPCR.FZ (bit 24).
    const MASK: Word = 1 << 24;

    pub(super) const fn denormals_enabled(word: Word) -> bool {
        word & MASK != MASK
    }

    pub(super) const fn with_ftz(word: Word) -> Word {
        word | MASK
    }

    #[inline(always)]
    pub(super) fn read() -> Word {
        let fpcr: Word;
        // SAFETY: reads FPCR into a general register. FPCR holds no NZCV or
        // FPSR state, so no flag changes.
        unsafe {
            asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
        }
        fpcr
    }

    /// # Safety
    ///
    /// The caller must accept the resulting FP environment.
    #[inline(always)]
    pub(super) unsafe fn write(fpcr: Word) {
        // SAFETY: writes FPCR from a general register; NZCV and FPSR are
        // untouched. No `nomem`: the block also orders memory the compiler
        // cannot prove private (not data behind a `&mut`).
        unsafe {
            asm!("msr fpcr, {}", in(reg) fpcr, options(nostack, preserves_flags));
        }
    }
}

#[cfg(not(any(
    all(
        not(miri),
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse"
    ),
    all(not(miri), target_arch = "aarch64", target_feature = "neon"),
)))]
mod imp {
    // `u32`, not `()`: a unit word trips clippy `let_unit_value`/`unit_arg`.
    pub(super) type Word = u32;
    pub(super) const SUPPORTED: bool = false;

    pub(super) const fn denormals_enabled(_word: Word) -> bool {
        false
    }

    pub(super) const fn with_ftz(word: Word) -> Word {
        word
    }

    #[inline(always)]
    pub(super) fn read() -> Word {
        0
    }

    /// # Safety
    ///
    /// Always safe; `unsafe` only to match the supported variants.
    #[inline(always)]
    pub(super) unsafe fn write(_word: Word) {}
}

// Every test whose result depends on the target asserts it against
// `is_supported()` instead of skipping unsupported targets; the other tests
// hold on every target. On a no-op target the tests therefore check that the
// guard really is a no-op, nothing shows as passed without running, and the
// target predicates exist only once (in `imp`).
#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::f32::consts::PI;
    use std::hint::black_box;
    use std::panic;

    use super::DenormalDisabler;
    use crate::config::EchoCanceller;
    use crate::{AudioProcessing, Config, StreamConfig};

    thread_local! {
        /// Test-only: makes `DenormalDisabler::new` a no-op on this thread,
        /// so the regression test can run a control without the guard.
        pub(super) static BYPASS: Cell<bool> = const { Cell::new(false) };
    }

    /// `MIN_POSITIVE / 123.125` is subnormal under IEEE and zero under FTZ.
    /// `black_box` keeps LLVM from folding it at compile time, where the FTZ
    /// bit is invisible.
    fn runtime_division_is_subnormal() -> bool {
        (black_box(f32::MIN_POSITIVE) / black_box(123.125f32)).is_subnormal()
    }

    /// `2^-149 * 2^100` is `2^-49` (normal) under IEEE; DAZ reads the
    /// subnormal input as zero.
    fn subnormal_input_reads_as_zero() -> bool {
        black_box(f32::from_bits(1)) * black_box(2.0f32.powi(100)) == 0.0
    }

    fn guard() -> DenormalDisabler {
        // SAFETY: test code; tolerates flushed subnormals.
        unsafe { DenormalDisabler::new() }
    }

    /// Every test assumes it starts in Rust's default FP environment. If a
    /// platform starts threads with FTZ set, fail here, loudly, rather than
    /// in a confusing later assertion.
    fn assert_ieee_default() {
        assert!(
            runtime_division_is_subnormal() && !subnormal_input_reads_as_zero(),
            "precondition: the test thread must start in the IEEE default FP environment"
        );
    }

    /// Upstream parity (`ZeroDenormals`): subnormal results become zero
    /// inside the guard. This is the FTZ half of the fix.
    #[test]
    fn guard_flushes_subnormal_results() {
        assert_ieee_default();
        let _g = guard();
        assert_eq!(
            !runtime_division_is_subnormal(),
            DenormalDisabler::is_supported()
        );
    }

    /// Subnormal *inputs* must read as zero too (x86 DAZ; FPCR.FZ covers
    /// inputs on aarch64). Without DAZ, x86 still takes a microcode assist for
    /// every subnormal operand the caller feeds in; the FTZ test above cannot
    /// tell 0x8000 from 0x8040.
    #[test]
    fn guard_treats_subnormal_inputs_as_zero() {
        assert_ieee_default();
        let _g = guard();
        assert_eq!(
            subnormal_input_reads_as_zero(),
            DenormalDisabler::is_supported()
        );
    }

    /// Upstream parity (`RestoreDenormalsEnabled`): a leaked FTZ bit would
    /// silently change the numerics of unrelated code on the caller's thread.
    #[test]
    fn guard_restores_caller_environment() {
        assert_ieee_default();
        {
            let _g = guard();
        }
        assert_ieee_default();
    }

    /// Upstream's nesting rule: a nested guard, or a caller that already set
    /// FTZ, must keep the outer state after the inner guard drops.
    #[test]
    fn nested_guard_keeps_outer_state() {
        assert_ieee_default();
        let _outer = guard();
        {
            let _inner = guard();
        }
        assert_eq!(
            !runtime_division_is_subnormal(),
            DenormalDisabler::is_supported()
        );
    }

    /// Upstream parity (`DoNotZeroInfNan`): FTZ must not touch Inf or NaN.
    #[test]
    fn guard_keeps_inf_and_nan() {
        assert_ieee_default();
        let _g = guard();
        assert!((black_box(f32::MAX) * black_box(2.0f32)).is_infinite());
        assert!(black_box(-1.0f32).sqrt().is_nan());
    }

    /// The C API catches panics and the caller's thread carries on, so the
    /// environment must be restored during unwinding as well.
    #[test]
    fn guard_restores_on_unwind() {
        assert_ieee_default();
        let r = panic::catch_unwind(|| {
            let _g = guard();
            panic!("unwind through the guard");
        });
        assert!(r.is_err());
        assert_ieee_default();
    }

    fn ec_48k_mono() -> AudioProcessing {
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        };
        let stream = StreamConfig::new(48_000, 1);
        AudioProcessing::builder()
            .config(config)
            .capture_config(stream)
            .render_config(stream)
            .build()
    }

    /// Calls each public processing entry point once and returns after each
    /// call so the caller can inspect the thread's FP environment.
    fn for_each_process_call(apm: &mut AudioProcessing, mut check: impl FnMut(&str)) {
        let n = 480;
        let src = vec![0.01f32; n];
        let mut dst = vec![0.0f32; n];
        let src_i16 = vec![100i16; n];
        let mut dst_i16 = vec![0i16; n];
        apm.process_render_f32(&[&src], &mut [&mut dst]).unwrap();
        check("process_render_f32");
        apm.process_capture_f32(&[&src], &mut [&mut dst]).unwrap();
        check("process_capture_f32");
        apm.process_render_i16(&src_i16, &mut dst_i16).unwrap();
        check("process_render_i16");
        apm.process_capture_i16(&src_i16, &mut dst_i16).unwrap();
        check("process_capture_i16");
    }

    /// sonora must hand the caller's thread back in the environment it
    /// found, through the real API: IEEE for a default caller, and FTZ kept
    /// for a caller (such as a DAW audio thread) that already set it.
    #[test]
    fn process_calls_leave_caller_environment_unchanged() {
        assert_ieee_default();
        let mut apm = ec_48k_mono();
        for_each_process_call(&mut apm, |call| {
            assert!(runtime_division_is_subnormal(), "{call} leaked FTZ");
        });
        let _caller_ftz = guard();
        for_each_process_call(&mut apm, |call| {
            assert_eq!(
                !runtime_division_is_subnormal(),
                DenormalDisabler::is_supported(),
                "{call} cleared the caller's FTZ"
            );
        });
    }

    /// Counts nonzero values below `f32::MIN_POSITIVE` per state field in the
    /// pretty `Debug` dump of the whole processor (every state struct derives
    /// `Debug`). Keys are the field path, e.g.
    /// `inner.render.render_audio.splitting_filter.state.analysis_state1`.
    fn subnormals_by_field(apm: &AudioProcessing) -> BTreeMap<String, usize> {
        let dump = format!("{apm:#?}");
        let mut stack: Vec<Option<&str>> = Vec::new();
        let mut counts = BTreeMap::new();
        for line in dump.lines() {
            let body = line.trim_start();
            stack.truncate((line.len() - body.len()) / 4);
            let field = body.split_once(": ").map(|(name, _)| name).filter(|name| {
                name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                    && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            });
            stack.push(field);
            let n = body
                .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '-' | '+')))
                .filter_map(|t| t.parse::<f64>().ok())
                .filter(|v| *v != 0.0 && v.abs() < f64::from(f32::MIN_POSITIVE))
                .count();
            if n > 0 {
                let key: Vec<&str> = stack.iter().flatten().copied().collect();
                *counts.entry(key.join(".")).or_insert(0) += n;
            }
        }
        counts
    }

    /// Deterministic speech-like source: LCG noise with a 4 Hz envelope.
    fn speech_like(frame: usize, n: usize, out: &mut [f32]) {
        let mut seed = (frame as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
        for (i, s) in out.iter_mut().enumerate().take(n) {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = (frame * n + i) as f32 / 48_000.0;
            let env = 0.5 + 0.5 * (2.0 * PI * 4.0 * t).sin();
            *s = 0.2 * env * noise;
        }
    }

    /// Frames (10 ms) of 1 s speech-like render and echo.
    const SPEECH_END: usize = 100;
    /// Then exact-zero render (the reported case) with capture noise at about
    /// -75 dBFS RMS and -66 dBFS peak, until checkpoint A.
    const CHECKPOINT_A: usize = 300;
    /// Then exact-zero render and capture, until checkpoint B. The latest
    /// onset in the control (`y2_smoothed`, about frame 650) leaves a margin
    /// of 150 frames; keep one if either moves.
    const CHECKPOINT_B: usize = 800;

    /// Runs the issue #34 scenario through the f32 or i16 API and returns the
    /// subnormal state per field at checkpoints A and B.
    fn silent_render_scenario(use_i16: bool) -> [BTreeMap<String, usize>; 2] {
        let mut apm = ec_48k_mono();
        let n = 480;
        let mut render = vec![0.0f32; n];
        let mut capture = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let mut render_out = vec![0.0f32; n];
        let mut out_i16 = vec![0i16; n];
        let to_i16 = |x: &[f32]| -> Vec<i16> { x.iter().map(|v| (v * 32767.0) as i16).collect() };
        let mut checkpoint_a = BTreeMap::new();
        for frame in 0..CHECKPOINT_B {
            if frame < SPEECH_END {
                speech_like(frame, n, &mut render);
                for (c, r) in capture.iter_mut().zip(&render) {
                    *c = 0.5 * r;
                }
            } else if frame < CHECKPOINT_A {
                render.fill(0.0);
                speech_like(frame, n, &mut capture);
                for c in &mut capture {
                    *c *= 0.005;
                }
            } else {
                render.fill(0.0);
                capture.fill(0.0);
            }
            if use_i16 {
                apm.process_render_i16(&to_i16(&render), &mut out_i16)
                    .unwrap();
                apm.process_capture_i16(&to_i16(&capture), &mut out_i16)
                    .unwrap();
            } else {
                apm.process_render_f32(&[&render], &mut [&mut render_out])
                    .unwrap();
                apm.process_capture_f32(&[&capture], &mut [&mut out])
                    .unwrap();
            }
            if frame + 1 == CHECKPOINT_A {
                checkpoint_a = subnormals_by_field(&apm);
            }
        }
        [checkpoint_a, subnormals_by_field(&apm)]
    }

    /// Recursive states that settle in the subnormal range without the guard,
    /// with the checkpoint by which each must hold a subnormal in the control
    /// run. The control proves the scenario still reaches every known source;
    /// without it, a scenario that stopped producing subnormals would pass.
    const SOURCES: &[(usize, &str)] = &[
        // Render side, reached in the reported case (onset about frame 110-140).
        (
            CHECKPOINT_A,
            "render_audio.splitting_filter.state.analysis_state1",
        ),
        (
            CHECKPOINT_A,
            "render_audio.splitting_filter.state.synthesis_state1",
        ),
        (
            CHECKPOINT_A,
            "render_decimator.anti_aliasing_filter.biquads.y",
        ),
        (
            CHECKPOINT_A,
            "render_decimator.noise_reduction_filter.biquads.y",
        ),
        // Slower render-driven smoothers (onset about frame 330 and 500).
        (CHECKPOINT_B, "aec_state.avg_render_reverb.reverb"),
        (CHECKPOINT_B, "residual_echo_estimator.echo_reverb.reverb"),
        (CHECKPOINT_B, "low_render_detector.average_power"),
        // Capture side, once capture is exact zero too (onset about 310-650).
        (
            CHECKPOINT_B,
            "capture_audio.splitting_filter.state.analysis_state1",
        ),
        (
            CHECKPOINT_B,
            "capture_audio.splitting_filter.state.synthesis_state1",
        ),
        (CHECKPOINT_B, "high_pass_filter.filters.biquads.y"),
        (
            CHECKPOINT_B,
            "capture_decimator.anti_aliasing_filter.biquads.y",
        ),
        (CHECKPOINT_B, "echo_remover.cng.y2_smoothed"),
    ];

    /// Issue #34. After the render reference goes silent, the pipeline's
    /// recursive filters (QMF all-pass, decimator and high-pass biquads,
    /// reverb models, smoothers) settle a few ulps above zero under IEEE
    /// gradual underflow and never reach zero, so every later frame computes
    /// on subnormals: a microcode assist per operation on x86. With the guard
    /// in every `process_*` entry point, no subnormal may survive in any
    /// state. Deterministic, no timing, so it runs on real x86_64 in CI.
    #[test]
    fn silent_render_leaves_no_subnormal_state() {
        assert_ieee_default();
        BYPASS.set(true);
        let control = silent_render_scenario(false);
        BYPASS.set(false);
        for &(checkpoint, source) in SOURCES {
            let at = &control[usize::from(checkpoint == CHECKPOINT_B)];
            assert!(
                at.iter().any(|(k, &v)| k.ends_with(source) && v > 0),
                "control: scenario no longer drives {source} subnormal by frame {checkpoint}"
            );
        }
        for use_i16 in [false, true] {
            for (at, checkpoint) in silent_render_scenario(use_i16)
                .iter()
                .zip([CHECKPOINT_A, CHECKPOINT_B])
            {
                let total: usize = at.values().sum();
                assert_eq!(
                    total == 0,
                    DenormalDisabler::is_supported(),
                    "{total} subnormal state values at frame {checkpoint} (i16 API: {use_i16}): {at:?}"
                );
            }
        }
    }
}
