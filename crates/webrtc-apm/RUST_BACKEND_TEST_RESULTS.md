# Rust Backend Test Validation Results

Test results from running the C++ test suite against the Rust audio processing
backend via the `RustAudioProcessing` C++ wrapper class.

## Summary

| Metric | Count |
|--------|-------|
| Total tests | 2432 |
| Passed | 2359 |
| Failed | 36 |
| Disabled | 185 |
| **Pass rate** | **97.0%** |

The Rust backend is wired in via `BuiltinAudioProcessingBuilder::Build()` when
the `rust-backend` Meson option is enabled. Tests that inject custom components
(mock echo controllers, echo detectors, etc.) fall through to the C++
`AudioProcessingImpl` — only tests using the default builder path exercise the
Rust backend.

## Build and run

```bash
meson setup builddir-rust -Dtests=enabled -Drust-backend=true
ninja -C builddir-rust
meson test -C builddir-rust -v
```

## Failure Breakdown

### 1. AGC1 input volume controller (32 tests) — expected, not ported

**Tests:** `AudioProcessingImplTest/ApmInputVolumeControllerParametrizedTest.EnforceMinInputVolumeAtStartupWithZeroVolume/{0..15}` (16),
`AudioProcessingImplTest/ApmInputVolumeControllerParametrizedTest.EnforceMinInputVolumeAtStartupWithNonZeroVolume/{0..15}` (16)

**Root cause:** These tests enable AGC1 (`gain_controller1.enabled = true`,
`mode = kAdaptiveAnalog`) which is not ported to Rust. The AGC1 analog gain
controller enforces minimum input volume at startup — a behavior that cannot
be replicated without the full AGC1 implementation. AGC1 is deprecated upstream
(tracked by bugs.webrtc.org/7494).

### 2. AGC1 stream parameter preconditions (2 tests) — expected, not ported

**Tests:** `ApmTest.StreamParametersInt`, `ApmTest.StreamParametersFloat`

**Root cause:** These tests enable AGC1 and expect `kStreamParameterNotSetError`
(-11) when `set_stream_analog_level()` hasn't been called before
`ProcessStream()`. Since AGC1 is not ported, the Rust backend does not enforce
this precondition.

### 3. AGC1 volume recommendation (2 tests) — expected, not ported

**Tests:** `ApmTest.QuantizedVolumeDoesNotGetStuck`,
`ApmTest.ManualVolumeChangeIsPossible`

**Root cause:** These tests exercise AGC1's analog volume recommendation loop
(`kAdaptiveAnalog` mode). The Rust backend passes through the applied volume
via the fallback chain `recommended.or(applied).unwrap_or(255)`, but AGC1's
volume adjustment logic is not present.

## What passes

All core audio processing tests pass, including:
- Echo cancellation (AEC3) internal tests (30+ tests)
- All 53 CommonFormats tests (including stereo 48k→32k resampling paths)
- Noise suppression tests
- AGC2 gain controller tests (adaptive digital, fixed digital, limiter)
- High-pass filter tests
- RNN VAD tests
- Audio buffer, resampler, and DSP primitive tests
- Format validation with correct error codes and output buffer filling
- Reverse stream processing with rate conversion
- i16 processing at all rates [8000, 384000]
- Configuration round-trips
- Runtime settings
- Statistics reporting (echo detector disabled/enabled)
- Channel count validation (0 channels → correct error code)

## Modules not ported (expected failures)

- **AGC1** (legacy gain controller): Deprecated upstream, uses SPL library.
  All 36 AGC1-dependent test failures are expected.
- **AECM** (echo control mobile): Removed upstream in M146
- **Core VAD**: Depends on SPL; modern pipeline uses AGC2's RNN VAD
