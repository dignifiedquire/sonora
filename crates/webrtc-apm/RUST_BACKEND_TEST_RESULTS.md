# Rust Backend Test Validation Results

Test results from running the C++ test suite against the Rust audio processing
backend via the `RustAudioProcessing` C++ wrapper class.

## Summary

| Metric | Count |
|--------|-------|
| Total tests | 2432 |
| Passed | 2274 |
| Failed | 121 |
| Disabled | 185 |
| **Pass rate** | **93.5%** |

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

### 1. Format validation — channel count mismatches (54 tests)

**Tests:** `FormatValidation/ApmFormatHandlingTest.IntApi/{0..35}` (32),
`FormatValidation/ApmFormatHandlingTest.FloatApi/{6..35}` (22)

**Root cause:** The Rust backend rejects channel count combinations that C++
handles via internal downmix/upmix (e.g., 3ch input → 2ch output). The Rust
i16 path requires matching input/output channel counts; the float path requires
output channels == 1 or == input channels.

C++ returns an error but still writes a copy of the first input channel to
the output buffer. Rust returns an error without modifying the output buffer,
leaving it uninitialized from the test's perspective.

Additionally, the i16 path rejects mismatched input/output sample rates (e.g.,
16000→8000) because the Rust implementation requires native rates with matching
input/output. C++ handles this via internal resampling.

### 2. Resampling path — SNR degradation (26 tests)

**Tests:** `CommonFormats/AudioProcessingTest.Formats/{1,2,4..8,13,14,16,17,20,24,26,27,29,32,36,37,39,40,42,43,45,46,47}`

**Root cause:** When the reverse (render) stream sample rate differs from the
capture stream rate, the Rust backend produces output with low SNR (~-2.6 dB
vs the expected ≥30 dB). This occurs because the `RustAudioProcessing` wrapper
creates a throwaway output buffer in `AnalyzeReverseStream()` but the
rate-conversion interaction between capture and reverse paths isn't fully
equivalent to C++'s internal resampling pipeline.

Same-rate configurations pass with >48 dB SNR.

### 3. Input volume controller (32 tests)

**Tests:** `AudioProcessingImplTest/ApmInputVolumeControllerParametrizedTest.EnforceMinInputVolumeAtStartupWithNonZeroVolume/{0..15}` (16),
`AudioProcessingImplTest/ApmInputVolumeControllerParametrizedTest.EnforceMinInputVolumeAtStartupWithZeroVolume/{0..15}` (16)

**Root cause:** `recommended_stream_analog_level()` returns 0 instead of a
positive value. The Rust input volume controller implementation doesn't
fully enforce the minimum input volume at startup in the same way as C++.
The input volume controller was a late addition to the Rust port and has
limited functionality compared to C++ AGC1's analog controller.

### 4. Non-native i16 sample rates (1 test)

**Tests:** `ApmTest.NoProcessingWhenAllComponentsDisabledInt`

**Root cause:** The test iterates over rates including 22050 Hz and 44100 Hz.
The Rust i16 path only accepts native rates (8000, 16000, 32000, 48000) and
rejects non-native rates with `kBadSampleRateError`. C++ handles these via
internal resampling. Native rates pass fine.

### 5. Error code and pre-condition differences (4 tests)

**Tests:** `ApmTest.StreamParametersInt`, `ApmTest.StreamParametersFloat`,
`ApmTest.Channels`, `ApmTest.SampleRatesInt`

**Root cause:**
- `StreamParameters*`: C++ requires `set_stream_delay_ms()` before
  `ProcessStream()` and returns `kStreamParameterNotSetError`. Rust does not
  enforce this pre-condition.
- `Channels`: Rust returns `kBadStreamParameterWarning` (-13) for an invalid
  channel configuration where C++ returns `kBadNumberChannelsError` (-9).
- `SampleRatesInt`: Error code differences for non-native sample rates in
  the i16 path.

### 6. AGC volume recommendation (2 tests)

**Tests:** `ApmTest.QuantizedVolumeDoesNotGetStuck`,
`ApmTest.ManualVolumeChangeIsPossible`

**Root cause:** These tests exercise the analog volume recommendation loop
which depends on the input volume controller producing non-trivial
recommendations. Related to the same input volume controller limitation
described in category 3.

### 7. Statistics and format edge cases (2 tests)

**Tests:** `ApmStatistics.GetStatisticsReportsNoEchoDetectorStatsWhenDisabled`,
`ApmAnalyzeReverseStreamFormatTest.AnalyzeReverseStream`

**Root cause:**
- `GetStatisticsReportsNoEchoDetectorStatsWhenDisabled`: The test disables
  the echo detector but still expects certain stats fields. The Rust backend's
  stats population differs slightly.
- `AnalyzeReverseStream`: Reverse stream format handling edge case.

## What passes

All core audio processing tests pass, including:
- Echo cancellation (AEC3) internal tests (30+ tests)
- Noise suppression tests
- AGC2 gain controller tests (adaptive digital, fixed digital, limiter)
- High-pass filter tests
- RNN VAD tests
- Audio buffer, resampler, and DSP primitive tests
- Basic processing pipeline (same-rate mono/stereo capture and render)
- Configuration round-trips
- Runtime settings

## Modules not ported (expected failures in disabled tests)

- **AECM** (echo control mobile): Removed upstream in M146
- **AGC1** (legacy gain controller): Deprecated, uses SPL library
- **Core VAD**: Depends on SPL; modern pipeline uses AGC2's RNN VAD
