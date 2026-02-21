# Rust API review plan (upstream compatibility, quality, and performance)

This document captures a deeper review of the current Rust API surface and a concrete improvement plan.

## Key findings

1. **Typed, validated configuration paths are needed for public APIs**
   - Legacy constructors such as `StreamConfig::new(...)` allow invalid values (e.g. zero channels).
   - **Action taken:** added `CheckedStreamConfig` with `NonZeroU16` channels plus sample-rate validation (`8000..=384000`, 10ms-aligned) and explicit `StreamConfigError`.
   - Added integration points in `AudioProcessingBuilder` and `AudioProcessing::initialize_checked(...)` so callers can choose a type-safe path.

2. **Public API panic path in `high_pass_filter`**
   - `HighPassFilter::new` previously was the only constructor and panicked on unsupported sample rates.
   - **Action taken:** added `HighPassFilter::try_new` and `HighPassFilterError`; `new` remains as a compatibility shim.

3. **Validation behavior differences across modules**
   - `AudioProcessing` methods often return typed errors (`Error`) while some lower-level helper APIs still rely on debug assertions/panics.
   - **Plan:** continue adding `try_*` / checked variants and route panicking constructors through validated internals.

4. **Test coverage opportunities**
   - Existing tests are strong for signal paths, but there is room for additional API-contract tests:
     - boundary-value tests for all public config fields,
     - explicit tests for fallible constructors and error messages,
     - parity tests between Rust and FFI error mapping for newly added APIs.

5. **Performance follow-ups**
   - Several public interfaces allocate temporary vectors for de/interleaving in tests and helper flows.
   - **Plan:** add micro-benchmarks for high-frequency call paths and evaluate reusable scratch buffers in internal APIs where allocations are avoidable.

## Proposed phased roadmap

### Phase 1 (short term)
- Expand checked configuration wrappers that leverage Rust types (`NonZero*`, enums, domain-specific newtypes).
- Keep existing constructors for compatibility, but document checked alternatives as preferred.
- Add unit tests for both success and failure paths for every checked API.

### Phase 2 (medium term)
- Introduce a checked builder path (e.g. `build_checked`) to fully validate cross-field invariants before initialization.
- Expand API docs with “error semantics” and “invariants” sections.
- Add FFI parity tests ensuring Rust `Error` ↔ `WapError` mapping remains stable.

### Phase 3 (performance + quality)
- Add criterion benches around capture/render hot paths for common sample-rate/channel tuples.
- Measure allocation counts and introduce reusable scratch storage where it reduces overhead without API breakage.
- Add clippy/rustfmt checks to CI for consistent linting and style once toolchain components are available in CI/container images.
