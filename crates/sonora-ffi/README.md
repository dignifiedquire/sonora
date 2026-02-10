# sonora-ffi

C-compatible FFI layer for the [sonora](https://crates.io/crates/sonora) audio
processing library. Provides a stable C API for integrating echo cancellation,
noise suppression, and automatic gain control into C/C++ projects.

## Building

```sh
cargo build --release -p sonora-ffi
```

This produces a static library (e.g. `target/release/libsonora_ffi.a`) and
auto-generates the C header at `crates/sonora-ffi/include/wap_audio_processing.h`.

## Usage from C

```c
#include "wap_audio_processing.h"

WapConfig config = wap_config_default();
config.noise_suppression_enabled = true;
WapAudioProcessing *apm = wap_create_with_config(config);

WapStreamConfig stream = { .sample_rate_hz = 48000, .num_channels = 1 };
wap_initialize(apm, stream, stream, stream, stream);

// Process 10 ms frames (48 kHz = 480 samples)
float buf[480];
float *channels[] = { buf };
wap_process_stream_f32(apm, (const float *const *)channels,
                       stream, stream, channels);

wap_destroy(apm);
```

## API Overview

All functions are prefixed with `wap_` and types with `Wap`. The API is **not
thread-safe** -- all calls on the same handle must be serialized by the caller.

| Category | Functions |
|----------|-----------|
| Lifecycle | `wap_create`, `wap_create_with_config`, `wap_destroy` |
| Config | `wap_config_default`, `wap_apply_config`, `wap_get_config` |
| Init | `wap_initialize` |
| Processing | `wap_process_stream_f32`, `wap_process_stream_i16`, `wap_process_reverse_stream_f32`, `wap_process_reverse_stream_i16` |
| Runtime | `wap_set_capture_pre_gain`, `wap_set_capture_post_gain`, `wap_set_capture_fixed_post_gain`, `wap_set_playout_volume`, `wap_set_playout_audio_device`, `wap_set_capture_output_used` |
| AGC | `wap_set_stream_analog_level`, `wap_recommended_stream_analog_level` |
| Delay | `wap_set_stream_delay_ms`, `wap_stream_delay_ms` |
| Query | `wap_get_statistics`, `wap_version` |

## License

BSD-3-Clause -- see [LICENSE](../../LICENSE) in the repository root.
