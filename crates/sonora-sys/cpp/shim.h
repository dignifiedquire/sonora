// C++ shim for bridging the abstract AudioProcessing interface to cxx.
//
// Since AudioProcessing has pure virtual methods that cxx cannot handle
// directly, this shim provides concrete free functions that delegate to
// the virtual methods through an opaque handle.

#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace webrtc_shim {

// Opaque handle wrapping a scoped_refptr<AudioProcessing>
struct ApmHandle;

// Creation
std::unique_ptr<ApmHandle> create_apm();

// Configuration
void apply_config(
    ApmHandle& handle,
    bool ec_enabled,
    bool ns_enabled,
    uint8_t ns_level,
    bool agc2_enabled);

// Processing - interleaved i16
int32_t process_stream_i16(
    ApmHandle& handle,
    rust::Slice<const int16_t> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<int16_t> dest);

// Processing - deinterleaved f32 (single channel)
int32_t process_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest);

// Reverse stream - deinterleaved f32 (single channel)
int32_t process_reverse_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest);

}  // namespace webrtc_shim
