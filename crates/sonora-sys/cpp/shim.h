// C++ shim for bridging the abstract AudioProcessing interface to cxx.
//
// Since AudioProcessing has pure virtual methods that cxx cannot handle
// directly, this shim provides concrete free functions that delegate to
// the virtual methods through an opaque handle.
//
// Per-component handles (FilterBankHandle, HpfHandle, NsHandle) expose
// individual DSP components for comparison testing against the Rust port.

#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"
#include "webrtc/api/audio/audio_processing.h"
#include "webrtc/api/scoped_refptr.h"
#include "webrtc/modules/audio_processing/three_band_filter_bank.h"
#include "webrtc/modules/audio_processing/high_pass_filter.h"
#include "webrtc/modules/audio_processing/ns/noise_suppressor.h"
#include "webrtc/modules/audio_processing/audio_buffer.h"

namespace webrtc_shim {

// ── Full pipeline handle ─────────────────────────────────────────────────────

// Handle wrapping a scoped_refptr<AudioProcessing>.
// Defined here so cxx-generated code can see the complete type.
struct ApmHandle {
    webrtc::scoped_refptr<webrtc::AudioProcessing> apm;
};

// ── Per-component handles ────────────────────────────────────────────────────

struct FilterBankHandle {
    webrtc::ThreeBandFilterBank bank;
};

struct HpfHandle {
    std::unique_ptr<webrtc::HighPassFilter> hpf;
};

struct NsHandle {
    std::unique_ptr<webrtc::NoiseSuppressor> ns;
    std::unique_ptr<webrtc::AudioBuffer> buf;
};

// Creation
std::unique_ptr<ApmHandle> create_apm();

// Configuration
void apply_config(
    ApmHandle& handle,
    bool ec_enabled,
    bool ns_enabled,
    uint8_t ns_level,
    bool agc2_enabled,
    bool hpf_enabled);

// Processing - interleaved i16
int32_t process_stream_i16(
    ApmHandle& handle,
    rust::Slice<const int16_t> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<int16_t> dest);

// Processing - deinterleaved f32 (mono)
int32_t process_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest);

// Processing - deinterleaved f32 (stereo)
int32_t process_stream_f32_2ch(
    ApmHandle& handle,
    rust::Slice<const float> src_l,
    rust::Slice<const float> src_r,
    int32_t sample_rate,
    rust::Slice<float> dest_l,
    rust::Slice<float> dest_r);

// Reverse stream - deinterleaved f32 (mono)
int32_t process_reverse_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest);

// ── Per-component: ThreeBandFilterBank ────────────────────────────────────────

std::unique_ptr<FilterBankHandle> create_filter_bank();

// Analysis: in = 480 floats (fullband), out = 480 floats (3×160 bands packed).
void filter_bank_analysis(
    FilterBankHandle& handle,
    rust::Slice<const float> in,
    rust::Slice<float> out);

// Synthesis: in = 480 floats (3×160 bands packed), out = 480 floats (fullband).
void filter_bank_synthesis(
    FilterBankHandle& handle,
    rust::Slice<const float> in,
    rust::Slice<float> out);

// ── Per-component: HighPassFilter ────────────────────────────────────────────

std::unique_ptr<HpfHandle> create_hpf(
    int32_t sample_rate_hz,
    size_t num_channels);

// Process a single channel in-place. Data length = sample_rate / 100.
void hpf_process(
    HpfHandle& handle,
    rust::Slice<float> ch0);

// ── Per-component: NoiseSuppressor ───────────────────────────────────────────

std::unique_ptr<NsHandle> create_ns(
    uint8_t level,
    int32_t sample_rate_hz,
    size_t num_channels);

// Analyze band0 (160 floats). Must be called before ns_process.
void ns_analyze(
    NsHandle& handle,
    rust::Slice<const float> band0);

// Process band0 in-place (160 floats).
void ns_process(
    NsHandle& handle,
    rust::Slice<float> band0);

}  // namespace webrtc_shim
