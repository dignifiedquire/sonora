// C++ shim implementation.
//
// Wraps the abstract AudioProcessing interface for cxx interop.
// Also provides per-component shims for comparison testing.

#include "sonora-sys/cpp/shim.h"

#include <vector>

#include "webrtc/api/array_view.h"
#include "webrtc/api/audio/builtin_audio_processing_builder.h"
#include "webrtc/api/environment/environment_factory.h"
#include "webrtc/api/field_trials.h"
#include "webrtc/modules/audio_processing/ns/ns_config.h"

namespace webrtc_shim {

std::unique_ptr<ApmHandle> create_apm() {
    webrtc::AudioProcessing::Config config;
    webrtc::Environment env = webrtc::CreateEnvironment();
    auto apm = webrtc::BuiltinAudioProcessingBuilder(config).Build(env);
    if (!apm) {
        return nullptr;
    }
    auto handle = std::make_unique<ApmHandle>();
    handle->apm = std::move(apm);
    return handle;
}

std::unique_ptr<ApmHandle> create_apm_with_field_trials(rust::Str field_trials) {
    std::string ft_str(field_trials.data(), field_trials.size());
    auto ft = std::make_unique<webrtc::FieldTrials>(ft_str);
    webrtc::Environment env = webrtc::CreateEnvironment(std::move(ft));

    webrtc::AudioProcessing::Config config;
    auto apm = webrtc::BuiltinAudioProcessingBuilder(config).Build(env);
    if (!apm) {
        return nullptr;
    }
    auto handle = std::make_unique<ApmHandle>();
    handle->apm = std::move(apm);
    return handle;
}

void apply_config(
    ApmHandle& handle,
    bool ec_enabled,
    bool ns_enabled,
    uint8_t ns_level,
    bool agc2_enabled,
    bool hpf_enabled) {
    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = ec_enabled;
    config.noise_suppression.enabled = ns_enabled;
    config.noise_suppression.level =
        static_cast<webrtc::AudioProcessing::Config::NoiseSuppression::Level>(ns_level);
    config.gain_controller2.enabled = agc2_enabled;
    config.high_pass_filter.enabled = hpf_enabled;
    handle.apm->ApplyConfig(config);
}

int32_t process_stream_i16(
    ApmHandle& handle,
    rust::Slice<const int16_t> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<int16_t> dest) {

    webrtc::StreamConfig input_config(input_sample_rate, input_channels);
    webrtc::StreamConfig output_config(output_sample_rate, output_channels);

    return handle.apm->ProcessStream(
        src.data(), input_config, output_config,
        const_cast<int16_t*>(dest.data()));
}

int32_t process_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest) {

    webrtc::StreamConfig input_config(input_sample_rate, input_channels);
    webrtc::StreamConfig output_config(output_sample_rate, output_channels);

    const float* src_ptrs[1] = { src.data() };
    float* dest_ptrs[1] = { dest.data() };

    return handle.apm->ProcessStream(
        src_ptrs, input_config, output_config, dest_ptrs);
}

int32_t process_stream_f32_2ch(
    ApmHandle& handle,
    rust::Slice<const float> src_l,
    rust::Slice<const float> src_r,
    int32_t sample_rate,
    rust::Slice<float> dest_l,
    rust::Slice<float> dest_r) {

    webrtc::StreamConfig config(sample_rate, 2);

    const float* src_ptrs[2] = { src_l.data(), src_r.data() };
    float* dest_ptrs[2] = { dest_l.data(), dest_r.data() };

    return handle.apm->ProcessStream(
        src_ptrs, config, config, dest_ptrs);
}

int32_t process_reverse_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,
    int32_t input_sample_rate,
    size_t input_channels,
    int32_t output_sample_rate,
    size_t output_channels,
    rust::Slice<float> dest) {

    webrtc::StreamConfig input_config(input_sample_rate, input_channels);
    webrtc::StreamConfig output_config(output_sample_rate, output_channels);

    const float* src_ptrs[1] = { src.data() };
    float* dest_ptrs[1] = { dest.data() };

    return handle.apm->ProcessReverseStream(
        src_ptrs, input_config, output_config, dest_ptrs);
}

// ── Per-component: ThreeBandFilterBank ────────────────────────────────────────

std::unique_ptr<FilterBankHandle> create_filter_bank() {
    return std::make_unique<FilterBankHandle>();
}

void filter_bank_analysis(
    FilterBankHandle& handle,
    rust::Slice<const float> in,
    rust::Slice<float> out) {
    constexpr int kNumBands = webrtc::ThreeBandFilterBank::kNumBands;
    constexpr int kSplitBandSize = webrtc::ThreeBandFilterBank::kSplitBandSize;
    constexpr int kFullBandSize = webrtc::ThreeBandFilterBank::kFullBandSize;

    // Wrap the packed output slice as 3 separate ArrayViews.
    std::array<webrtc::ArrayView<float>, kNumBands> out_bands = {
        webrtc::ArrayView<float>(out.data(), kSplitBandSize),
        webrtc::ArrayView<float>(out.data() + kSplitBandSize, kSplitBandSize),
        webrtc::ArrayView<float>(out.data() + 2 * kSplitBandSize, kSplitBandSize),
    };
    // Zero-init the output.
    for (auto& band : out_bands) {
        std::fill(band.begin(), band.end(), 0.f);
    }

    handle.bank.Analysis(
        webrtc::ArrayView<const float, kFullBandSize>(in.data(), kFullBandSize),
        webrtc::ArrayView<const webrtc::ArrayView<float>, kNumBands>(
            out_bands.data(), kNumBands));
}

void filter_bank_synthesis(
    FilterBankHandle& handle,
    rust::Slice<const float> in,
    rust::Slice<float> out) {
    constexpr int kNumBands = webrtc::ThreeBandFilterBank::kNumBands;
    constexpr int kSplitBandSize = webrtc::ThreeBandFilterBank::kSplitBandSize;
    constexpr int kFullBandSize = webrtc::ThreeBandFilterBank::kFullBandSize;

    // Wrap the packed input slice as 3 separate ArrayViews.
    // Need non-const copy since ArrayView<const ArrayView<float>> requires ArrayView<float>.
    std::array<float, kFullBandSize> in_copy;
    std::copy(in.data(), in.data() + kFullBandSize, in_copy.data());

    std::array<webrtc::ArrayView<float>, kNumBands> in_bands = {
        webrtc::ArrayView<float>(in_copy.data(), kSplitBandSize),
        webrtc::ArrayView<float>(in_copy.data() + kSplitBandSize, kSplitBandSize),
        webrtc::ArrayView<float>(in_copy.data() + 2 * kSplitBandSize, kSplitBandSize),
    };

    handle.bank.Synthesis(
        webrtc::ArrayView<const webrtc::ArrayView<float>, kNumBands>(
            in_bands.data(), kNumBands),
        webrtc::ArrayView<float, kFullBandSize>(out.data(), kFullBandSize));
}

// ── Per-component: HighPassFilter ────────────────────────────────────────────

std::unique_ptr<HpfHandle> create_hpf(
    int32_t sample_rate_hz,
    size_t num_channels) {
    auto handle = std::make_unique<HpfHandle>();
    handle->hpf = std::make_unique<webrtc::HighPassFilter>(
        sample_rate_hz, num_channels);
    return handle;
}

void hpf_process(
    HpfHandle& handle,
    rust::Slice<float> ch0) {
    // Use the vector<vector<float>>* overload of Process.
    std::vector<std::vector<float>> audio(1);
    audio[0].assign(ch0.data(), ch0.data() + ch0.size());
    handle.hpf->Process(&audio);
    std::copy(audio[0].begin(), audio[0].end(), ch0.data());
}

// ── Per-component: NoiseSuppressor ───────────────────────────────────────────

std::unique_ptr<NsHandle> create_ns(
    uint8_t level,
    int32_t sample_rate_hz,
    size_t num_channels) {
    auto handle = std::make_unique<NsHandle>();

    webrtc::NsConfig config;
    config.target_level =
        static_cast<webrtc::NsConfig::SuppressionLevel>(level);

    handle->ns = std::make_unique<webrtc::NoiseSuppressor>(
        config, static_cast<size_t>(sample_rate_hz), num_channels);

    // Create an AudioBuffer sized for the split band (160 samples at 16kHz
    // internal rate). For NS, the buffer rate is always 16000 (single band).
    size_t buffer_rate = 16000;
    handle->buf = std::make_unique<webrtc::AudioBuffer>(
        buffer_rate, num_channels,
        buffer_rate, num_channels,
        buffer_rate);

    return handle;
}

void ns_analyze(
    NsHandle& handle,
    rust::Slice<const float> band0) {
    // Copy input into the AudioBuffer's channel 0.
    auto& buf = *handle.buf;
    float* ch = buf.channels()[0];
    std::copy(band0.data(), band0.data() + band0.size(), ch);

    handle.ns->Analyze(buf);
}

void ns_process(
    NsHandle& handle,
    rust::Slice<float> band0) {
    // Copy input into the AudioBuffer's channel 0.
    auto& buf = *handle.buf;
    float* ch = buf.channels()[0];
    std::copy(band0.data(), band0.data() + band0.size(), ch);

    handle.ns->Process(&buf);

    // Copy result back.
    const float* out = buf.channels_const()[0];
    std::copy(out, out + band0.size(), band0.data());
}

}  // namespace webrtc_shim
