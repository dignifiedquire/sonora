// C++ shim implementation.
//
// Wraps the abstract AudioProcessing interface for cxx interop.

#include "sonora-sys/cpp/shim.h"

#include "webrtc/api/audio/builtin_audio_processing_builder.h"
#include "webrtc/api/environment/environment_factory.h"

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

void apply_config(
    ApmHandle& handle,
    bool ec_enabled,
    bool ns_enabled,
    uint8_t ns_level,
    bool agc2_enabled) {
    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = ec_enabled;
    config.noise_suppression.enabled = ns_enabled;
    config.noise_suppression.level =
        static_cast<webrtc::AudioProcessing::Config::NoiseSuppression::Level>(ns_level);
    config.gain_controller2.enabled = agc2_enabled;
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

}  // namespace webrtc_shim
