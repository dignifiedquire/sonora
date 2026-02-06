// C++ shim implementation.
//
// Wraps the abstract AudioProcessing interface for cxx interop.

#include "webrtc-apm-sys/cpp/shim.h"

#include "webrtc/api/audio/audio_processing.h"
#include "webrtc/api/audio/builtin_audio_processing_builder.h"
#include "webrtc/api/environment/environment_factory.h"
#include "webrtc/api/scoped_refptr.h"

namespace webrtc_shim {

struct ApmHandle {
    webrtc::scoped_refptr<webrtc::AudioProcessing> apm;
};

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

}  // namespace webrtc_shim
