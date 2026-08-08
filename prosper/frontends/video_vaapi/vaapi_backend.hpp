#pragma once

#include "hle/video_backend.hpp"

#include <memory>

namespace prosper::video {

// Native Linux AvPlayer backend. FFmpeg handles MP4 demux and audio conversion while decoded
// video must come from its VA-API hardware-device path. Hardware surfaces are transferred and
// converted to bounded CPU-side NV12 packets for the guest AvPlayer ABI. Software video decode is
// available only through the explicit PROSPER_AVP_ALLOW_SOFTWARE=1 diagnostic override.
class VaapiBackend final : public VideoBackend {
public:
    VaapiBackend();
    ~VaapiBackend() override;

    VaapiBackend(const VaapiBackend&) = delete;
    VaapiBackend& operator=(const VaapiBackend&) = delete;

    bool available() const;
    int open(const std::string& host_path) override;
    int open_memory(const std::string& debug_name, const uint8_t* data, size_t bytes) override;
    bool info(int id, StreamInfo& out) override;
    bool next_video(int id, VideoFrame& out) override;
    bool next_audio(int id, AudioFrame& out) override;
    bool eof(int id) override;
    bool seek(int id, uint64_t position_us) override;
    void close(int id) override;

    // sceVideodec2's access-unit path (#2270). Software decode: the guest submits one compressed
    // access unit and expects at most one picture, so there is no stream to bind a hardware surface
    // pool to at open time, and a 1080p H.264 software decode is well inside budget for the movie
    // playback these titles use it for. Hardware acceleration can be layered on later without
    // changing this interface.
    int open_decoder(uint32_t codec) override;
    bool decode_au(int id, const uint8_t* au, size_t bytes, VideoFrame& out) override;
    void close_decoder(int id) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool install_vaapi_backend();
void uninstall_vaapi_backend();

} // namespace prosper::video
