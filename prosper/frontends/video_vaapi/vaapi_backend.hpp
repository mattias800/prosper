#pragma once

#include "hle/video/video_backend.hpp"

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
    bool peek_video(int id, VideoFrame& out) override;
    bool can_peek_video() const override { return true; }
    bool next_video(int id, VideoFrame& out) override;
    bool next_audio(int id, AudioFrame& out) override;
    bool eof(int id) override;
    bool seek(int id, uint64_t position_us) override;
    void close(int id) override;

    // The implementation behind vaapi_video_frames_dropped below.
    uint64_t video_frames_dropped(int id) override;
    uint64_t video_frames_dropped_for_test(int id);

    // sceVideodec2's access-unit path (#2270). The guest submits one compressed access unit and
    // expects at most one picture back, which is libavcodec's send_packet/receive_frame contract --
    // so this is a second entry point onto the same decoder, not a second decoder.
    //
    // VA-API is REQUESTED first and software is the fallback. (This comment used to say the path was
    // software-only with "hardware acceleration can be layered on later"; it was layered on, and the
    // comment was not updated. A stale claim about which path runs is the exact confusion #2586 is
    // about, so note that neither this comment nor open_decoder's log establishes the outcome --
    // only AuPicture::hardware does, because it is set from the pixel format a frame came back in.)
    int open_decoder(uint32_t codec) override;
    AuResult decode_au(int id, const uint8_t* au, size_t bytes,
                       uint8_t* dst, uint64_t dst_bytes, AuPicture& out) override;
    bool reset_decoder(int id) override;
    void close_decoder(int id) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool install_vaapi_backend();
void uninstall_vaapi_backend();

// TEST SEAM (#2899): how many decoded video frames this open session recycled because a live audio
// consumer was starved while the video queue was full. It is 0 for every session whose video IS
// consumed -- which is what makes it a DISCRIMINATOR rather than an observation: the audio-only arm
// and the consuming control arm are separated by this number, not only by how much each collected.
// Returns 0 for an unknown id or when a different backend is installed.
uint64_t vaapi_video_frames_dropped(int id);

} // namespace prosper::video
