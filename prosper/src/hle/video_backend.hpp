#pragma once
#include <cstdint>
#include <string>

// Host video-decode backend for the core sceAvPlayer HLE (#705).
//
// The core owns the guest-facing sceAvPlayer contract, per-player lifecycle, the guest event
// callback, and marshalling decoded frames into the guest SceAvPlayerFrameInfo buffers. The actual
// hardware decode is delegated to a backend that the APP (frontend) registers via set_backend() —
// Media Foundation (Windows) / VideoToolbox (macOS) / VA-API (Linux). Native OS media APIs do demux
// + hardware decode together, so a backend owns the whole open->demux->decode pipeline and just hands
// the core the next decoded frame.
//
// When NO backend is registered (headless boot_trace / ctest), the core runs a synthetic black-frame
// lifecycle instead (fires READY/PLAY, delivers blank frames over a short accelerated timeline, then
// STOP/EOF) so a title's video state machine still completes and the boot advances — with zero decode
// dependency. Requirement: real backends MUST use hardware decoding.
namespace prosper::video {

struct StreamInfo {
    uint32_t width = 0, height = 0;
    float    fps = 0.0f;
    bool     has_audio = false;
    uint32_t audio_channels = 0, audio_rate = 0;
    uint64_t duration_us = 0;
};

// One decoded video frame, NV12 (Y plane + interleaved UV plane) — the format PS5 AvPlayer delivers.
struct VideoFrame {
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    uint32_t width = 0, height = 0, y_stride = 0, uv_stride = 0;
    uint64_t pts_us = 0;
};

struct AudioFrame {
    const int16_t* pcm = nullptr;
    uint32_t channels = 0, samples = 0, sample_rate = 0;
    uint64_t pts_us = 0;
};

class VideoBackend {
public:
    virtual ~VideoBackend() = default;
    virtual int  open(const std::string& host_path) = 0;  // returns stream id (>=0), <0 on error; starts demux+HW decode
    virtual bool info(int id, StreamInfo& out) = 0;        // true once stream headers are parsed
    virtual bool next_video(int id, VideoFrame& out) = 0;  // false if no frame ready yet
    virtual bool next_audio(int id, AudioFrame& out) = 0;  // false if no audio ready yet
    virtual bool eof(int id) = 0;                          // true once fully played out
    virtual void close(int id) = 0;
};

// Registered by the app frontend; nullptr (default) selects the core synthetic fallback.
void         set_backend(VideoBackend* b);
VideoBackend* backend();

} // namespace prosper::video
