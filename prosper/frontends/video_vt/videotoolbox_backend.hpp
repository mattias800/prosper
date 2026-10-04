#pragma once

#include "hle/video/video_backend.hpp"

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::video {

// Native macOS AvPlayer backend. AVFoundation's AVAssetReader owns MP4 demux; video is decoded by
// VideoToolbox and read back as NV12 (420v, the format PS5 AvPlayer delivers) and audio is converted
// to interleaved signed 16-bit PCM. Decoded samples are copied into bounded CPU queues by one worker
// thread per session, so guest GetVideoData/GetAudioData calls never block on the reader.
//
// Compressed video is accepted only when VideoToolbox reports a hardware decoder for the track's
// codec (VTIsHardwareDecodeSupported). The diagnostic PROSPER_AVP_ALLOW_SOFTWARE=1 override is
// explicit and is reported in the log, as on Linux and Windows.
class VideoToolboxBackend final : public VideoBackend {
public:
    VideoToolboxBackend();
    ~VideoToolboxBackend() override;

    VideoToolboxBackend(const VideoToolboxBackend&) = delete;
    VideoToolboxBackend& operator=(const VideoToolboxBackend&) = delete;

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
    int video_queue_depth(int id) override;

    // The decoder this session's VTDecompressionSession REPORTS it is using
    // (kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder): 1 hardware, 0 software
    // (only reachable with PROSPER_AVP_ALLOW_SOFTWARE), -1 unknown id. An observation, not a
    // capability query -- the hardware policy is enforced on that same session.
    int decoder_uses_hardware(int id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Frontends call this during their post-HLE backend installation hook. It is intentionally separate
// from prosper_core so headless/core-only builds keep no AVFoundation dependency.
bool install_videotoolbox_backend();
void uninstall_videotoolbox_backend();

// TEST SEAMS over the real crop and pack code, for a caller-built pixel buffer (420v): the displayed
// rectangle the backend derives from the buffer alone (its clean aperture), as {x, y, w, h} with a
// top-left origin, and the packed NV12 it would deliver for that rectangle.
bool videotoolbox_display_rect_for_test(CVPixelBufferRef image, uint32_t rect[4]);
bool videotoolbox_pack_for_test(CVPixelBufferRef image, const uint32_t rect[4],
                                std::vector<uint8_t>* nv12, uint32_t* stride);

// TEST SEAM: the path prefix of open_memory's temporary copies in this process
// (`<temp dir>/prosper-avp-<pid>-`), so a test can count its own copies without seeing another
// process's.
std::string videotoolbox_temp_prefix();

}   // namespace prosper::video
