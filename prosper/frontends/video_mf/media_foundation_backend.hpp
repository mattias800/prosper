#pragma once

#include "hle/video_backend.hpp"

#include <memory>

namespace prosper::video {

// Native Windows AvPlayer backend. Media Foundation owns MP4 demux and decode; decoded samples are
// copied into bounded CPU queues so guest GetVideoData/GetAudioData calls never block in ReadSample.
// Compressed video is accepted only when Source Reader exposes a hardware video-decoder MFT. The
// diagnostic PROSPER_AVP_ALLOW_SOFTWARE=1 override is explicit and is reported in the log.
class MediaFoundationBackend final : public VideoBackend {
public:
    MediaFoundationBackend();
    ~MediaFoundationBackend() override;

    MediaFoundationBackend(const MediaFoundationBackend&) = delete;
    MediaFoundationBackend& operator=(const MediaFoundationBackend&) = delete;

    bool available() const;
    int open(const std::string& host_path) override;
    bool info(int id, StreamInfo& out) override;
    bool next_video(int id, VideoFrame& out) override;
    bool next_audio(int id, AudioFrame& out) override;
    bool eof(int id) override;
    void close(int id) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Frontends call this during their post-HLE backend installation hook. It is intentionally separate
// from prosper_core so headless/core-only builds retain no Media Foundation dependency.
bool install_media_foundation_backend();
void uninstall_media_foundation_backend();

} // namespace prosper::video
