#include "vaapi_backend.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

using namespace prosper::video;

namespace {
int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)
} // namespace

int main(int argc, char** argv) {
    std::puts("== test_video_vaapi ==");
    set_backend(nullptr);
    CHECK(install_vaapi_backend(), "FFmpeg VA-API backend is available");
    CHECK(backend() != nullptr, "native backend is registered explicitly");
    if (backend()) {
        CHECK(backend()->open("/prosper-missing/no-such-video.mp4") < 0,
              "missing source fails instead of fabricating a decode session");
        // Decode a committed test-pattern clip. On a headless host (no DRI render node / no working
        // VA driver — CI runners, containers) VA-API hardware decode is unavailable, so a successful
        // decode here proves REAL FFmpeg software decode is the default fallback (#320: without it,
        // sceAvPlayer could not open a title's movie and the game deadlocked in its black intro).
        // A path argument overrides the built-in asset (local diagnostics with a bigger clip).
        const char* asset = (argc == 2) ? argv[1] :
#ifdef PROSPER_TEST_VIDEO_ASSET
            PROSPER_TEST_VIDEO_ASSET;
#else
            nullptr;
#endif
        if (asset) {
            const int id = backend()->open(asset);
            CHECK(id >= 0, "MP4 opens with the selected VA-API or explicit diagnostic pipeline");
            if (id >= 0) {
                StreamInfo stream{};
                CHECK(backend()->info(id, stream) && stream.width && stream.height && stream.fps > 0,
                      "MP4 stream metadata contains dimensions and frame rate");
                unsigned video_frames = 0, audio_frames = 0;
                uint64_t previous_video_pts = 0, previous_audio_pts = 0;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (std::chrono::steady_clock::now() < deadline &&
                       (video_frames < 3 || (stream.has_audio && audio_frames == 0)) &&
                       !backend()->eof(id)) {
                    VideoFrame video{};
                    if (backend()->next_video(id, video)) {
                        CHECK(video.y && video.uv && video.width == stream.width &&
                                  video.height == stream.height && video.y_stride >= video.width &&
                                  video.uv_stride >= video.width,
                              "decoded video frame exposes stable NV12 planes and strides");
                        CHECK(video_frames == 0 || video.pts_us >= previous_video_pts,
                              "decoded video timestamps are monotonic");
                        previous_video_pts = video.pts_us;
                        ++video_frames;
                    }
                    AudioFrame audio{};
                    if (backend()->next_audio(id, audio)) {
                        CHECK(audio.pcm && audio.channels == stream.audio_channels &&
                                  audio.sample_rate == stream.audio_rate && audio.samples > 0,
                              "decoded audio frame exposes interleaved 16-bit PCM");
                        CHECK(audio_frames == 0 || audio.pts_us >= previous_audio_pts,
                              "decoded audio timestamps are monotonic");
                        previous_audio_pts = audio.pts_us;
                        ++audio_frames;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                CHECK(video_frames >= 3, "backend delivers multiple real video frames");
                if (stream.has_audio) CHECK(audio_frames > 0, "backend delivers real audio samples");
                if (stream.has_audio && video_frames >= 3) {
                    const auto eof_deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(30);
                    while (std::chrono::steady_clock::now() < eof_deadline &&
                           !backend()->eof(id)) {
                        VideoFrame video{};
                        if (backend()->next_video(id, video)) ++video_frames;
                        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    CHECK(backend()->eof(id),
                          "video-only consumption reaches EOF despite queued audio");
                }
                backend()->close(id);
            }
        }
    }
    uninstall_vaapi_backend();
    CHECK(backend() == nullptr, "native backend can be unregistered cleanly");
    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
