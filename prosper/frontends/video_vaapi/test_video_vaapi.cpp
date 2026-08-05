#include "vaapi_backend.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

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
            CHECK(id >= 0, "MP4 opens with hardware VA-API or the software-decode fallback");
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

                    // #1949 — sceAvPlayerJumpToTime needs a real reposition of a real container,
                    // and the hardest case is the one right here: a session whose decode already
                    // ran to completion. A seek that only flushed queues could not answer it.
                    auto pull_next = [&](VideoFrame& out) {
                        const auto limit = std::chrono::steady_clock::now() +
                                           std::chrono::seconds(10);
                        while (std::chrono::steady_clock::now() < limit) {
                            if (backend()->next_video(id, out)) return true;
                            if (backend()->eof(id)) return false;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        return false;
                    };
                    CHECK(backend()->seek(id, 500'000),
                          "seek repositions a decode session that already reached EOF");
                    CHECK(!backend()->eof(id), "a repositioned session is no longer at EOF");
                    VideoFrame sought{};
                    const bool got_sought = pull_next(sought);
                    CHECK(got_sought && sought.pts_us >= 500'000,
                          "the first frame after a seek is at or after the requested position");
                    CHECK(backend()->seek(id, 0), "seeking back to the start succeeds");
                    VideoFrame restarted{};
                    const bool got_restart = pull_next(restarted);
                    CHECK(got_restart && restarted.pts_us < 500'000,
                          "seeking back to the start rewinds the delivered timestamps");
                }
                backend()->close(id);
                CHECK(!backend()->seek(id, 0), "seeking a closed session fails instead of lying");
            }
            // #1955 — the same clip fed as BYTES rather than as a path. This is the route a title
            // that stores its media inside a container file must take, because only its own
            // sceAvPlayerInit file-replacement callbacks know where the media starts.
            {
                std::FILE* handle = std::fopen(asset, "rb");
                CHECK(handle != nullptr, "the test clip can be read into memory");
                std::vector<uint8_t> bytes;
                if (handle) {
                    std::fseek(handle, 0, SEEK_END);
                    const long size = std::ftell(handle);
                    std::fseek(handle, 0, SEEK_SET);
                    if (size > 0) {
                        bytes.resize(static_cast<size_t>(size));
                        if (std::fread(bytes.data(), 1, bytes.size(), handle) != bytes.size())
                            bytes.clear();
                    }
                    std::fclose(handle);
                }
                const int mem_id = bytes.empty()
                                       ? -1
                                       : backend()->open_memory("in-memory-testpattern",
                                                                bytes.data(), bytes.size());
                CHECK(mem_id >= 0, "an in-memory source opens through the custom AVIO path");
                if (mem_id >= 0) {
                    StreamInfo mem_stream{};
                    CHECK(backend()->info(mem_id, mem_stream) && mem_stream.width > 0 &&
                              mem_stream.height > 0 && mem_stream.fps > 0.0f,
                          "the in-memory source parses the same container's stream metadata");
                    unsigned mem_frames = 0;
                    const auto mem_deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    while (std::chrono::steady_clock::now() < mem_deadline && mem_frames < 3 &&
                           !backend()->eof(mem_id)) {
                        VideoFrame frame{};
                        if (backend()->next_video(mem_id, frame)) ++mem_frames;
                        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    CHECK(mem_frames >= 3, "the in-memory source decodes real frames");
                    // Seeking must work without a host file behind it: the session owns the bytes.
                    CHECK(backend()->seek(mem_id, 500'000),
                          "an in-memory source can be repositioned");
                    backend()->close(mem_id);
                }
                // Bytes that are not a media container must fail, not fabricate a session.
                const std::vector<uint8_t> garbage(4096, 0x5a);
                CHECK(backend()->open_memory("garbage", garbage.data(), garbage.size()) < 0,
                      "undecodable in-memory bytes fail to open instead of fabricating a stream");
            }

            // Deterministic software-fallback guard for ANY host (incl. machines with working
            // VA-API): force VA-API device creation to fail by pointing it at a nonexistent render
            // node, then assert the SAME clip still decodes — via real software decode (#705). Without
            // the default fallback this open would fail, reproducing the PPSA02664 hang.
            setenv("PROSPER_AVP_VAAPI_DEVICE", "/prosper-missing/renderD999", 1);
            const int sw_id = backend()->open(asset);
            CHECK(sw_id >= 0, "clip decodes via software fallback when VA-API device is unavailable");
            if (sw_id >= 0) {
                unsigned sw_frames = 0;
                const auto sw_deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (std::chrono::steady_clock::now() < sw_deadline && sw_frames < 3 &&
                       !backend()->eof(sw_id)) {
                    VideoFrame frame{};
                    if (backend()->next_video(sw_id, frame)) ++sw_frames;
                    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                CHECK(sw_frames >= 3, "software fallback delivers real frames with hardware forced off");
                backend()->close(sw_id);
            }
            unsetenv("PROSPER_AVP_VAAPI_DEVICE");
        }
    }
    uninstall_vaapi_backend();
    CHECK(backend() == nullptr, "native backend can be unregistered cleanly");
    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
