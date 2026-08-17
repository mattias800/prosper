#include "vaapi_backend.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <utility>
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

        // ---- sceVideodec2's access-unit path (#2270) -----------------------------------------
        //
        // A DIFFERENT contract from everything above: no container, no demux, no seeking — the
        // caller submits one compressed access unit and expects at most one picture back. The
        // committed asset is a raw Annex-B H.264 elementary stream whose framing is the one a live
        // guest actually submits: every access unit begins `00 00 00 01 09` (a 4-byte start code
        // and an access-unit delimiter), exactly as Tales of Graces f's criMvPly does, and the
        // first one carries the SPS at profile_idc 100 (High) that title configures.
        //
        // WHY EXACT HASHES, and why they are not circular. The expected values below were produced
        // by decoding this same file with the host `ffmpeg` COMMAND-LINE TOOL and hashing the raw
        // NV12 it wrote — a separate program, invoked outside this build, not the code under test.
        // H.264 reconstruction is normatively exact, so any conformant decoder must reproduce them
        // bit for bit; a wrong chroma plane, a swapped U/V, a stale reference frame or a dropped
        // access unit all change the hash while still producing something that looks like video.
        // "It decoded and the picture looks plausible" is precisely the claim this project has a
        // recorded trap for, so it is not the claim made here.
        if (const char* au_asset =
#ifdef PROSPER_TEST_H264_AU_ASSET
                PROSPER_TEST_H264_AU_ASSET
#else
                nullptr
#endif
        ) {
            std::vector<uint8_t> stream;
            if (std::FILE* h = std::fopen(au_asset, "rb")) {
                std::fseek(h, 0, SEEK_END);
                const long size = std::ftell(h);
                std::fseek(h, 0, SEEK_SET);
                if (size > 0) {
                    stream.resize(static_cast<size_t>(size));
                    if (std::fread(stream.data(), 1, stream.size(), h) != stream.size())
                        stream.clear();
                }
                std::fclose(h);
            }
            CHECK(!stream.empty(), "the committed H.264 access-unit asset can be read");

            // Split on access-unit delimiters, which is what the delimiter is FOR and what the
            // guest's own demuxer produces. Splitting per NAL instead would submit an SPS as its
            // own "access unit" and change the picture count for a reason unrelated to decoding.
            std::vector<std::pair<size_t, size_t>> units;
            for (size_t i = 0; i + 4 < stream.size(); ++i) {
                if (stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 0 &&
                    stream[i + 3] == 1 && (stream[i + 4] & 0x1F) == 9) {
                    if (!units.empty()) units.back().second = i - units.back().first;
                    units.emplace_back(i, stream.size() - i);
                }
            }
            CHECK(units.size() == 12, "the asset splits into the expected 12 access units");
            CHECK(!units.empty() && units[0].second > 0 && stream[units[0].first + 4] == 0x09,
                  "every access unit begins with the delimiter the live guest's stream carries");

            // Produced by: ffmpeg -i <asset> -f rawvideo -pix_fmt nv12 ref.nv12, then FNV-1a-64
            // over each 128*96*3/2 = 18432-byte frame. All twelve differ, so a decoder that
            // repeated one picture would fail rather than pass on a lucky constant.
            static const uint64_t kExpected[12] = {
                0x0954fe299ef88406ull, 0x06650c3b8b9ad8d7ull, 0x708abc6ac86599bfull,
                0xec63425d07c5a7bfull, 0xaff60e55974aa1f6ull, 0x588a5618a347d4cbull,
                0xafbdf999f2c63e11ull, 0x85c9459ed1f0aac2ull, 0x09f6e4532aaf8453ull,
                0xcecb63ab8c09525full, 0x7ccfc3e128c5e594ull, 0x0ad1a3f4d15f0157ull,
            };
            constexpr uint32_t kW = 128, kH = 96;

            // BOTH decode paths, against the same reference. decode_au converts differently in each
            // — the hardware path transfers an NV12 surface, the software path interleaves YUV420P
            // chroma on the CPU — so one set of hashes covering only whichever the host happened to
            // pick would leave the other completely unchecked.
            for (int pass = 0; pass < 2 && !stream.empty(); ++pass) {
                const bool force_software = pass == 1;
                if (force_software)
                    setenv("PROSPER_AVP_VAAPI_DEVICE", "/prosper-missing/renderD999", 1);
                else
                    unsetenv("PROSPER_AVP_VAAPI_DEVICE");
                const char* what = force_software ? "software" : "default (VA-API if present)";

                const int dec = backend()->open_decoder(1);
                CHECK(dec >= 0, force_software
                                    ? "AVC access-unit decoder opens with VA-API forced off"
                                    : "AVC access-unit decoder opens (codec=1)");
                if (dec < 0) continue;

                // The decoder writes into OUR buffer, so what is hashed below is exactly what a
                // caller receives — including the copy. Deliberately oversized by a guard region so
                // a decoder writing past the picture is caught rather than tolerated.
                constexpr size_t kNv12 = static_cast<size_t>(kW) * kH * 3 / 2;
                std::vector<uint8_t> dst(kNv12 + 256, 0xC3);
                std::vector<uint64_t> got;
                bool geometry_ok = true, guard_ok = true;
                for (const auto& u : units) {
                    VideoBackend::AuPicture pic{};
                    std::fill(dst.begin(), dst.end(), 0xC3);
                    if (backend()->decode_au(dec, stream.data() + u.first, u.second, dst.data(),
                                             kNv12, pic) != VideoBackend::AuResult::Decoded)
                        continue;
                    if (pic.width != kW || pic.height != kH || pic.y_stride < kW ||
                        pic.nv12_bytes != kNv12)
                        geometry_ok = false;
                    for (size_t i = kNv12; i < dst.size(); ++i)
                        if (dst[i] != 0xC3) { guard_ok = false; break; }
                    uint64_t hv = 0xcbf29ce484222325ull;
                    for (size_t i = 0; i < kNv12; ++i) { hv ^= dst[i]; hv *= 0x100000001b3ull; }
                    got.push_back(hv);
                }
                std::printf("  [info] %s pass: %zu pictures from %zu access units\n", what,
                            got.size(), units.size());
                CHECK(got.size() == 12,
                      force_software ? "12 access units yield 12 pictures (software)"
                                     : "12 access units yield 12 pictures");
                CHECK(geometry_ok, "every decoded picture reports the stream's 128x96 NV12 geometry");
                CHECK(guard_ok, "no decoded picture writes past the buffer size it was given");
                size_t matched = 0;
                for (size_t i = 0; i < got.size() && i < 12; ++i) {
                    if (got[i] == kExpected[i]) { ++matched; continue; }
                    std::printf("  [info] frame %zu: got 0x%016llx expected 0x%016llx\n", i,
                                static_cast<unsigned long long>(got[i]),
                                static_cast<unsigned long long>(kExpected[i]));
                }
                CHECK(matched == 12,
                      force_software
                          ? "every decoded picture is byte-identical to the independent decode"
                            " (software path)"
                          : "every decoded picture is byte-identical to the independent decode");
                backend()->close_decoder(dec);
            }
            unsetenv("PROSPER_AVP_VAAPI_DEVICE");

            // A frame buffer smaller than the picture reports its OWN outcome, distinct from "no
            // picture yet". Collapsed into one bool these are indistinguishable, and the benign one
            // is what a reader would assume — the exact confusion #2270 is about.
            {
                const int dec = backend()->open_decoder(1);
                if (dec >= 0) {
                    constexpr size_t kNv12 = static_cast<size_t>(128) * 96 * 3 / 2;
                    std::vector<uint8_t> small(kNv12 - 1, 0x5A);
                    VideoBackend::AuPicture pic{};
                    bool saw_too_small = false, small_untouched = true;
                    for (const auto& u : units) {
                        const auto r = backend()->decode_au(dec, stream.data() + u.first, u.second,
                                                            small.data(), small.size(), pic);
                        if (r == VideoBackend::AuResult::FrameTooSmall) { saw_too_small = true; break; }
                    }
                    for (uint8_t b : small) if (b != 0x5A) { small_untouched = false; break; }
                    CHECK(saw_too_small,
                          "a too-small destination reports FrameTooSmall, not NoPicture");
                    CHECK(pic.nv12_bytes == kNv12,
                          "FrameTooSmall still reports the exact size the picture needs");
                    CHECK(small_untouched, "a refused-for-size picture writes nothing at all");
                    backend()->close_decoder(dec);
                }
            }

            // A codec prosper has not identified must be REFUSED, not opened on a guess. #2270's
            // whole subject is an entry point that answered successfully when it could not deliver.
            CHECK(backend()->open_decoder(0xBADC0DE) < 0,
                  "an unidentified codec is refused rather than opened on a guess");
        }
    }
    uninstall_vaapi_backend();
    CHECK(backend() == nullptr, "native backend can be unregistered cleanly");
    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
