#include "vaapi_backend.hpp"

#include <algorithm>
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
            // #2899 -- AUDIO MUST NOT BE HOSTAGE TO THE VIDEO QUEUE.
            //
            // One demux/decode worker feeds both queues, so a video queue nobody drains used to stop
            // audio as well: the worker parked in enqueue_video and never got back to the demuxer. A
            // guest that clocks playback on audio (Unity's PS5VideoMedia pulls
            // sceAvPlayerGetAudioData and only asks for a video frame once the audio position moves)
            // therefore deadlocks -- no audio, so no video pull, so the video queue never drains.
            //
            // The arm below is that guest: it pulls ONLY audio. Without the fix it collects at most
            // one video-queue-worth of interleaved audio before the worker parks forever; with it,
            // audio keeps arriving because a starved audio consumer releases the video wait.
            {
                const int id = backend()->open(asset);
                CHECK(id >= 0, "#2899: the clip opens for the audio-only consumer arm");
                if (id >= 0) {
                    StreamInfo stream{};
                    const bool have = backend()->info(id, stream);
                    CHECK(have && stream.has_audio,
                          "#2899: the test clip really has an audio stream (the arm is not vacuous)");
                    unsigned audio_frames = 0;
                    const auto deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(20);
                    while (std::chrono::steady_clock::now() < deadline && !backend()->eof(id)) {
                        AudioFrame audio{};
                        if (backend()->next_audio(id, audio)) ++audio_frames;
                        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    // The clip is one second of 48 kHz stereo AAC, i.e. roughly 47 packets of 1024
                    // samples, and NOTHING here ever pulls a video frame.
                    //
                    // The threshold is MEASURED, not guessed, because the first one guessed did not
                    // discriminate: a parked worker still delivers everything it demuxed before the
                    // sixth video frame filled the queue, which for this asset is 26 packets, so a
                    // ">= 20" arm passed with the fix reverted. Measured on this asset: 26 with the
                    // pre-#2899 unbounded wait, 44 with it. 35 sits between them with margin on both
                    // sides and well under the whole track, so it separates "the worker kept going"
                    // from "the worker parked" without asserting an exact decode order.
                    CHECK(audio_frames >= 35,
                          "#2899: a consumer that pulls ONLY audio keeps receiving it "
                          "(the video queue's backpressure does not stop the session)");
                    std::printf("  [info] #2899 audio-only arm collected %u audio frame(s), "
                                "%llu video frame(s) recycled\n", audio_frames,
                                (unsigned long long)vaapi_video_frames_dropped(id));
                    // The CONTROL that makes the number above mean something: a session whose video
                    // IS consumed must never recycle a frame, so the two arms are separated by the
                    // mechanism and not merely by how long each ran.
                    backend()->close(id);
                }
                const int consuming = backend()->open(asset);
                CHECK(consuming >= 0, "#2899: the clip opens for the consuming control arm");
                if (consuming >= 0) {
                    const auto deadline =
                        std::chrono::steady_clock::now() + std::chrono::seconds(20);
                    unsigned video_frames = 0, audio_frames = 0;
                    while (std::chrono::steady_clock::now() < deadline && !backend()->eof(consuming)) {
                        VideoFrame video{};
                        AudioFrame audio{};
                        if (backend()->next_video(consuming, video)) ++video_frames;
                        if (backend()->next_audio(consuming, audio)) ++audio_frames;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    CHECK(video_frames > 0 && audio_frames > 0,
                          "#2899 control: a consumer that pulls both streams receives both");
                    CHECK(vaapi_video_frames_dropped(consuming) == 0,
                          "#2899 control: a session whose video is consumed recycles NO frame");
                    backend()->close(consuming);
                }
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
            //
            // #2586. Which path each pass ACTUALLY took, as opposed to which it requested. Recorded
            // per pass so the run can state its own coverage instead of claiming it.
            bool pass_hardware[2] = {false, false};
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
                int hardware_pictures = 0, software_pictures = 0;
                for (const auto& u : units) {
                    VideoBackend::AuPicture pic{};
                    std::fill(dst.begin(), dst.end(), 0xC3);
                    if (backend()->decode_au(dec, stream.data() + u.first, u.second, dst.data(),
                                             kNv12, pic) != VideoBackend::AuResult::Decoded)
                        continue;
                    if (pic.width != kW || pic.height != kH || pic.y_stride < kW ||
                        pic.nv12_bytes != kNv12)
                        geometry_ok = false;
                    if (pic.hardware) ++hardware_pictures; else ++software_pictures;
                    for (size_t i = kNv12; i < dst.size(); ++i)
                        if (dst[i] != 0xC3) { guard_ok = false; break; }
                    uint64_t hv = 0xcbf29ce484222325ull;
                    for (size_t i = 0; i < kNv12; ++i) { hv ^= dst[i]; hv *= 0x100000001b3ull; }
                    got.push_back(hv);
                }
                pass_hardware[pass] = hardware_pictures > 0;
                std::printf("  [info] %s pass: %zu pictures from %zu access units, decoded in %s "
                            "(%d hardware / %d software)\n", what, got.size(), units.size(),
                            hardware_pictures ? "HARDWARE (VA-API)" : "SOFTWARE",
                            hardware_pictures, software_pictures);
                // One decoder must not silently change path mid-stream in EITHER direction: the two
                // conversions differ (surface transfer vs CPU chroma interleave), so a pass that did
                // both would be reporting one hash set produced by two code paths.
                CHECK(hardware_pictures == 0 || software_pictures == 0,
                      "a single decoder stays on ONE decode path for the whole stream");
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

            // WHAT THE TWO PASSES ACTUALLY COVERED (#2586). The passes above REQUEST different
            // paths; a request is not an outcome. On a host with no usable VA-API device both are
            // software, and until AuPicture::hardware existed nothing here could say so -- the claim
            // "covered both the hardware and software paths" was then true as a request and
            // unverified as a result, on exactly the headless CI where it matters.
            //
            // A host without VA-API is a SUPPORTED configuration, so this must not become "hardware
            // is required" -- that would turn a diagnostic gap into a broken build. What is asserted
            // is only what is verifiable on any host; the rest is reported.
            //
            // THE RESIDUAL LIMIT, measured rather than reasoned about, because "what else could
            // satisfy this assertion" is the question #2586 exists to ask. Two mutation arms were run
            // against a host that HAS VA-API (12 of 12 pictures in the hardware format):
            //
            //   AuPicture::hardware forced TRUE  -> both assertions below FAIL. Detected.
            //   AuPicture::hardware forced FALSE -> this test PASSES, and prints "this host
            //                                       negotiated NO VA-API hardware decode" on a host
            //                                       that plainly has it. NOT detected.
            //
            // That asymmetry is not fixable from inside a single run, and it is worth stating rather
            // than hiding: "the flag is stuck false" and "this host has no VA-API" are the same
            // observation, and the second is a legitimate supported state. A device having been
            // CREATED does not settle it either -- libavcodec may negotiate a software format behind
            // an attached VA-API device, which is exactly why open_decoder's log says "requested".
            // So the honest guarantee is one-sided: this can prove a pass ran in software, and it can
            // prove two passes differed; it cannot prove a host's hardware path went unexercised.
            CHECK(!pass_hardware[1],
                  "the forced-software pass really did decode in software, not merely ask to");
            if (pass_hardware[0]) {
                CHECK(pass_hardware[0] != pass_hardware[1],
                      "the two passes exercised DIFFERENT decode paths: hardware and software");
                std::puts("  [info] coverage: BOTH decode paths exercised (hardware and software).");
            } else {
                // Stated rather than asserted, and stated plainly: this run is the one-path case.
                std::puts("  [info] coverage: this host negotiated NO VA-API hardware decode, so "
                          "BOTH passes ran the SOFTWARE path and the hardware surface-transfer path "
                          "is NOT covered by this run (#2586). Not a failure: a host without VA-API "
                          "is supported.");
            }

            // ---- sceVideodec2Reset: discard the buffered state, keep the decoder (#2585) --------
            //
            // Two halves of one contract, each with a hand-built positive instance and a control
            // that must come out the OTHER way. A control drawn from the same setup as the claim
            // would only show the machinery runs; these two show the case is expressible and that
            // the fix is what decides it.
            //
            // The asset is what makes this testable: it carries an IDR with its own SPS+PPS at
            // access unit 0 AND at access unit 6, so the second half can be re-fed with the
            // parameter sets stripped -- which is exactly a stream that supplies them once.
            {
                // Strip every SPS(7)/PPS(8) NAL out of one access unit. Annex-B start codes are 3 or
                // 4 bytes and this asset uses both, so both are walked.
                auto strip_parameter_sets = [](const uint8_t* p, size_t len) {
                    std::vector<std::pair<size_t, int>> starts;
                    for (size_t i = 0; i + 3 <= len;) {
                        if (p[i] == 0 && p[i + 1] == 0) {
                            if (i + 4 <= len && p[i + 2] == 0 && p[i + 3] == 1) {
                                starts.emplace_back(i, 4); i += 4; continue;
                            }
                            if (p[i + 2] == 1) { starts.emplace_back(i, 3); i += 3; continue; }
                        }
                        ++i;
                    }
                    std::vector<uint8_t> out;
                    for (size_t k = 0; k < starts.size(); ++k) {
                        const size_t s = starts[k].first;
                        const size_t e = (k + 1 < starts.size()) ? starts[k + 1].first : len;
                        const int nal = p[s + starts[k].second] & 0x1F;
                        if (nal == 7 || nal == 8) continue;
                        out.insert(out.end(), p + s, p + e);
                    }
                    return out;
                };
                constexpr size_t kNv12 = static_cast<size_t>(kW) * kH * 3 / 2;
                std::vector<uint8_t> dst(kNv12, 0);
                // Confirm the fixture really is the case under test before drawing conclusions from
                // it: if the asset ever stopped repeating its parameter sets at unit 6, the stripped
                // units below would be trivially undecodable and both arms would "pass" for the
                // wrong reason.
                const auto stripped_head =
                    strip_parameter_sets(stream.data() + units[6].first, units[6].second);
                CHECK(stripped_head.size() < units[6].second,
                      "access unit 6 really does carry parameter sets for the strip arm to remove");

                auto feed = [&](int dec, size_t from, size_t to, bool strip) {
                    int pictures = 0;
                    for (size_t i = from; i < to; ++i) {
                        VideoBackend::AuPicture pic{};
                        const std::vector<uint8_t> tmp =
                            strip ? strip_parameter_sets(stream.data() + units[i].first,
                                                         units[i].second)
                                  : std::vector<uint8_t>();
                        const uint8_t* p = strip ? tmp.data() : stream.data() + units[i].first;
                        const size_t n = strip ? tmp.size() : units[i].second;
                        if (backend()->decode_au(dec, p, n, dst.data(), kNv12, pic) ==
                            VideoBackend::AuResult::Decoded)
                            ++pictures;
                    }
                    return pictures;
                };

                // ARM 1 -- the parameter sets SURVIVE a Reset. This is #2585 itself: closing the
                // decoder and reopening (what this used to do) throws the parsed SPS/PPS away, and a
                // title whose stream carries them once then cannot decode again at all.
                {
                    const int dec = backend()->open_decoder(1);
                    CHECK(dec >= 0, "a decoder opens for the Reset arms");
                    if (dec >= 0) {
                        const int before = feed(dec, 0, 6, false);
                        CHECK(before == 6, "6 access units decode before the Reset");
                        CHECK(backend()->reset_decoder(dec),
                              "the backend performs an in-place Reset");
                        const int after = feed(dec, 6, 12, true);
                        CHECK(after == 6,
                              "after a Reset the decoder still decodes access units whose SPS/PPS "
                              "have been STRIPPED -- the parameter sets survived (#2585)");
                        backend()->close_decoder(dec);
                    }
                }
                // ARM 1's CONTROL, built by hand and outside the arm: a FRESH decoder is exactly
                // what close-and-reopen produced, and it must fail on the same bytes. Without this
                // the arm above would also pass if `reset_decoder` did nothing at all.
                {
                    const int dec = backend()->open_decoder(1);
                    if (dec >= 0) {
                        const int after = feed(dec, 6, 12, true);
                        CHECK(after == 0,
                              "CONTROL: a FRESH decoder -- what close-and-reopen gives -- decodes "
                              "NOTHING from the same stripped access units (#2585)");
                        backend()->close_decoder(dec);
                    }
                }

                // ARM 2 -- the DPB really is DROPPED. Reset's other half, and it must not be lost
                // while fixing the first: units 7..11 are non-IDR slices with no IDR among them, so
                // a decoder that still holds the pre-reset references decodes them and one that
                // forgot them cannot.
                {
                    const int dec = backend()->open_decoder(1);
                    if (dec >= 0) {
                        CHECK(feed(dec, 0, 6, false) == 6, "6 access units decode before the Reset");
                        CHECK(backend()->reset_decoder(dec), "the backend performs an in-place Reset");
                        CHECK(feed(dec, 7, 12, false) == 0,
                              "after a Reset, non-IDR access units decode to NOTHING -- every "
                              "decoded reference was forgotten (#2585)");
                        backend()->close_decoder(dec);
                    }
                }
                // ARM 2's CONTROL: the same units WITHOUT a Reset must decode, or the arm above
                // proves only that non-IDR units never decode.
                {
                    const int dec = backend()->open_decoder(1);
                    if (dec >= 0) {
                        CHECK(feed(dec, 0, 6, false) == 6, "6 access units decode before the control");
                        // `> 0` AND NOT A FIXED COUNT, deliberately. The real margin here is ONE
                        // picture, not five: the decoder errors out on the rest once the references
                        // diverge. Measured, and independently reproduced against libavcodec 8.1.2
                        // with a separate access-unit splitter. Do NOT "tighten" this to 5 -- a
                        // fixed count is brittle across libavcodec versions and would then fail for
                        // a reason that has nothing to do with Reset.
                        CHECK(feed(dec, 7, 12, false) > 0,
                              "CONTROL: without a Reset the same non-IDR units DO decode, against "
                              "the references the Reset is what discards (#2585)");
                        backend()->close_decoder(dec);
                    }
                }

                // Reset on an id the backend does not know must say so rather than report success.
                CHECK(!backend()->reset_decoder(0x5EED),
                      "Reset on an unknown decoder id is refused, not silently reported as done");
            }

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
