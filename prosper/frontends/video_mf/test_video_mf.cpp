#include "media_foundation_backend.hpp"
#include "vp9_uncompressed_header.hpp"

#include <windows.h>
#include <combaseapi.h>
#include <mfapi.h>
#include <mftransform.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---- sceVideodec2's access-unit path (#2983) ---------------------------------------------------
//
// The guest demuxes itself and submits one compressed access unit at a time, so these streams are
// fed exactly that way: no container around the units the decoder sees.
//
// WHY EXACT HASHES, and why they are not circular. Every expected value below was produced by
// decoding the same committed file with the host `ffmpeg` command-line tool (libavcodec's own
// decoders, not Media Foundation) and hashing the raw NV12 it wrote. H.264 and VP9 reconstruction
// are both normatively exact, so a conformant decoder must reproduce them bit for bit; a wrong
// chroma plane, a padded surface copied as if it were the picture, a hidden alt-ref frame returned as
// a picture, or a dropped unit all change the hash while still producing something that looks like
// video.
namespace {

std::vector<uint8_t> read_file(const char* path) {
    std::vector<uint8_t> bytes;
    if (std::FILE* f = std::fopen(path, "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (size > 0) {
            bytes.resize(static_cast<size_t>(size));
            if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
        }
        std::fclose(f);
    }
    return bytes;
}

uint64_t fnv1a(const uint8_t* data, size_t bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < bytes; ++i) { h ^= data[i]; h *= 0x100000001b3ull; }
    return h;
}

void set_software(bool on) { _putenv_s("PROSPER_VDEC2_MF_SOFTWARE", on ? "1" : ""); }

// Whether THIS HOST has a synchronous Media Foundation decoder for the format, asked of Media
// Foundation directly rather than of the code under test. Codecs are the host's: H.264 ships with
// Windows, VP9 comes from the separately installed VP9 Video Extensions, and a Windows Server CI image
// may have neither. A host without one is a supported configuration, so the decode assertions are
// conditional on this -- and the refusal is asserted instead, so a skip is never silent and a
// backend that fabricated a decoder where none exists would fail.
bool host_has_decoder(const GUID& subtype) {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFT_REGISTER_TYPE_INFO input{MFMediaType_Video, subtype};
    IMFActivate** activates = nullptr;
    UINT32 count = 0;
    const HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER,
                                 MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT |
                                     MFT_ENUM_FLAG_SORTANDFILTER,
                                 &input, nullptr, &activates, &count);
    for (UINT32 i = 0; SUCCEEDED(hr) && i < count; ++i) activates[i]->Release();
    CoTaskMemFree(activates);
    if (SUCCEEDED(com)) CoUninitialize();
    return SUCCEEDED(hr) && count > 0;
}

// The skip arm: with no decoder on the host, the codec must be REFUSED -- not opened into a decoder
// that then reports "no picture" forever, which is #2270's shape.
void check_refused_without_decoder(const char* label, uint32_t codec) {
    std::printf("  [info] SKIP %s decode: this host has no Media Foundation %s decoder\n", label,
                label);
    const int dec = backend()->open_decoder(codec);
    CHECK(dec < 0, "with no host decoder installed, the codec is refused rather than faked");
    if (dec >= 0) backend()->close_decoder(dec);
}

using Unit = std::pair<size_t, size_t>;   // offset, length

struct Stream {
    const char* label;
    uint32_t codec;
    uint32_t width, height;
    std::vector<uint8_t> bytes;
    std::vector<Unit> units;
    const uint64_t* expected;
    size_t expected_count;
};

// Decode `units` on decoder `dec` and hash every picture. Checks geometry and the guard region.
struct DecodeRun {
    std::vector<uint64_t> hashes;
    int hardware = 0, software = 0;
    bool geometry_ok = true, guard_ok = true;
};

DecodeRun decode_units(int dec, const Stream& s, const std::vector<Unit>& units) {
    DecodeRun run;
    const size_t nv12 = static_cast<size_t>(nv12_bytes(s.width, s.height));
    std::vector<uint8_t> dst(nv12 + 256);
    for (const Unit& u : units) {
        std::fill(dst.begin(), dst.end(), 0xC3);
        VideoBackend::AuPicture pic{};
        if (backend()->decode_au(dec, s.bytes.data() + u.first, u.second, dst.data(), nv12, pic) !=
            VideoBackend::AuResult::Decoded)
            continue;
        if (pic.width != s.width || pic.height != s.height || pic.nv12_bytes != nv12 ||
            pic.y_stride < s.width)
            run.geometry_ok = false;
        for (size_t i = nv12; i < dst.size(); ++i)
            if (dst[i] != 0xC3) { run.guard_ok = false; break; }
        if (pic.hardware) ++run.hardware; else ++run.software;
        run.hashes.push_back(fnv1a(dst.data(), nv12));
    }
    return run;
}

size_t count_matches(const std::vector<uint64_t>& got, const uint64_t* expected, size_t offset,
                     size_t count) {
    size_t matched = 0;
    for (size_t i = 0; i < got.size() && i < count; ++i) {
        if (got[i] == expected[offset + i]) { ++matched; continue; }
        std::printf("  [info] picture %zu: got 0x%016llx expected 0x%016llx\n", offset + i,
                    static_cast<unsigned long long>(got[i]),
                    static_cast<unsigned long long>(expected[offset + i]));
    }
    return matched;
}

// Both paths, against the same reference: DXVA hands back a D3D11 surface that is copied through a
// staging texture, software hands back a system-memory buffer with its own pitch, and one set of
// hashes over whichever the host picked would leave the other unchecked. Which path each pass
// ACTUALLY took is reported from AuPicture::hardware, never assumed from the request (#2586).
void check_stream_both_paths(const Stream& s) {
    bool pass_hardware[2] = {false, false};
    for (int pass = 0; pass < 2; ++pass) {
        const bool force_software = pass == 1;
        set_software(force_software);
        const int dec = backend()->open_decoder(s.codec);
        std::printf("  [info] %s %s pass: decoder id %d\n", s.label,
                    force_software ? "forced-software" : "default", dec);
        CHECK(dec >= 0, force_software ? "access-unit decoder opens with DXVA forced off"
                                       : "access-unit decoder opens");
        if (dec < 0) continue;
        const DecodeRun run = decode_units(dec, s, s.units);
        pass_hardware[pass] = run.hardware > 0;
        std::printf("  [info] %s %s pass: %zu pictures from %zu access units (%d DXVA / %d "
                    "software)\n", s.label, force_software ? "forced-software" : "default",
                    run.hashes.size(), s.units.size(), run.hardware, run.software);
        CHECK(run.hardware == 0 || run.software == 0,
              "a single decoder stays on ONE decode path for the whole stream");
        CHECK(run.hashes.size() == s.expected_count,
              "every shown frame yields exactly one picture (no hidden frame, none lost)");
        CHECK(run.geometry_ok, "every picture reports the stream's display size as packed NV12");
        CHECK(run.guard_ok, "no picture writes past the buffer size it was given");
        CHECK(count_matches(run.hashes, s.expected, 0, s.expected_count) == s.expected_count,
              "every picture is byte-identical to the independent ffmpeg decode");
        backend()->close_decoder(dec);
    }
    set_software(false);
    CHECK(!pass_hardware[1], "the forced-software pass really decoded in software");
    if (pass_hardware[0])
        std::printf("  [info] %s coverage: BOTH paths exercised (DXVA surface and software).\n",
                    s.label);
    else
        std::printf("  [info] %s coverage: this host's decoder gave NO DXVA surface, so both "
                    "passes ran in software and the surface copy is NOT covered by this run.\n",
                    s.label);
}

// IVF: a 32-byte file header, then per frame a 12-byte header (u32 size, u64 pts) and the frame.
// Each frame is one VP9 access unit exactly as a demuxer hands it on -- a superframe (hidden alt-ref
// plus the frame it serves) stays ONE unit, which is how a guest demuxer would submit it.
std::vector<Unit> split_ivf(const std::vector<uint8_t>& b) {
    std::vector<Unit> units;
    if (b.size() < 32 || std::memcmp(b.data(), "DKIF", 4) != 0) return units;
    size_t off = 32;
    while (off + 12 <= b.size()) {
        const uint32_t size = b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) |
                              (static_cast<uint32_t>(b[off + 3]) << 24);
        if (off + 12 + size > b.size()) break;
        units.emplace_back(off + 12, size);
        off += 12 + size;
    }
    return units;
}

// Access units begin with an access-unit delimiter, the framing a live guest submits.
std::vector<Unit> split_annexb_aud(const std::vector<uint8_t>& b) {
    std::vector<Unit> units;
    for (size_t i = 0; i + 4 < b.size(); ++i) {
        if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0 && b[i + 3] == 1 &&
            (b[i + 4] & 0x1F) == 9) {
            if (!units.empty()) units.back().second = i - units.back().first;
            units.emplace_back(i, b.size() - i);
        }
    }
    return units;
}

bool is_superframe(const uint8_t* p, size_t n) {
    return n > 0 && (p[n - 1] & 0xE0) == 0xC0;
}

void test_access_unit_path() {
    std::puts("-- sceVideodec2 access-unit decode (#2983) --");
    CHECK(backend()->open_decoder(0x7777) < 0,
          "an unidentified codec value is refused rather than guessed");

#ifdef PROSPER_TEST_VP9_AU_ASSET
    {
        // Produced by: ffmpeg -i vp9_testpattern.ivf -f rawvideo -pix_fmt nv12 ref.nv12, then
        // FNV-1a-64 over each 160*90 + 2*80*45 = 21600-byte frame. All 24 differ.
        static const uint64_t kVp9Expected[24] = {
            0xc5d881f5ad18d6e3ull, 0x0a104ad23033be37ull, 0x0c671d7cbaffd853ull,
            0xb050c7d781be8167ull, 0x30114c4d0db7d39dull, 0x64bde2949d311befull,
            0x437df2ad43354a19ull, 0x75ba29d1fff0cd40ull, 0x19f7fb26cfc95d38ull,
            0x3a15afbba23fda2eull, 0xbd6bfff70be35829ull, 0x762c1ff5bc183ce6ull,
            0x3218d18e7619cc16ull, 0x5069861f5e6eb052ull, 0xbcb8cbd90ac03d32ull,
            0xfcaab884215e640eull, 0xa8609984e0cd3d83ull, 0x895e7e704f12516aull,
            0xb33fcf120f8f743full, 0x9b769661ad98990bull, 0x0dc8693197c79bcbull,
            0x2ef06dd3c0b3d647ull, 0x4fee04f0e7dbf647ull, 0x82a50de1bd934ad4ull,
        };
        Stream s{"VP9", 2382845, 160, 90, read_file(PROSPER_TEST_VP9_AU_ASSET), {}, kVp9Expected,
                 24};
        s.units = split_ivf(s.bytes);
        CHECK(s.units.size() == 24, "the VP9 asset splits into 24 access units");
        size_t superframes = 0;
        for (const Unit& u : s.units)
            if (is_superframe(s.bytes.data() + u.first, u.second)) ++superframes;
        std::printf("  [info] VP9 asset: %zu of %zu access units are superframes\n", superframes,
                    s.units.size());
        // The case this asset exists for: a unit carrying a hidden alt-ref frame AND a shown one
        // must produce ONE picture. Without it the count assertion could not see a hidden frame
        // leaking out as a picture.
        CHECK(superframes >= 1, "the asset carries at least one superframe (hidden alt-ref)");

        uint32_t w = 0, h = 0;
        CHECK(!s.units.empty() &&
                  vp9_key_frame_size(s.bytes.data() + s.units[0].first, s.units[0].second, w, h) &&
                  w == 160 && h == 90,
              "the VP9 key-frame header parser reads 160x90 off the first access unit");
        CHECK(s.units.size() > 1 &&
                  !vp9_key_frame_size(s.bytes.data() + s.units[1].first, s.units[1].second, w, h),
              "the parser refuses an inter frame instead of inventing a size");
        const uint8_t truncated[3] = {0x82, 0x49, 0x83};
        CHECK(!vp9_key_frame_size(truncated, sizeof truncated, w, h),
              "the parser refuses a truncated key-frame header");

        if (!host_has_decoder(MFVideoFormat_VP90)) {
            check_refused_without_decoder("VP9", s.codec);
        } else if (s.units.size() == 24) {
            check_stream_both_paths(s);

            // A picture that does not fit must be reported as its OWN outcome, carrying the size it
            // needed -- not folded into "no picture" (#2571 review D6).
            {
                const int dec = backend()->open_decoder(s.codec);
                std::vector<uint8_t> small(1024);
                VideoBackend::AuPicture pic{};
                const auto r = dec >= 0
                    ? backend()->decode_au(dec, s.bytes.data() + s.units[0].first,
                                           s.units[0].second, small.data(), small.size(), pic)
                    : VideoBackend::AuResult::NoPicture;
                CHECK(r == VideoBackend::AuResult::FrameTooSmall && pic.nv12_bytes == 21600,
                      "a too-small destination is FrameTooSmall and names the 21600 bytes needed");
                if (dec >= 0) backend()->close_decoder(dec);
            }

            // sceVideodec2Reset: discard, keep the decoder (#2585). Decode the first six units,
            // reset, then feed the second group of pictures from its key frame at unit 12: the
            // same decoder must reproduce frames 12..23 exactly. A reset that left the decoder
            // unusable, or one that let a pre-reset picture out afterwards, fails this.
            {
                const int dec = backend()->open_decoder(s.codec);
                CHECK(dec >= 0, "decoder for the reset arm opens");
                if (dec >= 0) {
                    decode_units(dec, s, std::vector<Unit>(s.units.begin(), s.units.begin() + 6));
                    CHECK(backend()->reset_decoder(dec),
                          "reset is performed in place (the backend has a real discard)");
                    const DecodeRun after =
                        decode_units(dec, s, std::vector<Unit>(s.units.begin() + 12, s.units.end()));
                    std::printf("  [info] after reset: %zu pictures from 12 access units\n",
                                after.hashes.size());
                    CHECK(after.hashes.size() == 12 &&
                              count_matches(after.hashes, kVp9Expected, 12, 12) == 12,
                          "after reset the same decoder decodes frames 12..23 byte-exactly");
                    backend()->close_decoder(dec);
                }
            }
        }
    }
#endif

#ifdef PROSPER_TEST_H264_AU_ASSET
    {
        // The Linux backend's asset and its expected values (test_video_vaapi.cpp): ffmpeg -i
        // <asset> -f rawvideo -pix_fmt nv12, FNV-1a-64 per 128*96*3/2-byte frame.
        static const uint64_t kH264Expected[12] = {
            0x0954fe299ef88406ull, 0x06650c3b8b9ad8d7ull, 0x708abc6ac86599bfull,
            0xec63425d07c5a7bfull, 0xaff60e55974aa1f6ull, 0x588a5618a347d4cbull,
            0xafbdf999f2c63e11ull, 0x85c9459ed1f0aac2ull, 0x09f6e4532aaf8453ull,
            0xcecb63ab8c09525full, 0x7ccfc3e128c5e594ull, 0x0ad1a3f4d15f0157ull,
        };
        Stream s{"H.264", 1, 128, 96, read_file(PROSPER_TEST_H264_AU_ASSET), {}, kH264Expected, 12};
        s.units = split_annexb_aud(s.bytes);
        CHECK(s.units.size() == 12, "the H.264 asset splits into 12 access units");
        if (!host_has_decoder(MFVideoFormat_H264)) check_refused_without_decoder("H.264", s.codec);
        else if (s.units.size() == 12) check_stream_both_paths(s);
    }
#endif
#ifdef PROSPER_TEST_H264_BFRAMES_AU_ASSET
    {
        // REORDERING. The asset above has no B-frames, so it cannot tell a decoder that returns
        // pictures in DISPLAY order from one that returns them in decode order -- both give the same
        // sequence there. This one does: x264 High profile, two B-frames between references
        // (I B B P B B P ...), access-unit delimiters on. Low-latency mode, which open_decoder asks
        // the H.264 decoder for, is exactly the setting that could plausibly get this wrong.
        //
        // It is also 128x90, which H.264 cannot code: the stream carries 96 rows of macroblocks and
        // an SPS crop down to 90, so the decoder's surface is taller than the picture and the chroma
        // plane starts at row 96 of it, not row 90. That is the 1920x1080-in-1088 shape every
        // 1080p title's movie has, and the only case where copying the surface instead of the display
        // aperture changes the bytes.
        //
        // Expected values: ffmpeg -i <asset> -f rawvideo -pix_fmt nv12, FNV-1a-64 per
        // 128*90 + 2*64*45 = 17280-byte frame, in DISPLAY order. A decoder must still hold back the last pictures a B-frame
        // stream needs for reordering -- sceVideodec2Flush does not drain (#2562), so those are not
        // delivered by anything -- which is why the count is a floor, not 24.
        static const uint64_t kExpected[24] = {
            0x1f9d50f25f6500acull, 0xbdc621a22b411f45ull, 0xcb4dad8d78986716ull,
            0x7d3367e410b77fdbull, 0x25ae0a77e3f3ac9aull, 0x8aa69acf06f9d4f1ull,
            0x47d80f32ef60aa17ull, 0xce4e047e10de8515ull, 0xe7e51307a9d3addaull,
            0x5d6a24350aa59384ull, 0xce85896ca904b1b8ull, 0x384da39a08c91879ull,
            0x678b79ccde52f11aull, 0x3e9dffba7b91d47cull, 0xc9a2fd4b1244a80aull,
            0x816a551a31b8e1a5ull, 0xb1a0087b394bc8f6ull, 0x8d0556352de7d837ull,
            0x0c77594d47ec4165ull, 0x61ac2bccd97251c9ull, 0x6cfe017d9fcdb37bull,
            0xa093752e9d455a98ull, 0x60a54401e25427a3ull, 0xffea4db88db1fd64ull,
        };
        Stream s{"H.264 B-frames", 1, 128, 90, read_file(PROSPER_TEST_H264_BFRAMES_AU_ASSET), {},
                 kExpected, 24};
        s.units = split_annexb_aud(s.bytes);
        CHECK(s.units.size() == 24, "the B-frame H.264 asset splits into 24 access units");
        const bool have = host_has_decoder(MFVideoFormat_H264);
        for (int pass = 0; have && pass < 2 && s.units.size() == 24; ++pass) {
            set_software(pass == 1);
            const int dec = backend()->open_decoder(s.codec);
            CHECK(dec >= 0, "B-frame decoder opens");
            if (dec < 0) continue;
            const DecodeRun run = decode_units(dec, s, s.units);
            std::printf("  [info] B-frame %s pass: %zu pictures from 24 access units (%d DXVA / %d "
                        "software)\n", pass ? "forced-software" : "default", run.hashes.size(),
                        run.hardware, run.software);
            CHECK(run.hashes.size() >= 20 && run.hashes.size() <= 24,
                  "all but the reorder-held tail of a B-frame stream is delivered");
            CHECK(run.geometry_ok && run.guard_ok, "B-frame pictures are the 128x90 display aperture, not the 128x96 surface");
            CHECK(count_matches(run.hashes, kExpected, 0, run.hashes.size()) == run.hashes.size(),
                  "B-frame pictures come out in DISPLAY order, each byte-identical to ffmpeg");
            backend()->close_decoder(dec);
        }
        set_software(false);
    }
#endif
}

} // namespace

int main(int argc, char** argv) {
    std::puts("== test_video_mf ==");
    set_backend(nullptr);
    CHECK(install_media_foundation_backend(), "Media Foundation starts");
    CHECK(backend() != nullptr, "native backend is registered explicitly");
    if (backend()) {
        CHECK(backend()->open("Z:\\prosper-missing\\no-such-video.mp4") < 0,
              "missing source fails instead of fabricating a decode session");
        if (argc == 2) {
            const int id = backend()->open(argv[1]);
            CHECK(id >= 0, "dump-backed MP4 opens with an accepted decoder pipeline");
            if (id >= 0) {
                StreamInfo stream{};
                CHECK(backend()->info(id, stream) && stream.width && stream.height && stream.fps > 0,
                      "MP4 stream metadata contains dimensions and frame rate");
                unsigned video_frames = 0, audio_frames = 0;
                uint64_t previous_video_pts = 0, previous_audio_pts = 0;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (std::chrono::steady_clock::now() < deadline && video_frames < 3 &&
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
                CHECK(video_frames >= 3, "Media Foundation delivers multiple real video frames");
                if (stream.has_audio)
                    CHECK(audio_frames > 0, "Media Foundation delivers real audio samples");
                if (stream.has_audio && video_frames >= 3) {
                    // Stop consuming audio and drain only video. A bounded queue may still contain
                    // PCM at stream end, but that must not keep video-only playback active forever.
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
    if (backend()) test_access_unit_path();
    uninstall_media_foundation_backend();
    CHECK(backend() == nullptr, "native backend can be unregistered cleanly");
    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
