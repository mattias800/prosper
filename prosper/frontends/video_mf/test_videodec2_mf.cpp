// test_videodec2_mf (#2983) -- the guest's sceVideodec2 lifecycle, end to end, on Windows' real
// decoder.
//
// test_video_mf checks the Media Foundation access-unit decoder against independent frame hashes,
// calling the backend directly. test_videodec2_decode checks the HLE handlers against a FAKE
// backend. Neither covers the join: that the handlers a title actually calls -- CreateDecoder with
// the codec value Dragon Quest VII passes (2382845), Decode with the guest's own input/frame/output
// structs, DeleteDecoder -- reach THIS backend, and that what lands in the guest's frame buffer is
// the right picture. That join is what #2983 was missing: the HLE was complete and the backend set
// was empty, so every Decode answered NO DECODER.
//
// The access units are the committed libvpx stream, fed in the order a demuxer would, including two
// superframes (hidden alt-ref + shown frame), and every guest picture is hashed against the same
// ffmpeg-derived values test_video_mf uses.
//
// A host without a VP9 decoder (a Windows Server CI image has no VP9 Video Extensions) is a
// supported configuration; there the test asserts the refusal instead, and says so.

#include "media_foundation_backend.hpp"

#include "hle/dispatch/dispatch.hpp"
#include "hle/video/video_backend.hpp"

#include <windows.h>
#include <combaseapi.h>
#include <mfapi.h>
#include <mftransform.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

namespace {

// Mirrors of the guest-facing structs, pinned by size exactly as test_videodec2_decode pins them.
struct VdecConfig {
    uint64_t size; uint32_t resource, codec, profile, max_level;
    int32_t max_width, max_height, max_dpb; uint32_t input_depth;
    uint64_t compute_queue, affinity; int32_t priority;
    uint8_t optimize, check_memory, reserved0, reserved1; uint64_t extra;
};
struct VdecInput { uint64_t size, data, data_size, pts, dts, attached; };
struct VdecFrame { uint64_t size, data, data_size; uint8_t accepted, pad[7]; };
struct VdecOutput {
    uint64_t size; uint8_t valid, error, pictures, discarded;
    uint32_t codec, width, pitch, height; uint64_t frame, frame_size;
    uint32_t format, pitch_bytes;
};
static_assert(sizeof(VdecConfig) == 72, "test VdecConfig must mirror the HLE layout");
static_assert(sizeof(VdecInput) == 48, "test VdecInput must mirror the HLE layout");
static_assert(sizeof(VdecFrame) == 32, "test VdecFrame must mirror the HLE layout");
static_assert(sizeof(VdecOutput) == 56, "test VdecOutput must mirror the HLE layout");

// Same values as test_video_mf.cpp's kVp9Expected: ffmpeg -i vp9_testpattern.ivf -f rawvideo
// -pix_fmt nv12, FNV-1a-64 over each 21600-byte frame.
const uint64_t kExpected[24] = {
    0xc5d881f5ad18d6e3ull, 0x0a104ad23033be37ull, 0x0c671d7cbaffd853ull,
    0xb050c7d781be8167ull, 0x30114c4d0db7d39dull, 0x64bde2949d311befull,
    0x437df2ad43354a19ull, 0x75ba29d1fff0cd40ull, 0x19f7fb26cfc95d38ull,
    0x3a15afbba23fda2eull, 0xbd6bfff70be35829ull, 0x762c1ff5bc183ce6ull,
    0x3218d18e7619cc16ull, 0x5069861f5e6eb052ull, 0xbcb8cbd90ac03d32ull,
    0xfcaab884215e640eull, 0xa8609984e0cd3d83ull, 0x895e7e704f12516aull,
    0xb33fcf120f8f743full, 0x9b769661ad98990bull, 0x0dc8693197c79bcbull,
    0x2ef06dd3c0b3d647ull, 0x4fee04f0e7dbf647ull, 0x82a50de1bd934ad4ull,
};
constexpr uint32_t kW = 160, kH = 90;
constexpr uint32_t kCodecVp9 = 2382845;   // what Dragon Quest VII's CreateDecoder passes (#2983)

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

bool host_has_vp9_decoder() {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    MFT_REGISTER_TYPE_INFO input{MFMediaType_Video, MFVideoFormat_VP90};
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

} // namespace

int main() {
    std::puts("== test_videodec2_mf ==");
    const std::vector<uint8_t> ivf = read_file(PROSPER_TEST_VP9_AU_ASSET);
    std::vector<std::pair<size_t, size_t>> units;
    if (ivf.size() >= 32 && std::memcmp(ivf.data(), "DKIF", 4) == 0) {
        for (size_t off = 32; off + 12 <= ivf.size();) {
            const uint32_t n = ivf[off] | (ivf[off + 1] << 8) | (ivf[off + 2] << 16) |
                               (static_cast<uint32_t>(ivf[off + 3]) << 24);
            if (off + 12 + n > ivf.size()) break;
            units.emplace_back(off + 12, n);
            off += 12 + n;
        }
    }
    CHECK(units.size() == 24, "the committed VP9 stream splits into 24 access units");
    if (units.size() != 24) return 1;

    register_builtin_hle();
    auto alloc_queue   = Hle::lookup("eD+X2SmxUt4");
    auto query_compute = Hle::lookup("RnDibcGCPKw");
    auto query_decoder = Hle::lookup("qqMCwlULR+E");
    auto create        = Hle::lookup("CNNRoRYd8XI");
    auto decode        = Hle::lookup("852F5+q6+iM");
    auto destroy       = Hle::lookup("jwImxXRGSKA");
    CHECK(alloc_queue && query_compute && query_decoder && create && decode && destroy,
          "Videodec2 lifecycle NIDs resolve");
    if (!(alloc_queue && query_compute && query_decoder && create && decode && destroy)) return 1;

    // The REAL backend, installed the way prosper-app, screenshot and boot_trace install it.
    video::set_backend(nullptr);
    CHECK(video::install_media_foundation_backend(), "the Media Foundation backend installs");
    const bool have_vp9 = host_has_vp9_decoder();
    if (!have_vp9)
        std::puts("  [info] this host has no Media Foundation VP9 decoder (VP9 Video Extensions); "
                  "asserting the refusal instead of the decode");

    alignas(256) static uint8_t workspace[4][64u << 10];
    uint64_t compute_info[3] = {24, 0, 0};
    CHECK(query_compute((uint64_t)(uintptr_t)compute_info, 0, 0, 0, 0, 0) == 0,
          "QueryComputeMemoryInfo succeeds");
    compute_info[2] = (uint64_t)(uintptr_t)workspace[3];
    struct { uint64_t size; uint16_t pipe, queue; uint8_t check, r0; uint16_t r1; } cq{16, 0, 0, 0, 0, 0};
    uint64_t compute_queue = 0;
    CHECK(alloc_queue((uint64_t)(uintptr_t)&cq, (uint64_t)(uintptr_t)compute_info,
                      (uint64_t)(uintptr_t)&compute_queue, 0, 0, 0) == 0 && compute_queue,
          "AllocateComputeQueue publishes a queue");

    VdecConfig config{};
    config.size = sizeof config;
    config.resource = 1; config.codec = kCodecVp9; config.profile = 0; config.max_level = 0;
    config.max_width = kW; config.max_height = kH;
    config.max_dpb = -1; config.input_depth = 4; config.compute_queue = compute_queue;
    uint64_t decoder_memory[9] = {72};
    CHECK(query_decoder((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                        0, 0, 0, 0) == 0,
          "QueryDecoderMemoryInfo succeeds for the VP9 config");
    decoder_memory[2] = (uint64_t)(uintptr_t)workspace[0];
    decoder_memory[4] = (uint64_t)(uintptr_t)workspace[1];
    decoder_memory[6] = (uint64_t)(uintptr_t)workspace[2];
    uint64_t handle = 0;
    CHECK(create((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                 (uint64_t)(uintptr_t)&handle, 0, 0, 0) == 0 && handle,
          "CreateDecoder publishes a handle for codec 2382845");

    // The guest sizes its frame buffer from QueryDecoderMemoryInfo's max_frame_size; use it, plus a
    // guard tail the decode must not touch.
    const uint64_t max_frame = decoder_memory[7] ? decoder_memory[7] : video::nv12_bytes(kW, kH);
    const size_t nv12 = static_cast<size_t>(video::nv12_bytes(kW, kH));
    std::vector<uint8_t> guest_frame(static_cast<size_t>(max_frame) + 4096);

    unsigned pictures = 0, matched = 0, valid_geometry = 0;
    bool guard_ok = true;
    for (size_t i = 0; i < units.size(); ++i) {
        std::fill(guest_frame.begin(), guest_frame.end(), 0xCD);
        VdecInput input{sizeof(VdecInput), (uint64_t)(uintptr_t)(ivf.data() + units[i].first),
                        units[i].second, i, i, 0};
        VdecFrame frame{sizeof(VdecFrame), (uint64_t)(uintptr_t)guest_frame.data(), max_frame, 0, {}};
        VdecOutput out{}; out.size = sizeof out;
        if (decode(handle, (uint64_t)(uintptr_t)&input, (uint64_t)(uintptr_t)&frame,
                   (uint64_t)(uintptr_t)&out, 0, 0) != 0) continue;
        if (!(out.valid && out.pictures == 1)) continue;
        ++pictures;
        if (out.width == kW && out.height == kH && out.pitch == kW && frame.accepted == 1 &&
            out.frame == frame.data)
            ++valid_geometry;
        for (size_t k = static_cast<size_t>(max_frame); k < guest_frame.size(); ++k)
            if (guest_frame[k] != 0xCD) { guard_ok = false; break; }
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t k = 0; k < nv12; ++k) { h ^= guest_frame[k]; h *= 0x100000001b3ull; }
        if (pictures <= 24 && h == kExpected[pictures - 1]) ++matched;
    }
    std::printf("  [info] %u pictures from 24 guest access units, %u byte-exact\n", pictures, matched);
    if (have_vp9) {
        CHECK(pictures == 24, "every guest access unit yields its picture (superframes included)");
        CHECK(valid_geometry == 24,
              "every VdecOutput reports 160x90, pitch 160, in the guest's own frame buffer");
        CHECK(matched == 24, "every picture in the guest's buffer is byte-identical to ffmpeg's");
        CHECK(guard_ok, "no picture is written past the guest's max_frame_size");
    } else {
        CHECK(pictures == 0, "with no VP9 decoder installed, Decode reports no picture (refused)");
    }
    CHECK(destroy(handle, 0, 0, 0, 0, 0) == 0, "DeleteDecoder releases the decoder");

    video::uninstall_media_foundation_backend();
    std::puts(fails ? "== FAIL ==" : "== PASS ==");
    return fails ? 1 : 0;
}
