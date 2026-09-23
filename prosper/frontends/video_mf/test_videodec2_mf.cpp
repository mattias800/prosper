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

#include "au_test_streams.hpp"
#include "media_foundation_backend.hpp"

#include "hle/dispatch/dispatch.hpp"
#include "hle/video/video_backend.hpp"
#include "hle/video/videodec2_guest_abi.hpp"

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

using namespace prosper::test::vdec;
using namespace prosper::video::au_test;

constexpr uint32_t kW = kVp9Width, kH = kVp9Height;
constexpr uint32_t kCodecVp9 = 2382845;   // what Dragon Quest VII's CreateDecoder passes (#2983)

} // namespace

int main() {
    std::puts("== test_videodec2_mf ==");
    const std::vector<uint8_t> ivf = read_file(PROSPER_TEST_VP9_AU_ASSET);
    const std::vector<Unit> units = split_ivf(ivf);
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
    const bool have_vp9 = host_has_decoder(MFVideoFormat_VP90);
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
        if (pictures <= 24 && h == kVp9Expected[pictures - 1]) ++matched;
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
