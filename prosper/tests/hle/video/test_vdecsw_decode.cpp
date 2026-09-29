// test_vdecsw_decode (#3669) — libSceVdecsw software video decoder test.
//
// libSceVdecsw is the asynchronous software video decoder used by titles like Uncharted (PPSA05684).
// It shares set-up and backend decode plumbing with libSceVideodec2, but introduces:
//   1. An expanded 0x50-byte VdecswConfig struct (vs 0x48 in Videodec2), where reserved1 (+0x3f)
//      can be set to 1 by guest and must be adapted/zeroed for the shared validator.
//   2. An asynchronous FIFO queue for staging access units via sceVdecswSetDecodeInput. Staged units
//      must be copied at staging time because guest can immediately reuse the AU buffer.
//   3. Separate output frame registration via sceVdecswSetDecodeOutput (0x18 struct).
//   4. Input synchronization via sceVdecswTrySyncDecodeInput (0x18 struct), which zeroes the two
//      trailing qwords to prevent stack residue leakage to the guest.
//   5. Output polling via sceVdecswTrySyncDecodeOutput (0x38 struct), which consumes exactly one
//      staged AU per decode and places the decoded pictureCount at byte offset +0x0B (where Videodec2
//      has `discarded`). Polling when no AUs are staged returns "no picture" without re-decoding.
//   6. Clean lifecycle management: reset clears queued inputs/output configuration, and delete
//      destroys the decoder handle and clears all pending state in g_vdecsw.

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/video/video_backend.hpp"
#include "hle/video/videodec2_guest_abi.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

using namespace prosper::test::vdec;

class MockVdecBackend final : public video::VideoBackend {
public:
    static constexpr uint32_t kWidth = 64;
    static constexpr uint32_t kHeight = 32;

    int  open(const std::string&) override { return -1; }
    bool info(int, video::StreamInfo&) override { return false; }
    bool next_video(int, video::VideoFrame&) override { return false; }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int) override { return true; }
    void close(int) override {}

    int open_decoder(uint32_t codec) override {
        ++opens;
        last_codec = codec;
        return next_id++;
    }

    AuResult decode_au(int id, const uint8_t* au, size_t bytes,
                       uint8_t* dst, uint64_t dst_bytes, AuPicture& out) override {
        ++decodes;
        last_id = id;
        last_au_bytes = bytes;
        last_au_first = (au && bytes) ? au[0] : 0;
        decoded_tags.push_back(bytes >= 2 ? static_cast<uint16_t>(au[0] | (au[1] << 8)) :
                                       static_cast<uint16_t>(last_au_first));

        out.width = kWidth;
        out.height = kHeight;
        out.y_stride = kWidth;
        out.uv_stride = kWidth;
        const uint64_t y_bytes = static_cast<uint64_t>(kWidth) * kHeight;
        out.nv12_bytes = y_bytes + y_bytes / 2;
        if (!dst || dst_bytes < out.nv12_bytes) return AuResult::FrameTooSmall;

        for (uint64_t i = 0; i < y_bytes; ++i)
            dst[i] = static_cast<uint8_t>((last_au_first + i) & 0xFF);
        for (uint64_t i = 0; i < y_bytes / 2; ++i)
            dst[y_bytes + i] = static_cast<uint8_t>((last_au_first * 3u + i * 7u) & 0xFF);

        return AuResult::Decoded;
    }

    bool reset_decoder(int id) override {
        ++resets;
        last_reset = id;
        return true;
    }

    void close_decoder(int id) override {
        ++closes;
        last_closed = id;
    }

    int next_id = 1;
    int opens = 0, decodes = 0, closes = 0, resets = 0;
    int last_id = -1, last_closed = -1, last_reset = -1;
    uint32_t last_codec = 0;
    size_t last_au_bytes = 0;
    uint8_t last_au_first = 0;
    std::vector<uint16_t> decoded_tags;
};

// Error constants matching videodec2.cpp
constexpr uint64_t kVdecErrStruct    = 0x811d0101ull;
constexpr uint64_t kVdecErrArg       = 0x811d0102ull;
constexpr uint64_t kVdecErrDecoder   = 0x811d0103ull;
constexpr uint64_t kVdecErrFrameSize = 0x811d0106ull;
constexpr uint64_t kVdecErrFramePtr  = 0x811d0107ull;
constexpr uint64_t kVdecErrInputDepth = 0x811d0206ull;

} // namespace

int main() {
    printf("== test_vdecsw_decode ==\n");
    register_builtin_hle();

    auto query_compute  = Hle::lookup("0moTubWCsTM");
    auto alloc_queue    = Hle::lookup("hIgrg5h4V6s");
    auto query_decoder  = Hle::lookup("A+2M7EivuOU");
    auto create         = Hle::lookup("+L5ArV1tPGA");
    auto set_input      = Hle::lookup("aqMiF0AgUYI");
    auto set_output     = Hle::lookup("rgtMCOpyBSc");
    auto sync_input     = Hle::lookup("l4sQYy5wPkc");
    auto sync_output    = Hle::lookup("kMBw37oH8nI");
    auto destroy        = Hle::lookup("jwImxXRGSKA");
    auto reset          = Hle::lookup("wJXikG6QFN8");

    CHECK(query_compute && alloc_queue && query_decoder && create &&
          set_input && set_output && sync_input && sync_output && destroy && reset,
          "all 8 libSceVdecsw NIDs and lifecycle handlers resolve");
    if (!(query_compute && alloc_queue && query_decoder && create &&
          set_input && set_output && sync_input && sync_output && destroy && reset))
        return 1;

    MockVdecBackend mock;
    video::set_backend(&mock);

    alignas(256) static uint8_t workspace[4][64u << 10];

    // ---- 1. Compute Memory & Compute Queue Setup -----------------------------------------------
    uint64_t compute_info[3] = {24, 0, 0};
    CHECK(query_compute((uint64_t)(uintptr_t)compute_info, 0, 0, 0, 0, 0) == 0,
          "QueryComputeMemoryInfo succeeds with 0x18 struct");
    compute_info[2] = (uint64_t)(uintptr_t)workspace[3];

    struct { uint64_t size; uint16_t pipe, queue; uint8_t check, r0; uint16_t r1; } cq{16, 0, 0, 0, 0, 0};
    uint64_t compute_queue = 0;
    CHECK(alloc_queue((uint64_t)(uintptr_t)&cq, (uint64_t)(uintptr_t)compute_info,
                      (uint64_t)(uintptr_t)&compute_queue, 0, 0, 0) == 0 && compute_queue != 0,
          "AllocateComputeQueue publishes valid compute queue handle");

    // ---- 2. Decoder Config Adaptation & Validation (0x50 bytes) --------------------------------
    VdecswConfig config{};
    config.base.size = sizeof(VdecswConfig); // 0x50 == 80 bytes
    config.base.resource = 1;
    config.base.codec = 1;      // AVC
    config.base.profile = 100;  // High profile
    config.base.max_level = 52;
    config.base.max_width = MockVdecBackend::kWidth;
    config.base.max_height = MockVdecBackend::kHeight;
    config.base.max_dpb = 4;
    config.base.input_depth = 3;
    config.base.compute_queue = compute_queue;
    config.base.affinity = 0x3f;
    config.base.priority = -1;
    config.base.optimize = 1;
    config.base.reserved1 = 1;  // Vdecsw-specific byte at +0x3f; must be adapted & zeroed
    config.extra2 = 0;

    // Wrong size rejected
    VdecswConfig bad_cfg = config;
    bad_cfg.base.size = sizeof(VdecConfig); // 0x48 instead of 0x50
    CHECK(query_decoder((uint64_t)(uintptr_t)&bad_cfg, (uint64_t)(uintptr_t)workspace[0], 0, 0, 0, 0) == kVdecErrStruct,
          "QueryDecoderMemoryInfo rejects non-0x50 config size");

    uint64_t decoder_memory[9] = {72};
    CHECK(query_decoder((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                        0, 0, 0, 0) == 0,
          "QueryDecoderMemoryInfo succeeds with 0x50 config and adapts reserved1");

    decoder_memory[2] = (uint64_t)(uintptr_t)workspace[0];
    decoder_memory[4] = (uint64_t)(uintptr_t)workspace[1];
    decoder_memory[6] = (uint64_t)(uintptr_t)workspace[2];

    uint64_t handle = 0;
    CHECK(create((uint64_t)(uintptr_t)&config, (uint64_t)(uintptr_t)decoder_memory,
                 (uint64_t)(uintptr_t)&handle, 0, 0, 0) == 0 && handle != 0,
          "CreateDecoder succeeds with 0x50 config and returns valid handle");

    // ---- 3. TrySyncDecodeInput Zeroing & Validation --------------------------------------------
    VdecswSyncInput sync_in{sizeof(VdecswSyncInput), 0xDEADBEEFCAFEBABEull, 0x1234567890ABCDEFull};
    CHECK(sync_input(handle, (uint64_t)(uintptr_t)&sync_in, 0, 0, 0, 0) == 0,
          "TrySyncDecodeInput succeeds");
    CHECK(sync_in.rsvd0 == 0 && sync_in.rsvd1 == 0,
          "TrySyncDecodeInput zeroes trailing qwords to prevent stack residue (#2951)");

    VdecswSyncInput bad_sync{16, 0, 0};
    CHECK(sync_input(handle, (uint64_t)(uintptr_t)&bad_sync, 0, 0, 0, 0) == kVdecErrStruct,
          "TrySyncDecodeInput rejects bad struct size");
    CHECK(sync_input(handle + 0x99999, (uint64_t)(uintptr_t)&sync_in, 0, 0, 0, 0) == kVdecErrDecoder,
          "TrySyncDecodeInput rejects unknown decoder handle");

    // ---- 4. AU Input Staging (FIFO Queue & Staging-time Copy) -----------------------------------
    std::vector<uint8_t> au1(100, 0x11);
    std::vector<uint8_t> au2(200, 0x22);
    std::vector<uint8_t> au3(300, 0x33);

    VdecInput in1{sizeof(VdecInput), (uint64_t)(uintptr_t)au1.data(), au1.size(), 1000, 0, 0};
    VdecInput in2{sizeof(VdecInput), (uint64_t)(uintptr_t)au2.data(), au2.size(), 2000, 0, 0};
    VdecInput in3{sizeof(VdecInput), (uint64_t)(uintptr_t)au3.data(), au3.size(), 3000, 0, 0};

    CHECK(set_input(handle, (uint64_t)(uintptr_t)&in1, 0, 0, 0, 0) == 0, "SetDecodeInput queues AU 1");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&in2, 0, 0, 0, 0) == 0, "SetDecodeInput queues AU 2");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&in3, 0, 0, 0, 0) == 0, "SetDecodeInput queues AU 3");

    // Immediately corrupt/overwrite the guest's source AU buffers to prove staging-time copy
    std::fill(au1.begin(), au1.end(), 0xFF);
    std::fill(au2.begin(), au2.end(), 0xFF);
    std::fill(au3.begin(), au3.end(), 0xFF);

    VdecInput bad_in{32, (uint64_t)(uintptr_t)au1.data(), au1.size(), 0, 0, 0};
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&bad_in, 0, 0, 0, 0) == kVdecErrStruct,
          "SetDecodeInput rejects bad input struct size");
    CHECK(set_input(handle + 0x99999, (uint64_t)(uintptr_t)&in1, 0, 0, 0, 0) == kVdecErrDecoder,
          "SetDecodeInput rejects unknown decoder handle");

    // ---- 5. Output Frame Setup & Validation ----------------------------------------------------
    const size_t y_bytes = static_cast<size_t>(MockVdecBackend::kWidth) * MockVdecBackend::kHeight;
    const size_t nv12_bytes = y_bytes + y_bytes / 2;
    std::vector<uint8_t> guest_frame(nv12_bytes + 256, 0xAA);

    VdecswOutput out{};
    out.size = sizeof(VdecswOutput);

    // Calling TrySyncDecodeOutput before SetDecodeOutput must fail
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == kVdecErrFramePtr,
          "TrySyncDecodeOutput fails with VDEC_ERR_FRAME_PTR when output frame not set");

    VdecswFrame frame{sizeof(VdecswFrame), (uint64_t)(uintptr_t)guest_frame.data(), nv12_bytes};
    CHECK(set_output(handle, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0,
          "SetDecodeOutput succeeds");

    VdecswFrame bad_frame{sizeof(VdecswFrame), 0, nv12_bytes};
    CHECK(set_output(handle, (uint64_t)(uintptr_t)&bad_frame, 0, 0, 0, 0) == kVdecErrFramePtr,
          "SetDecodeOutput rejects null data pointer");
    CHECK(set_output(handle + 0x99999, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == kVdecErrDecoder,
          "SetDecodeOutput rejects unknown decoder handle");

    // ---- 6. Output Decoding, FIFO Consumption & pictureCount (+0x0B) ---------------------------
    // Poll 1: Consumes AU 1 (tag 0x11, 100 bytes)
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0,
          "TrySyncDecodeOutput succeeds for AU 1");
    CHECK(mock.decodes == 1 && mock.last_au_bytes == 100 && mock.last_au_first == 0x11,
          "AU 1 was decoded with original copied data (not corrupted by guest overwrite)");
    CHECK(out.valid == 1 && out.pictures == 1 && out.pictureCount == 1,
          "AU 1 sets pictureCount == 1 at byte offset +0x0B (#3669)");
    CHECK(guest_frame[0] == 0x11, "Luma frame buffer contains decoded picture from AU 1");

    // Poll 2: Consumes AU 2 (tag 0x22, 200 bytes)
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0,
          "TrySyncDecodeOutput succeeds for AU 2");
    CHECK(mock.decodes == 2 && mock.last_au_bytes == 200 && mock.last_au_first == 0x22,
          "AU 2 was decoded second (FIFO ordering preserved)");
    CHECK(out.valid == 1 && out.pictures == 1 && out.pictureCount == 1,
          "AU 2 sets pictureCount == 1 at byte offset +0x0B");
    CHECK(guest_frame[0] == 0x22, "Luma frame buffer contains decoded picture from AU 2");

    // Poll 3: Consumes AU 3 (tag 0x33, 300 bytes)
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0,
          "TrySyncDecodeOutput succeeds for AU 3");
    CHECK(mock.decodes == 3 && mock.last_au_bytes == 300 && mock.last_au_first == 0x33,
          "AU 3 was decoded third (FIFO ordering preserved)");
    CHECK(out.valid == 1 && out.pictures == 1 && out.pictureCount == 1,
          "AU 3 sets pictureCount == 1 at byte offset +0x0B");

    // Poll 4: Input queue is now empty — single consumption test
    const int decodes_before = mock.decodes;
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0,
          "TrySyncDecodeOutput succeeds on empty queue");
    CHECK(mock.decodes == decodes_before,
          "Polling empty queue does NOT re-decode previous access unit (single-consumption)");
    CHECK(out.valid == 0 && out.pictures == 0 && out.pictureCount == 0,
          "Empty queue reports valid=0, pictures=0, pictureCount=0 at +0x0B");

    // ---- 7. Reset Lifecycle & Discard Semantics ------------------------------------------------
    std::vector<uint8_t> au4(150, 0x44);
    VdecInput in4{sizeof(VdecInput), (uint64_t)(uintptr_t)au4.data(), au4.size(), 4000, 0, 0};
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&in4, 0, 0, 0, 0) == 0, "Queue AU 4 before Reset");

    CHECK(reset(handle, 0, 0, 0, 0, 0) == 0, "Reset succeeds");
    CHECK(mock.resets == 1, "Reset flushed video backend decoder");

    // Reset clears output state and queued inputs
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == kVdecErrFramePtr,
          "Reset cleared output frame registration (returns VDEC_ERR_FRAME_PTR)");

    // Re-register output frame and verify queued AU 4 was discarded by Reset
    CHECK(set_output(handle, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0,
          "Re-set output frame after Reset");
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0,
          "TrySyncDecodeOutput succeeds after Reset and output re-registration");
    CHECK(out.pictures == 0 && out.pictureCount == 0,
          "Staged AU 4 was discarded on Reset (no pictures decoded)");

    // A real title stages more than 64 AUs before its first output poll. Preserve every accepted
    // unit, including a possible SPS/PPS at the head, and retain their order through decoding.
    std::vector<std::vector<uint8_t>> burst(512, std::vector<uint8_t>(4));
    std::vector<uint16_t> expected_tags;
    bool burst_accepted = true;
    for (size_t i = 0; i < burst.size(); ++i) {
        burst[i][0] = i ? static_cast<uint8_t>(i) : 0x67;
        burst[i][1] = i ? static_cast<uint8_t>(i >> 8) : 0xFF;
        expected_tags.push_back(static_cast<uint16_t>(burst[i][0] | (burst[i][1] << 8)));
        VdecInput item{sizeof(VdecInput), (uint64_t)(uintptr_t)burst[i].data(),
                       burst[i].size(), i, 0, 0};
        burst_accepted &= set_input(handle, (uint64_t)(uintptr_t)&item, 0, 0, 0, 0) == 0;
    }
    CHECK(burst_accepted, "a burst above 64 access units is accepted without loss");
    const size_t decoded_before_burst = mock.decoded_tags.size();
    bool burst_drained = true;
    for (size_t i = 0; i < burst.size(); ++i)
        burst_drained &= sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0;
    CHECK(burst_drained && mock.decoded_tags.size() == decoded_before_burst + burst.size(),
          "each accepted burst access unit is decoded exactly once");
    bool fifo = mock.decoded_tags.size() == decoded_before_burst + burst.size();
    for (size_t i = 0; fifo && i < burst.size(); ++i)
        fifo &= mock.decoded_tags[decoded_before_burst + i] == expected_tags[i];
    CHECK(fifo, "burst decoding preserves independent access-unit order");

    // Tiny AUs must not bypass the count guard. Refusal leaves the original head intact, and
    // consuming one unit makes room for exactly one new unit.
    VdecInput empty_in{sizeof(VdecInput), 0, 0, 0, 0, 0};
    bool count_filled = true;
    for (int i = 0; i < 4096; ++i)
        count_filled &= set_input(handle, (uint64_t)(uintptr_t)&empty_in, 0, 0, 0, 0) == 0;
    CHECK(count_filled, "count guard accepts 4096 tiny access units");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&empty_in, 0, 0, 0, 0) == kVdecErrInputDepth,
          "count guard refuses the next unit without dropping queued input");
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0 &&
          mock.last_au_bytes == 0, "first queued tiny access unit remains next after refusal");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&empty_in, 0, 0, 0, 0) == 0,
          "count guard admits again after one output poll");

    CHECK(reset(handle, 0, 0, 0, 0, 0) == 0, "Reset discards count-limited backlog");
    CHECK(set_output(handle, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0,
          "output registration succeeds after backlog reset");
    std::vector<uint8_t> large_au(32u * 1024 * 1024, 0x81);
    std::vector<uint8_t> refused_au(1, 0xEE), recovered_au(1, 0xA5);
    VdecInput large_in{sizeof(VdecInput), (uint64_t)(uintptr_t)large_au.data(), large_au.size(), 0, 0, 0};
    VdecInput refused_in{sizeof(VdecInput), (uint64_t)(uintptr_t)refused_au.data(), refused_au.size(), 0, 0, 0};
    VdecInput recovered_in{sizeof(VdecInput), (uint64_t)(uintptr_t)recovered_au.data(), recovered_au.size(), 0, 0, 0};
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&large_in, 0, 0, 0, 0) == 0 &&
          set_input(handle, (uint64_t)(uintptr_t)&large_in, 0, 0, 0, 0) == 0,
          "byte budget accepts 64 MiB of staged input");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&refused_in, 0, 0, 0, 0) == kVdecErrInputDepth,
          "byte budget refuses additional input without evicting queued units");
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0 &&
          mock.last_au_first == 0x81, "byte-budget refusal retains the first queued unit");
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&recovered_in, 0, 0, 0, 0) == 0,
          "byte budget admits again after consumption");
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0 &&
          mock.last_au_first == 0x81 &&
          sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == 0 &&
          mock.last_au_first == 0xA5,
          "refused bytes never enter the queue and recovered bytes retain FIFO order");

    // ---- 8. DeleteDecoder Teardown & Cleanup ----------------------------------------------------
    CHECK(destroy(handle, 0, 0, 0, 0, 0) == 0, "DeleteDecoder succeeds");
    CHECK(mock.closes == 1 && mock.last_closed == mock.last_id,
          "DeleteDecoder closes video backend decoder");

    // Deleted handle must be rejected by all entry points
    CHECK(set_input(handle, (uint64_t)(uintptr_t)&in1, 0, 0, 0, 0) == kVdecErrDecoder,
          "SetDecodeInput rejects deleted decoder handle");
    CHECK(set_output(handle, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == kVdecErrDecoder,
          "SetDecodeOutput rejects deleted decoder handle");
    CHECK(sync_input(handle, (uint64_t)(uintptr_t)&sync_in, 0, 0, 0, 0) == kVdecErrDecoder,
          "TrySyncDecodeInput rejects deleted decoder handle");
    CHECK(sync_output(handle, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0) == kVdecErrDecoder,
          "TrySyncDecodeOutput rejects deleted decoder handle");

    video::set_backend(nullptr);
    printf(fails ? "FAILED (%d)\n" : "PASSED (%d failures)\n", fails);
    return fails ? 1 : 0;
}
