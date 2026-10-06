// test_oversize_buffer_window -- a buffer window far past the copy cap is clamped to its MAPPED run, and
// only when the mapping table proves nothing else in the window is mapped.
//
// THE DEFECT. Assassin's Creed Black Flag Resynced reads guest memory through raw buffer V#s whose
// NUM_RECORDS spans an aperture: `0x407ef42800` binds base 0x103dc00000 with a 2 GiB window and
// `0x407eece900` a 1000 MiB window. Prosper refuses any buffer over 256 MiB, so both programs (a
// compute pass and a fragment draw) were skipped. Measured on the live title, the 2 GiB window holds
// exactly one mapped run of 12 MiB at its start and nothing else: everything else in it is unmapped,
// which the GPU cannot touch without faulting.
//
// WHAT EACH TEST KILLS:
//   LeadingRunIsTheWindow              the clamp is wrong, or applies when no clamp is needed
//   IslandAfterTheRunIsRefused         a mapped island past the run is silently turned into zeros
//   RunOverCapIsRefused                an over-cap mapped run is admitted under the cap
//   NothingMappedIsRefused             an unmapped window is published as an empty buffer
//   ClampRoundsDownToWholeRecords      the clamped size splits a record
//   LiveWindowIsCachedUntilMappingsChange  the tail scan reruns every call, or survives a mapping change
//   WholeApertureWindowIsMeasuredInSixtyFourBits  the proof scans the decoder's saturated 4 GiB size
//                                      instead of the V#'s real window, so an island past 4 GiB is
//                                      never probed and the shader's reads there become zeros
//   ResolveClampsProvenMarksAndDropsTheRest  a refused mark reaches a consumer, or a proven one is lost
//   FoldReplayDoesNotReadTheMappingTable  the clamp is decided inside the fold, so a .prfold capture
//                                      replays differently once the mapping table has changed
#include "gpu/capture/fold_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/oversize_buffer_window.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kCap = 256 * kMiB;
constexpr uint64_t kBase = 0x103dc00000ull;

// A table of mapped [begin, end) ranges standing in for the guest mapping registry.
struct FakeMap {
    std::vector<std::pair<uint64_t, uint64_t>> runs;
    uint64_t scans = 0;

    MappedMemoryProbe probe() {
        MappedMemoryProbe p;
        p.readable_prefix = [this](uint64_t addr, uint64_t bytes) {
            uint64_t got = 0;
            for (;;) {
                bool extended = false;
                for (const auto& r : runs)
                    if (addr + got >= r.first && addr + got < r.second) {
                        got = std::min(r.second - addr, bytes);
                        extended = got < bytes;
                        break;
                    }
                if (!extended) break;
            }
            return got;
        };
        p.any_mapped = [this](uint64_t addr, uint64_t bytes) {
            ++scans;
            for (const auto& r : runs)
                if (addr < r.second && r.first < addr + bytes) return true;
            return false;
        };
        return p;
    }
};

}   // namespace

TEST(OversizeBufferWindow, LeadingRunIsTheWindow) {
    FakeMap map{{{kBase, kBase + 12 * kMiB}}};
    uint64_t clamped = 0;
    ASSERT_TRUE(clamp_oversized_buffer_window(kBase, 2048 * kMiB, 4, kCap, map.probe(), clamped));
    EXPECT_EQ(clamped, 12 * kMiB) << "the window is exactly its mapped run";
    uint64_t untouched = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window(kBase, 200 * kMiB, 4, kCap, map.probe(), untouched))
        << "a window already within the cap needs no clamp";
}

TEST(OversizeBufferWindow, IslandAfterTheRunIsRefused) {
    FakeMap map{{{kBase, kBase + 12 * kMiB}, {kBase + 1500 * kMiB, kBase + 1500 * kMiB + 0x4000}}};
    uint64_t clamped = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window(kBase, 2048 * kMiB, 4, kCap, map.probe(), clamped))
        << "a mapped island the shader could read must not become zeros";
}

TEST(OversizeBufferWindow, RunOverCapIsRefused) {
    FakeMap map{{{kBase, kBase + 300 * kMiB}}};
    uint64_t clamped = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window(kBase, 2048 * kMiB, 4, kCap, map.probe(), clamped));
}

TEST(OversizeBufferWindow, NothingMappedIsRefused) {
    FakeMap map;
    uint64_t clamped = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window(kBase, 2048 * kMiB, 4, kCap, map.probe(), clamped));
}

TEST(OversizeBufferWindow, ClampRoundsDownToWholeRecords) {
    FakeMap map{{{kBase, kBase + 12 * kMiB + 0x4000}}};
    uint64_t clamped = 0;
    ASSERT_TRUE(clamp_oversized_buffer_window(kBase, 2048 * kMiB, 48, kCap, map.probe(), clamped));
    EXPECT_EQ(clamped % 48, 0u);
    EXPECT_LE(clamped, 12 * kMiB + 0x4000);
    EXPECT_GT(clamped, 12 * kMiB - 48);
}

TEST(OversizeBufferWindow, LiveWindowIsCachedUntilMappingsChange) {
    // Real mappings: a flexible allocation is mapped at an address the HLE layer picks, and the window
    // declared over it is far larger than the cap.
    register_builtin_hle();
    HleFn flex = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    HleFn unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    ASSERT_NE(flex, nullptr);
    ASSERT_NE(unmap, nullptr);
    uint64_t addr = 0;
    ASSERT_EQ(flex((uint64_t)(uintptr_t)&addr, 0x400000, 3, 0, (uint64_t)(uintptr_t)"w", 0), 0u);
    ASSERT_NE(addr, 0u);

    uint64_t clamped = 0;
    ASSERT_TRUE(clamp_oversized_buffer_window_live(addr, 2048 * kMiB, 4, kCap, clamped))
        << "one mapped run at the base and nothing else is mapped inside the window";
    EXPECT_EQ(clamped, 0x400000u);
    uint64_t again = 0;
    EXPECT_TRUE(clamp_oversized_buffer_window_live(addr, 2048 * kMiB, 4, kCap, again));
    EXPECT_EQ(again, clamped) << "a repeat call answers from the cache";

    // Map a second run INSIDE the window: the table changed, so the proof must be redone and refused.
    uint64_t island = addr + 1024 * kMiB;
    uint64_t placed = island;
    ASSERT_EQ(flex((uint64_t)(uintptr_t)&placed, 0x4000, 3, 0x10, (uint64_t)(uintptr_t)"i", 0), 0u);
    ASSERT_EQ(placed, island)
        << "the island must land inside the window for this arm to mean anything";
    uint64_t after = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window_live(addr, 2048 * kMiB, 4, kCap, after))
        << "a mapping added inside the window invalidates the cached clamp";
    EXPECT_EQ(unmap(placed, 0x4000, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unmap(addr, 0x400000, 0, 0, 0, 0), 0u);
}

namespace {

// A raw V#: Base48, 14-bit stride, NUM_RECORDS, and a word 3 the raw-load path does not interpret.
std::array<uint32_t, 4> raw_v(uint64_t base, uint32_t stride, uint32_t records) {
    return {static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32) | (stride << 16),
            records, 0x00016204u};
}

}   // namespace

TEST(OversizeBufferWindow, WholeApertureWindowIsMeasuredInSixtyFourBits) {
    // NUM_RECORDS 0xFFFFFFFF at stride 16 is a 64 GiB window. decode_buffer_descriptor saturates its
    // size at 0xFFFFFFFF, which would make the island scan stop at base + 4 GiB.
    const auto v = raw_v(kBase, 16, 0xFFFFFFFFu);
    EXPECT_EQ(buffer_descriptor_window_bytes(v.data()), 0xFFFFFFFFull * 16u);

    constexpr uint64_t kGiB = 1024 * kMiB;
    FakeMap island{{{kBase, kBase + 12 * kMiB}, {kBase + 5 * kGiB, kBase + 5 * kGiB + 0x4000}}};
    uint32_t records = 0;
    EXPECT_FALSE(clamp_oversized_buffer_descriptor(v.data(), kCap, island.probe(), records))
        << "an island 5 GiB into a 64 GiB window is memory the shader may read";

    // A window past the scan limit is refused whole: scanning 64 GiB is four million granules.
    FakeMap run_only{{{kBase, kBase + 12 * kMiB}}};
    EXPECT_FALSE(clamp_oversized_buffer_descriptor(v.data(), kCap, run_only.probe(), records));
    EXPECT_EQ(run_only.scans, 0u) << "refused before any granule is probed";

    // Positive control on the same path: a 2 GiB window (stride 4) with only its leading run mapped.
    const auto two_gib = raw_v(kBase, 4, static_cast<uint32_t>(2 * kGiB / 4));
    ASSERT_TRUE(clamp_oversized_buffer_descriptor(two_gib.data(), kCap, run_only.probe(), records));
    EXPECT_EQ(records, 12 * kMiB / 4);
}

TEST(OversizeBufferWindow, ResolveClampsProvenMarksAndDropsTheRest) {
    constexpr uint64_t kOther = kBase + 0x1000000000ull;
    FakeMap map{{{kBase, kBase + 12 * kMiB},
                 {kOther, kOther + 12 * kMiB},
                 {kOther + 1500 * kMiB, kOther + 1500 * kMiB + 0x4000}}};
    std::vector<SrtUse> uses(4);
    uses[0].use_pc = 0;   // an ordinary use before the fold's range: untouched
    uses[1].use_pc = 1;   // proven: leading run only
    uses[1].oversize_window = true;
    const auto proven = raw_v(kBase, 4, static_cast<uint32_t>(2048 * kMiB / 4));
    std::copy(proven.begin(), proven.end(), uses[1].v4.begin());
    uses[2].use_pc = 2;   // refused: an island after the run
    uses[2].oversize_window = true;
    const auto refused = raw_v(kOther, 4, static_cast<uint32_t>(2048 * kMiB / 4));
    std::copy(refused.begin(), refused.end(), uses[2].v4.begin());
    uses[3].use_pc = 3;   // an unmarked use after it keeps its place
    EXPECT_EQ(resolve_oversized_buffer_windows(uses, 1, map.probe()), 1u);
    ASSERT_EQ(uses.size(), 3u);
    EXPECT_EQ(uses[0].use_pc, 0u);
    EXPECT_EQ(uses[1].use_pc, 1u);
    EXPECT_FALSE(uses[1].oversize_window) << "a resolved mark never reaches a consumer";
    EXPECT_EQ(uses[1].v4[2], 12 * kMiB / 4);
    EXPECT_EQ(uses[2].use_pc, 3u);
}

TEST(OversizeBufferWindow, FoldReplayDoesNotReadTheMappingTable) {
    register_builtin_hle();
    HleFn flex = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    HleFn unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    ASSERT_NE(flex, nullptr);
    ASSERT_NE(unmap, nullptr);
    uint64_t addr = 0;
    ASSERT_EQ(flex((uint64_t)(uintptr_t)&addr, 0x400000, 3, 0, (uint64_t)(uintptr_t)"r", 0), 0u);
    ASSERT_NE(addr, 0u);

    // buffer_load_dword v0, off, s[0:3]; s_endpgm -- a raw load through a user-SGPR V# whose 2 GiB
    // window holds one 4 MiB mapped run.
    const auto v = raw_v(addr, 4, static_cast<uint32_t>(2048 * kMiB / 4));
    FoldInputs in;
    in.code = {0xE0300000u, 0x80000000u, 0xBF810000u};
    in.user.assign(v.begin(), v.end());
    in.user_base = 0;
    in.code_address = 0x12340000;
    in.revision = "oversize-window-fixture";
    in.srt_requested = true;

    // The production entry, while the run is mapped: clamped to the run.
    std::vector<SrtUse> live;
    resolve_dynamic_fetch(in.code.data(), in.code.size(), in.user.data(),
                          static_cast<uint32_t>(in.user.size()), 0, &live);
    ASSERT_EQ(live.size(), 1u) << "the mapped run is published";
    EXPECT_FALSE(live[0].oversize_window);
    EXPECT_EQ(live[0].v4[2], 0x400000u / 4);

    // A capture taken now, then replayed after the table changed. The fold must not have read the
    // table, or the replay produces a different SrtUse and strict replay throws.
    const FoldCapture capture = capture_fold(in);
    ASSERT_TRUE(capture.complete);
    ASSERT_EQ(capture.uses.size(), 1u);
    EXPECT_TRUE(capture.uses[0].oversize_window)
        << "the fold marks the window and leaves its V# alone";
    EXPECT_EQ(capture.uses[0].v4[2], v[2]);

    ASSERT_EQ(unmap(addr, 0x400000, 0, 0, 0, 0), 0u);
    EXPECT_NO_THROW(replay_fold(capture));
    EXPECT_NO_THROW(replay_fold(decode_fold_capture(encode_fold_capture(capture))))
        << "the mark survives the .prfold codec";

    // And the production entry, after the unmap: nothing is mapped, so the use is dropped as before.
    std::vector<SrtUse> unmapped;
    resolve_dynamic_fetch(in.code.data(), in.code.size(), in.user.data(),
                          static_cast<uint32_t>(in.user.size()), 0, &unmapped);
    EXPECT_TRUE(unmapped.empty());
}
