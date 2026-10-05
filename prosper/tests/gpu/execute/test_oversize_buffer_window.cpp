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
#include "gpu/execute/oversize_buffer_window.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

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

}  // namespace

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
    ASSERT_EQ(placed, island) << "the island must land inside the window for this arm to mean anything";
    uint64_t after = 0;
    EXPECT_FALSE(clamp_oversized_buffer_window_live(addr, 2048 * kMiB, 4, kCap, after))
        << "a mapping added inside the window invalidates the cached clamp";
}
