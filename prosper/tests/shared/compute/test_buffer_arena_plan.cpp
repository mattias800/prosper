// The window-arena planner (buffer_arena_plan.hpp / buffer_arena_registry.hpp) without a device.
//
// What these guard, from the review of the first version:
//   * an arena never extends past memory the caller proved readable, even when an EXISTING arena
//     already reached past it (the first version's unsigned subtraction wrapped there);
//   * a reused arena is re-probed, so memory released since its creation is not read;
//   * an isolated window costs nothing: no arena, no guest probe;
//   * an arena's headroom is a fraction of what it covers and it never exceeds twice the largest
//     window it serves, so per-dispatch cost does not grow with a constant;
//   * a binding that aliases a writable binding is never served from an arena.
#include "shared/compute/buffer_arena_registry.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using namespace prosper::frontend;
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kBase = 0x4062000000ull;

// Deterministic generator for the randomized check. Not for any security purpose.
struct SplitMix {
    uint64_t state;
    explicit SplitMix(uint64_t seed) : state(seed) {}
    uint64_t operator()() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

// A guest whose readable memory is [lo, hi); counts how often it is asked.
struct Guest {
    uint64_t lo, hi;
    mutable int probes = 0;
    bool operator()(uint64_t addr, uint32_t bytes) const {
        ++probes;
        return addr >= lo && bytes <= hi - addr && addr + bytes <= hi;
    }
};

TEST(BufferArenaPlan, AnIsolatedWindowIsPrivateAndCostsNothing) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    const BufferArenaDecision d = select_buffer_arena(registry, kBase + 10 * kMiB, 4 * kMiB, 256, guest);
    EXPECT_EQ(d.action, BufferArenaAction::Private);
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_EQ(guest.probes, 0) << "no overlap evidence: nothing is probed";
}

TEST(BufferArenaPlan, AWindowBelowTheMinimumNeverGetsAnArena) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(select_buffer_arena(registry, kBase + i * 256, 512 * 1024, 256, guest).action,
                  BufferArenaAction::Private);
    EXPECT_EQ(registry.size(), 0u);
}

TEST(BufferArenaPlan, AnOverlappingSecondWindowBuildsABoundedArenaAndLaterWindowsReuseIt) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    const uint64_t window = 43 * kMiB;
    EXPECT_EQ(select_buffer_arena(registry, kBase + 0x60b900, window, 256, guest).action,
              BufferArenaAction::Private);
    const BufferArenaDecision d = select_buffer_arena(registry, kBase + 0x60c300, window, 256, guest);
    ASSERT_EQ(d.action, BufferArenaAction::Create);
    EXPECT_TRUE(d.arena.contains(kBase + 0x60b900, window));
    EXPECT_TRUE(d.arena.contains(kBase + 0x60c300, window));
    const uint64_t hull = window + 0xa00;
    EXPECT_LE(d.arena.bytes, hull + hull / 8 + 2 * kBufferArenaChunk)
        << "headroom is a fraction of the hull, not a constant";
    EXPECT_EQ(d.offset, kBase + 0x60c300 - d.arena.base);
    // A third window drifting a little further stays inside the arena.
    const BufferArenaDecision third = select_buffer_arena(registry, kBase + 0x60d100, window, 256, guest);
    EXPECT_EQ(third.action, BufferArenaAction::Reuse);
    EXPECT_EQ(registry.size(), 1u);
}

TEST(BufferArenaPlan, AnArenaNeverLeavesProvenReadableMemoryEvenFromAnExistingOverhang) {
    // The reviewer's scenario: an existing arena already ends past the readable limit when a later
    // window arrives. The first version computed `readable_hi - hi` unsigned there and extended the
    // new arena 6 MiB beyond mapped memory.
    BufferArenaRegistry registry;
    const uint64_t readable_lo = kBase, readable_hi = kBase + 200 * kMiB;
    // An existing arena whose end is 5 MiB past the readable limit (the memory was released since),
    // and a window that overlaps its start but is NOT contained in it.
    BufferArenaDecision seed;
    seed.action = BufferArenaAction::Create;
    seed.arena = {kBase + 100 * kMiB, 105 * kMiB};
    registry.commit(seed);
    const BufferArenaDecision d =
        registry.plan(kBase + 98 * kMiB, 8 * kMiB, readable_lo, readable_hi, 256);
    ASSERT_EQ(d.action, BufferArenaAction::Create) << "the window shares a ring with the overhanging arena";
    EXPECT_GE(d.arena.base, readable_lo);
    EXPECT_LE(d.arena.end(), readable_hi) << "clipped to readable memory, never extended past it";
    EXPECT_TRUE(d.arena.contains(kBase + 98 * kMiB, 8 * kMiB));
    for (const BufferArenaExtent& gone : d.replaces)
        EXPECT_TRUE(d.arena.contains(gone.base, gone.bytes))
            << "only an extent the new arena fully contains may be replaced";
    EXPECT_TRUE(d.replaces.empty()) << "the clipped arena no longer contains the overhanging one";
}

TEST(BufferArenaPlan, EveryPlanStaysReadableContainsItsWindowAndReplacesOnlyWhatItContains) {
    SplitMix rng(12345);
    for (int trial = 0; trial < 3000; ++trial) {
        const uint64_t readable_lo = kBase + (rng() % 64) * 64 * 1024;
        const uint64_t readable_hi = readable_lo + (1 + rng() % 220) * kMiB;
        std::vector<BufferArenaExtent> arenas, recent;
        for (unsigned k = rng() % 4; k > 0; --k)
            arenas.push_back({kBase + (rng() % 250) * kMiB, (1 + rng() % 90) * kMiB});   // may overhang
        for (unsigned k = rng() % 4; k > 0; --k)
            recent.push_back({kBase + (rng() % 250) * kMiB, (1 + rng() % 60) * kMiB});
        const uint64_t bytes = (1 + rng() % 60) * kMiB;
        const uint64_t addr = readable_lo + (rng() % 200) * 64 * 1024;
        const BufferArenaDecision d = plan_buffer_arena(arenas, recent, addr, bytes, readable_lo,
                                                        readable_hi, 256);
        if (d.action != BufferArenaAction::Create) continue;
        ASSERT_GE(d.arena.base, readable_lo) << trial;
        ASSERT_LE(d.arena.end(), readable_hi) << trial;
        ASSERT_TRUE(d.arena.contains(addr, bytes)) << trial;
        ASSERT_EQ(d.offset, addr - d.arena.base) << trial;
        ASSERT_LE(d.arena.bytes, kBufferArenaMaxBytes) << trial;
        for (const BufferArenaExtent& gone : d.replaces)
            ASSERT_TRUE(d.arena.contains(gone.base, gone.bytes)) << trial;
    }
}

TEST(BufferArenaPlan, TwoClustersOfWindowsOfOneRingConvergeToASingleArena) {
    // Black Flag's shape: windows cluster at the ring's start and about 5 MiB further along. The
    // readable range is proven over the whole hull, so the second Create covers both clusters and
    // replaces the first arena instead of leaving two overlapping copies.
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 72 * kMiB};
    const uint64_t window = 43 * kMiB;
    const uint64_t offsets[] = {0x60b900, 0x60c300, 0x60c700, 0x60d100, 0x60d500, 0x609e00,
                                0xb5f600, 0xb00000, 0xb73800};
    int creates = 0;
    for (int frame = 0; frame < 3; ++frame)
        for (uint64_t off : offsets)
            if (select_buffer_arena(registry, kBase + off, window, 256, guest).action ==
                BufferArenaAction::Create) ++creates;
    EXPECT_EQ(registry.size(), 1u) << "one arena serves every window of the ring";
    EXPECT_LE(creates, 2);
    for (uint64_t off : offsets)
        EXPECT_EQ(registry.find(kBase + off, window, 256).action, BufferArenaAction::Reuse);
}

TEST(BufferArenaPlan, AnArenaNeverGrowsPastTwiceTheLargestWindowItServes) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 4096 * kMiB};
    const uint64_t window = 8 * kMiB;
    // Windows sliding by half their size chain together; the hull must stop at twice the window.
    uint64_t largest_arena = 0;
    for (int i = 0; i < 40; ++i) {
        const BufferArenaDecision d =
            select_buffer_arena(registry, kBase + i * (window / 2), window, 256, guest);
        if (d.action != BufferArenaAction::Private) largest_arena = std::max(largest_arena, d.arena.bytes);
    }
    EXPECT_LE(largest_arena, 2 * window + 2 * (window * 2 / 16) + 2 * kBufferArenaChunk);
}

TEST(BufferArenaPlan, ANeighbourThatOnlyTouchesDoesNotShareARing) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    EXPECT_EQ(select_buffer_arena(registry, kBase, 4 * kMiB, 256, guest).action, BufferArenaAction::Private);
    EXPECT_EQ(select_buffer_arena(registry, kBase + 4 * kMiB, 4 * kMiB, 256, guest).action,
              BufferArenaAction::Private);
    EXPECT_EQ(registry.size(), 0u);
}

TEST(BufferArenaPlan, AReusedArenaIsReprobedAndDroppedWhenItsMemoryDisappears) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    const uint64_t window = 8 * kMiB;
    select_buffer_arena(registry, kBase + 50 * kMiB, window, 256, guest);
    const BufferArenaDecision made = select_buffer_arena(registry, kBase + 50 * kMiB + 4096, window, 256, guest);
    ASSERT_EQ(made.action, BufferArenaAction::Create);
    ASSERT_EQ(registry.size(), 1u);
    // The guest releases the memory above the window (a neighbouring allocation freed).
    guest.hi = kBase + 50 * kMiB + window;
    const BufferArenaDecision after = select_buffer_arena(registry, kBase + 50 * kMiB + 8192, window, 256, guest);
    EXPECT_NE(after.action, BufferArenaAction::Reuse) << "the stale arena must not be reused";
    if (after.action == BufferArenaAction::Create) EXPECT_LE(after.arena.end(), guest.hi);
}

TEST(BufferArenaPlan, AnUnalignedOffsetStaysPrivate) {
    BufferArenaRegistry registry;
    Guest guest{kBase, kBase + 200 * kMiB};
    select_buffer_arena(registry, kBase + 50 * kMiB, 8 * kMiB, 256, guest);
    // An offset that is not a multiple of the device alignment must not be bound at an arena offset.
    const BufferArenaDecision d = select_buffer_arena(registry, kBase + 50 * kMiB + 100, 8 * kMiB, 256, guest);
    EXPECT_EQ(d.action, BufferArenaAction::Private);
}

TEST(BufferArenaPlan, CommitReplacesOnlyContainedExtentsAndForgetsServedWindows) {
    BufferArenaRegistry registry;
    registry.remember(kBase, 4 * kMiB);
    EXPECT_EQ(registry.recent_size(), 1u);
    BufferArenaDecision first;
    first.action = BufferArenaAction::Create;
    first.arena = {kBase, 6 * kMiB};
    registry.commit(first);
    EXPECT_EQ(registry.recent_size(), 0u) << "the window the arena now serves is forgotten";
    BufferArenaDecision bigger;
    bigger.action = BufferArenaAction::Create;
    bigger.arena = {kBase - kMiB, 10 * kMiB};
    bigger.replaces = {first.arena};
    registry.commit(bigger);
    EXPECT_EQ(registry.size(), 1u);
}

TEST(BufferArenaPlan, TheRegistryIsBounded) {
    BufferArenaRegistry registry;
    for (uint64_t i = 0; i < 3 * BufferArenaRegistry::kMaxExtents; ++i) {
        BufferArenaDecision d;
        d.action = BufferArenaAction::Create;
        d.arena = {kBase + i * 100 * kMiB, 2 * kMiB};
        registry.commit(d);
        registry.remember(kBase + i * 100 * kMiB, 2 * kMiB);
    }
    EXPECT_LE(registry.size(), BufferArenaRegistry::kMaxExtents);
    EXPECT_LE(registry.recent_size(), BufferArenaRegistry::kMaxRecentWindows);
}

// ---- aliasing: a writable alias forbids the arena -----------------------------------------------

TEST(BufferAliasGroup, AReadOnlyBindingAliasingALaterWritableOneIsFlagged) {
    // The review's corruption scenario: the same V# bound read-only first and writable second. The
    // loop that discovers aliases one binding at a time would have turned the owner writable AFTER its
    // arena was chosen; the pre-pass answers it up front.
    const std::vector<BufferAliasKey> bindings = {
        {0x1000, 2 * kMiB, 0, 0, false},   // read-only owner
        {0x1000, 2 * kMiB, 0, 0, true},    // same range, writable
        {0x9000, 2 * kMiB, 0, 0, false},   // unrelated
    };
    EXPECT_TRUE(buffer_alias_group_has_writer(bindings, 0));
    EXPECT_TRUE(buffer_alias_group_has_writer(bindings, 1));
    EXPECT_FALSE(buffer_alias_group_has_writer(bindings, 2));
}

TEST(BufferAliasGroup, DifferentSizeOrHostBackingIsNotAnAlias) {
    const std::vector<BufferAliasKey> bindings = {
        {0x1000, 2 * kMiB, 0, 0, false},
        {0x1000, 3 * kMiB, 0, 0, true},      // same base, different size: not the same range
        {0x1000, 2 * kMiB, 0x77, 8, true},   // host-backed copy of the same address
    };
    EXPECT_FALSE(buffer_alias_group_has_writer(bindings, 0));
}

// ---- cache key ------------------------------------------------------------------------------------

struct FakeShape { uint64_t logical_bytes = 0, binding_bytes = 0; };
struct FakeKey { uint64_t gpu_addr = 0; uintptr_t host_data = 0; uint32_t bytes = 0; FakeShape materialization; };

TEST(BufferArenaPlan, TheArenaKeyIsTheExtentWithItsShapeResized) {
    FakeKey window{kBase + 100, 0x55, 4096, {4096, 4096}};
    const FakeKey key = buffer_arena_cache_key(window, BufferArenaExtent{kBase, 10 * kMiB});
    EXPECT_EQ(key.gpu_addr, kBase);
    EXPECT_EQ(key.host_data, 0u);
    EXPECT_EQ(key.bytes, 10 * kMiB);
    EXPECT_EQ(key.materialization.logical_bytes, 10 * kMiB);
    EXPECT_EQ(key.materialization.binding_bytes, 10 * kMiB);
}

}  // namespace
