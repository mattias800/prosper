// test_buffer_arena_plan -- overlapping windows of a read-only buffer share one resident arena.
//
// THE DEFECT. The compute buffer cache was keyed by a V#'s own start address. Black Flag's compute
// passes bind a 43 MiB constant-ring window whose base advances a few hundred bytes per dispatch, so
// nine near-identical copies (~390 MiB) overflowed the 512 MiB cache: every acquisition missed and
// each miss compared and re-uploaded ~43 MiB (ADR 0010, PERF-P9).
//
// WHAT EACH TEST KILLS:
//   FirstWindowCreatesAnArenaWithSlack     no arena is built, or it has no headroom for drift
//   DriftingWindowsReuseTheArena           a window inside the arena gets its own copy again
//   MeasuredBlackFlagWindowsShareOneArena  the real nine bases would not fit one arena
//   WindowBeyondTheArenaGrowsIt            a window past the arena is made private or a duplicate
//   ArenaNeverLeavesReadableMemory         slack or rounding reaches memory not proven readable
//   SmallAndMisalignedWindowsStayPrivate   a cheap window is shared, or an offset the device cannot bind
//   RegistryFindNeedsNoGuestProbe          the reuse lookup needs readable-range knowledge it cannot have
//   RegistryCommitReplacesSubsumedArenas   a grown arena leaves its predecessor registered (two copies)
//   RegistryIsBounded                      stale arenas accumulate without limit
#include "shared/compute/buffer_arena_registry.hpp"

#include <gtest/gtest.h>

using namespace prosper::frontend;

namespace {
constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kWindow = 45088768;   // 43 MiB, the measured V# span
constexpr uint64_t kAlign = 256;
// Readable memory around the measured ring: generous on both sides.
constexpr uint64_t kLo = 0x4060000000ull, kHi = 0x4068000000ull;
}  // namespace

TEST(BufferArenaPlan, FirstWindowCreatesAnArenaWithSlack) {
    const auto d = plan_buffer_arena({}, 0x406260b900ull, kWindow, kLo, kHi, kAlign);
    ASSERT_EQ(d.action, BufferArenaAction::Create);
    EXPECT_TRUE(d.arena.contains(0x406260b900ull, kWindow));
    EXPECT_GE(d.arena.bytes, kWindow + 2 * kBufferArenaSlackBytes - 2 * kBufferArenaChunk)
        << "headroom on both sides lets nearby windows land inside";
    EXPECT_EQ(d.arena.base % kBufferArenaChunk, 0u);
    EXPECT_EQ(d.arena.base + d.offset, 0x406260b900ull);
    EXPECT_TRUE(d.replaces.empty());
}

TEST(BufferArenaPlan, DriftingWindowsReuseTheArena) {
    const auto first = plan_buffer_arena({}, 0x406260b900ull, kWindow, kLo, kHi, kAlign);
    for (uint64_t shift : {0x400ull, 0xa00ull, 0x1800ull}) {
        const auto d = plan_buffer_arena({first.arena}, 0x406260b900ull + shift, kWindow, kLo, kHi,
                                         kAlign);
        ASSERT_EQ(d.action, BufferArenaAction::Reuse) << "shift " << shift;
        EXPECT_EQ(d.arena.base, first.arena.base);
        EXPECT_EQ(d.arena.base + d.offset, 0x406260b900ull + shift);
    }
}

TEST(BufferArenaPlan, MeasuredBlackFlagWindowsShareOneArena) {
    // The nine bases seen on the title (one dispatch each per frame), in first-seen order.
    const uint64_t bases[] = {0x406260b900ull, 0x406260c300ull, 0x406260c700ull, 0x406260d100ull,
                              0x406260d500ull, 0x4062609e00ull, 0x4062b5f600ull, 0x4062b00000ull,
                              0x4062b73800ull};
    std::vector<BufferArenaExtent> arenas;
    unsigned created = 0;
    for (uint64_t base : bases) {
        const auto d = plan_buffer_arena(arenas, base, kWindow, kLo, kHi, kAlign);
        ASSERT_NE(d.action, BufferArenaAction::Private) << std::hex << base;
        if (d.action == BufferArenaAction::Create) {
            ++created;
            for (const auto& gone : d.replaces)
                arenas.erase(std::remove_if(arenas.begin(), arenas.end(),
                                            [&](const BufferArenaExtent& a) {
                                                return a.base == gone.base && a.bytes == gone.bytes;
                                            }),
                             arenas.end());
            arenas.push_back(d.arena);
        }
    }
    EXPECT_LE(created, 2u) << "nine windows must not cost nine resident copies";
    EXPECT_EQ(arenas.size(), 1u);
}

TEST(BufferArenaPlan, WindowBeyondTheArenaGrowsIt) {
    const uint64_t hi = kHi + 64 * kMiB;   // room for a window that ends past the first arena
    const auto first = plan_buffer_arena({}, 0x406260b900ull, kWindow, kLo, hi, kAlign);
    const uint64_t far = first.arena.end() - 2 * kMiB;   // overlaps the arena, ends past it
    const auto d = plan_buffer_arena({first.arena}, far, kWindow, kLo, hi, kAlign);
    ASSERT_EQ(d.action, BufferArenaAction::Create);
    ASSERT_EQ(d.replaces.size(), 1u) << "the old arena is subsumed, not duplicated";
    EXPECT_TRUE(d.arena.contains(first.arena.base, first.arena.bytes));
    EXPECT_TRUE(d.arena.contains(far, kWindow));
}

TEST(BufferArenaPlan, ArenaNeverLeavesReadableMemory) {
    // Readable memory ends 3 MiB after the window and starts 1 MiB before it.
    const uint64_t addr = 0x4061100000ull;
    const uint64_t lo = addr - kMiB, hi = addr + kWindow + 3 * kMiB;
    const auto d = plan_buffer_arena({}, addr, kWindow, lo, hi, kAlign);
    ASSERT_EQ(d.action, BufferArenaAction::Create);
    EXPECT_GE(d.arena.base, lo);
    EXPECT_LE(d.arena.end(), hi);
    EXPECT_TRUE(d.arena.contains(addr, kWindow));
}

TEST(BufferArenaPlan, SmallAndMisalignedWindowsStayPrivate) {
    EXPECT_EQ(plan_buffer_arena({}, 0x406260b900ull, kMiB - 1, kLo, kHi, kAlign).action,
              BufferArenaAction::Private) << "a cheap window keeps its own entry";
    // A window whose offset inside the arena is not a multiple of the device alignment cannot be
    // bound as an offset view.
    const auto d = plan_buffer_arena({}, 0x406260b910ull, kWindow, kLo, kHi, kAlign);
    EXPECT_EQ(d.action, BufferArenaAction::Private);
    EXPECT_EQ(plan_buffer_arena({}, 0x4000ull, kWindow, kLo, kHi, kAlign).action,
              BufferArenaAction::Private) << "a window outside the readable range is refused";
}

TEST(BufferArenaRegistry, RegistryFindNeedsNoGuestProbe) {
    BufferArenaRegistry registry;
    EXPECT_EQ(registry.find(0x406260b900ull, kWindow, kAlign).action, BufferArenaAction::Private)
        << "an empty registry knows no arena";
    registry.commit(registry.plan(0x406260b900ull, kWindow, kLo, kHi, kAlign));
    const auto d = registry.find(0x406260c300ull, kWindow, kAlign);
    ASSERT_EQ(d.action, BufferArenaAction::Reuse);
    EXPECT_EQ(d.arena.base + d.offset, 0x406260c300ull);
    EXPECT_EQ(registry.find(0x406260c310ull, kWindow, kAlign).action, BufferArenaAction::Private)
        << "an offset the device cannot bind falls back to a private copy";
    EXPECT_EQ(registry.find(0x406260c300ull, kMiB - 1, kAlign).action, BufferArenaAction::Private);
}

TEST(BufferArenaRegistry, RegistryCommitReplacesSubsumedArenas) {
    BufferArenaRegistry registry;
    const uint64_t hi = kHi + 64 * kMiB;
    registry.commit(registry.plan(0x406260b900ull, kWindow, kLo, hi, kAlign));
    ASSERT_EQ(registry.size(), 1u);
    const auto first = registry.find(0x406260b900ull, kWindow, kAlign).arena;
    // A window that ends past the arena grows it; the registry keeps ONE extent.
    const auto grown = registry.plan(first.end() - 2 * kMiB, kWindow, kLo, hi, kAlign);
    ASSERT_EQ(grown.action, BufferArenaAction::Create);
    registry.commit(grown);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.find(0x406260b900ull, kWindow, kAlign).arena.base, grown.arena.base)
        << "windows of the old arena now resolve to the grown one";
}

TEST(BufferArenaRegistry, RegistryIsBounded) {
    BufferArenaRegistry registry;
    for (uint64_t i = 0; i < BufferArenaRegistry::kMaxExtents + 20; ++i) {
        // Disjoint windows: each needs its own arena, 1 GiB apart.
        const uint64_t addr = 0x1000000000ull + i * (1ull << 30);
        registry.commit(registry.plan(addr, 2 * kMiB, addr - 16 * kMiB, addr + 18 * kMiB, kAlign));
    }
    EXPECT_EQ(registry.size(), BufferArenaRegistry::kMaxExtents);
    EXPECT_EQ(registry.find(0x1000000000ull, 2 * kMiB, kAlign).action, BufferArenaAction::Private)
        << "the oldest extent was dropped";
}
