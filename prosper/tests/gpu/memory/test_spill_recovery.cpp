// spill_recovery — when a retained resource that spilled to system memory may be evicted so it is
// re-created device-local (#3905). Pure arithmetic over hand-chosen figures; every expected value is
// worked out by hand from the header's rules, not computed by the function under test.
#include "gpu/memory/spill_recovery.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

using namespace prosper::gpu;

static void check(bool ok, const char* what) { EXPECT_TRUE(ok) << what; }

static constexpr uint64_t MiB = 1024ull * 1024ull;
static constexpr uint64_t GiB = 1024ull * MiB;

static SpillRecoveryInputs budget(uint64_t now, uint64_t spilled, uint64_t heap_budget,
                                  uint64_t usage) {
    SpillRecoveryInputs in;
    in.now_ms = now; in.spilled_bytes = spilled; in.have_budget = true;
    in.heap_budget = heap_budget; in.heap_usage = usage;
    return in;
}

TEST(SpillRecovery, Contract) {
    // 1. Nothing spilled: never due, never granted, state untouched (the healthy-run cost).
    {
        SpillRecoveryState s;
        check(!spill_recovery_due(s, 1000, 0), "nothing spilled -> not due");
        check(spill_recovery_allowance(budget(1000, 0, 8 * GiB, 1 * GiB), s) == 0 && s.next_ms == 0,
              "nothing spilled -> no allowance, state untouched");
    }
    // 2. Headroom-limited: budget 8 GiB -> target = 8589934592 / 100 * 80 = 6871947600 bytes;
    //    usage 6 GiB + 400 MiB = 6861881344; headroom = 10066256 bytes (~9.6 MiB) < 100 MiB spilled.
    {
        SpillRecoveryState s;
        const uint64_t target = 8 * GiB / 100 * 80;
        const uint64_t usage = 6 * GiB + 400 * MiB;
        const uint64_t a = spill_recovery_allowance(budget(0, 100 * MiB, 8 * GiB, usage), s);
        check(a == target - usage && a == 10066256, "headroom limits the grant");
        check(s.next_ms == kSpillRecoveryIntervalMs, "next decision one interval later");
    }
    // 3. Spilled-limited and per-step cap.
    {
        SpillRecoveryState s;
        check(spill_recovery_allowance(budget(0, 40 * MiB, 16 * GiB, 1 * GiB), s) == 40 * MiB,
              "small spill fully granted when there is room");
        SpillRecoveryState t;
        check(spill_recovery_allowance(budget(0, 2 * GiB, 16 * GiB, 1 * GiB), t) ==
                  kSpillRecoveryMaxBytesPerStep,
              "large spill capped at the per-step maximum (256 MiB)");
    }
    // 4. No headroom: a card that is still full is not churned; no backoff change.
    {
        SpillRecoveryState s;
        check(spill_recovery_allowance(budget(0, 64 * MiB, 8 * GiB, 7 * GiB), s) == 0,
              "usage above the 80% target -> nothing");
        check(s.interval_ms == kSpillRecoveryIntervalMs && !s.granted,
              "no grant -> interval unchanged");
    }
    // 5. Rate limit: a second call inside the interval grants nothing.
    {
        SpillRecoveryState s;
        check(spill_recovery_allowance(budget(0, 32 * MiB, 16 * GiB, 1 * GiB), s) == 32 * MiB,
              "first step grants");
        check(!spill_recovery_due(s, 249, 32 * MiB) &&
                  spill_recovery_allowance(budget(249, 32 * MiB, 16 * GiB, 1 * GiB), s) == 0,
              "249 ms later -> not due, nothing");
        check(spill_recovery_due(s, 250, 32 * MiB), "250 ms later -> due");
    }
    // 6. Without a live budget: a 64 MiB probe per interval.
    {
        SpillRecoveryState s;
        SpillRecoveryInputs in; in.now_ms = 0; in.spilled_bytes = 500 * MiB;
        check(spill_recovery_allowance(in, s) == kSpillRecoveryProbeBytes, "no budget -> 64 MiB probe");
        SpillRecoveryState t;
        in.spilled_bytes = 10 * MiB;
        check(spill_recovery_allowance(in, t) == 10 * MiB, "no budget, small spill -> all of it");
    }
    // 7. Backoff: grant at spilled=100 MiB; the evicted bytes come back spilled (still 100 MiB at the
    //    next due step) -> interval doubles 250 -> 500 -> 1000; progress resets it to 250.
    {
        SpillRecoveryState s;
        SpillRecoveryInputs in; in.spilled_bytes = 100 * MiB;
        in.now_ms = 0;
        check(spill_recovery_allowance(in, s) == 64 * MiB && s.next_ms == 250, "grant 1");
        in.now_ms = 250;
        spill_recovery_allowance(in, s);
        check(s.interval_ms == 500 && s.next_ms == 750, "no progress -> interval 500");
        in.now_ms = 750;
        spill_recovery_allowance(in, s);
        check(s.interval_ms == 1000 && s.next_ms == 1750, "no progress again -> interval 1000");
        in.now_ms = 1750; in.spilled_bytes = 36 * MiB;
        spill_recovery_allowance(in, s);
        check(s.interval_ms == kSpillRecoveryIntervalMs && s.next_ms == 2000,
              "progress -> interval back to 250");
        // The cap: from a huge interval, doubling saturates at 60 s.
        s.interval_ms = 40'000; s.granted = true; s.spilled_at_grant = 36 * MiB;
        in.now_ms = s.next_ms;
        spill_recovery_allowance(in, s);
        check(s.interval_ms == kSpillRecoveryMaxBackoffMs, "backoff caps at 60 s");
    }
}
