// test_precise_sleep (#3074) — the post-condition every sleep_until_steady_ns() backend must
// satisfy: never return before `now_ns() >= deadline_ns`. The high-resolution Windows path already
// re-checked the clock after every timer signal; Win32SleepFallback did not, so a single early
// ::Sleep() return let the whole function come back before its documented deadline.
//
// That fallback only runs under _WIN32, so on this platform it cannot be driven directly (and even
// on Windows a wall-clock test would flake with host scheduling, which this project has already been
// burned by twice — see test_videoout.cpp's note on #1770/#1793). The fix is therefore split into a
// PURE FUNCTION, sleep_until_deadline_retry(), compiled and linked on every platform: it takes the
// clock read and the one-shot sleep as function pointers, so this test can supply a fake clock and a
// fake "sleep" that just advances it, and assert the loop's TERMINATION CONDITION — never a duration.
//
// Mutation-checked (see the PR): reverting sleep_until_deadline_retry()'s body from
// `while (now_ns() < deadline_ns) sleep_once(deadline_ns);` to the pre-fix single-call shape
// `if (now_ns() < deadline_ns) sleep_once(deadline_ns);` reddens arm 1 below — the fake sleep step is
// deliberately smaller than the deadline gap, so one call cannot reach it and the post-condition
// check catches exactly the #3074 defect.
#include "host/platform/precise_sleep.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// A fake clock/sleeper pair, file-scope so they can be plain function pointers (the API
// sleep_until_deadline_retry() takes, matching the real steady_now_ns()/sleep_until_with_std()
// signatures it is called with in precise_sleep.cpp). Reset at the top of each arm.
static uint64_t g_fake_now_ns = 0;
static int g_sleep_calls = 0;
static uint64_t g_sleep_step_ns = 0;   // how far one fake "sleep" advances the fake clock

static uint64_t fake_now_ns() { return g_fake_now_ns; }

// Simulates an OS sleep primitive that does not itself guarantee reaching the deadline — exactly
// what ::Sleep()'s ~15.6 ms tick quantization can do relative to a 16.68 ms vblank deadline (or,
// more generally, what any single OS wait is allowed to do: return marginally early). Ignores the
// deadline it is passed and always advances by the fixed step, so a caller that does not re-check
// and retry will return however far short of the deadline that leaves it.
static void fake_sleep_fixed_step(uint64_t /*deadline_ns*/) {
    ++g_sleep_calls;
    g_fake_now_ns += g_sleep_step_ns;
}

int main() {
    printf("== test_precise_sleep ==\n");

    // 1. THE #3074 CASE: each fake sleep advances less than the full gap to the deadline, so a
    //    single call cannot satisfy the post-condition. This is the arm the mutation check reddens.
    {
        g_fake_now_ns = 0;
        g_sleep_calls = 0;
        g_sleep_step_ns = 1000;                    // ten calls needed to cover the 10000 ns deadline
        constexpr uint64_t kDeadlineNs = 10000;
        host::sleep_until_deadline_retry(kDeadlineNs, fake_now_ns, fake_sleep_fixed_step);
        CHECK(g_fake_now_ns >= kDeadlineNs,
              "retry loop never returns before the deadline, even when each sleep undershoots it");
        CHECK(g_sleep_calls > 1,
              "reaching the deadline took more than one sleep call (proves it looped, not just "
              "called once)");
        CHECK(g_sleep_calls == 10,
              "it retried exactly as many times as the fixed step requires, not more (no busy spin "
              "past the deadline)");
    }

    // 2. A sleep that reaches the deadline in one call: the loop must not call again after the
    //    post-condition already holds (no extra, unnecessary sleep).
    {
        g_fake_now_ns = 0;
        g_sleep_calls = 0;
        g_sleep_step_ns = 10000;                   // exactly the deadline gap
        constexpr uint64_t kDeadlineNs = 10000;
        host::sleep_until_deadline_retry(kDeadlineNs, fake_now_ns, fake_sleep_fixed_step);
        CHECK(g_fake_now_ns >= kDeadlineNs, "the single-call case still satisfies the post-condition");
        CHECK(g_sleep_calls == 1, "a sleep that already reaches the deadline is not retried again");
    }

    // 3. A sleep that overshoots the deadline in one call: same as (2), one call is enough.
    {
        g_fake_now_ns = 0;
        g_sleep_calls = 0;
        g_sleep_step_ns = 50000;                   // far past the deadline
        constexpr uint64_t kDeadlineNs = 10000;
        host::sleep_until_deadline_retry(kDeadlineNs, fake_now_ns, fake_sleep_fixed_step);
        CHECK(g_fake_now_ns >= kDeadlineNs, "an overshooting sleep satisfies the post-condition");
        CHECK(g_sleep_calls == 1, "an overshooting sleep is not called a second time");
    }

    // 4. The deadline has ALREADY passed when the loop is entered: it must not sleep at all. This is
    //    the same early-out precise_sleep.cpp keeps before calling the retry loop, pinned here as a
    //    property of the loop itself.
    {
        g_fake_now_ns = 12345;
        g_sleep_calls = 0;
        g_sleep_step_ns = 1000;
        constexpr uint64_t kDeadlineNs = 10000;    // already behind g_fake_now_ns
        host::sleep_until_deadline_retry(kDeadlineNs, fake_now_ns, fake_sleep_fixed_step);
        CHECK(g_sleep_calls == 0, "a deadline already in the past is not slept on at all");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
