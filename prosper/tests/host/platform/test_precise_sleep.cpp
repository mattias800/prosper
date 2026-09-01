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
//
// ARMS 5-11 (#3069) cover a SECOND set of pure functions in the same header, extracted from
// scePthreadSemTimedwait's Windows poll loop for the same reason and testable the same way:
// timeout_ns_from_us(), deadline_ns_from(), poll_slice_ns() and sem_poll_step(). Two of them are
// saturating arms guarding a guest-controlled timeout, and before the extraction nothing reached
// them -- the only handle on that arithmetic was to call the real HLE and time it, which
// test_kernel_sem_timedwait.cpp's header records at length cannot discriminate anything here.
//
// The cases the fake clock buys, none of which a wall-clock test can produce on demand: an early
// wakeup, a deadline already in the past, a clock that runs backwards, the slice clamp binding, and
// arithmetic within a few nanoseconds of UINT64_MAX. Mutation results are in the PR.
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

// ---------------------------------------------------------------------------
// A driver for sem_poll_step() (#3069) that reproduces the real poll loop's structure exactly --
// read the clock, take a step, "sleep", repeat -- with the clock replaced by a variable. The one
// thing it varies is WHERE THE SLEEP ACTUALLY LANDS relative to the step's requested target, which
// is the seam a real host has and a test otherwise cannot control.
enum class Wake {
    Exact,       // the sleep returns precisely at the requested target
    Early,       // it returns roughly halfway there -- the #3074 shape, seen from this side
    Overshoot,   // it returns late, as ::Sleep()'s tick quantization does by 14-30 ms
};

struct LoopStats {
    int steps = 0;
    int fine_steps = 0;      // steps taken while inside the 8 ms fine window
    int coarse_steps = 0;    // steps taken after it
    int clamped_steps = 0;   // steps where the remaining time, not the slice, set the target
    bool invariants_ok = true;
    bool terminated = false;
    uint64_t last_now_ns = 0;
};

// Runs the loop to its timeout, asserting sem_poll_step()'s two invariants at EVERY step:
//     now_ns < sleep_until_ns <= deadline_ns
// and that the fake clock advances strictly, so a violation cannot hide behind a stalled loop.
static LoopStats drive_poll_loop(uint64_t start_ns, uint64_t timeout_us, Wake wake,
                                 int max_steps = 100000) {
    LoopStats st;
    const uint64_t timeout_ns = host::timeout_ns_from_us(timeout_us);
    const uint64_t deadline_ns = host::deadline_ns_from(start_ns, timeout_ns);
    uint64_t now_ns = start_ns;
    st.last_now_ns = now_ns;
    for (int i = 0; i < max_steps; ++i) {
        const host::SemPollStep step = host::sem_poll_step(start_ns, deadline_ns, now_ns);
        if (step.expired) { st.terminated = true; break; }
        ++st.steps;
        // THE INVARIANTS. The strict lower bound is what forbids a zero-length sleep (a busy spin
        // while time remains); the upper bound is the clamp, and it is also what keeps
        // `now_ns + slice` from wrapping when the deadline is near UINT64_MAX.
        if (!(step.sleep_until_ns > now_ns)) st.invariants_ok = false;
        if (!(step.sleep_until_ns <= deadline_ns)) st.invariants_ok = false;
        // Which regime this step was in, and whether the clamp bound it. Counted so the arms below
        // can assert that the branch they claim to cover was actually REACHED -- a step count alone
        // cannot tell a loop that exercised both regimes from one that never left the first.
        if ((now_ns - start_ns) < host::kSemPollFineWindowNs) ++st.fine_steps; else ++st.coarse_steps;
        if (step.sleep_until_ns == deadline_ns) ++st.clamped_steps;

        uint64_t next_ns = step.sleep_until_ns;
        if (wake == Wake::Early) {
            // Land strictly between now and the target. `+ 1` keeps it strictly ahead of `now_ns`
            // even when the gap is 1 ns, so the driver itself can never stall.
            next_ns = now_ns + (step.sleep_until_ns - now_ns) / 2 + 1;
            if (next_ns > step.sleep_until_ns) next_ns = step.sleep_until_ns;
        } else if (wake == Wake::Overshoot) {
            // Saturating, so an overshoot near UINT64_MAX does not wrap the DRIVER and fake a pass.
            next_ns = step.sleep_until_ns > ~0ull - 137000ull ? ~0ull : step.sleep_until_ns + 137000ull;
        }
        if (!(next_ns > now_ns)) { st.invariants_ok = false; break; }   // the driver must progress
        now_ns = next_ns;
        st.last_now_ns = now_ns;
    }
    return st;
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

    // =======================================================================
    // #3069: scePthreadSemTimedwait's poll-loop arithmetic, driven by a fake clock.
    // =======================================================================
    constexpr uint64_t kNsMax = ~0ull;

    // 5. SATURATING us -> ns. `timeout_us` is guest-controlled; a wrap turns a 584-year timeout
    //    into a few hundred nanoseconds. The naive `us * 1000` gives 384 for the first input past
    //    the boundary, which is why the property arm below is stated as "never less than its own
    //    input" rather than as a magic number.
    {
        CHECK(host::timeout_ns_from_us(0) == 0, "a zero timeout converts to zero");
        CHECK(host::timeout_ns_from_us(1) == 1000, "1 us is 1000 ns");
        CHECK(host::timeout_ns_from_us(5000) == 5000000,
              "the 5 ms timeout test_kernel_sem_timedwait uses converts exactly");
        // THE BOUNDARY, both sides of it. kNsMax/1000 is the largest input that does NOT saturate.
        constexpr uint64_t kLastExact = kNsMax / 1000ull;
        CHECK(host::timeout_ns_from_us(kLastExact) == kLastExact * 1000ull,
              "the largest non-saturating input converts exactly (no premature clamping)");
        CHECK(host::timeout_ns_from_us(kLastExact + 1) == kNsMax,
              "one microsecond past it saturates instead of wrapping");
        CHECK(host::timeout_ns_from_us(kNsMax) == kNsMax, "UINT64_MAX microseconds saturates");
        // The property, swept across the boundary: a saturating conversion is non-decreasing and
        // never returns less than its input. A wrap violates both, and this arm does not need to
        // know what the wrapped value would be.
        bool monotonic = true, never_shrinks = true;
        uint64_t prev = 0;
        for (uint64_t d = 0; d <= 8; ++d) {
            const uint64_t u = kLastExact - 4 + d;
            const uint64_t got = host::timeout_ns_from_us(u);
            if (got < prev) monotonic = false;
            if (got < u) never_shrinks = false;
            prev = got;
        }
        CHECK(monotonic, "the conversion is non-decreasing across the saturation boundary");
        CHECK(never_shrinks, "...and never returns a timeout SHORTER than the microseconds asked for");
    }

    // 6. SATURATING deadline addition. A wrapped deadline lands in the PAST, which turns a long
    //    wait into an instant timeout.
    {
        CHECK(host::deadline_ns_from(0, 0) == 0, "a zero start and zero timeout give a zero deadline");
        CHECK(host::deadline_ns_from(1000, 5000) == 6000, "the ordinary case is a plain addition");
        // The boundary, both sides. With start = kNsMax - 5, a timeout of exactly 5 still fits.
        CHECK(host::deadline_ns_from(kNsMax - 5, 5) == kNsMax,
              "a timeout that exactly reaches the maximum is not clamped early");
        CHECK(host::deadline_ns_from(kNsMax - 5, 6) == kNsMax,
              "one nanosecond past it saturates instead of wrapping into the past");
        CHECK(host::deadline_ns_from(kNsMax, 1) == kNsMax, "a start already at the maximum saturates");
        CHECK(host::deadline_ns_from(100, kNsMax) == kNsMax, "a maximal timeout saturates");
        // The property: the deadline is never BEHIND the start. That is the whole point of the
        // saturation, and the naive `start + timeout` fails it (kNsMax - 5 + 6 wraps to 0).
        bool never_in_past = true;
        for (uint64_t d = 0; d <= 8; ++d) {
            const uint64_t start = kNsMax - 4;   // fixed, near the top; `d` sweeps the TIMEOUT
            const uint64_t got = host::deadline_ns_from(start, d);
            if (got < start) never_in_past = false;
        }
        CHECK(never_in_past, "a saturated deadline never lands BEHIND its own start");
    }

    // 7. THE ADAPTIVE SLICE AND ITS CLAMP, tested directly rather than only through sem_poll_step,
    //    so a clamp regression is attributable to one function.
    {
        const uint64_t kFine = host::kSemPollFineSliceNs;
        const uint64_t kCoarse = host::kSemPollCoarseSliceNs;
        const uint64_t kWindow = host::kSemPollFineWindowNs;
        const uint64_t kPlenty = 1000ull * 1000ull * 1000ull;   // 1 s remaining: never the binding constraint

        // The three constants themselves, as LITERALS. Every other assertion in this arm reads them
        // SYMBOLICALLY -- which is right for testing the selection logic, but means a change to a
        // constant's VALUE cannot redden any of them. Mutation-checked, and it is not hypothetical:
        // swapping the fine slice to 2 ms survived every other assertion in this file. The triple is
        // a measured tuning decision (see the sleep-floor measurement above poll_slice_ns), so this
        // is a tripwire making a retune deliberate -- not a claim that these values are optimal.
        CHECK(kFine == 500000ull && kCoarse == 2000000ull && kWindow == 8000000ull,
              "the measured poll cadence is 500 us / 2 ms / 8 ms (tripwire on an accidental retune)");

        CHECK(host::poll_slice_ns(0, kPlenty) == kFine, "a young wait polls on the 500 us slice");
        CHECK(host::poll_slice_ns(kWindow - 1, kPlenty) == kFine,
              "...right up to the last nanosecond inside the 8 ms window");
        CHECK(host::poll_slice_ns(kWindow, kPlenty) == kCoarse,
              "at exactly 8 ms elapsed it switches to the 2 ms slice (the boundary is `<`, not `<=`)");
        CHECK(host::poll_slice_ns(kWindow + 1, kPlenty) == kCoarse, "...and stays there afterwards");

        // THE CLAMP. Each of these fails if the `remaining_ns < slice` clamp is removed.
        CHECK(host::poll_slice_ns(0, 1) == 1,
              "one nanosecond of remaining time clamps the fine slice to 1 ns, not 500 us");
        CHECK(host::poll_slice_ns(kWindow, 1) == 1, "...and clamps the coarse slice just the same");
        CHECK(host::poll_slice_ns(0, kFine - 1) == kFine - 1,
              "remaining just under the fine slice binds");
        CHECK(host::poll_slice_ns(0, kFine) == kFine,
              "remaining exactly equal to the slice is NOT clamped (the boundary is `<`, not `<=`)");
        CHECK(host::poll_slice_ns(kWindow, kCoarse - 1) == kCoarse - 1,
              "remaining just under the coarse slice binds");
        // The property that makes the clamp the anti-overflow guard, swept: the slice never exceeds
        // the remaining time, in either regime.
        bool never_exceeds = true;
        for (uint64_t rem = 0; rem <= 12; ++rem) {
            if (host::poll_slice_ns(0, rem) > rem) never_exceeds = false;
            if (host::poll_slice_ns(kWindow, rem) > rem) never_exceeds = false;
        }
        CHECK(never_exceeds, "the slice never exceeds the time remaining, in either regime");
    }

    // 8. A DEADLINE ALREADY IN THE PAST, and the exact instant it becomes one. The real loop cannot
    //    be asked for this: it would need a semaphore that stays unposted and a clock the test owns.
    {
        constexpr uint64_t kStart = 1000000ull, kDeadline = 1005000ull;
        CHECK(host::sem_poll_step(kStart, kDeadline, kDeadline).expired,
              "a clock exactly ON the deadline has expired (the test is `>=`, not `>`)");
        CHECK(host::sem_poll_step(kStart, kDeadline, kDeadline + 1).expired,
              "a clock past the deadline has expired");
        CHECK(host::sem_poll_step(kStart, kDeadline, kDeadline + 999999999ull).expired,
              "a wildly overshot wake still reports expiry rather than computing a slice");
        CHECK(!host::sem_poll_step(kStart, kDeadline, kDeadline - 1).expired,
              "one nanosecond short of the deadline has NOT expired");
        // ...and that last nanosecond is slept on, not skipped or spun through.
        const host::SemPollStep last = host::sem_poll_step(kStart, kDeadline, kDeadline - 1);
        CHECK(last.sleep_until_ns == kDeadline,
              "...and the final nanosecond is slept to the deadline exactly, never past it");
        // A zero-length timeout: expired on the very first look, without a single sleep.
        CHECK(host::sem_poll_step(kStart, kStart, kStart).expired,
              "a zero timeout expires immediately (deadline == start)");
    }

    // 9. THE FULL LOOP under three wake behaviours, including an EARLY wakeup. Every arm asserts
    //    both regimes were actually entered -- otherwise a loop that never left the fine window
    //    would pass while claiming to cover the crossover.
    {
        constexpr uint64_t kStart = 5000000000ull;   // an arbitrary, non-zero clock epoch
        constexpr uint64_t kTimeoutUs = 50000;       // 50 ms: well past the 8 ms fine window

        for (int w = 0; w < 3; ++w) {
            const Wake wake = w == 0 ? Wake::Exact : (w == 1 ? Wake::Early : Wake::Overshoot);
            const char* name = w == 0 ? "exact" : (w == 1 ? "early" : "overshooting");
            const LoopStats st = drive_poll_loop(kStart, kTimeoutUs, wake);
            printf("         (%s wakes: %d steps, %d fine, %d coarse, %d clamped)\n",
                   name, st.steps, st.fine_steps, st.coarse_steps, st.clamped_steps);
            CHECK(st.terminated, "the poll loop terminates");
            CHECK(st.invariants_ok, "now_ns < sleep_until_ns <= deadline_ns held at every step");
            // The lever moved: both regimes were reached, so the crossover really was crossed.
            CHECK(st.fine_steps > 0, "...and the fine (500 us) regime was actually entered");
            CHECK(st.coarse_steps > 0, "...and the coarse (2 ms) regime was actually entered");
            CHECK(st.clamped_steps > 0,
                  "...and the clamp bound at least once, on the final approach to the deadline");
            CHECK(st.last_now_ns >= kStart, "the fake clock never wrapped");
        }

        // An early wakeup costs strictly MORE iterations than an exact one for the same timeout --
        // which is the observable consequence of the loop recomputing from `now_ns` rather than
        // carrying a target forward. If these were equal the "early" arm would be exercising
        // nothing the "exact" arm does not.
        const LoopStats exact = drive_poll_loop(kStart, kTimeoutUs, Wake::Exact);
        const LoopStats early = drive_poll_loop(kStart, kTimeoutUs, Wake::Early);
        CHECK(early.steps > exact.steps,
              "an early-returning sleep really does drive more iterations (the arm is not a no-op)");
    }

    // 10. A CLOCK THAT RUNS BACKWARDS. steady_clock cannot do this, so the real call site cannot
    //     reach it -- but the arithmetic is expressible and this pins what it currently does, so a
    //     later change to the elapsed computation is a DELIBERATE one rather than a silent one.
    //
    //     Current behaviour, preserved unchanged by #3069's extraction: `now_ns - start_ns` wraps to
    //     a huge value, so the coarse slice is selected. Benign -- the loop polls at 2 ms instead of
    //     500 us and still times out at exactly the right moment, because the deadline test does not
    //     involve start_ns at all. THIS ARM ASSERTS THE STATUS QUO, NOT THAT THE STATUS QUO IS IDEAL.
    {
        constexpr uint64_t kStart = 5000000000ull;
        const uint64_t deadline = host::deadline_ns_from(kStart, host::timeout_ns_from_us(50000));
        const uint64_t now_behind = kStart - 1000000ull;   // the clock jumped 1 ms backwards
        const host::SemPollStep step = host::sem_poll_step(kStart, deadline, now_behind);
        CHECK(!step.expired, "a backwards clock before the deadline has not expired");
        CHECK(step.sleep_until_ns == now_behind + host::kSemPollCoarseSliceNs,
              "a backwards clock currently selects the COARSE slice (wrapped elapsed) -- pinned, "
              "not endorsed");
        CHECK(step.sleep_until_ns > now_behind && step.sleep_until_ns <= deadline,
              "...and the invariants still hold, so the loop is still correct, just slower");
        // The termination guarantee survives it: the deadline test reads only now_ns and
        // deadline_ns, so a backwards jump cannot make the loop run forever.
        CHECK(host::sem_poll_step(kStart, deadline, deadline).expired,
              "...and expiry is still decided by the deadline alone, never by the elapsed time");
    }

    // 11. ARITHMETIC WITHIN A FEW NANOSECONDS OF UINT64_MAX -- the case the clamp exists for, and
    //     the one a wall-clock test can never construct. Without the clamp, `now_ns + 500000` wraps
    //     here and the next wake lands in the PAST.
    {
        const uint64_t start = kNsMax - 10;
        const uint64_t deadline = host::deadline_ns_from(start, host::timeout_ns_from_us(kNsMax));
        CHECK(deadline == kNsMax, "a maximal timeout from a near-maximal start saturates to the max");

        const uint64_t now = kNsMax - 1;                 // one nanosecond of remaining time
        const host::SemPollStep step = host::sem_poll_step(start, deadline, now);
        CHECK(!step.expired, "one nanosecond short of a maximal deadline has not expired");
        CHECK(step.sleep_until_ns == kNsMax,
              "...and the wake target is clamped to the deadline itself, not wrapped past zero");
        CHECK(step.sleep_until_ns > now, "...so it is still strictly in the future (no wrap)");
        CHECK(step.sleep_until_ns <= deadline, "...and still no later than the deadline");

        // Walk the last few nanoseconds one at a time: every one must stay ordered and terminate.
        bool ok = true;
        for (uint64_t back = 8; back >= 1; --back) {
            const uint64_t n = kNsMax - back;
            const host::SemPollStep s2 = host::sem_poll_step(start, deadline, n);
            if (s2.expired) { ok = false; break; }
            if (!(s2.sleep_until_ns > n && s2.sleep_until_ns <= deadline)) { ok = false; break; }
        }
        CHECK(ok, "every step through the last 8 ns before UINT64_MAX stays ordered and bounded");
        CHECK(host::sem_poll_step(start, deadline, kNsMax).expired,
              "and the maximum itself is the timeout, reached rather than overshot into a wrap");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
