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
#include <initializer_list>

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

    // -----------------------------------------------------------------------
    // ARMS 12-17 (#3056) — the RELATIVE-microsecond timed waits' deadline arithmetic:
    // normalize_wait_deadline(), abs_deadline_from_rel_us() and wait_timeout_ms_ceil().
    //
    // wait_timeout_ms_ceil() is the conversion the WINDOWS condition wait feeds to WaitOnAddress,
    // and before the extraction it lived inside `#ifdef _WIN32` — so on this host it could not be
    // compiled at all, let alone driven through its rounding, its past-deadline case or its
    // saturation. That is the same reason arms 5-11 exist, one platform further out: those needed a
    // fake clock, these need a fake clock AND to escape the preprocessor.
    //
    // These are NOT accuracy tests. Nothing here asserts a duration; every arm is an identity of the
    // pure function, so none of it can flake under host load (the #1770/#1793 rule this file's
    // header already states).

    // 12. NORMALIZATION, both directions and both saturating ends.
    {
        const host::WaitDeadline in_range = host::normalize_wait_deadline({5, 999999999});
        CHECK(in_range.sec == 5 && in_range.nsec == 999999999,
              "an already-normalized value is returned unchanged");

        const host::WaitDeadline carry = host::normalize_wait_deadline({5, 2500000000ll});
        CHECK(carry.sec == 7 && carry.nsec == 500000000ll,
              "2.5e9 ns carries two whole seconds and leaves 0.5e9");

        // A NEGATIVE nanosecond field borrows rather than truncating toward zero, which is what
        // integer division alone would do: -1 ns must be {sec-1, 999999999}, never {sec, -1}.
        const host::WaitDeadline borrow = host::normalize_wait_deadline({5, -1});
        CHECK(borrow.sec == 4 && borrow.nsec == 999999999,
              "a negative nanosecond field borrows a second instead of truncating toward zero");

        const host::WaitDeadline big_borrow = host::normalize_wait_deadline({5, -2500000000ll});
        CHECK(big_borrow.sec == 2 && big_borrow.nsec == 500000000ll,
              "-2.5e9 ns borrows three seconds and leaves 0.5e9");

        const host::WaitDeadline hi = host::normalize_wait_deadline({INT64_MAX, 2000000000ll});
        CHECK(hi.sec == INT64_MAX && hi.nsec == 999999999,
              "a carry past INT64_MAX seconds saturates instead of wrapping negative");

        const host::WaitDeadline lo = host::normalize_wait_deadline({INT64_MIN, -2000000000ll});
        CHECK(lo.sec == INT64_MIN && lo.nsec == 0,
              "a borrow past INT64_MIN seconds saturates instead of wrapping positive");
    }

    // 13. RELATIVE MICROSECONDS -> ABSOLUTE DEADLINE, the conversion every Sony
    //     scePthread*Timedwait/Timedlock entry point performs on a guest-supplied scalar.
    {
        // The exact interval #3056 was filed on: 818 us.
        const host::WaitDeadline d = host::abs_deadline_from_rel_us({100, 0}, 818);
        CHECK(d.sec == 100 && d.nsec == 818000,
              "818 us lands 818,000 ns past a whole-second reading");

        CHECK(host::abs_deadline_from_rel_us({100, 0}, 0).sec == 100 &&
              host::abs_deadline_from_rel_us({100, 0}, 0).nsec == 0,
              "a zero timeout is the reading itself, not one tick later");

        const host::WaitDeadline whole = host::abs_deadline_from_rel_us({100, 0}, 2500000);
        CHECK(whole.sec == 102 && whole.nsec == 500000000ll,
              "2.5 s of microseconds splits into whole seconds plus a remainder");

        // The CARRY, which the code this replaced did with a single subtraction: it is correct only
        // because now.nsec < 1e9 and the added part is < 1e9, so at most one second is carried.
        const host::WaitDeadline carry = host::abs_deadline_from_rel_us({100, 999999000}, 2);
        CHECK(carry.sec == 101 && carry.nsec == 1000,
              "a remainder that crosses a second boundary carries exactly once");
        CHECK(carry.nsec >= 0 && carry.nsec < 1000000000ll,
              "...and the result is normalized, so a 32-bit host tv_nsec cannot truncate it");

        // Every returned nanosecond field is in range, for the widest inputs available. This is the
        // property the host-`struct timespec` adapter in hle_kernel.cpp depends on: it casts nsec to
        // a `long`, which is 32 bits on Windows x64 (#3038's truncation class).
        bool normalized = true;
        const uint64_t widths[] = {1ull, 999999ull, 1000000ull, 1000001ull,
                                   1000000000ull, ~0ull / 2, ~0ull};
        for (uint64_t us : widths) {
            for (int64_t base_ns : {0ll, 1ll, 500000000ll, 999999999ll}) {
                const host::WaitDeadline r = host::abs_deadline_from_rel_us({1700000000ll, base_ns}, us);
                if (r.nsec < 0 || r.nsec >= 1000000000ll) { normalized = false; break; }
            }
        }
        CHECK(normalized, "every result is normalized to [0, 1e9) across the whole input width");

        // SATURATION. Unreachable from a real clock reading -- UINT64_MAX microseconds is 5.8e5
        // years of seconds against int64's 2.9e11 -- and asserted anyway, because the arm is what
        // makes "this cannot wrap" a checked statement rather than an assumption. Nothing else in
        // the tree can reach it.
        const host::WaitDeadline sat = host::abs_deadline_from_rel_us({INT64_MAX - 1, 0}, ~0ull);
        CHECK(sat.sec == INT64_MAX && sat.nsec == 999999999,
              "a maximal timeout from a near-maximal reading saturates instead of wrapping negative");
        CHECK(sat.sec > 0, "...so the deadline stays in the FUTURE, which is the direction that matters");

        const host::WaitDeadline sat_carry = host::abs_deadline_from_rel_us({INT64_MAX, 999999999}, 1);
        CHECK(sat_carry.sec == INT64_MAX && sat_carry.nsec == 999999999,
              "a carry off the top of the seconds field saturates too");
    }

    // 14. ABSOLUTE DEADLINE -> WHOLE MILLISECONDS, ROUNDED UP. This is the Windows conversion, and
    //     the rounding direction is a CONTRACT: pthread_cond_timedwait must never report ETIMEDOUT
    //     before its deadline, so a positive remainder may not become a zero-millisecond wait.
    {
        const uint64_t kMaxMs = 0xFFFFFFFEull;   // what the caller passes: INFINITE - 1

        CHECK(host::wait_timeout_ms_ceil({100, 0}, {100, 818000}, kMaxMs) == 1,
              "the 0.818 ms request from #3056's census rounds UP to a 1 ms wait, never down to 0");
        CHECK(host::wait_timeout_ms_ceil({100, 0}, {100, 1}, kMaxMs) == 1,
              "one nanosecond of remaining time is one millisecond of wait, not zero");
        CHECK(host::wait_timeout_ms_ceil({100, 0}, {100, 1000000}, kMaxMs) == 1,
              "an exact millisecond is not rounded up to two");
        CHECK(host::wait_timeout_ms_ceil({100, 0}, {100, 1000001}, kMaxMs) == 2,
              "a nanosecond past an exact millisecond is");
        CHECK(host::wait_timeout_ms_ceil({100, 500000000ll}, {102, 250000000ll}, kMaxMs) == 1750,
              "a span crossing whole seconds is converted across the boundary, not within it");
    }

    // 15. THE PAST-DEADLINE CASE, which is the loop's TERMINATION condition: zero means "the
    //     deadline is reached", and the caller reports ETIMEDOUT on exactly that. Getting a
    //     nonzero answer for a past deadline would park a guest past its own timeout; getting zero
    //     for a future one is the early-ETIMEDOUT defect the re-arm loop exists to prevent.
    {
        const uint64_t kMaxMs = 0xFFFFFFFEull;
        CHECK(host::wait_timeout_ms_ceil({100, 0}, {100, 0}, kMaxMs) == 0,
              "a deadline exactly now has been reached");
        CHECK(host::wait_timeout_ms_ceil({100, 1}, {100, 0}, kMaxMs) == 0,
              "a deadline one nanosecond in the past has been reached");
        CHECK(host::wait_timeout_ms_ceil({101, 0}, {100, 999999999}, kMaxMs) == 0,
              "...including across a second boundary, where the nanosecond field alone reads FORWARD");
        CHECK(host::wait_timeout_ms_ceil({1000, 0}, {100, 0}, kMaxMs) == 0,
              "a deadline fifteen minutes in the past is still just zero, never a huge unsigned span");
        // The sign-borrow the other way: now.nsec > deadline.nsec but the deadline is still ahead.
        CHECK(host::wait_timeout_ms_ceil({100, 999000000ll}, {101, 1000000ll}, kMaxMs) == 2,
              "a borrow across the second boundary yields the real 2 ms, not a wrapped span");
    }

    // 16. SATURATION AND THE ABSENCE OF OVERFLOW at the widest spans representable. `seconds * 1000`
    //     would overflow for any span past ~2.9e8 years; the bound is applied BEFORE the multiply,
    //     so no input can reach it.
    {
        const uint64_t kMaxMs = 0xFFFFFFFEull;
        CHECK(host::wait_timeout_ms_ceil({0, 0}, {INT64_MAX, 999999999}, kMaxMs) == kMaxMs,
              "the widest forward span saturates at max_ms instead of overflowing the multiply");
        CHECK(host::wait_timeout_ms_ceil({INT64_MIN, 0}, {INT64_MAX, 0}, kMaxMs) == kMaxMs,
              "...including a span wider than INT64_MAX, where a signed subtraction would overflow");
        // Exactly at, and one below, the saturation boundary.
        const uint64_t boundary_sec = kMaxMs / 1000ull;              // 4,294,967 s
        CHECK(host::wait_timeout_ms_ceil({0, 0}, {(int64_t)boundary_sec, 0}, kMaxMs)
                  == boundary_sec * 1000ull,
              "a span exactly at the seconds bound is converted, not saturated");
        CHECK(host::wait_timeout_ms_ceil({0, 0}, {(int64_t)boundary_sec + 1, 0}, kMaxMs) == kMaxMs,
              "one second past it saturates");
        CHECK(host::wait_timeout_ms_ceil({0, 0}, {(int64_t)boundary_sec, 999999999}, kMaxMs) == kMaxMs,
              "...as does a remainder that would push the millisecond count past max_ms");
        // max_ms is a parameter so the saturation is reachable without INFINITE; check it binds.
        CHECK(host::wait_timeout_ms_ceil({0, 0}, {10, 0}, 5) == 5,
              "max_ms binds wherever the caller sets it, not only at INFINITE - 1");

        // A max_ms near UINT64_MAX, where the SUM of the whole and fractional parts is what wraps
        // rather than the multiply. Raised in review of #3235: a trailing `ms > max_ms ? max_ms : ms`
        // reads a wrapped tiny value as being under the limit and returns it. Not reachable from the
        // production caller (which passes INFINITE-1) -- asserted because a function whose whole
        // purpose is saturating should not have a wrap that only luck keeps unreachable.
        //
        // The input is chosen, not guessed: the seconds bound keeps spans up to `max_ms / 1000`
        // seconds, whose whole-millisecond part is `(max_ms/1000)*1000` -- which for max_ms =
        // UINT64_MAX leaves only 615 ms of headroom below the maximum. A fractional part of
        // 1000 ms (any nanosecond remainder at all, rounded up from 0.999999999 s) therefore
        // overshoots, and an unbounded `whole + frac` wraps to 384. An earlier revision of this arm
        // used a 1 ns remainder, whose frac_ms is 1 and fits in the 615 -- so it passed the mutation
        // and asserted nothing. That is recorded because it is this file's own recurring failure:
        // an arm whose INPUT cannot reach the case it names.
        {
            const uint64_t huge = ~0ull;
            const uint64_t at_bound = huge / 1000ull;         // the largest whole-second span kept
            const uint64_t r = host::wait_timeout_ms_ceil(
                {0, 0}, {(int64_t)at_bound, 999999999}, huge);
            CHECK(r == huge,
                  "a sum that would exceed a near-UINT64_MAX max_ms saturates rather than wrapping");
            CHECK(r >= at_bound * 1000ull,
                  "...so the answer is never LESS than the whole-seconds part it already earned");
        }
    }

    // 17. THE TWO FUNCTIONS COMPOSED, which is the real call sequence: a guest hands over a relative
    //     microsecond count, hle_kernel.cpp turns it into an absolute deadline, and sync_futex.cpp's
    //     Windows branch turns that back into a millisecond timeout. The property that must survive
    //     the round trip is that the wait is never SHORTER than what the guest asked for.
    {
        const uint64_t kMaxMs = 0xFFFFFFFEull;
        const host::WaitDeadline now{1700000000ll, 123456789ll};
        bool never_short = true;
        uint64_t worst_us = 0;
        for (uint64_t us : {1ull, 499ull, 818ull, 1000ull, 1001ull, 5330ull, 16667ull,
                            999999ull, 1000000ull, 1234567ull}) {
            const host::WaitDeadline dl = host::abs_deadline_from_rel_us(now, us);
            const uint64_t ms = host::wait_timeout_ms_ceil(now, dl, kMaxMs);
            // Ceil of the request in milliseconds: the smallest wait that cannot be short.
            const uint64_t want_ms = (us + 999ull) / 1000ull;
            if (ms != want_ms) { never_short = false; worst_us = us; break; }
        }
        CHECK(never_short, "the round trip yields exactly ceil(request), so no wait is ever short");
        if (!never_short) printf("       first divergence at %llu us\n", (unsigned long long)worst_us);

        // And the one case where the guest asked for nothing: a zero timeout must reach the
        // deadline immediately rather than becoming a one-millisecond park.
        const host::WaitDeadline zero = host::abs_deadline_from_rel_us(now, 0);
        CHECK(host::wait_timeout_ms_ceil(now, zero, kMaxMs) == 0,
              "a zero-microsecond request is already expired, not rounded up to 1 ms");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
