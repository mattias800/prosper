// precise_sleep — block until a deadline on std::chrono::steady_clock, without inheriting the
// Win32 process timer period. See precise_sleep.cpp for why this exists rather than a bare
// std::this_thread::sleep_until (#1765).
#pragma once
#include <cstdint>

namespace prosper::host {

// Which primitive the calling thread's most recently completed sleep_until_steady_ns() used.
// Thread-local and diagnostic: a thread that has not slept yet reports None. It exists so a test
// can assert the MECHANISM — that the high-resolution path is actually in force — instead of
// asserting a wall-clock duration, which on a loaded host is a future flake (see the note in
// test_videoout.cpp about #1770/#1793).
enum class SleepBackend {
    None,                       // this thread has not completed a wait yet
    PosixSleepUntil,            // std::this_thread::sleep_until (glibc: a nanosleep loop)
    Win32HighResolutionTimer,   // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION waitable timer
    Win32SleepFallback,         // the timer was unavailable; ::Sleep(ms) quantization is in force
};

// Block until `deadline_ns`, measured on std::chrono::steady_clock's epoch in nanoseconds. Returns
// immediately if the deadline has already passed.
void sleep_until_steady_ns(uint64_t deadline_ns);

// The post-condition sleep_until_steady_ns() promises on EVERY backend: never return before
// `now_ns() >= deadline_ns`. A single OS sleep primitive is not trusted to hit this on its own -- a
// waitable timer can signal marginally early, and so can the ::Sleep()-backed fallback used when no
// waitable timer is available (#3074) -- so a backend that cannot prove its own precision composes
// this retry loop instead of calling its one-shot sleep directly and returning.
//
// `now_ns` and `sleep_once` are function pointers rather than the real clock/sleep calls so the
// LOOP'S TERMINATION CONDITION is a pure function, testable on any host platform without executing
// a real sleep or depending on _WIN32: a test supplies a fake clock and a fake "sleep" that just
// advances it by a fixed amount, and asserts the loop keeps calling sleep_once until the fake clock
// reaches the deadline -- never a wall-clock duration, which is a flake by construction on a loaded
// host (see test_videoout.cpp's note on #1770/#1793, and next_grid_deadline_ns above for the same
// pure-function-over-real-time precedent from #3075).
void sleep_until_deadline_retry(uint64_t deadline_ns,
                                 uint64_t (*now_ns)(),
                                 void (*sleep_once)(uint64_t deadline_ns));

SleepBackend sleep_backend();

// A stable spelling of sleep_backend() for assertions and logs. Never null.
const char* sleep_backend_name();

// The next boundary of a fixed periodic grid strictly after `now_ns`, where the grid is every
// `origin_ns + k * period_ns`. Intended to be called after a wait on the previous boundary has
// completed, as the WHOLE of a pacing loop's per-iteration work: because the answer depends only
// on (origin, now, period) and not on the deadline just waited, one expression covers the
// ordinary case, an early return, and a stall of any length.
//
// Phase-exact by construction, which is the property that motivated it. The obvious alternative
// -- carry a deadline and advance it by one period, re-anchoring to `now + period` when it falls
// behind -- keeps the PERIOD but loses the PHASE at every re-anchor, and then holds the wrong
// phase forever. Two ways in, neither exotic: a loop whose first deadline is already stale
// because the grid was anchored earlier by someone else, and any wait that overruns by more than
// a period (which the Win32SleepFallback path does by 14-30 ms whenever the high-resolution timer
// is unavailable). Snapping to the grid instead cannot drift in phase or period, however late it
// is called. Caught in review of the #3024 vblank pump, where the consumers of two vblank clocks
// can compare them and phase agreement is the point.
//
// Returns ONE boundary, never a backlog: a caller that has missed several abandons them. Pacing
// consumers respond to events arriving, so replaying missed slots back to back hands them a burst
// they read as several periods of progress in one instant.
constexpr uint64_t next_grid_deadline_ns(uint64_t origin_ns, uint64_t now_ns,
                                        uint64_t period_ns) {
    // A zero period has no boundaries; yielding `now_ns` makes the caller's wait a no-op rather
    // than dividing by zero. It still spins, and deliberately so -- silently substituting some
    // period would invent a rate nobody asked for.
    if (period_ns == 0) return now_ns;
    // A grid anchored in the future: the first boundary IS the origin. Reachable through the
    // videoout test seam, which can anchor the epoch to a fake time.
    if (now_ns < origin_ns) return origin_ns;
    return origin_ns + ((now_ns - origin_ns) / period_ns + 1) * period_ns;
}

// The number of grid boundaries strictly BETWEEN two successive next_grid_deadline_ns() results on
// the same (origin, period) grid: 0 for the ordinary case (each call lands exactly one period after
// the last), and >0 whenever a caller's wait overran far enough to skip boundaries outright (#3075).
//
// Deliberately NOT folded into next_grid_deadline_ns() itself, and this split is the actual fix for
// #3075, not a widened tolerance. next_grid_deadline_ns()'s whole contract -- see its own comment --
// is that its answer depends only on (origin, now, period) and never on how many boundaries were
// missed, which is exactly what makes it phase-exact under a stall of any length: correct behaviour
// for SCHEDULING, because a pacing loop must resume on the next real boundary rather than replay a
// backlog. But that same contract means the scheduler's return value carries no trace of a drop, so
// before this function existed there was no second code path to disagree with it either -- nothing
// anywhere computed "how many boundaries did we just skip", which is why a half-rate pump reported
// perfect phase and nothing else. This function computes exactly that, from two of the scheduler's
// own return values, without changing what the scheduler returns or how it is chosen: the schedule
// stays phase-exact, and the MEASUREMENT of it stops discarding the fault.
constexpr uint64_t grid_boundaries_missed(uint64_t prev_deadline_ns, uint64_t next_deadline_ns,
                                           uint64_t period_ns) {
    if (period_ns == 0 || next_deadline_ns <= prev_deadline_ns) return 0;
    const uint64_t advanced = (next_deadline_ns - prev_deadline_ns) / period_ns;
    return advanced > 0 ? advanced - 1 : 0;
}

// ---------------------------------------------------------------------------
// scePthreadSemTimedwait's poll-loop arithmetic (#3069).
//
// Extracted UNCHANGED from the Windows poll loop in hle_kernel.cpp's k_sem_timedwait, for the same
// reason next_grid_deadline_ns and sleep_until_deadline_retry above were extracted: it was inline in
// an HLE() body, so the only way to reach it was to call the real HLE and time it -- and
// test_kernel_sem_timedwait.cpp's own header records at length why a wall-clock duration cannot
// discriminate anything here (a quantized wait returns at the next tick BOUNDARY, so the defect's
// latency distribution overlaps the fix's, for any tick length, and no single-run bound separates
// them). Two of the three pieces below are saturating arms guarding guest-controlled input, and no
// test reached either.
//
// Everything here is pure and takes the clock reading as a PARAMETER, so a test drives the whole
// loop with a fake clock: no _WIN32, no real sleep, no wall time. See test_precise_sleep.cpp.

// Saturating microseconds -> nanoseconds. `timeout_us` arrives straight from the guest, and a wrap
// here would turn a very long timeout into a short one. Matches guest_ns_from in
// hle_kernel_time.cpp (#3022), which states the rule for the whole family.
constexpr uint64_t timeout_ns_from_us(uint64_t timeout_us) {
    constexpr uint64_t kNsMax = ~0ull;
    return timeout_us > kNsMax / 1000ull ? kNsMax : timeout_us * 1000ull;
}

// Saturating start + timeout. A wrapped deadline lands in the PAST, which turns a long wait into an
// instant timeout -- the loud direction, but still wrong.
constexpr uint64_t deadline_ns_from(uint64_t start_ns, uint64_t timeout_ns) {
    constexpr uint64_t kNsMax = ~0ull;
    return timeout_ns > kNsMax - start_ns ? kNsMax : start_ns + timeout_ns;
}

// The poll cadence. 500 us while the wait is young, 2 ms after 8 ms have passed; the crossover is
// above one 5.33 ms audio grain period so a pacing wait never leaves the tight regime.
//
// 500 us and not less, because that is the FLOOR of a short high-resolution-timer sleep on this
// host, measured rather than assumed: requests of 0.05, 0.10 and 0.20 ms all cost a median of
// 0.52 ms (200 samples each; 0.50 ms -> 1.02, 1.00 -> 1.51, 2.00 -> 2.07). An earlier revision asked
// for 200 us, which reads as a tighter loop than the machine can actually run and would mislead the
// next person tuning it. What the fine window buys is a poll interval about 4x shorter than the
// coarse one (0.52 ms against 2.07), not the 10x the constants imply.
constexpr uint64_t kSemPollFineSliceNs   =  500000ull;   // 500 us
constexpr uint64_t kSemPollCoarseSliceNs = 2000000ull;   // 2 ms
constexpr uint64_t kSemPollFineWindowNs  = 8000000ull;   // 8 ms

// How long to sleep before polling the semaphore again, given how long the wait has already run and
// how much of its timeout is left.
//
// THE CLAMP AGAINST `remaining_ns` IS THE LOAD-BEARING HALF. Without it the loop would sleep past
// the deadline and report a timeout up to one coarse slice late; and near the representable maximum
// the caller's `now_ns + slice` would wrap outright, landing the next wake in the past. With it the
// result is always <= remaining_ns, so `now_ns + poll_slice_ns(...)` can never exceed the deadline
// and therefore can never overflow.
//
// Exposed separately from sem_poll_step() below -- its only caller -- so the clamp can be
// mutation-tested on its own rather than only through the composed step.
constexpr uint64_t poll_slice_ns(uint64_t elapsed_ns, uint64_t remaining_ns) {
    const uint64_t slice =
        elapsed_ns < kSemPollFineWindowNs ? kSemPollFineSliceNs : kSemPollCoarseSliceNs;
    return remaining_ns < slice ? remaining_ns : slice;
}

// One iteration of the poll loop's timing decision, taken after a trywait has already failed.
struct SemPollStep {
    bool expired;              // the deadline has passed; the caller reports ETIMEDOUT
    uint64_t sleep_until_ns;   // otherwise sleep until here, then poll again. Unset if `expired`.
};

// Two invariants hold for every non-expired result, and they are what the tests assert:
//
//     now_ns < sleep_until_ns <= deadline_ns
//
// The LOWER bound is strict, so the loop can never busy-spin while time remains (a zero-length sleep
// would poll at whatever rate the scheduler allows). The UPPER bound is the clamp, so it can never
// overshoot the deadline nor overflow. Both survive an early wakeup: the decision is recomputed from
// `now_ns` alone on each iteration and carries nothing forward, so a sleep that returns short simply
// produces another step from wherever it actually landed.
constexpr SemPollStep sem_poll_step(uint64_t start_ns, uint64_t deadline_ns, uint64_t now_ns) {
    if (now_ns >= deadline_ns) return {true, 0};
    // NOTE: on a clock that went BACKWARDS, `now_ns - start_ns` wraps to a huge value and so selects
    // the coarse slice. That is the behaviour of the code this was extracted from, preserved
    // deliberately rather than corrected, because #3069 is a pure extraction. It is unreachable
    // through the real call site -- start_ns and now_ns are both steady_clock reads, which cannot go
    // backwards -- and benign if it ever were: the loop polls at 2 ms instead of 500 us and STILL
    // times out at exactly the right moment, because the deadline check above does not involve
    // start_ns at all. Pinned by a test arm so a later change to the elapsed computation is a
    // deliberate one.
    const uint64_t elapsed_ns = now_ns - start_ns;
    return {false, now_ns + poll_slice_ns(elapsed_ns, deadline_ns - now_ns)};
}

// ---------------------------------------------------------------------------
// The RELATIVE-microsecond timed waits' deadline arithmetic (#3056).
//
// scePthreadCondTimedwait, scePthreadMutexTimedlock and scePthreadRwlockTimedrd/wrlock all take a
// RELATIVE microsecond scalar and have to turn it into an absolute deadline; the Windows condition
// wait then has to turn that absolute deadline BACK into the whole-millisecond count WaitOnAddress
// accepts. Both halves used to be inline -- the first in hle_kernel.cpp's abs_deadline_us(), the
// second inside sync_futex.cpp's `#ifdef _WIN32` branch, which a POSIX host cannot even compile.
// Extracted here for the same reason, and tested the same way, as sem_poll_step() above (#3069) and
// sleep_until_deadline_retry() (#3074): a fake clock reaches every case, where timing the real HLE
// reaches almost none of them.
//
// Fields are int64_t rather than a host `struct timespec`, deliberately. The guest is FreeBSD
// x86-64, where tv_nsec is 64-bit, while MinGW-w64 declares `long tv_nsec` -- 32 bits on Windows
// x64. #3038 records the truncation that costs the select()/pselect() callers; arithmetic written
// on explicit 64-bit fields cannot acquire it, and the same expression is then exercised on both
// platforms rather than only on the one that happens to be wide enough.
struct WaitDeadline {
    int64_t sec = 0;
    int64_t nsec = 0;   // [0, 1e9) on every value these functions RETURN; see normalize_wait_deadline
};

// Carry an out-of-range nanosecond field into the seconds, saturating rather than wrapping at
// either end. Every production caller hands in a clock_gettime() result, which is already
// normalized -- this exists so the functions below are TOTAL, i.e. so a test may drive them with a
// fake clock that is not, and so a garbage guest value cannot reach the arithmetic below unnormalized.
constexpr WaitDeadline normalize_wait_deadline(WaitDeadline t) {
    constexpr int64_t kNsPerSec = 1000000000ll;
    if (t.nsec >= 0 && t.nsec < kNsPerSec) return t;
    int64_t carry = t.nsec / kNsPerSec;
    int64_t nsec = t.nsec - carry * kNsPerSec;
    if (nsec < 0) { nsec += kNsPerSec; carry -= 1; }
    if (carry > 0 && t.sec > INT64_MAX - carry) return WaitDeadline{INT64_MAX, kNsPerSec - 1};
    if (carry < 0 && t.sec < INT64_MIN - carry) return WaitDeadline{INT64_MIN, 0};
    return WaitDeadline{t.sec + carry, nsec};
}

// `now` + `usec`, saturating at INT64_MAX seconds.
//
// The saturation is the rule guest_ns_from() (#3022) and timeout_ns_from_us() (#3069) already state
// for this family: the microsecond count arrives straight from a guest register, and a wrap turns a
// very long wait into an instant one. On this platform the addition happens to be unreachable --
// UINT64_MAX microseconds is 5.8e5 years of seconds against int64's 2.9e11 -- so this arm is a
// consistency guard, not a live defect, and saying which it is matters more than having it.
constexpr WaitDeadline abs_deadline_from_rel_us(WaitDeadline now, uint64_t usec) {
    constexpr int64_t kNsPerSec = 1000000000ll;
    constexpr WaitDeadline kSaturated{INT64_MAX, kNsPerSec - 1};
    const WaitDeadline base = normalize_wait_deadline(now);
    // < 1e9 + 1e9, so this cannot overflow whatever the guest passed.
    int64_t nsec = base.nsec + (int64_t)(usec % 1000000ull) * 1000ll;
    int64_t sec = base.sec;
    if (nsec >= kNsPerSec) {
        nsec -= kNsPerSec;
        if (sec == INT64_MAX) return kSaturated;
        sec += 1;
    }
    const uint64_t whole_seconds = usec / 1000000ull;
    if (sec >= 0 && whole_seconds > (uint64_t)(INT64_MAX - sec)) return kSaturated;
    return WaitDeadline{sec + (int64_t)whole_seconds, nsec};
}

// Whole milliseconds from `now` to `deadline`, ROUNDED UP, saturating at `max_ms`, and exactly zero
// when the deadline is already at or in the past. Feeds WaitOnAddress's DWORD timeout.
//
// ROUNDING UP IS THE LOAD-BEARING HALF, and it is a contract rather than a preference:
// pthread_cond_timedwait must not report ETIMEDOUT before its deadline has passed, so the conversion
// may never round a positive remainder down to zero. It is also the whole of the sub-millisecond
// overshoot this call has on Windows -- a 0.818 ms request becomes a 1 ms wait (#3056) -- which is
// worth knowing is INTENTIONAL before anyone reads the census ratio as an arithmetic bug.
//
// `max_ms` is a parameter rather than INFINITE-1 baked in, so the saturation is reachable from a
// test without INFINITE (a Windows macro) appearing in this header.
constexpr uint64_t wait_timeout_ms_ceil(WaitDeadline now, WaitDeadline deadline, uint64_t max_ms) {
    constexpr int64_t kNsPerSec = 1000000000ll;
    const WaitDeadline a = normalize_wait_deadline(now);
    const WaitDeadline b = normalize_wait_deadline(deadline);
    if (b.sec < a.sec) return 0;
    // Unsigned subtraction of the two's-complement patterns is the EXACT difference here, and is
    // well-defined for every pair: b.sec >= a.sec, so the mathematical difference is in [0, 2^64),
    // where the signed subtraction would overflow for a span wider than INT64_MAX.
    uint64_t seconds = (uint64_t)b.sec - (uint64_t)a.sec;
    int64_t nanos = b.nsec - a.nsec;            // both normalized, so in (-1e9, 1e9)
    if (nanos < 0) {
        if (seconds == 0) return 0;             // deadline strictly in the past
        nanos += kNsPerSec;
        seconds -= 1;
    }
    // Bound BEFORE multiplying, so `seconds * 1000` cannot overflow for any span...
    if (seconds > max_ms / 1000ull) return max_ms;
    const uint64_t whole_ms = seconds * 1000ull;                       // <= max_ms, by the bound
    const uint64_t frac_ms = ((uint64_t)nanos + 999999ull) / 1000000ull;   // 0..1000
    // ...and bound the SUM the same way rather than adding and clamping afterwards. A trailing
    // `ms > max_ms ? max_ms : ms` looks equivalent and is not: for a max_ms near UINT64_MAX the
    // addition itself wraps, and the clamp then reads a tiny number as being under the limit.
    // Unreachable from the one production caller, which passes INFINITE-1 -- but a function whose
    // whole purpose is to saturate should not have a reachable-only-by-luck wrap in it (raised in
    // review of #3235).
    return frac_ms > max_ms - whole_ms ? max_ms : whole_ms + frac_ms;
}

}  // namespace prosper::host
