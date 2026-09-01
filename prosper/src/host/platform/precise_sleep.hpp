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

}  // namespace prosper::host
