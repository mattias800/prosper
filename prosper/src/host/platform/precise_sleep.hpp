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

SleepBackend sleep_backend();

// A stable spelling of sleep_backend() for assertions and logs. Never null.
const char* sleep_backend_name();

// Advance an absolute periodic deadline after a wait on it has completed. `now_ns` is the time
// the wait actually returned; the result is the next deadline on the SAME grid, so overshoot
// does not accumulate the way it does when each iteration sleeps for a relative period.
//
// This is a free function with its own tests because of one subtlety that reads as correct and
// is not: after sleep_until_steady_ns(deadline_ns) returns, `now_ns` is normally ALREADY past
// deadline_ns by the wait's own small overshoot. So a "have we fallen behind?" test written
// against the OLD deadline is true on every ordinary iteration, and re-anchoring there silently
// costs one extra period per iteration -- halving the rate, while the surrounding code still
// says 60 Hz. The comparison has to be against the ADVANCED deadline. Caught in review of the
// #3024 vblank pump, where it would have turned a 33 Hz defect into a 30 Hz one.
constexpr uint64_t next_periodic_deadline_ns(uint64_t deadline_ns, uint64_t now_ns,
                                            uint64_t period_ns) {
    if (period_ns == 0) return now_ns;   // a zero period would busy-spin on a fixed deadline
    const uint64_t advanced = deadline_ns + period_ns;
    // A whole period late: abandon the missed slots and re-anchor to now. Deliberately NOT a
    // catch-up loop -- consumers pace on events ARRIVING, so replaying the missed ones back to
    // back hands them a burst they read as several periods of progress in one instant.
    return advanced < now_ns ? now_ns + period_ns : advanced;
}

}  // namespace prosper::host
