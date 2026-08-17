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

}  // namespace prosper::host
