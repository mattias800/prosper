// precise_sleep — block until a steady-clock deadline with a wait primitive whose resolution does
// not depend on the process's Win32 timer period (#1765).
//
// WHY THIS IS NOT `std::this_thread::sleep_until`, on Windows only. Three facts compose:
//
//  1. libstdc++'s `sleep_until` on a STEADY clock forwards to `sleep_for(deadline - now)` and
//     returns — it does not re-check the clock (`bits/this_thread_sleep.h`).
//  2. On mingw-w64 libstdc++ is configured with `_GLIBCXX_USE_NANOSLEEP`, `_GLIBCXX_HAVE_SLEEP` and
//     `_GLIBCXX_HAVE_USLEEP` all UNdefined and `_GLIBCXX_USE_WIN32_SLEEP` defined, so `sleep_for`
//     ends in exactly one `::Sleep(DWORD milliseconds)`. Read from `bits/c++config.h`, and
//     confirmed against the shipped library: `libstdc++.a(thread.o)` names exactly one sleep-ish
//     import, `__imp_Sleep`, and `__sleep_for` tail-calls it.
//  3. Windows quantizes `::Sleep` to the system timer period, and nothing in this tree raises it
//     (`timeBeginPeriod` / `NtSetTimerResolution` appear nowhere outside `third_party/`). prosper's
//     own measurement on a Windows host — recorded in `tests/host/x86/test_win_exception_delivery.cpp` for
//     #2242 — is `Sleep(1)` = 15.554 ms by default against 1.930 ms after `timeBeginPeriod(1)`.
//
// So a ~16.68 ms display-pacing wait became `::Sleep(17)`, which cannot complete before the second
// ~15.6 ms tick: roughly 31 ms, i.e. a display thread paced at ~32 Hz instead of ~60. The waitable
// timer below is scheduled off the high-resolution timer wheel instead, so its expiry does not wait
// for the coarse tick and the process-wide timer period is left alone (raising it globally costs
// power and perturbs every other timer in the process).
//
// CONFIDENCE: HIGH on the mechanism — (1) and (2) are read from the toolchain's own headers,
// generated config and compiled library, (3) from a measurement already in this repository.
// CONFIDENCE: MED on the resulting numbers for any particular Windows host: the author could not
// execute on Windows, so the improvement from ~31 ms to ~16.7 ms per wait is derived rather than
// observed. The fallback keeps the previous behaviour if the timer cannot be created, so the worst
// case is what master already does.
#ifdef _WIN32
// Must precede EVERY header, including <chrono>/<thread>: those pull in MinGW's _mingw.h, which
// defaults _WIN32_WINNT to 0x0601, after which a later `#ifndef _WIN32_WINNT` is a silent no-op and
// CreateWaitableTimerExW goes undeclared. Same placement rule as hle_kernel.cpp, hle_kernel_mem.cpp
// and exec_image_win.cpp.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Windows 10 1803+. Defined defensively: an older mingw-w64 header set declares
// CreateWaitableTimerExW without this flag, and the value is part of the published contract.
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif  // _WIN32

#include "host/platform/precise_sleep.hpp"

#include <chrono>
#include <thread>

namespace prosper::host {
namespace {

thread_local SleepBackend g_backend = SleepBackend::None;

uint64_t steady_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sleep_until_with_std(uint64_t deadline_ns) {
    std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(deadline_ns))));
}

#ifdef _WIN32
// One auto-reset timer per waiting thread. Auto-reset so the wait itself consumes the signal and no
// stale signalled state can make the NEXT wait return instantly.
struct ThreadTimer {
    HANDLE handle = nullptr;
    ThreadTimer() {
        handle = CreateWaitableTimerExW(nullptr, nullptr,
                                        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                        TIMER_ALL_ACCESS);
    }
    ~ThreadTimer() { if (handle) CloseHandle(handle); }
    ThreadTimer(const ThreadTimer&) = delete;
    ThreadTimer& operator=(const ThreadTimer&) = delete;
};
thread_local ThreadTimer g_timer;
#endif

}  // namespace

void sleep_until_steady_ns(uint64_t deadline_ns) {
#ifdef _WIN32
    for (;;) {
        const uint64_t now = steady_now_ns();
        if (now >= deadline_ns) return;
        const uint64_t remaining_ns = deadline_ns - now;
        if (!g_timer.handle) break;   // no high-resolution timer on this system

        LARGE_INTEGER due;
        // NEGATIVE is a relative interval in 100 ns units; a positive value would be an absolute
        // FILETIME, i.e. a wait until the year 1601 — which returns instantly and would silently
        // turn this into a spin. Round the division UP so the timer can never expire early.
        due.QuadPart = -(LONGLONG)((remaining_ns + 99) / 100);
        if (due.QuadPart == 0) due.QuadPart = -1;
        // The timeout is a safety net, not the pacing mechanism: the timer object signals first on
        // every healthy wait, so this only bounds a wait that would otherwise hang. It is itself
        // tick-quantized, hence the generous slack.
        const uint64_t timeout_ms = remaining_ns / 1000000ull + 100ull;
        if (!SetWaitableTimer(g_timer.handle, &due, 0, nullptr, nullptr, FALSE)) break;
        // Only a real signal counts as the high-resolution path. WAIT_TIMEOUT means the safety net
        // fired instead, which is a tick-quantized wake and must not be labelled as the timer's --
        // it also means the deadline is long past, so the re-check below returns without sleeping
        // again.
        if (WaitForSingleObject(g_timer.handle,
                                timeout_ms > 0xFFFFFFFEull ? INFINITE : (DWORD)timeout_ms)
                != WAIT_OBJECT_0)
            break;
        g_backend = SleepBackend::Win32HighResolutionTimer;
        // Loop rather than return: a waitable timer may be released marginally early, and the
        // contract here is "do not come back before the deadline". The re-read of the clock makes
        // that true without trusting the primitive's own precision.
    }
    // Anything above that did not work out lands here — no worse than master's behaviour, and
    // reported distinctly so a test can tell the two apart rather than silently pacing at ~32 Hz.
    if (steady_now_ns() >= deadline_ns) return;
    sleep_until_with_std(deadline_ns);
    g_backend = SleepBackend::Win32SleepFallback;
#else
    // glibc: sleep_until -> sleep_for -> a nanosleep loop that also restarts on EINTR, which matters
    // because prosper delivers its own signals. Deliberately a single call, identical to what this
    // path did before precise_sleep existed.
    sleep_until_with_std(deadline_ns);
    g_backend = SleepBackend::PosixSleepUntil;
#endif
}

SleepBackend sleep_backend() { return g_backend; }

const char* sleep_backend_name() {
    switch (g_backend) {
        case SleepBackend::PosixSleepUntil:          return "posix-sleep-until";
        case SleepBackend::Win32HighResolutionTimer: return "win32-high-resolution-timer";
        case SleepBackend::Win32SleepFallback:       return "win32-sleep-fallback";
        case SleepBackend::None:                     break;
    }
    return "none";
}

}  // namespace prosper::host
