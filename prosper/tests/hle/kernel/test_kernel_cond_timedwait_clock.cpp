// test_kernel_cond_timedwait_clock (#3056) - scePthreadCondTimedwait measures its RELATIVE
// microsecond timeout against the CONDITION VARIABLE'S OWN CLOCK.
//
// The Sony spelling takes a relative microsecond scalar and used to convert it with a hardcoded
// CLOCK_REALTIME reading, while its POSIX sibling scePthread/pthread_cond_timedwait resolved
// SCE_KERNEL_CLOCK_VIRTUAL / _PROF / _MONOTONIC through the condvar's registered clock id. Same
// object, same question, two answers depending on which name the guest called - the divergence
// class #1873 records between two libraries, here between two spellings of one.
//
// WHY THE DISCRIMINATOR IS THE **VIRTUAL** CLOCK AND NOT A DURATION.
//
// For SCE_KERNEL_CLOCK_REALTIME the old path was self-consistent (a realtime deadline handed to a
// realtime wait), and for _MONOTONIC the two clocks only diverge across a wall-clock STEP, which a
// test cannot induce. So neither can separate the fix from the defect at all. _VIRTUAL is process
// USER CPU time, which advances by microseconds while a thread is parked - so a request for 300 ms
// of it cannot be satisfied inside a second of wall time by an idle process, and the two
// implementations disagree about something DISCRETE:
//
//     defect  - the deadline is 300 ms of WALL time, so the wait times out and returns
//               SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c) at roughly 300 ms;
//     fixed   - the deadline is 300 ms of CPU time, which an idle process has not spent, so the
//               wait is still parked when the signal arrives at ~600 ms and returns 0.
//
// The assertion is therefore `rc == 0` against `rc == ETIMEDOUT`, a return-value identity rather
// than a wall-clock bound. That matters here for the reason test_kernel_sem_timedwait.cpp's header
// spends forty lines on: a bound placed between two overlapping latency distributions passes the
// defect silently. There is no overlap between "signalled" and "timed out".
//
// The one way this could flake is the process genuinely spending 300 ms of USER CPU inside the
// ~600 ms window — impossible while both of its threads are blocked, and unaffected by load from
// OTHER processes, since getrusage(RUSAGE_SELF) / GetProcessTimes are per-process.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;
using clk = std::chrono::steady_clock;

namespace {
// Sony condattr clock ids, as scePthreadCondattrSetclock validates them (hle_kernel.cpp).
constexpr uint64_t kSceKernelClockRealtime = 0;
constexpr uint64_t kSceKernelClockVirtual = 1;

struct CondWaitFamily {
    HleFn mutex_init = nullptr, mutex_lock = nullptr, mutex_unlock = nullptr, attr_init = nullptr,
          attr_clock = nullptr, cond_init = nullptr, timedwait = nullptr, cond_signal = nullptr;
    bool complete() const {
        return mutex_init && mutex_lock && mutex_unlock && attr_init && attr_clock && cond_init &&
               timedwait && cond_signal;
    }
};
CondWaitFamily cond_wait_family() {
    register_builtin_hle();
    CondWaitFamily f;
    f.mutex_init = Hle::lookup(nid_hash("scePthreadMutexInit").c_str());
    f.mutex_lock = Hle::lookup(nid_hash("scePthreadMutexLock").c_str());
    f.mutex_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock").c_str());
    f.attr_init = Hle::lookup(nid_hash("scePthreadCondattrInit").c_str());
    f.attr_clock = Hle::lookup(nid_hash("scePthreadCondattrSetclock").c_str());
    f.cond_init = Hle::lookup(nid_hash("scePthreadCondInit").c_str());
    f.timedwait = Hle::lookup(nid_hash("scePthreadCondTimedwait").c_str());
    f.cond_signal = Hle::lookup(nid_hash("scePthreadCondSignal").c_str());
    return f;
}

uint64_t U(const void* p) {
    return reinterpret_cast<uint64_t>(p);
}

// Every guest handle here is a POINTER CELL that the init handler writes through, not the
// object itself — `k_cond_init` does `*(void**)a0 = cond`. Same shape as the semaphore slot in
// test_kernel_sem_timedwait.cpp, and for the same reason it must be pointer-sized.
struct GuestMutex {
    void* slot = nullptr;
    uint64_t handle() const { return U(&slot); }
};
}  // namespace

TEST(CondTimedwaitClock, TheGuestMutexCondattrAndWaitEntryPointsAreRegistered) {
    const CondWaitFamily f = cond_wait_family();
    EXPECT_TRUE(f.mutex_init && f.mutex_lock && f.mutex_unlock)
        << "the guest mutex entry points are registered";
    EXPECT_TRUE(f.attr_init && f.attr_clock && f.cond_init)
        << "the condattr and cond init entry points are registered";
    EXPECT_TRUE(f.timedwait && f.cond_signal)
        << "scePthreadCondTimedwait and scePthreadCondSignal are registered";
}

TEST(CondTimedwaitClock, TheDefaultRealtimePathStillTimesOutWithTheGuestEncoding) {
    // ARM 1 — the DEFAULT realtime path still times out and still encodes as the guest reads it.
    //
    // NOT a discriminator, and said so plainly: this arm passes both before and after the fix,
    // because for SCE_KERNEL_CLOCK_REALTIME the fixed code takes the identical route
    // (interruptible_cond_clock_timedwait forwards a realtime deadline straight to
    // interruptible_cond_timedwait). Its job is to pin that the common case — every title that
    // never sets a condattr clock, including the Blasphemous 2 census #3056 was filed from — did
    // not change, and to pin the 0x8002003c encoding, which nothing else in the suite covers for
    // this entry point.
    const CondWaitFamily f = cond_wait_family();
    ASSERT_TRUE(f.complete());
    GuestMutex mutex;
    void* rt_attr_slot = nullptr;
    void* rt_cond_slot = nullptr;
    const uint64_t rt_attr_h = U(&rt_attr_slot);
    const uint64_t rt_cond_h = U(&rt_cond_slot);
    ASSERT_EQ(f.mutex_init(mutex.handle(), 0, 0, 0, 0, 0), 0u) << "scePthreadMutexInit succeeds";
    ASSERT_EQ(f.attr_init(rt_attr_h, 0, 0, 0, 0, 0), 0u) << "scePthreadCondattrInit succeeds";
    ASSERT_EQ(f.attr_clock(rt_attr_h, kSceKernelClockRealtime, 0, 0, 0, 0), 0u)
        << "scePthreadCondattrSetclock accepts SCE_KERNEL_CLOCK_REALTIME";
    ASSERT_EQ(f.cond_init(rt_cond_h, rt_attr_h, 0, 0, 0, 0), 0u)
        << "scePthreadCondInit succeeds with a realtime condattr";

    ASSERT_EQ(f.mutex_lock(mutex.handle(), 0, 0, 0, 0, 0), 0u)
        << "the guest holds the mutex before waiting";
    const auto t0 = clk::now();
    const uint64_t rc =
        f.timedwait(rt_cond_h, mutex.handle(), 20000, 0, 0, 0);  // 20 ms, no signal
    const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    EXPECT_EQ(f.mutex_unlock(mutex.handle(), 0, 0, 0, 0, 0), 0u) << "and holds it again on return";

    EXPECT_EQ(rc, prosper::hle::kSceKernelErrorETIMEDOUT)
        << "a realtime timeout returns SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c), not FreeBSD 60";
    // A stub/unit guard, not the discriminator: it catches a handler that returns instantly or
    // one that read the microseconds as some other unit. The upper bound is deliberately loose.
    EXPECT_GE(ms, 15.0) << "it actually waited (a handler returning at once fails this)";
    EXPECT_LT(ms, 2000.0)
        << "and on the right order of magnitude (a wrong unit or a hang fails this)";
    std::printf("         (realtime 20 ms timeout took %.2f ms)\n", ms);
}

TEST(CondTimedwaitClock, AVirtualClockTimeoutIsSpentInCpuTimeSoTheIdleWaitIsSignalled) {
    // ARM 2 — THE DISCRIMINATOR. A SCE_KERNEL_CLOCK_VIRTUAL condvar's relative timeout is spent in
    // process CPU time, so an idle process cannot reach it and the wait is still parked when the
    // signal lands.
    const CondWaitFamily f = cond_wait_family();
    ASSERT_TRUE(f.complete());
    GuestMutex mutex;
    void* attr_slot = nullptr;
    void* cond_slot = nullptr;
    const uint64_t attr_h = U(&attr_slot);
    const uint64_t cond_h = U(&cond_slot);
    ASSERT_EQ(f.mutex_init(mutex.handle(), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.attr_init(attr_h, 0, 0, 0, 0, 0), 0u) << "scePthreadCondattrInit succeeds";
    ASSERT_EQ(f.attr_clock(attr_h, kSceKernelClockVirtual, 0, 0, 0, 0), 0u)
        << "scePthreadCondattrSetclock accepts SCE_KERNEL_CLOCK_VIRTUAL";
    ASSERT_EQ(f.cond_init(cond_h, attr_h, 0, 0, 0, 0), 0u)
        << "scePthreadCondInit records the virtual clock on the condvar";

    constexpr uint64_t kTimeoutUs = 300000;  // 300 ms of PROCESS CPU TIME
    constexpr int kSignalAfterMs = 600;      // ...signalled at twice that in WALL time

    std::atomic<bool> entered{false};
    std::atomic<uint64_t> rc{~0ull};
    std::atomic<double> waited_ms{0.0};

    std::thread waiter([&] {
        f.mutex_lock(mutex.handle(), 0, 0, 0, 0, 0);
        entered.store(true, std::memory_order_release);
        const auto t0 = clk::now();
        const uint64_t r = f.timedwait(cond_h, mutex.handle(), kTimeoutUs, 0, 0, 0);
        waited_ms.store(std::chrono::duration<double, std::milli>(clk::now() - t0).count(),
                        std::memory_order_relaxed);
        rc.store(r, std::memory_order_release);
        f.mutex_unlock(mutex.handle(), 0, 0, 0, 0, 0);
    });

    // Wait for the worker to have taken the mutex before starting the clock, so the signal
    // below cannot land before the wait exists. The wait itself releases the mutex, which is
    // what lets the signal path take it.
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(kSignalAfterMs));

    // Signal under the mutex, the ordinary discipline: it closes the window where the waiter is
    // between two slices of the clock-conversion loop and would otherwise miss the wake.
    // (guest_cond_advance's generation counter closes it too — this is belt and braces.)
    f.mutex_lock(mutex.handle(), 0, 0, 0, 0, 0);
    f.cond_signal(cond_h, 0, 0, 0, 0, 0);
    f.mutex_unlock(mutex.handle(), 0, 0, 0, 0, 0);
    waiter.join();

    const uint64_t result = rc.load(std::memory_order_acquire);
    const double ms = waited_ms.load(std::memory_order_relaxed);
    std::printf(
        "         (virtual 300 ms CPU-time wait returned 0x%llx after %.2f ms of wall time)\n",
        (unsigned long long)result, ms);

    // THE ARM. Pre-fix this is SCE_KERNEL_ERROR_ETIMEDOUT at ~300 ms, because the 300 ms was
    // spent against the wall clock; post-fix it is 0, because 300 ms of process CPU time has
    // not elapsed and the signal arrived first.
    EXPECT_EQ(result, 0u)
        << "a SCE_KERNEL_CLOCK_VIRTUAL timeout is spent in CPU time, so the idle wait is "
           "signalled rather than timed out";
    EXPECT_NE(result, prosper::hle::kSceKernelErrorETIMEDOUT)
        << "...and specifically did NOT time out, which is what the wall-clock deadline did";
    // A STUB/HANG GUARD, and explicitly NOT a second discriminator: the pre-fix code returns at
    // ~300 ms, which clears this bound comfortably (measured: it passes under the mutation while
    // the two arms above redden). Raising it to ~400 ms WOULD redden — and would then be a
    // wall-clock bound placed between two timing distributions, which is the shape
    // test_kernel_sem_timedwait.cpp's header records passing the defect silently. The return
    // value is the discriminator; this line only catches a handler that does not wait at all.
    EXPECT_GE(ms, 50.0) << "the wait actually blocked (a handler returning at once fails this)";
}