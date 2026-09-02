// test_kernel_cond_timedwait_clock (#3056) — scePthreadCondTimedwait measures its RELATIVE
// microsecond timeout against the CONDITION VARIABLE'S OWN CLOCK.
//
// The Sony spelling takes a relative microsecond scalar and used to convert it with a hardcoded
// CLOCK_REALTIME reading, while its POSIX sibling scePthread/pthread_cond_timedwait resolved
// SCE_KERNEL_CLOCK_VIRTUAL / _PROF / _MONOTONIC through the condvar's registered clock id. Same
// object, same question, two answers depending on which name the guest called — the divergence
// class #1873 records between two libraries, here between two spellings of one.
//
// WHY THE DISCRIMINATOR IS THE **VIRTUAL** CLOCK AND NOT A DURATION.
//
// For SCE_KERNEL_CLOCK_REALTIME the old path was self-consistent (a realtime deadline handed to a
// realtime wait), and for _MONOTONIC the two clocks only diverge across a wall-clock STEP, which a
// test cannot induce. So neither can separate the fix from the defect at all. _VIRTUAL is process
// USER CPU time, which advances by microseconds while a thread is parked — so a request for 300 ms
// of it cannot be satisfied inside a second of wall time by an idle process, and the two
// implementations disagree about something DISCRETE:
//
//     defect  — the deadline is 300 ms of WALL time, so the wait times out and returns
//               SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c) at roughly 300 ms;
//     fixed   — the deadline is 300 ms of CPU time, which an idle process has not spent, so the
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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;
using clk = std::chrono::steady_clock;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Sony condattr clock ids, as scePthreadCondattrSetclock validates them (hle_kernel.cpp).
static constexpr uint64_t kSceKernelClockRealtime = 0;
static constexpr uint64_t kSceKernelClockVirtual  = 1;

int main() {
    printf("== test_kernel_cond_timedwait_clock ==\n");
    register_builtin_hle();

    HleFn mutex_init  = Hle::lookup(nid_hash("scePthreadMutexInit").c_str());
    HleFn mutex_lock  = Hle::lookup(nid_hash("scePthreadMutexLock").c_str());
    HleFn mutex_unlock= Hle::lookup(nid_hash("scePthreadMutexUnlock").c_str());
    HleFn attr_init   = Hle::lookup(nid_hash("scePthreadCondattrInit").c_str());
    HleFn attr_clock  = Hle::lookup(nid_hash("scePthreadCondattrSetclock").c_str());
    HleFn cond_init   = Hle::lookup(nid_hash("scePthreadCondInit").c_str());
    HleFn timedwait   = Hle::lookup(nid_hash("scePthreadCondTimedwait").c_str());
    HleFn cond_signal = Hle::lookup(nid_hash("scePthreadCondSignal").c_str());

    CHECK(mutex_init && mutex_lock && mutex_unlock, "the guest mutex entry points are registered");
    CHECK(attr_init && attr_clock && cond_init, "the condattr and cond init entry points are registered");
    CHECK(timedwait && cond_signal, "scePthreadCondTimedwait and scePthreadCondSignal are registered");
    if (!mutex_init || !mutex_lock || !mutex_unlock || !attr_init || !attr_clock ||
        !cond_init || !timedwait || !cond_signal) {
        printf("== FAIL (unresolved) ==\n");
        return 1;
    }

    // Every guest handle here is a POINTER CELL that the init handler writes through, not the
    // object itself — `k_cond_init` does `*(void**)a0 = cond`. Same shape as the semaphore slot in
    // test_kernel_sem_timedwait.cpp, and for the same reason it must be pointer-sized.
    void* mutex_slot = nullptr;
    void* cond_slot = nullptr;
    void* attr_slot = nullptr;
    const uint64_t mutex_h = (uint64_t)(uintptr_t)&mutex_slot;
    const uint64_t cond_h  = (uint64_t)(uintptr_t)&cond_slot;
    const uint64_t attr_h  = (uint64_t)(uintptr_t)&attr_slot;

    CHECK(mutex_init(mutex_h, 0, 0, 0, 0, 0) == 0, "scePthreadMutexInit succeeds");

    // ---------------------------------------------------------------------
    // ARM 1 — the DEFAULT realtime path still times out and still encodes as the guest reads it.
    //
    // NOT a discriminator, and said so plainly: this arm passes both before and after the fix,
    // because for SCE_KERNEL_CLOCK_REALTIME the fixed code takes the identical route
    // (interruptible_cond_clock_timedwait forwards a realtime deadline straight to
    // interruptible_cond_timedwait). Its job is to pin that the common case — every title that
    // never sets a condattr clock, including the Blasphemous 2 census #3056 was filed from — did
    // not change, and to pin the 0x8002003c encoding, which nothing else in the suite covers for
    // this entry point.
    {
        void* rt_attr_slot = nullptr;
        void* rt_cond_slot = nullptr;
        const uint64_t rt_attr_h = (uint64_t)(uintptr_t)&rt_attr_slot;
        const uint64_t rt_cond_h = (uint64_t)(uintptr_t)&rt_cond_slot;
        CHECK(attr_init(rt_attr_h, 0, 0, 0, 0, 0) == 0, "scePthreadCondattrInit succeeds");
        CHECK(attr_clock(rt_attr_h, kSceKernelClockRealtime, 0, 0, 0, 0) == 0,
              "scePthreadCondattrSetclock accepts SCE_KERNEL_CLOCK_REALTIME");
        CHECK(cond_init(rt_cond_h, rt_attr_h, 0, 0, 0, 0) == 0,
              "scePthreadCondInit succeeds with a realtime condattr");

        CHECK(mutex_lock(mutex_h, 0, 0, 0, 0, 0) == 0, "the guest holds the mutex before waiting");
        const auto t0 = clk::now();
        const uint64_t rc = timedwait(rt_cond_h, mutex_h, 20000, 0, 0, 0);   // 20 ms, nothing signals
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        CHECK(mutex_unlock(mutex_h, 0, 0, 0, 0, 0) == 0, "and holds it again on return");

        CHECK(rc == prosper::hle::kSceKernelErrorETIMEDOUT,
              "a realtime timeout returns SCE_KERNEL_ERROR_ETIMEDOUT (0x8002003c), not FreeBSD 60");
        // A stub/unit guard, not the discriminator: it catches a handler that returns instantly or
        // one that read the microseconds as some other unit. The upper bound is deliberately loose.
        CHECK(ms >= 15.0, "it actually waited (a handler returning at once fails this)");
        CHECK(ms < 2000.0, "and on the right order of magnitude (a wrong unit or a hang fails this)");
        printf("         (realtime 20 ms timeout took %.2f ms)\n", ms);
    }

    // ---------------------------------------------------------------------
    // ARM 2 — THE DISCRIMINATOR. A SCE_KERNEL_CLOCK_VIRTUAL condvar's relative timeout is spent in
    // process CPU time, so an idle process cannot reach it and the wait is still parked when the
    // signal lands.
    CHECK(attr_init(attr_h, 0, 0, 0, 0, 0) == 0, "scePthreadCondattrInit succeeds");
    CHECK(attr_clock(attr_h, kSceKernelClockVirtual, 0, 0, 0, 0) == 0,
          "scePthreadCondattrSetclock accepts SCE_KERNEL_CLOCK_VIRTUAL");
    CHECK(cond_init(cond_h, attr_h, 0, 0, 0, 0) == 0,
          "scePthreadCondInit records the virtual clock on the condvar");
    {
        constexpr uint64_t kTimeoutUs = 300000;      // 300 ms of PROCESS CPU TIME
        constexpr int kSignalAfterMs = 600;          // ...signalled at twice that in WALL time

        std::atomic<bool> entered{false};
        std::atomic<uint64_t> rc{~0ull};
        std::atomic<double> waited_ms{0.0};

        std::thread waiter([&] {
            mutex_lock(mutex_h, 0, 0, 0, 0, 0);
            entered.store(true, std::memory_order_release);
            const auto t0 = clk::now();
            const uint64_t r = timedwait(cond_h, mutex_h, kTimeoutUs, 0, 0, 0);
            waited_ms.store(std::chrono::duration<double, std::milli>(clk::now() - t0).count(),
                            std::memory_order_relaxed);
            rc.store(r, std::memory_order_release);
            mutex_unlock(mutex_h, 0, 0, 0, 0, 0);
        });

        // Wait for the worker to have taken the mutex before starting the clock, so the signal
        // below cannot land before the wait exists. The wait itself releases the mutex, which is
        // what lets the signal path take it.
        while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(kSignalAfterMs));

        // Signal under the mutex, the ordinary discipline: it closes the window where the waiter is
        // between two slices of the clock-conversion loop and would otherwise miss the wake.
        // (guest_cond_advance's generation counter closes it too — this is belt and braces.)
        mutex_lock(mutex_h, 0, 0, 0, 0, 0);
        cond_signal(cond_h, 0, 0, 0, 0, 0);
        mutex_unlock(mutex_h, 0, 0, 0, 0, 0);
        waiter.join();

        const uint64_t result = rc.load(std::memory_order_acquire);
        const double ms = waited_ms.load(std::memory_order_relaxed);
        printf("         (virtual 300 ms CPU-time wait returned 0x%llx after %.2f ms of wall time)\n",
               (unsigned long long)result, ms);

        // THE ARM. Pre-fix this is SCE_KERNEL_ERROR_ETIMEDOUT at ~300 ms, because the 300 ms was
        // spent against the wall clock; post-fix it is 0, because 300 ms of process CPU time has
        // not elapsed and the signal arrived first.
        CHECK(result == 0,
              "a SCE_KERNEL_CLOCK_VIRTUAL timeout is spent in CPU time, so the idle wait is "
              "signalled rather than timed out");
        CHECK(result != prosper::hle::kSceKernelErrorETIMEDOUT,
              "...and specifically did NOT time out, which is what the wall-clock deadline did");
        // A STUB/HANG GUARD, and explicitly NOT a second discriminator: the pre-fix code returns at
        // ~300 ms, which clears this bound comfortably (measured: it passes under the mutation while
        // the two arms above redden). Raising it to ~400 ms WOULD redden — and would then be a
        // wall-clock bound placed between two timing distributions, which is the shape
        // test_kernel_sem_timedwait.cpp's header records passing the defect silently. The return
        // value is the discriminator; this line only catches a handler that does not wait at all.
        CHECK(ms >= 50.0, "the wait actually blocked (a handler returning at once fails this)");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
