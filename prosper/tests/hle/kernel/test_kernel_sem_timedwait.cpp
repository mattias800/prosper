// test_kernel_sem_timedwait (#3013) - the guest semaphore timed wait.
//
// THIS TEST'S FIRST VERSION COULD NOT FAIL, which is why the mechanism arm below exists.
//
// It asserted wall-clock bounds: a 5 ms timeout under a 40 ms ceiling, and the latency of a POSTED
// semaphore. Review showed none of the three arms could redden on the pre-fix code. 40 ms sits 2.6x
// above the 15.57 ms defect it claimed to catch, so the tick passed it comfortably; and a posted
// semaphore is served by UNFIXED winpthreads in 0.01 ms - better than the poll loop that replaced it -
// so the wake arms preferred the defect. Meanwhile the file header and two arm comments all stated
// that the tick was pinned. Confident labels over assertions that could not discriminate.
//
// The fix is the pattern this repo already prescribes and uses. test_kernel_nanosleep.cpp says it in
// words - "precise_sleep exposes sleep_backend() so the MECHANISM can be asserted instead" - and
// test_videoout.cpp:274/:315 is the worked example: assert WHICH primitive served the wait, not how
// long it took. A mechanism assertion cannot be satisfied by a lucky duration, and it does not flake
// on a loaded host.
//
// So the arms are:
//   1. MECHANISM  - the wait was served by the high-resolution timer, not ::Sleep's tick. Reddens on
//                   the pre-fix code, which never touches precise_sleep at all.
//   2. ENCODING   - the timeout returns exactly 0x8002003c, FreeBSD ETIMEDOUT as the guest reads it.
//                   Nothing else in the suite covers this path's encoding.
//   3. BOUNDS     - a lower bound as well as an upper one. The upper alone cannot catch a wait that
//                   returns instantly; the lower alone cannot catch the tick.
//   4. FAST PATH  - an already-posted semaphore never enters the loop.
//
// Deliberately NOT asserted: the poll slice length. A short high-resolution sleep on this host has a
// ~0.52 ms floor, so the adaptive 0.5 ms slice and a flat 2 ms one produce overlapping latency
// distributions and no honest single-run bound separates them. That measurement lives at the call site.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"
#include "host/platform/precise_sleep.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <semaphore.h>
#include <thread>

using namespace prosper;
using clk = std::chrono::steady_clock;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_kernel_sem_timedwait ==\n");
    register_builtin_hle();

    HleFn sem_init_fn = Hle::lookup(nid_hash("scePthreadSemInit").c_str());
    HleFn timedwait   = Hle::lookup(nid_hash("scePthreadSemTimedwait").c_str());
    HleFn post        = Hle::lookup(nid_hash("scePthreadSemPost").c_str());
    CHECK(sem_init_fn != nullptr, "scePthreadSemInit is registered");
    CHECK(timedwait != nullptr, "scePthreadSemTimedwait is registered");
    CHECK(post != nullptr, "scePthreadSemPost is registered");
    if (!sem_init_fn || !timedwait || !post) { printf("== FAIL (unresolved) ==\n"); return 1; }

    // The guest handle is an opaque slot; ensure_sem backs it with a real sem_t. Initialised through
    // the HLE and CHECKED - the first version declared a bare `sem_t slot;` and called init with its
    // result ignored, so an init that silently failed would have left every arm below operating on an
    // uninitialised object.
    sem_t slot;
    const uint64_t handle = (uint64_t)(uintptr_t)&slot;
    CHECK(sem_init_fn(handle, 0, 0, 0, 0, 0) == 0, "the semaphore initialises to a count of 0");

    // ARM 1 + 2 + 3: one unposted wait, three independent properties.
    {
        // "none" first, so the mechanism assertion below cannot be satisfied by an accessor that is a
        // compiled-in constant. test_videoout.cpp uses the same guard for the same reason.
        CHECK(strcmp(host::sleep_backend_name(), "none") == 0,
              "no precise wait has run on this thread yet");

        const auto t0 = clk::now();
        const uint64_t rc = timedwait(handle, 5000, 0, 0, 0, 0);   // 5 ms, nothing will post
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

        // ARM 2: the encoding the guest actually compares against.
        CHECK(rc == prosper::hle::kSceKernelErrorETIMEDOUT,
              "a timeout returns FreeBSD ETIMEDOUT encoded as the guest reads it (0x8002003c)");

        // ARM 3: bounded on BOTH sides. An upper bound alone passes a wait that returns instantly;
        // a lower bound alone passes the 15.6 ms tick.
        CHECK(ms >= 4.0, "it actually waited (a stub returning at once fails this)");
        CHECK(ms < 12.0, "and returned inside one winpthreads tick of the request, not after it");

        // ARM 1: WHICH primitive served it. This is the arm that reddens on the pre-fix code, which
        // delegates to winpthreads and never enters precise_sleep, leaving the backend at "none".
#ifdef _WIN32
        CHECK(strcmp(host::sleep_backend_name(), "win32-high-resolution-timer") == 0,
              "the wait was served by the high-resolution timer, NOT ::Sleep's ~15.6 ms tick");
#else
        CHECK(strcmp(host::sleep_backend_name(), "posix-sleep-until") == 0,
              "POSIX keeps the native timed wait; no polling was introduced there");
#endif
        printf("         (timeout took %.2f ms via %s)\n", ms, host::sleep_backend_name());
    }

    // ARM 4: an already-posted semaphore is taken by the trywait fast path and never enters the loop.
    {
        CHECK(post(handle, 0, 0, 0, 0, 0) == 0, "post succeeds");
        const auto t0 = clk::now();
        const uint64_t rc = timedwait(handle, 1000000, 0, 0, 0, 0);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        CHECK(rc == 0, "an already-posted semaphore is acquired");
        CHECK(ms < 1.0, "...immediately, without entering the poll loop");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
