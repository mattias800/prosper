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
//   3. BOUNDS     - a stub/unit/hang guard, and NOT the discriminator. With the fix reverted the
//                   defect delivered at 14.69, 12.06, 17.75, 10.00 and 15.32 ms -- so a 12 ms
//                   ceiling would have PASSED the defect, not merely flaked. A quantized wait
//                   returns at the next tick boundary, so a 5 ms request lands anywhere in
//                   ~[5, 20.6] ms and the two distributions overlap. The same 12 ms ceiling
//                   also flaked on macOS/Rosetta at 18.54 ms, where the native call is
//                   untouched by this change anyway. Arm 1 caught all five.
//   4. FAST PATH  - an already-posted semaphore never enters the loop.
//
// The header being right about the arms is part of the test. Its first version was wrong about
// them in the same breath as the arms were wrong, and the confident label is what stopped anyone
// re-deriving it -- so a bound that changes changes this list too.
//
// Deliberately NOT asserted: the poll slice length. A short high-resolution sleep on this host has a
// ~0.52 ms floor, so the adaptive 0.5 ms slice and a flat 2 ms one produce overlapping latency
// distributions and no honest single-run bound separates them. That measurement lives at the call site.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"
#include "host/platform/precise_sleep.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <semaphore.h>

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
    // Asserts that init REPORTS success, not that the count is observably 0 -- the count is then
    // established by arm 1 timing out rather than being acquired. The earlier message claimed the
    // stronger thing.
    CHECK(sem_init_fn(handle, 0, 0, 0, 0, 0) == 0, "scePthreadSemInit reports success");

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

        // ARM 3: BOUNDS, and they are deliberately NOT the discriminator. This is the second
        // correction to this arm and the reasoning is measured rather than argued.
        //
        // The first version used a 40 ms ceiling, 2.6x above the defect, so it could not fail.
        // The second used 12 ms, which reddened -- and then flaked on macOS/Rosetta at 18.54 ms,
        // because on POSIX the native call is untouched by this change and the ceiling was purely
        // an assertion about the host's scheduler.
        //
        // Tightening it per-platform was the obvious next move, and it is wrong in the direction
        // that matters: it lets the defect THROUGH. Measured by reverting the fix and running
        // five times -- 14.69, 12.06, 17.75, 10.00, 15.32 ms. The 10.00 ms run passes a 12 ms
        // ceiling, so that bound was not merely fragile, it was unsound. It has to be: a
        // quantized wait returns at the next tick BOUNDARY, so a 5 ms request lands anywhere in
        // roughly [5, 20.6] ms depending on where it falls within the ~15.6 ms tick. The
        // defect's timing distribution OVERLAPS the fix's, so no single-run wall-clock bound
        // separates them on any platform. The mechanism arm caught all five.
        //
        // So the bounds are a stub/unit/hang guard on both platforms, and the MECHANISM arm below
        // is the discriminator -- which is what this file's header has said from the start, and
        // what the repo's own guidance says: assert which primitive served the wait, because that
        // reads a state variable instead of a clock and cannot flake.
        CHECK(ms >= 4.0, "it actually waited (a stub returning at once fails this)");
        CHECK(ms < 500.0, "and on the right order of magnitude (a wrong unit or a hang fails this)");

        // ARM 1: WHICH primitive served it. This is the arm that reddens on the pre-fix code, which
        // delegates to winpthreads and never enters precise_sleep, leaving the backend at "none".
#ifdef _WIN32
        CHECK(strcmp(host::sleep_backend_name(), "win32-high-resolution-timer") == 0,
              "the wait was served by the high-resolution timer, NOT ::Sleep's ~15.6 ms tick");
#else
        // STILL "none" on POSIX, and that is the assertion. The POSIX branch is the native
        // sem_timedwait and never enters sleep_until_steady_ns, so the backend is unchanged by the
        // wait. An earlier revision asserted "posix-sleep-until" here -- which, with the "none"
        // pre-check above, asserted two different values for one unchanged state, so ONE of the two
        // arms failed on every POSIX host regardless of the starting value. A Windows-only run
        // cannot see that, and Linux/macOS CI had not yet built this file.
        //
        // Asserting "none" is not a weaker check for being the unchanged value: it is a real guard
        // against someone later unifying both platforms onto the poll loop, which would enter
        // precise_sleep here and redden this line.
        CHECK(strcmp(host::sleep_backend_name(), "none") == 0,
              "POSIX keeps the native timed wait; no polling was introduced there");
#endif
        printf("         (timeout took %.2f ms via %s)\n", ms, host::sleep_backend_name());
    }

    // ARM 4: an already-posted semaphore is taken by the trywait fast path and never enters the loop.
    {
        CHECK(post(handle, 0, 0, 0, 0, 0) == 0, "post succeeds");
        const auto t0 = clk::now();
        // A ONE SECOND timeout, deliberately: the assertion is that the wait returned nowhere near
        // it, so the gap between the bound and the timeout is the whole strength of the arm.
        const uint64_t rc = timedwait(handle, 1000000, 0, 0, 0, 0);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        CHECK(rc == 0, "an already-posted semaphore is acquired");
        // Same asymmetry as ARM 3, and the same reasoning: 1 ms is a fair bound on a real x86
        // host, and on an emulated-x86 CI VM it is a latent flake that happened not to fire yet.
        // 100 ms still fails a fast path that fell through to waiting out the 1 s timeout, which
        // is the only defect this arm can see.
#ifdef _WIN32
        CHECK(ms < 1.0, "...immediately, without entering the poll loop");
#else
        CHECK(ms < 100.0, "...immediately, nowhere near the 1 s timeout it was given");
#endif
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
