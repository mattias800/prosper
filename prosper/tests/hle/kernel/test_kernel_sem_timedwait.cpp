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
//   1. MECHANISM  - WHICH primitive served the wait. On Windows: the high-resolution timer, not
//                   ::Sleep's tick -- reddens on the pre-fix code, which never touches
//                   precise_sleep at all. On POSIX the assertion is the OPPOSITE value,
//                   "none", because the native sem_timedwait is kept there and must not start
//                   polling; the two are different assertions, not one with a platform tweak.
//   2. ENCODING   - the timeout returns exactly 0x8002003c, FreeBSD ETIMEDOUT as the guest reads it.
//                   Nothing else in the suite covers this path's encoding.
//   3. BOUNDS     - a stub/unit guard, and NOT the discriminator. THE canonical figures for this
//                   file, referenced from CMakeLists.txt rather than restated there:
//
//                     defect, Windows, fix reverted:  14.69  12.06  17.75  10.00  15.32 ms
//                     fix, Windows:                    5.49   5.47   5.39 ms
//                     macOS/Rosetta, native path:     13.58  18.54  23.22  46.46 ms
//
//                   The 13.58 ms is the run that actually reddened CI and became #3067 -- the
//                   macOS job on PR #3063, whose log reads `(timeout took 13.58 ms via none)`
//                   under `[FAIL] and returned inside one winpthreads tick`. It is BELOW the
//                   18.54 ms this list already carried, which sharpens rather than repeats the
//                   point: the 12 ms ceiling was not a bound that occasionally lost to an
//                   outlier, it was a bound sitting inside the ordinary spread of a native
//                   POSIX wait under binary translation. Nothing in this file changed to fix
//                   it; #3066 was already merged 49 minutes after #3067 was filed.
//
//                   A 12 ms ceiling would have PASSED the defect (the 10.00 ms run), not
//                   merely flaked -- silently, which is worse. And it cannot be repaired by
//                   moving it, for a reason that needs no measurement at all: a quantized
//                   wait returns at the next tick BOUNDARY, so its latency is T-p or 2T-p for
//                   phase p within the tick, and as p approaches T-N from below the defect's
//                   latency approaches the REQUESTED 5 ms exactly -- below the fix's own
//                   5.39 ms, for any tick length T. The distributions overlap by
//                   construction. (Second review of #3066 supplied that argument; the five
//                   samples are kept because they also discriminate the MODEL -- a
//                   whole-ticks-from-the-request model predicts clustering at ~15.6/31.2 ms,
//                   which 10.00 and 12.06 falsify.)
//
//                   Arm 1 caught all five reverted runs. On Rosetta the native path is
//                   untouched by this change, and arm 4's posted acquire measures SUB-
//                   MILLISECOND there, so the 100 ms bound has over 100x headroom on a
//                   measured value rather than a guessed one.
//   4. FAST PATH  - an already-posted semaphore is acquired at once, asserted as a COUNT of 0
//                   afterwards rather than as a latency. The latency bound is kept as well but
//                   is Windows-tight / POSIX-loose, and it cannot see the fast path at all:
//                   the pre-loop trywait and the first call inside the loop are identical, so
//                   deleting the fast path changes no duration. "Never enters the loop" is
//                   therefore NOT what this arm asserts, and there is no loop on POSIX.
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
    // Optional: used only to make ARM 4 assert the COUNT rather than a duration. Guarded so a
    // build where it is unregistered loses one assertion instead of the whole test.
    HleFn getvalue    = Hle::lookup(nid_hash("scePthreadSemGetvalue").c_str());
    CHECK(sem_init_fn != nullptr, "scePthreadSemInit is registered");
    CHECK(timedwait != nullptr, "scePthreadSemTimedwait is registered");
    CHECK(post != nullptr, "scePthreadSemPost is registered");
    if (!sem_init_fn || !timedwait || !post) { printf("== FAIL (unresolved) ==\n"); return 1; }

    // The guest handle is a POINTER CELL, not a semaphore. k_sem_init does
    //     `*(void**)(uintptr_t)a0 = s;`
    // i.e. it heap-allocates the sem_t and writes the POINTER through the handle, and every
    // other member of the family reads it back with an 8-byte load (`ensure_sem`). So the slot
    // must be pointer-sized.
    //
    // This was `sem_t slot;`, which is a latent stack overwrite that happens to be invisible
    // on two of three platforms: winpthreads' sem_t is a pointer (8 bytes) and glibc's is 32,
    // but **Darwin's is `int`** -- so on macOS the 8-byte write ran four bytes past a 4-byte
    // automatic. Found in review of the very PR whose purpose was to green the macOS job.
    void* slot = nullptr;
    const uint64_t handle = (uint64_t)(uintptr_t)&slot;
    // Asserts that init REPORTS success, not that the count is observably 0 -- the count is then
    // established by arm 1 timing out rather than being acquired.
    //
    // Until #3068, k_sem_init DISCARDED sem_init's return and reported success unconditionally, so
    // this CHECK could not fail as the code stood -- it was kept only as a tripwire. #3068 made init
    // forward the real host result, so this now genuinely exercises the success path (value 0 is
    // always a legal initial count) rather than an unconditional 0. The FAILURE path -- a value
    // sem_init genuinely rejects, and the guest slot staying unpublished when it does -- is covered
    // by test_kernel_sem_init_error.cpp, which is what actually reddens without #3068's fix; this
    // CHECK alone still cannot (init with 0 succeeds on every platform this suite runs on).
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

    // ARM 4: an already-posted semaphore is acquired at once and its count is consumed.
    // NOT "never enters the loop" -- that phrase was here and is wrong, see the header and the
    // note over the count assertion below. There is no loop on POSIX at all.
    {
        CHECK(post(handle, 0, 0, 0, 0, 0) == 0, "post succeeds");
        const auto t0 = clk::now();
        // A ONE SECOND timeout, deliberately: the assertion is that the wait returned nowhere near
        // it, so the gap between the bound and the timeout is the whole strength of the arm.
        const uint64_t rc = timedwait(handle, 1000000, 0, 0, 0, 0);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        CHECK(rc == 0, "an already-posted semaphore is acquired");
        // The count is the observable, and it is what "acquired" MEANS: it catches a path that
        // returns rc==0 WITHOUT consuming, which no timing bound can see. Mutation-checked
        // with a peek-and-return in place of the trywait.
        //
        // What it does NOT catch, stated because the obvious reading of it is wrong: deleting
        // the pre-loop sem_trywait entirely. The loop's FIRST statement is the identical
        // call, so it consumes the count anyway and every assertion here still passes. That
        // deletion is unobservable at the HLE boundary by construction -- the fast path is a
        // latency optimisation, not a semantic one -- so it is a fact to record rather than a
        // gap to close. Second review of #3066.
        // The lookup is CHECKed rather than merely guarded: an optional lookup that silently
        // skips means a rename makes both assertions below vanish GREEN.
        CHECK(getvalue != nullptr, "scePthreadSemGetvalue is registered");
        if (getvalue) {
            int v = -1;
            CHECK(getvalue(handle, (uint64_t)(uintptr_t)&v, 0, 0, 0, 0) == 0,
                  "getvalue succeeds after the acquire");
            CHECK(v == 0, "...and the count really was consumed, not merely reported as taken");
        }
        // Same asymmetry as ARM 3, and the same reasoning: 1 ms is a fair bound on a real x86
        // host, and on an emulated-x86 CI VM it is a latent flake that happened not to fire yet.
        // 100 ms still fails a fast path that fell through to waiting out the 1 s timeout, which
        // is the only defect this arm can see.
#ifdef _WIN32
        CHECK(ms < 1.0, "...immediately, on the order of a trywait rather than a wait");
#else
        CHECK(ms < 100.0, "...immediately, nowhere near the 1 s timeout it was given");
#endif
        // Printed for the same reason ARM 3 prints its figure: a green ctest run shows no
        // per-test output, so a margin that is never printed cannot be quoted from a passing
        // log -- which is exactly how #3044 came to be merged on a green Rosetta job whose
        // numbers nobody had seen.
        //
        // Printing it was only half the fix and the other half was missing until #3067: CI ran
        // `ctest --output-on-failure`, which shows a test's output ONLY when it fails, so these
        // figures reached a log exclusively on runs that were already red. This test now carries
        // the `timing-margin` ctest label and the Linux, Windows MinGW and macOS jobs re-run that
        // label with `-V`, so every green run records what the margin actually was. To read the
        // current headroom on any platform, open a recent CI log's "Report the measured wait
        // margins" step rather than re-deriving it.
        printf("         (posted acquire took %.2f ms)\n", ms);
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
