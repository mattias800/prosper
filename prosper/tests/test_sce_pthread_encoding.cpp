// #2178: a Sony pthread spelling must never hand the guest a BARE FreeBSD errno.
//
// Every other libkernel entry point returns `0x80020000 | errno`. A Sony name registered straight
// onto a fallible POSIX body returns the bare value instead, so the guest gets a number in the wrong
// space -- #1984's shape, the one that killed Oregon Trail until 38619f29. #2158 fixed two
// instances; this file guards the rule rather than the instances.
//
// The assertion is deliberately an INVARIANT over the whole family, not a list of expected values.
// A list would have to be extended by hand every time a function is added, and the failure mode of
// forgetting is silent -- which is exactly how the instances below survived. Anything that returns
// non-zero must have the libkernel prefix; what the errno IS is the POSIX body's business and is
// tested elsewhere.

#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what.c_str());
}

using prosper::Hle;
using prosper::HleFn;

int main() {
    std::fprintf(stderr, "== test_sce_pthread_encoding ==\n");
    prosper::register_kernel_hle();

    // Each entry is called with arguments chosen to FAIL: a null output slot, an invalid enum, or a
    // bad handle. The point is to reach a return path that carries an errno at all -- a call that
    // succeeds proves nothing about encoding.
    struct Case { const char* name; uint64_t a0, a1, a2; };
    const Case cases[] = {
        {"scePthreadOnce",                0, 0, 0},   // null control / null routine -> EINVAL
        {"scePthreadAttrInit",            0, 0, 0},   // null attr slot
        {"scePthreadAttrSetguardsize",    0, 0, 0},
        {"scePthreadAttrSetdetachstate",  0, 99, 0},  // 99 is not a detach state
        {"scePthreadAttrGetdetachstate",  0, 0, 0},
        {"scePthreadKeyCreate",           0, 0, 0},
        // scePthreadCreate is DELIBERATELY absent, and it must not be added back with a null entry
        // point. `scePthreadCreate(0,0,0,0,0,0)` is not a failure path at all: `k_pthread_create`
        // never validates a2, so it SPAWNS A REAL HOST THREAD on `thread_trampoline`, publishes its
        // name, and returns 0. The row therefore took the `rc == 0` branch below on every run since
        // it was written -- it asserted nothing and never counted toward `exercised` -- while
        // leaving a joinable thread nobody joins, racing `main`'s return.
        //
        // That race is #2427, the macOS/Rosetta `sce_pthread_encoding` SEGFAULT, and it is the
        // single largest source of red on that job. Measured, not guessed: 15 of 502 macOS job
        // attempts (2026-08-08..17) died here -- 3.0% of attempts and 71% of every failure the job
        // produced in that window -- with every failing log stopping after the last loop line and
        // before the `exercised` line. A core dump from the same crash reproduced locally under
        // `taskset -c 0` names the mechanism exactly --
        //
        //   thread 2 (main): exit() -> __run_exit_handlers -> _dl_fini
        //   thread 1:        thread_trampoline (hle_kernel.cpp:2008)
        //                      -> retire_guest_thread_name (hle_kernel.cpp:1735)
        //                        -> std::unordered_map::find on g_guest_thread_names  -> SIGSEGV
        //
        // -- i.e. the spawned thread reaches the trampoline's tail after `exit()` has already run
        // the static destructors for that registry. Rosetta widens the window (every first-execution
        // path in the new thread is translated), which is why this fires on macOS and not on native
        // Linux: 0 of 400 native runs, but 1 of 300 pinned to one CPU with `taskset -c 0`, which is
        // how the core dump above was obtained. With this row gone the thread is never created at
        // all -- `PROSPER_SYNCLOG=1` prints no `[thread] create` line -- so the race is removed by
        // construction rather than made rarer: 0 of 2000 runs under the same `taskset -c 0`.
        //
        // The registry lifetime is a SEPARATE and still-open defect, tracked as #2613: any prosper
        // process that exits while a guest thread is inside the trampoline's tail can fault the same
        // way, and the thread-stack registry next to it has the same shape. Removing this row fixes
        // the CI signal, not that.
        //
        // Re-adding a row for it needs an argument pattern that reaches a RETURN, not a spawn, and
        // none of the zero-argument shapes does: `k_pthread_create`'s only error returns come from
        // `pthread_attr_setstacksize`, the ThreadStart allocation, and `pthread_create` itself, and
        // the first is floored to 1 MiB so it cannot be provoked portably. The alias itself
        // (`SCE_PTHREAD_ALIAS(k_sce_pthread_create, k_pthread_create)`) is consequently NOT guarded
        // by this file today. That gap is real and tracked in #2615, together with the separate
        // question of whether `scePthreadCreate` should accept a null entry point at all -- it is a
        // smaller cost than a test that crashes the suite at random, and the gap already existed:
        // this row never once asserted anything about the alias.
        //
        // The families #2158 and #2359 already fixed, kept here so a regression in ANY of them is
        // caught by this one file rather than by whichever test happened to cover it.
        {"scePthreadMutexUnlock",         0, 0, 0},
        {"scePthreadRwlockattrInit",      0, 0, 0},
        {"scePthreadRwlockattrDestroy",   0, 0, 0},
        {"scePthreadCondDestroy",         0, 0, 0},
    };

    size_t exercised = 0;
    std::vector<const char*> inert;   // rows whose argument pattern did not reach an error path
    for (const Case& c : cases) {
        HleFn fn = Hle::lookup(prosper::nid_hash(c.name));
        if (!fn) { check(false, std::string(c.name) + " is registered"); continue; }
        const uint64_t rc = fn(c.a0, c.a1, c.a2, 0, 0, 0);
        // This argument pattern succeeded, so it says nothing about encoding. Record the name: the
        // `exercised` floor below is a TOTAL, so a row that is inert for its whole life hides behind
        // the rows that are not. That is exactly how the scePthreadCreate row survived (see above) --
        // it was inert from the day it was written and the floor still read 8 >= 6. Naming them
        // makes the next one visible on a passing run instead of only after it does damage; the two
        // this currently reports are #2615.
        if (rc == 0) { inert.push_back(c.name); continue; }
        ++exercised;
        const bool encoded = (rc & 0xffff0000ull) == 0x80020000ull;
        check(encoded, std::string(c.name) + " returns an ENCODED error (0x8002xxxx), not a bare "
                       "errno -- got 0x" + [&] {
                           char buf[32]; std::snprintf(buf, sizeof buf, "%llx",
                                                       (unsigned long long)rc); return std::string(buf);
                       }());
    }

    // Without this the file would pass vacuously if every call above started returning 0 -- which is
    // a real possibility, since the arguments are chosen to provoke failure and a future change
    // could make one of them legal. A green run that exercised nothing is the #2130 trap, and it is
    // the reason this counter exists rather than a comment saying the arms are meaningful.
    check(exercised >= 6,
          "at least 6 of the calls actually reached an error path (got " +
              std::to_string(exercised) + ") -- otherwise the assertions above are vacuous");
    for (const char* name : inert)
        std::fprintf(stderr, "  [info] %s answered 0 for this argument pattern, so it asserted "
                             "nothing about encoding\n", name);

    // The CONTROL, and the half that proves the test can tell the two spellings apart: the POSIX
    // name must still return the bare errno. If encoding had been applied to the shared body instead
    // of the Sony wrapper, every assertion above would pass and this one would fail.
    if (HleFn posix_attr_init = Hle::lookup(prosper::nid_hash("pthread_attr_init"))) {
        const uint64_t rc = posix_attr_init(0, 0, 0, 0, 0, 0);
        check(rc == 0 || (rc & 0xffff0000ull) == 0,
              "control: the POSIX spelling still returns a BARE errno, so encoding was added to the "
              "Sony wrapper rather than to the shared body");
    }

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
