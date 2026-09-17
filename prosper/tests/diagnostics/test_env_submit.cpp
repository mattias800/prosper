// test_env_submit — the contract of diagnostics/env_submit.hpp, asserted in BOTH directions and on
// BOTH paths.
//
// A per-submit sample sits between two failure modes and only a two-sided test separates it from
// either. Too sticky and it is `PROSPER_ENV_ON` in disguise: a test that arms a switch and then
// calls the renderer goes vacuous, which is the #2214 failure this header exists to avoid. Too
// eager and it is the live `getenv` it replaces, which buys nothing. So: frozen WITHIN a scope,
// visible at the NEXT one.
//
// And the macro has two paths, not one -- the cached path inside a `SubmitEnvScope` and a live
// `getenv` outside every scope. Correct output on the optimized path cannot prove the fallback is
// still live, and correct output on the fallback cannot prove anything was cached, so each is
// asserted with the property the other one does not have.
#include "diagnostics/env_submit.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#ifdef _WIN32
#include <stdlib.h>
#endif

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Bounded: an unbounded `while (x != want)` in a test never dies when the handshake breaks, it
// just hangs ctest. Returns false so the caller can mark the arm void rather than pass it.
static bool spin_until(std::atomic<int>& phase, int want) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (phase.load(std::memory_order_acquire) != want) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// One call site per probe, deliberately: the macro's unit is the textual expansion, so two probes
// reading the same name must not share state. `probe_a` and `probe_b` exist to assert that.
static bool probe_a() { return PROSPER_ENV_ON_PER_SUBMIT("PROSPER_TEST_ENV_SUBMIT"); }
static bool probe_b() { return PROSPER_ENV_ON_PER_SUBMIT("PROSPER_TEST_ENV_SUBMIT"); }

int main() {
    printf("== test_env_submit ==\n");
    set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);

    // --- the FALLBACK path: no scope anywhere, so the read must be live ------------------------
    // This is the behaviour every call site had before the header existed, and it is what keeps
    // `realize_retained_draw`, `diagnose_resource_provenance` and the tests that call
    // `build_stage_table` directly from silently freezing on their first read.
    CHECK(prosper::diag::submit_env_window() == 0, "no submit in progress reports no window");
    CHECK(!probe_a(), "outside a submit, an unset switch reads false");
    set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
    CHECK(probe_a(), "outside a submit, an arm is visible IMMEDIATELY -- the read is live");
    set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
    CHECK(!probe_a(), "...and so is a clear");

    // --- the OPTIMIZED path: inside a scope, frozen for the window -----------------------------
    {
        const prosper::diag::SubmitEnvScope scope;
        CHECK(prosper::diag::submit_env_window() != 0, "a scope opens a sampling window");
        CHECK(!probe_a(), "an unset switch reads false");
        // Arm it WITHOUT opening a new scope. The fallback arm above proves a live getenv would
        // see this at once; the cached path must not, or the two probes below could disagree
        // about the same submit.
        set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
        CHECK(!probe_a(), "...and stays false for the rest of the submit that sampled it");
    }
    {
        const prosper::diag::SubmitEnvScope scope;
        CHECK(probe_a(), "the next submit observes the arm");
        set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
        CHECK(probe_a(), "...and holds it for that submit even after the switch is cleared");
    }
    {
        const prosper::diag::SubmitEnvScope scope;
        CHECK(!probe_a(), "the submit after that observes the clear");
    }

    // A second call site is independent, and the arm has to be shaped so that a SHARED cache gives
    // a different answer -- "B agrees with A" passes either way and proves nothing. So: A samples
    // while the switch is clear, the switch is then armed, and B reads for the first time. Per
    // call site, B samples afresh and sees the arm; from a shared cache it would inherit A's stale
    // false. (Two sites in one window can therefore disagree. That is inherent to "the unit is the
    // call site", exactly as it is for PROSPER_ENV_ON, and it is what the header claims.)
    set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
    {
        const prosper::diag::SubmitEnvScope scope;
        CHECK(!probe_a(), "site A samples the cleared switch");
        set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
        CHECK(probe_b(), "a second call site samples on ITS first read, not from A's cache");
    }

    // Leaving the scope must restore the live read rather than leave the last sample standing --
    // the destructor closing the window is what makes "forgetting a scope costs speed, never
    // correctness" true.
    //
    // The switch is ARMED here rather than cleared, and that is the whole arm. Site A's cached
    // value at this point is `false` (it sampled a cleared switch just above), so an arm expecting
    // `false` would be satisfied by the stale cache: it would pass whether or not the scope closed,
    // which is a vacuous arm sitting next to the property it claims to test.
    set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
    CHECK(probe_a(), "after the last scope closes, an arm made outside it is visible at once");

    // Worker threads realize a submit's draws, so the generation must be global rather than
    // thread_local: a per-thread counter is never bumped on a worker, which would freeze the value
    // for the life of that thread -- process-lifetime behaviour on exactly the threads that do the
    // per-draw work.
    //
    // ONE worker spanning TWO submits, not one worker per submit. A fresh thread starts with fresh
    // thread_locals, so a per-submit-thread arm samples correctly even with a thread_local
    // generation and proves nothing. The worker below outlives the first scope, which is the only
    // shape in which the two designs differ.
    {
        set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
        std::atomic<int> phase{0};
        bool first = true, second = false;
        std::thread worker([&] {
            if (!spin_until(phase, 1)) return;
            first = probe_a();
            phase.store(2, std::memory_order_release);
            if (!spin_until(phase, 3)) return;
            second = probe_a();
            phase.store(4, std::memory_order_release);
        });
        bool handshake = true;
        {
            const prosper::diag::SubmitEnvScope scope;
            phase.store(1, std::memory_order_release);
            handshake = spin_until(phase, 2);
        }
        set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
        {
            const prosper::diag::SubmitEnvScope scope;
            phase.store(3, std::memory_order_release);
            handshake = spin_until(phase, 4) && handshake;
        }
        worker.join();
        CHECK(handshake, "the worker handshake completed (otherwise the arm below is void)");
        CHECK(!first && second,
              "one worker spanning two submits observes each submit's value, not a frozen first "
              "sample");
    }

    // Nesting must not strand the window open: an inner scope's destructor closes only its own
    // level, and the outer one still holds the window. Only the outermost close restores the live
    // read. `realize_gpustate_draws` nests inside `execute_ordered_gpustate` on the eager path, so
    // this is a shape production reaches, not a hypothetical.
    set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
    {
        const prosper::diag::SubmitEnvScope outer;
        CHECK(!probe_a(), "outer scope samples the cleared switch");
        { const prosper::diag::SubmitEnvScope inner; (void)inner; }
        // The inner scope started a window of its own, so this read re-samples -- more eager than
        // the contract, which is the safe direction. What matters is what happens AFTER it.
        (void)probe_a();
        set_test_env("PROSPER_TEST_ENV_SUBMIT", "1");
        CHECK(prosper::diag::submit_env_window() != 0,
              "an inner scope closing leaves the outer window open");
        CHECK(!probe_a(), "...so the read is still cached, not live");
    }
    CHECK(probe_a(), "closing the outermost scope restores the live read");

    set_test_env("PROSPER_TEST_ENV_SUBMIT", nullptr);
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
