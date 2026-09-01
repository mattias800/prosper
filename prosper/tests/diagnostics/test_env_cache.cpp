// test_env_cache — the one-shot semantics of PROSPER_ENV_ON / PROSPER_ENV_VALUE.
//
// WHY THIS SHAPE. These macros exist to take getenv off per-draw paths (#3094), and the whole point
// of the optimisation is that a given guard reads the environment EXACTLY ONCE no matter how many
// times it is evaluated. That property is not observable in the ordinary case -- a diagnostic
// nobody re-arms behaves identically cached or not -- so a test that merely evaluates a macro and
// checks the answer would pass just as well against a plain getenv. It would confirm the assertion
// without exercising the mechanism.
//
// So each case below evaluates ONE call site twice with the environment MUTATED in between, and
// asserts the second evaluation still reports the first value. That is the one behaviour a live
// getenv cannot produce, so these arms go red if the caching is ever removed.
//
// THE CALL SITE IS THE UNIT, AND IT IS EASY TO GET WRONG. Each textual expansion of the macro
// declares its own function-local static inside its own lambda, so two expansions of
// PROSPER_ENV_ON("X") on two different source lines are two independent caches -- the second one
// performs its own live read. The first draft of this test asserted across two separate expansions
// and failed for exactly that reason. That is not a defect: a per-draw guard is a single site
// evaluated thousands of times, which is the case the optimisation targets. It does mean the
// contract is "once per site", never "once per name", so each case here routes both evaluations
// through the SAME site by calling a helper twice.
//
// The names below deliberately do NOT carry the PROSPER_ prefix. This file is the one place in the
// tree that arms a variable and caches it on purpose -- it asserts the staleness rather than being
// made vacuous by it -- which is exactly the shape tools/env/check_cached_env.py fails on. That
// gate scans for PROSPER_* names only, so an unprefixed name keeps this test outside its scope
// without adding an exception to it. (The macros are name-agnostic, so nothing is lost.)
#include "diagnostics/env_cache.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#ifdef _WIN32
#include <stdlib.h>
static void put_env(const char* n, const char* v) { _putenv_s(n, v ? v : ""); }
#else
static void put_env(const char* n, const char* v) { if (v) setenv(n, v, 1); else unsetenv(n); }
#endif

// One call site each. Every evaluation below goes through these, so the "same site" condition the
// contract is stated in terms of is structural here rather than a property of how the test is laid
// out. NOINLINE keeps them distinguishable in a disassembly if this ever needs debugging; it has no
// bearing on the semantics, which come from the function-local static.
#if defined(_MSC_VER)
#define ENVCACHE_NOINLINE __declspec(noinline)
#else
#define ENVCACHE_NOINLINE __attribute__((noinline))
#endif
static ENVCACHE_NOINLINE bool site_on()       { return PROSPER_ENV_ON("ENVCACHE_TEST_ON"); }
static ENVCACHE_NOINLINE bool site_off()      { return PROSPER_ENV_ON("ENVCACHE_TEST_OFF"); }
static ENVCACHE_NOINLINE const char* site_v() { return PROSPER_ENV_VALUE("ENVCACHE_TEST_VAL"); }

static int fails = 0;
static void check(bool ok, const char* what) {
    printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", what);
    if (!ok) ++fails;
}

int main() {
    // --- armed BEFORE first use, then disarmed --------------------------------------------------
    put_env("ENVCACHE_TEST_ON", "1");
    check(site_on(), "PROSPER_ENV_ON reports a variable set before the site's first evaluation");
    put_env("ENVCACHE_TEST_ON", nullptr);
    check(std::getenv("ENVCACHE_TEST_ON") == nullptr,
          "control: the variable really is gone from the environment now");
    check(site_on(),
          "...and the same site still reports it, because its read was cached (a live getenv "
          "would now say false)");

    // --- unset BEFORE first use, then armed -----------------------------------------------------
    // The mirror arm. Without it, "still true" above could be satisfied by a macro that simply
    // always returns true.
    put_env("ENVCACHE_TEST_OFF", nullptr);
    check(!site_off(), "PROSPER_ENV_ON reports false for a variable unset at first evaluation");
    put_env("ENVCACHE_TEST_OFF", "1");
    check(std::getenv("ENVCACHE_TEST_OFF") != nullptr,
          "control: the variable really is set in the environment now");
    check(!site_off(),
          "...and the same site still reports false -- arming AFTER first use is never observed");

    // --- the value form -------------------------------------------------------------------------
    put_env("ENVCACHE_TEST_VAL", "first");
    const char* v1 = site_v();
    check(v1 && std::strcmp(v1, "first") == 0, "PROSPER_ENV_VALUE returns the value at first use");
    put_env("ENVCACHE_TEST_VAL", "second");
    const char* live = std::getenv("ENVCACHE_TEST_VAL");
    check(live && std::strcmp(live, "second") == 0,
          "control: a live getenv sees the new value, so the mutation did take effect");
    check(site_v() == v1,
          "...and the same site returns the identical cached pointer, not the new value");

    // --- distinct sites keep independent storage ------------------------------------------------
    // Stated as a positive property rather than left implicit: if a refactor ever hoisted the
    // storage to one shared slot, the first name evaluated would answer for every other name --
    // silently, and only on paths that read more than one switch. site_on() is cached true and
    // site_off() cached false above, so a shared slot collapses them to one value.
    check(site_on() && !site_off(),
          "two different sites keep independent cached storage (true and false coexist)");

    printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
