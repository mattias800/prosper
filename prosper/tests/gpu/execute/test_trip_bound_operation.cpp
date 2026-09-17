// test_trip_bound_operation — the coherence contract of TripBoundOperation (#3714).
//
// The property under test is NOT "fewer getenv calls". It is that one shader-cache operation
// resolves the PROSPER_CFG_TRIP_BOUND* selectors ONCE and uses that value for both halves of the
// operation: the cache key, and the compilation the key will name. Before this scope existed,
// gpu_executor sampled the settings into the key and the recompiler then re-read the environment
// from four further sites. A change landing in between keys an entry under settings A while the
// module in it was built under settings B, and no later lookup can detect that -- the entry is
// simply wrong from then on.
//
// So every arm here is about AGREEMENT between two reads, and each has a negative control that
// makes the two disagree. The read-count arms come last and are a work counter: correct settings
// cannot show that the sampling happened once rather than four times.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "diagnostics/env_submit.hpp"
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <stdlib.h>
#endif

using prosper::gpu::ComputeTripBoundSettings;
using prosper::gpu::TripBoundOperation;
using prosper::gpu::compute_trip_bound_settings;
using prosper::gpu::parse_trip_bound_settings;
using prosper::gpu::trip_bound_parses;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

static void clear_all() {
    set_env("PROSPER_CFG_TRIP_BOUND", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
    set_env("PROSPER_CFG_TRIP_BOUND_ORDINAL", nullptr);
}

// Compare every selector, not just `bound`. The cache key mixes in all four, so an operation that
// pinned only the bound would still let the other three drift between key and module -- and an arm
// that checked only `bound` would not see it.
static bool same(const ComputeTripBoundSettings& a, const ComputeTripBoundSettings& b) {
    return a.bound == b.bound && a.only_program == b.only_program &&
           a.only_phase == b.only_phase && a.only_ordinal == b.only_ordinal;
}

static void arm(const char* bound, const char* program, const char* phase, const char* ordinal) {
    set_env("PROSPER_CFG_TRIP_BOUND", bound);
    set_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", program);
    set_env("PROSPER_CFG_TRIP_BOUND_PHASE", phase);
    set_env("PROSPER_CFG_TRIP_BOUND_ORDINAL", ordinal);
}

int main() {
    printf("== test_trip_bound_operation ==\n");
    clear_all();

    // --- default / unset -------------------------------------------------------------------
    {
        const ComputeTripBoundSettings off = compute_trip_bound_settings();
        CHECK(off.bound == 0 && off.only_program == 0 &&
                  off.only_phase == ComputeTripBoundSettings::kAllPhases &&
                  off.only_ordinal == ComputeTripBoundSettings::kAllOrdinals,
              "unset: disarmed, and every selector at its all-inclusive default");
        const TripBoundOperation op;
        CHECK(same(op.settings(), off), "an operation on an unset environment pins the same value");
    }

    // --- UNSCOPED reads stay LIVE ------------------------------------------------------------
    // This is the behaviour test_cfg_trip_bound depends on: arm, compile, disarm, compile again in
    // one process. If the scope ever froze reads process-wide this arm goes red.
    {
        clear_all();
        CHECK(compute_trip_bound_settings().bound == 0, "outside any operation: disarmed");
        arm("4", nullptr, "0", nullptr);
        CHECK(compute_trip_bound_settings().bound == 4,
              "outside any operation, an arm is visible IMMEDIATELY -- the read is live");
        clear_all();
        CHECK(compute_trip_bound_settings().bound == 0, "...and so is a disarm");
    }

    // --- THE CONTRACT: key and compilation cannot disagree ------------------------------------
    // `key` is the value gpu_executor puts in ShaderCompileKey; `compiler` is what the recompiler
    // reads later in the same operation. They must be equal even though the environment changed
    // in between, because the entry is about to be keyed under the first and built under the
    // second.
    {
        arm("4", "0x1000", "0", "7");
        ComputeTripBoundSettings key{}, compiler{};
        {
            const TripBoundOperation op;
            key = op.settings();
            arm("8", "0x2000", "1", "9");          // the change that used to corrupt the entry
            compiler = compute_trip_bound_settings();
        }
        CHECK(same(key, compiler),
              "within one operation the key and the compiler see the SAME settings");
        CHECK(key.bound == 4 && key.only_program == 0x1000 && key.only_phase == 0 &&
                  key.only_ordinal == 7,
              "...and it is the value sampled when the operation opened, all four selectors");

        // NEGATIVE CONTROL, and the whole reason the arm above is not vacuous: run the identical
        // sequence with no operation open and the two reads disagree, which is the defect.
        arm("4", "0x1000", "0", "7");
        const ComputeTripBoundSettings unscoped_key = compute_trip_bound_settings();
        arm("8", "0x2000", "1", "9");
        const ComputeTripBoundSettings unscoped_compiler = compute_trip_bound_settings();
        CHECK(!same(unscoped_key, unscoped_compiler),
              "control: without an operation the same sequence DIVERGES -- the defect reproduces");
    }

    // --- the pin is released, and released to the right value ---------------------------------
    {
        clear_all();
        arm("4", nullptr, "0", nullptr);
        { const TripBoundOperation op; (void)op; }
        arm("8", nullptr, "1", nullptr);
        CHECK(compute_trip_bound_settings().bound == 8,
              "after the operation closes, reads are live again");
    }

    // --- nesting ADOPTS rather than re-samples -------------------------------------------------
    // A chained vertex program opens an operation inside the one its prolog opened. If the inner
    // scope re-sampled, the prolog and its continuation could be compiled under different bounds
    // while sharing one key.
    {
        clear_all();
        arm("4", nullptr, "0", nullptr);
        const TripBoundOperation outer;
        arm("8", nullptr, "1", nullptr);
        const TripBoundOperation inner;
        CHECK(same(inner.settings(), outer.settings()) && inner.settings().bound == 4,
              "a nested operation adopts the enclosing value rather than re-sampling");
    }
    clear_all();

    // --- WORK COUNTER: one parse per submit, not one per operation ----------------------------
    // Correct settings cannot show this; only the counter can. Inside a submit window the sampled
    // value is reused, which is the whole reduction (#3714: 3.9 M reads on a 240 s routed window).
    {
        const uint64_t before = trip_bound_parses();
        {
            const prosper::diag::SubmitEnvScope submit;
            for (int i = 0; i < 16; ++i) { const TripBoundOperation op; (void)op.settings(); }
        }
        const uint64_t parses = trip_bound_parses() - before;
        CHECK(parses == 1,
              "16 operations inside one submit parse the environment exactly ONCE");
    }

    // ...and the complement, which is what keeps unscoped callers honest rather than merely fast.
    {
        const uint64_t before = trip_bound_parses();
        for (int i = 0; i < 16; ++i) { const TripBoundOperation op; (void)op.settings(); }
        const uint64_t parses = trip_bound_parses() - before;
        CHECK(parses == 16,
              "the same 16 operations outside any submit parse 16 times -- live, not frozen");
    }

    // A new submit re-samples: a change made between two submits must reach the next one, or the
    // reduction has quietly become a process-lifetime cache.
    {
        clear_all();
        uint32_t first = 0xffffffffu, second = 0xffffffffu;
        {
            const prosper::diag::SubmitEnvScope submit;
            const TripBoundOperation op;
            first = op.settings().bound;
        }
        arm("12", nullptr, "0", nullptr);
        {
            const prosper::diag::SubmitEnvScope submit;
            const TripBoundOperation op;
            second = op.settings().bound;
        }
        CHECK(first == 0 && second == 12,
              "a change between two submits is observed by the next submit");
    }

    // The parser itself is never skipped when it is called directly -- it is the thing the counter
    // counts, so a counter that stopped incrementing would make every arm above vacuous.
    {
        const uint64_t before = trip_bound_parses();
        (void)parse_trip_bound_settings();
        (void)parse_trip_bound_settings();
        CHECK(trip_bound_parses() - before == 2,
              "the work counter tracks direct parses (so the counts above mean something)");
    }

    clear_all();
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
