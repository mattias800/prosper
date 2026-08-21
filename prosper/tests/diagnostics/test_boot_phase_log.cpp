// test_boot_phase_log.cpp — the boot-phase line is REACHABLE in a default build.
//
// The defect this pins: `boot_program()` records seven phases, and until `boot_phase_log.cpp` none
// of them could ever be printed — `PROSPER_DIAGNOSTICS` is off by default, `DiagnosticContext::
// enable()` was never called anywhere in the tree, and no subscriber was ever attached to the bus.
// So the interesting property is not "the formatter formats"; it is that a DEFAULT build, driven
// only by an environment variable, actually emits the lines. A unit test that called the formatter
// directly would pass just as happily with `record_boot_phase()` still a no-op, which is exactly
// the state being fixed.
//
// It is therefore checked end to end, through a child process, in three arms:
//
//   ON   PROSPER_BOOTPHASE=1  -> the child's stderr carries one line per phase
//   OFF  (variable unset)     -> no lines
//   OFF  PROSPER_BOOTPHASE=0  -> no lines; an explicit disable is honoured
//
// Both OFF arms are negative controls, and a negative control's whole claim is that it stays quiet
// — so "quiet because the rule permits it" and "quiet because the child never ran" are the same
// observation (instrument trap 190). Each arm therefore asserts its own precondition out loud: the
// child prints `EMITTED <n>` on STDOUT from the same code path that would have logged, and every
// arm requires that marker before it is allowed to interpret the stderr it collected. An arm whose
// child failed to launch fails; it does not quietly pass.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "diagnostics/diagnostics.hpp"
#include "diagnostics/boot_phase_log.hpp"

using prosper::diagnostics::BootPhase;
using prosper::diagnostics::boot_phase_log_name;
using prosper::diagnostics::record_boot_phase;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

static const unsigned kPhaseCount = static_cast<unsigned>(BootPhase::_COUNT);

// Child mode: drive every phase through the REAL entry point boot_program() uses, then announce on
// stdout that it did so. stdout and stderr are collected separately by the parent, so the marker
// cannot be mistaken for the output being measured.
static int emit_mode() {
    for (unsigned i = 0; i < kPhaseCount; ++i)
        record_boot_phase(static_cast<BootPhase>(i));
    printf("EMITTED %u\n", kPhaseCount);
    fflush(stdout);
    return 0;
}

#ifndef _WIN32
struct ArmResult {
    bool ran = false;        // the child reached the emit path and said so
    unsigned emitted = 0;    // how many phases it drove
    int phase_lines = 0;     // "[bootphase]" lines on its stderr
};

// Run this same executable in --emit mode with `env_prefix` prepended, and split its two streams.
static ArmResult run_arm(const char* self, const char* env_prefix) {
    ArmResult r;
    char errfile[512];
    snprintf(errfile, sizeof errfile, "boot_phase_log_arm_%d_%p.err",
             (int)getpid(), (const void*)env_prefix);
    std::string cmd = std::string(env_prefix) + " '" + self + "' --emit 2>'" + errfile + "'";
    if (FILE* out = popen(cmd.c_str(), "r")) {
        char line[512];
        while (fgets(line, sizeof line, out))
            if (sscanf(line, "EMITTED %u", &r.emitted) == 1) r.ran = true;
        pclose(out);
    }
    if (FILE* errf = fopen(errfile, "r")) {
        char line[512];
        while (fgets(line, sizeof line, errf))
            if (strstr(line, "[bootphase]")) r.phase_lines++;
        fclose(errf);
    }
    remove(errfile);
    return r;
}
#endif

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--emit") == 0) return emit_mode();

    printf("== test_boot_phase_log ==\n");

    // 1. The name table covers every enumerator and refuses to invent one past the end. Without
    //    this, adding a phase to BootPhase silently prints "UNKNOWN" in the middle of the boot
    //    somebody is trying to read.
    bool all_named = true;
    for (unsigned i = 0; i < kPhaseCount; ++i) {
        const char* n = boot_phase_log_name(i);
        if (!n || !*n || strcmp(n, "UNKNOWN") == 0) all_named = false;
    }
    CHECK(all_named, "every BootPhase enumerator has a name");
    CHECK(strcmp(boot_phase_log_name(kPhaseCount), "UNKNOWN") == 0,
          "a value past the last enumerator reports UNKNOWN rather than reading off the table");
    CHECK(strcmp(boot_phase_log_name(0), "PROCESS_START") == 0, "first phase names PROCESS_START");
    CHECK(strcmp(boot_phase_log_name(kPhaseCount - 1), "BOOT_COMPLETE") == 0,
          "last phase names BOOT_COMPLETE");

#ifndef _WIN32
    const char* self = argv[0];
    if (strchr(self, '\'')) {
        printf("  [FAIL] executable path contains a quote; cannot build the child command\n");
        fails++;
    } else {
        // ON arm.
        ArmResult on = run_arm(self, "PROSPER_BOOTPHASE=1");
        CHECK(on.ran && on.emitted == kPhaseCount,
              "ON arm: the child ran and drove every phase (precondition)");
        CHECK(on.phase_lines == (int)kPhaseCount,
              "ON arm: PROSPER_BOOTPHASE=1 emits exactly one line per phase");
        printf("         (ON arm saw %d line(s) for %u phase(s))\n", on.phase_lines, on.emitted);

        // OFF arm — negative control. It must prove it was pointed at something.
        ArmResult off = run_arm(self, "env -u PROSPER_BOOTPHASE");
        CHECK(off.ran && off.emitted == kPhaseCount,
              "OFF arm: the child ran and drove every phase (precondition -- without this, "
              "'no output' would be indistinguishable from 'no child')");
        CHECK(off.phase_lines == 0, "OFF arm: an unset PROSPER_BOOTPHASE emits nothing");

        // Explicit-disable arm — also a negative control, same precondition rule.
        ArmResult zero = run_arm(self, "PROSPER_BOOTPHASE=0");
        CHECK(zero.ran && zero.emitted == kPhaseCount,
              "DISABLE arm: the child ran and drove every phase (precondition)");
        CHECK(zero.phase_lines == 0, "DISABLE arm: PROSPER_BOOTPHASE=0 is honoured as OFF");
    }
#else
    printf("  [skip] subprocess arms are POSIX-only\n");
#endif

    printf(fails ? "\ntest_boot_phase_log: %d FAILURE(S)\n" : "\ntest_boot_phase_log: all ok\n", fails);
    return fails ? 1 : 0;
}
