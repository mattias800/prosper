// test_diagnostics_disabled.cpp — Verify diagnostics disabled path is zero-cost.
//
// When PROSPER_DIAGNOSTICS is OFF (default), all calls must be no-ops:
// - is_enabled() returns false
// - record_phase() produces no events
// - event_count() stays 0
//
// This test runs in ALL builds (no PROSPER_DIAGNOSTICS gate) to verify
// the stub path compiles and behaves correctly without the feature.

#include <cstdio>
#include "diagnostics/diagnostics.hpp"

// The diagnostics types live in prosper::diagnostics namespace.
using prosper::diagnostics::DiagnosticContext;
using prosper::diagnostics::record_boot_phase;
using prosper::diagnostics::BootPhase;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_diagnostics_disabled: verifying stub behavior ==\n");

    auto& ctx = DiagnosticContext::instance();

    // Stub context must report disabled.
    CHECK(!ctx.is_enabled(), "is_enabled() returns false when disabled");

    // Recording phases must be a no-op.
    size_t before = ctx.event_count();
    record_boot_phase(BootPhase::PROCESS_START);
    record_boot_phase(BootPhase::LINKING);
    record_boot_phase(BootPhase::BOOT_COMPLETE);
    size_t after = ctx.event_count();

    CHECK(before == after, "record_phase() produces no events when disabled");
    CHECK(after == 0, "event_count() remains 0");

    // Enable/disable on stubs are no-ops.
    ctx.enable();
    CHECK(!ctx.is_enabled(), "enable() is no-op on stub");
    ctx.disable();
    CHECK(!ctx.is_enabled(), "disable() is no-op on stub");

    printf(fails ? "\ntest_diagnostics_disabled: %d FAILURE(S)\n"
                 : "\ntest_diagnostics_disabled: all ok\n", fails);
    return fails ? 1 : 0;
}
