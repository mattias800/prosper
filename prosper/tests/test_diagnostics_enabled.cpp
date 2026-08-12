// test_diagnostics_enabled.cpp — Verify diagnostics enabled path captures events.
//
// Requires PROSPER_DIAGNOSTICS=ON. Tests:
// - DiagnosticContext enables/disables correctly
// - record_phase() captures events with monotonic timestamps
// - EventBus publishes to subscribers
// - event_count() reflects recorded events
// - clear() resets state

#ifdef PROSPER_DIAGNOSTICS

#include <cstdio>
#include <vector>
#include "diagnostics/diagnostics.hpp"

// The diagnostics types live in prosper::diagnostics namespace.
using prosper::diagnostics::DiagnosticContext;
using prosper::diagnostics::record_boot_phase;
using prosper::diagnostics::BootPhase;
using prosper::diagnostics::BootEvent;
using prosper::diagnostics::event_bus;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_diagnostics_enabled: verifying event capture ==\n");

    auto& ctx = DiagnosticContext::instance();
    ctx.clear();  // start clean

    // Initially disabled after clear/reset.
    ctx.disable();
    CHECK(!ctx.is_enabled(), "initially disabled");

    // Enable and verify.
    ctx.enable();
    CHECK(ctx.is_enabled(), "enable() works");

    // Record phases and verify capture.
    ctx.clear();
    ctx.enable();
    record_boot_phase(BootPhase::PROCESS_START);
    record_boot_phase(BootPhase::LINKING);
    record_boot_phase(BootPhase::HLE_REGISTERED);

    CHECK(ctx.event_count() == 3, "captured 3 phase events");

    // Verify event order and phases.
    const auto& events = ctx.events();
    CHECK(events[0].phase == BootPhase::PROCESS_START, "event 0 is PROCESS_START");
    CHECK(events[1].phase == BootPhase::LINKING, "event 1 is LINKING");
    CHECK(events[2].phase == BootPhase::HLE_REGISTERED, "event 2 is HLE_REGISTERED");

    // Verify timestamps are non-negative and monotonically increasing.
    CHECK(events[0].timestamp_ms >= 0, "timestamp >= 0");
    CHECK(events[1].timestamp_ms >= events[0].timestamp_ms, "monotonic timestamps");

    // Verify EventBus subscriber receives events.
    std::vector<BootEvent> received;
    auto sub_handle = event_bus().subscribe(
        [&received](const BootEvent& ev) { received.push_back(ev); });

    record_boot_phase(BootPhase::MODULES_MAPPED);
    CHECK(received.size() == 1, "subscriber received 1 event");
    CHECK(received[0].phase == BootPhase::MODULES_MAPPED, "correct phase in subscriber");

    event_bus().unsubscribe(sub_handle);

    // Disable stops recording.
    ctx.disable();
    size_t count_at_disable = ctx.event_count();
    record_boot_phase(BootPhase::BOOT_COMPLETE);
    CHECK(ctx.event_count() == count_at_disable, "no events when disabled");

    // Clear resets everything.
    ctx.enable();
    ctx.clear();
    CHECK(ctx.event_count() == 0, "clear() resets event count");

    printf(fails ? "\ntest_diagnostics_enabled: %d FAILURE(S)\n"
                 : "\ntest_diagnostics_enabled: all ok\n", fails);
    return fails ? 1 : 0;
}

#else  // PROSPER_DIAGNOSTICS not defined — this test is vacuous.

#include <cstdio>
int main() {
    printf("== test_diagnostics_enabled: SKIPPED (PROSPER_DIAGNOSTICS not enabled) ==\n");
    return 0;  // Not a failure — feature is off.
}

#endif
