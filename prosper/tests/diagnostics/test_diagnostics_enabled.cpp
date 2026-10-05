// test_diagnostics_enabled.cpp - Verify the diagnostics enabled path captures events.
//
// Requires PROSPER_DIAGNOSTICS=ON, which is also the only configuration CMake registers this target
// in. The `#else` arm is kept so the file is self-consistent if it is ever compiled directly, and it
// REPORTS a skip instead of exiting 0: the pre-GTest version printed "SKIPPED" and returned success,
// which ctest records identically to a pass.
#include <gtest/gtest.h>

#ifdef PROSPER_DIAGNOSTICS

#include <vector>

#include "diagnostics/diagnostics.hpp"

// The diagnostics types live in prosper::diagnostics namespace.
using prosper::diagnostics::BootEvent;
using prosper::diagnostics::BootPhase;
using prosper::diagnostics::DiagnosticContext;
using prosper::diagnostics::event_bus;
using prosper::diagnostics::record_boot_phase;

TEST(DiagnosticsEnabled, EnableAndDisableFlipTheRecordingSwitch) {
    auto& ctx = DiagnosticContext::instance();
    ctx.clear();  // start clean

    // Initially disabled after clear/reset.
    ctx.disable();
    EXPECT_FALSE(ctx.is_enabled()) << "initially disabled";

    ctx.enable();
    EXPECT_TRUE(ctx.is_enabled()) << "enable() works";
}

TEST(DiagnosticsEnabled, RecordBootPhaseCapturesEventsInOrderWithMonotonicTimestamps) {
    auto& ctx = DiagnosticContext::instance();
    ctx.clear();
    ctx.enable();
    record_boot_phase(BootPhase::PROCESS_START);
    record_boot_phase(BootPhase::LINKING);
    record_boot_phase(BootPhase::HLE_REGISTERED);

    ASSERT_EQ(ctx.event_count(), 3u) << "captured 3 phase events";
    const auto& events = ctx.events();
    ASSERT_GE(events.size(), 3u);
    EXPECT_EQ(events[0].phase, BootPhase::PROCESS_START) << "event 0 is PROCESS_START";
    EXPECT_EQ(events[1].phase, BootPhase::LINKING) << "event 1 is LINKING";
    EXPECT_EQ(events[2].phase, BootPhase::HLE_REGISTERED) << "event 2 is HLE_REGISTERED";
    EXPECT_GE(events[0].timestamp_ms, 0) << "timestamp >= 0";
    EXPECT_GE(events[1].timestamp_ms, events[0].timestamp_ms) << "monotonic timestamps";
}

TEST(DiagnosticsEnabled, AnEventBusSubscriberReceivesTheEventAndUnsubscribeStopsIt) {
    auto& ctx = DiagnosticContext::instance();
    ctx.clear();
    ctx.enable();
    std::vector<BootEvent> received;
    const auto sub_handle =
        event_bus().subscribe([&received](const BootEvent& ev) { received.push_back(ev); });

    record_boot_phase(BootPhase::MODULES_MAPPED);
    // Unsubscribe before asserting: an early ASSERT return must not leave a subscriber on the
    // process-wide bus holding a reference to this destroyed local.
    event_bus().unsubscribe(sub_handle);
    ASSERT_EQ(received.size(), 1u) << "subscriber received 1 event";
    EXPECT_EQ(received[0].phase, BootPhase::MODULES_MAPPED) << "correct phase in subscriber";
    record_boot_phase(BootPhase::BOOT_COMPLETE);
    EXPECT_EQ(received.size(), 1u) << "unsubscribe stops delivery";
}

TEST(DiagnosticsEnabled, RecordingStopsWhileDisabledAndClearResetsTheCount) {
    auto& ctx = DiagnosticContext::instance();
    ctx.clear();
    ctx.enable();
    record_boot_phase(BootPhase::PROCESS_START);

    // Disable stops recording.
    ctx.disable();
    const size_t count_at_disable = ctx.event_count();
    record_boot_phase(BootPhase::BOOT_COMPLETE);
    EXPECT_EQ(ctx.event_count(), count_at_disable) << "no events when disabled";

    // Clear resets everything.
    ctx.enable();
    ctx.clear();
    EXPECT_EQ(ctx.event_count(), 0u) << "clear() resets event count";
}

#else   // PROSPER_DIAGNOSTICS not defined - there is no enabled path to observe.

TEST(DiagnosticsEnabled, RequiresTheDiagnosticsFeature) {
    GTEST_SKIP() << "PROSPER_DIAGNOSTICS is off, so there is no enabled path to capture through";
}

#endif