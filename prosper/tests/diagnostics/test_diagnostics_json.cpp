// test_diagnostics_json.cpp - Verify the JSON output format (PROSPER_DIAGNOSTICS only).
//
// Tests that JsonWriter produces valid JSON:
// - Empty events -> "[]"
// - Single event -> correct object format
// - Multiple events -> array with proper commas
// - Phase names are human-readable
//
// Requires PROSPER_DIAGNOSTICS=ON, which is also the only configuration CMake registers this target
// in. The `#else` arm REPORTS a skip instead of exiting 0, which ctest records identically to a pass.
#include <gtest/gtest.h>

#ifdef PROSPER_DIAGNOSTICS

#include <cstddef>
#include <string>
#include <vector>

#include "diagnostics/diagnostics.hpp"

using namespace prosper::diagnostics;

TEST(DiagnosticsJson, AnEmptyEventListWritesAnEmptyArray) {
    const std::vector<BootEvent> empty;
    EXPECT_EQ(JsonWriter::write_events(empty), "[]") << "empty events -> []";
}

TEST(DiagnosticsJson, ASingleEventIsABalancedObjectCarryingItsPhaseAndTimestamp) {
    const BootEvent single(BootPhase::BOOT_COMPLETE, 42.5);
    const std::string single_json = JsonWriter::write_event(single);
    EXPECT_NE(single_json.find("\"phase\""), std::string::npos) << "contains 'phase' key";
    EXPECT_NE(single_json.find("BOOT_COMPLETE"), std::string::npos) << "contains phase name";
    EXPECT_NE(single_json.find("\"timestamp_ms\""), std::string::npos)
        << "contains 'timestamp_ms' key";
    ASSERT_FALSE(single_json.empty());
    EXPECT_EQ(single_json.front(), '{') << "object opens with a brace";
    EXPECT_EQ(single_json.back(), '}') << "object closes with a brace";
}

TEST(DiagnosticsJson, SeveralEventsBecomeAnArrayWithSeparatorsBetweenElements) {
    const std::vector<BootEvent> multi = {
        {BootPhase::PROCESS_START, 0.0},
        {BootPhase::LINKING, 1.5},
        {BootPhase::BOOT_COMPLETE, 10.0},
    };
    const std::string multi_json = JsonWriter::write_events(multi);
    ASSERT_FALSE(multi_json.empty());
    EXPECT_EQ(multi_json.front(), '[') << "array opens with a bracket";
    EXPECT_EQ(multi_json.back(), ']') << "array closes with a bracket";
    // Verify commas separate elements (N-1 commas for N elements in compact form,
    // but pretty-printed output may have newlines — just check at least N-1 separators).
    size_t comma_count = 0;
    for (const char c : multi_json)
        if (c == ',') ++comma_count;
    EXPECT_GE(comma_count, 2u) << "commas present between elements";
}

TEST(DiagnosticsJson, EveryKnownPhaseHasItsHumanReadableName) {
    EXPECT_EQ(std::string(phase_name(BootPhase::PROCESS_START)), "PROCESS_START");
    EXPECT_EQ(std::string(phase_name(BootPhase::LINKING)), "LINKING");
    EXPECT_EQ(std::string(phase_name(BootPhase::_COUNT)), "UNKNOWN") << "_COUNT -> UNKNOWN";
}

#else   // PROSPER_DIAGNOSTICS not defined - there is no writer to exercise.

TEST(DiagnosticsJson, RequiresTheDiagnosticsFeature) {
    GTEST_SKIP() << "PROSPER_DIAGNOSTICS is off, so JsonWriter is not built";
}

#endif