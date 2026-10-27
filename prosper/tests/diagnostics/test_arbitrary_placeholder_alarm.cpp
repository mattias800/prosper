// A real admission fires immediately; a checked program with no admission is quiet.
#include <vector>
#include <string_view>
#include <cstddef>
#include "diagnostics/perf/perf_ledger.hpp"
#include "diagnostics/perf/perf_alarm_rules.hpp"
#include "diagnostics/perf/arbitrary_placeholder.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include <gtest/gtest.h>
#include <algorithm>

namespace {
using namespace prosper::diagnostics::perf;
bool has_alarm(const std::vector<AlarmFiring>& alarms) {
    return std::any_of(alarms.begin(), alarms.end(), [](const auto& alarm) {
        return std::string_view(alarm.rule) == "arbitrary-placeholder-admitted";
    });
}
} // namespace

TEST(ArbitraryPlaceholderAlarm, AdmittedProgramsAndReadsFireInTheirFirstWindow) {
    Ledger ledger;
    AlarmEngine engine({.window_ns = 1'000'000'000, .log = nullptr});
    engine.on_flip(1'000'000'000, ledger, 60);
    ledger.counters[static_cast<size_t>(Counter::ArbitraryPlaceholderChecks)] = 3;
    ledger.counters[static_cast<size_t>(Counter::ArbitraryPlaceholderPrograms)] = 2;
    ledger.counters[static_cast<size_t>(Counter::ArbitraryPlaceholderReads)] = 7;
    const auto alarms = engine.on_flip(2'000'000'000, ledger, 60);
    ASSERT_TRUE(has_alarm(alarms));
    const auto found = std::find_if(alarms.begin(), alarms.end(), [](const auto& alarm) {
        return std::string_view(alarm.rule) == "arbitrary-placeholder-admitted";
    });
    EXPECT_EQ(found->value, 2);
    EXPECT_NE(found->detail.find("reads=7"), std::string::npos);
    EXPECT_EQ(sustain_windows("arbitrary-placeholder-admitted"), 1u);
}

TEST(ArbitraryPlaceholderAlarm, CheckedProgramsWithoutAdmissionsStayQuiet) {
    WindowSample window;
    window.seconds = 5;
    window.counters[static_cast<size_t>(Counter::ArbitraryPlaceholderChecks)] = 3;
    EXPECT_TRUE(rule_has_data("arbitrary-placeholder-admitted", window));
    EXPECT_FALSE(has_alarm(evaluate_rules(window, {})));
    window.counters[static_cast<size_t>(Counter::ArbitraryPlaceholderChecks)] = 0;
    EXPECT_FALSE(rule_has_data("arbitrary-placeholder-admitted", window));
}

TEST(ArbitraryPlaceholderAlarm, SummaryNamesAddressesAndDoesNotInventAnUnknownAddress) {
    const auto summary = format_arbitrary_placeholder_summary({
        {0x549700u, {2, 3, 4}},
        {0, {1, 1, 0}},
    });
    EXPECT_NE(summary.find("program=0x549700 compilations=2 reads=7"), std::string::npos);
    EXPECT_NE(summary.find("unidentified-program=0x0"), std::string::npos);
    EXPECT_TRUE(format_arbitrary_placeholder_summary({}).empty());
}
