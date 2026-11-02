// Actual native Wave64 groups observe entry mask words after a loop and a partial VCC write.
#include "fixtures/native_entry_exec_words.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstdint>

namespace {
void device(bool partial, bool data_mask = false) {
    namespace fixture = prosper::test::native_entry_exec_words;
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    const auto c = fixture::config(partial);
    const auto result = fixture::compile(fixture::program(data_mask), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    fixture::Words actual;
    prosper::test::run_compute(result.module, {0.0f}, c.threads_x, 1, {}, fixture::output(partial),
                               &actual, 64, nullptr, nullptr, nullptr, 64, nullptr, c.user_sgprs);
    auto expected = fixture::output(partial);
    for (uint32_t group = 0; group < (partial ? 65u : 2u); ++group)
        fixture::expected_group(expected, partial, group, data_mask);
    EXPECT_EQ(actual, expected);
}
} // namespace
TEST(NativeEntryExecWordsGpu, CompleteGroupsKeepTheirEntryHighWord) {
    device(false);
}
TEST(NativeEntryExecWordsGpu, TheLastGroupKeepsItsActualFiftySixEntryBits) {
    device(true);
}

TEST(NativeEntryExecWordsGpu, CurrentNumericWordsSupplyTheMaskAfterScalarMultiplication) {
    device(false, true);
    device(true, true);
}

TEST(NativeEntryExecWordsGpu, ACurrentOverwriteRetiresTheReadfirstArithmeticRelay) {
    namespace fixture = prosper::test::native_entry_exec_words;
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    auto c = fixture::config(false);
    c.threads_x = 64;
    c.dispatch_launch.dimensions[0] = 1;
    const auto result = fixture::compile(fixture::readfirst_program(true), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    fixture::Words actual;
    prosper::test::run_compute(result.module, {0.0f}, 64, 1, {},
                               fixture::Words(64, fixture::untouched), &actual, 64, nullptr,
                               nullptr, nullptr, 64, nullptr, c.user_sgprs);
    fixture::Words expected(64, 0);
    expected[0] = 1;
    EXPECT_EQ(actual, expected);
}

TEST(NativeEntryExecWordsGpu, ACurrentOverwriteRetiresThePcZeroBackedgePublication) {
    namespace fixture = prosper::test::native_entry_exec_words;
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    auto c = fixture::config(false);
    c.user_sgprs.resize(16);
    c.threads_x = 64;
    c.dispatch_launch.dimensions[0] = 1;
    const auto result = fixture::compile(fixture::pc0_program(true), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    fixture::Words actual;
    prosper::test::run_compute(result.module, {0.0f}, 64, 1, {},
                               fixture::Words(64, fixture::untouched), &actual, 64, nullptr,
                               nullptr, nullptr, 64, nullptr, c.user_sgprs);
    fixture::Words expected(64, 0);
    expected[0] = 1;
    EXPECT_EQ(actual, expected);
}
