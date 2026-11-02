// A post-loop EXEC copy must keep its real entry HIGH word through CMPX and a LOW write.
#include "fixtures/native_entry_exec_words.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <cstdint>

namespace {
namespace fixture = prosper::test::native_entry_exec_words;
}

TEST(NativeEntryExecWords, CompleteGroupsKeepTheActualEntryPairAcrossTheLoop) {
    const auto c = fixture::config(false);
    const auto result = fixture::compile(fixture::program(), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    for (uint32_t group : {0u, 1u}) {
        bpermute_oracle::Interpreter vm(result.module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = c.user_sgprs;
        const auto actual = vm.run_buffers(64, {{3, fixture::output(false)}}, 3, group);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        auto expected = fixture::output(false);
        fixture::expected_group(expected, false, group);
        EXPECT_EQ(actual, expected);
    }
}

TEST(NativeEntryExecWords, TheLastPartialGroupKeepsItsRealFiftySixEntryBits) {
    const auto c = fixture::config(true);
    const auto result = fixture::compile(fixture::program(), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    for (uint32_t group : {0u, 64u}) {
        bpermute_oracle::Interpreter vm(result.module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = c.user_sgprs;
        const auto actual = vm.run_buffers(64, {{3, fixture::output(true)}}, 3, group);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        auto expected = fixture::output(true);
        fixture::expected_group(expected, true, group);
        EXPECT_EQ(actual, expected);
    }
}

TEST(NativeEntryExecWords, NormalizedGeometryCannotReplaceMissingOrInconsistentLaunchWords) {
    auto c = fixture::config(true);
    c.dispatch_launch = {};
    EXPECT_TRUE(fixture::compile(fixture::program(), c).module.empty());
    c = fixture::config(true);
    c.dispatch_launch.num_thread[0] ^= 1u << 16;
    EXPECT_TRUE(fixture::compile(fixture::program(), c).module.empty());
}

TEST(NativeEntryExecWords, ARealMissingSpillCannotBecomeAnEntryMask) {
    auto code = fixture::program();
    code[0] = 0xbe82041au; // saved s[2:3] comes from the missing s[26:27] source instead
    code = prosper::test::arbitrary_placeholder::with_missing_spill_root(code, 26);
    EXPECT_TRUE(fixture::compile(code, fixture::config(true)).module.empty());
}

TEST(NativeEntryExecWords, AScalarMultiplyRetiresOnlyTheOldPredicateNotTheCurrentWords) {
    for (const bool partial : {false, true}) {
        const auto c = fixture::config(partial);
        const auto result = fixture::compile(fixture::program(true), c);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        for (const uint32_t group : {0u, partial ? 64u : 1u}) {
            bpermute_oracle::Interpreter vm(result.module);
            vm.native_subgroup_width = 64;
            vm.supplied_push_constants = c.user_sgprs;
            const auto actual = vm.run_buffers(64, {{3, fixture::output(partial)}}, 3, group);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            auto expected = fixture::output(partial);
            fixture::expected_group(expected, partial, group, true);
            EXPECT_EQ(actual, expected);
        }
    }
}

TEST(NativeEntryExecWords, MissingForceStartCannotAuthenticateThePartialEntryMask) {
    auto c = fixture::config(true);
    c.dispatch_launch.initiator &= ~uint64_t{4};
    EXPECT_TRUE(fixture::compile(fixture::program(), c).module.empty());
}

TEST(NativeEntryExecWords, ArithmeticCannotLaunderAnUnprovedReadfirstPublication) {
    const auto c = fixture::config(false);
    const auto result = fixture::compile(fixture::readfirst_program(false), c);
    EXPECT_TRUE(result.module.empty()) << "a real broadcast is required";
    if (result.module.empty()) return;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = c.user_sgprs;
    const auto actual = vm.run_buffers(64, {{3, fixture::Words(64, fixture::untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    fixture::Words expected(64, 0);
    expected[0] = 1; // hardware READFIRST selects local lane zero, so every lane sees scalar 1
    EXPECT_EQ(actual, expected);
}

TEST(NativeEntryExecWords, ACompleteCurrentOverwriteRetiresTheUnprovedArithmeticRelay) {
    const auto c = fixture::config(false);
    const auto result = fixture::compile(fixture::readfirst_program(true), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = c.user_sgprs;
    const auto actual = vm.run_buffers(64, {{3, fixture::Words(64, fixture::untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    fixture::Words expected(64, 0);
    expected[0] = 1;
    EXPECT_EQ(actual, expected);
}

TEST(NativeEntryExecWords, APcZeroBackedgeCannotHideAReadfirstPublication) {
    auto c = fixture::config(false);
    c.user_sgprs.resize(16);
    const auto result = fixture::compile(fixture::pc0_program(false), c);
    EXPECT_TRUE(result.module.empty());
    if (result.module.empty()) return;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = c.user_sgprs;
    const auto actual = vm.run_buffers(64, {{3, fixture::Words(64, fixture::untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    fixture::Words expected(64, 0);
    expected[0] = 1;
    EXPECT_EQ(actual, expected);
}

TEST(NativeEntryExecWords, ACurrentOverwriteRetiresThePcZeroBackedgePublication) {
    auto c = fixture::config(false);
    c.user_sgprs.resize(16);
    const auto result = fixture::compile(fixture::pc0_program(true), c);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = c.user_sgprs;
    const auto actual = vm.run_buffers(64, {{3, fixture::Words(64, fixture::untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    fixture::Words expected(64, 0);
    expected[0] = 1;
    EXPECT_EQ(actual, expected);
}
