// Mutation: guard the stored inactive word as an active source of a complete-word writer.
#include "fixtures/inactive_destination_storage.hpp"
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>

namespace {
namespace f = prosper::test::inactive_destination_storage;
void execute(const f::Words& module, uint32_t input, const f::Words& expected) {
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = {input};
    const auto actual = vm.run_buffers(64, {{0, f::Words(64)}, {1, f::Words(64, f::sentinel)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, expected);
}
} // namespace
TEST(InactiveDestinationStorage, ACompleteActiveWriteDoesNotReadStoredInactiveBits) {
    for (bool native : {false, true}) {
        SCOPED_TRACE(native);
        const auto result = f::compile(native);
        ASSERT_FALSE(result.module.empty()) << result.roots;
        EXPECT_EQ(prosper::gpu::arbitrary_placeholder_reads(result.module).total(), 0u);
        for (uint32_t input : {13u, 23u}) execute(result.module, input, f::expected(input));
    }
}
TEST(InactiveDestinationStorage, AStoredGenuineValueSurvivesUntilTheWiderRead) {
    for (bool native : {false, true}) {
        const auto result =
            f::compile(native, f::OldValue::Supplied, f::Operation::Copy, f::Observation::AllLanes);
        ASSERT_FALSE(result.module.empty()) << result.roots;
        execute(result.module, 13, f::expected(13, f::Observation::AllLanes));
    }
}
TEST(InactiveDestinationStorage, APartialWriteCannotRecoverMissingOrReservedOldStorage) {
    for (bool native : {false, true})
        for (auto old : {f::OldValue::Missing, f::OldValue::Reserved}) {
            const auto result = f::compile(native, old);
            EXPECT_TRUE(result.module.empty());
            EXPECT_NE(result.roots.find("origin=launch-input word=v0"), std::string::npos);
        }
}
TEST(InactiveDestinationStorage, ASubsequentWiderReadStillNeedsTheRealInactiveInput) {
    for (bool native : {false, true}) {
        const auto result = f::compile(native, f::OldValue::Placeholder, f::Operation::Copy,
                                       f::Observation::AllLanes);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.roots.find("origin=launch-input word=v0"), std::string::npos);
        // An overbroad mutation must expose the wrong retained zero rather than hide behind
        // a verdict assertion: real TIDX is lane 16..63 in the newly observed inactive lanes.
        if (!result.module.empty())
            execute(result.module, 13, f::expected(13, f::Observation::AllLanes));
    }
}
TEST(InactiveDestinationStorage, AnAccumulatorStillReadsItsOldWordInActiveLanes) {
    for (bool native : {false, true}) {
        const auto result = f::compile(native, f::OldValue::Placeholder, f::Operation::Accumulate,
                                       f::Observation::IndependentSink);
        EXPECT_TRUE(result.module.empty());
        EXPECT_NE(result.roots.find("origin=launch-input word=v0"), std::string::npos);
        EXPECT_EQ(result.rejected_pc, 3u);   // the FMAC, before any later ordinary source guard
        EXPECT_EQ(result.rejected_opcode, 0x2bu);
        const auto supplied = f::compile(native, f::OldValue::Supplied, f::Operation::Accumulate,
                                         f::Observation::IndependentSink);
        ASSERT_FALSE(supplied.module.empty()) << supplied.roots;
        execute(supplied.module, 0x3f000000u, f::Words(64, f::sentinel));
    }
}
TEST(InactiveDestinationStorage, AGenuineAccumulatorStillExecutesItsCompleteOperation) {
    for (bool native : {false, true}) {
        const auto result = f::compile(native, f::OldValue::Supplied, f::Operation::Accumulate);
        ASSERT_FALSE(result.module.empty()) << result.roots;
        execute(result.module, 0x3f000000u, f::expected(0x3f400000u));   // 0.5*1.0 + 0.25
    }
}
