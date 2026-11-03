// Real required Wave64 lanes must retain exactly the same active/inactive split.
#include "fixtures/inactive_destination_storage.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>

namespace {
namespace f = prosper::test::inactive_destination_storage;
void execute(f::OldValue old, f::Operation operation, f::Observation observation, uint32_t input,
             const f::Words& expected) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    const auto result = f::compile(true, old, operation, observation);
    ASSERT_FALSE(result.module.empty()) << result.roots;
    const auto floats =
        prosper::test::run_compute(result.module, {0.0f}, 64, 64, {}, {}, nullptr, 64, nullptr,
                                   nullptr, nullptr, 64, nullptr, {input});
    f::Words actual;
    for (float value : floats) actual.push_back(std::bit_cast<uint32_t>(value));
    EXPECT_EQ(actual, expected);
}
} // namespace
TEST(InactiveDestinationStorageGpu, StoredInactiveBitsDoNotBlockACompleteActiveWrite) {
    execute(f::OldValue::Placeholder, f::Operation::Copy, f::Observation::ActiveOnly, 23,
            f::expected(23));
}
TEST(InactiveDestinationStorageGpu, GenuineInactiveBitsSurviveACompleteActiveWrite) {
    execute(f::OldValue::Supplied, f::Operation::Copy, f::Observation::AllLanes, 13,
            f::expected(13, f::Observation::AllLanes));
}
TEST(InactiveDestinationStorageGpu, AGenuineAccumulatorUsesTheOldActiveBits) {
    execute(f::OldValue::Supplied, f::Operation::Accumulate, f::Observation::ActiveOnly,
            0x3f000000u, f::expected(0x3f400000u));
}
