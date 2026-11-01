// The runtime bank must use its declared allocation and current VGPR0, not a zero fallback.
// Omitting the allocated lowering or the per-candidate Hard guard must redden these arms.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/allocated_movrels.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_relative_vgpr_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::allocated_movrels;
void execute(uint32_t offset, uint32_t wave, bool rewrite = false) {
    SCOPED_TRACE(offset);
    SCOPED_TRACE(wave);
    auto module = fixture::compile(offset, wave, rewrite);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    vm.supplied_push_constants = fixture::config(offset, wave).user_sgprs;
    const auto result = vm.run(wave, std::vector<uint32_t>(wave, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(result, fixture::expected(offset, wave, rewrite));
}
} // namespace
TEST(AllocatedMovrels, RuntimeOffsetsReadTheDeclaredBankAndOobVgprZero) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t offset : {0u, 1u, 6u, 8u, 22u, 23u, 255u, UINT32_MAX}) execute(offset, wave);
}
TEST(AllocatedMovrels, OutOfRangeUsesCurrentRatherThanLaunchVgprZero) {
    execute(23, 64, true);
    execute(UINT32_MAX, 32, true);
}
TEST(AllocatedMovrels, UserValuesDoNotSpecializeTheRuntimeChoice) {
    EXPECT_EQ(fixture::compile(0), fixture::compile(23));
}
TEST(AllocatedMovrels, UnknownAllocationCannotAuthorizeAnUnboundedIndex) {
    auto c = fixture::config(23);
    c.allocated_vgprs = 0;
    const auto words = fixture::program();
    const auto resources = prosper::test::arbitrary_placeholder::table();
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), &resources, c).empty());
}
TEST(AllocatedMovrels, ADeclaredAllocationCannotSupplyALostCandidate) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    auto c = fixture::config(1);
    const auto words = prosper::test::arbitrary_placeholder::relative_after_writelane();
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), &resources, c).empty());
    const auto provided = prosper::test::arbitrary_placeholder::relative_after_writelane(false);
    EXPECT_FALSE(recompile_compute(provided.data(), provided.size(), &resources, c).empty());
}
TEST(AllocatedMovrels, AllocationMustAgreeWithRsrc1GranularityAndWaveSize) {
    EXPECT_TRUE(compute_vgpr_allocation_is_canonical(0, UINT32_MAX, 64));
    EXPECT_TRUE(compute_vgpr_allocation_is_canonical(24, 5, 64));
    EXPECT_TRUE(compute_vgpr_allocation_is_canonical(24, 2, 32));
    for (uint32_t words : {1u, 20u, 28u, 260u})
        EXPECT_FALSE(compute_vgpr_allocation_is_canonical(words, 5, 64));
    EXPECT_FALSE(compute_vgpr_allocation_is_canonical(24, UINT32_MAX, 64));
    EXPECT_FALSE(compute_vgpr_allocation_is_canonical(24, 5, 32));
    auto c = fixture::config(23);
    c.allocated_vgprs = 20;
    const auto words = fixture::program();
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), nullptr, c).empty());
}

TEST(AllocatedMovrels, RelativeWritesToAllocationPaddingSurviveACfgReload) {
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t offset : {20u, 22u, 23u}) {
            auto module = fixture::compile(offset, wave, false, true);
            ASSERT_FALSE(module.empty()) << wave << ',' << offset;
            bpermute_oracle::Interpreter vm(module);
            vm.native_subgroup_width = wave;
            vm.supplied_push_constants = fixture::config(offset, wave).user_sgprs;
            const auto output = vm.run(wave, std::vector<uint32_t>(wave, 0xdeadbeefu));
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(output, offset < 23 ? std::vector<uint32_t>(wave, 105u)
                                          : fixture::expected(offset, wave));
        }
    }
}
TEST(AllocatedMovrels, AnExplicitWordBeyondTheDeclaredAllocationStaysUnsupported) {
    auto c = fixture::config(0);
    c.allocated_vgprs = 20;
    c.compute_pgm_rsrc1 = (c.compute_pgm_rsrc1 & ~63u) | 4u;
    const auto words = fixture::program(); // explicit v20 does not fit in v0..v19
    const auto resources = prosper::test::arbitrary_placeholder::table();
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), &resources, c).empty());
}
TEST(AllocatedMovrels, StrictReadsCannotLoseSpillStorageOrTheCurrentZeroFallback) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    auto c = fixture::config(0);
    c.user_sgprs[0] = 13;
    for (bool fallback : {false, true}) {
        for (bool alias : {false, true}) {
            const auto lost = fixture::strict_spill_source(fallback, alias, true);
            EXPECT_TRUE(recompile_compute(lost.data(), lost.size(), &resources, c).empty())
                << fallback << ',' << alias;
            const auto provided = fixture::strict_spill_source(fallback, alias, false);
            const auto module = recompile_compute(provided.data(), provided.size(), &resources, c);
            ASSERT_FALSE(module.empty());
            bpermute_oracle::Interpreter vm(module);
            vm.native_subgroup_width = 64;
            vm.supplied_push_constants = c.user_sgprs;
            const auto output = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(output, std::vector<uint32_t>(64, 7u));
        }
    }
}
TEST(AllocatedMovrels, AllocationAlsoBoundsTheSecondTerminatingArm) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    for (uint32_t condition : {0u, 1u}) {
        const auto c = fixture::config(condition);
        const auto outside = fixture::terminating_arms(24);
        EXPECT_TRUE(recompile_compute(outside.data(), outside.size(), &resources, c).empty());
        const auto inside = fixture::terminating_arms(23);
        const auto module = recompile_compute(inside.data(), inside.size(), &resources, c);
        ASSERT_FALSE(module.empty());
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = c.user_sgprs;
        const auto output = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(output, std::vector<uint32_t>(64, condition == 0 ? 7u : 13u));
    }
}
TEST(AllocatedMovrels, AllocationBoundsArmsExposedByConstantBranchSpecialization) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    for (uint32_t condition : {0u, 1u}) {
        const auto c = fixture::config(condition);
        const auto outside = fixture::terminating_arms(24, true);
        EXPECT_TRUE(recompile_compute(outside.data(), outside.size(), &resources, c).empty());
        const auto inside = fixture::terminating_arms(23, true);
        const auto module = recompile_compute(inside.data(), inside.size(), &resources, c);
        ASSERT_FALSE(module.empty());
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = c.user_sgprs;
        const auto output = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(output, std::vector<uint32_t>(64, condition == 0 ? 7u : 13u));
    }
}
