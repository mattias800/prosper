// SMEM must preserve a genuine SCC branch: losing it discards physical mask-word backing.
// Preserving a missing SCC must not invent the participation needed to publish those words.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr uint32_t untouched = 0xdeadbeefu;
struct Options {
    bool memory = true, branch = true, compare = true;
    uint32_t condition = 0;
};
std::vector<uint32_t> compile(Options o) {
    auto config = prosper::test::native_execution_launch();
    config.user_sgprs.resize(12);
    config.user_sgprs[4] = o.condition;
    auto resources = prosper::test::arbitrary_placeholder::table();
    resources.resources[0].fetch_pc = 1;
    std::vector<uint32_t> code{o.compare ? 0xbf068004u : 0xbf800000u};
    code.reserve(12);
    if (o.memory) code.insert(code.end(), {0xf4000380u, 0xfa000000u}); // load s14
    if (o.branch) code.push_back(0xbf840005u); // SCC0 skips the mask producer and output
    code.insert(code.end(), {0x7d8800f9u, 0x068694a8u, // s[20:21] = 40 > local_id.x
                             0x7e020215u,             // v1 = physical high mask word
                             0xe0702000u, 0x80020100u, 0xbf810000u});
    return recompile_compute(code.data(), code.size(), &resources, config);
}
uint32_t expected(Options o) {
    return o.branch && o.condition ? untouched : 255u;
}
void interpreter(Options o) {
    SCOPED_TRACE(o.memory);
    SCOPED_TRACE(o.branch);
    SCOPED_TRACE(o.condition);
    const auto module = compile(o);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants.resize(12);
    vm.push_constants[4] = o.condition;
    const auto result = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(result, std::vector<uint32_t>(64, expected(o)));
}
} // namespace

TEST(NativeSccPreservation, MemoryKeepsBothOutcomesOfTheGenuineBranch) {
    for (uint32_t condition : {0u, 1u}) interpreter({.condition = condition});
}
TEST(NativeSccPreservation, NoMemoryAndNoBranchRemainIndependentControls) {
    for (bool memory : {false, true})
        for (bool branch : {false, true})
            for (uint32_t condition : {0u, 1u})
                if (!memory || !branch)
                    interpreter({.memory = memory, .branch = branch, .condition = condition});
}
TEST(NativeSccPreservation, MemoryCannotInventAnEntryCondition) {
    EXPECT_TRUE(compile({.compare = false}).empty());
}
TEST(NativeSccPreservation, BothBranchOutcomesExecuteOnRequiredWave64) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t condition : {0u, 1u}) {
        const Options o{.condition = condition};
        const auto module = compile(o);
        ASSERT_FALSE(module.empty());
        std::vector<uint32_t> constants(12);
        constants[4] = condition;
        std::vector<uint32_t> result;
        prosper::test::run_compute(module, {0.0f}, 64, 1, {}, std::vector<uint32_t>(64, untouched),
                                   &result, 64, nullptr, nullptr, nullptr, 64, nullptr, constants);
        EXPECT_EQ(result, std::vector<uint32_t>(64, expected(o)));
    }
}
