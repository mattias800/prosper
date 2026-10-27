// A wide source's high word and an implicit condition can control EXEC without a data read.
// Removing those reader checks must not turn missing real inputs into a zero condition.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>

namespace {
namespace fixture = prosper::test::arbitrary_placeholder;
using namespace prosper::gpu;
void witness(const fixture::Words& words, const ComputeShaderConfig& launch, uint32_t expected) {
    const auto resources = fixture::table();
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = launch.user_sgprs;
    const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, fixture::Words(64, expected));
}
} // namespace

TEST(ArbitraryPlaceholderScalarWords, AnUnknownHighWordCannotControlExecThroughScc) {
    const auto words = fixture::bfe_scc_exec();
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs.resize(13);
    constexpr uint64_t address = 0x5497b04000ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("origin=unknown-launch word=s13"),
              std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=4 "), std::string::npos);
}

TEST(ArbitraryPlaceholderScalarWords, ASuppliedHighWordControlsTheWholeWave) {
    for (uint32_t high : {0u, 1u}) {
        SCOPED_TRACE(high);
        auto launch = fixture::config();
        launch.user_sgprs.resize(14);
        launch.user_sgprs[13] = high;
        witness(fixture::bfe_scc_exec(), launch, high ? 7 : 11);
    }
}

TEST(ArbitraryPlaceholderScalarWords, AnUnloadedHighWordKeepsItsConsistentZero) {
    witness(fixture::bfe_scc_exec(true), fixture::config(), 11);
}

TEST(ArbitraryPlaceholderScalarWords, AnUnknownImplicitSccCannotSelectExec) {
    const auto words = fixture::implicit_scc_exec();
    const auto resources = fixture::table();
    constexpr uint64_t address = 0x5497b04100ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("origin=unknown-launch word=scc"),
              std::string::npos);
}

TEST(ArbitraryPlaceholderScalarWords, ASuppliedImplicitSccControlsTheWholeWave) {
    for (uint32_t input : {0u, 1u}) {
        SCOPED_TRACE(input);
        auto launch = fixture::config();
        launch.user_sgprs[4] = input;
        witness(fixture::implicit_scc_exec(1), launch, input ? 11 : 7);
    }
}

TEST(ArbitraryPlaceholderScalarWords, AnUnloadedImplicitSccKeepsItsComputedValue) {
    witness(fixture::implicit_scc_exec(2), fixture::config(), 7);
}
