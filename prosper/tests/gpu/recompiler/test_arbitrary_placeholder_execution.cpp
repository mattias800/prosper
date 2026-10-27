// Removing unloaded-only admission must refuse these real compiler/interpreter witnesses.
// Unknown launch and genuinely unsupplied pixel launch words remain refused twins.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <atomic>

namespace {
namespace fixture = prosper::test::arbitrary_placeholder;
using namespace prosper::gpu;
struct Compilation {
    fixture::Words module;
    std::string reason;
};
Compilation compile(const fixture::Words& words, const ComputeShaderConfig& launch) {
    auto resources = fixture::table();
    static std::atomic<uint64_t> next{0x5497b00000ull};
    const auto address = next.fetch_add(0x100);
    auto result = recompile_compute(words.data(), words.size(), &resources, launch,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(result), last_terminal_reject_reason(address)};
}
void witness(const fixture::Words& words, bool narrowed = false) {
    const auto result = compile(words, fixture::config());
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = fixture::config().user_sgprs;
    const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, fixture::expected(narrowed));
}
} // namespace

TEST(ArbitraryPlaceholderExecution, AScalarRelayRetainsOneConsistentValue) {
    witness(fixture::scalar_relay());
}

TEST(ArbitraryPlaceholderExecution, ASdwaSavedPredicateRetainsTheSameArbitraryValue) {
    witness(fixture::sdwa_predicate());
}

TEST(ArbitraryPlaceholderExecution, TheNarrowedWriterKeepsOldValuesInTheOtherHalf) {
    witness(fixture::narrowed_writer(), true);
}

TEST(ArbitraryPlaceholderExecution, ARepeatedLoopReadRetainsItsCarriedValue) {
    witness(fixture::loop_relay());
}

TEST(ArbitraryPlaceholderExecution, RemovingTheLaunchInventoryKeepsTheSameRelayHard) {
    auto launch = fixture::config();
    launch.dispatch_launch = {};
    const auto result = compile(fixture::scalar_relay(), launch);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("origin=unknown-launch word=s60"), std::string::npos)
        << result.reason;
}

TEST(ArbitraryPlaceholderExecution, ChangingTheSameRootToAMissingSpillStillRefuses) {
    const auto words = fixture::with_missing_spill_root(fixture::scalar_relay(), 60);
    const auto result = compile(words, fixture::config());
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("origin=spill word=spill-v255:0"), std::string::npos)
        << result.reason;
}

TEST(ArbitraryPlaceholderExecution, ALoadedButUnsuppliedPixelWordStillRefuses) {
    const fixture::Words words{0x7e0c0300u, 0x7d840d06u, 0x7e000280u, 0xf800080fu, 0u, 0xbf810000u};
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1;
    constexpr uint64_t address = 0x5497b00100ull;
    EXPECT_TRUE(recompile_fragment(words.data(), words.size(), nullptr, &inputs, UINT32_MAX,
                                   nullptr, false, {RecompileDiagnosticStage::Fragment, address})
                    .empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("origin=launch-input word=v0"), std::string::npos) << reason;
}
