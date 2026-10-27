// A relative bank's unloaded alternatives must not hide a lost provided candidate.
// Removing the actual-bank guard admits zero instead of WRITELANE's retained [13,7,...].
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <gtest/gtest.h>

namespace {
namespace fixture = prosper::test::arbitrary_placeholder;
using namespace prosper::gpu;
void witness(bool provide_word, uint64_t address, bool alias_destination = false) {
    const auto words = fixture::relative_after_writelane(false, provide_word, alias_destination);
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[4] = 1;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = launch.user_sgprs;
    const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, fixture::Words(64, provide_word ? 7 : 0));
}
} // namespace

TEST(ArbitraryPlaceholderRelative, ALostProvidedCandidateIsHardBesideUnloadedAlternatives) {
    const auto words = fixture::relative_after_writelane();
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    launch.user_sgprs[4] = 1;
    constexpr uint64_t address = 0x5497b03000ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (module.empty()) {
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=lost-definition word=v6"),
                  std::string::npos);
    } else {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = launch.user_sgprs;
        auto expected = fixture::Words(64, 7);
        expected[0] = 13;
        const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, expected);
    }
}

TEST(ArbitraryPlaceholderRelative, AProvidedRuntimeSelectedWordKeepsItsValue) {
    witness(true, 0x5497b03100ull);
}

TEST(ArbitraryPlaceholderRelative, AnActuallyUnloadedSelectedWordKeepsItsPlaceholder) {
    witness(false, 0x5497b03200ull);
}

TEST(ArbitraryPlaceholderRelative, ALostSelectedWordStaysHardWhenTheDestinationAliasesIt) {
    const auto words = fixture::relative_after_writelane(true, true, true);
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    launch.user_sgprs[4] = 1;
    constexpr uint64_t address = 0x5497b03600ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (module.empty()) {
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=lost-definition word=v6"),
                  std::string::npos);
    } else {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.supplied_push_constants = launch.user_sgprs;
        auto expected = fixture::Words(64, 7);
        expected[0] = 13;
        const auto actual = vm.run(64, fixture::Words(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, expected);
    }
}

TEST(ArbitraryPlaceholderRelative, AnAliasedProvidedWordKeepsItsRuntimeValue) {
    witness(true, 0x5497b03800ull, true);
}

TEST(ArbitraryPlaceholderRelative, AnAliasedUnloadedWordKeepsItsPlaceholder) {
    witness(false, 0x5497b03900ull, true);
}
