// A one-word scalar write must not erase an independent Wave32 mask in the other VCC word.
// Interpreter/GPU observations distinguish retained saved EXEC from stale overwritten predicates.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::arbitrary_placeholder;
ComputeShaderConfig config() {
    auto result = fixture::config();
    result.local_x = result.wave_size = 32;
    result.native_subgroup_size = 0;
    prosper::test::complete_compute_launch(result);
    return result;
}
fixture::Words program(bool writes_high, uint32_t source = 60) {
    auto words = fixture::scalar_relay();
    words.pop_back();
    if (writes_high) words.push_back(0x7e02028bu);   // definite inactive output = 11
    words.push_back(0x7d860085u);   // VCC_LO = (5 <= lane)
    if (!writes_high) words.push_back(0xbeeb3c6au);   // save full EXEC_LO in VCC_HI
    words.push_back((writes_high ? 0x8f6b8400u : 0x8f6a8400u) | source);
    // Scalar scratch is non-projectable; only the independent, existing B32 mask survives.
    if (writes_high) {
        // Observe the mask directly; CNDMASK's separate VCC source-mark pair guard stays intact.
        const fixture::Words suffix{
            0xbefe036au, 0x7e020287u,   // observe retained VCC_LO through EXEC, output 7
            0xbefe03c1u,   // restore full EXEC before storing the preserved old values
            0xe0702000u, 0x80020100u, 0xbf810000u,
        };
        words.insert(words.end(), suffix.begin(), suffix.end());
    } else {
        const fixture::Words suffix{
            0xbefe036bu,   // restore saved EXEC_LO, including the first five lanes
            0x7e020287u, 0xe0702000u, 0x80020100u, 0xbf810000u,
        };
        words.insert(words.end(), suffix.begin(), suffix.end());
    }
    return words;
}
struct Compilation {
    fixture::Words module;
    std::string reason;
};
Compilation compile(const fixture::Words& words, const ComputeShaderConfig& launch = config()) {
    const auto resources = fixture::table();
    static std::atomic<uint64_t> next{0x5039330000ull};
    const auto address = next.fetch_add(0x100);
    auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
fixture::Words expected(bool writes_high) {
    fixture::Words result(32, 7u);
    if (writes_high)
        for (unsigned lane = 0; lane < 5; ++lane) result[lane] = 11u;
    return result;
}
void interpreter(const fixture::Words& words, const fixture::Words& want) {
    const auto result = compile(words);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    ASSERT_GT(arbitrary_placeholder_reads(result.module).fresh_mask, 0u);
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = config().user_sgprs;
    const auto actual = vm.run(32, fixture::Words(32, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, want);
}
}   // namespace

TEST(Wave32VccMaskLifetime, LowScalarWritePreservesSavedHighExec) {
    interpreter(program(false), expected(false));
}

TEST(Wave32VccMaskLifetime, HighScalarWritePreservesLowPredicate) {
    interpreter(program(true), expected(true));
}

TEST(Wave32VccMaskLifetime, AnActualOverwriteReplacesTheSavedHighExec) {
    auto words = program(false);
    words.insert(words.end() - 5, 0xbeeb0380u);   // overwrite VCC_HI with real zero before restore
    interpreter(words, fixture::Words(32, 11u));   // only the preceding predicate store executes
}

TEST(Wave32VccMaskLifetime, NumericScratchDoesNotRetainItsOwnMask) {
    for (bool writes_high : {false, true}) {
        auto words = program(writes_high);
        words.erase(words.end() - 5, words.end() - 3);
        // Replace the final value producer with a read of the word just overwritten as DATA.
        const uint32_t mask = writes_high ? 107 : 106;
        words.insert(words.end() - 3, {0xd1010001u, 0x96u | (0x8bu << 9u) | (mask << 18u)});
        EXPECT_TRUE(compile(words).module.empty()) << writes_high;
    }
}

TEST(Wave32VccMaskLifetime, PairOverwriteStillRetiresTheHighMask) {
    auto words = program(false);
    words[8] = 0x8fea843cu;   // S_LSHL_B64 overwrites both VCC words
    EXPECT_TRUE(compile(words).module.empty());
}

TEST(Wave32VccMaskLifetime, AnOverlappingPairIsNotAnIndependentLowMask) {
    auto words = program(true);
    words[7] = 0xbeea047eu;   // a saved B64 EXEC alias overlaps the following VCC_HI write
    EXPECT_TRUE(compile(words).module.empty());
}

TEST(Wave32VccMaskLifetime, MissingLaunchAndSpillInputsRemainHard) {
    auto unknown = config();
    unknown.dispatch_launch = {};
    EXPECT_TRUE(compile(program(false), unknown).module.empty());
    for (bool writes_high : {false, true}) {
        EXPECT_TRUE(compile(program(writes_high, 12)).module.empty());
        const auto spill = compile(fixture::with_missing_spill_root(program(writes_high), 60));
        EXPECT_TRUE(spill.module.empty());
        EXPECT_NE(spill.reason.find("origin=spill"), std::string::npos) << spill.reason;
    }
}

TEST(Wave32VccMaskLifetime, BothIndependentWordsExecuteOnTheGpu) {
    for (bool writes_high : {false, true}) {
        const auto result = compile(program(writes_high));
        ASSERT_FALSE(result.module.empty()) << result.reason;
        std::vector<uint32_t> actual;
        prosper::test::run_compute(result.module, {0.0f}, 32, 1, {},
                                   fixture::Words(32, 0xdeadbeefu), &actual, 32, nullptr, nullptr,
                                   nullptr, 0, nullptr, config().user_sgprs);
        EXPECT_EQ(actual, expected(writes_high));
    }
}
