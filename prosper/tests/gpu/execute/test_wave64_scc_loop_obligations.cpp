// Wave64 SCC writers must carry loop obligations into numeric CSELECT results.
// Dropping the roots makes a genuine counter look Hard; keeping stale roots can admit a spill.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::arbitrary_placeholder;
using Words = fixture::Words;
constexpr uint32_t untouched = 0xdeadbeefu;
Words program(bool arithmetic) {
    // LLVM gfx1030 Wave64: s30=0; compare or OR writes SCC; select 1/2 into VCC_LO;
    // compare with lane; select output 7/5; increment; loop while s30<s12; store v1.
    Words words{
        0xbe9e0380u, 0xbf06811eu, 0x856a8281u, 0x7d84006au, 0xd5010001u, 0x01a90e85u,
        0x801e811eu, 0xbf0a0c1eu, 0xbf85fff8u, 0xe0702000u, 0x80020100u, 0xbf810000u,
    };
    if (arithmetic) words[1] = 0x881f801eu; // s_or_b32 s31,s30,0: SCC=(s30!=0)
    return words;
}
ComputeShaderConfig launch(bool native, uint32_t trips = 3) {
    auto config = fixture::config();
    config.native_subgroup_size = native ? 64 : 0;
    config.user_sgprs.resize(13);
    config.user_sgprs[12] = trips;
    return config;
}
struct Compilation {
    Words module;
    std::string reason;
};
Compilation compile(const Words& words, const ComputeShaderConfig& config) {
    const auto resources = fixture::table();
    static std::atomic<uint64_t> next{0x5521cc0000ull};
    const uint64_t address = next.fetch_add(0x100);
    auto module = recompile_compute(words.data(), words.size(), &resources, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
Words expected(bool arithmetic, uint32_t trips) {
    const uint32_t counter = std::max(trips, 1u) - 1u;
    const uint32_t selected = (arithmetic ? counter != 0 : counter == 1) ? 1u : 2u;
    Words result(64, 5u);
    result[selected] = 7u;
    return result;
}
void interpreter(bool arithmetic, bool native, uint32_t trips) {
    SCOPED_TRACE(arithmetic);
    SCOPED_TRACE(native);
    SCOPED_TRACE(trips);
    const auto config = launch(native, trips);
    const auto result = compile(program(arithmetic), config);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module);
    // Both translator configurations execute as one 64-lane instance in this oracle.
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = config.user_sgprs;
    const auto actual = vm.run_buffers(64, {{3, Words(64, untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, expected(arithmetic, trips));
}
void gpu(bool native) {
    for (bool arithmetic : {false, true})
        for (uint32_t trips : {1u, 2u, 3u, 4u}) {
            SCOPED_TRACE(arithmetic);
            SCOPED_TRACE(trips);
            const auto config = launch(native, trips);
            const auto result = compile(program(arithmetic), config);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            Words actual;
            prosper::test::run_compute(result.module, {0.0f}, 64, 1, {}, Words(64, untouched),
                                       &actual, 64, nullptr, nullptr, nullptr, native ? 64 : 0,
                                       nullptr, config.user_sgprs);
            EXPECT_EQ(actual, expected(arithmetic, trips));
        }
}
} // namespace

TEST(Wave64SccLoopObligations, CompareAndArithmeticSelectBothOutcomesInTheInterpreter) {
    for (bool arithmetic : {false, true})
        for (bool native : {false, true})
            for (uint32_t trips : {1u, 2u, 3u, 4u}) interpreter(arithmetic, native, trips);
}

TEST(Wave64SccLoopObligations, MissingSpillCannotSupplyTheCounter) {
    auto words = program(false);
    words[0] = 0xbf800000u; // mutation: remove the genuine counter initializer
    words = fixture::with_missing_spill_root(words, 30);
    for (bool native : {false, true}) {
        const auto result = compile(words, launch(native));
        EXPECT_TRUE(result.module.empty()) << "a conditional missing spill remains Hard";
    }
}

TEST(Wave64SccLoopObligations, RootRepairDoesNotInventAnArbitraryAdmission) {
    const auto result = compile(program(false), launch(false));
    ASSERT_FALSE(result.module.empty()) << result.reason;
    EXPECT_EQ(arbitrary_placeholder_reads(result.module).total(), 0u);
}

TEST(Wave64SccLoopObligations, MissingSpillCannotReplaceTheSuppliedLoopBound) {
    const auto words = fixture::with_missing_spill_root(program(false), 12);
    for (bool native : {false, true}) {
        const auto result = compile(words, launch(native));
        EXPECT_TRUE(result.module.empty()) << "old launch s12 cannot hide its missing replacement";
    }
}

TEST(Wave64SccLoopObligations, PortableWave64ExecutesOnTheGpu) {
    gpu(false);
}

TEST(Wave64SccLoopObligations, RequiredWave64ExecutesOnTheGpu) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    gpu(true);
}
