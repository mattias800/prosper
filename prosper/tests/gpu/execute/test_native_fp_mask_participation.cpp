// Genuine FP comparison results must preserve native control participation, even for NaNs.
// Treating only integer compares as defined loses later saved-mask words after a real loop.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/compute_runner.h"
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
using Words = std::vector<uint32_t>;
constexpr uint32_t untouched = 0xdeadbeefu;
Words program(bool integer = false) {
    // LLVM gfx1030 Wave64: load per-lane bits; repeat a VCC branch three times;
    // publish a fresh s[14:15] compare, then observe the physical HIGH word plus the loop count.
    Words code{0xe0302000u, 0x80010200u, 0xbf8c3f70u, 0xbe940380u, 0xbe9e0301u, 0x7c040502u,
               0xbf870001u, 0x80148114u, 0x809e811eu, 0xbf07801eu, 0xbf85fffau, 0x7d8200f9u,
               0x06868e00u, 0x7e02020fu, 0x4a020214u, 0xe0702000u, 0x80020100u, 0xbf810000u};
    if (integer) code[5] = 0x7d840502u;   // u32 self equality: every bit pattern compares equal
    return code;
}
ComputeShaderConfig config(uint32_t trips) {
    auto c = fixture::config();
    c.user_sgprs[0] = 39;
    c.user_sgprs[1] = trips;
    c.float_transport = {FloatTransportProfile::ExplicitNonFinite32};
    return c;
}
ShaderResourceTable table() {
    auto result = fixture::table();
    auto input = result.resources.front();
    input.binding = 2;
    input.sgpr_base = 4;
    result.resources.push_back(input);
    return result;
}
struct Compilation {
    Words module;
    std::string reason;
};
Compilation compile(const Words& code, uint32_t trips = 3) {
    auto resources = table();
    const auto c = config(trips);
    static std::atomic<uint64_t> next{0x5039ffaa00ull};
    const auto address = next.fetch_add(0x100);
    auto module = recompile_compute(code.data(), code.size(), &resources, c,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
Words input(unsigned shape) {
    Words words(64, shape == 0 ? 0x3f800000u : 0x7fc00123u);
    if (shape == 2) words[63] = 0x7f800000u;   // one infinity in the other Wave32 half
    return words;
}
Words expected(unsigned shape, uint32_t trips, bool integer = false) {
    // s0=39 < local lane: bits40..63 are set, so HIGH=0xffffff00.
    // Only an all-NaN FP wave takes the counted body; the scalar branch is wave-wide.
    return Words(64, 0xffffff00u + (!integer && shape == 1 ? trips : 0u));
}
void interpreter(unsigned shape, uint32_t trips, bool integer = false) {
    SCOPED_TRACE(shape);
    SCOPED_TRACE(trips);
    const auto result = compile(program(integer), trips);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = config(trips).user_sgprs;
    const auto actual = vm.run_buffers(64, {{2, input(shape)}, {3, Words(64, untouched)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, expected(shape, trips, integer));
}
}   // namespace

TEST(NativeFpMaskParticipation, FiniteNanAndMixedWavesExecuteTheLoopAndPhysicalHighWord) {
    for (unsigned shape : {0u, 1u, 2u})
        for (uint32_t trips : {1u, 3u}) interpreter(shape, trips);
}

TEST(NativeFpMaskParticipation, TheExistingIntegerControlAcceptsEveryRawBitPattern) {
    for (unsigned shape : {0u, 1u, 2u}) interpreter(shape, 3, true);
}

TEST(NativeFpMaskParticipation, ARealMissingSpillComparandCannotSupplyThePredicate) {
    auto code = program();
    code.insert(code.begin() + 5, 0x7e04021au);   // v2 = s26, before the whole loop
    code = fixture::with_missing_spill_root(code, 26);
    EXPECT_TRUE(compile(code).module.empty());
}

TEST(NativeFpMaskParticipation, ARealMissingHighWordCannotSupplyAPairComparison) {
    auto code = program();
    code[5] = 0x7c440502u;   // F64 self comparison reads both v2 and v3
    code.insert(code.begin() + 5, 0x7e06021au);   // v3 = the missing spill result s26
    code = fixture::with_missing_spill_root(code, 26);
    EXPECT_TRUE(compile(code).module.empty());
}

TEST(NativeFpMaskParticipation, TheRequiredWave64DeviceExecutesBothBranchOutcomes) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 unavailable";
    for (unsigned shape : {0u, 1u, 2u})
        for (uint32_t trips : {1u, 3u}) {
            SCOPED_TRACE(shape);
            SCOPED_TRACE(trips);
            const auto c = config(trips);
            const auto result = compile(program(), trips);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            Words actual;
            prosper::test::run_compute(result.module, {0.0f}, 64, 1, input(shape),
                                       Words(64, untouched), &actual, 64, nullptr, nullptr, nullptr,
                                       64, nullptr, c.user_sgprs);
            EXPECT_EQ(actual, expected(shape, trips));
        }
}
