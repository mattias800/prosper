// Owned native structural fixtures must declare physical launch authority, not only a width.
#include "fixtures/native_structural_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 6> vcc_high = {
    0x7e160280u,                // v_mov_b32 v11, 0: a genuine owned compare input
    0xbeeb03ffu, 0x3f400000u,   // vcc_hi = 0.75f, overwritten by the Wave64 compare
    0x7d841680u,                // v_cmp_eq_u32 vcc, 0, v11
    0x7e4e026bu,                // v_mov_b32 v39, vcc_hi
    0xbf810000u,
};
constexpr std::array<uint32_t, 6> exec_pair = {
    0xbefe04c1u,   // exec = -1
    0x80907e80u,   // s16 = 0 - exec_lo
    0x82917f80u,   // s17 = 0 - exec_hi - borrow
    0x7e040210u, 0x7e060211u, 0xbf810000u,
};
ComputeShaderConfig config() {
    ComputeShaderConfig parent;
    parent.wave_size = parent.native_subgroup_size = parent.local_x = 32;
    parent.user_sgprs = {1u, 1u};
    return prosper::test::native_structural::wave64_config(parent);
}
std::vector<uint32_t> compile(const std::array<uint32_t, 6>& code,
                              const ComputeShaderConfig& launch) {
    return recompile_compute(code.data(), code.size(), nullptr, launch);
}
size_t ballots(const std::vector<uint32_t>& code) {
    size_t count = 0;
    for (size_t pc = 5; pc < code.size();) {
        const size_t width = code[pc] >> 16;
        if (!width || width > code.size() - pc) return 0;
        count += (code[pc] & 0xffffu) == 339u;
        pc += width;
    }
    return count;
}
}   // namespace

TEST(NativeStructuralLaunch, VccHighIsTheCurrentBallotWord) {
    const auto code = compile(vcc_high, config());
    ASSERT_FALSE(code.empty());
    EXPECT_GE(ballots(code), 1u) << "the overwritten 0.75f scalar cannot satisfy the mask read";
}

TEST(NativeStructuralLaunch, ExecIntegerPairUsesBothPhysicalBallots) {
    const auto code = compile(exec_pair, config());
    ASSERT_FALSE(code.empty());
    EXPECT_EQ(ballots(code), 2u);
}

TEST(NativeStructuralLaunch, WidthAndNormalizedGeometryAloneAreInsufficient) {
    auto launch = config();
    launch.dispatch_launch = {};
    EXPECT_TRUE(compile(vcc_high, launch).empty());
    EXPECT_TRUE(compile(exec_pair, launch).empty());
}

TEST(NativeStructuralLaunch, PartialGroupCannotSupplyPhysicalWords) {
    auto launch = config();
    launch.dispatch_launch.initiator |= 2u;
    EXPECT_TRUE(compile(vcc_high, launch).empty());
    EXPECT_TRUE(compile(exec_pair, launch).empty());
}

TEST(NativeStructuralLaunch, EveryRawProducingWordIsRequired) {
    const auto complete = config();
    for (uint32_t bit = 0; bit < 7; ++bit) {
        auto launch = complete;
        launch.dispatch_launch.known &= ~(1u << bit);
        EXPECT_TRUE(compile(vcc_high, launch).empty()) << "missing raw word " << bit;
        EXPECT_TRUE(compile(exec_pair, launch).empty()) << "missing raw word " << bit;
    }
}

TEST(NativeStructuralLaunch, WrongNativeWidthCannotSupplyPhysicalWords) {
    for (uint32_t width : {0u, 32u}) {
        auto launch = config();
        launch.native_subgroup_size = width;
        EXPECT_TRUE(compile(vcc_high, launch).empty()) << "native width " << width;
        EXPECT_TRUE(compile(exec_pair, launch).empty()) << "native width " << width;
    }
}
