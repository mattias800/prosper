// #5225: separate product rounding loses exact, normal half results in both mixed destinations.
#include "fixtures/mixed_f32_fma.hpp"
#include "fixtures/half_fma_finite.hpp"
#include "fixtures/half_fma_reference.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::mixed_f32;
namespace {
std::vector<uint32_t> run(const std::vector<uint32_t>& module, const LaneInputs& in) {
    bpermute_oracle::Interpreter vm(module);
    vm.half_rte_preserve_denorm = true;
    auto out = vm.run_buffers(64, {{3, prosper::test::half_fma::buffer(in)}}, 3);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}

uint32_t placed(uint32_t opcode, uint32_t old, uint32_t half) {
    return opcode == 0x21 ? (old & 0xffff0000u) | half : (old & 0xffffu) | (half << 16);
}

// Independent nearest-neighbour search for the finite, bounded fixture values. Distances
// between an F32 value and F16 candidates are exact in double; ties choose the even code.
uint32_t pack_rte(double value) {
    const double magnitude = std::fabs(value);
    uint32_t low = 0, high = 0x7bff;
    while (high - low > 1) {
        const uint32_t mid = (low + high) / 2;
        if (expand_half(mid) > magnitude)
            high = mid;
        else
            low = mid;
    }
    const double below = magnitude - expand_half(low), above = expand_half(high) - magnitude;
    const uint32_t nearest = below < above || (below == above && !(low & 1u)) ? low : high;
    return nearest | (std::signbit(value) ? 0x8000u : 0u);
}
} // namespace

TEST(IrMixedHalfFma, ExactHalfResiduesSurviveBothDestinationsAndExecPatterns) {
    // (8192+t)*(8192-t)-2^26 = -t^2, t=1,3,5,7. Each result is exactly F16.
    // Separately rounded F32 products instead leave 0, -8, -24 and -48.
    constexpr std::array<uint32_t, 4> halves{0xbc00, 0xc880, 0xce40, 0xd220};
    for (uint32_t opcode : {0x21u, 0x22u})
        for (uint32_t wave : {32u, 64u})
            for (bool empty : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "op=" << opcode << " wave=" << wave << " empty=" << empty);
                auto in = inputs();
                for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
                    const float t = float(2 * (lane % 4) + 1);
                    in.s0[lane] = std::bit_cast<uint32_t>(8192.0f + t);
                    in.s1[lane] = std::bit_cast<uint32_t>(8192.0f - t);
                    in.s2[lane] = 0xcc800000u;
                }
                if (empty) std::fill(in.active.begin(), in.active.end(), false);
                const auto p = program({}, wave, opcode);
                const auto module =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
                ASSERT_FALSE(module.empty());
                std::vector<uint32_t> data;
                for (const auto* source : {&in.s0, &in.s1, &in.old_d})
                    data.insert(data.end(), source->begin(), source->end());
                for (bool active : in.active) data.push_back(active);
                data.insert(data.end(), in.s2.begin(), in.s2.end());
                bpermute_oracle::Interpreter vm(module);
                vm.half_rte_preserve_denorm = true;
                const auto actual = vm.run_buffers(64, {{3, data}}, 3);
                ASSERT_TRUE(vm.error.empty()) << vm.error;
                ASSERT_EQ(actual.size(), data.size());
                for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto half = halves[lane % 4];
                    const auto written = opcode == 0x21 ? (in.old_d[lane] & 0xffff0000u) | half
                                                        : (in.old_d[lane] & 0xffffu) | (half << 16);
                    EXPECT_EQ(actual[lane], in.active[lane] ? written : in.old_d[lane])
                        << "lane " << lane;
                }
            }
}

TEST(IrMixedHalfFma, SelectorsModifiersAliasesAndBothWavesKeepTheOtherHalf) {
    for (const auto shape : shapes())
        for (uint32_t opcode : {0x21u, 0x22u})
            for (uint32_t wave : {32u, 64u}) {
                SCOPED_TRACE(::testing::Message() << "half=" << shape.half << " high=" << shape.high
                                                  << " opcode=" << opcode);
                const auto p = program(shape, wave, opcode);
                const auto module =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
                ASSERT_FALSE(module.empty());
                const auto in = inputs(shape);
                const auto actual = run(module, in);
                ASSERT_GE(actual.size(), in.lanes());
                for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto bits = expected(shape, in, lane);
                    const auto want =
                        placed(opcode, in.old_d[lane], pack_rte(std::bit_cast<float>(bits)));
                    EXPECT_EQ(actual[lane], in.active[lane] ? want : in.old_d[lane]) << lane;
                }
            }
}

TEST(IrMixedHalfFma, FullWidthSourcesSelectOneFmaAndAllHalfKeepsItsGraph) {
    for (uint32_t opcode : {0x21u, 0x22u})
        for (uint32_t half = 0; half < 8; ++half) {
            const auto p = program({half, half}, 64, opcode);
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            ASSERT_FALSE(module.empty());
            std::size_t fused = 0, separate = 0;
            for (std::size_t pc = 5; pc < module.size(); pc += module[pc] >> 16) {
                const uint32_t count = module[pc] >> 16, op = module[pc] & 65535;
                ASSERT_GT(count, 0u);
                ASSERT_LE(pc + count, module.size());
                if (op == Op_ExtInst && count >= 5 && module[pc + 4] == Glsl_Fma) ++fused;
                if (op == Op_FMul || op == Op_FAdd) ++separate;
            }
            EXPECT_EQ(fused, half == 7 ? 0u : 1u);
            EXPECT_EQ(separate, half == 7 ? 2u : 0u);
        }
}

TEST(IrMixedHalfFma, KnownDoubleRoundingMixedResultLimitation) {
    // CONFIDENCE: MED. Known double-rounding limitation, not strict direct-F16 rounding.
    // Exact (0x3f803000 * 1) - 2^-25 = 1 + 3*2^-11 - 2^-25, just below the
    // half midpoint. Direct F16 RTE is 0x3c01; F32 Fma reaches the midpoint and packs 0x3c02.
    auto in = inputs();
    std::fill(in.s0.begin(), in.s0.end(), 0x3f803000u);
    std::fill(in.s1.begin(), in.s1.end(), 0x3f800000u);
    std::fill(in.s2.begin(), in.s2.end(), 0xb3000000u);
    EXPECT_EQ(pack_rte(1.0 + 3.0 * std::ldexp(1.0, -11) - std::ldexp(1.0, -25)), 0x3c01u);
    EXPECT_EQ(pack_rte(std::bit_cast<float>(0x3f803000u)), 0x3c02u);
    for (uint32_t opcode : {0x21u, 0x22u})
        for (uint32_t wave : {32u, 64u}) {
            const auto p = program({}, wave, opcode);
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            ASSERT_FALSE(module.empty());
            const auto actual = run(module, in);
            ASSERT_GE(actual.size(), in.lanes());
            for (std::size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(actual[lane],
                          in.active[lane] ? placed(opcode, in.old_d[lane], 0x3c02) : in.old_d[lane])
                    << lane;
        }
}

TEST(IrMixedHalfFma, KnownDoubleRoundingPackedFallbackLimitationAndExactModeControl) {
    // CONFIDENCE: MED. Known double-rounding limitation under the hosted RTE pack/unpack
    // reference: 1.5h*(1+2^-10)h - 2^-24 = 25190399/16777216. Direct RTE is 0x3e01,
    // whereas the exact F32 product plus rounded F32 add packs to 0x3e02.
    namespace hf = prosper::test::half_fma;
    auto in = hf::inputs(hf::Kind::Packed);
    std::fill(in.s0.begin(), in.s0.end(), 0x3e003e00u);
    std::fill(in.s1.begin(), in.s1.end(), 0x3c013c01u);
    std::fill(in.s2.begin(), in.s2.end(), 0x80018001u);
    EXPECT_EQ(hf::reference(0x3e00, 0x3c01, 0x8001).bits, 0x3e01u);
    for (uint32_t wave : {32u, 64u})
        // Unknown launch MODE leaves the exact route unselected; this is a hosted-reference
        // limitation, not a claim about guest denormal handling under an unknown MODE.
        for (uint32_t mode : {UINT32_MAX, 0xc0000u}) {
            const auto module = hf::compile(hf::Kind::Packed, {}, wave, mode);
            ASSERT_FALSE(module.empty());
            const auto actual = run(module, in);
            ASSERT_GE(actual.size(), in.lanes());
            const uint32_t active = mode == UINT32_MAX ? 0x3e023e02u : 0x3e013e01u;
            for (std::size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(actual[lane], in.active[lane] ? active : in.old_d[lane]) << lane;
        }
}
