// Inexact I32/U32 -> F32 needs actual RTE; zero's stored bits must be positive zero.
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include <optional>
#include "gpu/recompiler/ir/integer_float_rte_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_integer_float_rte_e64.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <ios>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace cases = prosper::test::ir::integer_float_rte;
TEST(IrIntegerFloatRteValues, HandWorkedTieParityCarrySignednessAndPositiveZero) {
    for (const auto& def : vector_integer_float_rte_e64_defs())
        for (const auto& h : cases::hand) {
            const uint32_t want = def.opcode == 0x185 ? h.signed_bits : h.unsigned_bits;
            EXPECT_EQ(cases::reference(h.raw, def.opcode == 0x185), want);
            EXPECT_EQ(eval_conversion_op(def, h.raw), want) << std::hex << h.raw;
        }
}
TEST(IrIntegerFloatRteValues, EveryRoundingBoundaryAndVaryingInputsAgreeWithIntegerOracle) {
    for (const auto& def : vector_integer_float_rte_e64_defs()) {
        uint32_t seed = 5394;
        for (uint32_t i = 0; i < 4096; ++i) {
            seed = seed * 1664525u + 1013904223u;
            EXPECT_EQ(eval_conversion_op(def, seed), cases::reference(seed, def.opcode == 0x185));
        }
        for (uint32_t exponent = 24; exponent < 32; ++exponent)
            for (uint32_t parity : {0u, 1u})
                for (uint32_t delta : {0u, 1u, 2u}) {
                    const uint32_t half = 1u << (exponent - 24u);
                    const uint32_t word = (1u << exponent) + (parity * 2u + 1u) * half + delta - 1u;
                    for (uint32_t raw : {word, 0u - word})
                        EXPECT_EQ(eval_conversion_op(def, raw),
                                  cases::reference(raw, def.opcode == 0x185));
                }
        EXPECT_EQ(eval_conversion_op(def, std::nullopt), std::nullopt);
        const auto& effect = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_EQ(effect.vdst_dwords, 1u);
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(effect.implicit_reads || effect.implicit_writes || effect.vdst_half);
    }
}
