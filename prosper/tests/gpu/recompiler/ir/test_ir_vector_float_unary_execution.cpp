// Existing VOP1 F32 unary definitions execute in real emitted SPIR-V with differing lanes
// and partial EXEC. This is a nominal finite-domain reference, not device approximation bits.
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
LaneInputs unary_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    // Positive finite normal inputs keep all thirteen forms in their shared specified domain.
    for (auto& word : in.s0) word = std::bit_cast<uint32_t>(float((word % 65535u) + 1u) / 4096.0f);
    return in;
}
}   // namespace
TEST(IrVectorFloatUnaryExecution, EveryDefinitionMatchesEachActiveAndInactiveLane) {
    for (const auto& def : vector_float_unary_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = unary_inputs(seed * 5229);
                const auto run = run_vector_op_per_lane({vop1(def.opcode)}, in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want = in.active[lane]
                                          ? eval_vector_op(def, in.s0[lane], std::nullopt)
                                          : Known(in.old_d[lane]);
                    if (!want) {
                        ADD_FAILURE() << "unknown lane " << lane;
                        continue;
                    }
                    EXPECT_EQ(run.d[lane], *want)
                        << "wave " << wave << " seed " << seed << " lane " << lane;
                }
            }
    }
}
TEST(IrVectorFloatUnaryExecution, HarnessDistinguishesInputsAndInactiveDestinations) {
    const auto in = unary_inputs(37);
    const auto run = run_vector_op_per_lane({vop1(0x2a)}, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
}
TEST(IrVectorFloatUnaryExecution, HandWorkedIsaRoundingAndExactPowerCases) {
    struct Example {
        uint32_t op, input, want;
    };
    // AMD 70648 12.8, worked independently: fractions are x-floor(x), ties go to even,
    // and the listed powers/roots/reciprocals are exact. SIN/COS inputs are revolutions.
    for (auto x : {Example{0x20, 0xbfa00000, 0x3f400000}, Example{0x21, 0xbfe00000, 0xbf800000},
                   Example{0x22, 0xbfe00000, 0xbf800000}, Example{0x23, 0x40200000, 0x40000000},
                   Example{0x23, 0x40600000, 0x40800000}, Example{0x23, 0xc0200000, 0xc0000000},
                   Example{0x23, 0xbe800000, 0x80000000}, Example{0x24, 0xbfa00000, 0xc0000000},
                   Example{0x25, 0x40400000, 0x41000000}, Example{0x27, 0x41000000, 0x40400000},
                   Example{0x2a, 0x40000000, 0x3f000000}, Example{0x2b, 0x40800000, 0x3e800000},
                   Example{0x2e, 0x40800000, 0x3f000000}, Example{0x33, 0x40800000, 0x40000000},
                   Example{0x35, 0x3e800000, 0x3f800000}, Example{0x36, 0x3f000000, 0xbf800000}}) {
        SCOPED_TRACE(x.op);
        const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.input, std::nullopt), Known(x.want));
        auto in = unary_inputs(7);
        std::fill(in.s0.begin(), in.s0.end(), x.input);
        const auto run = run_vector_op_per_lane({vop1(x.op)}, in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorFloatUnaryExecution, TrigQuarterTurnRetainsTheBackendApproximationEnvelope) {
    // AMD 70648's ideal cos(1/4 revolution) is zero. The retained rounded-2*pi lowering
    // need not produce zero bits: Vulkan's SPIR-V precision table permits absolute error
    // <= 2^-11 for Cos in [-pi,pi]. This checks that bound, not native-unit bit identity.
    // docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation
    auto in = unary_inputs(7);
    std::fill(in.s0.begin(), in.s0.end(), 0x3e800000u);
    const auto run = run_vector_op_per_lane({vop1(0x36)}, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, 0x36);
    ASSERT_NE(def, nullptr);
    const auto want = eval_vector_op(*def, 0x3e800000u, std::nullopt);
    if (!want) {
        ADD_FAILURE() << "quarter-turn reference is unknown";
        return;
    }
    EXPECT_LE(std::fabs(std::bit_cast<float>(*want)), 0x1p-11f);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            EXPECT_EQ(run.d[lane], *want);
            EXPECT_LE(std::fabs(std::bit_cast<float>(run.d[lane])), 0x1p-11f);
        } else
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
}
TEST(IrVectorFloatUnaryExecution, EffectsAndDefinitionsKeepTheExistingOneWordLaneDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_float_unary_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half);
        EXPECT_FALSE(effects.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        Block block;
        Builder ir(block);
        def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (uint32_t op = uint32_t(Opcode::F32Fract); op <= uint32_t(Opcode::F32Cos); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op)));
    EXPECT_EQ(find_vector_float_unary_op(Rdna2Format::VOP1, 0x02), nullptr);
    EXPECT_EQ(find_vector_float_unary_op(Rdna2Format::VOP3, 0x1a0), nullptr);
}
TEST(IrVectorFloatUnaryExecution, UnknownAndUnspecifiedDomainsAreNotGuessed) {
    for (const auto& def : vector_float_unary_defs()) {
        EXPECT_FALSE(eval_vector_op(def, std::nullopt, 0x3f800000));
        EXPECT_TRUE(eval_vector_op(def, 0x3fa00000, std::nullopt));
        EXPECT_FALSE(eval_vector_op(def, 0x7fc00001, std::nullopt));
    }
    for (auto op : {0x27u, 0x2eu, 0x33u}) {
        const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, op);
        ASSERT_NE(def, nullptr);
        EXPECT_FALSE(eval_vector_op(*def, 0xbf800000, std::nullopt));
    }
}
