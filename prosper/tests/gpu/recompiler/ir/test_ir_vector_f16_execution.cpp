// Half arithmetic/RTZ packing against actual emitted SPIR-V. The oracle explicitly selects RTE
// and denormal preservation; this does not establish a host driver's implicit float environment.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_half.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_half.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
LaneInputs half_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const auto normal = [](uint32_t word, uint32_t exponent) {
        return (word & 0x8000u) | (exponent << 10) | (word & 1023u);
    };
    for (size_t i = 0; i < in.lanes(); ++i) {
        in.s0[i] = normal(in.s0[i], 1 + i % 25) | (normal(in.s0[i] >> 16, 2 + i % 23) << 16);
        in.s1[i] = normal(in.s1[i], 2 + i % 24) | (normal(in.s1[i] >> 16, 1 + i % 22) << 16);
    }
    return in;
}
uint32_t half_source(uint32_t word, bool high) {
    return std::bit_cast<uint32_t>(half_to_float(static_cast<uint16_t>(high ? word >> 16 : word)));
}
struct Mode {
    bool sdwa, high, clamp, pad;
};
std::vector<uint32_t> words(uint32_t op, Mode mode) {
    if (!mode.sdwa) return {vop2(op)};
    // llvm-mc gfx1030: v_add_f16_sdwa v1,v2,v3 clamp WORD_0/PRESERVE -> 640206f9 04043402.
    const uint32_t dst = mode.pad ? 6u : mode.high ? 0x15u : 0x14u;
    const uint32_t src1 = mode.high ? 5u : 4u;
    return {vop2(op, kVdst, 249),
            (src1 << 24) | (4u << 16) | ((dst | (mode.clamp ? 0x20u : 0u)) << 8) | kVsrc0};
}
}   // namespace

TEST(IrVectorF16Execution, EachLaneMatchesEvaluationAcrossHalfSelectionsAndExec) {
    for (const auto& def : vector_half_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        SCOPED_TRACE(effects.name);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 3; ++seed) {
                const auto in =
                    effects.vdst_half ? half_inputs(seed * 997) : lane_inputs(seed * 997);
                for (Mode mode : {Mode{false, false, false, false}, Mode{true, true, false, false},
                                  Mode{true, false, true, false}, Mode{true, false, false, true}}) {
                    if (!effects.vdst_half && mode.sdwa) continue;
                    const auto run =
                        run_vector_op_per_lane(words(def.opcode, mode), in, wave, true);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        if (!in.active[lane]) {
                            EXPECT_EQ(run.d[lane], in.old_d[lane]);
                            continue;
                        }
                        const auto want =
                            effects.vdst_half
                                ? eval_half_op(def, half_source(in.s0[lane], false),
                                               half_source(in.s1[lane], mode.high), mode.clamp)
                                : eval_half_op(def, in.s0[lane], in.s1[lane]);
                        if (!want) {
                            ADD_FAILURE() << "unknown lane " << lane;
                            continue;
                        }
                        const uint32_t placed = !effects.vdst_half || mode.pad ? *want
                                                : mode.high
                                                    ? (in.old_d[lane] & 0xffffu) | (*want << 16)
                                                    : (in.old_d[lane] & 0xffff0000u) | *want;
                        EXPECT_EQ(run.d[lane], placed) << "lane " << lane << " wave " << wave;
                    }
                }
            }
    }
}

TEST(IrVectorF16Execution, HarnessSeesDistinctSourcesAndInactiveFullDestinations) {
    const auto in = half_inputs(7);
    const auto run = run_vector_op_per_lane({vop2(0x32)}, in, 64, true);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> distinct;
    size_t inactive = 0;
    for (size_t i = 0; i < in.lanes(); ++i) {
        if (in.active[i])
            distinct.insert(run.d[i] & 0xffffu);
        else {
            ++inactive;
            EXPECT_EQ(run.d[i], in.old_d[i]);
        }
    }
    EXPECT_GT(distinct.size(), 16u);
    EXPECT_GE(inactive, 2u);
}

TEST(IrVectorF16Execution, HandWorkedIsaArithmeticAndRtzCases) {
    struct Example {
        uint32_t op, a, b, want;
    };
    // AMD70648 12.7: 1+2=3, 1-2=-1, 1.5*2=3, min/max(-1,0.5).
    for (auto x :
         {Example{0x32, 0x3c00, 0x4000, 0x4200}, Example{0x33, 0x3c00, 0x4000, 0xbc00},
          Example{0x35, 0x3e00, 0x4000, 0x4200}, Example{0x39, 0xbc00, 0x3800, 0x3800},
          Example{0x3a, 0xbc00, 0x3800, 0xbc00}, Example{0x2f, 0x3f801000, 0xc7800000, 0xfbff3c00},
          Example{0x2f, 0x7f800000, 0xff800000, 0xfc007c00}}) {
        auto in = half_inputs(5);
        for (size_t i = 0; i < in.lanes(); ++i) {
            in.s0[i] = x.a;
            in.s1[i] = x.b;
        }
        const auto* def = find_vector_half_op(Rdna2Format::VOP2, x.op);
        ASSERT_NE(def, nullptr);
        const bool half = rdna2_instruction_effects(def->fmt, def->opcode).vdst_half;
        EXPECT_EQ(eval_half_op(*def, half ? half_source(x.a, false) : x.a,
                               half ? half_source(x.b, false) : x.b),
                  Known(x.want));
        const auto run = run_vector_op_per_lane({vop2(x.op)}, in, 64, true);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t i = 0; i < in.lanes(); ++i)
            EXPECT_EQ(run.d[i], !in.active[i] ? in.old_d[i]
                                : half        ? (in.old_d[i] & 0xffff0000u) | x.want
                                              : x.want);
    }
}

TEST(IrVectorF16Execution, HandWorkedRteTiesExponentCarryAndSubnormals) {
    // Half has 10 fraction bits. At 1, tie 2^-11 goes to even 1; 3*2^-11 goes to the
    // even second successor. Just below 2 carries the significand into the next exponent.
    struct Example {
        uint32_t input, want;
    };
    for (auto x :
         {Example{0x3f801000, 0x3c00}, Example{0x3f803000, 0x3c02}, Example{0x3fffffff, 0x4000},
          Example{0x33800000, 1}, Example{0x33000000, 0}, Example{0x33000001, 1},
          Example{0x477fe000, 0x7bff}, Example{0x477ff000, 0x7c00}, Example{0x80000000, 0x8000}})
        EXPECT_EQ(pack_half_rte(x.input), Known(x.want));
    EXPECT_FALSE(pack_half_rte(0x7fc00001));
}

TEST(IrVectorF16Execution, EffectsAndPrimitiveCoverageDoNotAdmitLegacyAccumulators) {
    std::set<Opcode> seen;
    for (const auto& def : vector_half_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.accumulates);
        Block block;
        Builder ir(block);
        const auto a = ir.input(Type::U32), b = ir.input(Type::U32);
        def.build(ir, {a, b, true});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    EXPECT_TRUE(seen.contains(Opcode::BoolAnd));
    EXPECT_TRUE(seen.contains(Opcode::F32Clamp01));
    EXPECT_TRUE(seen.contains(Opcode::PackHalfLo));
    EXPECT_EQ(find_vector_half_op(Rdna2Format::VOP2, 0x3c), nullptr);
    const auto* def = find_vector_half_op(Rdna2Format::VOP2, 0x32);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_half_op(*def, std::nullopt, 0x3f800000));
}
