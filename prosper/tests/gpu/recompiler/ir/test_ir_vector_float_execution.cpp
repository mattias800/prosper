// Binary32 VOP2 definitions against real emitted SPIR-V, with independent per-lane operands
// and EXEC-off destinations. Hand-worked ISA examples distinguish a shared wrong definition.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include "gpu/recompiler/ir/ops/vector_float.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <set>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
LaneInputs float_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0,          0x3f800000, 0xbf800000, 0x3f000000,
                              0xc0200000, 0x3f800001, 0x00800000, 0x47800000};
    for (size_t i = 0; i < in.lanes(); ++i) {
        // Finite normal random inputs keep execution inside SPIR-V's specified domain.
        in.s0[i] =
            i < std::size(edges) ? edges[i] : (in.s0[i] & 0x807fffffu) | ((100u + i % 40u) << 23);
        in.s1[i] = i < std::size(edges) ? edges[(i * 3 + 1) % std::size(edges)]
                                        : (in.s1[i] & 0x807fffffu) | ((105u + i % 35u) << 23);
    }
    return in;
}
}   // namespace

TEST(IrVectorFloatExecution, EveryDefinitionMatchesEachLaneInBothWaveSizes) {
    for (const auto& def : vector_float_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        for (uint32_t wave : {64u, 32u}) {
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = float_inputs(seed * 0x5220);
                const auto run = run_vector_op_per_lane({vop2(def.opcode)}, in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want = in.active[lane]
                                          ? eval_vector_op(def, in.s0[lane], in.s1[lane])
                                          : Known(in.old_d[lane]);
                    if (want) {
                        EXPECT_EQ(run.d[lane], *want)
                            << "wave " << wave << " seed " << seed << " lane " << lane;
                    } else {
                        ADD_FAILURE() << "unknown result at lane " << lane;
                    }
                }
            }
        }
    }
}

TEST(IrVectorFloatExecution, HarnessDistinguishesInputsAndInactiveDestinations) {
    const auto in = float_inputs(9);
    const auto run = run_vector_op_per_lane({vop2(0x04)}, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active_results;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (in.active[lane]) {
            active_results.insert(run.d[lane]);
        } else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    }
    EXPECT_GT(active_results.size(), 16u);
    EXPECT_GE(inactive, 2u);
}

TEST(IrVectorFloatExecution, DefinitionsAgreeWithEffectsAndUseEveryFloatIrOpcode) {
    std::set<Opcode> seen;
    for (const auto& def : vector_float_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        ASSERT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.accumulates);
        Block block;
        Builder ir(block);
        def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (uint32_t op = uint32_t(Opcode::F32Add); op <= uint32_t(Opcode::F32NMax); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op)));
    EXPECT_EQ(find_vector_float_op(Rdna2Format::VOP2, 0x28), nullptr);
    EXPECT_EQ(find_vector_float_op(Rdna2Format::VOP3, 0x103), nullptr);
}

TEST(IrVectorFloatExecution, HandWorkedIsaValuesDoNotComeFromTheEvaluator) {
    // AMD 70648 12.7: 1+2=3; 1-2=-1; reverse gives +1; 1.5*2=3.
    // MIN/MAX return the numeric operand for one quiet NaN. No folder scope is widened.
    struct Example {
        uint32_t op, a, b, want;
    };
    for (const auto& x : {Example{0x03, 0x3f800000, 0x40000000, 0x40400000},
                          Example{0x04, 0x3f800000, 0x40000000, 0xbf800000},
                          Example{0x05, 0x3f800000, 0x40000000, 0x3f800000},
                          Example{0x08, 0x3fc00000, 0x40000000, 0x40400000},
                          Example{0x0f, 0x7fc00001, 0xbf800000, 0xbf800000},
                          Example{0x10, 0x3f800000, 0x7fc00001, 0x3f800000}}) {
        const auto* def = find_vector_float_op(Rdna2Format::VOP2, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.a, x.b), Known(x.want));
        auto in = float_inputs(7);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        const auto run = run_vector_op_per_lane({vop2(x.op)}, in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}

TEST(IrVectorFloatExecution, UnknownOperandsAndUnspecifiedResultsStayUnknown) {
    const auto* add = find_vector_float_op(Rdna2Format::VOP2, 0x03);
    const auto* min = find_vector_float_op(Rdna2Format::VOP2, 0x0f);
    ASSERT_NE(add, nullptr);
    ASSERT_NE(min, nullptr);
    EXPECT_FALSE(eval_vector_op(*add, std::nullopt, 0x3f800000));
    EXPECT_FALSE(eval_vector_op(*min, 0x7fc00001, 0x7fc00002));
    EXPECT_FALSE(eval_vector_op(*min, 0, 0x80000000));
}
