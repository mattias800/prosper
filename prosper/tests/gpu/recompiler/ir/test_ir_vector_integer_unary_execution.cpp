// Every migrated VOP1 integer operation executes in actual emitted SPIR-V, per lane with partial
// EXEC. Hand-worked ISA results distinguish a shared wrong evaluator/emitter definition.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/ops/vector_integer_unary.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

TEST(IrVectorIntegerUnaryExecution, EveryDefinitionMatchesEachActiveAndInactiveLane) {
    for (const auto& def : vector_integer_unary_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                auto in = lane_inputs(seed * 5227);
                in.active[12] =
                    true; // a zero source is explicitly active, including scan sentinels
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
                    EXPECT_EQ(run.d[lane], *want) << "lane " << lane << " wave " << wave;
                }
            }
    }
}

TEST(IrVectorIntegerUnaryExecution, HarnessDistinguishesSourcesAndInactiveDestinations) {
    const auto in = lane_inputs(23);
    const auto run = run_vector_op_per_lane({vop1(0x37)}, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> distinct;
    size_t inactive = 0;
    for (size_t i = 0; i < in.lanes(); ++i) {
        if (in.active[i])
            distinct.insert(run.d[i]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[i], in.old_d[i]);
            EXPECT_NE(run.d[i], ~in.s0[i]);
        }
    }
    EXPECT_GT(distinct.size(), 16u);
    EXPECT_GE(inactive, 2u);
}

TEST(IrVectorIntegerUnaryExecution, HandWorkedIsaValuesPinScanDirectionAndZeroSentinels) {
    struct Example {
        uint32_t op, input, want;
    };
    // 70648 12.8: FFBH counts leading zeroes; FFBL gives the LSB index. Empty is all ones.
    // 0x01234567 reverses bitwise into 0xe6a2c480 (not merely a byte reversal).
    for (auto x :
         {Example{0x01, 0x12345678, 0x12345678}, Example{0x37, 0, 0xffffffff},
          Example{0x37, 0x80000000, 0x7fffffff}, Example{0x38, 1, 0x80000000},
          Example{0x38, 0x01234567, 0xe6a2c480}, Example{kVop1OpcodeFfbhU32, 0, 0xffffffff},
          Example{kVop1OpcodeFfbhU32, 1, 31}, Example{kVop1OpcodeFfbhU32, 0x800000ff, 0},
          Example{kVop1OpcodeFfbhU32, 0x100000ff, 3}, Example{kVop1OpcodeFfblB32, 0, 0xffffffff},
          Example{kVop1OpcodeFfblB32, 0xff000008, 3}, Example{kVop1OpcodeFfblB32, 0xffff0000, 16},
          Example{kVop1OpcodeFfblB32, 0x80000000, 31}}) {
        const auto* def = find_vector_integer_unary_op(Rdna2Format::VOP1, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.input, std::nullopt), Known(x.want));
        auto in = lane_inputs(7);
        for (auto& source : in.s0) source = x.input;
        const auto run = run_vector_op_per_lane({vop1(x.op)}, in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t i = 0; i < in.lanes(); ++i)
            EXPECT_EQ(run.d[i], in.active[i] ? x.want : in.old_d[i]);
    }
}

TEST(IrVectorIntegerUnaryExecution, EffectsRequireExactlyOneWholeWordSource) {
    std::set<Opcode> seen;
    for (const auto& def : vector_integer_unary_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise)
            << "unary results stay in their source lane";
        EXPECT_EQ(effects.vdst_dwords, 1u) << "one complete destination word";
        EXPECT_FALSE(effects.vdst_half) << "integer unary results do not preserve a halfword";
        EXPECT_FALSE(effects.accumulates) << "the old destination is not an arithmetic input";
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u)
            << "only source zero is consumed";
        Block block;
        Builder ir(block);
        const auto a = ir.input(Type::U32), unused = ir.input(Type::U32);
        def.build(ir, {a, unused});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    EXPECT_TRUE(seen.contains(Opcode::FfbhU32));
    EXPECT_EQ(find_vector_integer_unary_op(Rdna2Format::VOP1, 0x02), nullptr);
    EXPECT_EQ(find_vector_integer_unary_op(Rdna2Format::VOP3, 0x1b7), nullptr);
}

TEST(IrVectorIntegerUnaryExecution, AnUnknownSourceIsNeverAnEmptyScan) {
    for (const auto& def : vector_integer_unary_defs()) {
        EXPECT_FALSE(eval_vector_op(def, std::nullopt, 0xffffffff));
        EXPECT_TRUE(eval_vector_op(def, 0x12345678, std::nullopt));
    }
}
