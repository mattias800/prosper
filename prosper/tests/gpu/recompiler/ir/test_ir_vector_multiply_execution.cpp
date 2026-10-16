// Actual emitted SPIR-V and shared eval agree per lane, with independent hand-worked ISA rails.
#include "gpu/recompiler/ir/ops/vector_multiply.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
std::vector<uint32_t> words(uint32_t opcode, uint32_t unused = 0) {
    return {0xd4000000u | (opcode << 16) | kVdst,
            (256u + kVsrc0) | ((256u + kVsrc1) << 9) | (unused << 18)};
}
}   // namespace
TEST(IrVectorMultiplyExecution, EveryDefinitionMatchesEachLaneInBothWaves) {
    for (const auto& def : vector_multiply_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = lane_inputs(seed * 5238);
                const auto run = run_vector_op_per_lane(words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want = eval_vector_op(def, in.s0[lane], in.s1[lane]);
                    if (!want) {
                        ADD_FAILURE() << "unknown lane " << lane;
                        continue;
                    }
                    EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
                        << "lane " << lane << " wave " << wave;
                }
            }
    }
}
TEST(IrVectorMultiplyExecution, HarnessSeesDifferentProductsAndPreservesInactiveDestinations) {
    const auto in = lane_inputs(23);
    const auto run = run_vector_op_per_lane(words(kVop3OpcodeMulLoU32), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> distinct;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            EXPECT_EQ(run.d[lane], in.s0[lane] * in.s1[lane]);
            distinct.insert(run.d[lane]);
        } else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(distinct.size(), 16u);
    EXPECT_GE(inactive, 2u);
}
TEST(IrVectorMultiplyExecution, HandWorkedIsaResultsDistinguishLowUnsignedHighAndSignedHigh) {
    struct Example {
        uint32_t opcode, a, b, want;
    };
    // AMD 70648 12.12: (2^32-1)^2 = 0xfffffffe00000001; (-1)^2 = 1.
    // (-2^31)*2 = 0xffffffff00000000; (-2^31)^2 = 0x4000000000000000.
    // Low products wrap modulo 2^32. These expectations are independent of eval.
    for (auto x : {Example{kVop3OpcodeMulLoU32, 0xffffffff, 0xffffffff, 1},
                   Example{kVop3OpcodeMulLoU32, 0x80000000, 2, 0},
                   Example{kVop3OpcodeMulLoU32, 0x12345678, 16, 0x23456780},
                   Example{kVop3OpcodeMulLoU32, 0, 0xffffffff, 0},
                   Example{kVop3OpcodeMulHiU32, 0xffffffff, 0xffffffff, 0xfffffffe},
                   Example{kVop3OpcodeMulHiU32, 0x80000000, 2, 1},
                   Example{kVop3OpcodeMulHiU32, 0x80000000, 0x80000000, 0x40000000},
                   Example{kVop3OpcodeMulHiU32, 0x7fffffff, 2, 0},
                   Example{kVop3OpcodeMulHiU32, 0, 0xffffffff, 0},
                   Example{kVop3OpcodeMulHiI32, 0xffffffff, 0xffffffff, 0},
                   Example{kVop3OpcodeMulHiI32, 0xffffffff, 2, 0xffffffff},
                   Example{kVop3OpcodeMulHiI32, 0x80000000, 2, 0xffffffff},
                   Example{kVop3OpcodeMulHiI32, 0x80000000, 0x80000000, 0x40000000},
                   Example{kVop3OpcodeMulHiI32, 0x7fffffff, 2, 0},
                   Example{kVop3OpcodeMulHiI32, 0x80000000, 0xffffffff, 0}}) {
        SCOPED_TRACE(::testing::Message() << "op " << x.opcode << " a " << x.a << " b " << x.b);
        const auto* def = find_vector_multiply_op(Rdna2Format::VOP3, x.opcode);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.a, x.b), Known(x.want));
        auto in = lane_inputs(7);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        const auto run = run_vector_op_per_lane(words(x.opcode), in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorMultiplyExecution, InstructionEffectsAndEncodedSourceCountStayAuthoritative) {
    std::set<Opcode> seen;
    const auto in = lane_inputs(31);
    for (const auto& def : vector_multiply_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_FALSE(e.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        const auto code = words(def.opcode, 127); // reserved SRC2 must not become an EXEC read
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].n_src, 2u);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::None);
        const auto normal = run_vector_op_per_lane(words(def.opcode), in);
        const auto reserved = run_vector_op_per_lane(code, in);
        ASSERT_TRUE(normal.error.empty()) << normal.error;
        ASSERT_TRUE(reserved.error.empty()) << reserved.error;
        EXPECT_EQ(normal.d, reserved.d);
        Block block;
        Builder ir(block);
        def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (auto op : {Opcode::IMul, Opcode::UMulHi, Opcode::SMulHi}) EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_multiply_op(Rdna2Format::VOP3, 0x16b), nullptr);
    EXPECT_EQ(find_vector_multiply_op(Rdna2Format::VOP3, kVop3OpcodeAddCoU32), nullptr);
    EXPECT_EQ(find_vector_multiply_op(Rdna2Format::VOP2, kVop3OpcodeMulLoU32), nullptr);
}
TEST(IrVectorMultiplyExecution, ExistingClampAndUnsupportedEncodingRefusalsRemain) {
    const auto in = lane_inputs(51);
    for (const auto& def : vector_multiply_defs()) {
        const auto good = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def.opcode);
        code[0] |= 1u << 15;
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
    const auto bad = run_vector_op_per_lane(words(0x16b), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
}
TEST(IrVectorMultiplyExecution, UnknownInputsRemainUnknown) {
    for (const auto& def : vector_multiply_defs()) {
        EXPECT_FALSE(eval_vector_op(def, std::nullopt, 7));
        EXPECT_FALSE(eval_vector_op(def, 7, std::nullopt));
    }
}
