// VOP1 f16 unary math: actual per-lane emitted SPIR-V against the shared definitions,
// with an explicit RTE/denormal-preserving half oracle. Driver/native-unit bits are not claimed.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/ops/vector_half_unary.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
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
LaneInputs unary_half_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const uint32_t word = in.s0[lane];
        const uint32_t lo = ((12u + lane % 4u) << 10) | (word & 1023u);
        const uint32_t hi = ((11u + lane % 4u) << 10) | ((word >> 16) & 1023u);
        in.s0[lane] = lo | (hi << 16);
    }
    return in;
}
uint32_t source_half(uint32_t word, bool high) {
    return std::bit_cast<uint32_t>(half_to_float(static_cast<uint16_t>(high ? word >> 16 : word)));
}
struct Mode {
    bool sdwa, src_high, dst_high;
};
std::vector<uint32_t> words(uint32_t opcode, Mode m) {
    if (!m.sdwa) return {vop1(opcode)};
    // VOP1 SDWA: WORD source, WORD destination + UNUSED_PRESERVE, v2 as SRC0.
    return {vop1(opcode, kVdst, 249),
            ((m.src_high ? 5u : 4u) << 16) | ((0x10u | (m.dst_high ? 5u : 4u)) << 8) | kVsrc0};
}
}   // namespace
TEST(IrVectorHalfUnaryExecution, AllDefinitionsMatchEachLaneInPlainAndWordSdwaForms) {
    for (const auto& def : vector_half_unary_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 3; ++seed) {
                const auto in = unary_half_inputs(seed * 5232);
                for (Mode mode : {Mode{false, false, false}, Mode{true, true, false},
                                  Mode{true, false, true}}) {
                    const auto run =
                        run_vector_op_per_lane(words(def.opcode, mode), in, wave, true);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        if (!in.active[lane]) {
                            EXPECT_EQ(run.d[lane], in.old_d[lane]);
                            continue;
                        }
                        const auto want = eval_vector_op(
                            def, source_half(in.s0[lane], mode.src_high), std::nullopt);
                        if (!want) {
                            ADD_FAILURE() << "unknown lane " << lane;
                            continue;
                        }
                        const uint32_t placed = mode.dst_high
                                                    ? (in.old_d[lane] & 0xffffu) | (*want << 16)
                                                    : (in.old_d[lane] & 0xffff0000u) | *want;
                        EXPECT_EQ(run.d[lane], placed) << "lane " << lane << " wave " << wave;
                    }
                }
            }
    }
}
TEST(IrVectorHalfUnaryExecution, HarnessDistinguishesInputsAndInactiveWords) {
    const auto in = unary_half_inputs(79);
    const auto run = run_vector_op_per_lane({vop1(0x54)}, in, 64, true);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> values;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            values.insert(run.d[lane] & 0xffffu);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(values.size(), 16u);
    EXPECT_GE(inactive, 2u);
}
TEST(IrVectorHalfUnaryExecution, HandWorkedIsaRoundingRootsPowersAndTurns) {
    struct Example {
        uint32_t op, half, want;
    };
    // AMD 70648 VOP1 f16: -2 reciprocal=-1/2, sqrt(4)=2, log2(8)=3, exp2(3)=8;
    // fractions use x-floor(x), round-nearest ties go to even, and trig takes revolutions.
    for (auto x : {Example{0x54, 0xc000, 0xb800}, Example{0x55, 0x4400, 0x4000},
                   Example{0x56, 0x4400, 0x3800}, Example{0x57, 0x4800, 0x4200},
                   Example{0x58, 0x4200, 0x4800}, Example{0x5b, 0xbd00, 0xc000},
                   Example{0x5c, 0xbf00, 0xbc00}, Example{0x5d, 0xbf00, 0xbc00},
                   Example{0x5e, 0x4100, 0x4000}, Example{0x5e, 0x4300, 0x4400},
                   Example{0x5e, 0xc100, 0xc000}, Example{0x5e, 0xb400, 0x8000},
                   Example{0x5f, 0xbd00, 0x3a00}, Example{0x60, 0x3400, 0x3c00},
                   Example{0x61, 0x3800, 0xbc00}}) {
        SCOPED_TRACE(x.op);
        const auto* def = find_vector_half_unary_op(Rdna2Format::VOP1, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, source_half(x.half, false), std::nullopt), Known(x.want));
        auto in = unary_half_inputs(7);
        std::fill(in.s0.begin(), in.s0.end(), 0x12340000u | x.half);
        const auto run = run_vector_op_per_lane({vop1(x.op)}, in, 64, true);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane],
                      in.active[lane] ? (in.old_d[lane] & 0xffff0000u) | x.want : in.old_d[lane]);
    }
}
TEST(IrVectorHalfUnaryExecution, EffectsKeepTheOneSourceHalfResultDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_half_unary_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        SCOPED_TRACE(e.name);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_TRUE(e.vdst_half);
        EXPECT_FALSE(e.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        Block block;
        Builder ir(block);
        def.build(ir, {ir.input(Type::U32), ir.input(Type::U32)});
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    EXPECT_TRUE(seen.contains(Opcode::PackHalfLo));
    for (uint32_t op = uint32_t(Opcode::F32Fract); op <= uint32_t(Opcode::F32Cos); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op)));
    EXPECT_EQ(find_vector_half_unary_op(Rdna2Format::VOP1, 0x50), nullptr);
    EXPECT_EQ(find_vector_half_unary_op(Rdna2Format::VOP3, 0x1d4), nullptr);
}
TEST(IrVectorHalfUnaryExecution, UnknownOrUnspecifiedValuesAreNotGuessed) {
    for (const auto& def : vector_half_unary_defs()) {
        EXPECT_FALSE(eval_vector_op(def, std::nullopt, 0x3f800000));
        EXPECT_TRUE(eval_vector_op(def, 0x3fa00000, std::nullopt));
        EXPECT_FALSE(eval_vector_op(def, 0x7fc00001, std::nullopt));
    }
}
