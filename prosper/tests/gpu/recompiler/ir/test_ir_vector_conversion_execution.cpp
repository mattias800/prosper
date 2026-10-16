// Existing VOP1 conversion semantics: real emitted SPIR-V vs shared eval on differing lanes,
// partial EXEC and both waves. Hand-worked AMD 70648 cases independently pin rounding/rails.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "support/death_test.hpp"
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
LaneInputs conversion_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const uint32_t word = in.s0[lane];
        // Finite as F32, low/high F16 and arbitrary signed/unsigned integer words.
        const uint32_t half =
            ((14u + uint32_t(lane % 7)) << 10) | (word & 1023u) | ((word >> 3) & 0x8000u);
        in.s0[lane] = ((125u + uint32_t(lane % 4)) << 23) | half | (word & 0x80000000u);
    }
    return in;
}
const ConversionDef* definition(uint32_t op) {
    if (const auto* d = find_vector_conversion_op(Rdna2Format::VOP1, op)) return d;
    return find_vector_half_integer_conversion_op(Rdna2Format::VOP1, op);
}
void expect_plain(const ConversionDef& def, const LaneInputs& in, uint32_t wave) {
    const auto run = run_vector_op_per_lane({vop1(def.opcode)}, in, wave, true);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto d = eval_conversion_op(def, in.s0[lane]);
        if (!d) {
            ADD_FAILURE() << "unknown conversion value";
            continue;
        }
        const uint32_t placed = effects.vdst_half ? (in.old_d[lane] & 0xffff0000u) | *d : *d;
        EXPECT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " op " << def.opcode;
    }
}
std::vector<uint32_t> sdwa(uint32_t op, uint32_t src, uint32_t dst, bool preserve = true) {
    return {vop1(op, kVdst, 249), (src << 16) | ((dst | (preserve ? 0x10u : 0u)) << 8) | kVsrc0};
}
}   // namespace
TEST(IrVectorConversionExecution, EveryDefinitionMatchesEveryLaneInBothWaves) {
    for (auto defs : {vector_conversion_defs(), vector_half_integer_conversion_defs()})
        for (const auto& def : defs) {
            SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 5; ++seed)
                    expect_plain(def, conversion_inputs(seed * 5231), wave);
        }
}
TEST(IrVectorConversionExecution, HarnessSeesDifferentInputsAndInactiveDestinations) {
    const auto in = conversion_inputs(17);
    const auto* def = definition(0x06);
    ASSERT_NE(def, nullptr);
    const auto run = run_vector_op_per_lane({vop1(def->opcode)}, in);
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
TEST(IrVectorConversionExecution, HalfIntegerSdwaSelectsAndPreservesEitherWord) {
    for (const auto& def : vector_half_integer_conversion_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t source : {4u, 5u, 6u})
                for (uint32_t dst : {4u, 5u}) {
                    const auto in = conversion_inputs(97);
                    const auto run =
                        run_vector_op_per_lane(sdwa(def.opcode, source, dst), in, wave, true);
                    ASSERT_TRUE(run.error.empty()) << run.error << " op " << def.opcode;
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        const auto d =
                            eval_conversion_op(def, source == 5 ? in.s0[lane] >> 16 : in.s0[lane]);
                        if (!d) {
                            ADD_FAILURE() << "unknown conversion value";
                            continue;
                        }
                        const uint32_t placed = dst == 5 ? (in.old_d[lane] & 0xffffu) | (*d << 16)
                                                         : (in.old_d[lane] & 0xffff0000u) | *d;
                        EXPECT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane]);
                    }
                }
}
TEST(IrVectorConversionExecution, FloatToIntegerSdwaPreservesOtherBytesAndWords) {
    for (uint32_t op : {0x07u, 0x08u})
        for (uint32_t wave : {64u, 32u})
            for (uint32_t dst = 0; dst < 6; ++dst) {
                const auto* def = definition(op);
                ASSERT_NE(def, nullptr);
                const auto in = conversion_inputs(139);
                const auto run = run_vector_op_per_lane(sdwa(op, 6, dst), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                const uint32_t width = dst <= 3 ? 8u : 16u;
                const uint32_t shift = dst <= 3 ? dst * 8 : dst == 5 ? 16 : 0;
                const uint32_t mask = ((1u << width) - 1u) << shift;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto d = eval_conversion_op(*def, in.s0[lane]);
                    if (!d) {
                        ADD_FAILURE() << "unknown conversion value";
                        continue;
                    }
                    const uint32_t placed = (in.old_d[lane] & ~mask) | ((*d << shift) & mask);
                    EXPECT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane]);
                }
            }
}
TEST(IrVectorConversionExecution, F32HalfSdwaSourceSelectionAndDestinationPadding) {
    const auto in = conversion_inputs(181);
    for (uint32_t wave : {64u, 32u}) {
        const auto* unpack = definition(0x0b);
        const auto* pack = definition(0x0a);
        ASSERT_NE(unpack, nullptr);
        ASSERT_NE(pack, nullptr);
        for (uint32_t src : {4u, 5u, 6u}) {
            const auto run = run_vector_op_per_lane(sdwa(0x0b, src, 6, false), in, wave, true);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const auto d = eval_conversion_op(*unpack, in.s0[lane], src == 5);
                if (!d) {
                    ADD_FAILURE() << "unknown conversion value";
                    continue;
                }
                EXPECT_EQ(run.d[lane], in.active[lane] ? *d : in.old_d[lane]);
            }
        }
        for (uint32_t dst : {4u, 5u, 6u}) {
            const auto run = run_vector_op_per_lane(sdwa(0x0a, 6, dst, dst != 6), in, wave, true);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const auto d = eval_conversion_op(*pack, in.s0[lane]);
                if (!d) {
                    ADD_FAILURE() << "unknown conversion value";
                    continue;
                }
                const uint32_t placed = dst == 6   ? *d
                                        : dst == 5 ? (in.old_d[lane] & 0xffffu) | (*d << 16)
                                                   : (in.old_d[lane] & 0xffff0000u) | *d;
                EXPECT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane]);
            }
        }
    }
}
TEST(IrVectorConversionExecution, HandWorkedIsaRoundingSaturationNibblesAndBytes) {
    struct Example {
        uint32_t op, input, want;
    };
    // AMD 70648 12.8, binary32/binary16 encodings calculated by hand. RPI ties go toward +inf;
    // nibble 8 means -8/16; byte conversion selects the named byte, never sign-extends it.
    for (auto x : {Example{0x05, 0xffffffff, 0xbf800000}, Example{0x06, 0xffffffff, 0x4f800000},
                   Example{0x07, 0x7fc00001, 0},          Example{0x07, 0x7f800000, 0xffffffff},
                   Example{0x07, 0xff800000, 0},          Example{0x07, 0x80000000, 0},
                   Example{0x08, 0x7fc00001, 0},          Example{0x08, 0x7f800000, 0x7fffffff},
                   Example{0x08, 0xff800000, 0x80000000}, Example{0x08, 0xbfe00000, 0xffffffff},
                   Example{0x0c, 0x3effffff, 0},          Example{0x0c, 0x3f000000, 1},
                   Example{0x0c, 0xbfc00000, 0xffffffff}, Example{0x0c, 0xbf000000, 0},
                   Example{0x0d, 0xbfe00000, 0xfffffffe}, Example{0x0e, 8, 0xbf000000},
                   Example{0x0e, 0xdeadbeef, 0xbd800000}, Example{0x0e, 7, 0x3ee00000},
                   Example{0x11, 0x12345678, 0x42f00000}, Example{0x12, 0x12345678, 0x42ac0000},
                   Example{0x13, 0x12345678, 0x42500000}, Example{0x14, 0x12345678, 0x41900000}}) {
        SCOPED_TRACE(x.op);
        SCOPED_TRACE(x.input);
        const auto* def = definition(x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_conversion_op(*def, x.input), Known(x.want));
        auto in = conversion_inputs(13);
        std::fill(in.s0.begin(), in.s0.end(), x.input);
        const auto run = run_vector_op_per_lane({vop1(x.op)}, in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorConversionExecution, HandWorkedHalfBoundariesAndTruncation) {
    struct Example {
        uint32_t op, input, want;
    };
    // Half RTE: 1+2^-11 ties to 1; 1+3*2^-11 ties upward to the even mantissa. The
    // smallest half subnormal is 2^-24; 65520 is the half overflow tie. Signed rails clamp.
    for (auto x : {Example{0x0a, 0x3f801000, 0x3c00}, Example{0x0a, 0x3f803000, 0x3c02},
                   Example{0x0a, 0x33800000, 1},      Example{0x0a, 0x80000000, 0x8000},
                   Example{0x0a, 0x477fe000, 0x7bff}, Example{0x0a, 0x477ff000, 0x7c00},
                   Example{0x0b, 1, 0x33800000},      Example{0x0b, 0x8000, 0x80000000},
                   Example{0x0b, 0x7c00, 0x7f800000}, Example{0x0b, 0xb800, 0xbf000000},
                   Example{0x50, 0xffff, 0x7c00},     Example{0x51, 0xffff, 0xbc00},
                   Example{0x51, 0x8000, 0xf800},     Example{0x52, 0xbc00, 0},
                   Example{0x52, 0x7bff, 0xffe0},     Example{0x52, 0x7c00, 0xffff},
                   Example{0x52, 0x3e00, 1},          Example{0x53, 0x7bff, 0x7fff},
                   Example{0x53, 0xfbff, 0x8000},     Example{0x53, 0xbf00, 0xffff},
                   Example{0x53, 0x7c00, 0x7fff},     Example{0x53, 0xfc00, 0x8000}}) {
        SCOPED_TRACE(x.op);
        SCOPED_TRACE(x.input);
        const auto* def = definition(x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_conversion_op(*def, x.input), Known(x.want));
        auto in = conversion_inputs(23);
        std::fill(in.s0.begin(), in.s0.end(), x.input);
        const auto run = run_vector_op_per_lane({vop1(x.op)}, in, 64, true);
        ASSERT_TRUE(run.error.empty()) << run.error;
        const bool half = rdna2_instruction_effects(def->fmt, def->opcode).vdst_half;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const uint32_t placed = half ? (in.old_d[lane] & 0xffff0000u) | x.want : x.want;
            EXPECT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane]);
        }
    }
}
TEST(IrVectorConversionExecution, EffectsAndPrimitivesAgreeWithTheDefinitionDomain) {
    std::set<Opcode> seen;
    for (auto defs : {vector_conversion_defs(), vector_half_integer_conversion_defs()})
        for (const auto& def : defs) {
            const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
            EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
            EXPECT_EQ(e.vdst_dwords, 1u);
            EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
            EXPECT_FALSE(e.accumulates);
            Block block;
            Builder ir(block);
            def.build(ir, {ir.input(Type::U32), false});
            for (const auto& inst : block.insts()) seen.insert(inst.op);
        }
    for (uint32_t op = uint32_t(Opcode::I32ToF32); op <= uint32_t(Opcode::F32Ge); ++op)
        EXPECT_TRUE(seen.contains(Opcode(op)));
    EXPECT_EQ(find_vector_conversion_op(Rdna2Format::VOP3, 0x05), nullptr);
    EXPECT_EQ(find_vector_conversion_op(Rdna2Format::VOP1, 0x50), nullptr);
    EXPECT_EQ(find_vector_half_integer_conversion_op(Rdna2Format::VOP1, 0x0a), nullptr);
}
TEST(IrVectorConversionExecution, UnknownHalfNaNsAndInvalidComponentsAreNotGuessed) {
    for (auto defs : {vector_conversion_defs(), vector_half_integer_conversion_defs()})
        for (const auto& def : defs) EXPECT_FALSE(eval_conversion_op(def, std::nullopt));
    EXPECT_FALSE(eval_conversion_op(*definition(0x0b), 0x7e01));
    EXPECT_FALSE(eval_conversion_op(*definition(0x52), 0x7e01));
    EXPECT_FALSE(evaluate_op(Opcode::UnpackHalf, 2, 0x3c00, 0, 0));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Block block;
            Builder ir(block);
            ir.unpack_half(ir.c32(0x3c00), 2);
        },
        "half component out of range");
}
