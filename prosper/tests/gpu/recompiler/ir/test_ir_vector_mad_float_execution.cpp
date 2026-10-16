// Non-fused MAD over real emitted SPIR-V: independent per-lane operands and oldD,
// the multiply-round/add-round witness, and the AMD70648 source/result modifier contract.
#include "gpu/recompiler/ir/ops/vector_mad_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
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
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edge[] = {0,          0x3f800000, 0xbf800000, 0x3f000000,
                             0xc0200000, 0x3f800001, 0x40000000, 0x47800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 8 ? edge[lane]
                               : (in.s0[lane] & 0x807fffffu) |
                                     ((110u + static_cast<uint32_t>(lane) % 25) << 23);
        in.s1[lane] = lane < 8 ? edge[(lane * 3 + 1) % 8]
                               : (in.s1[lane] & 0x807fffffu) |
                                     ((112u + static_cast<uint32_t>(lane) % 21) << 23);
        seed = seed * 1664525u + 1013904223u;
        in.s2.push_back(lane < 8 ? edge[(lane * 5 + 3) % 8]
                                 : (seed & 0x807fffffu) |
                                       ((108u + static_cast<uint32_t>(lane) % 27) << 23));
    }
    return in;
}
std::vector<uint32_t> words(uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 4 ? 242 : mode == 5 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 4 ? 244 : mode == 6 ? 255 : 256 + kVsrc1;
    const uint32_t c = mode == 3 || mode == 4 ? 241 : mode == 7 ? 255 : 256 + kVsrc2;
    std::vector<uint32_t> code = {0xd5410000u | kVdst | (abs << 8) | (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (c << 18) | (omod << 27) | (neg << 29)};
    if (mode >= 5) code.push_back(mode == 5 ? 0x3fc00000 : mode == 6 ? 0x3f7ffffe : 0xbf800000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0) {
    return {mode == 1 || mode == 4 ? 0x3f800000u
            : mode == 5            ? 0x3fc00000u
                                   : in.s0[lane],
            mode == 2 || mode == 4 ? 0x40000000u
            : mode == 6            ? 0x3f7ffffeu
                                   : in.s1[lane],
            mode == 3 || mode == 4 ? 0xbf000000u
            : mode == 7            ? 0xbf800000u
                                   : in.s2[lane]};
}
void compare(const MadFloatDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_mad_float_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown MAD lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorMadFloatExecution, EveryDefinitionMatchesDifferingLanesInBothWaves) {
    for (const auto& def : vector_mad_float_defs()) {
        const auto code = words();
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, inputs(seed * 5257), wave);
    }
}
TEST(IrVectorMadFloatExecution, HarnessDistinguishesTheThreeInputsAndInactiveDestinations) {
    auto in = inputs(113);
    const auto run = run_vector_op_per_lane(words(), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        EXPECT_NE(in.s2[lane], in.old_d[lane]);
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
    in.s2.pop_back();
    const auto bad = run_vector_op_per_lane(words(), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorMadFloatExecution, HandWorkedSeparateRoundingAndDyadicExamples) {
    struct Example {
        uint32_t a, b, c, want;
    };
    // (1+2^-23)*(1-2^-23) rounds to1 before subtracting1: MAD gives0,
    // whereas fused arithmetic would retain -2^-46 (a8800000). Other examples are exact dyadics.
    for (auto x : {Example{0x3fc00000, 0x40000000, 0x3f000000, 0x40600000},
                   Example{0x3fc00000, 0x40000000, 0xbf000000, 0x40200000},
                   Example{0xbf800000, 0x40000000, 0x40400000, 0x3f800000},
                   Example{0, 0x47800000, 0x3e800000, 0x3e800000},
                   Example{0x80000000, 0x40000000, 0x80000000, 0x80000000},
                   Example{0x3f800001, 0x3f7ffffe, 0xbf800000, 0},
                   Example{0x40000000, 0x3f000000, 0xbf800000, 0},
                   Example{0x47800000, 0x40000000, 0x3f800000, 0x48000040}}) {
        const auto* def = find_vector_mad_float_op(Rdna2Format::VOP3, 0x141);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_mad_float_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = inputs(19);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        std::fill(in.s2.begin(), in.s2.end(), x.c);
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(words(), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
        }
    }
}
TEST(IrVectorMadFloatExecution, InlineFloatAndLiteralEachSourceKeepTheirWordBits) {
    for (const auto& def : vector_mad_float_defs())
        for (uint32_t mode = 1; mode <= 7; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, mode);
}
TEST(IrVectorMadFloatExecution, EverySourceAndResultModifierCombinationPreservesTheCallerContract) {
    const auto* def = find_vector_mad_float_op(Rdna2Format::VOP3, 0x141);
    ASSERT_NE(def, nullptr);
    auto in = inputs(71);
    std::fill(in.s0.begin(), in.s0.end(), 0xbfc00000);
    std::fill(in.s1.begin(), in.s1.end(), 0xc0000000);
    std::fill(in.s2.begin(), in.s2.end(), 0xbf000000);
    for (uint32_t abs = 0; abs < 8; ++abs)
        for (uint32_t neg = 0; neg < 8; ++neg) {
            std::array<Known, 3> args;
            const uint32_t raw[] = {0xbfc00000, 0xc0000000, 0xbf000000};
            for (uint32_t k = 0; k < 3; ++k) {
                uint32_t v = raw[k];
                if (abs & (1u << k)) v &= 0x7fffffffu;
                if (neg & (1u << k)) v ^= 0x80000000u;
                args[k] = v;
            }
            const auto result = eval_mad_float_op(*def, args);
            ASSERT_TRUE(result.has_value());
            if (!result) continue;
            for (uint32_t omod = 0; omod < 4; ++omod)
                for (bool clamp : {false, true}) {
                    float value = std::bit_cast<float>(*result);
                    if (omod) value *= omod == 1 ? 2.0f : omod == 2 ? 4.0f : 0.5f;
                    if (clamp) value = std::clamp(value, 0.0f, 1.0f);
                    const uint32_t want = std::bit_cast<uint32_t>(value);
                    for (uint32_t wave : {64u, 32u}) {
                        const auto run =
                            run_vector_op_per_lane(words(0, abs, neg, omod, clamp), in, wave);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        for (size_t lane = 0; lane < in.lanes(); ++lane)
                            EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                    }
                }
        }
}
TEST(IrVectorMadFloatExecution, HandWorkedAbsThenNegAndOmodThenClampOrder) {
    struct Example {
        uint32_t abs, neg, omod;
        bool clamp;
        uint32_t want;
    };
    auto in = inputs(83);
    std::fill(in.s0.begin(), in.s0.end(), 0xbfc00000);
    std::fill(in.s1.begin(), in.s1.end(), 0xc0000000);
    std::fill(in.s2.begin(), in.s2.end(), 0xbf000000);
    for (auto x : {Example{1, 0, 0, false, 0xc0600000}, Example{1, 1, 0, false, 0x40200000},
                   Example{0, 4, 0, false, 0x40600000}, Example{7, 0, 0, false, 0x40600000},
                   Example{7, 7, 0, false, 0x40200000}, Example{0, 0, 1, false, 0x40a00000},
                   Example{0, 0, 2, false, 0x41200000}, Example{0, 0, 3, false, 0x3fa00000},
                   Example{0, 0, 3, true, 0x3f800000}, Example{1, 0, 0, true, 0},
                   Example{0, 1, 0, false, 0xc0600000}, Example{0, 0, 0, true, 0x3f800000}}) {
        for (uint32_t wave : {64u, 32u}) {
            const auto run =
                run_vector_op_per_lane(words(0, x.abs, x.neg, x.omod, x.clamp), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
        }
    }
}
TEST(IrVectorMadFloatExecution, DefinitionFactsRequiredSourcesAndNonfusedScopeAreExplicit) {
    std::set<Opcode> seen;
    for (const auto& def : vector_mad_float_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> slots;
        def.build(ir, TernaryInputs(ir, slots));
        EXPECT_EQ((std::set<uint32_t>(slots.begin(), slots.end())), (std::set<uint32_t>{0, 1, 2}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
        for (uint32_t missing = 0; missing < 3; ++missing) {
            std::array<Known, 3> args = {0x3f800000u, 0x40000000u, 0x40400000u};
            args[missing] = std::nullopt;
            EXPECT_FALSE(eval_mad_float_op(def, args));
        }
        EXPECT_FALSE(eval_mad_float_op(def, {0x7fc00001u, 0x3f800000u, 0x40000000u}));
        EXPECT_EQ(eval_mad_float_op(def, {0x3f800000u, 0x40000000u, 0x40400000u}),
                  Known(0x40a00000));
    }
    EXPECT_TRUE(seen.contains(Opcode::F32Mul));
    EXPECT_TRUE(seen.contains(Opcode::F32Add));
    EXPECT_EQ(find_vector_mad_float_op(Rdna2Format::VOP3, 0x14b), nullptr);
    EXPECT_EQ(find_vector_mad_float_op(Rdna2Format::VOP3, 0x34b), nullptr);
    EXPECT_EQ(find_vector_mad_float_op(Rdna2Format::VOP2, 0x141), nullptr);
}
