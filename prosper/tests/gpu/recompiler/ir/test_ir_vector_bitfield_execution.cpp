// AMD 70648 12.12 VOP3 bitfields: emitted SPIR-V vs shared definitions,
// with three independent per-lane inputs, old D independent of S2, and partial EXEC in both waves.
#include "gpu/recompiler/ir/ops/vector_bitfield.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
LaneInputs ternary_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0u, 1u, 31u, 32u, 33u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1103515245u + 12345u;
        uint32_t value = lane < 16 ? edges[(lane * 3 + 5) % 8] : seed;
        if (value == in.old_d[lane]) value ^= 1;
        in.s2.push_back(value);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode) {
    return {0xd4000000u | (opcode << 16) | kVdst,
            (256u + kVsrc0) | ((256u + kVsrc1) << 9) | ((256u + kVsrc2) << 18)};
}
}   // namespace
TEST(IrVectorBitfieldExecution, EveryDefinitionMatchesEachLaneInBothWaves) {
    for (const auto& def : vector_bitfield_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        const auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = ternary_inputs(seed * 5241);
                const auto run = run_vector_op_per_lane(code, in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want =
                        eval_ternary_integer_op(def, {in.s0[lane], in.s1[lane], in.s2[lane]});
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
TEST(IrVectorBitfieldExecution, HarnessSeesAllThreeInputsAndInactiveDestinations) {
    auto in = ternary_inputs(43);
    const auto code = words(0x14a);
    const auto run = run_vector_op_per_lane(code, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            active.insert(run.d[lane]);
            EXPECT_EQ(run.d[lane], ((in.s0[lane] & in.s1[lane]) | (~in.s0[lane] & in.s2[lane])));
            EXPECT_NE(in.s2[lane], in.old_d[lane]);
        } else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
    in.s2.pop_back();
    const auto bad = run_vector_op_per_lane(code, in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorBitfieldExecution, HandWorkedIsaFieldBoundsSignAndSelectionBits) {
    struct Example {
        uint32_t op, a, b, c, want;
    };
    // 70648 12.12: offset and width use low five bits. Width zero extracts zero.
    // A four-bit signed field 0xb is -5. Extraction stops at the source word's end.
    // BFI(00ff00ff,12345678,abcdef01) combines 00340078 | ab00ef00 = ab34ef78.
    for (auto x :
         {Example{0x148, 0xabcdeff0, 4, 8, 0xff}, Example{0x148, 0xf0f0f0f0, 36, 40, 0x0f},
          Example{0x148, 0x80000000, 31, 31, 1}, Example{0x148, 0xffffffff, 32, 4, 0xf},
          Example{0x148, 0xffffffff, 0, 32, 0}, Example{0x148, 0x12345678, 31, 0, 0},
          Example{0x149, 0x000000b0, 4, 4, 0xfffffffb}, Example{0x149, 0x00000070, 4, 4, 7},
          Example{0x149, 0x80000000, 31, 31, 0xffffffff}, Example{0x149, 0xffffffff, 0, 0, 0},
          Example{0x149, 0x12345678, 32, 8, 0x78},
          Example{0x14a, 0, 0x12345678, 0xabcdef01, 0xabcdef01},
          Example{0x14a, 0xffffffff, 0x12345678, 0xabcdef01, 0x12345678},
          Example{0x14a, 0x00ff00ff, 0x12345678, 0xabcdef01, 0xab34ef78}}) {
        SCOPED_TRACE(::testing::Message() << x.op << " " << x.a << " " << x.b << " " << x.c);
        const auto* def = find_vector_bitfield_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = ternary_inputs(13);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        std::fill(in.s2.begin(), in.s2.end(), x.c);
        const auto run = run_vector_op_per_lane(words(x.op), in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorBitfieldExecution, ExistingClampRefusalRemains) {
    const auto in = ternary_inputs(151);
    for (const auto& def : vector_bitfield_defs()) {
        const auto good = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def.opcode);
        code[0] |= 1u << 15;
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
}
TEST(IrVectorBitfieldExecution, EffectsAndSourceSlotsAgreeWithTheDefinitionDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_bitfield_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_FALSE(e.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> sources;
        def.build(ir, TernaryInputs(ir, sources));
        EXPECT_EQ(sources.size(), block.input_count());
        EXPECT_EQ(std::set<uint32_t>(sources.begin(), sources.end()),
                  (std::set<uint32_t>{0, 1, 2}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (auto op : {Opcode::BfeU, Opcode::BfeS, Opcode::And, Opcode::Or, Opcode::Not})
        EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_bitfield_op(Rdna2Format::VOP2, 0x148), nullptr);
    EXPECT_EQ(find_vector_bitfield_op(Rdna2Format::VOP3, 0x128), nullptr);
}
TEST(IrVectorBitfieldExecution, UnknownSourceWordsRemainUnknown) {
    for (const auto& def : vector_bitfield_defs())
        for (unsigned source = 0; source < 3; ++source) {
            std::array<Known, 3> in{1, 2, 3};
            in[source] = std::nullopt;
            EXPECT_FALSE(eval_ternary_integer_op(def, in));
        }
}
