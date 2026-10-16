// AMD 70648 12.12 unsigned 24-bit multiply-add: emitted SPIR-V vs shared definitions,
// with three independent per-lane inputs, old D independent of S2, and partial EXEC in both waves.
#include "gpu/recompiler/ir/ops/vector_mad24.hpp"
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
TEST(IrVectorMad24Execution, EveryDefinitionMatchesEachLaneInBothWaves) {
    for (const auto& def : vector_mad24_defs()) {
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
                const auto in = ternary_inputs(seed * 5245);
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
TEST(IrVectorMad24Execution, HarnessSeesAllThreeInputsAndInactiveDestinations) {
    auto in = ternary_inputs(43);
    const auto code = words(0x143);
    const auto run = run_vector_op_per_lane(code, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            active.insert(run.d[lane]);
            EXPECT_EQ(run.d[lane],
                      ((in.s0[lane] & 0xffffffu) * (in.s1[lane] & 0xffffffu) + in.s2[lane]));
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
TEST(IrVectorMad24Execution, HandWorkedIsaTruncationFullAccumulatorAndOverflow) {
    struct Example {
        uint32_t a, b, c, want;
    };
    // (2^24-1)^2 = 2^48-2^25+1, whose low32 bits are0xfe000001.
    // Adding0x02000000 wraps to1. Only multiplicands are truncated; S2 keeps all32 bits.
    for (auto x :
         {Example{0xff000002, 0xaa000003, 5, 11}, Example{0xffffffff, 0xffffffff, 0, 0xfe000001},
          Example{0x00ffffff, 0x00ffffff, 0x02000000, 1}, Example{0x00800000, 2, 0, 0x01000000},
          Example{0xffffff, 0x100, 0x12345678, 0x12345578},
          Example{0x01000000, 0xffffffff, 0x89abcdef, 0x89abcdef},
          Example{1, 1, 0xff000001, 0xff000002}, Example{2, 3, 0xffffffff, 5},
          Example{0, 0, 0xffffffff, 0xffffffff},
          Example{0x800000, 0x800000, 0x12345678, 0x12345678}}) {
        const auto* def = find_vector_mad24_op(Rdna2Format::VOP3, 0x143);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = ternary_inputs(13);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        std::fill(in.s2.begin(), in.s2.end(), x.c);
        const auto run = run_vector_op_per_lane(words(0x143), in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorMad24Execution, ExistingClampRefusalRemains) {
    const auto in = ternary_inputs(151);
    for (const auto& def : vector_mad24_defs()) {
        const auto good = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def.opcode);
        code[0] |= 1u << 15;
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
}
TEST(IrVectorMad24Execution, EffectsAndSourceSlotsAgreeWithTheDefinitionDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_mad24_defs()) {
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
    for (auto op : {Opcode::And, Opcode::IMul, Opcode::IAdd}) EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_mad24_op(Rdna2Format::VOP2, 0x143), nullptr);
    EXPECT_EQ(find_vector_mad24_op(Rdna2Format::VOP3, 0x142), nullptr);
}
TEST(IrVectorMad24Execution, UnknownSourceWordsRemainUnknown) {
    for (const auto& def : vector_mad24_defs())
        for (unsigned source = 0; source < 3; ++source) {
            std::array<Known, 3> in{1, 2, 3};
            in[source] = std::nullopt;
            EXPECT_FALSE(eval_ternary_integer_op(def, in));
        }
}
