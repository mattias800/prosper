// Newly admitted e64 rounding must read only S0, use the caller's numeric result,
// preserve mask state, and decline foreign/reserved shapes before publication.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include <cstddef>
#include "gpu/recompiler/ir/lower/vop3_round_float_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_round_float.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op = 0x1a0) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = op;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    return in;
}
struct Round : LoweringFixture {
    std::vector<int> reads;
    std::vector<uint32_t> modified;
    ir::BinaryFloatSource source = [&](int slot) {
        reads.push_back(slot);
        return b.uconst(0xbfa00000);
    };
    ir::BinaryFloatResult result = [&](uint32_t value) {
        modified.push_back(value);
        return value;
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_round_float(b, rs, in, ok, source, result);
    }
};
}   // namespace
TEST(IrLowerVop3RoundFloat, MatchesExactlyTheFiveDefinedUnaryForms) {
    size_t count = 0;
    for (uint32_t op = 0; op < 1024; ++op) {
        Round t;
        const auto begin = t.code_words();
        if (!t.lower(instruction(op))) {
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(t.code_words(), begin);
            EXPECT_TRUE(t.reads.empty());
            EXPECT_TRUE(t.modified.empty());
            EXPECT_TRUE(t.rs.vreg.empty());
            continue;
        }
        ++count;
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0}));
        ASSERT_EQ(t.modified.size(), 1u);
        EXPECT_EQ(t.rs.vreg.at(1), t.modified[0]);
        const auto& effects = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half);
        EXPECT_FALSE(effects.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 1u);
    }
    EXPECT_EQ(count, 5u);
}
TEST(IrLowerVop3RoundFloat, EveryUnusedModifierRefusesBeforeSourceOrDestinationEffects) {
    for (const auto& def : ir::vector_round_float_defs())
        for (uint32_t flag = 0; flag < 13; ++flag) {
            Round t;
            auto in = instruction(def.opcode);
            if (flag < 4)
                in.words[0] |= 1u << (11 + flag);
            else if (flag < 6)
                in.src_abs[flag - 3] = true;
            else if (flag < 8)
                in.src_neg[flag - 5] = true;
            else if (flag == 8)
                in.has_dpp = true;
            else if (flag == 9)
                in.has_sdwa = true;
            else if (flag == 10)
                in.words[1] |= 50u << 9;
            else if (flag == 11)
                in.words[1] |= 510u << 18;
            else
                in.words[1] |= 255u << 9;
            EXPECT_TRUE(t.lower(in));
            EXPECT_FALSE(t.ok);
            EXPECT_TRUE(t.reads.empty());
            EXPECT_TRUE(t.modified.empty());
            EXPECT_TRUE(t.rs.vreg.empty());
        }
}
TEST(IrLowerVop3RoundFloat, SourceFailureDoesNotPublishAnInventedResult) {
    Round t;
    t.source = [&](int slot) {
        t.reads.push_back(slot);
        t.ok = false;
        return t.b.uconst(0);
    };
    EXPECT_TRUE(t.lower(instruction()));
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(t.reads, (std::vector<int>{0}));
    EXPECT_TRUE(t.modified.empty());
    EXPECT_TRUE(t.rs.vreg.empty());
}
TEST(IrLowerVop3RoundFloat, MissingInputModeKeepsTheOldDestinationAndSkipsOutputModifiers) {
    for (const auto& def : ir::vector_round_float_defs()) {
        if (!rdna2_instruction_effects(def.fmt, def.opcode).f32_rounding_input_sensitive) continue;
        Round t;
        t.b.compute_pgm_rsrc1 = UINT32_MAX;
        const auto old = t.b.uconst(0xd00d);
        t.rs.vreg[1] = old;
        t.source = [&](int slot) {
            t.reads.push_back(slot);
            return t.b.id();
        };
        const auto before = t.code_words();
        EXPECT_TRUE(t.lower(instruction(def.opcode)));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0}));
        EXPECT_TRUE(t.modified.empty());
        EXPECT_EQ(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.code_words(), before);
        EXPECT_EQ(t.b.stage_reject_reason, "f32-round-input-mode-unavailable");
    }
}
TEST(IrLowerVop3RoundFloat, ResultAndMaskBookkeepingRemainWithTheCaller) {
    Round t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto marker = t.b.uconst(0x12345678);
    t.result = [&](uint32_t) { return marker; };
    auto in = instruction();
    in.src_abs[0] = in.src_neg[0] = true;
    in.omod = 3;
    in.clamp = true;
    ASSERT_TRUE(t.lower(in));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vreg.at(1), marker);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
}
TEST(IrLowerVop3RoundFloat, ForeignEncodingsAndUnmodelledUnaryFamiliesRemainUnmatched) {
    for (auto op : {0x101u, 0x103u, 0x1a5u, 0x1abu, 0x1afu, 0x270u}) {
        Round t;
        EXPECT_FALSE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
    }
    Round t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP1;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
}
TEST(IrLowerVop3RoundFloat, MissingNumericServicesAbortOnlyForDefinedForms) {
    Round t;
    EXPECT_FALSE(ir::lower_vop3_round_float(t.b, t.rs, instruction(0x101), t.ok, {}, {}));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3_round_float(t.b, t.rs, instruction(), t.ok, {}, t.result);
        },
        "rounding float callbacks missing");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3_round_float(t.b, t.rs, instruction(), t.ok, t.source, {});
        },
        "rounding float callbacks missing");
}

TEST(IrLowerVop3RoundFloat, DecoderConsumesOnlyTheEffectsDeclaredUnarySource) {
    size_t count = 0;
    for (uint32_t op = 0x100; op < 1024; ++op) {
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        if (!e.describes_vector_result() || e.lane_flow != Rdna2LaneFlow::LaneWise ||
            e.vdst_dwords != 1 || e.vdst_half || e.accumulates ||
            e.m0_relative != Rdna2M0Relative::None || e.implicit_reads || e.implicit_writes ||
            rdna2_valu_source_count(Rdna2Format::VOP3, op) != 1)
            continue;
        // Deliberately nonzero fields prove this is operand inventory, not just canonical s0.
        const uint32_t raw[] = {0xd4000001u | (op << 16), 258u | (50u << 9) | (510u << 18)};
        const auto in = rdna2_decode_one(raw, 2);
        EXPECT_EQ(in.n_src, 1u) << e.name;
        EXPECT_EQ(in.src[0].kind, OperandKind::VGPR);
        EXPECT_EQ(in.src[0].value, 2);
        EXPECT_EQ(in.src[1].kind, OperandKind::None);
        EXPECT_EQ(in.src[2].kind, OperandKind::None);
        ++count;
    }
    EXPECT_EQ(count, 38u);
    // Reserved interpolation/mask/cross-lane families and an unknown form retain their inventory.
    for (uint32_t op : {0x101u, 0x182u, 0x270u, 0x3ffu}) {
        const uint32_t raw[] = {0xd4000001u | (op << 16), 258u};
        const auto in = rdna2_decode_one(raw, 2);
        EXPECT_EQ(in.n_src, 3u) << op;
    }
}
