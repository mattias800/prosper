// Preserve legacy multiply source order, DX9 zero selection and root-owned modifiers.
// A lost zero selector or misplaced modifier can corrupt the published lane value.
#include "gpu/recompiler/ir/lower/vop3_legacy_mul_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x107) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    return in;
}
struct FloatLegacy : LoweringFixture {
    std::vector<int> reads;
    std::array<size_t, 3> read_at{};
    std::vector<uint32_t> modified;
    size_t modify_at = 0;
    std::array<uint32_t, 3> bits = {0x3f800000, 0x40000000, 0x40400000};
    int bad = -1;
    ir::LegacyMulSource source = [this](int slot) {
        reads.push_back(slot);
        read_at.at(static_cast<size_t>(slot)) = code_words();
        if (slot == bad) {
            ok = false;
            return b.uconst(0);
        }
        return b.uconst(bits.at(static_cast<size_t>(slot)));
    };
    ir::LegacyMulResult result = [this](uint32_t value) {
        modified.push_back(value);
        modify_at = code_words();
        return value;
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_legacy_mul(b, rs, in, ok, false, source, result);
    }
    size_t count(uint32_t opcode) const {
        size_t out = 0;
        for (size_t pc = 0; pc < b.code.size(); pc += b.code[pc] >> 16)
            out += (b.code[pc] & 0xffffu) == opcode;
        return out;
    }
};
}   // namespace
TEST(IrLowerVop3LegacyMul, TwoSequentialSourcesAndTheDx9SelectorKeepTheOldStageBoundaries) {
    FloatLegacy t;
    const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, 0x107);
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_FALSE(e.vdst_half);
    EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, 0x107), 2u);
    ASSERT_TRUE(t.lower(instruction()));
    ASSERT_TRUE(t.ok);
    EXPECT_EQ(t.reads.size(), 2u);
    EXPECT_EQ((std::set<int>(t.reads.begin(), t.reads.end())), (std::set<int>{0, 1}));
    EXPECT_EQ(t.reads, (std::vector<int>{0, 1}));
    EXPECT_GE(t.modify_at, t.read_at[1]);
    EXPECT_EQ(t.count(Op_FOrdEqual), 2u);
    EXPECT_EQ(t.count(Op_FMul), 1u);
    EXPECT_EQ(t.count(Op_Select), 1u);
    ASSERT_EQ(t.modified.size(), 1u);
    EXPECT_EQ(t.rs.vreg.at(1), t.modified[0]);
}
TEST(IrLowerVop3LegacyMul, ResultCallbackReturnValueIsThePublishedWord) {
    FloatLegacy t;
    const uint32_t marker = t.b.uconst(0x76543210);
    size_t calls = 0;
    t.result = [&](uint32_t value) {
        EXPECT_NE(value, 0u);
        ++calls;
        EXPECT_EQ(t.count(Op_Select), 1u);
        return marker;
    };
    ASSERT_TRUE(t.lower(instruction()));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(t.rs.vreg.at(1), marker);
}
TEST(IrLowerVop3LegacyMul, SourceErrorsStillRunTheOriginalValueAndModifierStages) {
    for (int bad : {0, 1}) {
        FloatLegacy t;
        t.bad = bad;
        ASSERT_TRUE(t.lower(instruction()));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_EQ(t.modified.size(), 1u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.count(Op_FMul), 1u);
        EXPECT_EQ(t.count(Op_Select), 1u);
    }
}
// This helper owns only the DX9 value select. The execution suite checks inactive-lane
// publication through the root predicate; this direct call checks mask-state preservation.
TEST(IrLowerVop3LegacyMul, HelperPreservesMaskStateAndOwnsOnlyTheDx9Select) {
    FloatLegacy t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto selects = t.count(Op_Select);
    auto in = instruction();
    in.src_abs[0] = true;
    in.src_neg[1] = true;
    in.omod = 2;
    in.clamp = true;
    ASSERT_TRUE(t.lower(in));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.reads.size(), 2u);
    EXPECT_EQ(t.modified.size(), 1u);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    EXPECT_EQ(t.count(Op_Select), selects + 1);
}
TEST(IrLowerVop3LegacyMul, FusedFmaCarryAndForeignFormatsRemainUnmatched) {
    for (auto in :
         {instruction(0x14b), instruction(0x108), instruction(0x176), instruction(0x02)}) {
        FloatLegacy t;
        const auto before = t.code_words();
        EXPECT_FALSE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.modified.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    FloatLegacy t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.reads.empty());
    EXPECT_TRUE(t.rs.vreg.empty());
    in.fmt = Rdna2Format::VOP3;
    in.opcode = 0x14b;
    EXPECT_FALSE(ir::lower_vop3_legacy_mul_erased(t.b, t.rs, in, t.ok, false, {}, {}));
}
TEST(IrLowerVop3LegacyMul, MissingCallbacksForAHandledInstructionStopLoudly) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            FloatLegacy t;
            ir::lower_vop3_legacy_mul_erased(t.b, t.rs, instruction(), t.ok, false, {},
                                             std::cref(t.result));
        },
        "legacy multiply callbacks missing");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            FloatLegacy t;
            ir::lower_vop3_legacy_mul_erased(t.b, t.rs, instruction(), t.ok, false,
                                             std::cref(t.source), {});
        },
        "legacy multiply callbacks missing");
}
