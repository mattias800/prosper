// The packed integer helper owns both destination halves and its original predicate call.
#include "gpu/recompiler/ir/lower/vop3p_packed_integer_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x0a) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3P;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.n_src = 3;
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    in.vop3p_opsel_hi = 7;
    return in;
}
struct Packed : LoweringFixture {
    Packed() {
        rs.vreg[1] = b.uconst(0xd00df00d);
        rs.vreg[2] = b.uconst(0xffff0002);
        rs.vreg[3] = b.uconst(0x00030004);
        rs.vreg[5] = b.uconst(0x00050006);
    }
    bool lower(const Rdna2Inst& in) { return ir::lower_vop3p_packed_integer(b, rs, in, ok, false); }
};
size_t count_opcode(const std::vector<uint32_t>& words, uint32_t opcode) {
    size_t n = 0;
    for (size_t at = 0; at < words.size();) {
        const auto width = words[at] >> 16;
        if (!width) break;
        if ((words[at] & 0xffff) == opcode) ++n;
        at += width;
    }
    return n;
}
}   // namespace
TEST(IrLowerVop3pPackedInteger, EveryExistingIntegerOperationPublishesOneFullWord) {
    for (uint32_t op = 0; op <= 0x0d; ++op) {
        Packed t;
        const auto before = t.rs.vreg.at(1);
        const auto& effect = rdna2_instruction_effects(Rdna2Format::VOP3P, op);
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effect.vdst_dwords, 1u);
        EXPECT_FALSE(effect.vdst_half);
        ASSERT_TRUE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), before);
        EXPECT_TRUE(t.constant_cached(0xffff));
        EXPECT_TRUE(t.constant_cached(16));
    }
}
TEST(IrLowerVop3pPackedInteger, HelperKeepsItsOriginalNarrowedExecPredicateCall) {
    Packed full, narrow;
    narrow.rs.exec = narrow.b.bfalse();
    narrow.rs.exec_narrowed = true;
    const auto old = narrow.rs.vreg.at(1), exec = narrow.rs.exec;
    const auto vcc = narrow.rs.vcc, scc = narrow.rs.scc;
    ASSERT_TRUE(full.lower(instruction()));
    ASSERT_TRUE(narrow.lower(instruction()));
    EXPECT_TRUE(full.ok);
    EXPECT_TRUE(narrow.ok);
    EXPECT_EQ(count_opcode(narrow.b.code, Op_Select), count_opcode(full.b.code, Op_Select) + 1);
    EXPECT_EQ(narrow.rs.exec, exec);
    EXPECT_EQ(narrow.rs.vcc, vcc);
    EXPECT_EQ(narrow.rs.scc, scc);
    bool found = false;
    for (size_t at = 0; at < narrow.b.code.size();) {
        const auto width = narrow.b.code[at] >> 16;
        ASSERT_GT(width, 0u);
        if ((narrow.b.code[at] & 0xffff) == Op_Select && width == 6) {
            const auto* p = &narrow.b.code[at];
            if (p[2] == narrow.rs.vreg.at(1)) {
                found = true;
                EXPECT_EQ(p[3], exec);
                EXPECT_EQ(p[5], old);
            }
        }
        at += width;
    }
    EXPECT_TRUE(found);
}
TEST(IrLowerVop3pPackedInteger,
     OnlyRealSourcesAreReadAndTheirFailureStillPublishesTheOriginalStage) {
    for (uint32_t op = 0; op <= 0x0d; ++op) {
        Packed t;
        auto in = instruction(op);
        in.src[2] = {OperandKind::None, 0};
        ASSERT_TRUE(t.lower(in));
        EXPECT_EQ(t.ok, rdna2_valu_source_count(Rdna2Format::VOP3P, op) == 2u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
    Packed t;
    auto in = instruction();
    in.src[0] = {OperandKind::None, 0};
    ASSERT_TRUE(t.lower(in));
    EXPECT_FALSE(t.ok);
    EXPECT_TRUE(t.rs.vreg.contains(1));
}
TEST(IrLowerVop3pPackedInteger, InlineAndNegWrapperPreservationKeepsExistingMaterialization) {
 // CONFIDENCE: LOW on integer inline floats (#4407), MED on guest-derived integer NEG.
    Packed t;
    auto in = instruction();
    in.src[0] = {OperandKind::InlineFloat, 242};
    in.src[1] = {OperandKind::InlineInt, -1};
    in.src_neg[0] = true;
    in.vop3p_neg_hi = 2;
    ASSERT_TRUE(t.lower(in));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.constant_cached(0x3c00));
    EXPECT_TRUE(t.constant_cached(0x8000));
}
TEST(IrLowerVop3pPackedInteger, FloatDotMixAndForeignFormatsRemainUnclaimed) {
    for (auto op : {0x0eu, 0x13u, 0x14u, 0x20u, 0x7fu}) {
        Packed t;
        const auto before = t.code_words();
        const auto dest = t.rs.vreg.at(1);
        EXPECT_FALSE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.code_words(), before);
        EXPECT_EQ(t.rs.vreg.at(1), dest);
    }
    Packed t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3;
    EXPECT_FALSE(t.lower(in));
}
