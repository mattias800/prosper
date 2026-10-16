// Native half extrema retain selected sources, the opposite destination half and caller state.
#include "gpu/recompiler/ir/lower/vop3_minmax_half_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t op = 0x351, uint32_t sel = 0) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = op;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    in.vop3p_opsel = static_cast<uint8_t>(sel);
    return in;
}
void sources(LoweringFixture& t) {
    t.rs.vreg[2] = t.b.uconst(0xbc003c00);
    t.rs.vreg[3] = t.b.uconst(0x40003800);
    t.rs.vreg[5] = t.b.uconst(0xc0004200);
}
struct Result {
    uint32_t opcode;
    std::vector<uint32_t> operands;
};
std::map<uint32_t, Result> results(const LoweringFixture& t, size_t begin) {
    std::map<uint32_t, Result> out;
    for (size_t pc = begin; pc < t.b.code.size();) {
        size_t n = t.b.code[pc] >> 16;
        if (n < 3 || n > t.b.code.size() - pc) {
            ADD_FAILURE();
            break;
        }
        out[t.b.code[pc + 2]] = {t.b.code[pc] & 0xffffu,
                                 {t.b.code.begin() + pc + 3, t.b.code.begin() + pc + n}};
        pc += n;
    }
    return out;
}
}   // namespace
TEST(IrLowerVop3MinmaxHalf, MatchesExactlyThreeLaneWiseHalfForms) {
    size_t count = 0;
    for (uint32_t op = 0; op < 1024; ++op) {
        LoweringFixture t;
        sources(t);
        auto old = t.b.uconst(0xdeadbeef);
        auto begin = t.code_words();
        auto modify = [](uint32_t x) { return x; };
        if (!ir::lower_vop3_minmax_half(t.b, t.rs, instruction(op), t.ok, false, old, modify)) {
            EXPECT_TRUE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.code_words(), begin);
            continue;
        }
        ++count;
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_TRUE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 3u);
    }
    EXPECT_EQ(count, 3u);
}
TEST(IrLowerVop3MinmaxHalf, BothDestinationHalvesKeepTheOppositeOldBits) {
    for (uint32_t sel = 0; sel < 16; ++sel) {
        LoweringFixture t;
        sources(t);
        auto old = t.b.uconst(0xdeadbeef);
        auto begin = t.code_words();
        size_t calls = 0;
        auto modify = [&](uint32_t x) {
            ++calls;
            return x;
        };
        ASSERT_TRUE(ir::lower_vop3_minmax_half(t.b, t.rs, instruction(0x351, sel), t.ok, false, old,
                                               modify));
        ASSERT_TRUE(t.ok);
        EXPECT_EQ(calls, 1u);
        auto ops = results(t, begin);
        const auto& d = ops.at(t.rs.vreg.at(1));
        EXPECT_EQ(d.opcode, Op_BitwiseOr);
        ASSERT_EQ(d.operands.size(), 2u);
        const auto& keep = ops.at(d.operands[0]);
        EXPECT_EQ(keep.opcode, Op_BitwiseAnd);
        EXPECT_EQ(keep.operands, (std::vector<uint32_t>{
                                     old, t.b.uconst_cache.at((sel & 8) ? 0xffffu : 0xffff0000u)}));
        if (sel & 8) EXPECT_EQ(ops.at(d.operands[1]).opcode, Op_ShiftLeftLogical);
    }
}
TEST(IrLowerVop3MinmaxHalf, WrongFormatAndDeferredFusedFormHaveNoEffects) {
    for (auto fmt : {Rdna2Format::VOP3, Rdna2Format::VOP3P, Rdna2Format::VOP2}) {
        LoweringFixture t;
        sources(t);
        auto in = instruction(fmt == Rdna2Format::VOP3 ? 0x34b : 0x351);
        in.fmt = fmt;
        auto old = t.b.uconst(7);
        auto begin = t.code_words();
        auto modify = [](uint32_t x) { return x; };
        EXPECT_FALSE(ir::lower_vop3_minmax_half(t.b, t.rs, in, t.ok, false, old, modify));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_FALSE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3MinmaxHalf, NonOperandSourceSetsFailure) {
    LoweringFixture t;
    sources(t);
    auto in = instruction();
    in.src[1] = {OperandKind::None, 0};
    auto modify = [](uint32_t x) { return x; };
    auto old = t.b.uconst(7);
    ASSERT_TRUE(ir::lower_vop3_minmax_half(t.b, t.rs, in, t.ok, false, old, modify));
    EXPECT_FALSE(t.ok);
}
TEST(IrLowerVop3MinmaxHalf, ResultCallbackIsAppliedBeforeHalfPacking) {
    LoweringFixture t;
    sources(t);
    auto old = t.b.uconst(0xdeadbeef);
    const auto marker = t.b.uconst(0x3f000000);
    uint32_t received = 0;
    auto modify = [&](uint32_t x) {
        received = x;
        return marker;
    };
    ASSERT_TRUE(ir::lower_vop3_minmax_half(t.b, t.rs, instruction(), t.ok, false, old, modify));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(received, 0u);
    EXPECT_NE(received, marker);
    EXPECT_TRUE(t.rs.vreg.contains(1));
}
TEST(IrLowerVop3MinmaxHalf, MissingCallbackIsAnInternalErrorOnlyForMatchedForms) {
    LoweringFixture t;
    sources(t);
    auto old = t.b.uconst(7);
    ir::MinmaxHalfResult empty;
    EXPECT_FALSE(
        ir::lower_vop3_minmax_half_erased(t.b, t.rs, instruction(0x34b), t.ok, false, old, empty));
    EXPECT_DEATH(
        ir::lower_vop3_minmax_half_erased(t.b, t.rs, instruction(), t.ok, false, old, empty),
        "half extrema callback missing");
}
