// The relocated packed route must preserve both source halves, the old destination and EXEC.
#include "gpu/recompiler/ir/lower/vop3p_packed_values.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
struct PackedValues : LoweringFixture {
    Rdna2Inst instruction(uint32_t opcode) {
        auto in = decode({0xcc000001u | (opcode << 16), 0x04160702u});
        rs.vreg[1] = b.uconst(0x42003c00);
        rs.vreg[2] = b.uconst(0x3c004000);
        rs.vreg[3] = b.uconst(0x40003c00);
        rs.vreg[5] = b.uconst(0x38003400);
        return in;
    }
    size_t count(uint32_t opcode) const {
        size_t out = 0;
        for (size_t at = 0; at < b.code.size(); at += b.code[at] >> 16)
            out += (b.code[at] & 0xffffu) == opcode;
        return out;
    }
};
} // namespace

TEST(IrLowerVop3pPackedValues, BothResultHalvesArePackedAndPublished) {
    for (uint32_t opcode = 0x0e; opcode <= 0x12; ++opcode) {
        PackedValues t;
        const auto in = t.instruction(opcode);
        const auto old = t.rs.vreg.at(1);
        ASSERT_EQ(ir::lower_vop3p_packed_values(t.b, t.rs, in, t.ok, true), true);
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vreg.at(1), old);
        EXPECT_EQ(t.count(Op_ShiftLeftLogical), 1u);
        EXPECT_GE(t.count(Op_BitwiseOr), 1u);
    }
}

TEST(IrLowerVop3pPackedValues, NarrowedExecSelectsTheReachingDestination) {
    PackedValues t;
    auto in = t.instruction(0x0e);
    t.rs.exec_narrowed = true;
    t.rs.exec = t.b.bfalse();
    const auto old = t.rs.vreg.at(1), exec = t.rs.exec;
    ASSERT_EQ(ir::lower_vop3p_packed_values(t.b, t.rs, in, t.ok, true), true);
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    ASSERT_GE(t.b.code.size(), 6u);
    const auto at = t.b.code.size() - 6;
    ASSERT_EQ(t.b.code[at] & 0xffffu, Op_Select);
    EXPECT_EQ(t.b.code[at + 2], t.rs.vreg.at(1));
    EXPECT_EQ(t.b.code[at + 3], exec);
    EXPECT_EQ(t.b.code[at + 5], old);
}

TEST(IrLowerVop3pPackedValues, UnhandledShapesLeaveRegisterAndCodeStateAlone) {
    PackedValues t;
    const auto in = t.instruction(0x20);
    const auto size = t.code_words();
    const auto registers = t.rs.vreg;
    EXPECT_FALSE(ir::lower_vop3p_packed_values(t.b, t.rs, in, t.ok, true).has_value());
    EXPECT_EQ(t.code_words(), size);
    EXPECT_EQ(t.rs.vreg, registers);
    EXPECT_TRUE(t.ok);
}
