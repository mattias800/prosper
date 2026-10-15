// VOP2 float lowering keeps source order, the EXEC predicate and the reserved VCC route.
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct FloatVop2 : LoweringFixture {
    FloatVop2() {
        rs.vreg[kVsrc0] = b.uconst(0x3f800000);
        rs.vreg[kVsrc1] = b.uconst(0x40000000);
        rs.vreg[kVdst] = b.uconst(0x40400000);
    }
    bool lower(uint32_t op) { return ir::lower_vop2(b, rs, decode({vop2(op)}), ok, false); }
    size_t count(uint32_t opcode) const {
        size_t n = 0;
        for (size_t i = 0; i < b.code.size(); i += b.code[i] >> 16)
            n += (b.code[i] & 0xffffu) == opcode;
        return n;
    }
};
}   // namespace

TEST(IrLowerVop2Float, FullExecWritesOneAdd) {
    FloatVop2 t;
    ASSERT_TRUE(t.lower(0x03));
    ASSERT_TRUE(t.ok);
    EXPECT_EQ(t.count(Op_FAdd), 1u);
    EXPECT_EQ(t.count(Op_Select), 0u);
    EXPECT_NE(t.rs.vreg[kVdst], t.b.uconst(0x40400000));
}

TEST(IrLowerVop2Float, NarrowedExecKeepsTheOldDestination) {
    FloatVop2 t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    ASSERT_TRUE(t.lower(0x08));
    ASSERT_TRUE(t.ok);
    EXPECT_EQ(t.count(Op_FMul), 1u);
    EXPECT_EQ(t.count(Op_Select), 1u);
}

TEST(IrLowerVop2Float, ReservedCarryStillRequiresVcc) {
    FloatVop2 t;
    t.rs.vcc = 0;
    t.lower(0x28);
    EXPECT_FALSE(t.ok);
}
