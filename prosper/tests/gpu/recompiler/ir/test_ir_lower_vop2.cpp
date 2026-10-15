// ir/lower/vop2.cpp: the operation stage takes the IR definition when one exists and the old switch
// otherwise; both then go through the same predicated VDST write.
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct Vop2 : LoweringFixture {
    bool lower(uint32_t op) { return ir::lower_vop2(b, rs, decode({vop2(op)}), ok, false); }
    Vop2() {
        rs.vreg[kVsrc0] = b.uconst(5);
        rs.vreg[kVsrc1] = b.uconst(7);
        rs.vreg[kVdst] = b.uconst(9);
    }
};
} // namespace

TEST(IrLowerVop2, DefinedOpWritesVdstOnce) {
    Vop2 t;
    const size_t words = t.code_words();
    ASSERT_TRUE(t.lower(0x25));   // v_add_nc_u32, full EXEC: one IAdd, no select
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.code_words() - words, 5u);
    EXPECT_NE(t.rs.vreg[kVdst], t.b.uconst(9));
}

TEST(IrLowerVop2, NarrowedExecKeepsTheOldVdst) {
    Vop2 t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    const size_t words = t.code_words();
    ASSERT_TRUE(t.lower(0x1b));   // v_and_b32
    EXPECT_TRUE(t.ok);
    EXPECT_GT(t.code_words() - words, 5u) << "BitwiseAnd, then the EXEC-predicated write";
}

TEST(IrLowerVop2, UndefinedOpsStillTakeTheSwitch) {
    Vop2 f32;
    ASSERT_TRUE(f32.lower(0x03));   // v_add_f32 now uses its IR definition
    EXPECT_TRUE(f32.ok);
    Vop2 carry;
    carry.rs.vcc = 0;
    carry.lower(0x28);   // v_add_co_ci_u32 without a VCC carry-in refuses
    EXPECT_FALSE(carry.ok);
}
