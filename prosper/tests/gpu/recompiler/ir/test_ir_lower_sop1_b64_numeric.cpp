#include "gpu/recompiler/ir/scalar_integer64_lower_test.hpp"
using namespace prosper::gpu;
using namespace prosper::test::ir;

TEST(IrLowerSop1B64Numeric, PopcountWritesDataAndRetiresOverlappingMask) {
    LoweringFixture t;
    t.set_sgpr(50, 0xffffffff);
    t.set_sgpr(51, 0xffffffff);
    t.rs.sreg_bool[59] = t.b.btrue();
    t.rs.sreg_bool[60] = t.b.btrue();
    ASSERT_TRUE(
        ir::lower_sop1(t.b, t.rs, t.decode({sop1(kSop1OpcodeBcnt1I32B64, 60, 50)}), t.ok, false));
    ASSERT_TRUE(t.ok);
    EXPECT_FALSE(t.rs.sreg_bool.contains(59));
    EXPECT_FALSE(t.rs.sreg_bool.contains(60));
    expect_scalar64_word(t, t.rs.sreg[60], 64);
}
TEST(IrLowerSop1B64Numeric, MissingHighAndExecDestinationStayRefused) {
    for (bool missing : {true, false}) {
        LoweringFixture t;
        t.set_sgpr(50, 1);
        if (!missing) t.set_sgpr(51, 0);
        ASSERT_TRUE(ir::lower_sop1(t.b, t.rs,
                                   t.decode({sop1(kSop1OpcodeBcnt1I32B64, missing ? 60 : 126, 50)}),
                                   t.ok, false));
        EXPECT_FALSE(t.ok);
    }
}
