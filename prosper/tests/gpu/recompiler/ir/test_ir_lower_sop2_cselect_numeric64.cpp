#include "gpu/recompiler/ir/scalar_integer64_lower_test.hpp"
using namespace prosper::gpu;
using namespace prosper::test::ir;

TEST(IrLowerSop2CselectNumeric64, BothWordsFollowSccWithoutChangingIt) {
    for (bool condition : {false, true})
        for (unsigned output = 0; output < 2; ++output) {
            LoweringFixture t;
            t.set_sgpr(50, 1);
            t.set_sgpr(51, 2);
            t.set_sgpr(54, 3);
            t.set_sgpr(55, 4);
            t.rs.scc = condition ? t.b.btrue() : t.b.bfalse();
            const auto before = t.rs.scc;
            ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, t.decode({sop2(0x0b, 60, 50, 54)}), t.ok, false));
            ASSERT_TRUE(t.ok);
            EXPECT_EQ(t.rs.scc, before);
            expect_scalar64_word(t, t.rs.sreg[60 + output], (condition ? 1 : 3) + output);
        }
}
TEST(IrLowerSop2CselectNumeric64, PoisonedSccOrMissingHighStillRefuses) {
    for (bool missing : {false, true}) {
        LoweringFixture t;
        t.set_sgpr(50, 1);
        t.set_sgpr(51, 2);
        t.set_sgpr(54, 3);
        if (!missing) t.set_sgpr(55, 4);
        t.rs.scc = missing ? t.b.btrue() : 0;
        ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, t.decode({sop2(0x0b, 60, 50, 54)}), t.ok, false));
        EXPECT_FALSE(t.ok);
    }
}
