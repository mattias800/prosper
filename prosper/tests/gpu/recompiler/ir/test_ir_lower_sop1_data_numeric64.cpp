#include "gpu/recompiler/ir/scalar_integer64_lower_test.hpp"
using namespace prosper::gpu;
using namespace prosper::test::ir;

TEST(IrLowerSop1DataNumeric64, FlbitHighWordWinsAndSccIsUnchanged) {
    LoweringFixture t;
    t.set_sgpr(50, 0xffffffff);
    t.set_sgpr(51, 1);
    t.rs.scc = t.b.bfalse();
    const auto before = t.rs.scc;
    ASSERT_TRUE(ir::lower_sop1(t.b, t.rs, t.decode({sop1(0x16, 60, 50)}), t.ok, false));
    ASSERT_TRUE(t.ok);
    EXPECT_EQ(t.rs.scc, before);
    expect_scalar64_word(t, t.rs.sreg[60], 31);
}
TEST(IrLowerSop1DataNumeric64, BitreplicatePublishesBothWordsWithoutSccWrite) {
    for (unsigned output = 0; output < 2; ++output) {
        LoweringFixture t;
        t.set_sgpr(50, 0x80000001);
        t.rs.scc = t.b.btrue();
        const auto before = t.rs.scc;
        ASSERT_TRUE(ir::lower_sop1(
            t.b, t.rs, t.decode({sop1(kSop1OpcodeBitreplicateB64B32, 60, 50)}), t.ok, false));
        ASSERT_TRUE(t.ok);
        EXPECT_EQ(t.rs.scc, before);
        EXPECT_TRUE(t.rs.sreg_written.contains(61));
        expect_scalar64_word(t, t.rs.sreg[60 + output], output ? 0xc0000000 : 3);
    }
}
TEST(IrLowerSop1DataNumeric64, FlbitUnownedHighWordStillRefuses) {
    LoweringFixture t;
    t.set_sgpr(50, 1);
    ASSERT_TRUE(ir::lower_sop1(t.b, t.rs, t.decode({sop1(0x16, 60, 50)}), t.ok, false));
    EXPECT_FALSE(t.ok);
}
