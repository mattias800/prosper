// Scalar-pair numeric results, unchanged publication and the pre-existing source refusals.
#include "gpu/recompiler/ir/scalar_integer64_lower_test.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;

TEST(IrLowerSop2Numeric64, BfePublishesBothWordsAndNonzeroScc) {
    for (unsigned output = 0; output < 3; ++output) {
        LoweringFixture t;
        t.set_sgpr(50, 0x89abcdef);
        t.set_sgpr(51, 0x01234567);
        t.set_sgpr(54, 0x00400000);
        ASSERT_TRUE(ir::lower_sop2(t.b, t.rs, t.decode({sop2(kSop2OpcodeBfeU64, 60, 50, 54)}), t.ok,
                                   false));
        ASSERT_TRUE(t.ok);
        ASSERT_TRUE(t.rs.sreg.contains(61));
        expect_scalar64_word(t,
                             output == 2 ? t.b.sel(t.rs.scc, t.b.uconst(1), t.b.uconst(0))
                                         : t.rs.sreg[60 + output],
                             output == 0   ? 0x89abcdef
                             : output == 1 ? 0x01234567
                                           : 1);
    }
}
TEST(IrLowerSop2Numeric64, InlineFloatSourceStillRefuses) {
    LoweringFixture t;
    t.set_sgpr(54, 0x100000);
    ASSERT_TRUE(
        ir::lower_sop2(t.b, t.rs, t.decode({sop2(kSop2OpcodeBfeU64, 60, 242, 54)}), t.ok, false));
    EXPECT_FALSE(t.ok);
}
