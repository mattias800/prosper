// Scalar-pair numeric results, unchanged publication and the pre-existing source refusals.
#include "gpu/recompiler/ir/scalar_integer64_lower_test.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/scalar_encoding.hpp"
#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::test::ir;

TEST(IrLowerSopcNumeric64, HighOnlyDifferenceAndComplementWriteOnlyScc) {
    for (uint32_t op : {0x12u, 0x13u}) {
        LoweringFixture t;
        t.set_sgpr(50, 7);
        t.set_sgpr(51, 1);
        t.set_sgpr(54, 7);
        t.set_sgpr(55, 2);
        const auto before = t.rs.sreg;
        ASSERT_TRUE(ir::lower_sopc(t.b, t.rs, t.decode({sopc(op, 50, 54)}), t.ok, false));
        ASSERT_TRUE(t.ok);
        EXPECT_EQ(t.rs.sreg, before);
        expect_scalar64_word(t, t.b.sel(t.rs.scc, t.b.uconst(1), t.b.uconst(0)), op == 0x13);
    }
}
TEST(IrLowerSopcNumeric64, MissingHighAndInlineDoubleStillRefuse) {
    for (uint32_t source : {54u, 242u}) {
        LoweringFixture t;
        t.set_sgpr(50, 0);
        t.set_sgpr(51, 0);
        t.set_sgpr(54, 0);
        ASSERT_TRUE(ir::lower_sopc(t.b, t.rs, t.decode({sopc(0x12, 50, source)}), t.ok, false));
        EXPECT_FALSE(t.ok);
    }
}
