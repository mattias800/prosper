// ir/lower/sopc.cpp: scalar compares write only SCC; the u64 form needs both high words.
#include "gpu/recompiler/ir/lowering_fixture.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct Sopc : LoweringFixture {
    bool lower(uint32_t op, uint32_t s0 = 20, uint32_t s1 = 21) {
        return ir::lower_sopc(b, rs, decode({sopc(op, s0, s1)}), ok, false);
    }
    Sopc() {
        set_sgpr(20, 5);
        set_sgpr(21, 7);
        rs.scc = b.bfalse();
    }
};
} // namespace

TEST(IrLowerSopc, ComparesWriteOnlyScc) {
    for (uint32_t op = 0x00; op <= 0x0d; ++op) {   // s_cmp_*_i32/u32, s_bitcmp0/1_b32
        Sopc t;
        const uint32_t scc_before = t.rs.scc;
        const auto sreg_before = t.rs.sreg;
        ASSERT_TRUE(t.lower(op));
        EXPECT_TRUE(t.ok) << op;
        EXPECT_NE(t.rs.scc, scc_before) << op;
        EXPECT_EQ(t.rs.sreg, sreg_before) << op;
    }
}

TEST(IrLowerSopc, U64CompareNeedsBothHighWords) {
    Sopc missing;
    missing.lower(0x12);   // s_cmp_eq_u64 s[20:21], s[21:22]: s22 has no value
    EXPECT_FALSE(missing.ok);
    Sopc complete;
    complete.set_sgpr(22, 9);
    complete.lower(0x12);
    EXPECT_TRUE(complete.ok);
}

TEST(IrLowerSopc, UnmodelledCompareRefuses) {
    Sopc t;
    t.lower(0x0e);   // s_bitcmp0_b64
    EXPECT_FALSE(t.ok);
}
