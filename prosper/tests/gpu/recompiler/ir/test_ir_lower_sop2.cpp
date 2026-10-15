// ir/lower/sop2.cpp: the SOP2 plain scalar-data path. Values are test_ir_scalar_alu_execution's
// job; this pins what the lowering does with them -- which registers and SCC it writes, the
// descriptor tag it retires, and the shapes it must refuse.
#include "gpu/recompiler/ir/lowering_fixture.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct Sop2 : LoweringFixture {
    bool lower(uint32_t op, uint32_t d = 22, uint32_t s0 = 20, uint32_t s1 = 21) {
        const Rdna2Inst in = decode({sop2(op, d, s0, s1)});
        return ir::lower_sop2(b, rs, in, ok, false);
    }
    Sop2() {
        set_sgpr(20, 5);
        set_sgpr(21, 7);
        rs.scc = b.bfalse();
    }
};
} // namespace

TEST(IrLowerSop2, DataOpWritesDestinationAndScc) {
    Sop2 t;
    t.rs.sreg_srt[22] = 0x40;   // a descriptor tag the new value must retire
    const uint32_t scc_before = t.rs.scc;
    const size_t words = t.code_words();
    ASSERT_TRUE(t.lower(0x00));   // s_add_u32
    EXPECT_TRUE(t.ok);
    ASSERT_TRUE(t.rs.sreg.contains(22));
    EXPECT_NE(t.rs.sreg[22], t.rs.sreg[20]);
    EXPECT_NE(t.rs.scc, scc_before) << "s_add_u32 writes SCC (carry)";
    EXPECT_FALSE(t.rs.sreg_srt.contains(22));
    EXPECT_EQ(t.code_words() - words, 2u * 5u) << "IAdd then ULessThan";
}

TEST(IrLowerSop2, OpsWithoutAnSccWriteKeepIt) {
    for (uint32_t op : {0x26u, 0x24u, 0x35u, 0x36u}) {   // s_mul_i32, s_bfm_b32, s_mul_hi_u32/i32
        Sop2 t;
        const uint32_t scc_before = t.rs.scc;
        ASSERT_TRUE(t.lower(op));
        EXPECT_TRUE(t.ok) << op;
        EXPECT_EQ(t.rs.scc, scc_before) << op;
        EXPECT_TRUE(t.rs.sreg.contains(22)) << op;
    }
}

TEST(IrLowerSop2, SccReadersRefuseAPoisonedScc) {
    for (uint32_t op : {0x04u, 0x05u, 0x0au}) {   // s_addc_u32, s_subb_u32, s_cselect_b32
        Sop2 poisoned;
        poisoned.rs.scc = 0;
        poisoned.lower(op);
        EXPECT_FALSE(poisoned.ok) << op;
        Sop2 known;
        known.lower(op);
        EXPECT_TRUE(known.ok) << op;
    }
}

TEST(IrLowerSop2, UnmodelledOpsAndExecHalvesRefuse) {
    Sop2 bfe_i32;
    bfe_i32.lower(0x28);   // s_bfe_i32 has no definition
    EXPECT_FALSE(bfe_i32.ok);
    Sop2 exec_lo;
    exec_lo.lower(0x00, 126);   // s_add_u32 exec_lo: a scalar data write into the wave mask
    EXPECT_FALSE(exec_lo.ok);
}

TEST(IrLowerSop2, PackRouteComputesThroughTheDefinition) {
    Sop2 t;
    const uint32_t scc_before = t.rs.scc;
    ASSERT_TRUE(t.lower(0x33));   // s_pack_lh_b32_b16
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.sreg.contains(22));
    EXPECT_EQ(t.rs.scc, scc_before);
    EXPECT_TRUE(t.constant_cached(0xffff0000u)) << "S1's high half mask";
}
