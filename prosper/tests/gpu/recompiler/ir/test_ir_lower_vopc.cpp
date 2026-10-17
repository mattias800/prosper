// ir/lower/vopc*.cpp: a compare's value comes from its definition (or the remaining switch), and
// write_vopc_mask then sends EXEC & value to VCC, an SGPR pair or (V_CMPX) EXEC, recording the
// exec_subset_of fact the loop-EXEC proof reads (#5039).
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {

uint32_t vopc(uint32_t op, uint32_t src0 = 256 + kVsrc0) {
    return 0x7c000000u | (op << 17) | (kVsrc1 << 9) | src0;
}

struct Vopc : LoweringFixture {
    bool allow_exec_update = true;
    bool lower(const std::vector<uint32_t>& words) {
        return ir::lower_vopc(b, rs, decode(words), ok, false, allow_exec_update);
    }
    Vopc() {
        rs.vreg[kVsrc0] = b.uconst(0x3f800000u);
        rs.vreg[kVsrc1] = b.uconst(0x40000000u);
        rs.vcc = b.bfalse();
    }
};

} // namespace

TEST(IrLowerVopc, CompareUnderFullExecWritesTheValueToVcc) {
    Vopc t;
    const uint32_t exec = t.rs.exec;
    ASSERT_TRUE(t.lower({vopc(0x01)}));   // v_cmp_lt_f32 vcc, v2, v3
    EXPECT_TRUE(t.ok);
    EXPECT_NE(t.rs.vcc, t.b.bfalse());
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_TRUE(t.rs.sreg_bool_narrowed[106]);
    EXPECT_TRUE(t.rs.sreg_bool_narrowed[107]) << "Wave64 replaces both VCC words";
    EXPECT_EQ(t.rs.vcc_wave_uniform, 0u) << "VGPR sources are not wave-uniform";
    EXPECT_FALSE(t.rs.exec_subset_of.contains(t.rs.vcc)) << "full EXEC: no select, no fact";
}

TEST(IrLowerVopc, NarrowedExecMasksTheValueAndRecordsTheSubset) {
    Vopc t;
    t.rs.exec = t.b.fcmp(Op_FOrdLessThan, t.b.uconst(1), t.b.uconst(2));   // some runtime Bool
    t.rs.exec_narrowed = true;
    ASSERT_TRUE(t.lower({vopc(0x09)}));   // v_cmp_nge_f32
    EXPECT_TRUE(t.ok);
    ASSERT_TRUE(t.rs.exec_subset_of.contains(t.rs.vcc));
    EXPECT_EQ(t.rs.exec_subset_of.at(t.rs.vcc), t.rs.exec) << "VCC = EXEC & test is within EXEC";
}

TEST(IrLowerVopc, Vop3SgprDestinationLeavesVccAlone) {
    Vopc t;
    const uint32_t vcc = t.rs.vcc;
    // v_cmp_gt_f32_e64 s[20:21], v2, v3
    ASSERT_TRUE(t.lower({0xd4040014u, (256 + kVsrc0) | ((256 + kVsrc1) << 9)}));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vcc, vcc);
    ASSERT_TRUE(t.rs.sreg_bool.contains(20));
    EXPECT_TRUE(t.rs.sreg_bool_narrowed[20]);
}

TEST(IrLowerVopc, CmpxNarrowsExecOnlyAndRecordsTheSubset) {
    Vopc t;
    const uint32_t vcc = t.rs.vcc, exec = t.rs.exec;
    ASSERT_TRUE(t.lower({vopc(0x13)}));   // v_cmpx_le_f32
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vcc, vcc) << "gfx10 V_CMPX writes EXEC only (#464)";
    EXPECT_NE(t.rs.exec, exec);
    EXPECT_TRUE(t.rs.exec_narrowed);
    ASSERT_TRUE(t.rs.exec_subset_of.contains(t.rs.exec));
    EXPECT_EQ(t.rs.exec_subset_of.at(t.rs.exec), exec);
}

TEST(IrLowerVopc, CmpxIsRefusedWhereExecMayNotChange) {
    Vopc t;
    t.allow_exec_update = false;
    const uint32_t exec = t.rs.exec;
    ASSERT_TRUE(t.lower({vopc(0x11)}));   // v_cmpx_lt_f32
    EXPECT_FALSE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
}

TEST(IrLowerVopc, ScalarSourcesPublishVccWaveUniformity) {
    Vopc t;
    t.set_sgpr(4, 0x3f800000u);
    t.rs.vreg[kVsrc1] = t.b.uconst(0x40000000u);
    // v_cmp_lt_f32 vcc, s4, v3: v3 is a VGPR, so not uniform; then s4 against an inline constant.
    ASSERT_TRUE(t.lower({vopc(0x01, 4)}));
    EXPECT_EQ(t.rs.vcc_wave_uniform, 0u);
    // v_cmp_lt_f32_e64 vcc, s4, 1.0
    ASSERT_TRUE(t.lower({0xd401006au, 4u | (0xf2u << 9)}));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.vcc_wave_uniform, t.rs.vcc);
}

TEST(IrLowerVopc, OpcodesWithoutAValueAreRefused) {
    Vopc undefined;
    ASSERT_TRUE(undefined.lower({vopc(0x80)}));   // v_cmp_f_i32: no definition and no case
    EXPECT_FALSE(undefined.ok);
    Vopc integer;
    ASSERT_TRUE(integer.lower({vopc(0xc1)}));   // v_cmp_lt_u32 still takes the switch
    EXPECT_TRUE(integer.ok);
}
