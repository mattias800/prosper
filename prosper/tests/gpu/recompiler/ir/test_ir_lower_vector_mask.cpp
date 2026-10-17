// The mask halves of the lane-mask VALU ops, called directly: ir/lower/vop3_carry.cpp,
// ir/lower/vop3_cndmask.cpp, ir/lower/vop2_mask.cpp and the shared carry_mask_write.cpp. Pins where
// the carry mask is written (VCC, an SGPR pair, the Wave32 B32 domain), the EXEC masking, the
// mask reads, and every refusal.
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mask_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/vector_mask_harness.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {

struct Mask : LoweringFixture {
    uint32_t old_vcc;
    Mask() {
        rs.vreg[kVsrc0] = b.uconst(5);
        rs.vreg[kVsrc1] = b.uconst(7);
        rs.vreg[kVdst] = b.uconst(9);
        old_vcc = rs.vcc = b.bfalse();
    }
    bool carry_out(uint32_t op, uint32_t sdst) {
        return ir::lower_vop3_carry_out(b, rs, decode(vop3_words(op, sdst, 0)), ok, false);
    }
    bool carry_in_out(uint32_t op, uint32_t sdst, uint32_t src2) {
        return ir::lower_vop3_carry_in_out(b, rs, decode(vop3_words(op, sdst, src2)), ok, false);
    }
    bool cndmask(uint32_t src2) {
        const ir::Vop3MaskSource fv = [this](int k) { return rs.vreg[k == 0 ? kVsrc0 : kVsrc1]; };
        return ir::lower_vop3_cndmask(b, rs, decode(vop3_words(0x101, 0, src2)), ok, false, fv);
    }
    bool vop2_op(uint32_t op) { return ir::lower_vop2(b, rs, decode({vop2(op)}), ok, false); }
};

}   // namespace

TEST(IrLowerVectorMask, CarryOutToAnSgprPairIsATrackedMask) {
    Mask t;
    ASSERT_TRUE(t.carry_out(0x30F, kCarryOut));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(t.rs.vreg[kVdst], t.b.uconst(9));
    ASSERT_TRUE(t.rs.sreg_bool.contains(kCarryOut));
    EXPECT_FALSE(t.rs.sreg_bool_b32.contains(kCarryOut)) << "Wave64: a pair, not the B32 domain";
    EXPECT_EQ(t.rs.vcc, t.old_vcc);
}

TEST(IrLowerVectorMask, CarryOutToVccReplacesVcc) {
    for (const uint32_t vcc : {106u, 107u}) {
        Mask t;
        ASSERT_TRUE(t.carry_out(0x310, vcc));
        EXPECT_TRUE(t.ok);
        EXPECT_NE(t.rs.vcc, t.old_vcc);
        EXPECT_FALSE(t.rs.sreg_bool.contains(vcc));
    }
}

TEST(IrLowerVectorMask, Wave32CarryIsOneWordInTheB32Domain) {
    Mask t;
    t.b.wave_size = 32;
    t.b.allow_b32_masks = true;
    t.set_sgpr(kCarryOut, 0x1234);
    ASSERT_TRUE(t.carry_out(0x319, kCarryOut));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.sreg_bool_b32.contains(kCarryOut));
    EXPECT_TRUE(t.rs.sreg_bool_narrowed[kCarryOut]);
    EXPECT_FALSE(t.rs.sreg.contains(kCarryOut)) << "the old data word is retired";
}

TEST(IrLowerVectorMask, NarrowedExecMasksTheCarry) {
    Mask full, narrowed;
    narrowed.rs.exec = narrowed.b.bfalse();
    narrowed.rs.exec_narrowed = true;
    const size_t full_words = full.code_words(), narrowed_words = narrowed.code_words();
    ASSERT_TRUE(full.carry_out(0x30F, kCarryOut));
    ASSERT_TRUE(narrowed.carry_out(0x30F, kCarryOut));
    EXPECT_EQ((narrowed.code_words() - narrowed_words) - (full.code_words() - full_words), 5u)
        << "one OpLogicalAnd of EXEC and the carry";
}

TEST(IrLowerVectorMask, CarryOutToANonMaskDestinationRefuses) {
    // The decoder names every SDST an SGPR; a lowering that is handed anything else refuses.
    Mask t;
    Rdna2Inst in = t.decode(vop3_words(0x30F, kCarryOut, 0));
    in.sdst.kind = OperandKind::Special;
    ASSERT_TRUE(ir::lower_vop3_carry_out(t.b, t.rs, in, t.ok, false));
    EXPECT_FALSE(t.ok);
    EXPECT_FALSE(t.rs.sreg_bool.contains(kCarryOut));
}

TEST(IrLowerVectorMask, CarryInReadsVccOrATrackedPairAndRefusesAnUntrackedOne) {
    Mask vcc;
    vcc.rs.vcc = vcc.b.btrue();
    ASSERT_TRUE(vcc.carry_in_out(0x128, kVcc, kVcc));
    EXPECT_TRUE(vcc.ok);
    EXPECT_NE(vcc.rs.vcc, vcc.b.btrue()) << "the carry-out replaced the carry-in";

    Mask pair;
    pair.rs.sreg_bool[kCarryIn] = pair.b.btrue();
    ASSERT_TRUE(pair.carry_in_out(0x129, kCarryOut, kCarryIn));
    EXPECT_TRUE(pair.ok);
    EXPECT_TRUE(pair.rs.sreg_bool.contains(kCarryOut));

    Mask untracked;
    untracked.set_sgpr(kCarryIn, 1);   // scalar DATA, not a mask
    ASSERT_TRUE(untracked.carry_in_out(0x12A, kCarryOut, kCarryIn));
    EXPECT_FALSE(untracked.ok);
    EXPECT_EQ(untracked.rs.vreg[kVdst], untracked.b.uconst(9)) << "VDST untouched on refusal";
    EXPECT_FALSE(untracked.rs.sreg_bool.contains(kCarryOut));
}

TEST(IrLowerVectorMask, CndmaskReadsVccOrATrackedPair) {
    Mask vcc;
    vcc.rs.vcc = vcc.b.btrue();
    ASSERT_TRUE(vcc.cndmask(kVcc));
    EXPECT_TRUE(vcc.ok);
    EXPECT_NE(vcc.rs.vreg[kVdst], vcc.b.uconst(9));

    Mask pair;
    pair.rs.sreg_bool[kCarryOut] = pair.b.btrue();
    ASSERT_TRUE(pair.cndmask(kCarryOut));
    EXPECT_TRUE(pair.ok);

    Mask untracked;
    ASSERT_TRUE(untracked.cndmask(kCarryOut));
    EXPECT_FALSE(untracked.ok);

    Mask vcc_hi;   // VCC_HI needs its own B32 lifetime; it never falls back to VCC
    vcc_hi.rs.vcc = vcc_hi.b.btrue();
    ASSERT_TRUE(vcc_hi.cndmask(107));
    EXPECT_FALSE(vcc_hi.ok);
}

TEST(IrLowerVectorMask, OtherOpcodesAreNotTheirs) {
    Mask t;
    EXPECT_FALSE(t.carry_out(0x128, kVcc)) << "a carry-in op is not a carry-out-only op";
    EXPECT_FALSE(t.carry_in_out(0x30F, kVcc, kVcc));
    EXPECT_FALSE(
        ir::lower_vop3_carry_out(t.b, t.rs, t.decode(vop3_words(0x103, 0, 0)), t.ok, false));
    EXPECT_TRUE(t.ok);
}

TEST(IrLowerVectorMask, Vop2CarryReadsAndReplacesVcc) {
    Mask t;
    t.rs.vcc = t.b.btrue();
    ASSERT_TRUE(t.vop2_op(0x28));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(t.rs.vcc, t.b.btrue());
    EXPECT_NE(t.rs.vreg[kVdst], t.b.uconst(9));

    Mask cnd;
    cnd.rs.vcc = cnd.b.btrue();
    const uint32_t vcc = cnd.rs.vcc;
    ASSERT_TRUE(cnd.vop2_op(0x01));
    EXPECT_TRUE(cnd.ok);
    EXPECT_EQ(cnd.rs.vcc, vcc) << "v_cndmask_b32 reads VCC and leaves it";

    Mask no_vcc;
    no_vcc.rs.vcc = 0;
    no_vcc.vop2_op(0x01);
    EXPECT_FALSE(no_vcc.ok);
}
