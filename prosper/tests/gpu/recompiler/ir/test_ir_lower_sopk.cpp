// ir/lower/sopk.cpp: the immediate forms share the SOP2/SOPC definitions; what is SOPK's own is
// how SIMM16 is extended (signed except for the unsigned compares) and that SDST is the source.
#include "gpu/recompiler/ir/lowering_fixture.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct Sopk : LoweringFixture {
    bool lower(uint32_t op, uint32_t simm16, uint32_t d = 22) {
        return ir::lower_sopk(b, rs, decode({sopk(op, d, simm16)}), ok, false);
    }
    Sopk() {
        set_sgpr(22, 5);
        rs.scc = b.bfalse();
    }
};
} // namespace

TEST(IrLowerSopk, MovkSignExtends) {
    Sopk t;
    ASSERT_TRUE(t.lower(0x00, 0xffff));
    uint32_t value = 0;
    ASSERT_TRUE(t.b.uconst_literal(t.rs.sreg[22], &value));
    EXPECT_EQ(value, 0xffffffffu);
}

TEST(IrLowerSopk, SignedComparesSignExtendAndUnsignedOnesDoNot) {
    Sopk is;
    is.lower(0x03, 0x8000);   // s_cmpk_eq_i32
    EXPECT_TRUE(is.ok);
    EXPECT_TRUE(is.constant_cached(0xffff8000u));
    Sopk un;
    const uint32_t s22 = un.rs.sreg[22];
    un.lower(0x09, 0x8000);   // s_cmpk_eq_u32
    EXPECT_TRUE(un.ok);
    EXPECT_TRUE(un.constant_cached(0x00008000u));
    EXPECT_FALSE(un.constant_cached(0xffff8000u));
    EXPECT_EQ(un.rs.sreg[22], s22) << "a compare leaves SDST alone";
}

TEST(IrLowerSopk, ReadModifyWriteForms) {
    Sopk addk;
    const uint32_t scc_before = addk.rs.scc;
    const uint32_t old = addk.rs.sreg[22];
    ASSERT_TRUE(addk.lower(0x0f, 1));   // s_addk_i32: SCC = signed overflow
    EXPECT_NE(addk.rs.sreg[22], old);
    EXPECT_NE(addk.rs.scc, scc_before);
    Sopk mulk;
    ASSERT_TRUE(mulk.lower(0x10, 3));   // s_mulk_i32: no SCC
    EXPECT_EQ(mulk.rs.scc, scc_before);
    EXPECT_NE(mulk.rs.sreg[22], old);
}

TEST(IrLowerSopk, WaitcntIsANoOpAndSetregNeedsCompute) {
    Sopk wait;
    const size_t words = wait.code_words();
    ASSERT_TRUE(wait.lower(0x18, 0, 125));   // s_waitcnt_vmcnt null, 0
    EXPECT_TRUE(wait.ok);
    EXPECT_EQ(wait.code_words(), words);
    Sopk setreg;
    setreg.b.is_compute = false;
    setreg.lower(0x13, (20u | (31u << 11)), 4);   // s_setreg_b32 hwreg(20, 0, 32), s4
    EXPECT_FALSE(setreg.ok);
}
