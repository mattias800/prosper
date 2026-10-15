// ir/lower/sop1.cpp: the SOP1 plain scalar-data path (s_mov_b32, s_not_b32, s_brev_b32,
// s_abs_i32) and its refusals.
#include "gpu/recompiler/ir/lowering_fixture.hpp"

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct Sop1 : LoweringFixture {
    bool lower(uint32_t op, uint32_t d = 22, uint32_t s0 = 20) {
        return ir::lower_sop1(b, rs, decode({sop1(op, d, s0)}), ok, false);
    }
    Sop1() {
        set_sgpr(20, 0x80000000u);
        rs.scc = b.bfalse();
    }
};
} // namespace

TEST(IrLowerSop1, MovPublishesTheSourceWord) {
    Sop1 t;
    ASSERT_TRUE(t.lower(0x03));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.sreg[22], t.rs.sreg[20]) << "a move emits nothing and copies the id";
}

TEST(IrLowerSop1, DefinitionsWriteDestinationAndTheirScc) {
    struct Case {
        uint32_t op;
        bool writes_scc;
    };
    for (Case c : {Case{0x07, true}, Case{0x0b, false}, Case{0x34, true}}) {   // not, brev, abs
        Sop1 t;
        t.rs.sreg_srt[22] = 0x40;
        const uint32_t scc_before = t.rs.scc;
        ASSERT_TRUE(t.lower(c.op));
        EXPECT_TRUE(t.ok) << c.op;
        EXPECT_NE(t.rs.sreg[22], t.rs.sreg[20]) << c.op;
        EXPECT_EQ(t.rs.scc != scc_before, c.writes_scc) << c.op;
        EXPECT_FALSE(t.rs.sreg_srt.contains(22)) << c.op;
    }
}

TEST(IrLowerSop1, UnmodelledOpRefuses) {
    Sop1 t;
    t.lower(0x19);   // s_sext_i32_i8: no definition and no route
    EXPECT_FALSE(t.ok);
}
