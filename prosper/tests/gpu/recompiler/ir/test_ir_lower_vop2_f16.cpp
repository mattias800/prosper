// Half-result lowering preserves the unselected half, while packed RTZ writes a whole dword.
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {
struct HalfVop2 : LoweringFixture {
    HalfVop2() {
        rs.vreg[kVsrc0] = b.uconst(0x3c003c00);
        rs.vreg[kVsrc1] = b.uconst(0x40004000);
        rs.vreg[kVdst] = b.uconst(0x12344200);
    }
    bool lower(uint32_t op) { return ir::lower_vop2(b, rs, decode({vop2(op)}), ok, false); }
};
}   // namespace

TEST(IrLowerVop2F16, PlainHalfResultKeepsTheHighHalf) {
    HalfVop2 t;
    ASSERT_TRUE(t.lower(0x32));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.constant_cached(0xffff0000));
    EXPECT_NE(t.rs.vreg[kVdst], t.b.uconst(0x12344200));
}

TEST(IrLowerVop2F16, PackedRtzHasFiniteOverflowSaturation) {
    HalfVop2 t;
    ASSERT_TRUE(t.lower(0x2f));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.constant_cached(0x7bff));
}

TEST(IrLowerVop2F16, UnsupportedSiblingDoesNotGainAdmission) {
    HalfVop2 t;
    t.lower(0x34);
    EXPECT_FALSE(t.ok);
}
