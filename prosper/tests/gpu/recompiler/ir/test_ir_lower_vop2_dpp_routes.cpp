// The row-XOR value must leave VOP2's admission, source registers and flag ownership intact.
#include "gpu/recompiler/ir/lower/vector_alu_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using prosper::test::ir::LoweringFixture;
namespace {
Rdna2Inst instruction() {
    return LoweringFixture::decode(
        {(0x1cu << 25) | (4u << 17) | (3u << 9) | 250u, 0xff080002u | (0x164u << 8)});
}
uint32_t shuffles(const std::vector<uint32_t>& words, size_t first) {
    uint32_t count = 0;
    for (size_t i = first; i < words.size();) {
        const uint32_t n = words[i] >> 16;
        if (!n || n > words.size() - i) return 0;
        count += (words[i] & 0xffffu) == Op_GroupNonUniformShuffle;
        i += n;
    }
    return count;
}
TEST(IrLowerVop2DppRoutes, NativeSourceTransformKeepsMasksAndOtherRegisters) {
    const auto in = instruction();
    const auto& effects = rdna2_instruction_effects(in);
    ASSERT_TRUE(effects.describes_vector_result());
    EXPECT_EQ(effects.vdst_dwords, 1u);
    EXPECT_EQ(effects.implicit_writes, kRdna2ImplicitNone);
    for (uint32_t wave : {32u, 64u}) {
        LoweringFixture t;
        t.b.wave_size = t.b.native_subgroup_size = wave;
        t.rs.vreg[2] = t.b.load_input(0);
        t.rs.vreg[3] = t.b.load_input(1);
        t.rs.vreg[4] = t.b.uconst(0x53960000u);
        t.rs.vcc = t.b.bfalse();
        t.rs.scc = t.b.btrue();
        const auto a = t.rs.vreg[2], c = t.rs.vreg[3];
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        const auto start = t.b.code.size();
        EXPECT_TRUE(lower_vop2(t.b, t.rs, in, t.ok, true));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(shuffles(t.b.code, start), 2u);
        EXPECT_EQ(t.b.compute_min_subgroup_size, 16u);
        EXPECT_EQ(t.rs.vreg.at(2), a);
        EXPECT_EQ(t.rs.vreg.at(3), c);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
    }
}
TEST(IrLowerVop2DppRoutes, PortableAndNonComputeDoNotAcquireTheNativeValue) {
    for (bool compute : {false, true}) {
        LoweringFixture t;
        t.b.is_compute = compute;
        t.b.native_subgroup_size = compute ? 0 : 64;
        t.rs.vreg[2] = t.b.load_input(0);
        t.rs.vreg[3] = t.b.load_input(1);
        const auto old = t.rs.vreg[4] = t.b.uconst(0x53960000u);
        const auto start = t.b.code.size();
        EXPECT_TRUE(lower_vop2(t.b, t.rs, instruction(), t.ok, true));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(shuffles(t.b.code, start), 0u);
        EXPECT_EQ(t.b.compute_min_subgroup_size, 0u);
        EXPECT_EQ(t.rs.vreg.at(4), old);
    }
}
} // namespace
