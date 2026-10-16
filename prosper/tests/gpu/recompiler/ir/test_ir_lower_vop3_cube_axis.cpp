// Cube ID/MA must preserve caller modifiers, reads and publication without owning wave masks.
#include "gpu/recompiler/ir/lower/vop3_cube_axis_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x144) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    for (int k = 0; k < 3; ++k) in.src[k] = {OperandKind::VGPR, k + 2};
    return in;
}
struct Cube : LoweringFixture {
    std::vector<int> reads;
    std::vector<uint32_t> modified;
    bool bad = false;
    ir::CubeAxisSource source = [this](int k) {
        reads.push_back(k);
        if (bad) ok = false;
        return b.uconst(0x3f800000u + static_cast<uint32_t>(k) * 0x800000u);
    };
    ir::CubeAxisResult result = [this](uint32_t v) {
        modified.push_back(v);
        return v;
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_cube_axis(b, rs, in, ok, false, std::cref(source), std::cref(result));
    }
};
}   // namespace
TEST(IrLowerVop3CubeAxis, ReadsEverySourceOnceBeforePublishing) {
    for (auto op : {0x144u, 0x147u}) {
        Cube t;
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 3u);
        ASSERT_TRUE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0, 1, 2}));
        ASSERT_EQ(t.modified.size(), 1u);
        EXPECT_EQ(t.rs.vreg.at(1), t.modified[0]);
    }
}
TEST(IrLowerVop3CubeAxis, CallerResultAndWaveStateArePreserved) {
    for (auto op : {0x144u, 0x147u}) {
        Cube t;
        auto in = instruction(op);
        in.src_abs[0] = true;
        in.src_neg[1] = true;
        in.clamp = true;
        in.omod = 2;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc,
                   marker = t.b.uconst(0x13572468);
        size_t calls = 0;
        t.result = [&](uint32_t v) {
            EXPECT_NE(v, 0u);
            EXPECT_EQ(t.reads, (std::vector<int>{0, 1, 2}));
            ++calls;
            return marker;
        };
        ASSERT_TRUE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(t.rs.vreg.at(1), marker);
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
    }
}
TEST(IrLowerVop3CubeAxis, SourceFailureRetainsOriginalResultStage) {
    for (auto op : {0x144u, 0x147u}) {
        Cube t;
        t.bad = true;
        ASSERT_TRUE(t.lower(instruction(op)));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(t.modified.size(), 1u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3CubeAxis, CoordinateAndForeignOperationsRemainUnclaimed) {
    for (auto op : {0x143u, 0x145u, 0x146u, 0x148u, kVop3OpcodeAddCoU32}) {
        Cube t;
        const auto before = t.code_words();
        EXPECT_FALSE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.modified.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    Cube t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.rs.vreg.empty());
}
TEST(IrLowerVop3CubeAxis, MatchedRouteRequiresBothCallbacks) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Cube t;
            ir::lower_vop3_cube_axis(t.b, t.rs, instruction(), t.ok, false, {},
                                     std::cref(t.result));
        },
        "cube callbacks missing");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Cube t;
            ir::lower_vop3_cube_axis(t.b, t.rs, instruction(), t.ok, false, std::cref(t.source),
                                     {});
        },
        "cube callbacks missing");
}
