// The value-only helper borrows both half readers; root owns modifiers and publication.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x0f) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3P;
    in.opcode = opcode;
    return in;
}
struct PackedFloat : LoweringFixture {
    std::vector<std::pair<int, bool>> reads;
    ir::PackedFloatHalfSource half = [this](int source, bool high) {
        reads.emplace_back(source, high);
        return b.uconst(source == 0 ? 0x3fc00000u : 0x40000000u);
    };
    uint32_t r = 0;
    bool lower(const Rdna2Inst& in, bool high = false) {
        return ir::lower_vop3p_packed_float(b, in, half, high, r);
    }
    size_t count(uint32_t op) const {
        size_t n = 0;
        for (size_t at = 0; at < b.code.size();) {
            const auto word = b.code[at];
            if ((word & 0xffffu) == op) ++n;
            at += word >> 16;
        }
        return n;
    }
};
}   // namespace
TEST(IrLowerVop3pPackedFloat, BothOperandsReceiveTheUnchangedHalfSelector) {
    for (auto op : {0x0fu, 0x10u, 0x11u, 0x12u})
        for (bool high : {false, true}) {
            PackedFloat t;
            ASSERT_TRUE(t.lower(instruction(op), high));
            EXPECT_TRUE(t.ok);
            EXPECT_NE(t.r, 0u);
            ASSERT_EQ(t.reads.size(), 2u);
            const std::set<std::pair<int, bool>> want = {{0, high}, {1, high}};
            EXPECT_EQ((std::set<std::pair<int, bool>>(t.reads.begin(), t.reads.end())), want);
            EXPECT_EQ(t.count(op == 0x0f ? Op_FAdd : op == 0x10 ? Op_FMul : Op_ExtInst), 1u);
        }
}
TEST(IrLowerVop3pPackedFloat, ReaderErrorsDoNotSkipTheOtherReadOrArithmetic) {
    for (int bad : {0, 1}) {
        PackedFloat t;
        t.half = [&](int source, bool high) {
            t.reads.emplace_back(source, high);
            if (source == bad) t.ok = false;
            return t.b.uconst(source == 0 ? 0x3f800000u : 0x40000000u);
        };
        ASSERT_TRUE(t.lower(instruction()));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_NE(t.r, 0u);
        EXPECT_EQ(t.count(Op_FAdd), 1u);
    }
}
TEST(IrLowerVop3pPackedFloat, RegisterAndWavePublicationRemainWithTheCaller) {
    PackedFloat t;
    t.rs.exec = t.b.bfalse();
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    const auto selects = t.count(Op_Select);
    auto in = instruction();
    in.clamp = true;
    in.vop3p_opsel = 3;
    in.vop3p_neg_hi = 3;
    ASSERT_TRUE(t.lower(in, true));
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.count(Op_Select), selects);
    EXPECT_EQ(t.count(Op_FMul), 0u);
}
TEST(IrLowerVop3pPackedFloat, FusedIntegerDotAndForeignOperationsAreUntouched) {
    for (auto op : {0x0eu, 0x00u, 0x14u, 0x20u}) {
        PackedFloat t;
        auto begin = t.code_words();
        t.r = 123;
        EXPECT_FALSE(t.lower(instruction(op)));
        EXPECT_TRUE(t.reads.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_EQ(t.r, 123u);
    }
    PackedFloat t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.reads.empty());
}
TEST(IrLowerVop3pPackedFloat, ARequiredMissingCallbackStopsLoudly) {
    PackedFloat t;
    EXPECT_FALSE(ir::lower_vop3p_packed_float(t.b, instruction(0x0e), {}, false, t.r));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3p_packed_float(t.b, instruction(), {}, false, t.r);
        },
        "packed float half callback missing");
}
