// RTZ consumes two source words; root modifiers and final EXEC publication stay outside.
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x12f) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    return in;
}
struct RtzPack : LoweringFixture {
    std::vector<int> reads;
    std::array<size_t, 3> read_at{};
    std::array<uint32_t, 3> bits = {0x3f800000, 0x40000000, 0x40400000};
    int bad = -1;
    ir::RtzPackSource source = [this](int slot) {
        reads.push_back(slot);
        read_at.at(static_cast<size_t>(slot)) = code_words();
        if (slot == bad) {
            ok = false;
            return b.uconst(0);
        }
        return b.uconst(bits.at(static_cast<size_t>(slot)));
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_rtz_pack(b, rs, in, ok, false, source);
    }
    size_t count(uint32_t opcode) const {
        size_t out = 0;
        for (size_t pc = 0; pc < b.code.size(); pc += b.code[pc] >> 16)
            out += (b.code[pc] & 0xffffu) == opcode;
        return out;
    }
};
}   // namespace
TEST(IrLowerVop3RtzPack, MatchesExactlyTheExistingFullWordForm) {
    size_t count = 0;
    for (uint32_t op = 0; op < 1024; ++op) {
        RtzPack t;
        auto before = t.code_words();
        if (!t.lower(instruction(op))) {
            EXPECT_TRUE(t.ok);
            EXPECT_TRUE(t.reads.empty());
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.code_words(), before);
            continue;
        }
        ++count;
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 2u);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_EQ((std::set<int>(t.reads.begin(), t.reads.end())), (std::set<int>{0, 1}));
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
    EXPECT_EQ(count, 1u);
}
TEST(IrLowerVop3RtzPack, FullDestinationPublicationDoesNotKeepTheOldWord) {
    RtzPack t;
    auto old = t.b.uconst(0xaabbccdd);
    t.rs.vreg[1] = old;
    ASSERT_TRUE(t.lower(instruction()));
    EXPECT_TRUE(t.ok);
    EXPECT_NE(t.rs.vreg.at(1), old);
    EXPECT_EQ(t.reads.size(), 2u);
    EXPECT_GT(t.count(Op_BitwiseOr), 0u);
}
TEST(IrLowerVop3RtzPack, SourceFailuresStillReachTheOriginalValueStage) {
    for (int bad : {0, 1}) {
        RtzPack t;
        t.bad = bad;
        ASSERT_TRUE(t.lower(instruction()));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3RtzPack, CallerRetainsModifiersAndFinalExecPublication) {
    RtzPack plain, modified;
    auto in = instruction();
    in.src_abs[0] = true;
    in.src_abs[1] = true;
    in.src_neg[0] = true;
    in.src_neg[1] = true;
    in.omod = 2;
    in.clamp = true;
    ASSERT_TRUE(plain.lower(instruction()));
    ASSERT_TRUE(modified.lower(in));
    EXPECT_TRUE(plain.ok);
    EXPECT_TRUE(modified.ok);
    EXPECT_EQ(plain.b.code, modified.b.code);
    auto exec = modified.rs.exec, vcc = modified.rs.vcc, scc = modified.rs.scc;
    ASSERT_TRUE(modified.lower(in));
    EXPECT_EQ(modified.rs.exec, exec);
    EXPECT_EQ(modified.rs.vcc, vcc);
    EXPECT_EQ(modified.rs.scc, scc);
}
TEST(IrLowerVop3RtzPack, RawHalfPackedAndForeignFormsHaveNoEffects) {
    for (auto in : {instruction(0x311), instruction(0x2f), instruction(0x34b)}) {
        RtzPack t;
        auto before = t.code_words();
        EXPECT_FALSE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    RtzPack t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3P;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_TRUE(t.reads.empty());
}
TEST(IrLowerVop3RtzPack, MatchedFormNeedsItsBorrowedSourceCallback) {
    RtzPack t;
    EXPECT_FALSE(ir::lower_vop3_rtz_pack_erased(t.b, t.rs, instruction(0x311), t.ok, false, {}));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3_rtz_pack_erased(t.b, t.rs, instruction(), t.ok, false, {});
        },
        "RTZ source callback missing");
}
