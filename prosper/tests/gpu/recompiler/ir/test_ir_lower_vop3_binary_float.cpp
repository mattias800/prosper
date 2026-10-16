// Pin existing binary float source/result boundaries and ADD/MUL observation order.
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include "gpu/recompiler/fragment_arithmetic_observation.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x103) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::VGPR, 5};
    return in;
}
struct FloatBinary : LoweringFixture {
    std::vector<int> reads;
    std::array<size_t, 3> read_at{};
    std::vector<uint32_t> modified;
    size_t modify_at = 0;
    std::array<uint32_t, 3> bits = {0x3f800000, 0x40000000, 0x40400000};
    int bad = -1;
    ir::BinaryFloatSource source = [this](int slot) {
        reads.push_back(slot);
        read_at.at(static_cast<size_t>(slot)) = code_words();
        if (slot == bad) {
            ok = false;
            return b.uconst(0);
        }
        return b.uconst(bits.at(static_cast<size_t>(slot)));
    };
    ir::BinaryFloatResult result = [this](uint32_t value) {
        modified.push_back(value);
        modify_at = code_words();
        return value;
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_binary_float(b, rs, in, ok, false, source, result);
    }
    size_t count(uint32_t opcode) const {
        size_t out = 0;
        for (size_t pc = 0; pc < b.code.size(); pc += b.code[pc] >> 16)
            out += (b.code[pc] & 0xffffu) == opcode;
        return out;
    }
};
}   // namespace
TEST(IrLowerVop3BinaryFloat, MatchesExactlySixBinaryLaneWiseForms) {
    size_t count = 0;
    for (uint32_t op = 0; op < 1024; ++op) {
        FloatBinary t;
        auto begin = t.code_words();
        if (!t.lower(instruction(op))) {
            EXPECT_TRUE(t.ok);
            EXPECT_TRUE(t.reads.empty());
            EXPECT_TRUE(t.modified.empty());
            EXPECT_EQ(t.code_words(), begin);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            continue;
        }
        ++count;
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, op);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, op), 2u);
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_EQ((std::set<int>(t.reads.begin(), t.reads.end())), (std::set<int>{0, 1}));
        ASSERT_EQ(t.modified.size(), 1u);
        EXPECT_EQ(t.rs.vreg.at(1), t.modified[0]);
        EXPECT_GE(t.modify_at, t.read_at[0]);
        EXPECT_GE(t.modify_at, t.read_at[1]);
    }
    EXPECT_EQ(count, 6u);
}
TEST(IrLowerVop3BinaryFloat, ResultCallbackReturnsThePublishedWord) {
    for (auto op : {0x103u, 0x104u, 0x105u, 0x108u, 0x10fu, 0x110u}) {
        FloatBinary t;
        auto marker = t.b.uconst(0x76543210);
        size_t calls = 0;
        t.result = [&](uint32_t value) {
            EXPECT_NE(value, 0u);
            ++calls;
            return marker;
        };
        ASSERT_TRUE(t.lower(instruction(op)));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(t.rs.vreg.at(1), marker);
    }
}
TEST(IrLowerVop3BinaryFloat, SourceErrorsStillRunTheValueAndModifierStages) {
    for (int bad : {0, 1}) {
        FloatBinary t;
        t.bad = bad;
        ASSERT_TRUE(t.lower(instruction()));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads.size(), 2u);
        EXPECT_EQ(t.modified.size(), 1u);
        EXPECT_EQ(t.count(Op_FAdd), 1u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3BinaryFloat, CallerOwnsFinalPredicationAndMaskState) {
    FloatBinary t;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
    auto selects = t.count(Op_Select);
    auto in = instruction();
    in.src_abs[0] = true;
    in.src_neg[1] = true;
    in.omod = 2;
    in.clamp = true;
    ASSERT_TRUE(t.lower(in));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    EXPECT_EQ(t.rs.scc, scc);
    EXPECT_EQ(t.count(Op_Select), selects);
}
TEST(IrLowerVop3BinaryFloat, LegacyPackedFusedAndForeignFormsRemainUnmatched) {
    for (auto in :
         {instruction(0x107), instruction(0x12f), instruction(0x14b), instruction(0x102)}) {
        FloatBinary t;
        auto begin = t.code_words();
        EXPECT_FALSE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.modified.empty());
        EXPECT_EQ(t.code_words(), begin);
        EXPECT_TRUE(t.rs.vreg.empty());
    }
    FloatBinary t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.reads.empty());
    EXPECT_TRUE(t.rs.vreg.empty());
}
TEST(IrLowerVop3BinaryFloat, AddAndMulObservationsPrecedeSourceAndResultReads) {
    for (auto op : {0x103u, 0x104u, 0x105u, 0x108u, 0x10fu, 0x110u}) {
        FloatBinary t;
        FragmentArithmeticObservation observation;
        t.b.is_fragment = true;
        t.b.fragment_arithmetic_observation = &observation;
        auto in = instruction(op);
        in.pc = 172;
        const uint32_t expected = op == 0x103 || op == 0x108 ? 1 : 0;
        t.source = [&](int slot) {
            EXPECT_EQ(observation.site_count, expected);
            t.reads.push_back(slot);
            return t.b.uconst(t.bits.at(static_cast<size_t>(slot)));
        };
        t.result = [&](uint32_t value) {
            EXPECT_EQ(observation.site_count, expected);
            return value;
        };
        ASSERT_TRUE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(observation.site_count, expected);
        if (expected) {
            EXPECT_EQ(observation.sites[0].pc, 172u);
            EXPECT_EQ(observation.sites[0].family,
                      op == 0x103 ? FragmentArithmeticFamily::Add : FragmentArithmeticFamily::Mul);
        }
    }
}

TEST(IrLowerVop3BinaryFloat, MissingCallbacksStopOnlyForMatchedForms) {
    FloatBinary t;
    EXPECT_FALSE(
        ir::lower_vop3_binary_float_erased(t.b, t.rs, instruction(0x107), t.ok, false, {}, {}));
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3_binary_float_erased(t.b, t.rs, instruction(), t.ok, false, {}, t.result);
        },
        "binary float callbacks missing");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            ir::lower_vop3_binary_float_erased(t.b, t.rs, instruction(), t.ok, false, t.source, {});
        },
        "binary float callbacks missing");
}
