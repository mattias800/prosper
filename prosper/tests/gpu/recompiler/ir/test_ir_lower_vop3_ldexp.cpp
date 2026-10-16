// Pin the existing exponent bus, float callbacks and modifier refusal boundary.
#include "gpu/recompiler/ir/lower/vop3_ldexp_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x362) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::InlineInt, -2};
    in.src[2] = {OperandKind::None, 0};
    return in;
}
struct Scale : LoweringFixture {
    std::vector<int> reads;
    std::vector<uint32_t> modified;
    bool bad = false;
    ir::Vop3LdexpSource source = [this](int k) {
        reads.push_back(k);
        if (bad) ok = false;
        return b.uconst(0x40c00000);
    };
    ir::Vop3LdexpResult result = [this](uint32_t value) {
        modified.push_back(value);
        return value;
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_ldexp(b, rs, in, ok, false, std::cref(source), std::cref(result));
    }
};
}   // namespace
TEST(IrLowerVop3Ldexp, FloatAndSignedExponentKeepTheirDistinctReaders) {
    Scale t;
    const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, 0x362);
    EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
    EXPECT_EQ(e.vdst_dwords, 1u);
    EXPECT_FALSE(e.vdst_half);
    EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, 0x362), 2u);
    ASSERT_TRUE(t.lower(instruction()));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.reads, (std::vector<int>{0}));
    ASSERT_EQ(t.modified.size(), 1u);
    EXPECT_EQ(t.rs.vreg.at(1), t.modified[0]);
    EXPECT_TRUE(t.b.uconst_cache.contains(0xfffffffe));
}
TEST(IrLowerVop3Ldexp, ResultCallbackValueIsPublishedAndCallerKeepsWaveState) {
    Scale t;
    auto in = instruction();
    in.src_abs[0] = true;
    in.src_neg[0] = true;
    in.clamp = true;
    t.rs.exec = t.b.bfalse();
    t.rs.exec_narrowed = true;
    t.rs.vcc = t.b.btrue();
    t.rs.scc = t.b.bfalse();
    const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc, marker = t.b.uconst(0x13572468);
    size_t calls = 0;
    t.result = [&](uint32_t value) {
        EXPECT_NE(value, 0u);
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
TEST(IrLowerVop3Ldexp, ExistingExponentUnusedSourceAndOutputModifierRefusalsHaveNoPublication) {
    for (int field = 0; field < 5; ++field) {
        Scale t;
        auto in = instruction();
        if (field < 2)
            in.src_abs[field + 1] = true;
        else if (field < 4)
            in.src_neg[field - 1] = true;
        else
            in.omod = 1;
        const auto before = t.code_words();
        ASSERT_TRUE(t.lower(in));
        EXPECT_FALSE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.modified.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
}
TEST(IrLowerVop3Ldexp, OperandErrorsKeepTheOriginalValueAndResultStages) {
    for (bool bad_float : {false, true}) {
        Scale t;
        auto in = instruction();
        t.bad = bad_float;
        if (!bad_float) in.src[1] = {OperandKind::None, 0};
        ASSERT_TRUE(t.lower(in));
        EXPECT_FALSE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0}));
        EXPECT_EQ(t.modified.size(), 1u);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}
TEST(IrLowerVop3Ldexp, UnmatchedInstructionsAndFormatsHaveNoEffects) {
    for (auto in : {instruction(0x361), instruction(0x363), instruction(kVop3OpcodeAddCoU32)}) {
        Scale t;
        const auto before = t.code_words();
        EXPECT_FALSE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.modified.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    Scale t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP2;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.rs.vreg.empty());
    in.fmt = Rdna2Format::VOP3;
    in.opcode = 0x363;
    EXPECT_FALSE(ir::lower_vop3_ldexp(t.b, t.rs, in, t.ok, false, {}, {}));
}
TEST(IrLowerVop3Ldexp, HandledInstructionRequiresCallbacks) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Scale t;
            ir::lower_vop3_ldexp(t.b, t.rs, instruction(), t.ok, false, {}, std::cref(t.result));
        },
        "float callbacks missing");
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Scale t;
            ir::lower_vop3_ldexp(t.b, t.rs, instruction(), t.ok, false, std::cref(t.source), {});
        },
        "float callbacks missing");
}
