// Pin source callback sequencing, packed publication and the unchanged caller modifier boundary.
#include "gpu/recompiler/ir/lower/vop3_pack_norm_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode = 0x368) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    in.src[0] = {OperandKind::VGPR, 2};
    in.src[1] = {OperandKind::VGPR, 3};
    in.src[2] = {OperandKind::None, 0};
    return in;
}
struct Pack : LoweringFixture {
    std::vector<int> reads;
    std::array<size_t, 2> read_at{};
    int bad = -1;
    ir::Vop3PackNormSource source = [this](int k) {
        reads.push_back(k);
        read_at.at(static_cast<size_t>(k)) = code_words();
        if (k == bad) ok = false;
        return b.uconst(k ? 0xbf000000 : 0x3f000000);
    };
    bool lower(const Rdna2Inst& in) {
        return ir::lower_vop3_pack_norm(b, rs, in, ok, false, std::cref(source));
    }
    size_t count(uint32_t opcode) const {
        size_t n = 0;
        for (size_t pc = 0; pc < b.code.size(); pc += b.code[pc] >> 16)
            n += (b.code[pc] & 0xffffu) == opcode;
        return n;
    }
};
}   // namespace
TEST(IrLowerVop3PackNorm, BothDefinitionsKeepEffectsAndSequentialFloatReads) {
    for (uint32_t opcode : {0x368u, 0x369u}) {
        Pack t;
        const auto& e = rdna2_instruction_effects(Rdna2Format::VOP3, opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(Rdna2Format::VOP3, opcode), 2u);
        ASSERT_TRUE(t.lower(instruction(opcode)));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0, 1}));
        EXPECT_GT(t.read_at[1], t.read_at[0]);
        EXPECT_NE(t.rs.vreg.at(1), 0u);
        EXPECT_EQ(t.count(Op_FMul), 2u);
        EXPECT_EQ(t.count(Op_ShiftLeftLogical), 1u);
        EXPECT_EQ(t.count(Op_BitwiseOr), 1u);
    }
}
TEST(IrLowerVop3PackNorm, SourceErrorsKeepBothConversionsAndPackedPublication) {
    for (uint32_t opcode : {0x368u, 0x369u})
        for (int bad : {0, 1}) {
            Pack t;
            t.bad = bad;
            ASSERT_TRUE(t.lower(instruction(opcode)));
            EXPECT_FALSE(t.ok);
            EXPECT_EQ(t.reads, (std::vector<int>{0, 1}));
            EXPECT_TRUE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.count(Op_FMul), 2u);
        }
}
TEST(IrLowerVop3PackNorm, CallerKeepsWaveStateAndModifierAdmission) {
    for (uint32_t opcode : {0x368u, 0x369u}) {
        Pack t;
        t.rs.exec = t.b.bfalse();
        t.rs.exec_narrowed = true;
        t.rs.vcc = t.b.btrue();
        t.rs.scc = t.b.bfalse();
        const auto exec = t.rs.exec, vcc = t.rs.vcc, scc = t.rs.scc;
        auto in = instruction(opcode);
        for (uint32_t k = 0; k < 3; ++k) in.src_abs[k] = in.src_neg[k] = true;
        in.clamp = true;
        in.omod = 3;
        ASSERT_TRUE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.reads, (std::vector<int>{0, 1}));
        EXPECT_TRUE(t.rs.vreg.contains(1));
        EXPECT_EQ(t.rs.exec, exec);
        EXPECT_EQ(t.rs.vcc, vcc);
        EXPECT_EQ(t.rs.scc, scc);
    }
}
TEST(IrLowerVop3PackNorm, OtherOpcodesAndFormatsHaveNoEffects) {
    for (auto in : {instruction(0x367), instruction(0x36a), instruction(kVop3OpcodeAddCoU32)}) {
        Pack t;
        const auto before = t.code_words();
        EXPECT_FALSE(t.lower(in));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.reads.empty());
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), before);
    }
    Pack t;
    auto in = instruction();
    in.fmt = Rdna2Format::VOP3P;
    EXPECT_FALSE(t.lower(in));
    EXPECT_TRUE(t.rs.vreg.empty());
    in.fmt = Rdna2Format::VOP3;
    in.opcode = 0x36a;
    EXPECT_FALSE(ir::lower_vop3_pack_norm(t.b, t.rs, in, t.ok, false, {}));
}
TEST(IrLowerVop3PackNorm, AHandledOpcodeRequiresItsSourceCallback) {
    EXPECT_DEATH(
        {
            prosper::test::suppress_coredump();
            Pack t;
            ir::lower_vop3_pack_norm(t.b, t.rs, instruction(), t.ok, false, {});
        },
        "source callback missing");
}
