// Pin real destination flow and the original interleaving of source reads and bitfield operations.
#include "gpu/recompiler/ir/lower/vop3_bitfield_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
Rdna2Inst instruction(uint32_t opcode) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::VOP3;
    in.opcode = opcode;
    in.dst = {OperandKind::VGPR, 1};
    for (int source = 0; source < 3; ++source)
        in.src[source] = {OperandKind::InlineInt, source == 0 ? 59 : source == 1 ? 61 : 62};
    return in;
}
struct Result {
    uint32_t opcode;
    std::vector<uint32_t> operands;
};
std::map<uint32_t, Result> results(const LoweringFixture& t, size_t begin) {
    std::map<uint32_t, Result> out;
    for (size_t pc = begin; pc < t.b.code.size();) {
        const size_t count = t.b.code[pc] >> 16;
        if (count < 3 || count > t.b.code.size() - pc) {
            ADD_FAILURE() << "invalid instruction size at " << pc;
            break;
        }
        out[t.b.code[pc + 2]] = {t.b.code[pc] & 0xffffu,
                                 {t.b.code.begin() + pc + 3, t.b.code.begin() + pc + count}};
        pc += count;
    }
    return out;
}
}   // namespace
TEST(IrLowerVop3Bitfield, ExtractMasksOffsetAndCountBeforeReadingData) {
    for (auto opcode : {0x148u, 0x149u}) {
        LoweringFixture t;
        const size_t begin = t.code_words();
        ASSERT_TRUE(ir::lower_vop3_bitfield(t.b, t.rs, instruction(opcode), t.ok, false));
        ASSERT_TRUE(t.ok);
        ASSERT_TRUE(t.rs.vreg.contains(1));
        const auto ops = results(t, begin);
        std::vector<uint32_t> masks;
        for (const auto& [id, op] : ops)
            if (op.opcode == Op_BitwiseAnd) masks.push_back(id);
        ASSERT_EQ(masks.size(), 2u);
        EXPECT_EQ(ops.at(masks[0]).operands,
                  (std::vector<uint32_t>{t.b.uconst_cache.at(61), t.b.uconst_cache.at(31)}));
        EXPECT_EQ(ops.at(masks[1]).operands,
                  (std::vector<uint32_t>{t.b.uconst_cache.at(62), t.b.uconst_cache.at(31)}));
        EXPECT_LT(t.b.uconst_cache.at(61), masks[0]);
        EXPECT_LT(masks[0], t.b.uconst_cache.at(62));
        EXPECT_LT(t.b.uconst_cache.at(62), masks[1]);
        EXPECT_LT(masks[1], t.b.uconst_cache.at(59));
        auto d = t.rs.vreg.at(1);
        if (opcode == 0x149) {
            ASSERT_TRUE(ops.contains(d));
            EXPECT_EQ(ops.at(d).opcode, Op_Bitcast);
            ASSERT_EQ(ops.at(d).operands.size(), 1u);
            d = ops.at(d).operands[0];
        }
        ASSERT_TRUE(ops.contains(d));
        const auto& extract = ops.at(d);
        EXPECT_EQ(extract.opcode, opcode == 0x148 ? Op_BitFieldUExtract : Op_BitFieldSExtract);
        ASSERT_EQ(extract.operands.size(), 3u);
        auto data = extract.operands[0];
        if (opcode == 0x149) {
            ASSERT_TRUE(ops.contains(data));
            EXPECT_EQ(ops.at(data).opcode, Op_Bitcast);
            ASSERT_EQ(ops.at(data).operands.size(), 1u);
            data = ops.at(data).operands[0];
        }
        EXPECT_EQ(data, t.b.uconst_cache.at(59));
        const auto off = extract.operands[1];
        ASSERT_TRUE(ops.contains(off));
        EXPECT_EQ(ops.at(off).opcode, Op_ExtInst);
        EXPECT_EQ(ops.at(off).operands,
                  (std::vector<uint32_t>{t.b.glsl, Glsl_UMin, masks[0], t.b.uconst_cache.at(32)}));
    }
}
TEST(IrLowerVop3Bitfield, InsertDestinationUsesBothLegsAndDelaysTheLastSource) {
    LoweringFixture t;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_bitfield(t.b, t.rs, instruction(0x14a), t.ok, false));
    ASSERT_TRUE(t.ok);
    const auto ops = results(t, begin);
    const auto d = t.rs.vreg.at(1);
    ASSERT_TRUE(ops.contains(d));
    EXPECT_EQ(ops.at(d).opcode, Op_BitwiseOr);
    ASSERT_EQ(ops.at(d).operands.size(), 2u);
    const auto first = ops.at(d).operands[0], second = ops.at(d).operands[1];
    ASSERT_TRUE(ops.contains(first));
    ASSERT_TRUE(ops.contains(second));
    EXPECT_EQ(ops.at(first).opcode, Op_BitwiseAnd);
    EXPECT_EQ(ops.at(first).operands,
              (std::vector<uint32_t>{t.b.uconst_cache.at(59), t.b.uconst_cache.at(61)}));
    EXPECT_LT(first, t.b.uconst_cache.at(62));
    EXPECT_EQ(ops.at(second).opcode, Op_BitwiseAnd);
    ASSERT_EQ(ops.at(second).operands.size(), 2u);
    EXPECT_EQ(ops.at(second).operands[1], t.b.uconst_cache.at(62));
    const auto inverted = ops.at(second).operands[0];
    ASSERT_TRUE(ops.contains(inverted));
    EXPECT_EQ(ops.at(inverted).opcode, Op_Not);
    EXPECT_EQ(ops.at(inverted).operands, (std::vector<uint32_t>{t.b.uconst_cache.at(59)}));
}
TEST(IrLowerVop3Bitfield, ValueStageLeavesExecVccAndPredicationToItsCaller) {
    LoweringFixture t;
    t.rs.exec = t.b.ucmp(Op_IEqual, t.b.uconst(0), t.b.uconst(1));
    t.rs.vcc = t.b.btrue();
    const auto exec = t.rs.exec, vcc = t.rs.vcc;
    const size_t begin = t.code_words();
    ASSERT_TRUE(ir::lower_vop3_bitfield(t.b, t.rs, instruction(0x14a), t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_EQ(t.rs.exec, exec);
    EXPECT_EQ(t.rs.vcc, vcc);
    for (const auto& [id, op] : results(t, begin)) EXPECT_NE(op.opcode, Op_Select);
}
TEST(IrLowerVop3Bitfield, InvalidSourceKeepsTheExistingFailureSignal) {
    for (int source = -1; source < 3; ++source) {
        LoweringFixture t;
        auto in = instruction(0x14a);
        if (source >= 0) in.src[source].kind = OperandKind::None;
        EXPECT_TRUE(ir::lower_vop3_bitfield(t.b, t.rs, in, t.ok, false));
        EXPECT_EQ(t.ok, source < 0);
    }
}
TEST(IrLowerVop3Bitfield, OtherFormatsReservedAndNeighbouringOpcodesFallBackUntouched) {
    for (auto opcode : {0x128u, kVop3OpcodeAlignbitB32}) {
        LoweringFixture t;
        const size_t begin = t.code_words();
        EXPECT_FALSE(ir::lower_vop3_bitfield(t.b, t.rs, instruction(opcode), t.ok, false));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.empty());
        EXPECT_EQ(t.code_words(), begin);
    }
    LoweringFixture t;
    auto in = instruction(0x148);
    in.fmt = Rdna2Format::VOP2;
    const size_t begin = t.code_words();
    EXPECT_FALSE(ir::lower_vop3_bitfield(t.b, t.rs, in, t.ok, false));
    EXPECT_TRUE(t.ok);
    EXPECT_TRUE(t.rs.vreg.empty());
    EXPECT_EQ(t.code_words(), begin);
}
