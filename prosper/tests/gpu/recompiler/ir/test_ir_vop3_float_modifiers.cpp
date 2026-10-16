// The common numeric stages must retain ABS-before-sign-XOR and OMOD-before-CLAMP,
// without touching operand materialization or the caller's destination/proof state.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_float_modifiers.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace {
size_t count(const SpirvCompute& b, uint32_t opcode, uint32_t ext = 0) {
    size_t n = 0;
    for (size_t at = 0; at < b.code.size(); at += b.code[at] >> 16) {
        if ((b.code[at] & 0xffffu) == opcode &&
            (!ext || (b.code[at] >> 16) >= 5 && b.code[at + 4] == ext))
            ++n;
    }
    return n;
}
}   // namespace
TEST(IrVop3FloatModifiers, SourceReadsTheSelectedSlotOnceBeforeItsNumericModifiers) {
    for (int slot = 0; slot < 3; ++slot)
        for (uint32_t flags = 0; flags < 4; ++flags) {
            LoweringFixture t;
            Rdna2Inst in{};
            in.src[slot] = {OperandKind::VGPR, 7 + slot};
            in.src_abs[slot] = flags & 1;
            in.src_neg[slot] = flags & 2;
            std::vector<int> reads;
            auto read = [&](const Operand& operand) {
                EXPECT_EQ(count(t.b, Op_BitwiseXor), 0u);
                EXPECT_EQ(count(t.b, Op_ExtInst, Glsl_FAbs), 0u);
                reads.push_back(operand.value);
                return t.b.uconst(0xbfa00000);
            };
            bool routed = false;
            ir::Vop3FloatModifiers stage{t.b, in, read, routed};
            EXPECT_NE(stage.fv(slot), 0u);
            EXPECT_EQ(reads, (std::vector<int>{7 + slot}));
            EXPECT_EQ(count(t.b, Op_ExtInst, Glsl_FAbs), uint32_t(bool(flags & 1)));
            EXPECT_EQ(count(t.b, Op_BitwiseXor), uint32_t(bool(flags & 2)));
            if ((flags & 3) == 3) {
                size_t abs_at = 0, neg_at = 0;
                for (size_t at = 0; at < t.b.code.size(); at += t.b.code[at] >> 16) {
                    if ((t.b.code[at] & 0xffffu) == Op_ExtInst && t.b.code[at + 4] == Glsl_FAbs)
                        abs_at = at;
                    if ((t.b.code[at] & 0xffffu) == Op_BitwiseXor) neg_at = at;
                }
                EXPECT_LT(abs_at, neg_at);
            }
            EXPECT_FALSE(routed);
            EXPECT_TRUE(t.rs.vreg.empty());
        }
}
TEST(IrVop3FloatModifiers, ResultRoutesEvenWithoutModifiersAndScalesBeforeClamping) {
    for (uint32_t scale = 0; scale < 4; ++scale)
        for (bool clamp : {false, true}) {
            LoweringFixture t;
            Rdna2Inst in{};
            in.omod = scale;
            in.clamp = clamp;
            auto read = [&](const Operand&) -> uint32_t {
                ADD_FAILURE() << "result stage must never read a source";
                return 0;
            };
            bool routed = false;
            ir::Vop3FloatModifiers stage{t.b, in, read, routed};
            const auto input = t.b.uconst(0x3e800000);
            const auto result = stage.fresult(input);
            EXPECT_TRUE(routed);
            EXPECT_EQ(count(t.b, Op_FMul), uint32_t(scale != 0));
            EXPECT_EQ(count(t.b, Op_ExtInst, Glsl_NMax), uint32_t(clamp));
            EXPECT_EQ(count(t.b, Op_ExtInst, Glsl_NMin), uint32_t(clamp));
            if (!scale && !clamp) EXPECT_EQ(result, input);
            if (scale)
                EXPECT_TRUE(t.constant_cached(scale == 1   ? 0x40000000
                                              : scale == 2 ? 0x40800000
                                                           : 0x3f000000));
            if (scale && clamp) {
                size_t mul_at = 0, clamp_at = 0;
                for (size_t at = 0; at < t.b.code.size(); at += t.b.code[at] >> 16) {
                    if ((t.b.code[at] & 0xffffu) == Op_FMul) mul_at = at;
                    if ((t.b.code[at] & 0xffffu) == Op_ExtInst && t.b.code[at + 4] == Glsl_NMax)
                        clamp_at = at;
                }
                EXPECT_LT(mul_at, clamp_at);
            }
            EXPECT_TRUE(t.rs.vreg.empty());
        }
}
