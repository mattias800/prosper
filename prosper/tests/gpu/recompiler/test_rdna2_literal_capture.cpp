// test_rdna2_literal_capture — the mandatory literal K of the VOP2 mul-adds and the SOPK
// setreg data dword must be CAPTURED, not just counted.
//
// Length alone is not the contract: the walker sweep pins that the six K-carrying VOP2 mul-adds
// (v_madmk/madak_f32, v_fmamk/fmaak_f32, v_fmamk/fmaak_f16) and SOPK s_setreg_imm32_b32 are 2
// dwords, but a length fix that stopped capturing K (or captured the wrong dword) would keep
// every length green while the emitter folds a phantom constant. These arms assert has_literal
// and the exact captured value for each of the seven opcodes, plus that an ordinary VOP2 with
// plain sources captures nothing.
//
// Oracle: AMD RDNA2 ISA 70648 VOP2 K-table (v_madmk_f32 0x20, v_madak_f32 0x21, v_fmamk_f32 0x2C,
// v_fmaak_f32 0x2D, v_fmamk_f16 0x37, v_fmaak_f16 0x38 — the f16 pair are Table 75 ops 55/56)
// and SOPK Table 66 (only op 21, s_setreg_imm32_b32, carries trailing data). Pure decode.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace prosper::gpu;

namespace {

// VOP2 word: bit31=0, OP[30:25], VDST[24:17], VSRC1[16:9], SRC0[8:0]. Plain VGPR sources, so any
// literal comes only from the opcode's mandatory K slot.
uint32_t vop2_w0(uint32_t op) { return (op << 25) | (0u << 17) | (1u << 9) | 0x100u; }

}  // namespace

TEST(Rdna2LiteralCapture, Vop2MandatoryKIsCapturedForAllSixMulAdds) {
    const uint32_t ops[] = {0x20u, 0x21u, 0x2Cu, 0x2Du, 0x37u, 0x38u};
    for (uint32_t op : ops) {
        const uint32_t code[] = {vop2_w0(op), 0x3F800000u};
        const Rdna2Inst in = rdna2_decode_one(code, std::size(code));
        EXPECT_EQ(in.fmt, Rdna2Format::VOP2) << "op=" << op;
        EXPECT_EQ(in.opcode, op) << "op=" << op;
        EXPECT_EQ(in.len_dwords, 2u) << "op=" << op;
        EXPECT_TRUE(in.has_literal) << "K must be captured, op=" << op;
        EXPECT_EQ(in.literal, 0x3F800000u) << "K value must survive, op=" << op;
    }
}

TEST(Rdna2LiteralCapture, OrdinaryVop2WithPlainSourcesCapturesNothing) {
    // v_add_f32 (0x03) with VGPR sources: one dword, no literal. Guards K-capture leaking onto
    // ordinary ops (the converse miscompile of the arms above).
    const uint32_t code[] = {vop2_w0(0x03u)};
    const Rdna2Inst in = rdna2_decode_one(code, std::size(code));
    EXPECT_EQ(in.len_dwords, 1u);
    EXPECT_FALSE(in.has_literal);
}

TEST(Rdna2LiteralCapture, SopkSetregImm32CapturesItsDataDword) {
    // Exact llvm-mc gfx1030 words: s_setreg_imm32_b32 with trailing data 0x12345678.
    const uint32_t setreg[] = {0xBA80F801u, 0x12345678u};
    const Rdna2Inst keep = rdna2_decode_one(setreg, std::size(setreg));
    EXPECT_EQ(keep.fmt, Rdna2Format::SOPK);
    EXPECT_EQ(keep.opcode, 21u);
    EXPECT_EQ(keep.len_dwords, 2u);
    EXPECT_TRUE(keep.has_literal);
    EXPECT_EQ(keep.literal, 0x12345678u);
    // Neighbouring SOPK opcodes carry no literal: op 20 and 22 stay one dword.
    for (uint32_t op : {20u, 22u}) {
        const uint32_t word = 0xB0000000u | (op << 23) | 0x1234u;
        const Rdna2Inst in = rdna2_decode_one(&word, 1);
        EXPECT_EQ(in.fmt, Rdna2Format::SOPK) << "op=" << op;
        EXPECT_EQ(in.len_dwords, 1u) << "op=" << op;
        EXPECT_FALSE(in.has_literal) << "op=" << op;
    }
}
