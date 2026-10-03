// test_gap_opcode_refusals — fail-visible refusal pins for the six opcode
// families the AMDGPU survey flagged as decode-without-lowering.
//
// WHAT THIS PINS TODAY: per gap opcode, (1) the decoder RECOGNIZES the exact
// LLVM-attested word (right format, opcode and operands — not an "unknown
// encoding"), and (2) the compute recompiler REFUSES it (empty module) with
// a reason naming that pc/words/op, rather than silently mistranslating it.
// The observed mode is `unresolved-operand` (handled-but-unlowered), not
// `unknown-encoding`: the dispatch claims these instructions and then fails
// to lower them. That tag is pinned as-measured; per the mode taxonomy it
// reads as "fix the descriptor", while the truth for these six is "write
// the lowering" — the words + op in the reason are what keep the pin
// pointed at the opcode rather than at operand plumbing.
// Per the recompiler charter an unsupported op is a FATAL gap; the loud
// refusal is the backstop, and every live-boot hit is the next thing to
// implement. Any later lowering replaces these arms with execution tests;
// these arms do not require admission.
//
// Oracle: llvm-project llvm/test/MC/AMDGPU gfx10_asm_*.s (exact word
// encodings cited per arm). The shared CONTROL (a supported VOP3 sibling
// compiling through the same entry point) proves the refusal is about the
// opcode, not the program shape. No device is created; compile-only, runs
// everywhere.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

auto isV = [](const Operand& o, int n) {
    return o.kind == OperandKind::VGPR && o.value == (uint32_t)n;
};
auto isS = [](const Operand& o, int n) {
    return o.kind == OperandKind::SGPR && o.value == (uint32_t)n;
};

static int fails = 0;
#define CHECK(c, m) EXPECT_TRUE(c) << (m)

namespace {

// All source operands are defined in-shader: an unresolved-operand refusal
// would otherwise be indistinguishable from an opcode refusal, which is the
// exact vacuity these arms must not have.
std::vector<uint32_t> compile_with_addr(const uint32_t* code, size_t n, uint64_t addr) {
    return recompile_compute(code, n, nullptr, ComputeShaderConfig{},
                             RecompileDiagnosticContext{RecompileDiagnosticStage::Compute, addr});
}

}  // namespace

// Shared CONTROL: the supported VOP3 sibling (Worms v_sad_u32, Tsad1) through
// the same recompile_compute entry point. If the harness or the program
// shape were at fault, this would refuse too.
TEST(GapOpcodeRefusals, ControlSupportedVop3Compiles) {
    static const uint32_t code[] = {
        0x7ec20300u,              // v_mov_b32 v97, v0
        0x7ec002f4u,              // v_mov_b32 v96, 2.0
        0xd55d0062u, 0x057ec161u,  // v_sad_u32 v98, v97, v96, v95
        0xbf810000u,
    };
    const uint64_t addr = 0xA000ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    CHECK(!spv.empty(), "CONTROL compiles: the refusal arms below are about "
                        "their opcodes, not the harness or program shape");
    EXPECT_EQ(fails, 0);
}

// V_PERM_B32 (VOP3). LLVM gfx10_asm_vop3.s:
//   v_perm_b32 v5, v1, v2, v3 ; encoding: [0x05,0x00,0x44,0xd7,...]
TEST(GapOpcodeRefusals, PermB32) {
    static const uint32_t perm[] = {0xd7440005u, 0x040e0501u};
    Rdna2Inst dec = rdna2_decode_one(perm, 2);
    CHECK(dec.fmt == Rdna2Format::VOP3 && dec.opcode == 0x344u && dec.len_dwords == 2u &&
              isV(dec.dst, 5) && dec.n_src == 3u && isV(dec.src[0], 1) && isV(dec.src[1], 2) &&
              isV(dec.src[2], 3),
          "v_perm_b32 decodes as VOP3 op 0x344 with v5/v1/v2/v3");
    printf("  [perm] fmt=%d op=0x%x len=%u\n", (int)dec.fmt, dec.opcode, dec.len_dwords);
    static const uint32_t code[] = {
        0x7e020381u,              // v_mov_b32 v1, 1
        0x7e040382u,              // v_mov_b32 v2, 2
        0x7e060383u,              // v_mov_b32 v3, 3
        0xd7440005u, 0x040e0501u,  // v_perm_b32 v5, v1, v2, v3
        0xbf810000u,
    };
    const uint64_t addr = 0xA001ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [perm] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "v_perm_b32 refuses (no module)");
    CHECK(reason.find("d7440005") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}

// V_MAD_I64_I32 (VOP3). LLVM gfx10_asm_vop3.s:
//   v_mad_i64_i32 v[5:6], s12, v1, v2, v[3:4]
//   ; encoding: [0x05,0x0c,0x77,0xd5,...]
TEST(GapOpcodeRefusals, MadI64I32) {
    static const uint32_t mad[] = {0xd5770c05u, 0x040e0501u};
    Rdna2Inst dec = rdna2_decode_one(mad, 2);
    // fmt + op + length only: VOP3B carries src0 in word0 (here s12), which
    // the decoder does not model � it reads plain-VOP3 operands. That gap
    // is part of what the refusal below covers, so asserting the misread
    // operands as correct would bless it.
    CHECK(dec.fmt == Rdna2Format::VOP3 && dec.opcode == 0x177u && dec.len_dwords == 2u,
          "v_mad_i64_i32 decodes as VOP3 op 0x177");
    printf("  [mad] fmt=%d op=0x%x len=%u\n", (int)dec.fmt, dec.opcode, dec.len_dwords);
    static const uint32_t code[] = {
        0xbe8c0381u,              // s_mov_b32 s12, 1
        0xbe8d0382u,              // s_mov_b32 s13, 2
        0x7e020281u,              // v_mov_b32 v1, 1
        0x7e040282u,              // v_mov_b32 v2, 2
        0x7e060283u,              // v_mov_b32 v3, 3
        0x7e080280u,              // v_mov_b32 v4, 0
        0xd5770c05u, 0x040e0501u,  // v_mad_i64_i32 v[5:6], s12, v1, v2, v[3:4]
        0xbf810000u,
    };
    const uint64_t addr = 0xA011ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [mad] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "v_mad_i64_i32 refuses (no module)");
    CHECK(reason.find("d5770c05") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}

// V_DIV_FIXUP_F32 (VOP3). LLVM gfx10_asm_vop3.s:
//   v_div_fixup_f32 v5, v1, v2, v3 ; encoding: [0x05,0x00,0x5f,0xd5,...]
TEST(GapOpcodeRefusals, DivFixupF32) {
    static const uint32_t fix[] = {0xd55f0005u, 0x040e0501u};
    Rdna2Inst dec = rdna2_decode_one(fix, 2);
    CHECK(dec.fmt == Rdna2Format::VOP3 && dec.opcode == 0x15fu && dec.len_dwords == 2u &&
              isV(dec.dst, 5) && dec.n_src == 3u && isV(dec.src[0], 1) && isV(dec.src[1], 2) &&
              isV(dec.src[2], 3),
          "v_div_fixup_f32 decodes as VOP3 op 0x15f with v5/v1/v2/v3");
    printf("  [div] fmt=%d op=0x%x len=%u\n", (int)dec.fmt, dec.opcode, dec.len_dwords);
    static const uint32_t code[] = {
        0x7e020381u,              // v_mov_b32 v1, 1
        0x7e040382u,              // v_mov_b32 v2, 2
        0x7e060383u,              // v_mov_b32 v3, 3
        0xd55f0005u, 0x040e0501u,  // v_div_fixup_f32 v5, v1, v2, v3
        0xbf810000u,
    };
    const uint64_t addr = 0xA021ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [div] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "v_div_fixup_f32 refuses (no module)");
    CHECK(reason.find("d55f0005") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}

// V_SAD_U8 (VOP3). LLVM gfx10_asm_vop3.s:
//   v_sad_u8 v5, v1, v2, v3 ; encoding: [0x05,0x00,0x5a,0xd5,...]
// Sibling v_sad_u32 is lowered and executed (Tsad1/Tsad2); this pins the
// sub-word sibling to a refusal instead of a silent mistranslation.
TEST(GapOpcodeRefusals, SadU8) {
    static const uint32_t sad[] = {0xd55a0005u, 0x040e0501u};
    Rdna2Inst dec = rdna2_decode_one(sad, 2);
    CHECK(dec.fmt == Rdna2Format::VOP3 && dec.opcode == 0x15au && dec.len_dwords == 2u &&
              isV(dec.dst, 5) && dec.n_src == 3u && isV(dec.src[0], 1) && isV(dec.src[1], 2) &&
              isV(dec.src[2], 3),
          "v_sad_u8 decodes as VOP3 op 0x15a with v5/v1/v2/v3");
    printf("  [sad] fmt=%d op=0x%x len=%u\n", (int)dec.fmt, dec.opcode, dec.len_dwords);
    static const uint32_t code[] = {
        0x7e020381u,              // v_mov_b32 v1, 1
        0x7e040382u,              // v_mov_b32 v2, 2
        0x7e060383u,              // v_mov_b32 v3, 3
        0xd55a0005u, 0x040e0501u,  // v_sad_u8 v5, v1, v2, v3
        0xbf810000u,
    };
    const uint64_t addr = 0xA031ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [sad] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "v_sad_u8 refuses (no module)");
    CHECK(reason.find("d55a0005") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}

// S_MOVRELS_B32 (SOP1 op 0x2e). LLVM gfx10_asm_sop.s:
//   s_movrels_b32 s0, s1 ; encoding: [0x01,0x2e,0x80,0xbe]
// Scalar M0-relative read; the vector v_movrels_b32 sibling is lowered
// (rdna2_emit_alu case 0x43, kernels T16/T19).
TEST(GapOpcodeRefusals, MovrelsB32) {
    static const uint32_t rel[] = {0xbe802e01u};
    Rdna2Inst dec = rdna2_decode_one(rel, 1);
    CHECK(dec.fmt == Rdna2Format::SOP1 && dec.opcode == 0x2eu && dec.len_dwords == 1u &&
              isS(dec.dst, 0) && isS(dec.src[0], 1),
          "s_movrels_b32 decodes as SOP1 op 0x2e with s0/s1");
    static const uint32_t code[] = {
        0xbe820381u,  // s_mov_b32 s2, 1 (M0-adjacent scalar material)
        0xbe802e01u,  // s_movrels_b32 s0, s1
        0xbf810000u,
    };
    const uint64_t addr = 0xA041ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [movrels] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "s_movrels_b32 refuses (no module)");
    CHECK(reason.find("be802e01") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}

// IMAGE_GATHER4 (MIMG op 0x40, DMASK=0xf 2D — the image_sample 0x20 word
// with the opcode field advanced: 0xF0800F08 -> 0xF1000F08).
TEST(GapOpcodeRefusals, ImageGather4) {
    static const uint32_t gat[] = {0xF1000F08u, 0x00A30000u};
    Rdna2Inst dec = rdna2_decode_one(gat, 2);
    CHECK(dec.fmt == Rdna2Format::MIMG && dec.opcode == 0x40u && dec.len_dwords == 2u &&
              isV(dec.dst, 0) && isV(dec.src[0], 0) && isS(dec.src[1], 12) && isS(dec.src[2], 20) &&
              dec.mimg_dmask == 0xfu,
          "image_gather4 decodes as MIMG op 0x40, dmask 0xf, v0/s12/s20");
    printf("  [gather4] fmt=%d op=0x%x len=%u\n", (int)dec.fmt, dec.opcode, dec.len_dwords);
    static const uint32_t code[] = {
        0xF1000F08u,
        0x00A30000u,  // image_gather4 v[0:3], v[0:1], s[12:19], s[20:23]
        0xbf810000u,
    };
    const uint64_t addr = 0xA051ull;
    std::vector<uint32_t> spv = compile_with_addr(code, std::size(code), addr);
    const std::string reason = last_terminal_reject_reason(addr);
    printf("  [gather4] spv_words=%zu reason=%s\n", spv.size(), reason.c_str());
    CHECK(spv.empty(), "image_gather4 refuses (no module)");
    CHECK(reason.find("f1000f08") != std::string::npos,
          "refusal names the gap instruction's words");
    EXPECT_EQ(fails, 0);
}
