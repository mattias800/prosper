// rdna2_decode.hpp — walk an RDNA2 (gfx10.3) shader binary instruction by instruction.
//
// First stage of the RDNA2 -> SPIR-V shader recompiler (the piece that will replace the placeholder
// SPIR-V in test_vulkan_triangle). Before we can translate instructions we must be able to *walk*
// the ISA stream: classify each instruction's encoding format and compute its length in dwords
// (accounting for inline 32-bit literal constants and the 2-dword formats). Opcode semantics /
// operand decode / SPIR-V emission come in later stages.
//
// Encoding classification follows the RDNA2 ("next-gen") dispatch in Kyty ShaderParse.cpp and the
// AMD gfx10 ISA docs. This is pure (no I/O), so it is unit-tested against instructions assembled by
// llvm-mc for gfx1030 (test_rdna2_decode).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace prosper::gpu {

enum class Rdna2Format : uint8_t {
    SOP2, SOP1, SOPK, SOPC, SOPP,      // scalar ALU / constant / program-flow
    SMEM,                              // scalar memory
    VOP2, VOP1, VOPC, VOP3,           // vector ALU
    VINTRP,                            // interpolation
    DS, MUBUF, MTBUF, MIMG, FLAT,      // LDS / buffer / image / flat memory
    EXP,                               // export (render target / position)
    Unknown,
};

// An operand's kind + payload. For SGPR/VGPR, `value` is the register number; for InlineInt, the
// signed integer constant (-16..64); for InlineFloat, the AMD encoding code (240..248, see
// inline_float_value); for Literal, the constant is in Rdna2Inst::literal; for Special, the field
// code (VCC/EXEC/M0/...). See the AMD gfx10 ISA "SSRC/SRC operand" table.
enum class OperandKind : uint8_t { None, SGPR, VGPR, InlineInt, InlineFloat, Literal, Special };
struct Operand {
    OperandKind kind = OperandKind::None;
    int32_t     value = 0;
};

// Decode an AMD gfx10 source-operand field (8-bit scalar SSRC, or 9-bit vector SRC where 256..511 =
// VGPR). Pure; exposed for reuse/testing.
Operand decode_src_field(uint32_t field);
// The float value an InlineFloat operand encodes (0.5, 1.0, ... , 1/(2*pi)); 0 for non-float codes.
float inline_float_value(uint32_t code);

struct Rdna2Inst {
    Rdna2Format fmt = Rdna2Format::Unknown;
    uint32_t    pc = 0;            // dword offset from the start of the stream
    uint32_t    words[4] = {0, 0, 0, 0}; // the instruction dwords (not incl. a trailing literal); [2]/[3]
                                         // hold MIMG NSA extra address dwords (0 for every other encoding)
    uint32_t    len_dwords = 1;    // total length incl. any inline literal
    bool        has_literal = false;
    uint32_t    literal = 0;       // the inline 32-bit constant, if has_literal
    bool        is_end = false;    // S_ENDPGM
    bool        has_modifier = false;  // VOP SDWA/DPP form (2nd dword is a control word, not a literal);
                                       // decoded for correct length, but the recompiler rejects it
                                       // (sub-dword select / cross-lane semantics are not modeled)

    // Decoded operands (filled for the ALU formats: SOP1/2/K, VOP1/2/C, VOP3). `opcode` is the
    // format-local opcode; `dst` the destination; `src[0..n_src-1]` the sources. simm16 holds the
    // signed 16-bit immediate for SOPK/SOPP. Memory/interp/export formats leave these unset (fmt only).
    uint32_t opcode = 0;
    Operand  dst;
    Operand  src[4];        // up to 4 (EXP has 4 VGPR sources; VOP3 uses 3)
    uint8_t  n_src = 0;
    int32_t  simm16 = 0;

    // VOP3 float source modifiers (per source): abs (bits[10:8] of dword0) then neg (bits[63:61]). The
    // recompiler applies OpFAbs then OpFNegate for float ops. CLAMP/OMOD (which we don't model) instead
    // set has_modifier so the instruction is rejected rather than miscomputed. VOP2 has no source mods.
    bool     src_neg[4] = {false, false, false, false};
    bool     src_abs[4] = {false, false, false, false};
    // VOP3 output modifiers (float ops): CLAMP = saturate result to [0,1] (dword0[15]); OMOD scales the
    // result ×2 (1) / ×4 (2) / ×0.5 (3) (dword1[28:27]). Applied after the op, before writing dst.
    bool     clamp = false;
    uint8_t  omod  = 0;
    // VOP3B scalar destination (carry-out for v_add/sub_co_ci_u32 etc.): dword0[14:8]. Only set for the
    // VOP3B carry ops; kind==None otherwise. (VOP3B reuses the abs-modifier bits for sdst, so the decode
    // clears src_abs for these opcodes.)
    Operand  sdst;

    // EXP-only: export target (MRT0=0..7, MRTZ=8, NULL=9, POS0=12..15, PARAM0=32..) and the 4-bit
    // per-component enable mask. The 4 exported VGPRs are in src[0..3].
    uint32_t exp_target = 0;
    uint32_t exp_en = 0;
    bool     exp_compr = false;     // COMPR: the 4 channels are two f16x2 pairs in src[0] (r,g) / src[1] (b,a)

    // MIMG-only: destination component mask (dmask), image dimensionality (SQ_RSRC_IMG dim: 1D=0,
    // 2D=1, 3D=2, Cube=3, 1D_ARRAY=4, 2D_ARRAY=5, ...), and the unnormalized-coordinate flag. VDATA is
    // in `dst`, VADDR in src[0], SRSRC (T# base SGPR) in src[1], SSAMP (S# base SGPR) in src[2].
    uint32_t mimg_dmask = 0;
    uint32_t mimg_dim   = 0;
    bool     mimg_unorm = false;

    // VINTRP-only: the interpolated attribute (parameter location) and its component channel (0..3).
    // opcode = p1(0)/p2(1)/mov(2); dst = vdst; src[0] = vsrc (the i/j barycentric VGPR).
    uint32_t vintrp_attr = 0;
    uint32_t vintrp_chan = 0;
};

// Decode the single instruction at code[0..]; `max_dwords` bounds the read. On a truncated/unknown
// encoding, returns fmt=Unknown with len_dwords clamped so a walker still terminates.
Rdna2Inst rdna2_decode_one(const uint32_t* code, size_t max_dwords);

// Walk from code[0], appending each decoded instruction to `out`, until S_ENDPGM, an Unknown
// encoding, or the end of the buffer. Returns the number of dwords consumed.
size_t rdna2_walk(const uint32_t* code, size_t dwords, std::vector<Rdna2Inst>& out);

} // namespace prosper::gpu
