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
    VOP3P,                             // packed/mixed-precision 3-operand VALU (v_fma_mix*, v_pk_*)
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
    // True whenever an SDWA control word was present (src0 field 0xF9), including the modeled
    // subsets that clear has_modifier. Stage 2 needs this to distinguish the plain e32 encoding
    // (dst_sel defaults to DWORD=6, hardware PRESERVES the unwritten f16 half) from the SDWA
    // DWORD+UNUSED_PAD form (same field values, hardware ZERO-fills) — ISA Table 88 DST_U.
    bool        has_sdwa = false;
    // DPP16 QUAD_PERM (#273): a full-quad permute (ctrl < 0x100, row/bank masks 0xf, no neg/abs/FI) —
    // the manual-ddx/ddy idiom. The recompiler lowers it via screen-space derivatives (fragment only);
    // src[0] holds the REAL source VGPR (dword1[7:0]) and dpp_ctrl the 8-bit lane selector.
    bool        has_dpp = false;
    uint16_t    dpp_ctrl = 0;
    // SDWA sub-word selects (#273): WORD_0/WORD_1 dst/src selects with UNUSED_PRESERVE — the f16
    // half-packing idiom (DOLL's box-blur: `v_mul_f16_sdwa … dst_sel:WORD_1 preserve` then
    // `v_mov_b32_sdwa … dst_sel:WORD_0 src0_sel:WORD_1`). 6 = DWORD (the default / no select);
    // 4/5 = WORD_0/WORD_1. Only combos the recompiler models clear has_modifier.
    uint8_t     sdwa_dst_sel = 6, sdwa_dst_unused = 0, sdwa_src0_sel = 6, sdwa_src1_sel = 6;

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

    // MUBUF-only flags (ISA Table 98): GLC (bit 14) — for atomics, "return pre-op value to VGPR";
    // LDS (bit 16) — transfer between LDS and memory instead of VGPRs (rejected until modeled).
    bool     mubuf_glc = false;
    bool     mubuf_lds = false;
    // MTBUF carries the GFX10 combined 7-bit BUF_FMT at dword0[25:19], just like a Gen5 V#.
    // Interpreting these bits as the older PS4 DFMT/NFMT split maps valid gfx1030 formats to the
    // wrong type (for example 32_FLOAT is combined format 22).
    uint32_t mtbuf_format = 0;
    // MTBUF Texel Fault Enable (instruction bit 55 / dword1 bit 23) writes a status VGPR after the
    // data results. Kept explicit so the recompiler can reject until that observable write is modeled.
    bool     mtbuf_tfe = false;
    // DS-only: GDS flag. llvm-mc gfx1030 round-trip places it at dword0 bit 17 (ds_add_u32 gds =
    // 0xd8020000 vs 0xd8000000; Table 94's "GDS [16]" is a GFX9-era erratum — it also misplaces
    // OP). Bit 16 is likewise captured so an unknown flag rejects rather than silently running
    // against workgroup LDS with device-global (GDS) semantics expected.
    bool     ds_gds = false;
    // 2D=1, 3D=2, Cube=3, 1D_ARRAY=4, 2D_ARRAY=5, ...), and the unnormalized-coordinate flag. VDATA is
    // in `dst`, VADDR in src[0], SRSRC (T# base SGPR) in src[1], SSAMP (S# base SGPR) in src[2].
    uint32_t mimg_dmask = 0;
    uint32_t mimg_dim   = 0;
    bool     mimg_unorm = false;

    // VINTRP-only: the interpolated attribute (parameter location) and its component channel (0..3).
    // opcode = p1(0)/p2(1)/mov(2); dst = vdst; src[0] = vsrc (the i/j barycentric VGPR).
    uint32_t vintrp_attr = 0;
    uint32_t vintrp_chan = 0;

    // Packed/mixed f16 selector bitmasks. For VOP3P mix ops, bit k of vop3p_opsel_hi means source k
    // reads as an f16 half selected by vop3p_opsel[k]; packed add/mul use the two masks for the low
    // and high results. VOP3 v_fma_f16 reuses vop3p_opsel[2:0] for its sources and bit 3 for dst.
    // NEG lands in src_neg[], NEG_HI (abs for mix ops) in src_abs[], and CLAMP in clamp.
    uint8_t vop3p_opsel = 0, vop3p_opsel_hi = 0;
    uint8_t vop3p_neg_hi = 0;   // packed add/mul: per-source negate for the HIGH f16 result
};

// Decode the single instruction at code[0..]; `max_dwords` bounds the read. On a truncated/unknown
// encoding, returns fmt=Unknown with len_dwords clamped so a walker still terminates.
Rdna2Inst rdna2_decode_one(const uint32_t* code, size_t max_dwords);

// Walk from code[0], appending each decoded instruction to `out`, until S_ENDPGM, an Unknown
// encoding, or the end of the buffer. Returns the number of dwords consumed.
size_t rdna2_walk(const uint32_t* code, size_t dwords, std::vector<Rdna2Inst>& out);

// Minimum byte range touched by immediate s_load_dword[xN] operations whose 64-bit SBASE begins at
// `sgpr_base`. Unlike s_buffer_load, s_load consumes an address pair rather than a bounded V#; callers
// use this to size pointer-backed user-data tables from the shader's actual accesses.
uint32_t rdna2_sload_required_bytes(const uint32_t* code, size_t dwords, uint32_t sgpr_base);

} // namespace prosper::gpu
