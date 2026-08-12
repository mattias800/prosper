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

// GFX10.3 scalar ALU opcodes shared by emission and source-lifetime analysis.
inline constexpr uint32_t kSop1OpcodeMovB32 = 0x03;
inline constexpr uint32_t kSop1OpcodeMovB64 = 0x04;
inline constexpr uint32_t kSop1OpcodeCmovB32 = 0x05;
inline constexpr uint32_t kSop1OpcodeCmovB64 = 0x06;
inline constexpr uint32_t kSop1OpcodeNotB32 = 0x07;
inline constexpr uint32_t kSop1OpcodeBrevB32 = 0x0b;
inline constexpr uint32_t kSop1OpcodeBcnt1I32B64 = 0x10;
inline constexpr uint32_t kSop1OpcodeFf1I32B64 = 0x14;
inline constexpr uint32_t kSop1OpcodeFlbitI32B32 = 0x15;
inline constexpr uint32_t kSop1OpcodeFlbitI32B64 = 0x16;
inline constexpr uint32_t kSop1OpcodeBitset0B32 = 0x1b;
inline constexpr uint32_t kSop1OpcodeBitset1B32 = 0x1d;
inline constexpr uint32_t kSop1OpcodeGetpcB64 = 0x1f;
inline constexpr uint32_t kSop1OpcodeSetpcB64 = 0x20;
inline constexpr uint32_t kSop1OpcodeSwappcB64 = 0x21;
inline constexpr uint32_t kSop1OpcodeRfeB64 = 0x22;
inline constexpr uint32_t kSop1OpcodeAndSaveexecB64 = 0x24;
inline constexpr uint32_t kSop1OpcodeXnorSaveexecB64 = 0x2b;
inline constexpr uint32_t kSop1OpcodeQuadmaskB64 = 0x2d;
inline constexpr uint32_t kSop1OpcodeAbsI32 = 0x34;
inline constexpr uint32_t kSop1OpcodeAndn1SaveexecB64 = 0x37;
inline constexpr uint32_t kSop1OpcodeOrn1SaveexecB64 = 0x38;
inline constexpr uint32_t kSop2OpcodeCselectB32 = 0x0a;
inline constexpr uint32_t kSop2OpcodeAddI32 = 0x02;
inline constexpr uint32_t kSop1OpcodeMovreldB32 = 0x30;
inline constexpr uint32_t kSop1OpcodeMovreldB64 = 0x31;
inline constexpr uint32_t kSop1OpcodeMovrelsd2B32 = 0x49;
inline constexpr uint32_t kSop2OpcodeCselectB64 = 0x0b;
inline constexpr uint32_t kSop2OpcodeAndB32 = 0x0e;
inline constexpr uint32_t kSop2OpcodeAndB64 = 0x0f;
inline constexpr uint32_t kSop2OpcodeOrB32 = 0x10;
inline constexpr uint32_t kSop2OpcodeOrB64 = 0x11;
inline constexpr uint32_t kSop2OpcodeXorB32 = 0x12;
inline constexpr uint32_t kSop2OpcodeAndn2B32 = 0x14;
inline constexpr uint32_t kSop2OpcodeOrn2B32 = 0x16;
inline constexpr uint32_t kSop2OpcodeNandB32 = 0x18;
inline constexpr uint32_t kSop2OpcodeNorB32 = 0x1a;
inline constexpr uint32_t kSop2OpcodeNorB64 = 0x1b;
inline constexpr uint32_t kSop2OpcodeXnorB32 = 0x1c;
inline constexpr uint32_t kSop2OpcodeBfmB32 = 0x24;
inline constexpr uint32_t kSop2OpcodeBfmB64 = 0x25;
inline constexpr uint32_t kSop2OpcodeBfeU64 = 0x29;
inline constexpr uint32_t kSopkOpcodeMovkI32 = 0x00;
inline constexpr uint32_t kSopkOpcodeCmovkI32 = 0x02;
inline constexpr uint32_t kSopkOpcodeCmpkFirst = 0x03;
inline constexpr uint32_t kSopkOpcodeCmpkLast = 0x0e;
inline constexpr uint32_t kSopkOpcodeAddkI32 = 0x0f;
inline constexpr uint32_t kSopkOpcodeMulkI32 = 0x10;
inline constexpr uint32_t kSopkOpcodeSetregB32 = 0x13;
inline constexpr uint32_t kSopkOpcodeWaitcntVscnt = 0x17;
inline constexpr uint32_t kSopkOpcodeWaitcntLgkmcnt = 0x1a;
inline constexpr uint32_t kSopkOpcodeCallB64 = 0x16;
inline constexpr uint32_t kSopkOpcodeSubvectorLoopBegin = 0x1b;
inline constexpr uint32_t kSopkOpcodeSubvectorLoopEnd = 0x1c;
inline constexpr uint32_t kSoppOpcodeBranch = 0x02;
inline constexpr uint32_t kSoppOpcodeCbranchScc0 = 0x04;
inline constexpr uint32_t kSoppOpcodeCbranchExecz = 0x08;
inline constexpr uint32_t kSoppOpcodeCbranchExecnz = 0x09;
inline constexpr uint32_t kSoppOpcodeBarrier = 0x0a;
inline constexpr uint32_t kSoppOpcodeTrap = 0x12;
inline constexpr uint32_t kSoppOpcodeCbranchCdbgsys = 0x17;
inline constexpr uint32_t kSoppOpcodeCbranchCdbguser = 0x18;
inline constexpr uint32_t kSoppOpcodeCbranchCdbgsysOrUser = 0x19;
inline constexpr uint32_t kSoppOpcodeCbranchCdbgsysAndUser = 0x1a;

inline constexpr bool sopp_opcode_is_direct_branch(uint32_t opcode) {
    return opcode == kSoppOpcodeBranch ||
           (opcode >= kSoppOpcodeCbranchScc0 && opcode <= kSoppOpcodeCbranchExecnz);
}

// GFX10.3 scalar-memory load opcodes shared by provenance analysis and emission. These are
// compile-time constants, so naming them has no runtime cost; it keeps width/family tests from
// becoming opaque hexadecimal comparisons at each consumer.
inline constexpr uint32_t kSmemOpcodeLoadDword          = 0x00;
inline constexpr uint32_t kSmemOpcodeLoadDwordX2        = 0x01;
inline constexpr uint32_t kSmemOpcodeLoadDwordX4        = 0x02;
inline constexpr uint32_t kSmemOpcodeLoadDwordX8        = 0x03;
inline constexpr uint32_t kSmemOpcodeLoadDwordX16       = 0x04;
inline constexpr uint32_t kSmemOpcodeBufferLoadDword    = 0x08;
inline constexpr uint32_t kSmemOpcodeBufferLoadDwordX2  = 0x09;
inline constexpr uint32_t kSmemOpcodeBufferLoadDwordX4  = 0x0a;
inline constexpr uint32_t kSmemOpcodeBufferLoadDwordX8  = 0x0b;
inline constexpr uint32_t kSmemOpcodeBufferLoadDwordX16 = 0x0c;

inline constexpr bool smem_opcode_is_buffer_load(uint32_t opcode) {
    return opcode >= kSmemOpcodeBufferLoadDword &&
           opcode <= kSmemOpcodeBufferLoadDwordX16;
}

inline constexpr bool sop2_is_b32_logical(uint32_t opcode) {
    return opcode == kSop2OpcodeAndB32 || opcode == kSop2OpcodeOrB32 ||
           opcode == kSop2OpcodeXorB32 || opcode == kSop2OpcodeAndn2B32 ||
           opcode == kSop2OpcodeOrn2B32 || opcode == kSop2OpcodeNandB32 ||
           opcode == kSop2OpcodeNorB32 || opcode == kSop2OpcodeXnorB32;
}

// GFX10.3 VOP1 opcodes used outside the decoder itself. Naming cross-component opcodes at this
// boundary keeps dispatcher admission and ALU emission on one compile-time constant.
inline constexpr uint32_t kVop1OpcodeFfbhU32 = 0x39;
inline constexpr uint32_t kVop1OpcodeFfblB32 = 0x3a;
inline constexpr uint32_t kVop1OpcodeMovreldB32 = 0x42;

// GFX10.3 VOP3 opcodes shared by source-lifetime and exact packet-shape proofs. The shift-reverse
// B64 pair consumes a B32 shift count followed by a B64 value; the arithmetic forms consume three
// independent B32 operands.
inline constexpr uint32_t kVop3OpcodeLshlrevB64 = 0x2ff;
inline constexpr uint32_t kVop3OpcodeLshrrevB64 = 0x300;
inline constexpr uint32_t kVop3OpcodeLshlAddU32 = 0x346;
inline constexpr uint32_t kVop3OpcodeAdd3U32 = 0x36d;
inline constexpr uint32_t kVop3OpcodeAndOrB32 = 0x371;

// GFX10.3 DS cross-lane/float-atomic opcodes shared by decode-side write accounting, control-flow
// admission, and emission. These inline constants compile to the same immediate comparisons as raw
// literals.
inline constexpr uint32_t kDsOpcodeBpermuteB32 = 0xb3;
inline constexpr uint32_t kDsOpcodeMinF32 = 0x12;
inline constexpr uint32_t kDsOpcodeMaxF32 = 0x13;

// GFX10.3 MUBUF opcodes shared by descriptor discovery and emission. These inline constants compile
// to the same immediate comparisons as the old literals.
inline constexpr uint32_t kMubufOpcodeStoreDword   = 0x1c;
inline constexpr uint32_t kMubufOpcodeStoreDwordX2 = 0x1d;
inline constexpr uint32_t kMubufOpcodeStoreDwordX4 = 0x1e;
inline constexpr uint32_t kMubufOpcodeStoreDwordX3 = 0x1f;

// GFX10.3 unsigned 64-bit scalar equality compare. Kept with the other scalar opcodes because the
// guarded-null resource proof and scalar emission must identify the same instruction.
inline constexpr uint32_t kSopcOpcodeCmpEqU64 = 0x12;

inline constexpr uint32_t kMubufOpcodeLoadDword         = 0x0c;
inline constexpr uint32_t kMubufOpcodeAtomicSwap        = 0x30;
inline constexpr uint32_t kMubufOpcodeAtomicCompareSwap = 0x31;
inline constexpr uint32_t kMubufOpcodeAtomicAdd         = 0x32;
inline constexpr uint32_t kMubufOpcodeAtomicSub         = 0x33;
inline constexpr uint32_t kMubufOpcodeAtomicConditionalSub = 0x34;
inline constexpr uint32_t kMubufOpcodeAtomicSmin        = 0x35;
inline constexpr uint32_t kMubufOpcodeAtomicUmin        = 0x36;
inline constexpr uint32_t kMubufOpcodeAtomicSmax        = 0x37;
inline constexpr uint32_t kMubufOpcodeAtomicUmax        = 0x38;
inline constexpr uint32_t kMubufOpcodeAtomicAnd         = 0x39;
inline constexpr uint32_t kMubufOpcodeAtomicOr          = 0x3a;
inline constexpr uint32_t kMubufOpcodeAtomicXor         = 0x3b;
inline constexpr uint32_t kMubufOpcodeAtomicInc         = 0x3c;
inline constexpr uint32_t kMubufOpcodeAtomicDec         = 0x3d;
inline constexpr uint32_t kMubufOpcodeAtomicFmin        = 0x3f;
inline constexpr uint32_t kMubufOpcodeAtomicFmax        = 0x40;
inline constexpr uint32_t kMubufOpcodeAtomicSwapX2      = 0x50;
inline constexpr uint32_t kMubufOpcodeAtomicOrX2        = 0x5a;

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
struct Rdna2Inst;
// True when a VOPC opcode is the `v_cmpx_*` form, which writes EXEC instead of a VCC/SGPR mask.
// Every one sits at its `v_cmp_*` counterpart + 0x10, so `opcode - 0x10` recovers the base compare.
// The six windows and the two invalid holes are enumerated (and sourced) at the definition in
// rdna2_decode.cpp. This is a property of the ENCODING, so it is shared: the decoder's SDWA
// admission and the recompiler's EXEC/mask bookkeeping must not carry separate copies (#2120).
bool vopc_is_cmpx(uint32_t opcode);
// True for every decoded instruction that can explicitly or implicitly write EXEC. This is shared
// by control-flow analysis and instruction-scoped value proofs: overlooking a mask mutation can
// turn a predicated VALU definition into a false all-lanes fact.
bool rdna2_instruction_may_change_exec(const Rdna2Inst& in);
// Number of consecutive data VGPRs an instruction writes from its decoded destination. This
// inventory is shared by control-flow analyses and instruction-scoped value proofs so scalar VALU
// results are not mistaken for four-register memory payloads, while actual wide results still
// invalidate every register they define. Known store encodings return zero because their VDATA
// field is a source. Appended TFE status and dynamic destinations such as v_movreld_b32 are separate:
// consumers must also use `rdna2_tfe_status_vgpr` and handle dynamic destinations fail-closed.
uint32_t rdna2_vgpr_write_count(const Rdna2Inst& in);
// Number of consecutive VGPR dwords consumed by one decoded source operand. Most encoded vector
// sources name one dword; a small set of B64 operations names a pair through one base operand.
// Keeping this beside destination accounting makes CFG register discovery and shader sizing agree.
uint32_t rdna2_vgpr_source_span(const Rdna2Inst& in, uint32_t source_index);
// Appended TFE status destination, or -1 when the instruction has none. Unlike the consecutive data
// result above, a store's decoded VDATA is a source prefix and only the trailing status is written.
// This remains exact for decoded MTBUF forms that emission rejects, including packed-D16 forms.
int rdna2_tfe_status_vgpr(const Rdna2Inst& in);
// Width of the complete consecutive VGPR field rooted at `dst`, including VDATA sources on stores
// and an appended TFE status destination. Register-file sizing needs this broader footprint even
// though dataflow invalidation must distinguish sources from writes.
uint32_t rdna2_vgpr_destination_span(const Rdna2Inst& in);
// True for the VOP1 f16 unary family: one f16 operand in, one f16 result out, no side effects —
// 0x54-0x58 (rcp/sqrt/rsq/log/exp) and 0x5B-0x61 (floor/ceil/trunc/rndne/fract/sin/cos). The two
// FREXP opcodes 0x59/0x5A sit inside that numeric span and are deliberately excluded: 0x59 returns a
// mantissa and 0x5A returns an **i16** exponent, so neither is the shape the shared lowering
// implements. Like `vopc_is_cmpx`, this is a property of the ENCODING and is shared rather than
// copied: the decoder's SDWA admission and the recompiler's `emit_f16_unary` must have exactly one
// domain between them, or an opcode gains support in one form and not the other (#2013).
bool vop1_is_f16_unary(uint32_t opcode);

struct Rdna2Inst {
    Rdna2Format fmt = Rdna2Format::Unknown;
    uint32_t    pc = 0;            // dword offset from the start of the stream
    uint32_t    words[5] = {0, 0, 0, 0, 0}; // instruction dwords (not incl. a trailing literal); [2]..[4]
                                            // hold MIMG NSA address dwords (0 for every other encoding)
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
    // DPP16 QUAD_PERM/ROW_SHR (#273/#1390): an admitted operation with no neg/abs/FI. The
    // recompiler lowers full-mask permutations through stage-appropriate lane operations; exact
    // partial-mask shapes keep their row/bank fields so a narrow stage model can preserve masked
    // destinations. src[0] holds the REAL source VGPR (dword1[7:0]).
    bool        has_dpp = false;
    uint16_t    dpp_ctrl = 0;
    bool        dpp_bound_ctrl = false;
    uint8_t     dpp_bank_mask = 0;
    uint8_t     dpp_row_mask = 0;
    // V_PERMLANE16_B32 / V_PERMLANEX16_B32 overload OPSEL[0:1] as Fetch-Inactive and
    // BOUND_CTRL. Keep these separate from DPP's control word: permlane is a native VOP3 op whose
    // 64-bit selector comes from SRC1:SRC2.
    bool        permlane_fetch_inactive = false;
    bool        permlane_bound_ctrl = false;
    // SDWA sub-word selects (#273): WORD_0/WORD_1 dst/src selects with UNUSED_PRESERVE — the f16
    // half-packing idiom (DOLL's box-blur: `v_mul_f16_sdwa … dst_sel:WORD_1 preserve` then
    // `v_mov_b32_sdwa … dst_sel:WORD_0 src0_sel:WORD_1`). 6 = DWORD (the default / no select);
    // 4/5 = WORD_0/WORD_1. Only combos the recompiler models clear has_modifier.
    uint8_t     sdwa_dst_sel = 6, sdwa_dst_unused = 0, sdwa_src0_sel = 6, sdwa_src1_sel = 6;
    // SDWA S0_SEXT / S1_SEXT (dword1 bits 19 and 27): the selected sub-dword source field is
    // SIGN-extended to 32 bits rather than zero-extended. A different operation from the
    // zero-extending form, so these are only set for encodings whose lowering honours them —
    // `v_mov_b32_sdwa` into a full DWORD destination and the integer VOP2 SDWA ops (Sonic Racing:
    // CrossWorlds' compute kernels, #2013). Every other SDWA admission still requires the bits
    // clear, which keeps an unmodelled sign extension fail-visible. A set bit is only ever admitted
    // alongside a real sub-dword select: SEXT of a full DWORD field is not exercised by any live
    // encoding here, so it stays rejected rather than assumed to be a no-op.
    bool        sdwa_src0_sext = false, sdwa_src1_sext = false;

    // Decoded operands. `opcode` is the format-local opcode; `dst` the destination (or VDATA source
    // base for memory stores); `src[0..n_src-1]` the remaining sources. simm16 holds the signed
    // 16-bit immediate for SOPK/SOPP. Most memory/interp/export formats populate the same fields.
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

    // MUBUF/MTBUF flags (ISA Table 98): GLC (bit 14) — for atomics, "return pre-op value to VGPR";
    // DLC (bit 15) — device-level cache policy for ordinary loads; LDS (bit 16, MUBUF only) —
    // transfer between LDS and memory instead of VGPRs (rejected until modeled).
    bool     mubuf_glc = false;
    bool     mubuf_dlc = false;
    bool     mubuf_lds = false;
    // MTBUF carries the GFX10 combined 7-bit BUF_FMT at dword0[25:19], just like a Gen5 V#.
    // Interpreting these bits as the older PS4 DFMT/NFMT split maps valid gfx1030 formats to the
    // wrong type (for example 32_FLOAT is combined format 22).
    uint32_t mtbuf_format = 0;
    // MTBUF Texel Fault Enable (instruction bit 55 / dword1 bit 23) writes a status VGPR after the
    // data results. Kept explicit so the recompiler can reject until that observable write is modeled.
    bool     mtbuf_tfe = false;
    // FLAT/GLOBAL/SCRATCH share the 0x37 encoding. Segment 0=flat, 1=scratch, 2=global; OFFSET is a
    // signed 12-bit byte immediate stored sign-extended in `literal`. src[0] is VADDR (None for the
    // canonical scratch `off, sN` form), src[1] is the seven-bit SADDR field, and dst is VDST for
    // loads or VDATA for stores. Cache-policy flags do not change private scratch data semantics;
    // LDS requests a memory-to/from-LDS transfer and must not be treated as an ordinary VGPR access.
    uint32_t flat_segment = 0;
    bool     flat_glc = false;
    bool     flat_slc = false;
    bool     flat_dlc = false;
    bool     flat_lds = false;
    // DS-only: GDS flag. llvm-mc gfx1030 round-trip places it at dword0 bit 17 (ds_add_u32 gds =
    // 0xd8020000 vs 0xd8000000; Table 94's "GDS [16]" is a GFX9-era erratum — it also misplaces
    // OP). Bit 16 is likewise captured so an unknown flag rejects rather than silently running
    // against workgroup LDS with device-global (GDS) semantics expected.
    bool     ds_gds = false;
    // 2D=1, 3D=2, Cube=3, 1D_ARRAY=4, 2D_ARRAY=5, ...), plus every MIMG control field from ISA
    // Table 100. Keeping the controls explicit is correctness-critical: unlike VOP encodings, MIMG
    // never sets `has_modifier`, and silently ignoring A16/TFE/etc. changes address/result layouts.
    // VDATA is in `dst`, VADDR in src[0], SRSRC (T# base SGPR) in src[1], SSAMP (S# base SGPR) in
    // src[2]. `mimg_reserved` covers dword0 bits 6/14 and dword1 bits 26..29.
    uint32_t mimg_nsa   = 0;
    uint32_t mimg_dmask = 0;
    uint32_t mimg_dim   = 0;
    bool     mimg_unorm = false;
    bool     mimg_glc   = false;
    bool     mimg_dlc   = false;
    bool     mimg_r128  = false;
    bool     mimg_tfe   = false;
    bool     mimg_lwe   = false;
    bool     mimg_slc   = false;
    bool     mimg_a16   = false;
    bool     mimg_d16   = false;
    bool     mimg_reserved = false;

    // VINTRP-only: the interpolated attribute (parameter location) and its component channel (0..3).
    // opcode = p1(0)/p2(1)/mov(2); dst = vdst; src[0] = vsrc (the i/j barycentric VGPR).
    uint32_t vintrp_attr = 0;
    uint32_t vintrp_chan = 0;

    // Packed/mixed f16 selector bitmasks. For VOP3P mix ops, bit k of vop3p_opsel_hi means source k
    // reads as an f16 half selected by vop3p_opsel[k]; packed add/mul use the two masks for the low
    // and high results. VOP3 scalar-f16 operations reuse vop3p_opsel[2:0] for their sources and bit
    // 3 for dst (currently v_fma_f16 and v_max3_f16).
    // NEG lands in src_neg[], NEG_HI (abs for mix ops) in src_abs[], and CLAMP in clamp.
    uint8_t vop3p_opsel = 0, vop3p_opsel_hi = 0;
    uint8_t vop3p_neg_hi = 0;   // packed add/mul: per-source negate for the HIGH f16 result
};

// Decode the single instruction at code[0..]; `max_dwords` bounds the read. On a truncated/unknown
// encoding, returns fmt=Unknown with len_dwords clamped so a walker still terminates.
Rdna2Inst rdna2_decode_one(const uint32_t* code, size_t max_dwords);

// Packet half of GTA V's optional-null buffer convention: one plain idxen-only RAW dword load with
// no instruction offset, SOFFSET, cache/status, LDS, or reserved controls. Descriptor provenance and
// launch/index geometry are independent proofs supplied by the front half.
bool rdna2_optional_null_raw_load_shape(const Rdna2Inst& in);

// Walk from code[0], appending each decoded instruction to `out`, until S_ENDPGM, an Unknown
// encoding, or the end of the buffer. Returns the number of dwords consumed.
size_t rdna2_walk(const uint32_t* code, size_t dwords, std::vector<Rdna2Inst>& out);

// Exact IMAGE_LOAD_MIP / IMAGE_STORE_MIP packet subset whose mip operand may be specialized after
// an independent value proof. Returns the dimension-specific mip VGPR when requested. This checks
// only packet shape; callers must still prove that VGPR is zero and that the resource has one
// materialized, uncompressed mip.
bool rdna2_mimg_zero_mip_shape(const Rdna2Inst& in, uint32_t* mip_vgpr = nullptr);

// Minimum byte range touched by immediate s_load_dword[xN] operations whose 64-bit SBASE begins at
// `sgpr_base`. Unlike s_buffer_load, s_load consumes an address pair rather than a bounded V#; callers
// use this to size pointer-backed user-data tables from the shader's actual accesses.
uint32_t rdna2_sload_required_bytes(const uint32_t* code, size_t dwords, uint32_t sgpr_base);

} // namespace prosper::gpu
