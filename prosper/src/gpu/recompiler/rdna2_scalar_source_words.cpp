// scalar_alu_source_words, defined once out of line (#5153). It used to be an inline definition in
// rdna2_cfg_support.hpp, while rdna2_alu_support.hpp declares it non-inline for callers that do not
// include that header. Once #5150 shrank the body, GCC inlined it at every call site and emitted no
// standalone copy, so MinGW's static link found no definition for those callers.
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {
// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            if (in.opcode == kSop1OpcodeGetpcB64) return UINT32_MAX;   // s_getpc has no source
            // S_MOVRELS reads SGPR[src + M0], not SGPR[src] (#4538). The b32 form is lowered as a
            // select over every register from the encoded source up to s105 (rdna2_movrels.cpp),
            // so that whole range is what the instruction may read. Every consumer takes the
            // answer as a COUNT of consecutive words starting at the operand, and all but one
            // refuse when the count overlaps something they care about; the exception is the
            // dispatcher transfer the predicate above exists for. Charging the read one register
            // let a register above the base be proven dead across a merge. The b64 form is not
            // lowered; it gets the same range so nothing reasons about it as one fixed pair.
            // A source that is not an SGPR is refused by the emitter and keeps the default.
            if ((in.opcode == kSop1OpcodeMovrelsB32 || in.opcode == kSop1OpcodeMovrelsB64) &&
                source == 0 && in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
                in.src[0].value <= 105)
                return static_cast<uint32_t>(106 - in.src[0].value);
            [[fallthrough]];
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPC: {
            // Every other scalar ALU source width comes from the instruction-effects table (a B64
            // shift's count, BFM's operands and S_BITREPLICATE's source are B32; S_BCNT*/S_FF*/
            // S_FLBIT*_B64 read a pair). A source the instruction does not have, or an opcode the
            // table cannot classify, is charged a pair: every consumer treats wider as "prove more".
            const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
            if (!effects.known() || source >= 2 || !effects.ssrc_dwords[source]) return 2u;
            return effects.ssrc_dwords[source];
        }
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOPC:
        case Rdna2Format::VOP3: {
            // Every VALU source width comes from the instruction-effects table, derived from the
            // ISA (70648 12.7-12.12) and checked against llvm-mc: a B32 operand is one SGPR, an
            // F64/I64/U64/B64 operand a pair, and a lane mask (SRC2 of v_cndmask_b32 and the
            // carry-in forms) the Wave64 pair; a caller that knows the program is Wave32 asks the
            // wave-aware overload below, which charges it one SGPR (#5202). The MAD_U64/I64 SRC2
            // is a numeric addend pair.
            // It replaces an allowlist of B32
            // VOP3 opcodes over a "two dwords" default, extended one opcode at a time as titles
            // were refused (V_LSHL_ADD_U32 #2790, V_MUL_LO/HI_U32 #2952, V_MED3_F32 #4429,
            // V_BFE_U32 #4848, V_BCNT_U32_B32 #5148): every B32 form outside that list charged a
            // never-read sibling SGPR and could refuse a valid shader (#5146).
            // A source the instruction does not have (the decoder fills SRC2 of every VOP3) is
            // UINT32_MAX, as v_readlane_b32's and v_mbcnt's always were. An encoding the table
            // cannot classify is charged a pair.
            const Rdna2InstructionEffects& effects = rdna2_instruction_effects(in);
            if (!effects.describes_sources()) return 2u;
            if (source >= 3 || !effects.ssrc_dwords[source]) return UINT32_MAX;
            return effects.ssrc_dwords[source];
        }
        default: return 0;
    }
}

bool valu_lane_mask_source(const Rdna2Inst& in, uint32_t source) {
    // SRC2 of v_cndmask_b32_e64 (VOP3 0x101) and of the VOP3B carry-in forms v_add_co_ci_u32,
    // v_sub_co_ci_u32 and v_subrev_co_ci_u32 (VOP3 0x128-0x12a). Their VOP2 twins read VCC
    // implicitly and have no SRC2.
    return in.fmt == Rdna2Format::VOP3 && source == 2 &&
           (in.opcode == 0x101 || (in.opcode >= 0x128 && in.opcode <= 0x12a));
}

uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size) {
    // A lane mask holds one bit per lane. ISA 70648 6.2.3 ("Wave64 use of SGPRs"): a mask source
    // makes "each workitem receive a separate value, and two consecutive SGPRs are read" -- in
    // Wave64. In Wave32 there are 32 lanes and "the upper half of EXEC and VCC are ignored"
    // (2.1), so the mask is ONE SGPR: llvm-mc gfx1030 (wavefrontsize32) prints the operand as a
    // 32-bit register (`v_add_co_ci_u32_e64 v44, vcc_lo, 0, v32, s107`). Charging the pair there
    // reads a neighbour the instruction never reads; when that neighbour is absent the marks call
    // the mask fabricated and refuse a defined carry (#5202). CONFIDENCE: HIGH.
    const uint32_t words = scalar_alu_source_words(in, source);
    if (wave_size == 32 && words == 2 && valu_lane_mask_source(in, source)) return 1u;
    return words;
}

uint32_t scalar_source_mark_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size) {
    // VCC_LO keeps the pair, as a stand-in for a fact the marks do not otherwise carry. Every
    // compute/graphics shell starts with rs.vcc = false, which sreg_word_absent_unmasked counts
    // as a real VCC_LO, although no launch provides VCC. What made a read of that entry VCC hard
    // was the never-written VCC_HI beside it, charged as the second word of the pair. Charging
    // VCC_LO alone would publish data selected by the unprovided entry VCC
    // (Wave32ExecBoundary.ExplicitCndmaskCannotUseUnprovidedVccToPublishCurrentData). So a Wave32
    // VCC_LO mask still asks for VCC_HI until entry-VCC provenance is tracked directly; that only
    // ever refuses more. Every other Wave32 mask source (an SGPR, VCC_HI) is its one word.
    if (valu_lane_mask_source(in, source) && in.src[source].value == 106)
        return scalar_alu_source_words(in, source);
    return scalar_alu_source_words(in, source, wave_size);
}

}   // namespace prosper::gpu
