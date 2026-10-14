// scalar_alu_source_words, defined once out of line (#5153). It used to be an inline definition in
// rdna2_cfg_support.hpp, while rdna2_alu_support.hpp declares it non-inline for callers that do not
// include that header. Once #5150 shrank the body, GCC inlined it at every call site and emitted no
// standalone copy, so MinGW's static link found no definition for those callers.
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {
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
            // F64/I64/U64/B64 operand a pair, and a lane mask (SRC2 of v_cndmask_b32, the
            // carry-in forms and v_mad_u64_u32) the Wave64 pair. It replaces an allowlist of B32
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
}   // namespace prosper::gpu
