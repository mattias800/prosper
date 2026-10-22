#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>

namespace prosper::gpu {

// Whether `in` is an s_movrels_b32 the lowering can emit (rdna2_movrels.cpp). What it writes is
// then a scalar word: the select over what each candidate holds. A candidate in another domain (a
// saved mask half, an entry-M0 token) refuses the whole instruction. A candidate with no tracked
// value does NOT refuse; it contributes the zero placeholder every scalar read uses, so the
// result is exactly as good as the candidate M0 selects, and wrong when that one was untracked.
//
// The dispatcher's scalar-word transfer asks this INSTEAD of charging the candidates, and the
// reason is what failing that transfer means. It is not a refusal: the destination is dropped
// from the words a case boundary reloads, and reads the zero placeholder afterwards. So dropping
// can only turn a right result into zero, never a wrong one into a refusal. Charged the base
// alone, a table whose first register was never written lost its result at the next join.
// Charged the whole range s[base..105], every result did, since almost no program defines all of
// those words. scalar_alu_source_words describes the read range for refusing proofs; this is
// for the one that drops.
bool s_movrels_b32_result_is_scalar_data(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovrelsB32 &&
           in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
           in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 && in.src[0].value <= 105;
}
// Scalar instructions whose encoded destination is also an implicit source. These reads must be
// observed before writer transfer functions invalidate the old lifetime. Every SOPK form whose SDST
// field is READ belongs here, not merely the arithmetic band: none of them decodes a src operand at
// all, so a caller scanning `n_src` sees nothing for any of them. SOPK has no decoded src
// operands at all; conditional moves preserve the old destination on one SCC outcome, comparisons
// consume SDST without writing it, and ADDK/MULK are ordinary read-modify-write operations. SOP1
// conditional moves and bitset operations have an explicit source too, but it is the replacement
// value/bit index rather than the old destination.
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::SGPR) return 0;
    if (in.fmt == Rdna2Format::SOPK) {
        if (in.opcode == 0x02 ||   // s_cmovk_i32
            (in.opcode >= 0x03 && in.opcode <= 0x10) ||   // s_cmpk_*, s_addk, s_mulk
            // s_setreg_b32: the encoded SDST field is the SOURCE SGPR, which rdna2_emit_alu.cpp's
            // own handler says in as many words. It reads the register and writes none.
            in.opcode == 0x13 ||
            // s_subvector_loop_begin/end read SDST as the loop counter.
            in.opcode == 0x1b || in.opcode == 0x1c)
            return 1;
        return 0;
    }
    if (in.fmt != Rdna2Format::SOP1) return 0;
    switch (in.opcode) {
        case kSop1OpcodeCmovB32: return 1;
        case kSop1OpcodeCmovB64: return 2;
        case kSop1OpcodeBitset0B32: return 1;
        case 0x1c: return 2;   // s_bitset0_b64
        case kSop1OpcodeBitset1B32: return 1;
        case 0x1e: return 2;   // s_bitset1_b64
        default: return 0;
    }
}

}   // namespace prosper::gpu
