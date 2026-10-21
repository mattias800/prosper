// Register mark helpers moved from rdna2_to_spirv_internal.hpp.
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu {

bool scalar_is_lane_local(const RegState& rs, int sgpr) {
    if (rs.lane_local_scalars.empty()) return false;
    const auto it = rs.sreg.find(sgpr);
    return it != rs.sreg.end() && rs.lane_local_scalars.count(it->second) != 0;
}

bool has_wave64_mask_half_pair(const RegState& rs, int base) {
    const auto low = rs.sreg_wave64_mask_half.find(base);
    const auto high = rs.sreg_wave64_mask_half.find(base + 1);
    const auto low_index = rs.sreg_wave64_mask_half_index.find(base);
    const auto high_index = rs.sreg_wave64_mask_half_index.find(base + 1);
    return low != rs.sreg_wave64_mask_half.end() && high != rs.sreg_wave64_mask_half.end() &&
           low_index != rs.sreg_wave64_mask_half_index.end() &&
           high_index != rs.sreg_wave64_mask_half_index.end() && low_index->second == 0 &&
           high_index->second == 1;
}

// Expire the Bool and physical-half spellings together. The complete-mask alias is erased only
// when its two dedicated halves prove that it came from adjacent-half promotion; unrelated VCC or
// saved-mask lifetimes remain under their existing architectural transfer functions. A B64 mask
// writer may replace the promoted alias after emit_alu, so its new Bool can be preserved explicitly.
void expire_wave64_mask_half(RegState& rs, int reg, int preserved_pair) {
    for (int base : {reg - 1, reg}) {
        if (base < 0 || base == preserved_pair || !has_wave64_mask_half_pair(rs, base)) continue;
        rs.sreg_bool.erase(base);
        rs.sreg_bool_narrowed.erase(base);
    }
    rs.sreg_wave64_mask_half.erase(reg);
    rs.sreg_wave64_mask_half_index.erase(reg);
}

// The same architectural transfer function as expire_wave64_mask_half, for the ORDINARY saved B64
// mask spelling, and deliberately NARROWER than it: a saved wave mask names the bits currently in a
// physical SGPR pair, so a scalar write over the register it is keyed on ends that lifetime. It is
// not provenance attached to the register number forever. (expire_wave64_mask_half expires on either
// word of the pair; portable mask-slot reloads retain the historical ROOT policy below.)
//
// Without this, `sreg_bool` outlives its register and every later consumer reads a mask the hardware
// no longer holds. Two of them are load-bearing: `operand_bits` rejects an ordinary data read of
// such a word ("a persisted B64 wave mask has no ordinary scalar dword"), and the SOP2 mask lambdas
// would silently substitute the dead mask for a live one. R-Type Delta's sprite vertex shader is the
// worked example (#2783): it saves `s_cselect_b64 s[0:1], exec, 0` in its NGG fetch prologue, then
// rebuilds its PC-relative embedded-table address in that same s[0:1] with `s_getpc_b64` /
// `s_add_u32 s0, lit, s0`, and the whole shader was rejected at that add -- which dropped every
// sprite draw in the title.
//
// `snapshot_saved_b64_masks` must be taken BEFORE emit_alu, because emit_alu materializes the new
// lifetime for the same instruction. The staleness test is "present in the snapshot AND its Bool id
// is unchanged", a PROXY for "this instruction did not publish it": every publisher either stores a
// fresh id or is named by `preserved_pair`. Classifying publishers syntactically instead is not
// sufficient -- `scalar_write_is_b64_mask` knows the SOP1/SOP2/VOPC/VOP3B mask writers, but the
// `vgpr_lane_mask_slots` reload in rdna2_emit_alu.cpp republishes a spilled alias from v_readlane
// with no syntactic marker at all. That reload re-stores an IDENTICAL id whenever the register
// still holds the mask it spilled, which is the ordinary spill-then-reload of one register, so
// record_scalar_write names it in `preserved_pair` too. Before it did, the proxy dropped exactly that
// alias, and the outcome was SILENT, not fail-visible as this comment once claimed: the next
// consumer, a re-spill, read the register through the DATA domain as an untracked zero, so a saved
// mask restored an empty EXEC (#4600 shape 1).
//
// VCC (106/107) is deliberately out of scope, and NOT because it is unreachable -- SGPR-kind
// operands really can carry 106/107 (`sgpr()` masks to 7 bits, rdna2_decode.cpp), and SOPK's
// read-modify-write forms read their own destination through `val(in.dst)`, so
// `s_mulk_i32 vcc_lo, imm` does reach operand_bits with an SGPR-kind 106. What makes VCC different
// is that its mask state is mirrored in `rs.vcc`, and expiring `sreg_bool[106]` without a matching
// policy for `rs.vcc` would leave the two spellings disagreeing. That is a separate change with its
// own risk -- it would also alter established behavior for every `s_bfe_u32 vcc_lo` NGG preamble --
// and no observed defect requires it: reaching the reject through VCC additionally needs a writer
// that overwrites VCC_LO while leaving no scalar SSA value behind, since operand_bits consults
// `rs.sreg` first. Tracked in #2804. Wave32 B32 aliases are likewise left to
// record_scalar_write's own (narrower) rules.
// ---------------------------------------------------------------------------------------------
// Merge-placeholder and memory-pattern marks (#4706): which scalar words may hold the structured
// emitter's fabricated zero, or bits loaded from memory. See RegState::sreg_merge_placeholder.
// Ordinary vector words carry the same marks (#4787), including lane reads, VALU copies,
// predicated writes and structured joins. The dispatcher preserves their MAY relay facts too.

// Whether a DATA read of scalar register `r` may see a fabricated word: marked, or (for an
// ordinary SGPR) absent from `sreg` while no mask covers it, since `operand_bits` reads absence as
// `uconst(0)`. A Bool-domain mask is a real value; its ballot words are materialized where a data
// read needs them. An untracked VCC half (106/107) is the VCC Bool, materialized exactly or
// refused, so it counts only when marked; so do ttmp0-15 (108-123) and M0 (124), whose untracked
// data reads `operand_bits` refuses (#4725 review: both laundered the mark while unscanned).
//
// The explicit mark is tested first, so it always wins. Absence then counts only when nothing real
// stands behind it: a direct-descriptor word in `sreg_input` is the driver's own data, which
// `operand_bits` reads, and every scalar write, merge, loop and dispatcher case erases
// `sreg_input`, so a word still there was never written on any path. scalar_source_marks() asks
// this same question, so a COPY of such a word stays clean too (#4714 review). And a shell that
// seeds no launch state (recompile_coverage) cannot tell never-written from launch data at all.
bool sreg_word_absent_unmasked(const RegState& rs, int r) {
    const int last = rs.vector_wave_size == 32 ? 124 : 105;
    return r >= 0 && r <= last && !rs.sreg.contains(r) && !rs.sreg_bool.contains(r) &&
           !(r == 106 && rs.vcc) &&
           !(r > 0 && rs.sreg_bool.contains(r - 1) && !rs.sreg_bool_b32.contains(r - 1));
}
bool sreg_word_may_be_fabricated(const RegState& rs, int r) {
    if (r < 0 || r > 124) return false;
    if (rs.unavailable_saved_mask_words.contains(r) || rs.sreg_merge_placeholder.contains(r))
        return true;
    return rs.absent_sgpr_reads_fabricated_zero && !rs.sreg_input.contains(r) &&
           sreg_word_absent_unmasked(rs, r);
}
// Whether a source of a Wave64 B32 write into a VCC word is defined on every path, by #4706's
// per-word marks rather than the whole-path `scalar_presence_has_no_placeholders`, which ANY
// structured if clears even when the instruction reads none of the registers it merged (#4846).
// emit_alu accepts it for a low-only write or a complete, definite scalar pair. SCC must itself be
// defined when read explicitly as a 0/1 dword. PPSA28000
// 0x26005ba000 and PPSA29343 0x4580fe3400 run `s_and_b32 vcc_lo, vcc_lo, 12` straight after an if
// that wrote only s[14:15], then `s_cselect_b32 vcc_lo, 4, 5` on that SCC. A marked, absent or
// fabricated word (and every mask operand) is still refused.
bool b32_source_word_defined(const RegState& rs, const Operand& source) {
    switch (source.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal: return true;
        case OperandKind::SGPR:
            return (rs.sreg.contains(source.value) || rs.sreg_input.contains(source.value)) &&
                   !sreg_word_may_be_fabricated(rs, source.value);
        case OperandKind::Special:
            if (source.value == 253) return rs.scc != 0 && !rs.scc_merge_placeholder;
            return source.value >= 106 && source.value <= 124 && rs.sreg.contains(source.value) &&
                   !rs.sreg_merge_placeholder.contains(source.value);
        default: return false;
    }
}
bool b32_sources_defined(const RegState& rs, const Rdna2Inst& in) {
    return b32_source_word_defined(rs, in.src[0]) && b32_source_word_defined(rs, in.src[1]);
}

}   // namespace prosper::gpu
