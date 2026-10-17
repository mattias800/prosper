// vopc_mask_write.cpp — the lane-mask half of every VOPC compare (ADR 0012).
//
// A compare's per-lane test result (`cmp`, a Bool) is opcode-specific and comes from ir/ops. What
// the instruction then does with it is the same for all 193 compares, and lives here: a V_CMPX
// narrows EXEC; every other compare writes EXEC & cmp to VCC or to its explicit SGPR destination,
// with that destination's mask-domain bookkeeping. Moved verbatim from emit_alu's VOPC case.
#include "gpu/recompiler/ir/lower/vopc_mask_write.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_mask_write.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

namespace prosper::gpu::ir {

// A compare result is a per-lane mask. The e64/SDWAB form can target an SGPR pair (SDST)
// instead of VCC — track it in sreg_bool so a later v_cndmask_b32_e64 / s_cselect can read it.
// Otherwise it writes VCC; keep VCC's narrowed-state in sync (106/107 = VCC_LO/HI).
void write_vopc_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t cmp,
                     bool compare_wave_uniform) {
    if (vopc_is_cmpx(in.opcode)) {
        // v_cmpx writes EXEC ONLY on gfx10 (EXEC &= cmp) — it has NO VCC/SGPR destination.
        // The old shared handler fell into the `else` and set vcc = cmp for cmpx too,
        // clobbering a VCC value kept live ACROSS the cmpx, so a later v_cndmask/s_cbranch_vccz/
        // v_add_co_ci reading VCC got the compare mask instead of the real predicate (#464).
        // EXEC suppresses inactive vector comparisons. Select masks an unselected
        // comparison's poison; strict OpLogicalAnd would propagate it into EXEC.
        const uint32_t exec_before = rs.exec;
        rs.exec = b.bsel(rs.exec, cmp, b.bfalse()); rs.exec_narrowed = true;
        rs.exec_subset_of[rs.exec] = exec_before;   // EXEC' = EXEC & cmp
    } else {
        // ISA 3.9: "VCC[n] = EXEC[n] & (test passed for thread n)" — a lane inactive in
        // EXEC gets its mask bit forced to 0, never the raw test result. Masking at the
        // write keeps wave reductions (vccz votes, mbcnt publications, v_writelane mask
        // spills) from seeing phantom bits from inactive lanes. No-op when EXEC is full.
        // This is architectural predication, not scalar mask arithmetic: an inactive
        // lane writes defined zero even when its unevaluated comparison can be poison.
        const uint32_t masked = exec_masked_predicate(b, rs, cmp);
        if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
            rs.sreg_bool[in.dst.value] = masked; rs.sreg_bool_narrowed[in.dst.value] = true;
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32)))
                rs.sreg_bool_b32.insert(in.dst.value);
        } else if (b.allow_b32_masks && in.dst.kind == OperandKind::SGPR &&
                   (in.dst.value == 106 || in.dst.value == 107)) {
            // Wave32 SDWA/e64 compares can explicitly target either one-word VCC half as
            // an independent saved mask (Astro world-map PC1060 targets VCC_HI, then
            // consumes it with s_andn2_b32). Keep that physical destination distinct;
            // only VCC_LO is also the implicit condition used by vccz/cndmask forms.
            rs.sreg_bool[in.dst.value] = masked;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            rs.sreg_bool_b32.insert(in.dst.value);
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            if (in.dst.value == 106) rs.vcc = masked;
        } else {
            // VOPC replaces the architectural VCC predicate. In Wave32 the complete mask
            // occupies VCC_LO only; VCC_HI remains available as ordinary scalar scratch
            // (Astro's sibling traversal keeps a float there across an implicit compare).
            // Wave64 still replaces both physical words. Retain the low-word mask marker
            // so a following s_mov_b32 save sees the fresh predicate rather than stale
            // dispatcher-loaded scalar data.
            rs.vcc = masked;
            rs.vcc_wave_uniform = compare_wave_uniform ? masked : 0;
            if (b.wave_size == 32) {
                rs.sreg.erase(106);
                rs.sreg_srt.erase(106);
            } else {
                mask_write_clobbers_pair(rs, 106);
            }
            rs.sreg_bool_narrowed[106] = true;
            if (b.wave_size != 32) rs.sreg_bool_narrowed[107] = true;
            if (b.allow_b32_masks &&
                (b.is_fragment || (b.is_compute && b.wave_size == 32))) {
                rs.sreg_bool[106] = masked;
                rs.sreg_bool_b32.insert(106);
            }
        }
    }
}

} // namespace prosper::gpu::ir
