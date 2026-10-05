#pragma once

// INTERNAL recompiler companion: loop-carried VCC when a loop body recycles VCC as scalar scratch
// (#4508). Used by the divergent-loop emitter in rdna2_emit_cfg.cpp.
//
// A structured loop gives VCC a header phi whenever VCC holds a lane mask on entry, and the phi
// needs a Bool from the continue block. A body that writes vcc_lo/vcc_hi with scalar ops leaves
// none: `rs.vcc == 0` says the pair now holds scalar data, which lives in rs.sreg[106]/[107].
// Compilers do this constantly -- VCC is the nearest free scratch pair -- and the loop then
// recomputes its predicate at the header:
//
//     header: v_cmp_lt_i32 vcc, s30, v4      ; redefines the whole pair
//             s_cbranch_vccz exit
//     body:   s_lshl_b32 vcc_lo, s30, 2      ; scratch
//             s_and_b32  vcc_hi, s30, 3      ; scratch
//             ...
//             s_branch header
//
// Space Adventure Cobra's per-pixel light loop is exactly this (three fragment programs, pc
// 102..173). The mask the phi would carry round the back-edge is never observed, so any Bool is a
// correct phi input -- the same degradation the SCC phi beside it already takes. What makes that
// sound is a proof, not the shape: VCC must be dead at the header in EVERY domain, so that neither
// a mask consumer nor a scalar read of either half can see the placeholder or the first
// iteration's entry mask. `ScalarMergeProof::AnyRead` is that proof; `MaskDomainOnly` would also
// admit a scalar read at the header, whose value on the entry path is a mask this model cannot
// hand out as data.
//
// CONFIDENCE: HIGH for the back-edge (the placeholder is unobservable by the proof); HIGH for the
// exit state (it restores the emitter's own "a mask write clobbers the pair" rule, see finish_exit).
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <vector>

namespace prosper::gpu {

struct LoopVccCarry {
    // Whether each VCC half was tracked as scalar data at the END OF THE CHECK BLOCK. The canonical
    // exit leaves through that block, so this is the state the merge has to reproduce.
    bool check_tracks_half[2] = {false, false};
    // The back-edge was closed with a placeholder: the body left VCC as scalar data.
    bool placeholder_backedge = false;

    explicit LoopVccCarry(const RegState& at_check_end)
        : check_tracks_half{at_check_end.sreg.contains(106), at_check_end.sreg.contains(107)} {}

    // The Bool that closes the VCC header phi when the body left no mask. 0 means the loop cannot
    // be emitted; the terminal reason is logged here so the caller's `return false` is not silent.
    //
    // `merge_reads_body_vcc`: a direct break reaches the merge from inside the body, where VCC is
    // scalar data. The merge would then join a mask with data, which has no Bool.
    uint32_t backedge_value(SpirvCompute& b, const std::vector<Rdna2Inst>& ins, uint32_t header_pc,
                            bool merge_reads_body_vcc) {
        if (merge_reads_body_vcc) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "loop-carried VCC is scalar data at the back-edge and a "
                                     "direct break reaches the merge (header pc=%u)",
                                     header_pc);
            return 0;
        }
        for (int half : {106, 107}) {
            ScalarMergeBlocker blocker;
            if (sgpr_dead_at_merge(ins, header_pc, half, ScalarMergeProof::AnyRead, &blocker))
                continue;
            // The proof can also give up at an instruction it does not model, and then names no
            // blocker; "is read" would be a claim nobody checked.
            if (blocker.pc == UINT32_MAX)
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "loop-carried VCC is scalar data at the back-edge and "
                                         "s%d could not be proven dead at header pc=%u",
                                         half, header_pc);
            else
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "loop-carried VCC is scalar data at the back-edge and "
                                         "s%d is read from header pc=%u before it is redefined "
                                         "(blocker pc=%u kind=%s)",
                                         half, header_pc, blocker.pc, blocker.kind);
            return 0;
        }
        placeholder_backedge = true;
        return b.bfalse();
    }

    // Whether the merge can name VCC. With a placeholder back-edge only the check block reaches the
    // merge (backedge_value refused the direct-break shape), so the body's missing mask is not an
    // input. Logs the reason when it cannot.
    bool merge_has_mask(SpirvCompute& b, uint32_t check_vcc, uint32_t body_vcc,
                        uint32_t header_pc) const {
        if (check_vcc && (body_vcc || placeholder_backedge)) return true;
        log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                 "loop exit cannot name VCC: check-mask=%d body-mask=%d "
                                 "(header pc=%u)",
                                 check_vcc != 0, body_vcc != 0, header_pc);
        return false;
    }

    // The counted-loop emitter keeps its own, older VCC phi and still refuses a body that leaves
    // VCC as scalar data: no title has needed it, and its exit state differs (it hands on the
    // header phi, not the check block's mask). This only gives that refusal a name, so the caller
    // does not `return false` in silence.
    static bool reject_counted_backedge(SpirvCompute& b, uint32_t header_pc) {
        log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                 "counted-loop body leaves VCC as scalar data at the back-edge "
                                 "(header pc=%u)",
                                 header_pc);
        return false;
    }

    // After the merge phis. A VCC half that the loop writes as scalar data is loop-carried, so the
    // merge just installed its HEADER phi -- the scratch the previous iteration left. On the
    // canonical exit that is stale wherever the check block overwrote the half with a mask: every
    // mask write erases the scalar value it clobbers (`mask_write_clobbers_pair`), and
    // operand_bits serves a tracked rs.sreg entry before it considers the mask. Left in place, a
    // later `s_mov_b32 s1, vcc_lo` reads last iteration's scratch where hardware reads the exit
    // mask's dword. Untracked, it takes the exact ballot or rejects loudly, exactly as it does
    // after a compare in straight-line code.
    //
    // This runs for every loop, not only the placeholder path (#4526). Whether the back-edge
    // carried a real mask, a placeholder, or no VCC phi at all (no mask live before the loop)
    // makes no difference to what the check block did to the half on the way out. A direct break
    // can reach the merge with the half still scalar; a value that is a mask on one edge and data
    // on the other has no scalar representation either, so it is dropped there too.
    void finish_exit(RegState& rs) const {
        for (int half : {106, 107}) {
            if (check_tracks_half[half - 106]) continue;
            rs.sreg.erase(half);
            rs.sreg_srt.erase(half);
        }
    }
};

}   // namespace prosper::gpu
