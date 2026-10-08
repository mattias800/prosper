#pragma once
// Guest-level scalar-effect classification of the region an `s_cbranch_execz` skips (ADR 0028,
// route 2, migration step 1). Internal to the recompiler.
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Why a skipped region is NOT known to be free of effects that the guest's 64-lane wave would
// observe. On hardware the region's EXEC-masked vector work does nothing for an all-false wave, but
// its scalar work (SALU, SMEM, wave messages) executes whenever ANY lane is active and does not
// execute when the whole wave skips. A host that forces the vote TRUE (or evaluates it per narrower
// subgroup) would run, or skip, that scalar work where the guest wave did the opposite.
//
// Every field is fail-closed: an instruction this classifier does not positively know to be
// effect-free sets one, so an unknown or new encoding can only cost an admission.
struct ExecSkipRegionEffects {
    // SALU/VALU wrote an SGPR, VCC half, SCC, M0 or EXEC-derived scalar that is not proven dead at
    // the merge. SCC and EXEC have no liveness proof here, so any write to them counts.
    bool scalar_live_out = false;
    // Any SMEM operation other than a plain s_load/s_buffer_load (RDNA2 has no scalar stores or
    // atomics; this is the cache-control, timer and ATC-probe forms), or a scalar load whose
    // result is not proven dead at the merge.
    bool scalar_memory_effect = false;
    // s_sendmsg, s_barrier, s_setreg, s_sleep, s_trap, s_ttracedata, any other SOPP/SOPK this
    // classifier does not know to be a pure hint, or a cross-lane/EXEC-unpredicated VALU access:
    // DPP/SDWA forms it does not model, v_readfirstlane_b32, v_readlane_b32, v_writelane_b32.
    bool wave_side_effect = false;
    // A branch or jump in the region, an s_endpgm, or an indirect transfer. The only permitted
    // way out of the region is falling through to the merge.
    bool foreign_exit = false;
    // An instruction format outside the VALU-only scope of this certificate (vector memory, LDS,
    // export, unknown). Not one of the ADR's four conditions; refused for being unclassified.
    bool unclassified = false;

    bool clean() const {
        return !scalar_live_out && !scalar_memory_effect && !wave_side_effect && !foreign_exit &&
               !unclassified;
    }
};

// Classify the instructions in (branch_pc, target_pc) of `ins`, the region guarded by the forward
// `s_cbranch_execz` at `branch_pc` whose skip target (the merge) is `target_pc`. `ins` is the whole
// decoded program: scalar liveness at the merge is answered over it by `sgpr_dead_at_merge`.
// CONFIDENCE: HIGH for the effects it reports; MED that the whitelist is complete, which is why it
// is a whitelist.
ExecSkipRegionEffects classify_exec_skip_region(const std::vector<Rdna2Inst>& ins,
                                                uint32_t branch_pc, uint32_t target_pc);

}   // namespace prosper::gpu
