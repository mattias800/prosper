#include "gpu/recompiler/ir/lower/sopp_case_routes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu::ir {
// clang-format off
bool SoppCaseRoutes::lower() {
    // Control flow. The only branch we can safely linearize today is the common forward
    // s_cbranch_execz "skip the EXEC-predicated block if no lane is active" idiom. Branches
    // on SCC/VCC (or EXECNZ) can skip code for reasons not represented by EXEC predication,
    // so accepting them as no-ops would silently execute the wrong path.
    switch (in.opcode) {
        // Hints / sync with no effect in our synchronous SSA model — safe no-ops.
        case 0x10:   // s_sendmsg: no-op, except GS_ALLOC_REQ in the NGG subgroup shell
            if (b.ngg_alloc_request) ok = b.ngg_alloc_request(rs, in);
            break;
        case 0x00:   // s_nop
        case 0x0c:   // s_waitcnt        (no async memory latency to wait on)
        case 0x16:   // s_ttracedata     (sends M0 to the thread-trace stream — a profiling side
                     //                   channel with no architectural effect: it writes no SGPR,
                     //                   VGPR, or memory and does not branch. Prosper models no
                     //                   thread trace, so emitting nothing IS the correct lowering.
                     //                   Observed live in Greak (PPSA02849) vertex shaders, which
                     //                   ship it at pc=2 and were rejected whole. CONFIDENCE: HIGH
                     //                   — round-tripped against llvm-mc gfx1030.)
        case 0x17:   // s_cbranch_cdbgsys (Prosper exposes no attached GPU system debugger, so
                     //                    COND_DBG_SYS is permanently clear and the branch falls through)
        case 0x20:   // s_inst_prefetch  (I-cache hint)
        case 0x21:   // s_clause         (memory-clause scheduling hint)
        case 0x22:   // s_wait_idle
            break;   // (s_waitcnt_vscnt is SOPK on gfx10, not SOPP 0x7d — see the SOPK case)
        case 0x08:                                          // s_cbranch_execz
            if (in.simm16 < 0) ok = false;                 // backward = loop -> unsupported
            else if (rs.exec_narrowed && (!safe_execz || !safe_execz->count(in.pc))) ok = false;
            break;                                          // forward = no-op (predication covers it)
        case 0x0a:                                          // s_barrier
            if (b.is_compute || b.ngg_private_lds) b.barrier();
            else ok = false;                                // barrier only meaningful in compute
            break;
        case 0x04: case 0x05:                              // s_cbranch_scc0 / scc1
            // An alpha-test / clip() kill-mask early-out (SCC = "any lane survives", set by a 64-bit
            // wave-mask op — see mask_test_branches) is a pure wave optimization: per-invocation the
            // survivor mask narrows EXEC in the block below and the export OpKills the failed lanes,
            // so the branch LINEARIZES away exactly like a forward s_cbranch_execz. Only these
            // recognized branches are dropped; any other SCC branch is still a real uniform-if we
            // can't model straight-line, so it rejects.
            if (safe_execz && safe_execz->count(in.pc)) break;   // recognized kill-mask branch -> no-op
            ok = false;
            break;
        case 0x06: case 0x07:                              // s_cbranch_vccz / vccnz
            // A byte-exact Astro NGG terminal suffix may skip only trailing PARAM exports
            // after POS has already been exported. In the one-lane projection emitting those
            // otherwise-unused values is harmless; the gate proof records only that bounded
            // branch in safe_execz. General VCC branches remain unsupported below.
            if (safe_execz && safe_execz->count(in.pc)) break;
            ok = false;
            break;
        case 0x09:                                         // s_cbranch_execnz
            ok = false;
            break;
        default: ok = false;   // s_branch / s_sendmsg / s_setreg / etc. -> reject
    }
    return true;
}
// clang-format on
}   // namespace prosper::gpu::ir
