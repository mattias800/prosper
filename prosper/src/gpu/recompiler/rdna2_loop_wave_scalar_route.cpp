// See rdna2_loop_wave_scalar_route.hpp.
#include "gpu/recompiler/rdna2_loop_wave_scalar_route.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

bool refuse_wave_scalar_loop(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                             const DivLoop& loop, const LoopExecSubset& proof,
                             const uint32_t* code, size_t dwords) {
    // Fragment loops keep every invocation until the wave leaves (fragment_wave_any); VCC and SCC
    // compute loops are admitted only with a wave-uniform exit. The certificate already requires
    // every written wave scalar dead at the exit.
    if (!b.is_compute || b.is_fragment || loop.condition != DivLoop::Condition::Exec ||
        proof.admitted())
        return false;
    const int word = loop_live_wave_scalar(ins, loop);
    if (word < 0) return false;
    // CONFIDENCE: HIGH -- per-invocation trip counts cannot reproduce the wave's final scalar
    // state (#5201). Refusing loses no faithful lowering: the exact attempt is the only route.
    b.stage_reject_pc = loop.header_pc;
    b.stage_reject_reason = "wave-scalar-loop-live-out";
    if (word == kLoopLiveScc)
        log_recompile_diagnostic(b.diagnostic, "wave-scalar-loop-live-out", "terminal",
                                 "pc=%u exit=%u scc is live after a per-invocation loop",
                                 loop.header_pc, loop.exit_pc);
    else
        log_recompile_diagnostic(b.diagnostic, "wave-scalar-loop-live-out", "terminal",
                                 "pc=%u exit=%u s%d is live after a per-invocation loop",
                                 loop.header_pc, loop.exit_pc, word);
    b.compute_store_loop_data_retry = native_exact_retry_authentic(b, code, dwords);
    return true;
}

}   // namespace prosper::gpu
