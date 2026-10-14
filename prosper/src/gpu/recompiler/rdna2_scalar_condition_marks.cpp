#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <cstdint>
namespace prosper::gpu {
// Both answers come from the instruction-effects table. An instruction it cannot classify (a
// reserved scalar encoding, an undecodable word) is neither a writer nor a preserver, which
// record_wave32_scalar_condition below treats as "may add a dependency, cannot refresh one".
bool scalar_instruction_writes_scc(const Rdna2Inst& in) {
    return rdna2_writes_scc(in.fmt, in.opcode) == Rdna2Answer::Yes;
}
bool scalar_instruction_preserves_scc(const Rdna2Inst& in) {
    return rdna2_writes_scc(in.fmt, in.opcode) == Rdna2Answer::No;
}
void record_wave32_scalar_condition(RegState& rs, const Rdna2Inst& in,
                                    const ScalarSourceMarks& source) {
    if (scalar_instruction_preserves_scc(in)) return;
    if (!scalar_instruction_writes_scc(in)) {
        // Unknown effects may add a missing dependency, but cannot refresh an older one.
        if (!source.placeholder) return;
        const bool hard = (rs.scc_merge_placeholder && rs.scc_loop_blanket.empty()) || source.hard;
        rs.scc_merge_placeholder = true;
        if (hard)
            rs.scc_loop_blanket.clear();
        else
            rs.scc_loop_blanket.insert(source.blanket_roots.begin(), source.blanket_roots.end());
        return;
    }
    rs.scc_merge_placeholder = source.placeholder;
    if (source.placeholder && !source.hard)
        rs.scc_loop_blanket = source.blanket_roots;
    else
        rs.scc_loop_blanket.clear();
}
bool scalar_condition_source_defined(const SpirvCompute& b, const RegState& rs) {
    if (!(b.is_compute || b.is_fragment) || (b.wave_size != 32 && !rs.track_scc_definition))
        return rs.scc != 0;
    if (rs.scc && !rs.scc_merge_placeholder) return true;
    if (rs.scc && !rs.scc_loop_blanket.empty()) {
        // The existing back-edge proof owns these exact scalar/vector assumptions.
        b.loop_blanket_roots_read.insert(rs.scc_loop_blanket.begin(), rs.scc_loop_blanket.end());
        return true;
    }
    return false;
}
bool scalar_condition_defined(SpirvCompute& b, const RegState& rs, uint32_t pc) {
    if (scalar_condition_source_defined(b, rs)) return true;
    if (b.wave_size == 32) {
        b.stage_reject_pc = pc;
        b.stage_reject_reason = "poisoned-scc";
    }
    return false;
}
} // namespace prosper::gpu
