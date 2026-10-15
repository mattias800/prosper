// EXEC subset facts that survive a structured loop header (#5039).
//
// A word defined before a loop under EXEC E and accumulated inside it under a narrower EXEC stays
// defined in every lane the loop reads it from, but only if the walk can prove that the loop's EXEC
// is still within E. Two facts carry that proof across the header:
//   - a VOPC mask is EXEC & comparison, so it is a subset of the EXEC it was computed under, and
//     `s_mov_b64 exec, sX` / `s_andn2_b64 exec, sX, ...` keep EXEC inside it;
//   - the header EXEC phi is a subset of the preheader EXEC when every back edge carries a subset
//     of it. That is an induction over iterations: the walk assumes it at the header, and the back
//     edge checks it from the facts the body recorded, using the assumption for the iteration that
//     entered the body, as the induction step allows. A loop whose check fails is refused.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {

bool writes_exec(const Rdna2Inst& in) {
    bool exec = false;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        exec |= base <= 127 && base + static_cast<int>(width) > 126;
    });
    return exec || rdna2_instruction_may_change_exec(in);
}

bool sgpr_pair(const Operand& operand) {
    return operand.kind == OperandKind::SGPR ||
           (operand.kind == OperandKind::Special && (operand.value == 106 || operand.value == 126));
}

// The EXEC writers whose new value can be a subset of the EXEC they replace: a CMPX; an S_MOV_B64
// from a register pair; an S_AND_B64 / S_ANDN2_B64 into EXEC; an AND-family SAVEEXEC. Whether the
// written value really is a subset is not decided here: the back edge checks the recorded facts.
// Any other EXEC writer (an OR, a NOT, a literal, a relative move) makes the loop no candidate.
bool narrowing_exec_form(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::VOPC) return vopc_is_cmpx(in.opcode);
    if (in.has_literal) return false;
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeAndSaveexecB64) return true;
    if (in.dst.value != 126) return false;
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode == kSop1OpcodeMovB64 && sgpr_pair(in.src[0]);
    if (in.fmt == Rdna2Format::SOP2) return in.opcode == 0x0fu || in.opcode == 0x15u;
    return false;
}

}   // namespace

uint32_t exec_masked_predicate(SpirvCompute& b, RegState& rs, uint32_t cmp) {
    if (!rs.exec_narrowed) return cmp;
    const uint32_t masked = b.bsel(rs.exec, cmp, b.bfalse());
    // CONFIDENCE: HIGH -- ISA 70648 3.9: VCC[n] = EXEC[n] & (test passed for thread n). The
    // selected value is false wherever EXEC is, so it names a subset of this EXEC value.
    if (masked && masked != rs.exec && !rs.exec_subset_of.contains(masked))
        rs.exec_subset_of[masked] = rs.exec;
    return masked;
}

uint32_t merge_exec_arm(SpirvCompute& b, RegState& rs, uint32_t skipped_exec,
                        uint32_t skipped_block, uint32_t taken_exec, uint32_t taken_block) {
    const uint32_t merged =
        b.emit_phi_2way(b.t_bool, skipped_exec, skipped_block, taken_exec, taken_block);
    // CONFIDENCE: HIGH -- each lane takes one incoming value, and both are within skipped_exec
    // when the taken arm's EXEC is a proved subset of it, so the phi is too.
    if (merged && merged != skipped_exec && exec_within(rs, taken_exec, skipped_exec))
        rs.exec_subset_of[merged] = skipped_exec;
    return merged;
}

LoopExecSubset::LoopExecSubset(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                               const DivLoop& loop, const RegState& entry) {
    certified_ = b.is_compute && narrowing_loop_exec_admission(ins, loop, entry);
    // Fragment loops only. A fragment invocation stays in a structured loop until the complete
    // guest wave leaves it, so the header EXEC is a wave fact. A compute loop leaves per invocation,
    // and its certificate also requires a lane-local body (narrowing_loop_exec_admission), which
    // this checked proof does not establish.
    if (certified_ || !b.is_fragment) return;
    bool narrows = false;
    for (const auto& in : ins) {
        if (in.pc < loop.header_pc || in.pc >= loop.backedge_pc || !writes_exec(in)) continue;
        if (!narrowing_exec_form(in)) return;
        narrows = true;
    }
    checked_ = narrows;
}

bool LoopExecSubset::holds(SpirvCompute& b, const RegState& rs) const {
    if (certified_ || !checked_) return true;
    // CONFIDENCE: HIGH -- the induction step: the body entered with header EXEC H, H within the
    // preheader EXEC P by the hypothesis, and computed its final EXEC from H. If that value is
    // within P (directly or through H), every header value is within P.
    // An un-narrowed preheader EXEC is the whole wave, which contains every EXEC value.
    if (rs.exec && (preheader_full_ || exec_within(rs, rs.exec, preheader_exec_))) return true;
    log_recompile_diagnostic(b.diagnostic, "loop-exec-subset-unproved", "terminal",
                             "a loop's back-edge EXEC is not a proved subset of its preheader EXEC");
    return false;
}

}   // namespace prosper::gpu
