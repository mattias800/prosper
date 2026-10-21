#include "gpu/recompiler/rdna2_cfg_loop_admission.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
DivLoopRefusal admit_div_loop_exit(uint32_t& tgt, DivLoop& L, const prosper::gpu::Rdna2Inst& in,
                                   const bool& execnz, bool& exact_fragment_wave_breaks,
                                   const std::vector<Rdna2Inst>& ins) {
    if (tgt == L.exit_pc) {   // an exit test
        if (!L.exit_branch_pc) {   // first one = the canonical exit
            if (in.opcode == 0x08) {   // execz -> EXIT
                L.condition = DivLoop::Condition::Exec;
            } else if (in.opcode == 0x06 && !execnz &&
                       (exact_fragment_wave_breaks || vcc_exit_is_wave_uniform(ins, in.pc))) {
                // Fragment reduces the real VCC mask through a forced wave64 vote. Other
                // structured stages require the compare to be proven uniform first.
                L.condition = DivLoop::Condition::Vcc;
            } else if ((in.opcode == 0x04 || in.opcode == 0x05) && !execnz) {
                // A scalar SCC exit with an unconditional back-edge is the same canonical
                // top-tested while loop accepted by CountedLoop. Keep it in this multi-loop
                // region form when several disjoint/nested loops coexist in one shader.
                // scc0 exits while SCC==0 (continue on set); scc1 has the opposite polarity.
                L.condition = DivLoop::Condition::Scc;
                L.continue_on_set = in.opcode == 0x04;
            } else if (in.opcode == 0x06 && !execnz) {
                // Reached the vccz arm but the compare was not proven wave-uniform.
                return DivLoopRefusal{"vcc-exit-not-proven-wave-uniform", in.pc};
            } else if (execnz) {
                // A scalar/vcc exit test combined with an execnz back-edge: the bottom-tested
                // lowering owns that back-edge, so this exit shape is not the one modeled.
                return DivLoopRefusal{"scalar-exit-with-execnz-backedge", in.pc};
            } else {
                return DivLoopRefusal{"exit-branch-opcode-not-modeled", in.pc};
            }
            L.exit_branch_pc = in.pc;
        } else {   // later ones = breaks
            if (in.opcode != 0x06 && in.opcode != 0x08)
                return DivLoopRefusal{"break-not-vccz-or-execz", in.pc};   // vccz/execz only
            // With an execnz back-edge, a cleared lane skips to the next header check. With
            // an unconditional back-edge that could re-enable EXEC, only an EXECZ branch is
            // exact: the emitter sends the cleared lane directly to the loop merge instead.
            // A VCCZ break does not itself clear EXEC and remains unsupported for that shape.
            if (!execnz && in.opcode != 0x08) {
                // A VCCZ break does not clear per-lane EXEC, so the old lane-local loop
                // approximation could not carry it to the merge. The fragment shell now
                // reduces VCC over an enforced wave64 subgroup and can branch the complete
                // guest wave directly at the loop latch. Other stages remain conservative.
                if (!exact_fragment_wave_breaks || in.opcode != 0x06)
                    return DivLoopRefusal{"vccz-break-unsupported-stage", in.pc};
                L.direct_wave_breaks = true;
            } else if (!execnz) {
                L.direct_exec_breaks = true;
            }
            L.break_pcs.push_back(in.pc);
        }
    }
    return {};
}

DivLoopRefusal admit_div_loop_final_shape(DivLoop& L, const bool& scc_do_while,
                                          const std::vector<Rdna2Inst>& ins,
                                          uint32_t& backedge_op_li, const bool& execnz,
                                          const std::unordered_set<uint32_t>& safe) {
    if (!L.exit_branch_pc && scc_do_while) {
        // Bottom-tested SCC loop: `s_cbranch_scc1 HEADER` continues while SCC is set, scc0
        // while it is clear. The whole body runs before the first test, so it is emitted as
        // the condition region, and the test must read one compare on every path.
        if (scc_do_while_condition_pc(ins, L.header_pc, L.backedge_pc) == UINT32_MAX)
            return DivLoopRefusal{"scc-do-while-condition-not-a-straight-line-compare",
                                  L.backedge_pc};
        L.exit_branch_pc = L.backedge_pc;
        L.condition = DivLoop::Condition::Scc;
        L.continue_on_set = backedge_op_li == 0x05;
        L.bottom_tested = true;
    }
    if (!L.exit_branch_pc) {
        // Bottom-tested EXEC loop: the back-edge itself is `s_cbranch_execnz HEADER`, so the
        // complete body executes once before the first condition. Require the instruction
        // immediately preceding the branch to update EXEC; accepting a stale entry mask here
        // would turn an arbitrary backward branch into a do-while (and commonly an infinite
        // one). GTA V's nested lighting loop ends in V_CMPX; the regression uses S_MOV_B64
        // EXEC,0. Nested child branches were already validated and skipped above.
        if (!execnz) return DivLoopRefusal{"bottom-tested-needs-execnz-backedge", L.backedge_pc};
        const Rdna2Inst* condition_writer = nullptr;
        for (const auto& in : ins) {
            if (in.pc < L.header_pc) continue;
            if (in.pc >= L.backedge_pc) break;
            condition_writer = &in;
        }
        if (!condition_writer || !rdna2_instruction_may_change_exec(*condition_writer))
            return DivLoopRefusal{"bottom-tested-no-exec-writer", L.backedge_pc};
        L.exit_branch_pc = L.backedge_pc;
        L.condition = DivLoop::Condition::Exec;
        L.continue_on_set = true;
        L.bottom_tested = true;
    }
    if (!L.bottom_tested) {
        // A top-tested loop's canonical exit must be the FIRST branch, so its condition region
        // [header, exit_branch) is branch-free and can be emitted straight-line. A bottom-tested
        // loop instead puts its complete, already-validated structured body in this region.
        for (const auto& in : ins) {
            if (in.pc >= L.exit_branch_pc) break;
            if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
            if (in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03 && !safe.count(in.pc))
                return DivLoopRefusal{"condition-region-not-branch-free", in.pc};
        }
    }
    // The execnz flavor's unconditional-continue lowering requires the header check to
    // immediately re-test EXEC (empty condition region) — see the shape comment.
    if (execnz && !L.bottom_tested && L.exit_branch_pc != L.header_pc)
        return DivLoopRefusal{"execnz-nonempty-condition-region", L.header_pc};
    return {};
}

// Pass 2 for loop `li` (`backedge_op_li` is its back-edge opcode): validates its interior
// branches and fills its canonical exit, breaks, condition and shape flags.
DivLoopRefusal admit_div_loop(const std::vector<Rdna2Inst>& ins,
                              const std::unordered_set<uint32_t>& safe,
                              bool exact_fragment_wave_breaks, std::vector<DivLoop>& out, size_t li,
                              uint32_t backedge_op_li) {
    // Pass 2: validate each loop's interior branches and find the canonical exit + breaks. A branch
    // inside a NESTED child loop belongs to the child (which validates itself in its own pass-2
    // iteration) — skip it here, including the child's backward back-edge, which is exactly the
    // "second back-edge inside" that used to reject nesting outright.
    auto inside_nested_child = [&out](const DivLoop& L, uint32_t pc) {
        for (const auto& C : out)
            if (C.header_pc > L.header_pc && C.exit_pc <= L.backedge_pc && pc >= C.header_pc &&
                pc <= C.backedge_pc)
                return true;
        return false;
    };
    DivLoop& L = out[li];
    const bool execnz = backedge_op_li == 0x09;
    const bool scc_do_while = backedge_op_li == 0x04 || backedge_op_li == 0x05;
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= L.backedge_pc) break;
        if (in.pc < L.header_pc || in.fmt != Rdna2Format::SOPP) continue;
        if (inside_nested_child(L, in.pc)) continue;
        switch (in.opcode) {
            case 0x02:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09: break;
            default: continue;   // hints
        }
        if (in.simm16 < 0)
            return DivLoopRefusal{"nested-backedge-in-body",
                                  in.pc};   // second back-edge inside -> nested loop
        if (safe.count(in.pc)) continue;   // linearized (kill-mask / safe-execz)
        uint32_t tgt = branch_target(in);
        if (in.opcode == 0x02) {   // forward s_branch: an else-arm terminator
            if (tgt > L.backedge_pc)
                return DivLoopRefusal{"else-arm-branch-leaves-body",
                                      in.pc};   // may not leave the body
            continue;   // (validated by detect_forward_ifs)
        }
        if (tgt > L.exit_pc)
            return DivLoopRefusal{"conditional-branch-past-loop",
                                  in.pc};   // conditional jumping past the loop
        // An SCC do-while's only exit is falling through its back-edge. A second, interior
        // exit is a break this lowering does not model (CountedLoop refuses it the same way).
        if (scc_do_while && tgt == L.exit_pc)
            return DivLoopRefusal{"scc-do-while-interior-exit", in.pc};
        if (const DivLoopRefusal split_refusal =
                admit_div_loop_exit(tgt, L, in, execnz, exact_fragment_wave_breaks, ins);
            split_refusal.why)
            return split_refusal;
        // (tgt <= backedge_pc: an interior forward if — validated by detect_forward_ifs.)
    }
    if (const DivLoopRefusal split_refusal =
            admit_div_loop_final_shape(L, scc_do_while, ins, backedge_op_li, execnz, safe);
        split_refusal.why)
        return split_refusal;
    return {};
}
}   // namespace prosper::gpu
