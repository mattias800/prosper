// The counted-loop lowering's proof that a wave-empty EXEC guard around (or inside) the loop may be
// linearized. See the comment at the scan.
#include "gpu/recompiler/rdna2_counted_loop_guard.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <vector>

namespace prosper::gpu {

bool mark_counted_loop_exec_guards(const std::vector<Rdna2Inst>& ins, const CountedLoop& L,
                                   std::unordered_set<uint32_t>& safe) {
    bool guarded_narrow_entry = false;
    // saveexec -> execz -> matching EXEC restore around a side-effect-free counted region is a
    // whole-wave empty-work optimization. In the per-invocation shell we may run the uniform
    // scalar loop for every invocation while narrowed EXEC predicates vector writes; inactive
    // lanes retain their old VGPRs until the exact restore. Reject stores/exports/barriers and
    // unclassified memory so this never becomes a general branch-linearization escape hatch.
    // Scan inside-out so an already-proven nested guard may contribute its balanced save/restore
    // pair without making an otherwise-safe outer guarded loop look like it leaks narrowed EXEC.
    struct GuardedExecRegion { uint32_t save_pc, restore_pc; };
    std::vector<GuardedExecRegion> guarded_exec_regions;
    // Every SOPP that can transfer control: s_branch, the SCC/VCC/EXEC conditional branches and
    // the debugger-conditional ones (0x17-0x1a), whose taken edge must still count as an edge.
    auto is_branch_opcode = [](uint32_t op) {
        return (op >= 0x02 && op <= 0x09 && op != 0x03) || (op >= 0x17 && op <= 0x1a);
    };
    for (size_t branch_index = ins.size(); branch_index-- > 0;) {
        const Rdna2Inst& branch = ins[branch_index];
        if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 || branch.simm16 <= 0)
            continue;
        size_t previous = branch_index;
        while (previous > 0) {
            --previous;
            if (!sopp_is_noop(ins[previous])) break;
        }
        if (previous >= branch_index) continue;
        // Two encodings of "save EXEC, narrow it, skip the region if no lane is left":
        //   * s_and_saveexec_b64 sN, cond   -> s_cbranch_execz  (the saved mask is sN)
        //   * s_mov_b64 sN, exec ... v_cmpx_* -> s_cbranch_execz  (UE4's NGG vertex shaders,
        //     #4427: Kena's 13 lighting-loop programs save EXEC early and narrow it with a VOPC
        //     compare-and-write-EXEC directly before the branch)
        // Both leave EXEC = saved & cond and sN = saved. For the second form the save must reach
        // the compare along one straight-line block: no branch between them, no branch target
        // inside, and no write to EXEC or sN in between, so sN still holds the pre-narrow mask.
        const Rdna2Inst& narrow = ins[previous];
        const Rdna2Inst* saveexec = nullptr;
        if (narrow.fmt == Rdna2Format::SOP1 && (narrow.opcode == 0x24 || narrow.opcode == 0x25) &&
            narrow.dst.kind == OperandKind::SGPR && narrow.dst.value <= 104) {
            saveexec = &narrow;
        } else if (narrow.fmt == Rdna2Format::VOPC && vopc_is_cmpx(narrow.opcode)) {
            for (size_t k = previous; k-- > 0;) {
                const Rdna2Inst& candidate = ins[k];
                if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == kSop1OpcodeMovB64 &&
                    candidate.dst.kind == OperandKind::SGPR && candidate.dst.value <= 104 &&
                    reg_operand(candidate.src[0], 126)) {
                    saveexec = &candidate;
                    break;
                }
                if (rdna2_instruction_may_change_exec(candidate) ||
                    (candidate.fmt == Rdna2Format::SOPP && is_branch_opcode(candidate.opcode)))
                    break;
            }
            if (saveexec) {
                bool straight = true;
                for (const auto& other : ins) {
                    if (other.fmt != Rdna2Format::SOPP || !is_branch_opcode(other.opcode)) continue;
                    const uint32_t to = branch_target(other);
                    if (to > saveexec->pc && to <= branch.pc) { straight = false; break; }
                }
                for (const auto& between : ins) {
                    if (!straight) break;
                    if (between.pc <= saveexec->pc || between.pc >= narrow.pc) continue;
                    for_each_scalar_write(between, [&](int base, uint32_t width) {
                        straight &= !(base < static_cast<int>(saveexec->dst.value) + 2 &&
                                      static_cast<int>(saveexec->dst.value) <
                                          base + static_cast<int>(width));
                    });
                }
                if (!straight) saveexec = nullptr;
            }
        }
        if (!saveexec) continue;
        const uint32_t target = branch_target(branch);
        // The matching restore is `s_mov_b64 exec, sN`, at the branch target or a few scalar
        // instructions after it (UE4 schedules the next s_load ahead of the restore). The window
        // [target, restore) runs with the narrowed EXEC on both paths, so it is held to the same
        // side-effect rules as the region below, and it must be straight-line.
        const Rdna2Inst* restore = nullptr;
        constexpr uint32_t kRestoreWindow = 8;
        uint32_t window = 0;
        for (const auto& candidate : ins) {
            if (candidate.pc < target) continue;
            if (candidate.fmt == Rdna2Format::SOP1 && candidate.opcode == kSop1OpcodeMovB64 &&
                candidate.dst.value >= 126 && reg_operand(candidate.src[0], saveexec->dst.value)) {
                restore = &candidate;
                break;
            }
            if (++window > kRestoreWindow || candidate.is_end ||
                (candidate.fmt == Rdna2Format::SOPP && is_branch_opcode(candidate.opcode)) ||
                (candidate.fmt != Rdna2Format::SMEM && !sopp_is_noop(candidate)))
                break;
        }
        if (!restore) continue;
        const uint32_t restore_pc = restore->pc;
        if (restore_pc != target) {
            bool entered_inside = false;
            for (const auto& other : ins) {
                if (other.fmt != Rdna2Format::SOPP || !is_branch_opcode(other.opcode)) continue;
                const uint32_t to = branch_target(other);
                if (to > target && to <= restore_pc) { entered_inside = true; break; }
            }
            if (entered_inside) continue;
        }
        // A lexical save/restore pair is not necessarily balanced along the counted-loop CFG.
        // In particular, a save in the body with its restore after the backedge leaves EXEC
        // narrowed between iterations (EXEC has no loop phi), and a zero-trip path reaches an
        // undominated restore. Accept only a pair contained in one straight-line loop segment,
        // or a true preheader-to-postloop wrapper around the complete loop. "Ends before" tests
        // use the restore pc and "ends after" tests use the branch target, so a restore placed
        // after the target can only make each test stricter.
        const bool same_preloop = saveexec->pc < L.header_pc && restore_pc < L.header_pc;
        const bool same_condition = saveexec->pc >= L.header_pc && restore_pc < L.exit_branch_pc;
        const bool same_body = saveexec->pc > L.exit_branch_pc && restore_pc < L.backedge_pc;
        const bool same_postloop = saveexec->pc >= L.exit_pc;
        const bool wraps_loop = saveexec->pc < L.header_pc && target >= L.exit_pc;
        if (!same_preloop && !same_condition && !same_body && !same_postloop && !wraps_loop)
            continue;
        bool side_effect_free = true;
        for (const auto& candidate : ins) {
            if (candidate.pc <= branch.pc || candidate.pc >= restore_pc) continue;
            bool clobbers_guard_mask = false;
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                clobbers_guard_mask |= base < static_cast<int>(saveexec->dst.value) + 2 &&
                    static_cast<int>(saveexec->dst.value) < base + static_cast<int>(width);
            });
            bool balanced_nested_exec = false;
            for (const auto& nested : guarded_exec_regions) {
                if (nested.save_pc > branch.pc && nested.restore_pc < restore_pc &&
                    (candidate.pc == nested.save_pc || candidate.pc == nested.restore_pc)) {
                    balanced_nested_exec = true;
                    break;
                }
            }
            // Reads from a descriptor-bounded buffer or image are allowed: their results are
            // EXEC-predicated VGPR writes (inactive lanes keep the old value, as for any VALU here),
            // and an inactive lane's out-of-range address is bounded by the descriptor. Everything
            // that writes memory (rdna2_instruction_may_write_memory is fail-closed), LDS-mode
            // buffer loads, FLAT (an unbounded address), DS, exports and barriers stay refused.
            const bool bounded_read =
                (candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                 candidate.fmt == Rdna2Format::MIMG) &&
                !rdna2_instruction_may_write_memory(candidate) &&
                !(candidate.fmt == Rdna2Format::MUBUF && candidate.mubuf_lds);
            const bool memory_or_export =
                candidate.fmt == Rdna2Format::EXP || candidate.fmt == Rdna2Format::DS ||
                candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                candidate.fmt == Rdna2Format::MIMG || candidate.fmt == Rdna2Format::FLAT;
            // A scalar store ignores EXEC, so linearizing the skipped region would perform it
            // even when every lane was off; any memory writer, scalar included, is refused.
            if ((memory_or_export && !bounded_read) || rdna2_instruction_may_write_memory(candidate) ||
                (rdna2_instruction_may_change_exec(candidate) && !balanced_nested_exec) ||
                clobbers_guard_mask ||
                (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0x0a)) {
                side_effect_free = false;
                break;
            }
        }
        if (!side_effect_free) continue;
        safe.insert(branch.pc);
        guarded_exec_regions.push_back({saveexec->pc, restore_pc});
        if (branch.pc < L.header_pc && target >= L.exit_pc) guarded_narrow_entry = true;
    }
    return guarded_narrow_entry;
}

}  // namespace prosper::gpu
