// The counted-loop lowering's proof that a wave-empty EXEC guard around (or inside) the loop may be
// linearized. Moved out of rdna2_emit_cfg.cpp; see the comment at the scan.
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
        const Rdna2Inst& saveexec = ins[previous];
        if (saveexec.fmt != Rdna2Format::SOP1 ||
            (saveexec.opcode != 0x24 && saveexec.opcode != 0x25) ||
            saveexec.dst.kind != OperandKind::SGPR || saveexec.dst.value > 104) continue;
        const uint32_t target = branch_target(branch);
        const Rdna2Inst* restore = nullptr;
        for (const auto& candidate : ins) if (candidate.pc == target) { restore = &candidate; break; }
        if (!restore || restore->fmt != Rdna2Format::SOP1 || restore->opcode != 0x04 ||
            restore->dst.value < 126 || !reg_operand(restore->src[0], saveexec.dst.value)) continue;
        // A lexical save/restore pair is not necessarily balanced along the counted-loop CFG.
        // In particular, a save in the body with its restore after the backedge leaves EXEC
        // narrowed between iterations (EXEC has no loop phi), and a zero-trip path reaches an
        // undominated restore. Accept only a pair contained in one straight-line loop segment,
        // or a true preheader-to-postloop wrapper around the complete loop.
        const bool same_preloop = saveexec.pc < L.header_pc && target < L.header_pc;
        const bool same_condition = saveexec.pc >= L.header_pc && target < L.exit_branch_pc;
        const bool same_body = saveexec.pc > L.exit_branch_pc && target < L.backedge_pc;
        const bool same_postloop = saveexec.pc >= L.exit_pc;
        const bool wraps_loop = saveexec.pc < L.header_pc && target >= L.exit_pc;
        if (!same_preloop && !same_condition && !same_body && !same_postloop && !wraps_loop)
            continue;
        bool side_effect_free = true;
        for (const auto& candidate : ins) {
            if (candidate.pc <= branch.pc || candidate.pc >= target) continue;
            bool clobbers_guard_mask = false;
            for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                clobbers_guard_mask |= base < saveexec.dst.value + 2 &&
                    saveexec.dst.value < base + static_cast<int>(width);
            });
            bool balanced_nested_exec = false;
            for (const auto& nested : guarded_exec_regions) {
                if (nested.save_pc > branch.pc && nested.restore_pc < target &&
                    (candidate.pc == nested.save_pc || candidate.pc == nested.restore_pc)) {
                    balanced_nested_exec = true;
                    break;
                }
            }
            if (candidate.fmt == Rdna2Format::EXP || candidate.fmt == Rdna2Format::DS ||
                candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                candidate.fmt == Rdna2Format::MIMG || candidate.fmt == Rdna2Format::FLAT ||
                (rdna2_instruction_may_change_exec(candidate) && !balanced_nested_exec) ||
                clobbers_guard_mask ||
                (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0x0a)) {
                side_effect_free = false;
                break;
            }
        }
        if (!side_effect_free) continue;
        safe.insert(branch.pc);
        guarded_exec_regions.push_back({saveexec.pc, target});
        if (branch.pc < L.header_pc && target >= L.exit_pc) guarded_narrow_entry = true;
    }
    return guarded_narrow_entry;
}

}  // namespace prosper::gpu
