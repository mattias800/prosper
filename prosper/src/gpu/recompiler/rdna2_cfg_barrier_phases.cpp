// Existing barrier-phase analysis and terminal-guard closure ownership.
#include "gpu/recompiler/rdna2_cfg_barrier_phases.hpp"
#include "gpu/recompiler/rdna2_cfg_barrier_proof_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {

// Prove the SCC consumed by a terminal compute guard is identical for every wave in the workgroup,
// while allowing unrelated lane-local setup to precede it. The legacy barrier-phase proof required
// every instruction before the guard to be scalar; generated kernels commonly interleave vector
// address setup with a scalar descriptor load that independently produces the guard condition.
//
// This is deliberately a narrow backwards slice. The last SCC writer must be an SOPC, and every
// scalar word feeding that compare must trace through launch SGPRs, scalar ALU, or scalar memory.
// V_READFIRSTLANE/V_READLANE, vector mask writes, EXEC, and unresolved special registers terminate
// the proof. Existing all-scalar prefixes retain their established admission path below.
bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                             size_t guard_index) {
    // This is a textual backwards slice, not a reaching-definition analysis. Keep the prefix
    // branch-free so every scalar definition it finds dominates the terminal guard. In particular,
    // a VCC/EXEC branch may vary between guest waves even when the literal or SMEM value written in
    // one arm is itself uniform.
    for (size_t i = 0; i < guard_index; ++i) {
        const Rdna2Inst& prefix = ins[i];
        if ((prefix.fmt == Rdna2Format::SOPP && !sopp_is_noop(prefix)) ||
            (prefix.fmt == Rdna2Format::SOP1 && prefix.opcode >= 0x20u && prefix.opcode <= 0x22u))
            return false;
    }

    std::function<bool(int, uint32_t, uint32_t)> uniform_reg_at;
    auto operand_uniform_at = [&](const Operand& operand, uint32_t words, uint32_t use_pc,
                                  auto&& self) -> bool {
        switch (operand.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal: return true;
            case OperandKind::Special:
                if (operand.value == 125) return true;   // SGPR_NULL
                if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                    return false;   // EXEC and SCC are not scalar launch data.
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!self(operand.value + static_cast<int>(word), use_pc, 0)) return false;
                return true;
            default: return false;
        }
    };

    uniform_reg_at = [&](int reg, uint32_t use_pc, uint32_t depth) -> bool {
        return cfg_terminal_uniform_register(ins, uniform_reg_at, reg, use_pc, depth);
    };

    for (size_t i = guard_index; i-- > 0;) {
        const Rdna2Inst& candidate = ins[i];
        if (sopp_is_noop(candidate)) continue;
        if (candidate.fmt == Rdna2Format::SOPC) {
            const uint32_t words = candidate.opcode == 0x12 || candidate.opcode == 0x13 ? 2u : 1u;
            for (uint32_t source = 0; source < candidate.n_src; ++source)
                if (!operand_uniform_at(candidate.src[source], words, candidate.pc, uniform_reg_at))
                    return false;
            return candidate.n_src != 0;
        }
        // Vector/memory operations cannot write SCC. Any scalar ALU or SOPK instruction might, so
        // stop rather than accidentally consuming an older compare through an unmodeled SCC writer.
        if (candidate.fmt == Rdna2Format::SOP1 || candidate.fmt == Rdna2Format::SOP2 ||
            candidate.fmt == Rdna2Format::SOPK)
            return false;
    }
    return false;
}

// Recognize barrier-separated compute phases. A generated kernel may wrap the phases in one proven
// workgroup-uniform terminal SCC guard, but the guard is not essential: an unconditional top-level
// barrier is itself a valid phase boundary. In either form, every explicit scalar edge must stay in
// one phase and indirect PC changes fail closed. Keeping this proof shared is important: emit_body
// uses it to split the shader, while the native subgroup policy uses the same result to make nested
// wave votes legal inside an individual phase.
BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
    BarrierPhasedCompute result;
    result.end_index = ins.size();
    uint32_t end_pc = UINT32_MAX;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].is_end) continue;
        result.end_index = i;
        end_pc = ins[i].pc;
        break;
    }

    result.guard_index = ins.size();
    size_t first_barrier = ins.size();
    size_t branch_count = 0;
    for (size_t i = 0; i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a && first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03)
            continue;
        ++branch_count;
        if (result.guard_index == ins.size() && (in.opcode == 0x04 || in.opcode == 0x05) &&
            branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool guarded = result.guard_index < first_barrier && first_barrier < result.end_index;
    bool scalar_prefix = guarded;
    cfg_barrier_scalar_prefix(scalar_prefix, result, ins);
    if (guarded && !scalar_prefix)
        guarded = terminal_guard_scc_is_workgroup_uniform(ins, result.guard_index);

    // If a candidate terminal guard was not proved uniform, include it in the ordinary edge proof.
    // Since it jumps across every barrier to S_ENDPGM, that proof rejects the conditional barrier.
    // A proved guard is deliberately excluded: every invocation in the workgroup takes it together.
    result.guarded = guarded;
    const size_t body_begin = guarded ? result.guard_index + 1 : 0;
    // Safety comes from the cross-edge, trap and EXEC-save checks below. A straight-line kernel
    // whose only control-flow instruction is a barrier satisfies that proof trivially; requiring
    // an unrelated branch-count threshold dropped GTA V's partial-workgroup barrier passes.
    bool valid = result.end_index < ins.size();

    for (size_t i = body_begin; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a) result.barriers.push_back(i);
    valid = valid && !result.barriers.empty();

    for (size_t i = body_begin; valid && i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // A trap deactivates its guest wave in the dispatcher. Emitting a host workgroup barrier
        // afterward would then require an invocation that has left the phase to participate.
        // Final-phase traps are fine because no outer barrier follows them.
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12 && i < result.barriers.back())
            valid = false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) valid = false;
    }
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = body_begin; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid;
    return result;
}

}   // namespace prosper::gpu
