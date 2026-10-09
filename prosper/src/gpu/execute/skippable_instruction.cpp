// skippable_instruction.cpp -- see the header.
#include "gpu/execute/skippable_instruction.hpp"

#include "gpu/execute/sopp_cfg.hpp"

#include <cstddef>
#include <limits>
#include <unordered_map>

namespace prosper::gpu {

namespace {

// SOPP program ends on gfx10: s_endpgm (0x01), s_endpgm_saved (0x1b), s_endpgm_ordered_ps_done
// (0x1e). The decoder's `is_end` names the first; the other two end the wave the same way.
bool ends_program(const Rdna2Inst& in) {
    return in.is_end || (in.fmt == Rdna2Format::SOPP &&
                         (in.opcode == 0x01 || in.opcode == 0x1b || in.opcode == 0x1e));
}

// s_cbranch_scc0 / s_cbranch_scc1. SCC is one bit per wave, but it is not necessarily draw state:
// `s_andn2_b64 exec, exec, vcc` sets it from the lane mask, and branching on it is LLVM's
// early-terminate (alpha-kill) shape. See scc_set_by_uniform_compare.
bool scc_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05);
}

// Data a scalar compare may read for its result to count as draw-uniform: an SGPR below VCC
// (s0..s105) or a constant. VCC, EXEC, M0, the other special registers and VGPRs are excluded,
// because a lane mask in any of them makes the comparison a mask test.
bool uniform_scalar_operand(const Operand& op) {
    switch (op.kind) {
        case OperandKind::SGPR: return op.value >= 0 && op.value < 106;
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal: return true;
        default: return false;
    }
}

// An instruction that sets SCC from draw-uniform data: a SOPC compare (not s_setvskip 0x10 or
// s_set_gpr_idx_on 0x11, which do not compare), or an s_cmpk_* (SOPK 0x03..0x0e) on such an SGPR.
// Deliberately narrow: any other SCC writer, a scalar ALU op on uniform data included, is treated
// as a mask test. That costs a skip that could have been claimed, never a false one.
bool uniform_scc_compare(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SOPC && in.opcode != 0x10 && in.opcode != 0x11) {
        for (uint8_t k = 0; k < in.n_src && k < 2; ++k)
            if (!uniform_scalar_operand(in.src[k])) return false;
        return in.n_src >= 2;
    }
    if (in.fmt == Rdna2Format::SOPK && in.opcode >= 0x03 && in.opcode <= 0x0e)
        return in.dst.kind == OperandKind::SGPR && uniform_scalar_operand(in.dst);
    return false;
}

// May write SCC: every scalar ALU format. Vector, memory, export and non-branch SOPP instructions
// never do, so the backwards walk steps over them.
bool scalar_alu(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
           in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::SOPK;
}

} // namespace

bool program_may_skip_by_scalar_branch(const std::vector<Rdna2Inst>& instructions,
                                       uint32_t use_pc) {
    if (instructions.empty() || has_indirect_control_flow(instructions)) return false;
    // Index instructions by pc. The decoded stream may append out-of-line tail blocks after the
    // first s_endpgm (rdna2_decode.hpp), so the vector is not assumed to be sorted.
    constexpr size_t kNone = std::numeric_limits<size_t>::max();
    uint32_t max_pc = 0;
    for (const Rdna2Inst& in : instructions) {
        if (in.synthetic_terminator || in.fmt == Rdna2Format::Unknown || in.len_dwords == 0)
            return false;
        if (in.pc > max_pc) max_pc = in.pc;
    }
    // No guest program is this long: refuse rather than allocate the index.
    if (max_pc > (1u << 20)) return false;
    std::vector<size_t> at(static_cast<size_t>(max_pc) + 1u, kNone);
    for (size_t i = 0; i < instructions.size(); ++i) at[instructions[i].pc] = i;
    if (use_pc > max_pc || at[use_pc] == kNone || at[0] == kNone) return false;

    // Successors, with every edge required to land on a decoded instruction: an incomplete graph
    // cannot support the claim.
    const size_t n = instructions.size();
    std::vector<size_t> succ(2 * n, kNone);
    std::vector<std::vector<size_t>> pred(n);
    for (size_t i = 0; i < n; ++i) {
        const Rdna2Inst& in = instructions[i];
        if (ends_program(in)) continue;
        const int64_t fallthrough = static_cast<int64_t>(in.pc) + in.len_dwords;
        int64_t targets[2] = {fallthrough, -1};
        if (sopp_is_branch(in)) {
            targets[0] = sopp_branch_target(in);
            if (!sopp_is_unconditional_branch(in)) targets[1] = fallthrough;
        }
        for (size_t k = 0; k < 2; ++k) {
            if (k == 1 && targets[1] == -1) break;
            const int64_t pc = targets[k];
            if (pc < 0 || pc > static_cast<int64_t>(max_pc) || at[static_cast<size_t>(pc)] == kNone)
                return false;
            succ[2 * i + k] = at[static_cast<size_t>(pc)];
            pred[succ[2 * i + k]].push_back(i);
        }
    }

    // Which scc branches are draw-uniform choices (#4801 re-review B3'): walk back from the branch
    // within its straight-line block to the last instruction that may write SCC, and accept only a
    // uniform compare. Stepping past an instruction some branch targets, or reaching a branch, an
    // end or the stream start first, leaves the definition unknown, and unknown counts as a mask.
    std::vector<char> targeted(n, 0);
    for (size_t i = 0; i < n; ++i)
        if (sopp_is_branch(instructions[i]) && succ[2 * i] != kNone) targeted[succ[2 * i]] = 1;
    std::vector<size_t> before(n, kNone);   // the instruction that falls through into this one
    for (size_t i = 0; i < n; ++i) {
        const uint64_t next = uint64_t(instructions[i].pc) + instructions[i].len_dwords;
        if (next <= max_pc && at[next] != kNone) before[at[next]] = i;
    }
    std::vector<char> uniform_branch(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (!scc_branch(instructions[i])) continue;
        size_t cur = i;
        for (;;) {
            if (targeted[cur]) break;
            const size_t prev = before[cur];
            if (prev == kNone) break;
            const Rdna2Inst& p = instructions[prev];
            if (scalar_alu(p)) {
                uniform_branch[i] = uniform_scc_compare(p) ? 1 : 0;
                break;
            }
            if (sopp_is_branch(p) || ends_program(p)) break;
            cur = prev;
        }
    }

    // Attractor of the program ends for the scalar-branch player, with the use removed. A node joins
    // when any successor is winning at a scalar branch, or when all of them are anywhere else (a
    // straight-line instruction has exactly one). Least fixpoint, so a path that loops forever never
    // counts as reaching an end.
    std::vector<char> win(n, 0);
    std::vector<int> pending(n, 0);   // successors not yet winning, for the all-successors nodes
    std::vector<size_t> work;
    for (size_t i = 0; i < n; ++i) {
        if (instructions[i].pc == use_pc) {
            pending[i] = -1;   // never joins
            continue;
        }
        if (ends_program(instructions[i])) {
            win[i] = 1;
            work.push_back(i);
            continue;
        }
        const int count = succ[2 * i + 1] == kNone ? 1 : 2;
        pending[i] = uniform_branch[i] ? 1 : count;
        // A branch whose two edges reach the same instruction needs it only once.
        if (count == 2 && succ[2 * i] == succ[2 * i + 1]) pending[i] = 1;
    }
    while (!work.empty()) {
        const size_t node = work.back();
        work.pop_back();
        for (const size_t p : pred[node]) {
            if (win[p] || pending[p] <= 0) continue;
            if (--pending[p] == 0) {
                win[p] = 1;
                work.push_back(p);
            }
        }
    }
    return win[at[0]] != 0;
}

bool SkippableInstructionQuery::may_skip(uint32_t use_pc) {
    // Per-thread answer cache: draw realization runs on several workers, and a process-global lock
    // here would be a P4 hot-path lock. Bounded, and keyed by the owner's identity with a weak
    // reference so an address reused by a later program is never served a stale answer.
    struct Entry {
        std::weak_ptr<const void> owner;
        std::unordered_map<uint32_t, bool> answers;
    };
    thread_local std::unordered_map<const void*, Entry> cache;
    constexpr size_t kMaxPrograms = 256;
    Entry* entry = nullptr;
    if (owner_) {
        auto it = cache.find(owner_.get());
        if (it != cache.end() && it->second.owner.lock() != owner_) {
            cache.erase(it);
            it = cache.end();
        }
        if (it == cache.end()) {
            if (cache.size() >= kMaxPrograms) cache.clear();
            it = cache.emplace(owner_.get(), Entry{owner_, {}}).first;
        }
        entry = &it->second;
        if (const auto hit = entry->answers.find(use_pc); hit != entry->answers.end())
            return hit->second;
    }
    if (!decoded_) {
        decoded_ = true;
        rdna2_walk(words_.data(), words_.size(), instructions_);
        if (live_) (void)rdna2_append_closed_tail_blocks(live_, live_dwords_, instructions_);
    }
    const bool answer = program_may_skip_by_scalar_branch(instructions_, use_pc);
    if (entry) entry->answers.emplace(use_pc, answer);
    return answer;
}

}   // namespace prosper::gpu
