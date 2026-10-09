// skippable_instruction.cpp -- see the header.
#include "gpu/execute/skippable_instruction.hpp"

#include "gpu/execute/sopp_cfg.hpp"

#include <cstddef>
#include <limits>

namespace prosper::gpu {

namespace {

// SOPP program ends on gfx10: s_endpgm (0x01), s_endpgm_saved (0x1b), s_endpgm_ordered_ps_done
// (0x1e). The decoder's `is_end` names the first; the other two end the wave the same way.
bool ends_program(const Rdna2Inst& in) {
    return in.is_end || (in.fmt == Rdna2Format::SOPP &&
                         (in.opcode == 0x01 || in.opcode == 0x1b || in.opcode == 0x1e));
}

} // namespace

bool program_may_skip_instruction(const std::vector<Rdna2Inst>& instructions, uint32_t use_pc) {
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
    const auto index_of = [&](int64_t pc) -> size_t {
        if (pc < 0 || pc > static_cast<int64_t>(max_pc)) return kNone;
        return at[static_cast<size_t>(pc)];
    };

    // Depth-first search from the entry with the use removed from the graph. Reaching a program end
    // proves a path that never executes the use. Any edge into undecoded code makes the graph
    // incomplete, and an incomplete graph cannot support the claim, so it answers false.
    std::vector<char> seen(instructions.size(), 0);
    std::vector<size_t> stack{at[0]};
    seen[at[0]] = 1;
    bool reached_end = false;
    while (!stack.empty()) {
        const Rdna2Inst& in = instructions[stack.back()];
        stack.pop_back();
        if (in.pc == use_pc) continue;   // the removed node: no path continues through it
        if (ends_program(in)) {
            reached_end = true;
            continue;
        }
        const int64_t fallthrough = static_cast<int64_t>(in.pc) + in.len_dwords;
        int64_t successors[2] = {fallthrough, 0};
        size_t successor_count = 1;
        if (sopp_is_branch(in)) {
            successors[0] = sopp_branch_target(in);
            if (!sopp_is_unconditional_branch(in)) successors[successor_count++] = fallthrough;
        }
        for (size_t s = 0; s < successor_count; ++s) {
            const size_t next = index_of(successors[s]);
            if (next == kNone) return false;
            if (!seen[next]) {
                seen[next] = 1;
                stack.push_back(next);
            }
        }
    }
    return reached_end;
}

bool SkippableInstructionQuery::may_skip(uint32_t use_pc) {
    if (!decoded_) {
        decoded_ = true;
        rdna2_walk(words_.data(), words_.size(), instructions_);
        if (live_) (void)rdna2_append_closed_tail_blocks(live_, live_dwords_, instructions_);
    }
    return program_may_skip_instruction(instructions_, use_pc);
}

}   // namespace prosper::gpu
