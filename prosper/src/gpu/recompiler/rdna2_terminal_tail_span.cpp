#include "gpu/recompiler/rdna2_terminal_tail_span.hpp"

#include <algorithm>
#include <cstdint>

namespace prosper::gpu {

Rdna2TerminalTailSpan rdna2_terminal_tail_span(const uint32_t* code, size_t dwords,
                                               std::span<const Rdna2Inst> body) {
    Rdna2TerminalTailSpan result;
    if (!code || body.empty() || !body.back().is_end || body.back().synthetic_terminator)
        return result;
    const size_t body_end = size_t(body.back().pc) + body.back().len_dwords;
    if (body_end > dwords || body.back().len_dwords != 1 ||
        (code[body.back().pc] & 0xffff0000u) != 0xbf810000u)
        return result;
    result.retained_dwords = result.inspected_dwords = body_end;
    std::vector<size_t> targets;
    for (const auto& branch : body) {
        // The six architectural SCC/VCC/EXEC conditionals. Unconditional transfers and debug
        // branches are not the existing verified early-out contract this helper preserves.
        if (branch.fmt != Rdna2Format::SOPP || branch.opcode < 0x04 || branch.opcode > 0x09)
            continue;
        const int64_t target = int64_t(branch.pc) + branch.len_dwords + branch.simm16;
        if (target < int64_t(body_end)) continue;
        if (uint64_t(target) >= dwords) {
            result.source_limit_required = true;
            continue;
        }
        targets.push_back(static_cast<size_t>(target));
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    // Descending targets can share a NOP suffix. Reuse the nearest already-inspected target's
    // result, so even many overlapping targets read each available tail word at most once.
    size_t previous_target = dwords, previous_end = 0;
    bool previous_terminal = false, previous_open = false;
    for (auto at = targets.rbegin(); at != targets.rend(); ++at) {
        const size_t target = *at;
        size_t pc = target, terminal_end = 0;
        bool terminal = false, open = false;
        for (; pc < dwords; ++pc) {
            if (pc == previous_target) {
                terminal = previous_terminal;
                terminal_end = previous_end;
                open = previous_open;
                break;
            }
            result.inspected_dwords = std::max(result.inspected_dwords, pc + 1);
            // Both allowed instructions are one dword. Do not decode arbitrary operands or read
            // a literal beyond the supplied window just to establish that this is not a trailer.
            const uint32_t encoding = code[pc] & 0xffff0000u;
            if (encoding == 0xbf810000u) {
                // CONFIDENCE: HIGH -- this is an actual S_ENDPGM after only S_NOPs. Retain every
                // byte read by the early-out proof in the immutable compiler/capture code span.
                terminal = true;
                terminal_end = pc + 1;
                break;
            }
            if (encoding != 0xbf800000u) break;
        }
        open |= pc == dwords;
        result.source_limit_required |= open;
        if (terminal) {
            result.retained_dwords = std::max(result.retained_dwords, terminal_end);
            result.terminating_targets.push_back(static_cast<uint32_t>(target));
        }
        previous_target = target;
        previous_terminal = terminal;
        previous_end = terminal_end;
        previous_open = open;
    }
    std::reverse(result.terminating_targets.begin(), result.terminating_targets.end());
    return result;
}

}   // namespace prosper::gpu
