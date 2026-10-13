#include "gpu/recompiler/rdna2_fragment_terminal_tail.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {

Rdna2FragmentTerminalTail rdna2_fragment_terminal_tail(const uint32_t* code, size_t dwords,
                                                       std::span<const Rdna2Inst> body) {
    Rdna2FragmentTerminalTail result;
    if (!code || dwords > UINT32_MAX || body.empty() || !body.back().is_end ||
        body.back().synthetic_terminator)
        return result;
    const size_t body_end = size_t(body.back().pc) + body.back().len_dwords;
    if (body_end > dwords || body.back().len_dwords != 1 ||
        (code[body.back().pc] & 0xffff0000u) != 0xbf810000u)
        return result;
    result.retained_dwords = result.inspected_dwords = body_end;
    std::set<uint32_t> targets;
    for (const auto& branch : body) {
        if (branch.fmt != Rdna2Format::SOPP || branch.opcode < 0x04 || branch.opcode > 0x09)
            continue;
        const int64_t target = int64_t(branch.pc) + branch.len_dwords + branch.simm16;
        if (target < int64_t(body_end)) continue;
        if (uint64_t(target) >= dwords || uint64_t(target) > UINT32_MAX) {
            result.source_limit_required = true;
            continue;
        }
        targets.insert(static_cast<uint32_t>(target));
    }
    std::map<uint32_t, Rdna2Inst> tails;
    for (uint32_t target : targets) {
        std::vector<Rdna2Inst> tail;
        size_t pc = target;
        bool cleared_exec = false, exported = false, closed = false;
        // A terminal idiom is small. Bound work even when many body branches aim into one
        // very long NOP suffix; the existing NOP/END span service handles unlimited padding.
        while (pc < dwords && pc - target < 256) {
            result.inspected_dwords = std::max(result.inspected_dwords, pc + 1);
            const uint32_t word = code[pc];
            size_t width = 1;
            if ((word & 0xffff0000u) == 0xbf800000u) {
                // Architectural NOP, including its immediate, is transparent.
            } else if ((word & 0xffff0000u) == 0xbf810000u) {
                closed = !cleared_exec || exported;
            } else if (!cleared_exec && word == 0xbefe0480u) {
                cleared_exec = true;   // S_MOV_B64 EXEC, inline integer zero; preserves SCC.
            } else if (cleared_exec && !exported && (word & 0xffffe00fu) == 0xf8000000u &&
                       (((word >> 4) & 0x3fu) < 8u ||
                        (((word >> 4) & 0x3fu) == 8u && !(word & (1u << 10))))) {
                // EN0 observes no physical VGPR payload, including COMPR. Retain the encoded
                // payload word nevertheless. CONFIDENCE: HIGH -- AMD 70648 chapter 11/Table 56
                // also defines uncompressed target 8 as depth, with the same component enables
                // and EXEC validity mask. No enabled output or NULL/vertex target is licensed.
                width = 2;
                if (width > dwords - pc || width > 256 - (pc - target)) {
                    result.source_limit_required |= width > dwords - pc;
                    break;
                }
                result.inspected_dwords = std::max(result.inspected_dwords, pc + width);
                exported = true;
            } else {
                break;
            }
            auto instruction = rdna2_decode_one(code + pc, width);
            instruction.pc = static_cast<uint32_t>(pc);
            tail.push_back(instruction);
            pc += width;
            if (instruction.is_end) break;
        }
        result.source_limit_required |= pc == dwords && !closed;
        if (!closed) continue;
        result.retained_dwords = std::max(result.retained_dwords, pc);
        for (const auto& instruction : tail) tails.emplace(instruction.pc, instruction);
    }
    // Two accepted targets must agree on instruction boundaries. An EXP's second word cannot
    // become another target merely because its bits happen to spell a complete trailer.
    size_t previous_end = body_end;
    for (const auto& [pc, instruction] : tails) {
        if (pc < previous_end) {
            result.retained_dwords = body_end;
            result.instructions.clear();
            return result;
        }
        previous_end = size_t(pc) + instruction.len_dwords;
        result.instructions.push_back(instruction);
    }
    return result;
}

}   // namespace prosper::gpu
