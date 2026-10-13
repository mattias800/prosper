#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace prosper::gpu {

// Code authority and cache validation are different. Only a conditional forward target whose
// actual bounded words are [S_NOP]* S_ENDPGM extends retained_dwords. inspected_dwords also
// covers a failed probe's first nonterminal word, without treating it as executable/terminal.
// `body` is the original rdna2_walk prefix, never a synthetic phase terminator.
struct Rdna2TerminalTailSpan {
    size_t retained_dwords = 0;
    size_t inspected_dwords = 0;
    bool source_limit_required = false;
    std::vector<uint32_t> terminating_targets;
};

Rdna2TerminalTailSpan rdna2_terminal_tail_span(const uint32_t* code, size_t dwords,
                                               std::span<const Rdna2Inst> body);

} // namespace prosper::gpu
